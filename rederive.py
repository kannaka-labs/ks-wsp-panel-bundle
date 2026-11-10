#!/usr/bin/env python3
"""Re-derive the ks-wsp panel result (Amendment 1 score) from the stripped bundle, and compare it with the
published score file data/panel_score.a1.json.

Usage:  python rederive.py [bundle_dir]        (default: the directory this file is in)
Needs:  Python 3.9+, numpy (the published numbers were made with numpy 2.4.2; numpy's default_rng / PCG64 stream
        is stable across versions, but percentile defaults are what they are, so use numpy >= 1.22).

What it does, in the order the frozen scorer does it (frozen/panel_score.py, score()):
  1. checks frozen/panel_sim.py against its pinned sha256 and imports its decide(), boot_mean(), kalpha_interval();
  2. reads items (no text), the leak gate, the key, the two scored judges' verdict letters (J-C rerun, J-G);
  3. builds the P (L1 vs L2) and A (adapter vs base) arrays, drops blocked items, runs decide() with
     B = 10,000 and default_rng(20261112), then the reported statistics (each with a fresh default_rng(20261112);
     alpha CI with default_rng(20261113));
  4. prints the band, S, the CIs, anchor, alpha, and whether every compared number is identical to
     data/panel_score.a1.json (exact JSON equality, not a tolerance).
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

BUNDLE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
SIM_SHA = "6def10dc1510c93dd3225301335aee5b161c51fe92acb0cb9f5032d6b67b2337"
B_BOOT, SEED_S, SEED_ALPHA, INVALID_MAX = 10_000, 20261112, 20261113, 0.10
SCORED = {"J-C": "data/judges/judge_J-C.a1.jsonl", "J-G": "data/judges/judge_J-G.jsonl"}
HOSTED = {"J-C": True, "J-G": True}
LABELS = {"L2_PREFERRED": "L2 CLOSER TO THE CARD", "L1_PREFERRED": "L1 CLOSER TO THE CARD",
          "UNRESOLVED(alpha)": "UNRESOLVED (low anchor agreement)", "UNRESOLVED(family)": "UNRESOLVED (family-dependent)",
          "UNRESOLVED(length)": "UNRESOLVED (length-dependent)", "NO_DIFFERENCE": "NO DIFFERENCE",
          "UNRESOLVED": "UNRESOLVED AT THIS n", "UNRELIABLE": "PANEL UNRELIABLE"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


p_sim = BUNDLE / "frozen" / "panel_sim.py"
assert sha(p_sim) == SIM_SHA, "frozen/panel_sim.py does not match its pinned sha256"
spec = importlib.util.spec_from_file_location("panel_sim_frozen", p_sim)
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)


def arrays(items, key, rows_by_judge, judges, field="voice"):
    idx = {it: i for i, it in enumerate(items)}
    I = len(items)
    P, A = np.zeros((len(judges), I, 2)), np.zeros((len(judges), 2 * I, 2))
    have = np.zeros((len(judges), 3 * I, 2), bool)
    Pfirst = np.zeros((I, 2))
    for k in key.values():
        if k["item"] in idx and k["comp"] == "P":
            Pfirst[idx[k["item"]], k["order"]] = 1 if k["A_arm"] == "L2" else -1
    for jn, name in enumerate(judges):
        for r in rows_by_judge[name]:
            k = key[r["call"]]
            if k["item"] not in idx:
                continue
            i, o = idx[k["item"]], k["order"]
            pick = r[field]
            arm = k["A_arm"] if pick == "A" else (k["B_arm"] if pick == "B" else None)
            if k["comp"] == "P":
                P[jn, i, o] = 0 if arm is None else (1 if arm == "L2" else -1)
                have[jn, i, o] = True
            else:
                u = i if k["comp"] == "A1" else I + i
                A[jn, u, o] = 0 if arm is None else (-1 if arm == "base" else 1)
                have[jn, I + u, o] = True
    return P, A, Pfirst, have.all(axis=(1, 2))


def boot_S(scores, clusters):
    rng = np.random.default_rng(SEED_S)
    s, lo, hi = sim.boot_mean(scores, clusters, clusters.max() + 1, rng, B_BOOT)
    return [float(s), float(lo), float(hi)]


def reindex(cl):
    return np.unique(cl, return_inverse=True)[1].astype(int)


def alpha_ci(Akeep, I):
    v = Akeep.mean(2)
    a0 = float(sim.kalpha_interval(v))
    rng = np.random.default_rng(SEED_ALPHA)
    bs = []
    for _ in range(B_BOOT):
        it = rng.integers(0, I, I)
        bs.append(sim.kalpha_interval(v[:, np.concatenate([it, it + I])]))
    bs = np.array(bs, float)
    bs = bs[np.isfinite(bs)]
    return [a0, float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]


def score():
    items_meta = jl(BUNDLE / "data/items.jsonl")
    lg = json.loads((BUNDLE / "data/leakgate.json").read_text())
    key = json.loads((BUNDLE / "data/key.json").read_text())
    build = json.loads((BUNDLE / "data/build.json").read_text())
    assert build["key_sha256"] == sha(BUNDLE / "data/key.json")
    blocked = {r["item"]: r["blocked"] for r in lg["items"]}
    by_item = {it["item"]: it for it in items_meta}
    all_items = [it["item"] for it in items_meta]
    clean = [x for x in all_items if not blocked[x]]
    rows, meta = {}, {}
    for n, p in SCORED.items():
        rows[n] = jl(BUNDLE / p)
        meta[n] = json.loads((BUNDLE / p).with_suffix(".meta.json").read_text())
        assert meta[n]["packet_sha256"] == build["packet_sha256"]
    out = {"n_items": len(all_items), "n_clean": len(clean), "n_blocked": len(all_items) - len(clean), "judges": {}}
    names, pre = list(rows), {}
    for n in names:
        inv = sum(not r["valid"] for r in rows[n])
        P, A, Pf, comp = arrays(clean, key, {n: rows[n]}, [n])
        info = {"calls": len(rows[n]), "invalid": inv, "invalid_rate": inv / max(len(rows[n]), 1),
                "status": meta[n].get("status"), "pin": meta[n].get("pin"), "spent_usd": meta[n].get("spent_usd"),
                "complete_on_clean": bool(comp[0])}
        reason = None
        if meta[n].get("status") == "excluded":
            reason = meta[n].get("excluded") or "excluded by runner"
        elif info["invalid_rate"] > INVALID_MAX:
            reason = f"invalid {inv}/{len(rows[n])} > 10%"
        elif not comp[0]:
            reason = f"incomplete ({meta[n].get('status')})"
        nt = P[0] != 0
        info["position_first_share_P"] = float(((P[0] == Pf) & nt).sum() / max(nt.sum(), 1))
        info["ties_P"] = int((P[0] == 0).sum())
        if reason:
            pre[n] = reason
        info["pre_excluded"] = reason
        out["judges"][n] = info
    elig = [n for n in names if n not in pre]
    out["eligible"] = elig
    cl = reindex(np.array([by_item[x]["idx"] for x in clean]))
    early = np.array([any(by_item[x]["hit_stop"].values()) for x in clean])
    P, A, Pf, _ = arrays(clean, key, rows, elig)
    band, info = sim.decide(P, Pf, A, cl, early, [HOSTED[n] for n in elig], np.random.default_rng(SEED_S), B_BOOT)
    keep = [elig[k] for k in info["retained"]]
    out.update({"band_raw": band, "band": LABELS[band],
                "decide_info": {k: v for k, v in info.items() if k != "retained"},
                "retained": keep, "position_excluded": [n for n in elig if n not in keep]})
    kidx = [elig.index(n) for n in keep]
    I = len(clean)
    sc = P[kidx].mean(2).mean(0)
    out["S"] = boot_S(sc, cl)
    qP, _, _, _ = arrays(clean, key, rows, elig, field="quality")
    out["S_quality"] = boot_S(qP[kidx].mean(2).mean(0), cl)
    for dec in ("greedy", "t0.7"):
        m = np.array([by_item[x]["decode"] == dec for x in clean])
        out[f"S_{dec}"] = boot_S(sc[m], reindex(cl[m]))
    ne = ~early
    out["S_no_early_stop"] = boot_S(sc[ne], reindex(cl[ne]))
    out["n_no_early_stop"] = int(ne.sum())

    def anchor(ks, mask=None):
        units = np.arange(2 * I)
        if mask is not None:
            units = units[mask[units % I]]
        a_ = A[ks][:, units]
        w = (a_ == 1).sum((0, 2)) + 0.5 * (a_ == 0).sum((0, 2))
        return boot_S(w / (len(ks) * 2), np.concatenate([cl, cl])[units])
    out["anchor_share"] = anchor(kidx)
    out["anchor_share_no_early_stop"] = anchor(kidx, ~early)
    out["anchor_lojo"] = {elig[k]: anchor([x for x in kidx if x != k]) for k in kidx}
    out["S_lojo"] = {elig[k]: boot_S(P[[x for x in kidx if x != k]].mean(2).mean(0), cl) for k in kidx}
    out["alpha_anchor"] = alpha_ci(A[kidx], I)
    out["alpha_P"] = float(sim.kalpha_interval(P[kidx].mean(2)))
    for k, n in enumerate(elig):
        out["judges"][n]["S_clean"] = boot_S(P[k].mean(1), cl)
        out["judges"][n]["anchor_share_clean"] = anchor([k])
    suf = []
    if band in ("L2_PREFERRED", "L1_PREFERRED"):
        sg = 1 if band == "L2_PREFERRED" else -1
        if np.sign(out["S_greedy"][0]) == -sg:
            suf.append("decode-dependent")
        sq = out["S_quality"]
        if abs(sq[0]) >= 0.15 and (sq[1] > 0 or sq[2] < 0) and np.sign(sq[0]) == -sg:
            suf.append("card gain with a quality cost")
    out["suffixes"] = suf
    out["length"] = {"early_stop_rate_matched": {s: float(np.mean([by_item[x]["hit_stop"][s] for x in all_items]))
                                                 for s in ("base", "L1", "L2")},
                     "items_with_any_early_stop_clean": int(early.sum()), "n_clean": I}
    out["pre_excluded"] = pre
    return out


def main():
    r = json.loads(json.dumps(score()))  # same JSON round trip as the published file
    pub = json.loads((BUNDLE / "data/panel_score.a1.json").read_text())
    compared = [k for k in pub if k not in ("inputs", "amendment")]
    diff = [k for k in compared if json.dumps(r.get(k)) != json.dumps(pub[k])]
    extra = sorted(set(r) - set(pub))
    man = json.loads((BUNDLE / "MANIFEST.json").read_text())
    o = man["originals"]
    inp = pub["inputs"]
    hash_checks = {
        "inputs.key == sha(data/key.json)": inp["key"] == sha(BUNDLE / "data/key.json"),
        "inputs.leakgate == sha(data/leakgate.json)": inp["leakgate"] == sha(BUNDLE / "data/leakgate.json"),
        "inputs.build == sha(data/build.json)": inp["build"] == sha(BUNDLE / "data/build.json"),
        "inputs.panel_sim == sha(frozen/panel_sim.py)": inp["panel_sim"] == sha(BUNDLE / "frozen/panel_sim.py"),
        "inputs.matched == MANIFEST original matched.jsonl": inp["matched"] == o["matched.jsonl"],
        "inputs.judge_J-C == MANIFEST original judge_J-C.a1.jsonl": inp["judge_J-C"] == o["judge_J-C.a1.jsonl"],
        "inputs.judge_J-G == MANIFEST original judge_J-G.jsonl": inp["judge_J-G"] == o["judge_J-G.jsonl"],
        "build.matched_sha256 == inputs.matched": json.loads((BUNDLE / "data/build.json").read_text())[
            "matched_sha256"] == inp["matched"],
        "stripped files match MANIFEST.stripped": all(sha(BUNDLE / p) == h for p, h in man["stripped"].items()),
    }
    print(json.dumps({"band": r["band"], "band_raw": r["band_raw"], "suffixes": r["suffixes"], "S": r["S"],
                      "S_J-C": r["judges"]["J-C"]["S_clean"], "S_J-G": r["judges"]["J-G"]["S_clean"],
                      "anchor_share": r["anchor_share"], "anchor_decide": r["decide_info"]["anchor"],
                      "alpha_anchor": r["alpha_anchor"], "S_no_early_stop": r["S_no_early_stop"],
                      "S_greedy": r["S_greedy"], "S_t0.7": r["S_t0.7"], "S_quality": r["S_quality"]}, indent=1))
    for k, v in hash_checks.items():
        print(("PASS " if v else "FAIL ") + k)
    print(f"compared {len(compared)} top-level fields of data/panel_score.a1.json: "
          f"{len(compared) - len(diff)} identical, {len(diff)} different {diff}; extra fields {extra}")
    ok = not diff and not extra and all(hash_checks.values())
    print("RE-DERIVATION: " + ("EXACT MATCH" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
