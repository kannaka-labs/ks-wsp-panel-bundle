#!/usr/bin/env python3
"""ks-wsp panel scorer (32-panel-prereg.md s6-s7). Owed hash H3b together with panel_build.py.

The band is decided by the frozen pre-freeze simulation's own `decide()` (panel_sim/panel_sim.py, sha256 pinned
below and checked before import), so the decision logic is that function, not a re-implementation. It is called with
B = 10,000 bootstrap replicates and rng = numpy default_rng(20261112) (32 s6), on the CLEAN items only (all judges
score the same items), clusters = prompts (a prompt's greedy and sampled items move together).

Before decide(): judges excluded for > 10% invalid calls, for a changed model id (J-C pin), or for not finishing
their calls are removed (32 s4-s5). decide() then applies the position-bias rule (share of non-tie P choices that
pick the response shown first outside [0.35, 0.65] -> excluded), band 1 (fewer than 2 retained / no hosted judge
retained / anchor fails with all or with any one retained judge left out), the alpha gate on the anchors (alpha >= 0.40,
Krippendorff interval on the anchor item scores), Rule F (leave-one-judge-out), the length gate (no-early-stop items:
S keeps sign with CI excluding 0 and the anchor passes), NO DIFFERENCE (CI inside [-0.15, 0.15]) and UNRESOLVED.

Reported beside the band (not bands): alpha 90% CI (item-cluster bootstrap, 10,000, default_rng(20261113)), alpha on P,
per-judge S / anchor share / position-bias share / invalid rate, S on the greedy and T = 0.7 strata, the Q statistic
and the "card gain with a quality cost" suffix, "decode-dependent", the length report (early-stop rates, items
dropped for n < 25, S on no-early-stop items), J-Q's S on ALL items (sensitivity). Secondary bootstraps each use
a fresh default_rng(20261112) so they do not depend on the order of computation.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / "panel_sim" / "panel_sim.py"
SIM_SHA = "6def10dc1510c93dd3225301335aee5b161c51fe92acb0cb9f5032d6b67b2337"
B_BOOT = 10_000
SEED_S, SEED_ALPHA = 20261112, 20261113
INVALID_MAX = 0.10
LABELS = {"L2_PREFERRED": "L2 CLOSER TO THE CARD", "L1_PREFERRED": "L1 CLOSER TO THE CARD",
          "UNRESOLVED(alpha)": "UNRESOLVED (low anchor agreement)", "UNRESOLVED(family)": "UNRESOLVED (family-dependent)",
          "UNRESOLVED(length)": "UNRESOLVED (length-dependent)", "NO_DIFFERENCE": "NO DIFFERENCE",
          "UNRESOLVED": "UNRESOLVED AT THIS n", "UNRELIABLE": "PANEL UNRELIABLE"}
HOSTED = {"J-C": True, "J-G": True, "J-Q": False}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_sim(path=SIM, pin=SIM_SHA):
    assert sha(path) == pin, "panel_sim.py does not match its pinned sha256 (32 s11)"
    spec = importlib.util.spec_from_file_location("panel_sim_frozen", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def arrays(items, key, judge_rows, judges, field="voice"):
    """P (J, I, 2), A (J, 2I, 2) with +1 = L2 (P) / adapter (A), -1 = L1 / base, 0 = tie; Pfirst (I, 2);
    complete (J,) whether every needed call is present."""
    idx = {it: i for i, it in enumerate(items)}
    I = len(items)
    P = np.zeros((len(judges), I, 2))
    A = np.zeros((len(judges), 2 * I, 2))
    have = np.zeros((len(judges), 3 * I, 2), bool)
    Pfirst = np.zeros((I, 2))
    for cid, k in key.items():
        if k["item"] not in idx:
            continue
        i, o = idx[k["item"]], k["order"]
        if k["comp"] == "P":
            Pfirst[i, o] = 1 if k["A_arm"] == "L2" else -1
    for jn, name in enumerate(judges):
        for r in judge_rows[name]:
            k = key[r["call"]]
            if k["item"] not in idx:
                continue
            i, o = idx[k["item"]], k["order"]
            pick = r[field]
            arm = k["A_arm"] if pick == "A" else (k["B_arm"] if pick == "B" else None)
            if k["comp"] == "P":
                v = 0 if arm is None else (1 if arm == "L2" else -1)
                P[jn, i, o] = v
                have[jn, i, o] = True
            else:
                u = i if k["comp"] == "A1" else I + i
                v = 0 if arm is None else (-1 if arm == "base" else 1)
                A[jn, u, o] = v
                have[jn, I + u, o] = True
    return P, A, Pfirst, have.all(axis=(1, 2))


def boot_S(sim, scores, clusters, seed=SEED_S, B=B_BOOT):
    rng = np.random.default_rng(seed)
    nclu = clusters.max() + 1
    s, lo, hi = sim.boot_mean(scores, clusters, nclu, rng, B)
    return [float(s), float(lo), float(hi)]


def reindex(cl):
    _, inv = np.unique(cl, return_inverse=True)
    return inv.astype(int)


def alpha_ci(sim, Akeep, I, seed=SEED_ALPHA, B=B_BOOT):
    """alpha on the anchor item scores (units A1, A2) with an item-cluster bootstrap (both units of an item move together)."""
    v = Akeep.mean(2)  # (k, 2I)
    a0 = float(sim.kalpha_interval(v))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        it = rng.integers(0, I, I)
        u = np.concatenate([it, it + I])
        bs.append(sim.kalpha_interval(v[:, u]))
    bs = np.array(bs, float)
    bs = bs[np.isfinite(bs)]
    return [a0, float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]


def score(a):
    sim = load_sim()
    matched = load_jsonl(a.matched)
    lg = json.loads(Path(a.leakgate).read_text())
    key = json.loads(Path(a.key).read_text())
    build = json.loads(Path(a.build).read_text())
    assert build["key_sha256"] == sha(a.key) and build["matched_sha256"] == sha(a.matched)
    assert lg["matched_sha256"] == sha(a.matched)
    blocked = {r["item"]: r["blocked"] for r in lg["items"]}
    by_item = {it["item"]: it for it in matched}
    all_items = [it["item"] for it in matched]
    clean = [x for x in all_items if not blocked[x]]
    judge_rows, jmeta, pre_excl = {}, {}, {}
    for spec in a.judge:
        name, path = spec.split("=", 1)
        judge_rows[name] = load_jsonl(path)
        mp = Path(path).with_suffix(".meta.json")
        jmeta[name] = json.loads(mp.read_text()) if mp.exists() else {}
        assert jmeta[name].get("packet_sha256", build["packet_sha256"]) == build["packet_sha256"]
    out = {"n_items": len(all_items), "n_clean": len(clean), "n_blocked": len(all_items) - len(clean),
           "judges": {}, "inputs": {"matched": sha(a.matched), "leakgate": sha(a.leakgate), "key": sha(a.key),
                                    "build": sha(a.build), "panel_sim": SIM_SHA,
                                    **{f"judge_{n}": sha(s.split('=', 1)[1]) for n, s in zip(judge_rows, a.judge)}}}
    names = list(judge_rows)
    for n in names:
        rows = judge_rows[n]
        inv = sum(not r["valid"] for r in rows)
        P, A, Pf, complete = arrays(clean, key, {n: rows}, [n])
        info = {"calls": len(rows), "invalid": inv, "invalid_rate": inv / max(len(rows), 1),
                "status": jmeta[n].get("status"), "pin": jmeta[n].get("pin"), "spent_usd": jmeta[n].get("spent_usd"),
                "complete_on_clean": bool(complete[0])}
        reason = None
        if jmeta[n].get("status") == "excluded":
            reason = jmeta[n].get("excluded") or "excluded by runner"
        elif info["invalid_rate"] > INVALID_MAX:
            reason = f"invalid {inv}/{len(rows)} > 10%"
        elif not complete[0]:
            reason = f"incomplete ({jmeta[n].get('status')})"
        nt = P[0] != 0
        info["position_first_share_P"] = float(((P[0] == Pf) & nt).sum() / max(nt.sum(), 1))
        info["ties_P"] = int((P[0] == 0).sum())
        if reason:
            pre_excl[n] = reason
        info["pre_excluded"] = reason
        out["judges"][n] = info
    eligible = [n for n in names if n not in pre_excl]
    cl_all = reindex(np.array([by_item[x]["idx"] for x in clean]))
    early = np.array([any(by_item[x]["hit_stop"].values()) for x in clean])
    out["eligible"] = eligible
    if not eligible:
        out["band"], out["band_raw"], out["reason"] = "PANEL UNRELIABLE", "UNRELIABLE", "no judge eligible"
        return out
    P, A, Pf, _ = arrays(clean, key, judge_rows, eligible)
    rng = np.random.default_rng(SEED_S)
    band, info = sim.decide(P, Pf, A, cl_all, early, [HOSTED[n] for n in eligible], rng, B_BOOT)
    keep = [eligible[k] for k in info["retained"]]
    out.update({"band_raw": band, "band": LABELS[band], "decide_info": {k: v for k, v in info.items() if k != "retained"},
                "retained": keep, "position_excluded": [n for n in eligible if n not in keep]})
    kidx = [eligible.index(n) for n in keep]
    I = len(clean)
    # reasons for band 1
    if band == "UNRELIABLE":
        if len(keep) < 2:
            out["reason"] = f"fewer than 2 judges retained ({keep})"
        elif not any(HOSTED[n] for n in keep):
            out["reason"] = "no hosted judge (J-C or J-G) retained"
        else:
            out["reason"] = "anchor fails (with all retained judges or with one left out)"
    # full-precision stats
    if kidx:
        sc = P[kidx].mean(2).mean(0)
        out["S"] = boot_S(sim, sc, cl_all)
        sq_P, sq_A, _, _ = arrays(clean, key, judge_rows, eligible, field="quality")
        out["S_quality"] = boot_S(sim, sq_P[kidx].mean(2).mean(0), cl_all)
        for dec in ("greedy", "t0.7"):
            m = np.array([by_item[x]["decode"] == dec for x in clean])
            if m.any():
                out[f"S_{dec}"] = boot_S(sim, sc[m], reindex(cl_all[m]))
        ne = ~early
        out["S_no_early_stop"] = boot_S(sim, sc[ne], reindex(cl_all[ne])) if ne.any() else None
        out["n_no_early_stop"] = int(ne.sum())

        def anchor(ks, mask=None):
            units = np.arange(2 * I)
            if mask is not None:
                units = units[mask[units % I]]
            a_ = A[ks][:, units]
            w = (a_ == 1).sum((0, 2)) + 0.5 * (a_ == 0).sum((0, 2))
            return boot_S(sim, w / (len(ks) * 2), np.concatenate([cl_all, cl_all])[units])
        out["anchor_share"] = anchor(kidx)
        out["anchor_share_no_early_stop"] = anchor(kidx, ~early) if (~early).any() else None
        out["anchor_lojo"] = {eligible[k]: anchor([x for x in kidx if x != k]) for k in kidx} if len(kidx) > 1 else {}
        out["S_lojo"] = {eligible[k]: boot_S(sim, P[[x for x in kidx if x != k]].mean(2).mean(0), cl_all)
                         for k in kidx} if len(kidx) > 1 else {}
        if len(kidx) >= 2:
            out["alpha_anchor"] = alpha_ci(sim, A[kidx], I)
            out["alpha_P"] = float(sim.kalpha_interval(P[kidx].mean(2)))
        for k in range(len(eligible)):
            n = eligible[k]
            out["judges"][n]["S_clean"] = boot_S(sim, P[k].mean(1), cl_all)
            out["judges"][n]["anchor_share_clean"] = anchor([k])
    # J-Q sensitivity on ALL items (blocked included)
    if "J-Q" in eligible:
        cl_full = reindex(np.array([by_item[x]["idx"] for x in all_items]))
        Pq, Aq, Pfq, cq = arrays(all_items, key, judge_rows, ["J-Q"])
        if cq[0]:
            out["JQ_S_all_items"] = boot_S(sim, Pq[0].mean(1), cl_full)
            out["JQ_n_all_items"] = len(all_items)
    # suffixes
    suf = []
    if band in ("L2_PREFERRED", "L1_PREFERRED"):
        sg = 1 if band == "L2_PREFERRED" else -1
        if "S_greedy" in out and np.sign(out["S_greedy"][0]) == -sg:
            suf.append("decode-dependent")
        sq = out.get("S_quality")
        if sq and abs(sq[0]) >= 0.15 and (sq[1] > 0 or sq[2] < 0) and np.sign(sq[0]) == -sg:
            suf.append("card gain with a quality cost")
    out["suffixes"] = suf
    # length report
    out["length"] = {"early_stop_rate_matched": {s: float(np.mean([by_item[x]["hit_stop"][s] for x in all_items]))
                                                 for s in ("base", "L1", "L2")},
                     "items_with_any_early_stop_clean": int(early.sum()), "n_clean": I}
    out["pre_excluded"] = pre_excl
    return out


# ----------------------------------------------------------------------------- selftest (stub data only)
def selftest():
    import contextlib
    import io
    import os
    import tempfile
    sim = load_sim()
    ok, fails = 0, []

    def check(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name)
        ok += bool(cond)
        if not cond:
            fails.append(name)

    sys.path.insert(0, str(HERE))
    import panel_build as PB
    td = Path(tempfile.mkdtemp(prefix="ps-selftest-"))
    rng = np.random.default_rng(7)
    prompts = [{"idx": i, "qid": 100 + i, "prompt": f"stub request {i}"} for i in range(30)]
    PB.write_jsonl(td / "prompts.jsonl", prompts)
    items = []
    for i in range(30):
        for d in ("greedy", "t0.7"):
            items.append({"item": f"{i:02d}-{d}", "idx": i, "qid": 100 + i, "decode": d, "W": 40, "words": 34,
                          "texts": {s: f"STUB-{s}-{i}-{d} " + "x " * 30 for s in PB.ARMS},
                          "hit_stop": {s: bool(rng.random() < 0.25) for s in PB.ARMS}, "orig_words": {}})
    PB.write_jsonl(td / "matched.jsonl", items)
    lg = {"matched_sha256": sha(td / "matched.jsonl"), "hosted_allowed": True,
          "items": [{"item": it["item"], "blocked": n in (4, 9)} for n, it in enumerate(items)]}
    (td / "leakgate.json").write_text(json.dumps(lg))
    with contextlib.redirect_stdout(io.StringIO()):
        PB.build(argparse.Namespace(matched=str(td / "matched.jsonl"), leakgate=str(td / "leakgate.json"),
                                    prompts=str(td / "prompts.jsonl"), wsp3=None, out_dir=str(td / "pk")))

    class Oracle:  # stub judges that read the arm from the stub text marker (only possible on stub texts)
        def __init__(self, pL2, pAd, first_bias=0.0, seed=0):
            self.r = np.random.default_rng(seed)
            self.pL2, self.pAd, self.fb = pL2, pAd, first_bias

        def call(self, msg):
            A = msg.split("Response A:")[1].split("STUB-")[1].split("-")[0]
            B = msg.split("Response B:")[1].split("STUB-")[1].split("-")[0]
            pair = {A, B}
            if pair == {"L1", "L2"}:
                pA = self.pL2 if A == "L2" else 1 - self.pL2
            else:
                pA = self.pAd if A != "base" else 1 - self.pAd
            pA = min(max(pA + self.fb, 0.01), 0.99)
            v = "A" if self.r.random() < pA else "B"
            if self.r.random() < 0.05:
                v = "tie"
            return json.dumps({"voice": v, "quality": "A" if self.r.random() < 0.5 else "B"}), 0.0, "m", {}

    def run(tag, specs):
        paths = []
        for name, orc in specs:
            p = td / f"{tag}_{name}.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                PB.run_judge(argparse.Namespace(judge=name, packet=str(td / "pk" / "packet.jsonl"), out=str(p),
                                                url=None, cap=None, max_calls=None, progress=0), backend=orc)
            paths.append(f"{name}={p}")
        return score(argparse.Namespace(matched=str(td / "matched.jsonl"), leakgate=str(td / "leakgate.json"),
                                        key=str(td / "pk" / "key.json"), build=str(td / "pk" / "build.json"),
                                        judge=paths))

    r1 = run("strong", [("J-C", Oracle(0.9, 0.95, seed=1)), ("J-G", Oracle(0.9, 0.95, seed=2)),
                        ("J-Q", Oracle(0.9, 0.95, seed=3))])
    check("strong L2 preference + clear anchors -> L2 band, gates computed",
          r1["band_raw"] in ("L2_PREFERRED", "UNRESOLVED(alpha)") and r1["S"][0] > 0.5)
    check("scores use clean items only (58 of 60)", r1["n_clean"] == 58)
    check("J-Q sensitivity on all 60 items reported", "JQ_S_all_items" in r1 and r1["JQ_n_all_items"] == 60)
    r2 = run("null", [("J-C", Oracle(0.5, 0.95, seed=4)), ("J-G", Oracle(0.5, 0.95, seed=5)),
                      ("J-Q", Oracle(0.5, 0.95, seed=6))])
    check("no preference -> not a directional band", r2["band_raw"] in ("NO_DIFFERENCE", "UNRESOLVED"))
    r3 = run("noanchor", [("J-C", Oracle(0.5, 0.5, seed=7)), ("J-G", Oracle(0.5, 0.5, seed=8)),
                          ("J-Q", Oracle(0.5, 0.5, seed=9))])
    check("judges cannot tell adapters from base -> PANEL UNRELIABLE", r3["band_raw"] == "UNRELIABLE")
    r4 = run("bias", [("J-C", Oracle(0.5, 0.9, first_bias=0.4, seed=10)), ("J-G", Oracle(0.5, 0.9, seed=11)),
                      ("J-Q", Oracle(0.5, 0.9, seed=12))])
    check("position-driven judge excluded (share outside [0.35, 0.65])", "J-C" in r4["position_excluded"])
    r5 = run("onlylocal", [("J-Q", Oracle(0.9, 0.95, seed=13))])
    check("only the local judge -> PANEL UNRELIABLE (needs 2 judges incl. a hosted one)",
          r5["band_raw"] == "UNRELIABLE")
    # decide() parity: score's band equals panel_sim.decide on the same arrays computed independently
    key = json.loads((td / "pk" / "key.json").read_text())
    rows = {n: PB.load_jsonl(td / f"strong_{n}.jsonl") for n in ("J-C", "J-G", "J-Q")}
    clean = [it["item"] for n, it in enumerate(items) if n not in (4, 9)]
    P, A, Pf, comp = arrays(clean, key, rows, ["J-C", "J-G", "J-Q"])
    cl = reindex(np.array([int(x[:2]) for x in clean]))
    early = np.array([any(it["hit_stop"].values()) for n, it in enumerate(items) if n not in (4, 9)])
    b2, _ = sim.decide(P, Pf, A, cl, early, [True, True, False], np.random.default_rng(SEED_S), B_BOOT)
    check("band equals frozen panel_sim.decide() on the same data", b2 == r1["band_raw"] and comp.all())
    check("Pfirst is +1/-1 in both orders for every item", set(np.unique(Pf)) == {-1.0, 1.0} and
          (Pf.sum(1) == 0).all())
    # invalid > 10% -> excluded
    jr = PB.load_jsonl(td / "null_J-G.jsonl")
    for r in jr[:40]:
        r["valid"] = False
    PB.write_jsonl(td / "inv_J-G.jsonl", jr)
    (td / "inv_J-G.meta.json").write_text((td / "null_J-G.meta.json").read_text())
    r6 = score(argparse.Namespace(matched=str(td / "matched.jsonl"), leakgate=str(td / "leakgate.json"),
                                  key=str(td / "pk" / "key.json"), build=str(td / "pk" / "build.json"),
                                  judge=[f"J-C={td / 'null_J-C.jsonl'}", f"J-G={td / 'inv_J-G.jsonl'}",
                                         f"J-Q={td / 'null_J-Q.jsonl'}"]))
    check("judge with > 10% invalid calls excluded", "J-G" in r6["pre_excluded"])
    check("results carry no text", "STUB" not in json.dumps(r1))
    print(f"selftest: {ok}/{ok + len(fails)} PASS" + (f"; FAILED: {fails}" if fails else ""))
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched")
    ap.add_argument("--leakgate")
    ap.add_argument("--key")
    ap.add_argument("--build")
    ap.add_argument("--judge", action="append", default=[], help="NAME=path/to/judge.jsonl (meta beside it)")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    r = score(a)
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(json.dumps({k: r.get(k) for k in ("band", "band_raw", "retained", "S", "anchor_share", "alpha_anchor",
                                            "suffixes", "reason")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
