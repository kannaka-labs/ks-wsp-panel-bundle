#!/usr/bin/env python3
"""Pre-freeze simulation for 32-panel-prereg.md (the standing pre-freeze power check from 13-K-results.md).

Synthetic judges only; no sample, no corpus row, no network. It applies the panel's decision rules (32 §6-§7, as
revised after the 2026-11-09 review: position-bias exclusion, alpha on the anchors run in both orders, leave-one-
judge-out on the anchor gate and the directional bands, the no-early-stop length gate) to simulated panels with
3 judges and 30 prompt clusters, at true S = 0 and S = 0.15, and reports how often each band is reached and the
90% CI half-width of S.

Generative model (an assumption, stated in 32 §11):
  item contrast d_i = S + u_prompt + e_item (clipped to +-0.9); judge k sees d_i + eta_ik;
  P(tie) = tau; otherwise P(choose L2) = (1 + d_ik)/2 + beta_k * (+1 if L2 shown first else -1).
  Anchors identical with adapter advantage A (win share (1+A)/2) in place of S. Early-stop flag per item ~ Bern(p_es),
  independent of everything (a null length model).
"""
import json
import sys

import numpy as np

PB_LO, PB_HI = 0.35, 0.65
ALPHA_MIN = 0.40
DELTA = 0.15
ANCHOR_SHARE, ANCHOR_LB = 0.60, 0.50


def kalpha_interval(v):
    """Krippendorff's alpha, interval metric, complete data. v: (m coders, N units)."""
    m, N = v.shape
    n = m * N
    within = (2 * m * (v ** 2).sum(0) - 2 * v.sum(0) ** 2).sum() / (m - 1)
    allp = 2 * n * (v ** 2).sum() - 2 * v.sum() ** 2
    Do = within / n
    De = allp / (n * (n - 1))
    return 1.0 - Do / De if De > 0 else float("nan")


def boot_mean(scores, clusters, nclu, rng, B):
    """Point estimate and percentile 90% CI of the item mean, prompt-cluster bootstrap."""
    s = np.bincount(clusters, weights=scores, minlength=nclu)
    c = np.bincount(clusters, minlength=nclu).astype(float)
    idx = rng.integers(0, nclu, size=(B, nclu))
    bs = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    return scores.mean(), np.percentile(bs, 5), np.percentile(bs, 95)


def decide(P, Pfirst, A, clusters, early, hosted, rng, B):
    """P: (J, I, 2) choices +1 L2 / -1 L1 / 0 tie; Pfirst: (I, 2) +1 if L2 shown first in that order.
    A: (J, U, 2) anchor choices +1 adapter / -1 base / 0 (U = 2 I units: A1 and A2), unit u -> item u % I.
    Returns (band, info)."""
    J, I, _ = P.shape
    nclu = clusters.max() + 1
    # position bias on P
    keep = []
    for k in range(J):
        nt = P[k] != 0
        first = (P[k] == Pfirst) & nt
        share = first.sum() / max(nt.sum(), 1)
        if PB_LO <= share <= PB_HI:
            keep.append(k)
    info = {"retained": keep}
    if len(keep) < 2 or not any(hosted[k] for k in keep):
        return "UNRELIABLE", info
    aclu = np.concatenate([clusters, clusters])

    def anchor_ok(ks, mask_items=None):
        a = A[ks]  # (k, U, 2)
        units = np.arange(A.shape[1])
        if mask_items is not None:
            units = units[mask_items[units % I]]
        w = (a[:, units] == 1).sum((0, 2)) + 0.5 * (a[:, units] == 0).sum((0, 2))
        n = len(ks) * 2
        sh = w / n
        est, lo, hi = boot_mean(sh, aclu[units], nclu, rng, B)
        return est >= ANCHOR_SHARE and lo > ANCHOR_LB, est, lo

    ok, est, lo = anchor_ok(keep)
    info["anchor"] = (round(est, 3), round(lo, 3))
    for k in keep:  # LOJO on the anchor gate
        if not anchor_ok([x for x in keep if x != k])[0]:
            ok = False
    if not ok:
        return "UNRELIABLE", info
    alpha = kalpha_interval(A[keep].mean(2))
    info["alpha_anchor"] = round(float(alpha), 3)

    def S_of(ks, items=None):
        sc = P[ks].mean(2).mean(0)  # item score averaged over judges
        it = np.arange(I) if items is None else items
        return boot_mean(sc[it], clusters[it], nclu, rng, B)

    S, lo, hi = S_of(keep)
    info["S"] = (round(S, 3), round(lo, 3), round(hi, 3))
    if (S >= DELTA and lo > 0) or (S <= -DELTA and hi < 0):
        sg = np.sign(S)
        if not alpha >= ALPHA_MIN:
            return "UNRESOLVED(alpha)", info
        for k in keep:
            r = [x for x in keep if x != k]
            s2, l2, h2 = S_of(r)
            if not (np.sign(s2) == sg and (l2 > 0 or h2 < 0)):
                return "UNRESOLVED(family)", info
        ne = np.where(~early)[0]
        s3, l3, h3 = S_of(keep, ne)
        if not (np.sign(s3) == sg and (l3 > 0 or h3 < 0)) or not anchor_ok(keep, ~early)[0]:
            return "UNRESOLVED(length)", info
        return ("L2" if sg > 0 else "L1") + "_PREFERRED", info
    if -DELTA < lo and hi < DELTA:
        return "NO_DIFFERENCE", info
    return "UNRESOLVED", info


def simulate(S, Aadv, su, se, eta, tau, betas, p_es, nprompt, decodes, rng, B):
    J = len(betas)
    I = nprompt * decodes
    clusters = np.repeat(np.arange(nprompt), decodes)

    def latent(mu):
        u = rng.normal(0, su, nprompt)[clusters]
        return np.clip(mu + u + rng.normal(0, se, I), -0.9, 0.9)

    def calls(d, first):  # d (I,), first (I,2) -> (J, I, 2)
        out = np.zeros((J, I, 2))
        for k in range(J):
            dk = np.clip(d + rng.normal(0, eta, I), -0.9, 0.9)
            for o in range(2):
                p = np.clip((1 + dk) / 2 + betas[k] * first[:, o], 0.02, 0.98)
                ch = np.where(rng.random(I) < p, 1, -1)
                ch[rng.random(I) < tau] = 0
                out[k, :, o] = ch
        return out

    Pfirst = np.stack([-np.ones(I), np.ones(I)], 1)  # order 0: L1 first, order 1: L2 first
    P = calls(latent(S), Pfirst)
    dA = np.concatenate([latent(Aadv), latent(Aadv)])
    Afirst = np.stack([-np.ones(2 * I), np.ones(2 * I)], 1)
    # anchors: reuse the call model on 2I units
    out = np.zeros((J, 2 * I, 2))
    for k in range(J):
        dk = np.clip(dA + rng.normal(0, eta, 2 * I), -0.9, 0.9)
        for o in range(2):
            p = np.clip((1 + dk) / 2 + betas[k] * Afirst[:, o], 0.02, 0.98)
            ch = np.where(rng.random(2 * I) < p, 1, -1)
            ch[rng.random(2 * I) < tau] = 0
            out[k, :, o] = ch
    early = rng.random(I) < p_es
    hosted = [True, True, False]
    return decide(P, Pfirst, out, clusters, early, hosted, rng, B)


def main():
    nsim = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    B = 1000
    rng = np.random.default_rng(20261114)
    betas = [0.08, -0.05, 0.12]
    scen = []
    for strat, decodes in (("pooled 60 items / 30 prompts", 2), ("greedy only 30 items", 1)):
        for het, su, se, eta in (("low", 0.10, 0.10, 0.25), ("high", 0.30, 0.25, 0.25), ("sharp", 0.50, 0.40, 0.10)):
            for Aadv in (0.5, 0.3):
                for S in (0.0, 0.15, 0.30):
                    bands, hw, alph = {}, [], []
                    for _ in range(nsim):
                        b, info = simulate(S, Aadv, su, se, eta, 0.05, betas, 0.35, 30, decodes, rng, B)
                        bands[b] = bands.get(b, 0) + 1
                        if "S" in info:
                            hw.append((info["S"][2] - info["S"][1]) / 2)
                        if "alpha_anchor" in info:
                            alph.append(info["alpha_anchor"])
                    scen.append({"stratum": strat, "heterogeneity": het, "sd_prompt": su, "sd_item": se, "eta": eta, "anchor_share": (1 + Aadv) / 2, "S": S,
                                 "bands": {k: round(v / nsim, 3) for k, v in sorted(bands.items())},
                                 "ci_halfwidth_median": round(float(np.median(hw)), 3) if hw else None,
                                 "alpha_anchor_median": round(float(np.median(alph)), 3) if alph else None})
                    print(json.dumps(scen[-1]), flush=True)
    json.dump({"nsim": nsim, "boot_B": B, "betas": betas, "tau": 0.05, "p_early": 0.35,
               "scenarios": scen}, open(sys.argv[2] if len(sys.argv) > 2 else "panel_sim.json", "w"), indent=1)


if __name__ == "__main__":
    main()
