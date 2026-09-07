"""Paired cross-arm comparison over the test30 cohort (2026-08-18).

THE THREE UNIFIED-LOOP ARMS (user's framing: everything identical -- estimator, CG data step,
schedule, measurements -- only the PRIOR differs):

    fm       FM velocity net, physics (data) bridge   -- the method
    linear   FM velocity net, pixel-linear bridge     -- bridge-shape ablation (B1)
    w3dm     JRM-ADM's x0-DDPM, renoise adapter       -- prior-class benchmark (tmax 300)

Every run dir holds pNN/result.pt from run_posterior3d on the standing (split=test, run=i,
seed=1000+i) triples. Pairing is ASSERTED via theta_true (same discipline as
cmp_thies_vs_ours: a mismatched seed still converges and still prints a plausible table, so
it raises rather than warns).

Reports, per arm: mean +- std of x_t and OUTPUT metrics vs GT (rigid-aligned, the honest
number under the SE(3) gauge) and vs static FDK; paired Wilcoxon p and win counts against the
reference arm (--ref, default fm). Missing patients are listed, not silently skipped.

    python scripts/cmp_arms.py \
        --arm fm=data/fm3d_test30_databridge \
        --arm linear=data/linbridge_test30 \
        --arm w3dm=data/w3dm_test30
"""
import argparse
import os

import numpy as np
import torch


def _wilcoxon(d: np.ndarray):
    """Two-sided Wilcoxon signed-rank; returns (stat, p) or (nan, nan) without scipy."""
    try:
        from scipy.stats import wilcoxon
        s, p = wilcoxon(d)
        return float(s), float(p)
    except Exception:
        return float("nan"), float("nan")


KEYS = [("final_xt", "x_t vs GT"), ("final", "output vs GT"),
        ("final_xt_s", "x_t vs sFDK"), ("final_s", "output vs sFDK")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="name=dir, repeatable; first is the reference unless --ref is given")
    ap.add_argument("--ref", default=None, help="reference arm name (default: first --arm)")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    arms = {}
    for a in args.arm:
        name, d = a.split("=", 1)
        arms[name] = d
    ref = args.ref or next(iter(arms))
    assert ref in arms, f"--ref {ref!r} not among arms {list(arms)}"

    rows = {name: {} for name in arms}
    missing = []
    theta_ref = {}
    for i in range(args.n):
        tag = f"p{i:02d}"
        for name, d in arms.items():
            p = os.path.join(d, tag, "result.pt")
            if not os.path.exists(p):
                missing.append((name, tag))
                continue
            o = torch.load(p, map_location="cpu", weights_only=False)
            # pairing assertion across arms
            tt = o["theta_true"]
            if tag not in theta_ref:
                theta_ref[tag] = tt
            else:
                dev = float((tt - theta_ref[tag]).abs().max())
                if dev > 1e-6:
                    raise SystemExit(f"{name}/{tag}: NOT the same motion as the other arms "
                                     f"(max |dtheta| {dev:.3e}) -- cohorts are not paired.")
            rows[name][tag] = o

    if missing:
        print("MISSING (excluded pairwise):")
        for name, tag in missing:
            print(f"  {name}/{tag}")

    common = sorted(set.intersection(*(set(r) for r in rows.values())))
    print(f"\npaired patients: {len(common)} / {args.n}   arms: {list(arms)}   ref: {ref}\n")

    for key, label in KEYS:
        print(f"== {label} (aligned) " + "=" * 46)
        ref_ssim = np.array([rows[ref][t][key]["ssim_aligned"] for t in common])
        ref_psnr = np.array([rows[ref][t][key]["psnr_aligned"] for t in common])
        for name in arms:
            ssim = np.array([rows[name][t][key]["ssim_aligned"] for t in common])
            psnr = np.array([rows[name][t][key]["psnr_aligned"] for t in common])
            line = (f"  {name:8s} PSNR {psnr.mean():6.2f} +- {psnr.std():4.2f}   "
                    f"SSIM {ssim.mean():.4f} +- {ssim.std():.4f}")
            if name != ref:
                _, p_s = _wilcoxon(ref_ssim - ssim)
                _, p_p = _wilcoxon(ref_psnr - psnr)
                wins = int((ref_ssim > ssim).sum())
                line += (f"   | vs {ref}: dSSIM {ssim.mean() - ref_ssim.mean():+.4f} "
                         f"(p={p_s:.2g}), dPSNR {psnr.mean() - ref_psnr.mean():+.2f} dB "
                         f"(p={p_p:.2g}), {ref} wins {wins}/{len(common)}")
            print(line)
        print()


if __name__ == "__main__":
    main()
