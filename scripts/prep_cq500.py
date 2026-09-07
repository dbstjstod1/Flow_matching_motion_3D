"""Index CQ500, apply the Thies selection, print the split -- and say WHAT WAS DROPPED AND WHY.

The selection throws away most of the archive (Thies keep 320 of 491 scans), so the interesting
output of this script is not the survivors but the CASUALTIES. A filter that silently eats half
the data reads exactly like a filter that is working.

    python scripts/prep_cq500.py --root data/CQ500
    python scripts/prep_cq500.py --root data/CQ500 --cache_volumes 8   # warm the .npy cache
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_cq500 import index_cq500, select_series, split_patients


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/CQ500")
    ap.add_argument("--thin_mm", type=float, default=0.7)
    ap.add_argument("--count_tol", type=float, default=0.5)
    ap.add_argument("--min_slices", type=int, default=64)
    ap.add_argument("--split_counts", type=int, nargs=3, default=(150, 50, 120))
    ap.add_argument("--cache_volumes", type=int, default=0,
                    help="also load+cache this many volumes per split (0 = index only)")
    ap.add_argument("--shape", type=int, nargs=3, default=(256, 256, 256))
    ap.add_argument("--voxel_mm", type=float, default=1.0)
    args = ap.parse_args()

    recs = index_cq500(args.root)
    pats = sorted({r["patient"] for r in recs})
    print(f"\nARCHIVE   {len(recs)} series | {len(pats)} patients "
          f"(id {min(pats)}..{max(pats)}) | {sum(r['n_slices'] for r in recs)} slices")

    th = Counter(round(r["thickness_mm"], 3) for r in recs)
    print("\nslice thickness [mm] -> series")
    for t, n in sorted(th.items()):
        print(f"  {t:6.3f}  {n:5d} {'#' * min(n // 20, 40)}")

    thin = [r for r in recs if r["thickness_mm"] <= args.thin_mm
            and r["n_slices"] >= args.min_slices]
    thin_p = {r["patient"] for r in thin}
    print(f"\n[1] thin-slice filter (<= {args.thin_mm} mm, >= {args.min_slices} slices)")
    print(f"    kept {len(thin):4d} / {len(recs)} series | "
          f"{len(thin_p):3d} / {len(pats)} patients still have one")
    lost = sorted(set(pats) - thin_p)
    print(f"    patients with NO thin series: {len(lost)}  {lost[:12]}{' ...' if len(lost) > 12 else ''}")

    n = np.array([r["n_slices"] for r in thin])
    med = float(np.median(n))
    keep = np.abs(n / med - 1.0) <= args.count_tol
    print(f"\n[2] slice-count outlier cut (median {med:.0f}, tol +-{args.count_tol:.0%} "
          f"-> keep {med * (1 - args.count_tol):.0f}..{med * (1 + args.count_tol):.0f} slices)")
    print(f"    slice counts: min {n.min()} p5 {np.percentile(n, 5):.0f} "
          f"med {med:.0f} p95 {np.percentile(n, 95):.0f} max {n.max()}")
    print(f"    kept {int(keep.sum()):4d} / {len(thin)} series "
          f"({int((~keep).sum())} dropped: {int((n < med * (1 - args.count_tol)).sum())} too few, "
          f"{int((n > med * (1 + args.count_tol)).sum())} too many)")

    sel = select_series(recs, thin_mm=args.thin_mm, count_tol=args.count_tol,
                        min_slices=args.min_slices)
    print(f"\n[3] one series per patient (most slices wins)")
    print(f"    -> {len(sel)} patients   [Thies got 320 of 491]")
    sp = np.array([r["spacing_mm"][0] for r in sel])
    ns = np.array([r["n_slices"] for r in sel])
    tk = np.array([r["thickness_mm"] for r in sel])
    print(f"    thickness  {tk.min():.3f}..{tk.max():.3f} mm (mean {tk.mean():.3f})")
    print(f"    in-plane   {sp.min():.3f}..{sp.max():.3f} mm (mean {sp.mean():.3f})"
          f"   [Thies: 0.38..0.58, mean 0.472]")
    print(f"    slices     {ns.min()}..{ns.max()} (mean {ns.mean():.0f}) "
          f"-> {ns.mean() * tk.mean():.0f} mm of head on average")

    parts = split_patients(sel, counts=args.split_counts)
    print(f"\n[4] sequential patient-level split (no RNG)")
    for k, v in parts.items():
        ids = [r["patient"] for r in v]
        print(f"    {k:5s} {len(v):3d} patients | CQ500CT{ids[0]} .. CQ500CT{ids[-1]}")
    allids = [r["patient"] for v in parts.values() for r in v]
    assert len(allids) == len(set(allids)), "patient leaked across splits"
    print("    disjoint: OK")

    if args.cache_volumes:
        import torch
        from fm3d.dataset_cq500 import CQ500Generator
        from fm3d.geometry_3d import ConeBeam3DConfig
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for split in ("train", "val", "test"):
            g = CQ500Generator(args.root, cfg=ConeBeam3DConfig.thies(), device=dev, split=split,
                               shape=tuple(args.shape), voxel_mm=args.voxel_mm,
                               thin_mm=args.thin_mm, count_tol=args.count_tol,
                               split_counts=tuple(args.split_counts), verbose=False)
            for i in range(min(args.cache_volumes, len(g.records))):
                v = g.volume(i)
                hu = float((v.max() / g.mu_water - 1.0) * 1000.0)
                print(f"    cached {split} [{i}] patient {g.records[i]['patient']:3d} "
                      f"shape {tuple(v.shape[2:])} max {hu:.0f} HU")


if __name__ == "__main__":
    main()
