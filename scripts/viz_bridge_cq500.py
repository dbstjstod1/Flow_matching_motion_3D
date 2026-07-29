"""The GEOMETRY BRIDGE on real CQ500 heads, rendered so it can be judged BY EYE.

The bridge is the manifold the FM prior trains on and the inference ODE walks:

    x_t = FDK(y, P_nom @ T(t * theta))        t=0 the uncorrected recon, t=1 the clean one

Every intermediate state is a REAL reconstruction under a partially corrected geometry -- not a
pixel-space blend. This script renders it at t = 0, 1/4, 1/2, 3/4, 1 on an actual CQ500 patient in
the standard geometry, so the two things that only fail visibly can be checked: that the path is
MONOTONE (streaks retreat, they do not just move), and that t=1 is a clean head and not a mirrored
or sheared one (a sign error in the motion algebra reconstructs a plausible-looking MIRROR).

    python scripts/viz_bridge_cq500.py --patients 3
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig, measured_region_mask
from fm3d.rigid_motion import make_motion, params_to_Pmot

TS = (0.0, 0.25, 0.5, 0.75, 1.0)


def psnr(a, b, m):
    e = (a - b)[m]
    rng = float(b[m].max() - b[m].min())
    return float(20 * np.log10(rng / (e.pow(2).mean().sqrt().item() + 1e-12)))


def montage(path, gt, xs, ts, ps, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D, H, W = gt.shape
    lo, hi = 0.0, 1.4 * 0.02                       # mu window ~ [-1000, 400] HU
    cols = [("ground truth", gt, None)] + [(f"t={t:.2f}", x, p) for t, x, p in zip(ts, xs, ps)]
    fig, ax = plt.subplots(3, len(cols), figsize=(3.0 * len(cols), 9.2))
    for c, (name, v, p) in enumerate(cols):
        v = v.cpu()
        for r, sl in enumerate([v[D // 2], v[:, H // 2], v[:, :, W // 2]]):
            ax[r, c].imshow(sl, cmap="gray", vmin=lo, vmax=hi,
                            aspect="auto" if r else "equal", origin="lower")
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        ax[0, c].set_title(name + ("" if p is None else f"\n{p:.2f} dB"), fontsize=10)
    for r, n in enumerate(["axial", "coronal", "sagittal"]):
        ax[r, 0].set_ylabel(n, fontsize=10)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/CQ500")
    ap.add_argument("--out", default="data/bridge_cq500")
    ap.add_argument("--split", default="train")
    ap.add_argument("--patients", type=int, default=3)
    ap.add_argument("--shape", type=int, nargs=3, default=(256, 256, 256))
    ap.add_argument("--views", type=int, default=360)
    ap.add_argument("--motion_kind", default="akima")     # the literature's model (Thies et al.)
    ap.add_argument("--anchor", default="static", choices=["static", "gt", "none"])
    ap.add_argument("--trans_mm", type=float, default=10.0)    # PEAK-TO-PEAK (Thies evals at 5)
    ap.add_argument("--rot_deg", type=float, default=10.0)     # PEAK-TO-PEAK
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)
    cfg = ConeBeam3DConfig.thies(n_views=args.views)
    gen = CQ500Generator(args.root, cfg=cfg, device=dev, split=args.split,
                         shape=tuple(args.shape), voxel_mm=1.0)
    print(f"geometry: SOD {cfg.SOD} SDD {cfg.SDD} | {cfg.nu}x{cfg.nv} @ {cfg.du} mm | "
          f"{cfg.n_views} views | FOV {cfg.fov_diameter_mm():.0f} mm, "
          f"axial {cfg.axial_coverage_mm():.0f} mm")
    print(f"grid {gen.shape} @ 1 mm | self-normalized FDK", flush=True)

    meas = measured_region_mask(gen.shape, (gen.dz, gen.dy, gen.dx), cfg, device=dev)

    for i in range(min(args.patients, len(gen.records))):
        pid = gen.records[i]["patient"]
        gt = gen.volume(i)                                              # (1,1,D,H,W) mu
        # amplitudes are per-axis; the literature's evaluation motion is 5 mm / 5 deg (Thies,
        # JRM-ADM), applied isotropically here.
        theta = make_motion(args.motion_kind, cfg.n_views, device=dev, seed=args.seed + i,
                            trans_mm=(args.trans_mm,) * 3, rot_deg=(args.rot_deg,) * 3)
        with torch.no_grad():
            y = gen.project(gt, params_to_Pmot(theta, gen.P_nom)[None])  # the corrupted scan

            # a STATIC scan through the same operator: the ceiling the bridge can reach at t=1,
            # and the number that separates "motion is left" from "FDK/cone artefacts".
            y0 = gen.project(gt, gen.P_nom[None])
            static = gen.fdk(y0, gen.P_nom[None])[0]

            # THE ANCHOR (see train_fm3d.bridge_pair): the bare geometry bridge's endpoint is
            # NOT the clean image -- FDK handed the TRUE theta still sits 1-3 dB under a static
            # scan, because it is an inverse derived for a circular EQUIANGULAR orbit. The anchor
            # detrends the path so t=1 lands on the static reconstruction BY CONSTRUCTION, while
            # t=0 stays exactly the cold start.
            x1_geo = gen.fdk(y, params_to_Pmot(theta, gen.P_nom)[None])[0]
            dlt = ((static if args.anchor == "static" else gt[0, 0]) - x1_geo
                   if args.anchor != "none" else 0.0)

            xs, ps = [], []
            for t in TS:
                x = gen.fdk(y, params_to_Pmot(t * theta, gen.P_nom)[None])[0] + t * dlt
                xs.append(x)
                ps.append(psnr(x, gt[0, 0], meas))
                print(f"  patient {pid:3d}  t={t:.2f}  {ps[-1]:6.2f} dB", flush=True)

        p_static = psnr(static, gt[0, 0], meas)
        mono = all(b >= a - 0.05 for a, b in zip(ps, ps[1:]))
        print(f"  patient {pid:3d}  static FDK (no motion) {p_static:6.2f} dB  | "
              f"bridge {'MONOTONE' if mono else '*** NOT MONOTONE ***'} | "
              f"t=1 is {p_static - ps[-1]:+.2f} dB from the static ceiling", flush=True)
        montage(os.path.join(args.out, f"bridge_p{pid:03d}.png"), gt[0, 0], xs, TS, ps,
                f"CQ500 patient {pid} | geometry bridge (anchor={args.anchor}) | motion "
                f"{args.trans_mm} mm / {args.rot_deg} deg ({args.motion_kind}) | "
                f"static FDK {p_static:.2f} dB")


if __name__ == "__main__":
    main()
