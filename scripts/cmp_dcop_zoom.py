"""ROI-magnified by-eye comparison of the data-step runs, because the 4-panel montages are too
small to judge on: at 13 inches wide and 8 panels, a 256-wide slice is drawn at ~1/3 scale and
exactly the thing that decides this -- residual streaks and whether bone edges are doubled or
crisp -- is below the figure's own resolution.

Rebuilds each run's OUTPUT (FDK(theta_hat)) and CARRIED (x_t) volumes from the saved theta, aligns
them to the GT the same way the loop did, and draws a zoomed crop at native pixel scale.

    python scripts/cmp_dcop_zoom.py --runs A_adj D_fdk --out data/figures/dcop_zoom.png
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig, measured_region_mask
from fm3d.reg_metric import aligned_metrics
from fm3d.rigid_motion import make_motion, params_to_Pmot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/fm3d_cq500/ckpt_iter500000.pth")
    ap.add_argument("--runs", nargs="+", default=["A_adj", "D_fdk"])
    ap.add_argument("--dir", default="data")
    ap.add_argument("--split", default="val")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--motion_kind", default="mixed")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--crop", type=int, default=110, help="half-width of the zoom box, voxels")
    ap.add_argument("--out", default="data/figures/dcop_zoom.png")
    args = ap.parse_args()

    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    ca = ck["args"]
    cfg = ConeBeam3DConfig.thies(n_views=ca["views"])
    gen = CQ500Generator(ca["root"], cfg=cfg, device=dev, split=args.split,
                         shape=tuple(ca["shape"]), voxel_mm=1.0)
    spacing = (gen.dz, gen.dy, gen.dx)
    meas = measured_region_mask(gen.shape, spacing, cfg, device=dev)

    # the SAME corrupted scan every run saw (same motion_kind + seed as run_posterior3d's defaults)
    gt = gen.volume(args.run)
    theta_true = make_motion(args.motion_kind, cfg.n_views, device=dev, seed=args.seed)
    with torch.no_grad():
        y = gen.simulate(args.run, params_to_Pmot(theta_true, gen.P_nom)[None])
    gt3 = gt[0, 0]

    panels = []
    for tag in args.runs:
        p = os.path.join(args.dir, f"dcop_{tag}", "result.pt")
        if not os.path.isfile(p):
            print(f"skip {tag}: no {p}")
            continue
        r = torch.load(p, map_location=dev, weights_only=False)
        with torch.no_grad():
            x_fdk = gen.fdk(y, params_to_Pmot(r["theta"].to(dev), gen.P_nom)[None])[0]
        m, al = aligned_metrics(x_fdk, gt3, spacing, mask=meas, iters=300, return_aligned=True)
        # every panel states its OWN number -- a figure-level title cannot say which volume it
        # grades, and the ranking of these panels is the whole point of the figure
        panels.append((f"{tag}  FDK(theta_hat)\n{m['psnr_aligned']:.2f} dB / SSIM "
                       f"{m['ssim_aligned']:.3f}", al))
        print(f"{tag}: FDK(theta) {m['psnr_aligned']:.2f} dB / {m['ssim_aligned']:.3f}",
              flush=True)
    panels.append(("ground truth", gt3))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D, H, W = gt3.shape
    zc, yc, xc = D // 2, H // 2, W // 2
    c = args.crop
    ys, xs = slice(yc - c, yc + c), slice(xc - c, xc + c)
    n = len(panels)
    fig, ax = plt.subplots(2, n, figsize=(4.2 * n, 8.6))
    for j, (name, v) in enumerate(panels):
        ax[0, j].imshow(v[zc, ys, xs].cpu(), cmap="gray", vmin=0.0, vmax=0.05)
        ax[0, j].set_title(name, fontsize=10)
        ax[1, j].imshow(v[:, yc, xs].cpu(), cmap="gray", vmin=0.0, vmax=0.05, aspect="auto")
        for i in range(2):
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    ax[0, 0].set_ylabel("axial (zoom)", fontsize=10)
    ax[1, 0].set_ylabel("coronal", fontsize=10)
    fig.suptitle("data-step comparison, OUTPUT = FDK(theta_hat), zoomed", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
