"""PURE prior ODE on a motion-corrupted scan -- no data-prox, no TV, no motion estimation.

The control that proves the FM prior alone moves x_t. Everything the posterior loop adds is
switched off (alpha=0, kappa=0), so the only thing acting on the carried volume is

    x <- x + dt * v_FM(x, t),   t = k/N

MEASURED (val 0, 500k ckpt, uniform K=2): per-step |dx|/|x| = 0.72 -> 0.86 -> 0.71 % and the
CUMULATIVE displacement from x0 reaches 29.45 % -- i.e. ~96 % of the accumulated path length, so
the prior flows MONOTONICALLY with essentially no cancellation. Contrast the same measurement
with the soft-DC on at alpha=0.1: 101 % of path length produced only 15.8 % of net displacement
(16 % efficiency) -- the fixed-size normalized data step overshoots and ping-pongs. That is why
x_t looked static in the posterior montages, and it is a soft-DC problem, not a prior problem.

Panels: input (cold FDK) | x_t (pure ODE) | static FDK (what the bridge was anchored to) | GT.

    CUDA_VISIBLE_DEVICES=0 python scripts/exp_pure_ode.py
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
from fm3d.unet_3d import UNet3D
from run_posterior3d import fm_predict


def montage(path, panels, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = 0.0, 0.05                       # head bone window in mu; see run_posterior3d.montage
    n = len(panels)
    D, H, _ = panels[0][1].shape
    zc, yc = D // 2, H // 2
    fig, ax = plt.subplots(2, n, figsize=(3.3 * n, 6.6))
    for c, (name, v) in enumerate(panels):
        ax[0, c].imshow(v[zc].cpu(), cmap="gray", vmin=lo, vmax=hi)
        ax[0, c].set_title(name, fontsize=9)
        ax[1, c].imshow(v[:, yc].cpu(), cmap="gray", vmin=lo, vmax=hi, aspect="auto")
        for r in range(2):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    ax[0, 0].set_ylabel("axial", fontsize=9)
    ax[1, 0].set_ylabel("coronal", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/fm3d_cq500/ckpt_iter500000.pth")
    ap.add_argument("--split", default="val")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--out", default="data/pure_ode_val0")
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--every", type=int, default=5, help="save a montage every k steps")
    ap.add_argument("--motion_kind", default="mixed")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--blend", default="uniform", choices=["uniform", "hann"])
    ap.add_argument("--patch_offsets", type=int, default=2)
    args = ap.parse_args()

    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    ca = ck["args"]
    cfg = ConeBeam3DConfig.thies(n_views=ca["views"])
    gen = CQ500Generator(ca["root"], cfg=cfg, device=dev, split=args.split,
                         shape=tuple(ca["shape"]), voxel_mm=1.0)
    spacing = (gen.dz, gen.dy, gen.dx)
    meas = measured_region_mask(gen.shape, spacing, cfg, device=dev)
    model = UNet3D(in_ch=int(ck["ema"]["in_conv.weight"].shape[1]), base=ca["base"]).to(dev).eval()
    model.load_state_dict(ck["ema"])
    for q in model.parameters():
        q.requires_grad_(False)

    gt = gen.volume(args.run)
    theta_true = make_motion(args.motion_kind, cfg.n_views, device=dev, seed=args.seed)
    with torch.no_grad():
        y = gen.simulate(args.run, params_to_Pmot(theta_true, gen.P_nom)[None])
        x = gen.fdk(y, gen.P_nom[None])[0]                                  # cold start
        static = gen.fdk(gen.simulate(args.run, gen.P_nom[None]), gen.P_nom[None])[0]   # bridge target
    gt3, x0 = gt[0, 0], x.clone()
    patch = ca["patch"]
    gtor = torch.Generator(device=dev).manual_seed(args.seed)

    m0 = aligned_metrics(x0, gt3, spacing, mask=meas, iters=200)
    print(f"PURE ODE: alpha=0 (no data-prox), kappa=0 (no TV), no motion estimation | "
          f"{args.n_steps} steps, blend={args.blend} K={args.patch_offsets}")
    print(f"cold  {m0['psnr_aligned']:.2f} dB / SSIM {m0['ssim_aligned']:.3f}")
    print(f"{'step':>4} {'t':>5} {'perstep%':>9} {'cumul%':>8} {'aligned dB':>11} {'SSIM':>6}")

    N = args.n_steps
    for k in range(N):
        t, dt = k / N, 1.0 / N
        xo = x.clone()
        x = fm_predict(model, gen, x, t, dt, patch, context="global",
                       n_offsets=args.patch_offsets, generator=gtor, blend=args.blend)
        if k % args.every == 0 or k == N - 1:
            m = aligned_metrics(x, gt3, spacing, mask=meas, iters=150)
            ps = 100 * float((x - xo).norm() / xo.norm())
            cu = 100 * float((x - x0).norm() / x0.norm())
            print(f"{k:>4} {t:>5.2f} {ps:>9.2f} {cu:>8.2f} {m['psnr_aligned']:>11.2f} "
                  f"{m['ssim_aligned']:>6.3f}", flush=True)
            montage(os.path.join(args.out, f"step{k:03d}.png"),
                    [("input (cold FDK)", x0), ("x_t (pure ODE)", x),
                     ("static FDK (target)", static), ("ground truth", gt3)],
                    f"PURE ODE  step {k}  t={t:.2f}   x_t aligned {m['psnr_aligned']:.2f} dB / "
                    f"SSIM {m['ssim_aligned']:.3f}   (cold {m0['psnr_aligned']:.2f})")
    mf = aligned_metrics(x, gt3, spacing, mask=meas, iters=300)
    print(f"\nFINAL pure-ODE x_t: {mf['psnr_aligned']:.2f} dB / SSIM {mf['ssim_aligned']:.3f} "
          f"(cold {m0['psnr_aligned']:.2f} / {m0['ssim_aligned']:.3f})")
    montage(os.path.join(args.out, "final.png"),
            [("input (cold FDK)", x0), ("x_t (pure ODE, t=1)", x),
             ("static FDK (target)", static), ("ground truth", gt3)],
            f"PURE ODE FINAL   cold {m0['psnr_aligned']:.2f} -> {mf['psnr_aligned']:.2f} dB / "
            f"SSIM {m0['ssim_aligned']:.3f} -> {mf['ssim_aligned']:.3f}")
    torch.save({"x_final": x.cpu(), "x0": x0.cpu(), "final": mf, "cold": m0},
               os.path.join(args.out, "pure_ode.pt"))
    print(f"-> {args.out}/  (step*.png, final.png, pure_ode.pt)")


if __name__ == "__main__":
    main()
