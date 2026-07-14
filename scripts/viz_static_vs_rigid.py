"""Does a GLOBAL rigid motion really cost NOTHING? The claim was "34.42 vs 34.34 dB". Look at it.

A rigid transform applied identically to every view moves the source circle rigidly -- it is still
a circle, still equiangular. Reconstructing with the SAME moved matrices should therefore return
the object in its own frame, at exactly static quality. That is the load-bearing claim behind
"the projection matrix absorbs 6-DoF motion exactly", and a PSNR two decimal places apart is not
proof: a reconstruction can be mirrored, shifted by a voxel, or subtly blurred and still score
within 0.1 dB.

So this renders it: static FDK, the global-rigid oracle FDK, and their DIFFERENCE at 20x gain,
next to the difference between static and a genuinely broken reconstruction (the same rigid motion
reconstructed on the NOMINAL geometry) for scale. If the first difference map is structureless
noise and the second is a head, the claim holds.

    python scripts/viz_static_vs_rigid.py
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
from fm3d.rigid_motion import params_to_Pmot


def psnr(a, b, m):
    e = (a - b)[m]
    rng = float(b[m].max() - b[m].min())
    return float(20 * np.log10(rng / (e.pow(2).mean().sqrt().item() + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/CQ500")
    ap.add_argument("--out", default="data/diag_oracle")
    ap.add_argument("--shape", type=int, nargs=3, default=(256, 256, 256))
    ap.add_argument("--views", type=int, default=360)
    ap.add_argument("--gain", type=float, default=20.0)
    args = ap.parse_args()

    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)
    cfg = ConeBeam3DConfig.thies(n_views=args.views)
    gen = CQ500Generator(args.root, cfg=cfg, device=dev, split="train",
                         shape=tuple(args.shape), voxel_mm=1.0, verbose=False)
    meas = measured_region_mask(gen.shape, (1.0, 1.0, 1.0), cfg, device=dev)
    gt = gen.volume(0)
    g3 = gt[0, 0]
    V = cfg.n_views

    # the global rigid transform: 6 mm of translation and 5 deg about the gantry axis, on EVERY view
    th = torch.zeros(V, 6, device=dev)
    th[:, :3] = torch.tensor([5.0, -3.0, 2.0], device=dev)
    th[:, 5] = 5.0 * np.pi / 180
    P_mov = params_to_Pmot(th, gen.P_nom)[None]

    with torch.no_grad():
        y_static = gen.project(gt, gen.P_nom[None])
        static = gen.fdk(y_static, gen.P_nom[None])[0]

        y_mov = gen.project(gt, P_mov)                 # scan of a rigidly displaced patient
        rigid = gen.fdk(y_mov, P_mov)[0]               # reconstructed with the TRUE matrices
        broken = gen.fdk(y_mov, gen.P_nom[None])[0]    # ... and with the NOMINAL ones (the control)

    p_s, p_r, p_b = (psnr(static, g3, meas), psnr(rigid, g3, meas), psnr(broken, g3, meas))
    d_sr = (rigid - static)
    d_sb = (broken - static)
    rng = float(g3[meas].max() - g3[meas].min())
    print(f"static FDK                          {p_s:6.2f} dB")
    print(f"global-rigid, TRUE matrices         {p_r:6.2f} dB   ({p_r - p_s:+.2f})")
    print(f"global-rigid, NOMINAL matrices      {p_b:6.2f} dB   ({p_b - p_s:+.2f})   <- the control")
    print(f"\nrigid-vs-static  : RMS {float(d_sr[meas].pow(2).mean().sqrt()) / rng:.2e} of range, "
          f"max {float(d_sr[meas].abs().max()) / rng:.2e}")
    print(f"broken-vs-static : RMS {float(d_sb[meas].pow(2).mean().sqrt()) / rng:.2e} of range, "
          f"max {float(d_sb[meas].abs().max()) / rng:.2e}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D, H, W = gen.shape
    lo, hi = 0.0, 1.4 * 0.02
    dlim = (hi - lo) / args.gain
    cols = [("ground truth", g3, None), (f"static FDK\n{p_s:.2f} dB", static, None),
            (f"global rigid, TRUE P\n{p_r:.2f} dB", rigid, None),
            (f"(rigid - static)  x{args.gain:.0f}", d_sr, dlim),
            (f"CONTROL: rigid, NOMINAL P\n{p_b:.2f} dB", broken, None),
            (f"(control - static)  x{args.gain:.0f}", d_sb, dlim)]
    fig, ax = plt.subplots(2, len(cols), figsize=(2.9 * len(cols), 6.4))
    for c, (name, v, dl) in enumerate(cols):
        v = v.cpu()
        for r, sl in enumerate([v[D // 2], v[:, H // 2]]):
            if dl is None:
                ax[r, c].imshow(sl, cmap="gray", vmin=lo, vmax=hi,
                                aspect="auto" if r else "equal", origin="lower")
            else:
                ax[r, c].imshow(sl, cmap="bwr", vmin=-dl, vmax=dl,
                                aspect="auto" if r else "equal", origin="lower")
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        ax[0, c].set_title(name, fontsize=9)
    ax[0, 0].set_ylabel("axial", fontsize=9)
    ax[1, 0].set_ylabel("coronal", fontsize=9)
    fig.suptitle("A GLOBAL rigid motion keeps the orbit a circle: reconstructing with the true P "
                 "must equal the static scan.\nThe control is the same scan reconstructed on the "
                 "NOMINAL geometry -- that is what a real error looks like.", fontsize=10)
    fig.tight_layout()
    p = os.path.join(args.out, "static_vs_rigid.png")
    fig.savefig(p, dpi=110)
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
