"""Prior-ONLY validation of an fm3d checkpoint: render x_t(0) -> [prior ODE] -> x1_hat vs the GT.

The same quantity the sibling 4DCT project renders every `val_every` steps, run here as a STANDALONE
job so it never touches (or restarts) the training on the other GPU. It answers the one question
the loss number cannot: does the prior ALONE, integrated from the cold start with no data term and
no TV, walk the uncorrected FDK toward a clean head?

    x0 = FDK(y_motion, P_nom)            the inference cold start, in NET space
    x1_hat = prior_ode(model, x0, 50)    50-step Euler, patch-blended, global-context aware
    reference = the motion-free target the bridge was anchored to (static FDK, or the GT)

Run it on the FREE gpu against the latest checkpoint, in a loop if you like:

    CUDA_VISIBLE_DEVICES=1 python scripts/val_fm3d.py --ckpt logs/fm3d_cq500/ckpt_last.pth
    CUDA_VISIBLE_DEVICES=1 python scripts/val_fm3d.py --ckpt logs/fm3d_cq500 --watch
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig, measured_region_mask
from fm3d.prior_patch import prior_ode
from fm3d.reg_metric import aligned_metrics
from fm3d.rigid_motion import make_motion, params_to_Pmot
from fm3d.unet_3d import UNet3D


def montage(path, panels, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    # a HEAD-CT window in mu [1/mm]: soft tissue ~0.02, bone runs to ~0.06. The old NET-space
    # (-1, -0.3) window clipped everything above soft tissue to white and hid all the anatomy.
    lo, hi = 0.0, 0.045
    fig, ax = plt.subplots(3, n, figsize=(3.2 * n, 9.4))
    for j, (vol, ttl) in enumerate(panels):
        D, H, W = vol.shape
        for i, sl in enumerate([vol[D // 2], vol[:, H // 2], vol[:, :, W // 2]]):
            a = ax[i, j]
            a.imshow(sl.detach().cpu().numpy(), cmap="gray", vmin=lo, vmax=hi,
                     aspect="auto" if i else "equal", origin="lower")
            a.set_title(ttl if i == 0 else "", fontsize=9)
            a.set_xticks([]); a.set_yticks([])
    for i, nm in enumerate(["axial", "coronal", "sagittal"]):
        ax[i, 0].set_ylabel(nm, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=105)
    plt.close(fig)


def run_validation(model, gen, meas, out_dir, *, it=0, patients=3, patch=64, ode_steps=50,
                   anchor="static", trans_mm=5.0, rot_deg=5.0, writer=None, dev="cuda"):
    """Prior-ONLY ODE from the cold start, on `patients` fixed val cases. Returns the per-patient
    metrics AND the montage paths, and (if given) logs scalars + images to a tensorboard writer.

    Shared by the standalone `evaluate` and by train_fm3d's inline validation, so both walk the
    same code -- the number the training loop prints is exactly the number this script reproduces.
    The fixed seed (1000 + i) means the SAME motion is scored at every checkpoint, so the curve
    tracks the prior improving and not the luck of the draw."""
    spacing = (gen.dz, gen.dy, gen.dx)
    rows, paths = [], []
    for i in range(patients):
        # the forward operator and the prior ODE need no grad; the metric's rigid_align DOES
        # (it optimizes the alignment by backprop), so only THIS block is under no_grad.
        with torch.no_grad():
            gt = gen.volume(i)
            theta = make_motion("akima", gen.cfg.n_views, device=dev, seed=1000 + i,
                                trans_mm=(trans_mm,) * 3, rot_deg=(rot_deg,) * 3)
            y = gen.project(gt, params_to_Pmot(theta, gen.P_nom)[None])
            x0_mu = gen.fdk(y, gen.P_nom[None])[0]                      # cold start, MU
            static_mu = gen.fdk(gen.project(gt, gen.P_nom[None]), gen.P_nom[None])[0]
            x1_mu = gen.from_net(prior_ode(model, gen.to_net(x0_mu)[None, None], n_steps=ode_steps,
                                           patch=patch, stride=patch // 2, context="auto")[0, 0])
        ref_mu = static_mu if anchor == "static" else gt[0, 0]
        # GAUGE-AWARE: rigidly align to the target before scoring. Raw PSNR penalises the
        # unobservable global pose the prior is free to shift; the aligned number is the honest
        # one (see fm3d/reg_metric.py, and the SE(3) gauge in memory).
        m0 = aligned_metrics(x0_mu, ref_mu, spacing, mask=meas, iters=200)
        m1 = aligned_metrics(x1_mu, ref_mu, spacing, mask=meas, iters=200)
        pid = gen.records[i]["patient"]
        rows.append({"patient": pid, **{f"cold_{k}": v for k, v in m0.items()},
                     **{f"ode_{k}": v for k, v in m1.items()}})
        print(f"    val it {it:6d} p{pid:3d}: cold {m0['psnr_aligned']:5.2f} -> ODE "
              f"{m1['psnr_aligned']:5.2f} dB / SSIM {m0['ssim_aligned']:.3f} -> "
              f"{m1['ssim_aligned']:.3f}  (raw {m0['psnr_raw']:.2f} -> {m1['psnr_raw']:.2f})",
              flush=True)
        p = os.path.join(out_dir, f"val_it{it:06d}_p{i}.png")
        montage(p, [(x0_mu, f"x_t=0 cold FDK\n{m0['psnr_aligned']:.2f} dB"),
                    (x1_mu, f"prior ODE {ode_steps} -> t=1\n{m1['psnr_aligned']:.2f} dB"),
                    (ref_mu, f"target ({anchor})")],
                f"fm3d it {it} | CQ500 patient {pid} | prior-only ODE (aligned) | "
                f"cold {m0['psnr_aligned']:.2f} -> {m1['psnr_aligned']:.2f} dB")
        paths.append(p)
        if writer is not None:
            writer.add_scalar(f"val_psnr_aligned/p{pid}", m1["psnr_aligned"], it)
            writer.add_scalar(f"val_ssim_aligned/p{pid}", m1["ssim_aligned"], it)
    mean = float(np.mean([r["ode_psnr_aligned"] for r in rows]))
    print(f"    val it {it:6d}  MEAN aligned ODE PSNR {mean:.2f} dB  -> {out_dir}/", flush=True)
    if writer is not None:
        writer.add_scalar("val_psnr_aligned/mean", mean, it)
        # the montages as an image grid, so the ODE progress is watchable in tensorboard itself
        import matplotlib.image as mpimg
        for i, p in enumerate(paths):
            img = mpimg.imread(p)                                   # (H,W,4) float
            writer.add_image(f"val/p{i}", torch.from_numpy(img[..., :3]).permute(2, 0, 1), it)
    return it, mean, rows


def evaluate(ckpt, gen, meas, args, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    ca = ck["args"]
    it = ck.get("iter", 0)
    in_ch = int(ck["ema"]["in_conv.weight"].shape[1])
    model = UNet3D(in_ch=in_ch, base=ca["base"]).to(dev).eval()
    model.load_state_dict(ck["ema"])
    for q in model.parameters():
        q.requires_grad_(False)
    return run_validation(model, gen, meas, args.out, it=it, patients=args.patients,
                          patch=ca["patch"], ode_steps=args.ode_steps, anchor=args.anchor,
                          trans_mm=args.trans_mm, rot_deg=args.rot_deg, dev=dev)[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="a .pth, or a dir (uses ckpt_last.pth)")
    ap.add_argument("--root", default="data/CQ500")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="data/val_fm3d")
    ap.add_argument("--shape", type=int, nargs=3, default=(256, 256, 256))
    ap.add_argument("--patients", type=int, default=2)
    ap.add_argument("--ode_steps", type=int, default=50)       # the deploy loop's count
    ap.add_argument("--anchor", default="static", choices=["static", "gt"],
                    help="which target to SCORE against -- must match how the ckpt was trained")
    ap.add_argument("--trans_mm", type=float, default=5.0)
    ap.add_argument("--rot_deg", type=float, default=5.0)
    ap.add_argument("--watch", action="store_true",
                    help="re-evaluate ckpt_last.pth whenever its iter advances")
    ap.add_argument("--poll", type=int, default=600)
    args = ap.parse_args()

    dev = "cuda"
    cfg = ConeBeam3DConfig.thies()
    gen = CQ500Generator(args.root, cfg=cfg, device=dev, split=args.split,
                         shape=tuple(args.shape), voxel_mm=1.0, verbose=True)
    meas = measured_region_mask(gen.shape, (1.0, 1.0, 1.0), cfg, device=dev)

    def resolve(p):
        return os.path.join(p, "ckpt_last.pth") if os.path.isdir(p) else p

    if not args.watch:
        evaluate(resolve(args.ckpt), gen, meas, args, dev)
        return

    seen = -1
    while True:
        ck = resolve(args.ckpt)
        if os.path.exists(ck):
            it = torch.load(ck, map_location="cpu", weights_only=False).get("iter", 0)
            if it > seen:
                seen = it
                evaluate(ck, gen, meas, args, dev)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
