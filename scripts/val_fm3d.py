"""Prior-only validation of a checkpoint: cold start -> [prior ODE] -> x_1 estimate vs the static FDK.

Answers the one question the training loss cannot: does the prior ALONE, integrated from the
uncorrected FDK with no data term and no TV, walk toward a clean head?

    x0     = FDK(y_motion, P_nom)          the inference cold start
    x1_hat = prior_ode(model, x0, 50)      50 Euler steps, patch-blended, context-conditioned
    ref    = the motion-free static FDK    the bridge's t=1 image

Scores are computed after rigid alignment (SE(3) gauge). `run_validation` is also what
train_fm3d.py calls inline, so the standalone number reproduces the training curve.

    CUDA_VISIBLE_DEVICES=1 python scripts/val_fm3d.py --ckpt logs/fm3d_databridge/ckpt_last.pth
    CUDA_VISIBLE_DEVICES=1 python scripts/val_fm3d.py --ckpt logs/fm3d_databridge --watch
"""

from __future__ import annotations

import argparse
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
    lo, hi = 0.0, 0.045                    # head window in mu [1/mm]
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


# Per-patient constants (cold start, reference, cold-start metrics) are identical at every
# checkpoint for the fixed seeds, so they are computed once per process.
_VAL_CACHE: dict[tuple, dict] = {}


def run_validation(model, gen, meas, out_dir, *, it=0, patients=3, patch=64, ode_steps=50,
                   trans_mm=10.0, rot_deg=10.0, blend="uniform", n_offsets=2,
                   tile_batch=64, writer=None, dev="cuda"):
    """Prior-only ODE from the cold start on `patients` fixed val cases (motion seed 1000 + i).
    Returns (it, mean aligned PSNR, per-patient rows) and logs to tensorboard if `writer` is given.

    The patch->volume scheme defaults to the deployed inference scheme (uniform, K=2)."""
    if blend == "uniform" and n_offsets < 2:
        raise ValueError("blend='uniform' needs n_offsets >= 2 (a single non-overlapping pass "
                         "leaves tile seams)")
    spacing = (gen.dz, gen.dy, gen.dx)
    rows, paths = [], []
    for i in range(patients):
        pid = gen.records[i]["patient"]
        key = (gen.split, pid, 1000 + i, float(trans_mm), float(rot_deg))
        ent = _VAL_CACHE.get(key)
        if ent is None:
            with torch.no_grad():
                theta = make_motion("akima", gen.cfg.n_views, device=dev, seed=1000 + i,
                                    trans_mm=(trans_mm,) * 3, rot_deg=(rot_deg,) * 3)
                y = gen.simulate(i, params_to_Pmot(theta, gen.P_nom)[None])
                x0_mu = gen.fdk(y, gen.P_nom[None])[0]                  # cold start, MU
                ref_mu = gen.fdk(gen.simulate(i, gen.P_nom[None]), gen.P_nom[None])[0]
            m0 = aligned_metrics(x0_mu, ref_mu, spacing, mask=meas, iters=200)
            ent = {"x0": x0_mu.cpu(), "ref": ref_mu.cpu(), "m0": m0}
            _VAL_CACHE[key] = ent
        x0_mu = ent["x0"].to(dev)
        ref_mu = ent["ref"].to(dev)
        m0 = ent["m0"]
        with torch.no_grad():
            gtor = torch.Generator(device=dev).manual_seed(1000 + i) if n_offsets > 1 else None
            x1_mu = gen.from_net(prior_ode(model, gen.to_net(x0_mu)[None, None], n_steps=ode_steps,
                                           patch=patch, stride=patch // 2, batch=tile_batch,
                                           context="auto", blend=blend, n_offsets=n_offsets,
                                           generator=gtor, amp=True)[0, 0])
        m1 = aligned_metrics(x1_mu, ref_mu, spacing, mask=meas, iters=200)
        rows.append({"patient": pid, **{f"cold_{k}": v for k, v in m0.items()},
                     **{f"ode_{k}": v for k, v in m1.items()}})
        print(f"    val it {it:6d} p{pid:3d}: cold {m0['psnr_aligned']:5.2f} -> ODE "
              f"{m1['psnr_aligned']:5.2f} dB / SSIM {m0['ssim_aligned']:.3f} -> "
              f"{m1['ssim_aligned']:.3f}  (raw {m0['psnr_raw']:.2f} -> {m1['psnr_raw']:.2f})",
              flush=True)
        p = os.path.join(out_dir, f"val_it{it:06d}_p{i}.png")
        montage(p, [(x0_mu, f"x_t=0 cold FDK\n{m0['psnr_aligned']:.2f} dB"),
                    (x1_mu, f"prior ODE {ode_steps} -> t=1\n{m1['psnr_aligned']:.2f} dB"),
                    (ref_mu, "target (static FDK)")],
                f"fm3d it {it} | CQ500 patient {pid} | prior-only ODE (aligned, ref=static FDK, "
                f"blend={blend} K={n_offsets}) | "
                f"cold {m0['psnr_aligned']:.2f} -> {m1['psnr_aligned']:.2f} dB")
        paths.append(p)
        if writer is not None:
            writer.add_scalar(f"val_psnr_aligned/p{pid}", m1["psnr_aligned"], it)
            writer.add_scalar(f"val_ssim_aligned/p{pid}", m1["ssim_aligned"], it)
    mean = float(np.mean([r["ode_psnr_aligned"] for r in rows]))
    print(f"    val it {it:6d}  MEAN aligned ODE PSNR {mean:.2f} dB  -> {out_dir}/", flush=True)
    if writer is not None:
        writer.add_scalar("val_psnr_aligned/mean", mean, it)
        import matplotlib.image as mpimg
        for i, p in enumerate(paths):
            img = mpimg.imread(p)
            writer.add_image(f"val/p{i}", torch.from_numpy(img[..., :3]).permute(2, 0, 1), it)
    return it, mean, rows


def resolve_ckpt(p):
    """A .pth path, or a run dir (-> its ckpt_last.pth)."""
    return os.path.join(p, "ckpt_last.pth") if os.path.isdir(p) else p


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
                          patch=ca["patch"], ode_steps=args.ode_steps,
                          trans_mm=args.trans_mm, rot_deg=args.rot_deg,
                          blend=args.blend, n_offsets=args.n_offsets, dev=dev)[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="a .pth, or a run dir (uses ckpt_last.pth)")
    ap.add_argument("--root", default="data/CQ500")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="data/val_fm3d")
    ap.add_argument("--shape", type=int, nargs=3, default=(256, 256, 256))
    ap.add_argument("--patients", type=int, default=2)
    ap.add_argument("--ode_steps", type=int, default=50)
    ap.add_argument("--trans_mm", type=float, default=10.0, help="peak-to-peak [mm]")
    ap.add_argument("--rot_deg", type=float, default=10.0, help="peak-to-peak [deg]")
    ap.add_argument("--blend", default="uniform", choices=["hann", "uniform"])
    ap.add_argument("--n_offsets", type=int, default=2,
                    help="tilings blended per ODE step (K); must be >= 2 with --blend uniform")
    ap.add_argument("--watch", action="store_true",
                    help="re-evaluate ckpt_last.pth whenever its iter advances")
    ap.add_argument("--poll", type=int, default=600)
    args = ap.parse_args()

    dev = "cuda"
    ck_path = resolve_ckpt(args.ckpt)
    if args.watch:
        while not os.path.exists(ck_path):
            print(f"[watch] waiting for {ck_path} (poll every {args.poll}s)", flush=True)
            time.sleep(args.poll)
            ck_path = resolve_ckpt(args.ckpt)
    # geometry and simulation grid come off the checkpoint, as in run_posterior3d.build_world
    _ca = torch.load(ck_path, map_location="cpu", weights_only=False).get("args", {})
    cfg = ConeBeam3DConfig.thies(n_views=_ca.get("views", 360))
    gen = CQ500Generator(args.root, cfg=cfg, device=dev, split=args.split,
                         shape=tuple(args.shape), voxel_mm=1.0, verbose=True,
                         sim_native=(_ca.get("sim_grid", "native") == "native"))
    meas = measured_region_mask(gen.shape, (1.0, 1.0, 1.0), cfg, device=dev)

    if not args.watch:
        evaluate(ck_path, gen, meas, args, dev)
        return

    seen = -1
    while True:
        ck = resolve_ckpt(args.ckpt)
        if os.path.exists(ck):
            it = torch.load(ck, map_location="cpu", weights_only=False).get("iter", 0)
            if it > seen:
                seen = it
                evaluate(ck, gen, meas, args, dev)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
