"""Prior-ONLY validation of an fm3d checkpoint: render x_t(0) -> [prior ODE] -> x1_hat vs the GT.

The same quantity the sibling 4DCT project renders every `val_every` steps, run here as a STANDALONE
job so it never touches (or restarts) the training on the other GPU. It answers the one question
the loss number cannot: does the prior ALONE, integrated from the cold start with no data term and
no TV, walk the uncorrected FDK toward a clean head?

    x0 = FDK(y_motion, P_nom)            the inference cold start, in NET space
    x1_hat = prior_ode(model, x0, 50)    50-step Euler, patch-blended, global-context aware
    reference = the motion-free target the bridge was anchored to (static FDK, or the GT)

Run it on the FREE gpu against the latest checkpoint, in a loop if you like:

    CUDA_VISIBLE_DEVICES=1 python scripts/val_fm3d.py --ckpt logs/fm3d_cq500_leap/ckpt_last.pth
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


# Per-patient constants, shared across validation calls in one process. For the fixed seeds the
# cold start x0, the reference and the cold-start metrics are IDENTICAL at every checkpoint --
# recomputing them cost two forward projections, three FDKs and a 200-iter rigid alignment per
# patient per call. Keyed on everything they depend on; tensors parked on CPU (~128 MB/patient).
_VAL_CACHE: dict[tuple, dict] = {}


def run_validation(model, gen, meas, out_dir, *, it=0, patients=3, patch=64, ode_steps=50,
                   anchor="static", trans_mm=10.0, rot_deg=10.0, blend="uniform", n_offsets=2,
                   tile_batch=64, writer=None, dev="cuda"):
    """Prior-ONLY ODE from the cold start, on `patients` fixed val cases. Returns the per-patient
    metrics AND the montage paths, and (if given) logs scalars + images to a tensorboard writer.

    Shared by the standalone `evaluate` and by train_fm3d's inline validation, so both walk the
    same code -- the number the training loop prints is exactly the number this script reproduces.
    The fixed seed (1000 + i) means the SAME motion is scored at every checkpoint, so the curve
    tracks the prior improving and not the luck of the draw.

    THE PATCH->VOLUME SCHEME IS THE DEPLOYED ONE (uniform K=2, changed from hann K=1 on
    2026-07-31, user's call): the archive paper's non-overlapping random tilings
    (arXiv:2512.18161), which is exactly what `run_posterior3d.fm_predict` runs at inference.
    Validating in a scheme the deployed loop does not use measured a quantity nothing consumes.
    It is also 2.5x cheaper, which is what makes the inline val affordable at a shorter
    --val_every. The A/B that justifies BOTH halves (500k prior, 3 val patients, data/blend_cmp/):

        uniform K=2   25.35 dB / SSIM 0.765   11.2 min      <- deployed, and now validated
        hann    K=1   25.42 dB / SSIM 0.765   27.5 min      <- what this used to run
        uniform K=1   24.96 dB / SSIM 0.749                 <- seams intact; never use

    So the historical val curve stays comparable to the new one to within 0.07 dB (well under
    the run-to-run noise), and no checkpoint needs re-scoring. K=1 with `uniform` is REFUSED
    below rather than silently run, exactly as run_posterior3d refuses it.

    The REFERENCE is the static FDK for anchor in ("static", "none") -- for "none" too, because
    the scanner-achievable still image is the only honest yardstick a bare-bridge prior has (it
    was previously scored against the GT, silently) -- and the GT volume only for anchor="gt".
    `tile_batch` is the patch batch of the blended prior evaluation; 64 (the training batch)
    rather than prior_patch's tiny default 8, which left the GPU mostly idle."""
    if blend == "uniform" and n_offsets < 2:
        raise ValueError("blend='uniform' needs n_offsets >= 2 (a single non-overlapping pass "
                         "leaves tile seams, -0.4 dB; see data/blend_cmp/)")
    spacing = (gen.dz, gen.dy, gen.dx)
    rows, paths = [], []
    for i in range(patients):
        pid = gen.records[i]["patient"]
        ref_kind = "gt" if anchor == "gt" else "static"
        key = (gen.split, pid, 1000 + i, float(trans_mm), float(rot_deg), ref_kind)
        ent = _VAL_CACHE.get(key)
        if ent is None:
            with torch.no_grad():
                gt = gen.volume(i)
                theta = make_motion("akima", gen.cfg.n_views, device=dev, seed=1000 + i,
                                    trans_mm=(trans_mm,) * 3, rot_deg=(rot_deg,) * 3)
                # both y's come from the NATIVE simulation grid (dataset_cq500.simulate); `gt` is
                # the coarse inversion-grid volume and stays the metric's reference
                y = gen.simulate(i, params_to_Pmot(theta, gen.P_nom)[None])
                x0_mu = gen.fdk(y, gen.P_nom[None])[0]                  # cold start, MU
                static_mu = gen.fdk(gen.simulate(i, gen.P_nom[None]), gen.P_nom[None])[0]
                ref_mu = static_mu if ref_kind == "static" else gt[0, 0]
            # the metric's rigid_align needs grad (it optimizes the alignment by backprop),
            # so it sits OUTSIDE the no_grad block
            m0 = aligned_metrics(x0_mu, ref_mu, spacing, mask=meas, iters=200)
            ent = {"x0": x0_mu.cpu(), "ref": ref_mu.cpu(), "m0": m0}
            _VAL_CACHE[key] = ent
        x0_mu = ent["x0"].to(dev)
        ref_mu = ent["ref"].to(dev)
        m0 = ent["m0"]
        with torch.no_grad():
            gtor = torch.Generator(device=dev).manual_seed(1000 + i) if n_offsets > 1 else None
            # amp=True is EXPLICIT here because it changes the number this function reports.
            # The deployed loop's prior forward is fp16 (`run_posterior3d --prior_amp`, on by
            # default since 2026-07-25) and so is the regime the 500k weights were trained in;
            # this path silently stayed fp32 until 2026-08-04, costing 1.6x for a precision
            # nothing ships. The val curve therefore shifts slightly at that date -- by well
            # under the 0.07 dB the blend A/B already treats as noise, but it is a shift, so
            # pass amp=False to reproduce a pre-2026-08-04 val number exactly.
            x1_mu = gen.from_net(prior_ode(model, gen.to_net(x0_mu)[None, None], n_steps=ode_steps,
                                           patch=patch, stride=patch // 2, batch=tile_batch,
                                           context="auto", blend=blend, n_offsets=n_offsets,
                                           generator=gtor, amp=True)[0, 0])
        # GAUGE-AWARE: rigidly align to the target before scoring. Raw PSNR penalises the
        # unobservable global pose the prior is free to shift; the aligned number is the honest
        # one (see fm3d/reg_metric.py, and the SE(3) gauge in memory).
        m1 = aligned_metrics(x1_mu, ref_mu, spacing, mask=meas, iters=200)
        rows.append({"patient": pid, **{f"cold_{k}": v for k, v in m0.items()},
                     **{f"ode_{k}": v for k, v in m1.items()}})
        print(f"    val it {it:6d} p{pid:3d}: cold {m0['psnr_aligned']:5.2f} -> ODE "
              f"{m1['psnr_aligned']:5.2f} dB / SSIM {m0['ssim_aligned']:.3f} -> "
              f"{m1['ssim_aligned']:.3f}  (raw {m0['psnr_raw']:.2f} -> {m1['psnr_raw']:.2f}) "
              f"[ref={ref_kind}]", flush=True)
        p = os.path.join(out_dir, f"val_it{it:06d}_p{i}.png")
        montage(p, [(x0_mu, f"x_t=0 cold FDK\n{m0['psnr_aligned']:.2f} dB"),
                    (x1_mu, f"prior ODE {ode_steps} -> t=1\n{m1['psnr_aligned']:.2f} dB"),
                    (ref_mu, f"target (ref={ref_kind})")],
                f"fm3d it {it} | CQ500 patient {pid} | prior-only ODE (aligned, ref={ref_kind}, "
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
        # the montages as an image grid, so the ODE progress is watchable in tensorboard itself
        import matplotlib.image as mpimg
        for i, p in enumerate(paths):
            img = mpimg.imread(p)                                   # (H,W,4) float
            writer.add_image(f"val/p{i}", torch.from_numpy(img[..., :3]).permute(2, 0, 1), it)
    return it, mean, rows


def resolve_ckpt(p):
    """A .pth path, or a run dir (-> its ckpt_last.pth)."""
    return os.path.join(p, "ckpt_last.pth") if os.path.isdir(p) else p


def evaluate(ckpt, gen, meas, args, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    ca = ck["args"]
    it = ck.get("iter", 0)
    # NET normalization is operator-determined now (self-normalized FDK), so there is no
    # per-checkpoint scale to restore -- see projector_3d._fdk_physical_norm.
    in_ch = int(ck["ema"]["in_conv.weight"].shape[1])
    model = UNet3D(in_ch=in_ch, base=ca["base"]).to(dev).eval()
    model.load_state_dict(ck["ema"])
    for q in model.parameters():
        q.requires_grad_(False)
    return run_validation(model, gen, meas, args.out, it=it, patients=args.patients,
                          patch=ca["patch"], ode_steps=args.ode_steps, anchor=args.anchor,
                          trans_mm=args.trans_mm, rot_deg=args.rot_deg,
                          blend=args.blend, n_offsets=args.n_offsets, dev=dev)[:2]


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
    ap.add_argument("--trans_mm", type=float, default=10.0)   # PEAK-TO-PEAK
    ap.add_argument("--rot_deg", type=float, default=10.0)    # PEAK-TO-PEAK
    ap.add_argument("--blend", default="uniform", choices=["hann", "uniform"],
                    help="uniform (DEFAULT) = the DEPLOYED scheme: non-overlapping random "
                         "tilings averaged (family B, arXiv:2512.18161), i.e. what "
                         "run_posterior3d actually runs, and 2.5x cheaper. hann = overlapping "
                         "Hann-window blend (family A, ours) -- what the val curve used before "
                         "2026-07-31; the two agree to 0.07 dB (see run_validation).")
    ap.add_argument("--n_offsets", type=int, default=2,
                    help="tile grids blended per ODE step (family B's K; K=2 is optimal and is "
                         "the deployed value). MUST be >=2 with --blend uniform.")
    ap.add_argument("--watch", action="store_true",
                    help="re-evaluate ckpt_last.pth whenever its iter advances")
    ap.add_argument("--poll", type=int, default=600)
    args = ap.parse_args()

    dev = "cuda"
    ck_path = resolve_ckpt(args.ckpt)
    if args.watch:
        # --watch is meant to be launchable BEFORE the trainer writes its first checkpoint, so
        # the args read below must WAIT for the file rather than crash on it. resolve_ckpt is
        # re-run each poll because the run DIRECTORY itself may not exist yet either (a path
        # that is not a dir resolves to itself, then becomes dir/ckpt_last.pth once it appears).
        while not os.path.exists(ck_path):
            print(f"[watch] waiting for {ck_path} (poll every {args.poll}s)", flush=True)
            time.sleep(args.poll)
            ck_path = resolve_ckpt(args.ckpt)
    # GEOMETRY AND SIMULATION GRID COME OFF THE CHECKPOINT, not off this script's defaults --
    # the same contract build_world enforces in run_posterior3d. Reading them here matters
    # because `evaluate` below rebuilds only the MODEL from the ckpt: a ckpt trained at a
    # different --views, or with --sim_grid coarse, would otherwise be scored against a
    # silently different scan and the curve would not be comparable to the trainer's inline val.
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
