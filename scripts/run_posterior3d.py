"""Blind rigid motion correction: the PnP-TV predictor-corrector posterior loop, in 3D.

This is the configuration the 2D project converged on after a long ablation, transplanted whole.
Per Euler step k of N (t = k/N), starting cold from x = FDK(y, P_nom) and theta = 0:

    1. PREDICT   x_prior = x + dt * v_FM(x, t)            the learned prior moves first
    2. ESTIMATE  est.refine_global(x_prior, y, iters=PER) motion is fitted on the IMPROVED image
    3. CORRECT   PnP-TV forward-backward on z:
                    z <- z - alpha * ||z|| * unit(grad_x 0.5||A_{P(theta)}(z) - y||^2)
                    z <- z + kappa * (TV(z) - z)
    4. x = z

WHY IT IS IN THIS ORDER (predictor-corrector, not a simultaneous update): estimating the motion on
`x_prior` rather than on `x` is a Gauss-Seidel step, and it beat the simultaneous ("combined")
update on every test index in 2D. The motion estimator is only as good as the image you hand it,
so hand it the better one.

WHY TV AND NOT THE FM DENOISER inside the PnP loop: using the learned x1_hat as the PnP denoiser
diverges. The data-prox pushes `z` off the manifold the network was trained on, the network then
returns garbage for it, and the two reinforce each other. TV is OOD-safe -- it is a weaker prior,
but it cannot blow up -- so the carried state stays stable and this is a FAITHFUL PnP. (The 2D
project also has a `decouple` variant that carries the on-manifold FM image and uses the
data-prox'd z only as a reference for the estimator; it scores higher but is not a PnP
reconstruction. TV is the honest one and is the default here.)

The objective the 2D work settled on is STREAK-FREE BY EYE, not the metric -- PSNR ranks these
wrong (see fm3d/reg_metric.py on the SE(3) gauge). Montages are written every step. Look at them.

SCALE BOOKKEEPING. The FM ODE runs in NET space ([-1,1]); the estimator, the data-prox and the TV
run in MU space (1/mm). Convert at every crossing. Omitting it is a ~50x mismatch between A(x)
and y and it does not announce itself.

    python scripts/run_posterior3d.py --ckpt logs/fm3d_a/ckpt_last.pth
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_slab import AAPMSlabGenerator
from fm3d.geometry_3d import ConeBeam3DConfig, measured_region_mask
from fm3d.motion_estimation import make_estimator
from fm3d.prior_patch import predict_x1_patched
from fm3d.reg_metric import aligned_metrics
from fm3d.rigid_motion import make_motion, motion_error, params_to_Pmot
from fm3d.tv import sidky_dtv_denoise_3d
from fm3d.unet_3d import UNet3D

DATA = "/home/mirlab/Desktop/Flow_matching_motion/data/AAPM_head_data"


@torch.no_grad()
def fm_predict(model, gen, x_mu, t, dt, patch, context="auto", n_offsets=1, generator=None):
    """One Euler step of the FM ODE. Takes and returns a MU-space volume.

    The prior is evaluated PATCH-WISE and Hann-blended (`predict_x1_patched`), never on the whole
    slab at once, and that is not only about memory. UNet3D normalizes with GroupNorm, whose
    statistics are taken over the spatial extent -- so the same weights fed a 64x256x256 volume
    normalize differently than they did on the 64^3 patches they were trained on. Running the net
    at a spatial size it never saw is a silent train/test mismatch. The blending is identity-exact,
    so tiling costs nothing.

    With a context-conditioned prior (in_ch=5, arXiv:2512.18161) the tiles additionally carry the
    downsampled CURRENT x_t and their absolute position, so the global-context channel is rebuilt
    from the evolving volume at EVERY ODE step -- exactly as the bridge built it in training.

    `predict_x1_patched` blends in "x1 space" and we divide back out to a velocity. Read that as
    bookkeeping, NOT as a round trip through a clean image: this project trains v on the TRUE
    TANGENT of the (curved) geometry bridge, so x_t + (1-t)v is a first-order extrapolation, not
    an endpoint -- the name is inherited from Flowmatching-4DCT, which does regress the endpoint.
    It is nevertheless EXACT: v -> x1 is affine with a constant coefficient and the blend weights
    normalize to 1, so the (1-t) cancels and what is blended is v itself, to 6e-6 relative at the
    worst t. See fm3d/prior_patch.py. The one thing never to do is treat x1_hat as a clean image.
    """
    x_net = gen.to_net(x_mu)[None, None]
    x1 = predict_x1_patched(model, x_net, t, patch=patch, stride=patch // 2,
                            context=context, n_offsets=n_offsets, generator=generator)
    v = (x1 - x_net) / max(1.0 - t, 1e-3)
    return gen.from_net(x_net + dt * v)[0, 0]


def data_grad(x_mu, theta, y, gen):
    """grad_x 0.5 ||A_{P(theta)}(x) - y||^2, in mu space."""
    x = x_mu.detach().requires_grad_(True)
    P = params_to_Pmot(theta, gen.P_nom)[None]
    r = gen.project(x[None, None], P, n_samples=256) - y
    (0.5 * (r ** 2).sum()).backward()
    return x.grad


def montage(path, gt, x, xt, step, t, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D, H, _ = gt.shape
    zc, yc = D // 2, H // 2
    lo, hi = 0.0, 1.4 * 0.02
    fig, ax = plt.subplots(2, 3, figsize=(10, 6.6))
    for c, (name, v) in enumerate([("ground truth", gt), ("FDK(theta_hat)", x), ("x_t (carried)", xt)]):
        ax[0, c].imshow(v[zc].cpu(), cmap="gray", vmin=lo, vmax=hi)
        ax[0, c].set_title(name, fontsize=9)
        ax[1, c].imshow(v[:, yc].cpu(), cmap="gray", vmin=lo, vmax=hi, aspect="auto")
        for r in range(2):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    fig.suptitle(f"step {step}  t={t:.2f}   {title}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default="data/posterior3d")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--z0", type=int, default=0)
    ap.add_argument("--motion_kind", default="mixed")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n_steps", type=int, default=30)       # 2D deploy recipe: ~30
    ap.add_argument("--per", type=int, default=10)           # motion iters per ODE step: ~10
    ap.add_argument("--estimator", default="net")            # net = hashbl, the 2D default
    ap.add_argument("--loss", default="l2si")                # l2si | lncc | ncc | ramp
    ap.add_argument("--lncc_win", type=int, default=9)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--views_per_iter", type=int, default=24)
    ap.add_argument("--alpha", type=float, default=0.1)      # data-prox step
    ap.add_argument("--alpha_p", type=float, default=0.0)    # alpha * (1-t)^p decay; 0 = off
    ap.add_argument("--kappa", type=float, default=0.3)      # TV pull. 2D sweet spots: 0.3/0.5
    ap.add_argument("--tv_iters", type=int, default=15)
    ap.add_argument("--tv_step", type=float, default=0.30)
    ap.add_argument("--pnp_k", type=int, default=1)          # data-prox <-> denoise alternations
    ap.add_argument("--est_n_samples", type=int, default=384)
    ap.add_argument("--context", default="auto", choices=["auto", "global", "none"],
                    help="auto reads in_ch off the checkpoint's in_conv weight")
    ap.add_argument("--patch_offsets", type=int, default=1,
                    help="tile grids blended per ODE step; >1 adds randomly SHIFTED grids "
                         "(the FM analogue of the paper's recurrent noising, K=2 optimal there)")
    args = ap.parse_args()

    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    ca = ck["args"]
    cfg = ConeBeam3DConfig(det_bin=2, n_views=ca["views"])
    gen = AAPMSlabGenerator(args.data, cfg, device=dev, slab=ca["slab"], in_plane=ca["in_plane"])

    # in_ch comes off the WEIGHTS, not off ca["context"] -- the checkpoint's args are what the
    # run was launched with, the weights are what it actually trained. A mismatch here is a
    # silently wrong prior (the net would read the coord channels as image content), so refuse.
    in_ch_ck = int(ck["ema"]["in_conv.weight"].shape[1])
    if args.context == "auto":
        args.context = "global" if in_ch_ck >= 5 else "none"
    in_ch = 5 if args.context == "global" else 1
    if in_ch != in_ch_ck:
        raise SystemExit(f"--context {args.context} wants in_ch={in_ch} but the ckpt was "
                         f"trained with in_ch={in_ch_ck}")
    print(f"prior: UNet3D in_ch={in_ch} (context={args.context}), "
          f"{args.patch_offsets} tile grid(s)/step")

    model = UNet3D(in_ch=in_ch, base=ca["base"]).to(dev)
    model.load_state_dict(ck["ema"])
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)

    spacing = (gen.dz, gen.dy, gen.dx)
    meas = measured_region_mask(gen.shape, spacing, cfg, device=dev)

    # ---- simulate a motion-corrupted scan
    gt = gen.volume(args.run, args.z0)                                   # (1,1,D,H,W) mu
    theta_true = make_motion(args.motion_kind, cfg.n_views, device=dev, seed=args.seed)
    with torch.no_grad():
        y = gen.project(gt, params_to_Pmot(theta_true, gen.P_nom)[None])
    gt3 = gt[0, 0]

    est = make_estimator(
        args.estimator, cfg, gen.P_nom, gen.u_coords, gen.v_coords, dev,
        dx=gen.dx, dy=gen.dy, dz=gen.dz, loss=args.loss, lncc_win=args.lncc_win,
        n_samples=args.est_n_samples, views_per_iter=args.views_per_iter, lr=args.lr)

    with torch.no_grad():
        x = gen.fdk(y, gen.P_nom[None])[0]                               # cold start: uncorrected
    patch = ca["patch"]

    print(f"cold start  " + json.dumps({k: round(v, 4) for k, v in
          aligned_metrics(x, gt3, spacing, mask=meas, iters=200).items()}))

    hist = []
    gtile = torch.Generator(device=dev).manual_seed(args.seed)   # reproducible tile jitter
    N = args.n_steps
    for k in range(N):
        t = k / N
        dt = 1.0 / N

        # 1. PREDICT -- the prior moves first (patch-blended; see fm_predict)
        x_prior = fm_predict(model, gen, x, t, dt, patch, context=args.context,
                             n_offsets=args.patch_offsets, generator=gtile)

        # 2. ESTIMATE on the improved image (Gauss-Seidel, not simultaneous)
        loss = est.refine_global(x_prior, y[0], iters=args.per)
        theta = est.current_params()

        # 3. CORRECT -- PnP forward-backward, carrying z (a faithful PnP: the denoiser is TV,
        #    which cannot go out of distribution the way the learned denoiser does)
        a = args.alpha * ((1 - t) ** args.alpha_p if args.alpha_p > 0 else 1.0)
        z = x_prior
        for _ in range(args.pnp_k):
            g = data_grad(z, theta, y[0], gen)
            z = z - a * z.norm() * g / g.norm().clamp_min(1e-12)
            z = z + args.kappa * (sidky_dtv_denoise_3d(
                z[None, None], args.tv_iters, args.tv_step)[0, 0] - z)
        x = z.detach()

        with torch.no_grad():
            x_fdk = gen.fdk(y, params_to_Pmot(theta, gen.P_nom)[None])[0]
        m = aligned_metrics(x_fdk, gt3, spacing, mask=meas, iters=150)
        me = motion_error(theta, theta_true)
        hist.append({"step": k, "t": t, "loss": loss, **m, **me})
        print(f"step {k:3d} t={t:.2f} | fit {loss:.5f} | FBP aligned "
              f"{m['psnr_aligned']:5.2f} dB / SSIM {m['ssim_aligned']:.3f} "
              f"(raw {m['psnr_raw']:5.2f}) | theta rot {me['rot_rmse_deg']:.2f} deg, "
              f"trans(gauge-fit) {me['trans_rmse_mm']:.2f} mm", flush=True)
        montage(os.path.join(args.out, f"step{k:03d}.png"), gt3, x_fdk, x, k, t,
                f"aligned {m['psnr_aligned']:.2f} dB / SSIM {m['ssim_aligned']:.3f}")

    with torch.no_grad():
        theta = est.current_params()
        x_final = gen.fdk(y, params_to_Pmot(theta, gen.P_nom)[None])[0]     # output = FDK(theta)
    fm = aligned_metrics(x_final, gt3, spacing, mask=meas, iters=300)
    print("\nFINAL " + json.dumps({k: round(v, 4) for k, v in fm.items()}))
    torch.save({"theta": theta.cpu(), "theta_true": theta_true.cpu(), "hist": hist,
                "final": fm}, os.path.join(args.out, "result.pt"))
    montage(os.path.join(args.out, "final.png"), gt3, x_final, x, N, 1.0,
            f"FINAL aligned {fm['psnr_aligned']:.2f} dB / SSIM {fm['ssim_aligned']:.3f}")
    print(f"montages -> {args.out}/  (judge by eye: streak-free, not the number)")


if __name__ == "__main__":
    main()
