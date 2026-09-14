"""Blind rigid-motion correction: the predictor-corrector loop with a geometry-bridge prior
(Algorithm 1 of the paper).

Per Euler step k of N (t = k/N), starting cold from x = FDK(y, P_nom) and theta_hat = 0:

    1. PREDICT   x_pred = x + dt * v_phi(x, t)              frozen flow-matching prior, patch-blended
    2. ESTIMATE  theta_hat = argmin ||A_theta x_pred - y||    Akima spline coefficients,
                                                              warm-started GD, --per iterations
    3. CORRECT   z = CG(theta_hat, y; warm start x_pred)      --cg_iters conjugate-gradient iterations
                 z = z + kappa * (D_TV(z) - z)                relaxed TV update
    4. x = z

The motion is fitted on the PREDICTED image (a Gauss-Seidel sweep): the estimator is only as good
as the image it is handed, so it is handed the better one. The data step is a short CG solve rather
than a gradient step because the payoff of a better theta_hat lies in streaks and edges, which a
single adjoint step barely reaches.

Two volumes are saved in result.pt: the final iterate x_t (the primary output) and FDK(theta_hat)
(an analytic readout of the prior-assisted pose estimates). Every score is computed after rigid alignment
to the ground truth, because blind motion correction has an exact SE(3) gauge (fm3d/reg_metric.py).

    python scripts/run_posterior3d.py --ckpt logs/fm3d_databridge/ckpt_iter500000.pth \
        --split test --run 0 --seed 1000 --out data/test30/p00

Units: the prior runs in NET space ([-1, 1]); the estimator, CG and TV run in MU space (1/mm).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import (ConeBeam3DConfig, detector_coords_3d,
                              measured_region_mask)
from fm3d.motion_estimation import make_estimator
from fm3d.prior_patch import predict_x1_patched
from fm3d.reg_metric import aligned_metrics
from fm3d.rigid_motion import AMP_UNITS, make_motion, motion_error, params_to_Pmot
from fm3d.tv import grad_forward_3d, sidky_dtv_denoise_3d
from fm3d.unet_3d import UNet3D


@torch.no_grad()
def fm_predict(model, gen, x_mu, t, dt, patch, context="auto", n_offsets=1, generator=None,
               blend="uniform", batch=64, amp=False):
    """One Euler step of the FM ODE, x + dt * v_phi(x, t). Takes and returns a MU-space volume.

    The prior is evaluated PATCH-WISE and blended (`predict_x1_patched`), never on the whole
    volume: the U-Net was trained on `patch`^3 tiles and its GroupNorm statistics are taken over
    the spatial extent, so running it at another size is a silent train/test mismatch. With
    blend="uniform" the volume is covered by `n_offsets` randomly shifted non-overlapping tilings
    (arXiv:2512.18161); the global-context channels are rebuilt from the CURRENT x_t at every
    step, exactly as the bridge built them in training.

    `predict_x1_patched` blends in "x1 space" (x_t + (1-t) v) and we divide back to a velocity.
    That is bookkeeping, not a round trip through a clean image: v -> x1 is affine with a
    constant coefficient and the blend weights sum to one, so what is blended is v itself.
    """
    x_net = gen.to_net(x_mu)[None, None]
    x1 = predict_x1_patched(model, x_net, t, patch=patch, stride=patch // 2,
                            context=context, n_offsets=n_offsets, generator=generator,
                            blend=blend, batch=batch, amp=amp)
    v = (x1 - x_net) / max(1.0 - t, 1e-3)
    return gen.from_net(x_net + dt * v)[0, 0]


def _adjoint(s, P, gen):
    """A_P^T s -- LEAP's modular VD backprojector. Not the exact transpose of the Joseph forward
    (adjointness defect measured at 3.5e-4 on a real sinogram), so CG below runs on a slightly
    non-symmetric operator; at --cg_iters 5 this is measured to be harmless."""
    with torch.no_grad():
        return gen.adjoint(s[None], P[None])[0, 0]


def cg_dc_step(x_mu, theta, y, gen, iters=5):
    """A few conjugate-gradient iterations on the normal equations, warm-started at x:

        minimize_z 0.5 ||A_{P(theta)} z - y||^2      (z0 = x)

    i.e. CG on A^T A z = A^T y, the DDS data-consistency step (Chung et al., ICLR 2024). CG's k-th
    iterate applies a degree-k polynomial of A^T A that approximates its inverse over the
    residual's spectrum, so a handful of iterations recovers the high frequencies a gradient step
    starves. Early stopping is the regularizer; a still-wrong theta is stamped into z only as far
    as `iters` allows, which is why it stays small while theta is moving.

    Memory: each A(v) materializes a full sinogram (~480 MB at 360 x 500 x 700), so the
    intermediate is freed as soon as the adjoint has consumed it.
    """
    P = params_to_Pmot(theta, gen.P_nom)

    def A(v):
        with torch.no_grad():
            return gen.project(v[None, None], P[None])[0]

    def AT(s):
        return _adjoint(s, P, gen)

    def M(v):
        s = A(v)
        out = AT(s)
        del s
        return out

    # r0 = A^T y - A^T A z0 = A^T (y - A z0): one adjoint instead of two.
    z = x_mu.detach().clone()
    r = AT(y - A(z))
    p = r.clone()
    rs = float((r * r).sum())
    for _ in range(iters):
        Mp = M(p)
        alpha = rs / max(float((p * Mp).sum()), 1e-30)
        z = z + alpha * p
        r = r - alpha * Mp
        del Mp
        rs_new = float((r * r).sum())
        p = r + (rs_new / max(rs, 1e-30)) * p
        rs = rs_new
    return z


def _panel_label(name, m_gt=None, m_st=None):
    s = name
    if m_gt:
        s += f"\nvs GT   {m_gt['psnr_aligned']:.2f} dB / {m_gt['ssim_aligned']:.3f}"
    if m_st:
        s += f"\nvs sFDK {m_st['psnr_aligned']:.2f} dB / {m_st['ssim_aligned']:.3f}"
    return s


def montage(path, gt, x0, x, xt, ceil, step, t, title,
            m_in=None, m_out=None, m_xt=None,
            ms_in=None, ms_out=None, ms_xt=None, m_ceil=None):
    """Axial + coronal: cold FDK | FDK(theta_hat) | x_t | FDK(theta_true) | GT.

    All reconstructions must already be rigidly aligned to `gt` (each has its own gauge fit),
    otherwise the raw z = D//2 slices would show different anatomical planes side by side.
    `m_*` are scores vs the GT volume, `ms_*` vs the motion-free static FDK. The fourth panel is
    FDK at the TRUE motion: the ceiling a motion-corrected FDK of THIS data can reach (it still
    carries the cone-beam artifacts of the analytic operator), not the static FDK of a different,
    motion-free scan. Window: mu in [0, 0.05] 1/mm (soft tissue ~0.02, cortical bone to ~0.06)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D, H, _ = gt.shape
    zc, yc = D // 2, H // 2
    lo, hi = 0.0, 0.05
    panels = [(_panel_label("input (cold FDK, aligned)", m_in, ms_in), x0),
              (_panel_label("FDK(theta_hat)", m_out, ms_out), x),
              (_panel_label("x_t (final iterate)", m_xt, ms_xt), xt),
              (_panel_label("FDK(theta_true) = reachable FDK ceiling", m_ceil), ceil),
              ("ground truth", gt)]
    fig, ax = plt.subplots(2, 5, figsize=(16.5, 7.4))
    for c, (name, v) in enumerate(panels):
        ax[0, c].imshow(v[zc].cpu(), cmap="gray", vmin=lo, vmax=hi)
        ax[0, c].set_title(name, fontsize=8)
        ax[1, c].imshow(v[:, yc].cpu(), cmap="gray", vmin=lo, vmax=hi, aspect="auto")
        for r in range(2):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    ax[0, 0].set_ylabel("axial", fontsize=9)
    ax[1, 0].set_ylabel("coronal", fontsize=9)
    fig.suptitle(f"step {step}  t={t:.2f}   {title}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def build_world(*, ckpt, dev="cuda", root=None, split="val",
                run=0, motion_kind="akima", seed=3, trans_mm=10.0, rot_deg=10.0):
    """Rebuild the exact world the prior was trained in, plus the simulated corrupted scan.

    Geometry, grid and simulation grid all come off the checkpoint, not off script defaults.
    Everything here is deterministic given (ckpt, split, run, seed): the volume is a file read,
    `make_motion` runs on its own seeded generator, and the forward projector has no atomics --
    which is what lets `render_posterior3d.py` and the baseline drivers rebuild the same
    (patient, motion, sinogram) triple instead of storing a ~480 MB sinogram per run.
    """
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    ca = ck["args"]
    ds = ca.get("dataset")
    if ds != "cq500":
        raise SystemExit(f"checkpoint dataset {ds!r}: only 'cq500' is supported")
    cfg = ConeBeam3DConfig.thies(n_views=ca["views"])            # the trainer's geometry
    root = root or ca.get("root")
    if not root or not os.path.isdir(root):
        raise SystemExit(f"CQ500 root {root!r} not found -- pass --root")
    # y is simulated on the grid the prior was trained against (native 612^3 by default);
    # everything downstream inverts on the coarse 1 mm grid, exactly as in training.
    gen = CQ500Generator(root, cfg, device=dev, split=split,
                         shape=tuple(ca["shape"]), voxel_mm=1.0,
                         sim_native=(ca.get("sim_grid", "native") == "native"))
    print(f"dataset {ds}: grid {gen.shape} @ ({gen.dz:g},{gen.dy:g},{gen.dx:g}) mm | "
          f"{cfg.n_views} views")

    spacing = (gen.dz, gen.dy, gen.dx)
    meas = measured_region_mask(gen.shape, spacing, cfg, device=dev)

    gt = gen.volume(run)                                             # (1,1,D,H,W) mu
    # Amplitudes are full control-point sampling widths before zero-centering:
    # 10 means nodes in [-5, 5]; realized trajectory ranges can differ.
    amp = {}
    if trans_mm is not None:
        amp["trans_mm"] = trans_mm
    if rot_deg is not None:
        amp["rot_deg"] = rot_deg
    theta_true = make_motion(motion_kind, cfg.n_views, device=dev, seed=seed, **amp)
    with torch.no_grad():
        y = gen.simulate(run, params_to_Pmot(theta_true, gen.P_nom)[None])
        # the motion-free scan's FDK: the bridge's t=1 image and Thies' scoring reference
        y_static = gen.simulate(run, gen.P_nom[None])
        static_fdk = gen.fdk(y_static, gen.P_nom[None])[0]
    return dict(ck=ck, ca=ca, ds=ds, cfg=cfg, gen=gen, spacing=spacing, meas=meas,
                gt3=gt[0, 0], theta_true=theta_true, y=y, static_fdk=static_fdk)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="trained prior (train_fm3d.py checkpoint)")
    ap.add_argument("--root", default=None,
                    help="CQ500 root; default: the checkpoint's --root")
    ap.add_argument("--split", default="val", help="CQ500 split to draw the patient from")
    ap.add_argument("--out", default="data/posterior3d")
    ap.add_argument("--run", type=int, default=0, help="patient index within the split")
    ap.add_argument("--motion_kind", default="akima",
                    help="akima (the field standard, 10 zero-centred nodes per DoF) | mixed | "
                         "sinusoid | linear | jerk | step")
    ap.add_argument("--trans_mm", type=float, default=10.0,
                    help="full translation-node sampling width [mm], before zero-centering")
    ap.add_argument("--rot_deg", type=float, default=10.0,
                    help="full rotation-node sampling width [deg], before zero-centering")
    ap.add_argument("--seed", type=int, default=3,
                    help="seeds the motion draw, the tile jitter and the estimator (the cohort "
                         "convention is seed = 1000 + run)")
    # ---- the loop ---------------------------------------------------------------------------
    ap.add_argument("--n_steps", type=int, default=50, help="learned Euler updates (N)")
    ap.add_argument("--per", type=int, default=200,
                    help="pose-fitting iterations per outer step")
    ap.add_argument("--loss", default="l2",
                    help="projection-domain data term of the estimator: l2 | l2si | lncc | ncc")
    ap.add_argument("--estimator", default="akima_gd",
                    choices=["akima_gd", "bspline_rmsprop", "net"],
                    help="paper: akima_gd; pose ablation: bspline_rmsprop; legacy: net")
    ap.add_argument("--lr", type=float, default=None,
                    help="pose step size; defaults: Akima 1000, B-spline .001, legacy net .003")
    ap.add_argument("--views_per_iter", type=int, default=24,
                    help="random views per estimator iteration (stochastic view subsampling)")
    ap.add_argument("--est_coarse", type=int, default=2,
                    help="estimate the motion on a 1/N grid (volume pooled, panel binned) "
                         "until --est_coarse_until, then on the full grid (coarse-to-fine)")
    ap.add_argument("--est_coarse_until", type=float, default=0.5,
                    help="flow time at which the estimator switches to the full grid")
    ap.add_argument("--cg_iters", type=int, default=5,
                    help="CG iterations of the data-consistency step")
    ap.add_argument("--kappa", type=float, default=0.3, help="relaxed TV update weight")
    ap.add_argument("--tv_iters", type=int, default=5, help="TV denoiser iterations")
    ap.add_argument("--tv_step", type=float, default=0.015,
                    help="TV denoiser step, as a fraction of ||z|| per iteration")
    # ---- prior evaluation -------------------------------------------------------------------
    ap.add_argument("--context", default="auto", choices=["auto", "global", "none"],
                    help="conditioning channels; auto reads in_ch off the checkpoint")
    ap.add_argument("--blend", default="uniform", choices=["uniform", "hann"],
                    help="patch->volume scheme: uniform = --patch_offsets random non-overlapping "
                         "tilings averaged (arXiv:2512.18161); hann = overlapping Hann blend")
    ap.add_argument("--patch_offsets", type=int, default=2,
                    help="tilings blended per step (K); must be >= 2 for --blend uniform")
    ap.add_argument("--prior_batch", type=int, default=64, help="tiles per U-Net forward")
    ap.add_argument("--prior_amp", action="store_true", default=True,
                    help="fp16 autocast for the prior forward, as trained (--no-prior_amp: fp32)")
    ap.add_argument("--no-prior_amp", dest="prior_amp", action="store_false")
    ap.add_argument("--compile", action="store_true", default=True,
                    help="torch.compile the prior net (--no-compile to disable)")
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    # ---- readout ----------------------------------------------------------------------------
    ap.add_argument("--metric_every", type=int, default=5,
                    help="metrics/montage (or snapshot) every k steps; the last step always")
    ap.add_argument("--metric_mode", default="defer", choices=["defer", "inline"],
                    help="defer = write a snapshot (theta + fp16 x_t) per metric step and render "
                         "later with scripts/render_posterior3d.py; inline = score and draw "
                         "the montage inside the loop (~17 s per hit)")
    args = ap.parse_args()
    if args.blend == "uniform" and args.patch_offsets < 2:
        raise SystemExit("--blend uniform needs --patch_offsets >= 2 (a single non-overlapping "
                         "pass leaves tile seams)")

    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)
    # The estimator draws its network init and its per-iteration view subset from the global
    # RNG, so the global seed is what makes two runs of one command follow the same trajectory
    # (up to atomics in the kernels).
    torch.manual_seed(args.seed)

    world = build_world(ckpt=args.ckpt, dev=dev, root=args.root, split=args.split,
                        run=args.run, motion_kind=args.motion_kind, seed=args.seed,
                        trans_mm=args.trans_mm, rot_deg=args.rot_deg)
    ck, ca, cfg, gen = world["ck"], world["ca"], world["cfg"], world["gen"]

    # in_ch comes off the WEIGHTS: the checkpoint's args say what was launched, the weights say
    # what was trained, and a mismatch would feed the coordinate channels as image content.
    in_ch_ck = int(ck["ema"]["in_conv.weight"].shape[1])
    if args.context == "auto":
        args.context = "global" if in_ch_ck >= 5 else "none"
    in_ch = 5 if args.context == "global" else 1
    if in_ch != in_ch_ck:
        raise SystemExit(f"--context {args.context} wants in_ch={in_ch} but the ckpt was "
                         f"trained with in_ch={in_ch_ck}")
    print(f"prior: UNet3D in_ch={in_ch} (context={args.context}), "
          f"blend={args.blend} x {args.patch_offsets} tile grid(s)/step")

    model = UNet3D(in_ch=in_ch, base=ca["base"]).to(dev)
    model.load_state_dict(ck["ema"])
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    if args.compile:
        model = torch.compile(model)
        print("torch.compile: ON for the prior. --no-compile to disable.")

    spacing, meas = world["spacing"], world["meas"]
    gt3, theta_true = world["gt3"], world["theta_true"]
    y, static_fdk = world["y"], world["static_fdk"]

    # Spline coefficients persist across outer updates and the grid transition.
    default_lr = {"akima_gd": 1000.0, "bspline_rmsprop": 0.001, "net": 0.003}
    est_kw = dict(dx=gen.dx, dy=gen.dy, dz=gen.dz, loss=args.loss,
                  views_per_iter=args.views_per_iter,
                  lr=default_lr[args.estimator] if args.lr is None else args.lr)
    print(f"estimator: {args.estimator}, lr={est_kw['lr']} "
          f"views/iter={args.views_per_iter} loss={args.loss}")

    def build_grid(n):
        """Everything the estimator needs to work on a 1/n grid (volume pooled n, panel binned n)."""
        if n == 1:
            return dict(cfg=cfg, u=gen.u_coords, v=gen.v_coords, y=y[0], vox=gen.dx, n=1)
        g = ConeBeam3DConfig.thies(n_views=cfg.n_views, det_bin=cfg.det_bin * n)
        gu, gv = detector_coords_3d(g, device=dev)
        return dict(cfg=g, u=gu, v=gv, y=F.avg_pool2d(y[0][None], n)[0], vox=gen.dx * n, n=n)

    grids = {1: build_grid(1)}
    ec = args.est_coarse
    if ec > 1:
        grids[ec] = build_grid(ec)
        print(f"estimator grid: 1/{ec} -- volume {gen.shape[0] // ec}^3 @ {gen.dx * ec:g} mm, "
              f"panel {grids[ec]['cfg'].nv}x{grids[ec]['cfg'].nu}"
              + (f", switching to 1/1 at t={args.est_coarse_until:g}"
                 if args.est_coarse_until < 1.0 else ""))
    g0 = grids[ec]
    est_kw.update(dx=g0["vox"], dy=g0["vox"], dz=g0["vox"])
    est = make_estimator(args.estimator, g0["cfg"], gen.P_nom, g0["u"], g0["v"], dev, **est_kw)

    def use_grid(t):
        """Switch the fitting grid while retaining the fitted motion parameters."""
        gg = grids[ec] if (ec > 1 and t < args.est_coarse_until) else grids[1]
        est.cfg, est.u, est.v = gg["cfg"], gg["u"], gg["v"]
        est.dx = est.dy = est.dz = gg["vox"]
        return gg

    def est_ref(img, n):
        return img if n == 1 else F.avg_pool3d(img[None, None], n)[0, 0]

    with torch.no_grad():
        x = gen.fdk(y, gen.P_nom[None])[0]                               # cold start: uncorrected
    patch = ca["patch"]

    # Every volume in a montage is shown in the GT's frame: the input FDK's gauge is fitted once.
    m_cold, x0_input = aligned_metrics(x, gt3, spacing, mask=meas, iters=200, return_aligned=True)
    ms_cold = aligned_metrics(x, static_fdk, spacing, mask=meas, iters=200)
    print(f"cold start  " + json.dumps({k: round(v, 4) for k, v in m_cold.items()}))
    m_sfdk = aligned_metrics(static_fdk, gt3, spacing, mask=meas, iters=200)

    def tv_l1(v):
        return sum(float(g.abs().sum()) for g in grad_forward_3d(v[None, None]))

    with torch.no_grad():
        tv_gt = tv_l1(gt3)
        ceil_fdk = gen.fdk(y, params_to_Pmot(theta_true, gen.P_nom)[None])[0]
    m_ceil, ceil_aligned = aligned_metrics(ceil_fdk, gt3, spacing, mask=meas, iters=200,
                                           return_aligned=True)
    print(f"static FDK (vs GT)  {m_sfdk['psnr_aligned']:.2f} dB / SSIM {m_sfdk['ssim_aligned']:.3f}")
    print(f"FDK(theta_true) (vs GT)  {m_ceil['psnr_aligned']:.2f} dB / "
          f"SSIM {m_ceil['ssim_aligned']:.3f}")

    snap_dir = None
    if args.metric_mode == "defer":
        snap_dir = os.path.join(args.out, "snaps")
        os.makedirs(snap_dir, exist_ok=True)
        torch.save({"args": {**vars(args), "amp_units": AMP_UNITS}},
                   os.path.join(snap_dir, "meta.pt"))
        print(f"metric_mode=defer: snapshots -> {snap_dir}/  "
              f"(render: python scripts/render_posterior3d.py --out {args.out} [--watch])")

    hist = []
    theta_hist = []
    gauge_th = None                        # warm starts for the gauge fits
    xt_gauge = None
    gtile = torch.Generator(device=dev).manual_seed(args.seed)   # reproducible tile jitter
    N = args.n_steps
    for k in range(N):
        t_wall = time.time()
        t = k / N
        dt = 1.0 / N

        # 1. PREDICT
        x_prior = fm_predict(model, gen, x, t, dt, patch, context=args.context,
                             n_offsets=args.patch_offsets, generator=gtile, blend=args.blend,
                             batch=args.prior_batch, amp=args.prior_amp)

        # 2. ESTIMATE on the predicted image
        gg = use_grid(t)
        loss = est.refine_global(est_ref(x_prior, gg["n"]), gg["y"], iters=args.per)
        theta = est.current_params()
        theta_hist.append(theta.detach().clone())

        # 3. CORRECT: CG data step at theta_hat, then the relaxed TV update
        z = cg_dc_step(x_prior, theta, y[0], gen, iters=args.cg_iters)
        dp = float((z - x_prior).norm())
        z_post_data = z
        if args.kappa > 0:
            z = z + args.kappa * (sidky_dtv_denoise_3d(
                z[None, None], args.tv_iters, args.tv_step)[0, 0] - z)
        d_fm = float((x_prior - x).norm())
        d_tv = float((z - z_post_data).norm())
        tv_rel = tv_l1(z) / tv_gt                 # gradient energy relative to the ground truth
        ratio = dp / max(d_fm, 1e-12)
        x = z.detach()

        do_metric = k % max(args.metric_every, 1) == 0 or k == N - 1
        if do_metric and args.metric_mode == "defer":
            me = motion_error(theta, theta_true, cfg=cfg)
            sec = time.time() - t_wall
            hist.append({"step": k, "t": t, "loss": loss, "d_fm": d_fm, "d_dc": dp,
                         "d_tv": d_tv, "tv_rel": tv_rel, "dc_over_fm": ratio, "sec": sec, **me})
            tmp = os.path.join(snap_dir, f".step{k:03d}.tmp")          # write-then-rename
            torch.save({"step": k, "t": t, "theta": theta.detach().cpu(),
                        "x_t": x.half().cpu(), "loss": loss}, tmp)
            os.replace(tmp, os.path.join(snap_dir, f"step{k:03d}.pt"))
            print(f"step {k:3d} t={t:.2f} | fit {loss:.5f} | rot {me['rot_rmse_deg']:.2f} deg, "
                  f"obs {me['trans_obs_mm']:.2f} mm | dc/fm {ratio:.1f}x | tv {tv_rel:.3f} "
                  f"| {sec:.1f}s (snap)", flush=True)
        elif do_metric:
            with torch.no_grad():
                x_fdk = gen.fdk(y, params_to_Pmot(theta, gen.P_nom)[None])[0]
            m, gauge_th, x_fdk_al = aligned_metrics(x_fdk, gt3, spacing, mask=meas, iters=150,
                                                    init=gauge_th, return_theta=True,
                                                    return_aligned=True)
            mx, xt_gauge, x_al = aligned_metrics(x, gt3, spacing, mask=meas, iters=150,
                                                 init=xt_gauge, return_theta=True,
                                                 return_aligned=True)
            ms_out = aligned_metrics(x_fdk, static_fdk, spacing, mask=meas, iters=150)
            ms_xt = aligned_metrics(x, static_fdk, spacing, mask=meas, iters=150)
            me = motion_error(theta, theta_true, cfg=cfg)
            sec = time.time() - t_wall
            hist.append({"step": k, "t": t, "loss": loss, "d_fm": d_fm, "d_dc": dp, "d_tv": d_tv,
                         "dc_over_fm": ratio, "sec": sec, **m, **me,
                         **{f"xt_{q}": v for q, v in mx.items()},
                         **{f"s_{q}": v for q, v in ms_out.items()},
                         **{f"xts_{q}": v for q, v in ms_xt.items()}})
            print(f"step {k:3d} t={t:.2f} | fit {loss:.5f} | FDK(th) vsGT "
                  f"{m['psnr_aligned']:5.2f}/{m['ssim_aligned']:.3f} vsSFDK "
                  f"{ms_out['psnr_aligned']:5.2f}/{ms_out['ssim_aligned']:.3f} "
                  f"| x_t vsGT {mx['psnr_aligned']:5.2f}/{mx['ssim_aligned']:.3f} vsSFDK "
                  f"{ms_xt['psnr_aligned']:5.2f}/{ms_xt['ssim_aligned']:.3f} "
                  f"| rot {me['rot_rmse_deg']:.2f} deg, obs {me['trans_obs_mm']:.2f} mm "
                  f"| dc/fm {ratio:.1f}x | {sec:.1f}s", flush=True)
            montage(os.path.join(args.out, f"step{k:03d}.png"), gt3, x0_input, x_fdk_al, x_al,
                    ceil_aligned, k, t,
                    f"N={N} PER={args.per} cg{args.cg_iters} kappa={args.kappa:g} | theta rot "
                    f"{me['rot_rmse_deg']:.2f} deg, trans_obs {me['trans_obs_mm']:.2f} mm",
                    m_in=m_cold, m_out=m, m_xt=mx,
                    ms_in=ms_cold, ms_out=ms_out, ms_xt=ms_xt, m_ceil=m_ceil)
        else:
            print(f"step {k:3d} t={t:.2f} | fit {loss:.5f} | {time.time() - t_wall:.1f}s",
                  flush=True)

    # ---- final readout: FDK(theta_hat) and the final iterate, both vs GT and vs static FDK ----
    theta = est.current_params()
    with torch.no_grad():
        x_final = gen.fdk(y, params_to_Pmot(theta, gen.P_nom)[None])[0]
    fm, x_final_al = aligned_metrics(x_final, gt3, spacing, mask=meas, iters=300, init=gauge_th,
                                     return_aligned=True)
    _, x_al = aligned_metrics(x, gt3, spacing, mask=meas, iters=300, init=xt_gauge,
                              return_aligned=True)
    mx_final = aligned_metrics(x, gt3, spacing, mask=meas, iters=300, init=xt_gauge)
    fm_s = aligned_metrics(x_final, static_fdk, spacing, mask=meas, iters=300)
    mx_final_s = aligned_metrics(x, static_fdk, spacing, mask=meas, iters=300)
    print("\nFINAL FDK(theta_hat) vs GT   " + json.dumps({k: round(v, 4) for k, v in fm.items()}))
    print("FINAL FDK(theta_hat) vs sFDK " + json.dumps({k: round(v, 4) for k, v in fm_s.items()}))
    print("FINAL x_t            vs GT   " + json.dumps({k: round(v, 4) for k, v in mx_final.items()}))
    print("FINAL x_t            vs sFDK " + json.dumps({k: round(v, 4) for k, v in mx_final_s.items()}))
    torch.save({"theta": theta.cpu(), "theta_last": theta.cpu(),
                "theta_hist": torch.stack(theta_hist, 0).cpu(),   # (N,V,6)
                "theta_true": theta_true.cpu(), "hist": hist,
                "final": fm, "final_xt": mx_final,
                "final_s": fm_s, "final_xt_s": mx_final_s,       # vs the static-FDK reference
                "sfdk_vs_gt": m_sfdk,
                "x_final": x_final.half().cpu(), "x_t": x.half().cpu()},
               os.path.join(args.out, "result.pt"))
    montage(os.path.join(args.out, "final.png"), gt3, x0_input, x_final_al, x_al, ceil_aligned,
            N, 1.0, f"FINAL | N={N} PER={args.per} cg{args.cg_iters} kappa={args.kappa:g} "
            f"| {args.split} {args.run} seed {args.seed}",
            m_in=m_cold, m_out=fm, m_xt=mx_final,
            ms_in=ms_cold, ms_out=fm_s, ms_xt=mx_final_s, m_ceil=m_ceil)
    print(f"montages -> {args.out}/")


if __name__ == "__main__":
    main()
