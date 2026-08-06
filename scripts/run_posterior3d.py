"""Blind rigid motion correction: the PnP-TV predictor-corrector posterior loop, in 3D.

This is the configuration the 2D project converged on after a long ablation, transplanted whole.
Per Euler step k of N (t = k/N), starting cold from x = FDK(y, P_nom) and theta = 0:

    1. PREDICT   x_prior = x + dt * v_FM(x, t)            the learned prior moves first
    2. ESTIMATE  est.refine_global(x_prior, y, iters=PER) motion is fitted on the IMPROVED image
    3. CORRECT   PnP forward-backward on z:
                    z <- DATA_STEP(z, theta, y)           see --dc_op; DEFAULT = `cg`
                    z <- z + kappa * (TV(z) - z)
    4. x = z

THE DATA STEP WAS REPLACED ON 2026-07-24 and the default is now `cg`, not the 2D project's
normalized adjoint step. The adjoint direction A^T(Az - y) is dominated by LOW frequencies
(A^T A has a ~1/|k| kernel), so an improved theta -- whose payoff is streaks and edges, i.e. HIGH
frequencies -- barely reached the carried image: measured at the TRUE theta, an adjoint step
bought +0.09 dB where a filtered reconstruction of the same data bought +10.07 dB. The cure had
to be SPECTRAL. Diagonal preconditioners (SART, ASD-POCS) were tried first and LOST; a short
matched-adjoint CG solve (`cg`, DDS) and a ramp-filtered residual step (`fdk`) both won. See the
table at --dc_op, scripts/cmp_dcop.py, and data/dcop_*.

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
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.dataset_slab import AAPMSlabGenerator
from fm3d.geometry_3d import (ConeBeam3DConfig, detector_coords_3d,
                              measured_region_mask)
from fm3d.motion_estimation import make_estimator
from fm3d.prior_patch import predict_x1_patched
from fm3d.reg_metric import aligned_metrics
from fm3d.rigid_motion import AMP_UNITS, make_motion, motion_error, params_to_Pmot
from fm3d.tv import (div_adjoint_3d, grad_forward_3d, shrink, sidky_dtv_denoise_3d,
                     sidky_dtv_grad_3d)
from fm3d.unet_3d import UNet3D

DATA = "/home/mirlab/Desktop/Flow_matching_motion/data/AAPM_head_data"


@torch.no_grad()
def fm_predict(model, gen, x_mu, t, dt, patch, context="auto", n_offsets=1, generator=None,
               blend="uniform", batch=64, amp=False):
    """One Euler step of the FM ODE. Takes and returns a MU-space volume.

    The prior is evaluated PATCH-WISE and blended (`predict_x1_patched`), never on the whole slab
    at once, and that is not only about memory. UNet3D normalizes with GroupNorm, whose statistics
    are taken over the spatial extent -- so the same weights fed a 256^3 volume normalize
    differently than they did on the `--patch`^3 patches they were trained on (32^3 in the
    deployed checkpoint; `patch` comes off the ckpt args, never off a literal here). Running the
    net at a spatial size it never saw is a silent train/test mismatch. The blending is
    identity-exact, so tiling costs nothing.

    `blend`: "uniform" (DEFAULT) = the archive paper's non-overlapping random-tiling scheme
    (arXiv:2512.18161), which needs n_offsets>=2 (a single non-overlapping pass leaves seams).
    "hann" = overlapping stride patch/2 + Hann window. A/B on the 500k prior-only ODE (3 val
    patients, data/blend_cmp/): uniform K=2 = 25.35 dB / SSIM 0.765 in 11.2 min vs hann K=1 =
    25.42 / 0.765 in 27.5 min -- statistically indistinguishable quality, 2.5x cheaper, so uniform
    K=2 is the deployed default. K=1 uniform (24.96 / 0.749) is worse: seams intact, do not use.

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
                            context=context, n_offsets=n_offsets, generator=generator,
                            blend=blend, batch=batch, amp=amp)
    v = (x1 - x_net) / max(1.0 - t, 1e-3)
    return gen.from_net(x_net + dt * v)[0, 0]


def data_grad(x_mu, theta, y, gen, views=None):
    """grad_x 0.5 ||A_{P(theta)}(x) - y||^2, in mu space.

    `views` (a LongTensor of view indices) subsamples the sum over views -- an unbiased estimate
    of the same gradient direction, and the data-prox step only uses the normalized direction.
    None = all views (exact, and the default)."""
    x = x_mu.detach().requires_grad_(True)
    P = params_to_Pmot(theta, gen.P_nom)
    if views is not None:
        P, y = P[views], y[views]
    r = gen.project(x[None, None], P[None]) - y
    (0.5 * (r ** 2).sum()).backward()
    return x.grad


def _adjoint(s, P, gen):
    """A_P^T s -- LEAP's modular VD backprojector (gen.adjoint -> leap_backproject, mode
    'VD'). NOT the exact transpose of the forward: with the forward pinned to LEAP's Joseph
    kernel neither of LEAP's backprojectors is, and the defect is MEASURED, not asserted
    (gate_leap_projector T2: 3.5e-4 on a real sinogram, 1.3e-2 on white noise).

    HISTORY, because several regimes preceded this one and their lessons are in the memories: the
    autograd-of-a-zero-forward route (wasted a full march per call), the direct ray-march
    scatter (3.9 s/application, L2-footprint-bound floor), the toolkits' UNMATCHED voxel gather
    (35x cheaper but B A nonsymmetric; quality -0.5 dB in the A/B), and our own SF pair, which
    WAS matched to ~1e-6 and is now gates-only. The 2026-07-29/30 switch to LEAP traded that
    exactness back for one vendor-verified operator everywhere: we are again on an unmatched
    pair, at the defect measured above rather than at the toolkits' unmeasured one."""
    with torch.no_grad():
        return gen.adjoint(s[None], P[None])[0, 0]


def fdk_dc_step(x_mu, theta, y, gen, eta, meas, views=None):
    """Filtered-residual (FDK-preconditioned) soft data step:

        x <- x + eta * FDK_{P(theta)}( y - A_{P(theta)} x ),   on the measured barrel only.

    WHY. The raw adjoint direction A^T(y - Ax) is the unfiltered backprojection of the residual,
    and A^T A has a ~1/|k| kernel, so a gradient step injects the residual's LOW frequencies and
    starves the high ones -- but the payoff of an improved theta (streak removal, edges) is almost
    all HF. MEASURED at the TRUE theta on val 0: a normalized adjoint step bought +0.09 dB where a
    filtered reconstruction of the same data bought +10.07 dB. SART's row/column weights are
    DIAGONAL (spatial-coverage) corrections and cannot flatten that spectrum -- and the 3-way A/B
    (data/dcop_*, 2026-07-23) confirmed it: adj 31.11 dB beat sart 30.41 and sart+ASD 30.54.
    Putting the ramp filter INSIDE the step is the spectral preconditioner M ~= A^+, so the step
    is ~flat in frequency and the updated Pmat reaches x_t at every scale at once. This is also
    what the field does: Thies (arXiv:2401.09283) evaluates its autofocus metric on a
    differentiable FDK recon every iteration, never via adjoint steps.

    Because the FDK is self-normalized so FDK(A x) ~= x inside the barrel, the update is ~
    (1-eta) x + eta FDK(y) there -- but the RESIDUAL form (not a blend) is what leaves null-space
    and never-measured content to the prior. Outside `meas` the FDK of the residual is built from
    partial coverage, so the update is gated to the barrel (the `measured_region_mask` support).

    Two cautions, and the knobs that answer them:
      * FDK is NOT A^T (an unmatched projector/backprojector pair, Zeng & Gullberg 2000): eta = 1
        iterated can diverge. One step per ODE step at eta <= ~0.8 is safely inside the stable
        band; the default caps at 0.5.
      * A strong filtered step STAMPS a still-wrong theta into x_t (the 2D DDNM lesson). So eta
        RAMPS UP with t as theta converges -- the opposite schedule of the adjoint step's
        (1-t)^p decay, whose overshoot rationale (fixed-size normalized steps ping-ponging) does
        not apply to a residual-proportional step.
    """
    P = params_to_Pmot(theta, gen.P_nom)
    if views is not None:
        P, y = P[views], y[views]
    with torch.no_grad():
        resid = y - gen.project(x_mu[None, None], P[None])[0]
        corr = gen.fdk(resid[None], P[None])[0]
        return x_mu + eta * corr * meas.to(corr.dtype)


def cg_dc_step(x_mu, theta, y, gen, iters=5, lam=0.0, views=None):
    """A few conjugate-gradient iterations on the normal equations, warm-started at x:

        minimize_z 0.5||A_{P(theta)} z - y||^2 + 0.5*lam*||z - x||^2      (z0 = x)

    i.e. CG on (A^T A + lam I) z = A^T y + lam x. This is the DDS data-consistency step (Chung et
    al., ICLR 2024, arXiv:2303.05754): replace the one ill-conditioned gradient step with a short
    Krylov solve. CG's k-th iterate applies a degree-k polynomial of A^T A that approximates the
    INVERSE over the residual's spectrum, so a handful of iterations recovers the high frequencies
    a Landweber step starves (same disease fdk_dc_step treats). Early stopping is itself the
    regularizer (truncated-Krylov). lam > 0 adds an explicit proximal pull toward the warm start;
    lam = 0 relies on `iters` alone (the DDS default flavor).

    THE PAIR IS NO LONGER MATCHED, so `M` is not exactly symmetric and this is CG on a
    slightly non-symmetric operator. Since 2026-07-30 A is LEAP's Joseph modular forward and
    A^T is LEAP's VD backprojector (see `_adjoint`); the SF pair that WAS matched to ~1e-6 is
    gates-only. MEASURED on this geometry, 256^3 / 360 views / akima 10-10 motion:
        adjointness   <Au,s> vs <u,A^Ts>       3.5e-4 on a real sinogram, 1.25e-2 on noise
        symmetry      <Mu,w> vs <u,Mw>         7e-3 .. 1.2e-2 on image-like directions
    That is a real defect, not a rounding one, and CG has no convergence theorem here. What it
    does have is the measurement: run from the loop's own warm start (the cold FDK), ||Az - y||
    fell MONOTONICALLY over 8 iterations (2.13e3 -> 5.18e2) with p^T M p > 0 throughout -- no
    breakdown, no negative curvature. So the deployed --cg_iters 5 is empirically safe; treat
    a LARGE --cg_iters as unvalidated, since asymmetry compounds with Krylov depth.

    Wrong-theta stamping is bounded by `iters`, not by a step size: keep it small (3-5) while
    theta is still moving -- which is also what keeps the asymmetry above harmless.

    MEMORY. The volumes are trivial (256^3 fp32 = 64 MB) but each A(v) materializes a FULL
    SINOGRAM (~482 MB at 360 x 500 x 700). Peak is therefore set by how many sinograms are alive
    at once, so the intermediate is freed the moment the adjoint has consumed it -- without the
    `del`, Python keeps A(v) alive for the whole of AT() and the peak doubles.
    """
    P = params_to_Pmot(theta, gen.P_nom)
    if views is not None:
        P, y = P[views], y[views]

    def A(v):
        with torch.no_grad():
            return gen.project(v[None, None], P[None])[0]

    def AT(s):
        return _adjoint(s, P, gen)

    def M(v):                                   # (A^T A + lam I) v
        s = A(v)
        out = AT(s)
        del s                                   # free the sinogram before the next one is built
        return out + lam * v if lam > 0 else out

    # r0 = b - M(z0) with b = AT(y) + lam*z0 collapses, BY LINEARITY, to a single adjoint of the
    # data residual: AT(y) + lam*z0 - AT(A z0) - lam*z0 = AT(y - A z0), for ANY lam (the lam
    # terms cancel exactly at the warm start). The naive form spends TWO adjoints (AT(y) and the
    # one inside M) on the setup; the adjoint is the single most expensive kernel in the whole
    # loop (atomic scatter, ~5x the forward), so this is ~14% of the CG step for free.
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


def admm_dc_step(x_mu, theta, y, gen, state, *, rho, thresh, iters=5, views=None, dual=True):
    """ONE ADMM-TV sweep: the STANDARD form for "CG data solve + TV" (Boyd et al. 2011, Found. &
    Trends ML 3(1), Sec. 6.4.1), replacing our kappa-blend heuristic.

        minimize_z  0.5||A_{P(theta)} z - y||^2 + lam*||D z||_1      D = forward differences

        x:  z <- CG( A^T A + rho D^T D ,  A^T y + rho D^T(d - u) ,  warm start z )
        z:  d <- S_{lam/rho}( D z + u )                              exact prox, closed form
        u:  u <- u + D z - d                                         scaled dual (u = y_dual/rho)

    `state` is a dict carrying `d` and `u` ACROSS OUTER STEPS -- the "variable sharing" of
    DiffusionMBIR (Chung et al., CVPR 2023) and of DDS's own released 3D solver, which is exactly
    this routine with D = D_z only. One sweep per outer step is what both of them do. Pass a fresh
    {} to start; the tensors are allocated lazily on first use (2 x 3 x ~64 MB fp32 at 256^3).

    WHY THIS OVER THE KAPPA-BLEND. Our old corrector was `(1-kappa)z + kappa*D_tv(z)` with a
    normalized-gradient-descent D_tv. That has the SYNTAX of a Krasnosel'skii-Mann iteration but
    none of its semantics: KM needs a nonexpansive operator with a nonempty fixed-point set, and
    D_tv is provably expansive (its step is proportional to ||z|| while grad-TV is scale-free) and
    moves by a fixed length no matter how close z already is to a TV minimum, so Fix(D_tv) is just
    the constant images. Here the TV half is an EXACT prox and kappa is gone, replaced by (rho,
    lam) which have meanings: rho is the augmented-Lagrangian penalty (it only conditions the CG
    operator), and ONLY the ratio lam/rho enters the threshold.

    THE ONE SUBTLETY, and it is where a silent regression would come from. `cg_dc_step` computes
    its initial residual as `A^T(y - A z0)`, folding two adjoints into one -- valid there because
    the proximal centre IS the warm start, so the lam terms cancel. Here the centre is (d - u),
    NOT z0, so the cancellation does not happen and the rho term must be carried explicitly:

        r0 = A^T(y - A z0) + rho * D^T( d - u - D z0 )

    The extra piece is finite differences only, so the 7->6 adjoint saving survives intact.

    `dual=False` freezes u at zero, which turns this into HALF-QUADRATIC SPLITTING -- the free
    ablation (HQS is ADMM with the dual dropped). Note HQS then needs rho to grow to converge,
    which we do NOT schedule, so expect it to be the weaker arm.

    SCALE WARNING: our volumes are attenuation mu in [0, 0.06], not [0,1] images. DDS's published
    lam/rho = 4e-3 is ~7% of our entire dynamic range and would flatten the volume. Start around
    lam/rho ~ 2e-4 .. 6e-4 in mu units.
    """
    P = params_to_Pmot(theta, gen.P_nom)
    if views is not None:
        P, y = P[views], y[views]
    shape = x_mu.shape

    def A(v):
        with torch.no_grad():
            return gen.project(v[None, None], P[None])[0]

    def AT(s):
        # LEAP's VD backprojector, exactly as in cg_dc_step -- and carrying the same
        # non-symmetry of A^T A that `cg_dc_step`'s docstring measures. The rho*D^T D block
        # added below IS exactly symmetric, so it dilutes rather than compounds it.
        return _adjoint(s, P, gen)

    def DtD(v):
        return div_adjoint_3d(*grad_forward_3d(v[None, None]), (1, 1) + tuple(shape))[0, 0]

    if state.get("d") is None:                        # lazy init on the gradient field
        dz, dy, dx = grad_forward_3d(x_mu[None, None])
        state["d"] = [torch.zeros_like(dz), torch.zeros_like(dy), torch.zeros_like(dx)]
        state["u"] = [torch.zeros_like(dz), torch.zeros_like(dy), torch.zeros_like(dx)]
    d, u = state["d"], state["u"]

    # ---- x-update: CG on (A^T A + rho D^T D) z = A^T y + rho D^T (d - u) -------------------
    z = x_mu.detach().clone()

    def M(v):
        s = A(v)
        out = AT(s)
        del s
        return out + rho * DtD(v)

    dz0, dy0, dx0 = grad_forward_3d(z[None, None])
    r = AT(y - A(z)) + rho * div_adjoint_3d(d[0] - u[0] - dz0, d[1] - u[1] - dy0,
                                            d[2] - u[2] - dx0, (1, 1) + tuple(shape))[0, 0]
    del dz0, dy0, dx0
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

    # ---- z-update (exact prox) and u-update -------------------------------------------------
    gz = grad_forward_3d(z[None, None])
    for i in range(3):
        d[i] = shrink(gz[i] + u[i], thresh)      # thresh IS lam/rho; lam never appears alone
        if dual:
            u[i] = u[i] + gz[i] - d[i]
    return z.detach()


def _panel_label(name, m_gt=None, m_st=None):
    """"name" + up to TWO reference scores, because the two references RANK THESE VOLUMES
    OPPOSITELY (vs GT the carried x_t wins; vs static FDK the output wins) and the user wants both
    carried. A single figure-level title cannot say which volume it grades. `m_gt`/`m_st` are the
    aligned_metrics dicts vs the GT volume and vs the static FDK; either may be None."""
    s = name
    if m_gt:
        s += f"\nvs GT   {m_gt['psnr_aligned']:.2f} dB / {m_gt['ssim_aligned']:.3f}"
    if m_st:
        s += f"\nvs sFDK {m_st['psnr_aligned']:.2f} dB / {m_st['ssim_aligned']:.3f}"
    return s


def montage(path, gt, x0, x, xt, ceil, step, t, title,
            m_in=None, m_out=None, m_xt=None,
            ms_in=None, ms_out=None, ms_xt=None, m_ceil=None):
    """Axial + coronal: cold FDK | FDK(theta_hat) OUTPUT | x_t | FDK(theta_TRUE) | GT.

    THE RECONSTRUCTIONS MUST ALREADY BE RIGIDLY ALIGNED TO `gt` (see reg_metric's
    `return_aligned`). Blind motion recon has an exact SE(3) gauge, so each volume sits at its own
    arbitrary pose; slicing them raw at z = D//2 would show different anatomical planes side by
    side. Callers pass the aligned volumes, and each has its OWN gauge fit -- x_t is not FDK(theta).

    TWO reference columns are scored on each reconstruction: `m_*` vs the GT volume, `ms_*` vs the
    STATIC FDK (Thies' protocol). Pass both -- those numbers are unchanged.

    THE FOURTH PANEL IS FDK(theta_TRUE), NOT THE STATIC FDK (user, 2026-07-27), because the static
    FDK is the reconstruction of a DIFFERENT, motion-free scan and NO method working on this data
    can reach it with an FDK -- putting it in the panel labelled "ceiling" was misleading. The
    reachable ceiling for an FDK deliverable is this same data reconstructed with the TRUE motion,
    and the point of showing it is that IT STILL HAS STREAKS: FDK is an analytic inverse for a
    circular, equiangular orbit, and per-view rigid motion moves the source off that orbit relative
    to the object. Measured on val 0 (aligned, vs GT): static FDK 33.34 dB / 0.8256, FDK(theta_true)
    32.01 / 0.7294, CG(theta_true) 41.04 / 0.9844 -- i.e. motion costs FDK 0.096 SSIM and GAINS CG
    0.015, so the loss is the OPERATOR's, not the data's.

    Window is a HEAD-CT bone window in mu [1/mm]: soft tissue sits at ~0.02 (mu_water) and cortical
    bone runs to ~0.05-0.06.0.0-0.05 keeps bone on the ramp (a tighter vmax saturates the skull)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D, H, _ = gt.shape
    zc, yc = D // 2, H // 2
    lo, hi = 0.0, 0.05
    panels = [(_panel_label("input (cold FDK, aligned)", m_in, ms_in), x0),
              (_panel_label("FDK(theta_hat) = OUTPUT", m_out, ms_out), x),
              (_panel_label("x_t (carried PnP state)", m_xt, ms_xt), xt),
              (_panel_label("FDK(theta_TRUE) = reachable FDK ceiling", m_ceil), ceil),
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


def build_world(*, ckpt, dev="cuda", root=None, split="val", data=DATA,
                run=0, z0=0, motion_kind="akima", seed=3, trans_mm=10.0, rot_deg=10.0):
    """Rebuild the EXACT world the prior was trained in, plus the simulated corrupted scan.

    The dataset choice, geometry and grid all come off the checkpoint, not off script defaults:
    a CQ500 prior fed AAPM slabs under the non-Thies geometry loads without error and is silently
    out of distribution.

    SHARED with the deferred renderer (scripts/render_posterior3d.py), which reconstructs
    gt / y / static FDK from the run's (ckpt, run, seed) alone instead of shipping a ~480 MB
    sinogram per run dir. That works because everything here is DETERMINISTIC given those:
    the volume is a file read, `make_motion` runs on its own seeded generator, and the Triton
    forward kernel has no atomics. (The renderer must run the same code version as the loop --
    the standing assumption for every script in this repo.)
    """
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    ca = ck["args"]
    ds = ca.get("dataset")
    if ds is None:
        if "shape" in ca or "root" in ca:
            raise SystemExit("checkpoint has no 'dataset' key but carries CQ500-style args "
                             "(shape/root) -- refusing to guess which generator it trained on")
        ds = "aapm"        # checkpoints predating --dataset could only be AAPM slab runs
    if ds == "cq500":
        cfg = ConeBeam3DConfig.thies(n_views=ca["views"])            # the trainer's geometry
        root = root or ca.get("root")
        if not root or not os.path.isdir(root):
            raise SystemExit(f"CQ500 root {root!r} not found -- pass --root")
        # THE SIMULATION GRID COMES OFF THE CHECKPOINT TOO. y is the one thing the prior was
        # trained against, and `sim_grid` decides whether it is projected from the native
        # 612^3 truth or from the 1 mm inversion grid (the inverse crime). Both default to
        # "native", so this is currently a no-op -- but leaving it implicit is exactly how a
        # coarse-trained prior would get evaluated on native data without a word of warning,
        # the same failure class the dataset/shape/views checks above exist to stop.
        gen = CQ500Generator(root, cfg, device=dev, split=split,
                             shape=tuple(ca["shape"]), voxel_mm=1.0,
                             sim_native=(ca.get("sim_grid", "native") == "native"))
    elif ds == "aapm":
        cfg = ConeBeam3DConfig(det_bin=2, n_views=ca["views"])
        gen = AAPMSlabGenerator(data, cfg, device=dev, slab=ca["slab"],
                                in_plane=ca["in_plane"])
    else:
        raise SystemExit(f"unknown dataset {ds!r} in checkpoint")
    # No FBP scale to reconcile: the FDK is self-normalized by SOD*SDD/2
    # (projector_3d._fdk_physical_norm), so the operator constant is identical in the
    # trainer and here by construction rather than by injection.
    print(f"dataset {ds}: grid {gen.shape} @ ({gen.dz:g},{gen.dy:g},{gen.dx:g}) mm | "
          f"{cfg.n_views} views | anchor={ca.get('anchor', '?')}")

    spacing = (gen.dz, gen.dy, gen.dx)
    meas = measured_region_mask(gen.shape, spacing, cfg, device=dev)

    # ---- simulate the motion-corrupted scan, and the STATIC FDK = FDK of the MOTION-FREE scan
    # = Thies' reference AND the bridge's t=1 training target. Two roles: (1) scored beside the
    # GT in every montage/metric (vs GT and vs static FDK rank output-vs-x_t oppositely, and
    # only the vs-static number is comparable to Thies' 0.94); (2) itself an honest CEILING
    # panel (the best FDK can do with a perfect, motion-free orbit). Computed once.
    gt = gen.volume(run) if ds == "cq500" else gen.volume(run, z0)   # (1,1,D,H,W) mu
    # AMPLITUDE IS AN EXPLICIT ARGUMENT, because leaving it implicit silently changed how hard
    # the task was. make_motion's own defaults are (3,3,2) mm / (1.5,1.5,2) deg -- a head-scale
    # range we picked -- while THIES EVALUATES AT 5 mm / 5 deg on Akima splines. Every number this
    # project has compared to Thies' SSIM 0.94 was therefore produced on a SUBSTANTIALLY EASIER
    # problem (~60% of his translation, ~40% of his rotation, and a non-standard motion profile).
    # ALL AMPLITUDES ARE PEAK-TO-PEAK (fm3d/rigid_motion header). Our 10/10 is 2x Thies'
    # own 5/5 evaluation, kept deliberately. The
    # defaults below are kept so every earlier run in data/ stays reproducible.
    amp = {}
    if trans_mm is not None:
        amp["trans_mm"] = trans_mm
    if rot_deg is not None:
        amp["rot_deg"] = rot_deg
    theta_true = make_motion(motion_kind, cfg.n_views, device=dev, seed=seed, **amp)
    with torch.no_grad():
        if ds == "cq500":
            # THE DATA comes off the NATIVE simulation grid (dataset_cq500.simulate); everything
            # downstream -- the estimator's forward model, CG, the FDKs -- keeps inverting on the
            # coarse grid, exactly as in training. aapm slabs have no native source, so they stay.
            y = gen.simulate(run, params_to_Pmot(theta_true, gen.P_nom)[None])
            y_static = gen.simulate(run, gen.P_nom[None])
        else:
            y = gen.project(gt, params_to_Pmot(theta_true, gen.P_nom)[None])
            y_static = gen.project(gt, gen.P_nom[None])
        static_fdk = gen.fdk(y_static, gen.P_nom[None])[0]
    return dict(ck=ck, ca=ca, ds=ds, cfg=cfg, gen=gen, spacing=spacing, meas=meas,
                gt3=gt[0, 0], theta_true=theta_true, y=y, static_fdk=static_fdk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=DATA, help="AAPM data dir (aapm checkpoints only)")
    ap.add_argument("--root", default=None,
                    help="CQ500 root (cq500 checkpoints); default: the checkpoint's --root")
    ap.add_argument("--split", default="val",
                    help="CQ500 split to draw the test patient from (cq500 checkpoints only)")
    ap.add_argument("--out", default="data/posterior3d")
    ap.add_argument("--run", type=int, default=0,
                    help="volume index: patient index (cq500) / run index (aapm)")
    ap.add_argument("--z0", type=int, default=0, help="slab start slice (aapm only)")
    # AKIMA 5 mm / 5 deg IS THE DEFAULT, because that is what the REST OF THIS PIPELINE ALREADY
    # USES and what the field evaluates on:
    #     val_fm3d.py     make_motion("akima", ...), --trans_mm 10 --rot_deg 10 (p2p)
    #     train_fm3d.py   random_motion(kind="akima", n_nodes=10) -- the same FAMILY and node
    #                     count, but since 2026-07-28 the training AMPLITUDE follows Thies II-B
    #                     instead (--motion_amp thies: per-DoF u~U(0,1) of a 10 mm / 15 deg
    #                     maximum). That is a deliberate train-WIDER-than-eval split, not a
    #                     mismatch: the eval point 5/5 sits inside the training support, and the
    #                     paper's own rationale is coverage of the states an optimizer traverses.
    #     Thies (IEEE TMI 2025, refs/thies_*.txt:504)  Akima, 5 mm / 5 deg for evaluation
    # This loop used to default to `mixed` at make_motion's (3,3,2) mm / (1.5,1.5,2) deg, which
    # was a TRAIN/TEST MISMATCH INSIDE OUR OWN PIPELINE: the prior learned the bridge for Akima
    # 5/5 motion and we were testing it on a different motion FAMILY at roughly half the
    # amplitude. `random_motion`'s own docstring warns about exactly this -- "train the prior on
    # the same motion family the field evaluates on, or the bridge it learns is not the bridge
    # inference walks". Measured cost of the mismatch (2026-07-26, 3 patients, same config):
    # output vs static FDK 0.915 at 3mm/2° but 0.763 at akima 5/5, and theta rot 0.065 -> 0.738
    # deg. Everything in data/ from before this change (tvv_*, night_*, fair_*, cg_v*) was run at
    # the old default; reproduce those with `--motion_kind mixed --trans_mm 6 --rot_deg 4`
    # (p2p; they were written as 3 / 2 in the pre-2026-07-28 half-range units).
    ap.add_argument("--motion_kind", default="akima",
                    help="akima (DEFAULT; the field standard, matches our training) | mixed | "
                         "sinusoid | linear | jerk | step. The non-akima kinds are OURS and exist "
                         "as honesty controls, not for headline numbers.")
    ap.add_argument("--trans_mm", type=float, default=10.0,
                    help="peak translation [mm], isotropic. 5 = Thies' evaluation amplitude and "
                         "our training amplitude.")
    ap.add_argument("--rot_deg", type=float, default=10.0,
                    help="peak rotation [deg], isotropic. 5 = Thies' evaluation amplitude and "
                         "our training amplitude.")
    ap.add_argument("--seed", type=int, default=3)
    # 2D's full (N,PER) grid (motion-estimator-direction): the ORACLE CEILING rises MONOTONICALLY
    # with N (N10->50 = +2.6 dB / +0.027 SSIM) -- N is the ceiling knob, fewer ODE steps CANNOT be
    # recovered by more PER, and the user's by-eye optimum was N50. (The earlier "deploy N~30" note
    # was superseded by that grid.)
    ap.add_argument("--n_steps", type=int, default=50)
    # 2D deploys PER~10, but a 2D inner iteration sees the WHOLE sinogram while ours sees
    # `views_per_iter` (24) random views -- the same count is far less information here. This
    # project's own convergence sweep (24 views/iter, oracle image) reads 250 it = 0.84 deg,
    # 500 = 0.20, 1500 = 0.17, and REBOUNDS to 0.28 by 2250 with no lr decay. At n_steps=30 the
    # estimator warm-starts across steps, so PER=50 accumulates 30*50 = 1500 iters over the loop
    # -- landing at the knee and BEFORE the 2250-iter rebound.
    # PER 200 is the deployed value (user's call, 2026-08-03), down from 400. REPLACES the old
    # note here, whose premise is REFUTED: "the estimator is only ~15% of a step (49.0 s blind vs
    # 41.7 s oracle)" was measured on the RETIRED ray-march operator. Re-measured under LEAP on
    # the 500k prior: 949 s blind vs 223 s with --theta_oracle, i.e. the estimator is 76-84% OF
    # THE STEP. PER is therefore the only real lever on inference time, and 400 was buying the
    # tail of a converged fit.
    #
    # 3-patient blind A/B (500k ckpt, val 0/1/2, akima 10/10 p2p, everything else at the winning
    # set), x_t aligned vs GT -- the deliverable:
    #     c2f PER 400   948 s   38.53 dB / SSIM 0.9869   rot 0.100 deg
    #     c2f PER 200   558 s   38.60 dB / SSIM 0.9853   rot 0.142 deg   <- -41% wall clock
    # and the montages are indistinguishable by eye on both patients rendered.
    #
    # WHAT THE NOISE BAR ALLOWS YOU TO CONCLUDE. The loop is not bit-reproducible (atomics), and
    # the rerun spread is CONFIG-DEPENDENT: identical-command reruns differ by 0.0018 SSIM at
    # PER 400 but 0.0047 at PER 200 (less converged -> more variable). So the -0.00165 SSIM
    # deficit is NOT resolved -- "undetectable at this rig", not "free". Two residual hints that
    # it is real: the sign is the same on 3/3 patients, and rot genuinely degrades (0.100 ->
    # 0.142 deg, outside its own 0.03 deg bar). rot costs the OUTPUT nothing (FDK is already the
    # ceiling, 0.1 dB) but x_t 1.68 dB, so a harder regime -- larger amplitude, worse cold start
    # -- may break PER 200 first. Validated at 10/10 p2p only.
    #
    # WHAT IS BELOW THE KNEE. PER 100 does NOT converge: its rot curve is STILL DESCENDING at
    # step 49 (fine grid: 3.18 -> 1.62 @5 -> 0.36 @20 -> 0.22 @49, against PER 400's plateau at
    # ~0.16 by step 20), and it loses 0.0096 SSIM = 5x the bar. PER 50 collapses outright
    # (-3.3 dB / -0.019 SSIM). The loop is path-dependent, so the damage is done EARLY: PER 100
    # is at 1.62 deg where PER 400 is at 0.45 deg by step 5, and no amount of late refinement
    # undoes the geometry that got baked into x_t. Do not cut below 200 without re-running the
    # 3-patient A/B.
    ap.add_argument("--per", type=int, default=200)          # motion iters per ODE step
    ap.add_argument("--estimator", default="net")            # net = hashbl, the 2D default
    # PLAIN L2 (user's call, 2026-07-28). `l2si` was inherited from the 2D project, where the
    # flow-matching push and the data-residual update disagreed about the image's overall
    # brightness and a plain L2 sinogram term charged that scale drift to the motion parameters.
    # That failure mode does NOT exist here: `scripts/exp_loss_l2_geom.py` measures the optimal
    # scale c = <p,y>/<p,p> at 1.0007 +- 0.0001 on EVERY reference image the loop hands the
    # estimator (cold FDK, the carried x_t at each ODE step, GT), with cos(grad_l2, grad_l2si)
    # >= 0.997 and a norm ratio of 1.00. The CG data step is a least-squares fit to y, so it pins
    # x_t's magnitude every step and the drift never appears. Since L_l2si = ||y||^2 sin^2(angle)
    # is blind to magnitude BY CONSTRUCTION, keeping it would throw away a real degree of freedom
    # for a 0.07% nuisance. l2si remains available for datasets where the drift does exist.
    ap.add_argument("--loss", default="l2")                  # l2 | l2si | lncc | ncc | ramp | l1
    ap.add_argument("--lncc_win", type=int, default=9)
    ap.add_argument("--lr", type=float, default=None,
                    help="estimator lr; default None = the encoder's matched lr (see "
                         "--est_band: fullband 1e-3, hashbl 1e-2) or the estimator's own "
                         "default (direct/basis 0.3)")
    # ---- ENCODER BANDWIDTH, and why it comes WITH a learning rate ------------------------
    # Oracle sweep, 2026-07-24 (scripts/exp_est_sweep.py --suite oracle --ref gt, 2500 iters =
    # the loop's own N50 x PER50 budget, CQ500 val 0, l2si, rot RMSE):
    #
    #   encoder                  lr 1e-2            lr 1e-3
    #   hashbl   4/2/2.0         0.173 deg          0.363 deg   <- cannot converge in budget
    #   fullband 16/16/1.5       0.253 deg          0.035 deg   <- 5x better than the old point
    #
    # THE TWO AXES DO NOT SEPARATE. Wide bandwidth supplies the CAPACITY to represent the
    # trajectory; the small lr is the implicit regularizer that stops that capacity being spent
    # on view-to-view jitter -- fullband at lr 1e-2 peaks at 0.048 by iter 1425 and then DECAYS
    # to 0.253, the jitter-overfit signature the 2D project predicted. The band-limited encoder
    # is the OTHER cure for the same disease (and remains correct at lr 1e-2, which is what
    # [[motion-estimator-direction]] measured in 2D) -- it is simply the weaker of the two here,
    # and it cannot be run at 1e-3 because its small capacity then needs more than 2500 iters.
    # So bandwidth and lr are ONE knob and `--est_band` sets both; override lr explicitly only
    # if you mean to break the pair.
    # Cost is unchanged (both ran ~8 min at views 24), so this is free accuracy.
    ap.add_argument("--est_band", default="fullband", choices=["fullband", "hashbl"],
                    help="motion-encoder bandwidth. fullband (DEFAULT) = stock Instant-NGP "
                         "16/16/1.5 as in AI_Geocal, paired with lr 1e-3. hashbl = the 2D "
                         "band-limited 4/2/2.0, paired with lr 1e-2. The lr comes with the "
                         "choice unless --lr is given.")
    ap.add_argument("--views_per_iter", type=int, default=24)
    # SOFT-DC STEP. `alpha` is a FIXED fraction of ||z|| along a NORMALIZED direction, so it does
    # not shrink as the data term converges -- it overshoots and ping-pongs. MEASURED on val 0
    # (kappa=0, l2si): the data-prox moved x_t by exactly 10.00%/step while the FM prior moved it
    # 0.72%/step (14:1), and over 10 steps the 101% of accumulated path length produced only 15.8%
    # of net displacement -- 84% of the motion cancelled, so x_t oscillated in place and the
    # learned prior contributed essentially nothing to the carried state.
    # FIX: alpha 0.1 -> 0.02 puts the data step within ~3x of the prior step instead of 14x, and
    # alpha_p=2 (2D's adopted ALPHA_DECAY, alpha*(1-t)^p) anneals the overshoot away as t->1.
    ap.add_argument("--alpha", type=float, default=0.02)     # data-prox step
    ap.add_argument("--alpha_p", type=float, default=2.0)    # alpha * (1-t)^p decay; 0 = off
    # TV corrector, at the 2D project's by-eye sweet spot. The two settings are a PAIR and sit on
    # a trade-off ridge in kappa*step: (0.3, 0.03) and (0.5, 0.015) both read streak-free, while
    # (0.5, 0.03) streaks and (0.3, 0.015) blurs. kappa is the streak lever, not step.
    # THE TV TRIPLE (kappa 0.3, tv_step 0.03, tv_iters 5) IS THE 2D PROJECT'S WINNING SETTING,
    # transplanted whole. What its ablation actually established (10 indices, aligned SSIM):
    #   * kappa and tv_step sit on a TRADE-OFF RIDGE in their product: (0.3, 0.03) and (0.5, 0.015)
    #     both read streak-free by eye; (0.5, 0.03) streaks and (0.3, 0.015) blurs.
    #   * The ORACLE CEILING is FLAT over tv_iters 4-15, so iters is a LANDING knob, not a ceiling
    #     knob -- but iters <= 2 genuinely LOWERS the ceiling (too little TV -> streaky reference
    #     -> worse theta). Never go below 4. 5 is the pick: sharper x_t and 3x cheaper than 15.
    #   * PATH beats total: at matched step*iters, few-big-steps (low iters) keeps texture while
    #     many-tiny-steps (low step) over-converges to the flat TV minimum and blurs.
    #   * Uniform TV_WX/TV_WY scaling is a NO-OP -- the denoiser normalizes the gradient. The real
    #     strength knobs are exactly these three.
    # NOT YET VALIDATED HERE: every run in the 5-way data-step A/B had TV OFF (--kappa 0), so
    # cg-plus-TV is untested, and cg's x_t is already the cleanest volume in that comparison -- TV
    # may now cost sharpness rather than buy streak suppression. Reproduce the promoted data-step
    # numbers with `--kappa 0`; A/B kappa before trusting the default.
    ap.add_argument("--kappa", type=float, default=0.3)      # TV pull; pairs with tv_step 0.03
    # 5. The 2D by-eye K=1 sweet spot used 15, but on our 3D volume the user judged that TV too
    # strong (over-smoothed / bone washed out), so we run the sharper end of the 2D metric sweep,
    # where the ceiling was flat over iters 4-8 and x_t sharpness improved as iters fell. Do not go
    # below 4 (too little TV -> streaky reference -> worse theta).
    ap.add_argument("--tv_iters", type=int, default=5)
    # tv_step 0.015 (was 0.03) = the SOFTER corner of the 2D by-eye kappa*step ridge, adopted
    # 2026-07-25 from the 3-patient 2x2 (data/tvv_*, N=50, fullband, cg5, kappa 0.3):
    #   carried x_t vs GT   val0 36.56 -> 37.00   val1 36.17 -> 36.59   val2 37.74 -> 38.35
    # i.e. +0.49 dB mean, ~+0.3 dB vs the static FDK too, at ZERO cost -- and the OUTPUT
    # FDK(theta_hat) and theta rot are unchanged to within noise. A weaker TV smears the carried
    # state less; the output only ever sees theta, which TV does not move here.
    # It PAIRS with kappa 0.3 (the ridge's other coordinate, see --kappa): do not change one
    # without re-running the pair. The same sweep KILLED views_per_iter 48 -- its large
    # oracle-sweep rot win did NOT transfer in-loop (x_t +0.15 dB, below tv015's, at 1.3x cost).
    ap.add_argument("--tv_step", type=float, default=0.015)  # frac of ||z|| per TV iter
    # `pnp_k` and `cg_iters` LIVE AT DIFFERENT LEVELS and are not interchangeable.
    #   pnp_k     = PnP forward-backward ALTERNATIONS: (data step, denoise) repeated k times. It
    #               belongs to the splitting, not to the data operator. With --kappa 0 (and no
    #               --asd) the denoiser is off, so pnp_k IS simply "repeat the data step k times"
    #               -- k INDEPENDENT, memoryless steps, each one a fixed-size move along the
    #               current residual direction.
    #   cg_iters  = iterations INSIDE one cg_dc_step solve. CG's iterations are COUPLED: each
    #               builds a direction conjugate to all previous ones and takes the optimal step
    #               along it, so k iterations minimize over a k-dimensional Krylov subspace. That
    #               is not "k steps"; it is one solve of depth k.
    # The distinction is exactly the hypothesis under test (2026-07-24): is a memoryless k-step
    # walk equivalent to a coordinated k-dimensional solve? Theory says no, but that the walk
    # catches up if you give it ~sqrt(kappa) times as many steps.
    # CONSEQUENCE: for --dc_op cg, do NOT raise pnp_k to add depth -- `--pnp_k 5 --cg_iters 5`
    # RESTARTS a 5-iteration solve five times, discarding the Krylov basis each time, and is
    # strictly worse than one 25-iteration solve at the same cost. Deepen cg with --cg_iters.
    ap.add_argument("--pnp_k", type=int, default=1)          # data-prox <-> denoise alternations
    ap.add_argument("--dc_views", type=int, default=0,
                    help="random views per data-prox gradient; 0 = all views (exact, default)")
    # ---- data step operator -------------------------------------------------------------
    # THE DEFAULT IS `cg`, promoted 2026-07-24 after the 5-way A/B (val 0, 500k ckpt, l2si, N=50,
    # PER=50, uniform K=2, kappa 0; scripts/cmp_dcop.py, data/dcop_*):
    #
    #   dc_op          OUTPUT FDK(theta)   carried x_t     theta rot   rot<0.5deg at
    #   adj (was)      31.11 / 0.790       29.47 / 0.879   0.23 deg    step 24
    #   sart           30.41 / 0.779       27.68 / 0.856   0.38        step 23
    #   sart+ASD-POCS  30.54 / 0.790       25.25 / 0.804   0.49        step 44
    #   fdk            31.53 / 0.791       34.51 / 0.929   0.24        step 21
    #   cg  <-- WINS   31.76 / 0.798       36.13 / 0.957   0.19        step  8
    #
    # CAVEAT ON THAT TABLE: those five runs predate the global seeding below, so each followed its
    # own estimator trajectory. The size of that noise is measurable from the runs themselves --
    # step 0's theta is fitted BEFORE any data step, so all five should report the SAME rot, and
    # they report 1.51 / 1.61 / 1.78 / 1.86 / 1.99 deg. The adj-vs-spectral verdict is far outside
    # that band (5-7 dB on x_t, and it is visible by eye); the fdk-vs-cg ORDERING is not, and
    # wants a seeded repeat before it is treated as settled.
    # Both SPECTRAL steps beat both DIAGONAL ones and the raw adjoint on every axis, which is the
    # whole point: the soft-DC deficit was that A^T A has a ~1/|k| kernel, so a gradient step
    # injects the residual's LOW frequencies while the payoff of an improved Pmat is nearly all
    # HIGH -- and SART's row/column weights are a SPATIAL correction that cannot flatten a
    # spectrum. Confirmed by eye too (data/dcop_zoom_xt3.png): adj's x_t is waxy with a blunted
    # inner table, cg reproduces the GT's irregular one.
    # `sart` (and its --sart_beta / --sart_beta_red knobs) was REMOVED 2026-08-06: it lost the
    # 5-way A/B above on every axis and the old drivers that exercised it (drive_dcop_compare,
    # drive_dc_fair3, drive_isocost_dc, drive_day3) are archival records, not rerun targets.
    ap.add_argument("--dc_op", default="cg", choices=["adj", "fdk", "cg", "admm"],
                    help="cg (DEFAULT, winner) = DDS-style short CG solve with the matched "
                         "adjoint (see cg_dc_step). admm = the STANDARD ADMM-TV form (Boyd "
                         "Sec. 6.4.1; = DDS's own 3D solver) -- CG x-update + exact TV prox + "
                         "dual; it SUBSUMES the TV corrector, so --kappa is ignored with it. "
                         "fdk = filtered-residual FDK-preconditioned step, 2nd place, ~2x cheaper "
                         "(see fdk_dc_step). adj = normalized raw-adjoint soft step (the 2D "
                         "recipe; only +0.09 dB even at the TRUE theta).")
    # ---- fdk / cg data steps (the SPECTRAL preconditioners) ------------------------------
    # eta ramps UP with t -- eta_min + (eta - eta_min) * t^p -- because a filtered step stamps a
    # still-wrong theta into x_t (2D's DDNM lesson), and theta converges as t grows. This is the
    # OPPOSITE of the adjoint step's (1-t)^p decay: that decay answered fixed-size-step overshoot,
    # which a residual-proportional step does not have.
    ap.add_argument("--fdk_eta", type=float, default=0.5,
                    help="fdk step size at t=1 (<= ~0.8; FDK is an unmatched backprojector)")
    ap.add_argument("--fdk_eta_min", type=float, default=0.1, help="fdk step size at t=0")
    ap.add_argument("--fdk_eta_p", type=float, default=1.0, help="eta ramp exponent in t")
    ap.add_argument("--cg_iters", type=int, default=5,
                    help="CG iterations per data step (early stopping IS the regularizer)")
    ap.add_argument("--cg_lam", type=float, default=0.0,
                    help="proximal pull toward the warm start; 0 = truncated-Krylov only (DDS)")
    # ---- THE OPERATOR (2026-07-29/30, user decisions, retraining accepted) ----------------
    # There is no operator knob anymore. forward_project_3d_batched and gen.adjoint route every
    # call -- estimator (d/dP included), CG/ADMM pair, y simulation, trainer bridge, FDK -- to
    # LEAP modular-beam: forward = LEAP's JOSEPH kernel, PINNED (set_forceJosephModular in
    # leap_projector._model, via our patch to the vendored library, refs/LEAP/FM3D_PATCH.md,
    # so the geometry can no
    # longer flip the model mid-run); backward = LEAP's VD backprojector, which also carries
    # our FDK (leap_fdk_backproject folds LEAP's ray weight back to our 1/w^2 convention).
    # The retired flavors and why, so nobody reinvents them:
    #   ray-march + scatter transpose   exact pair, but the scatter was 3.9 s/application
    #                                   (L2-footprint-bound floor; kernel-launch-retune memory);
    #   unmatched voxel gather          the RTK/ASTRA/TIGRE standard, 35x cheaper -- and -0.5 dB
    #                                   in the A/B (B A nonsymmetric; unmatched-cg memory);
    #   our SF matched pair             matched to ~1e-6 and the only one with a geometry
    #                                   gradient at the time -- now gates-only (triton_sf);
    #                                   LEAP's own SF kernel is what the Joseph pin excludes.
    # Every production VALUE operator is LEAP's. The two DERIVATIVE kernels are ours -- the
    # estimator's d/dP (triton_leap_grad.leap_grad_P) and the bridge's ds-tangent
    # (leap_fdk_backproject_tangent) -- but both are EXACT derivatives of LEAP's own kernels
    # (LEAP ships none), so there is no second model anywhere. `reference_project_3d_batched`
    # is the gates' independent pair; nothing in production can reach it.
    # ---- ADMM-TV (the STANDARD form; see admm_dc_step) -----------------------------------
    # NOTE rho is NOT cg_lam: rho multiplies D^T D, cg_lam multiplies I. Different operators.
    # Only lam/rho sets the soft threshold; rho alone conditions the CG system.
    # The two INDEPENDENT knobs are rho and the threshold lam/rho -- lam never appears alone, so
    # it is not exposed. MEASURED on this geometry (power iteration, 2026-07-26):
    #     ||A^T A|| = 5.04e5      ||D^T D|| = 11.69  (= the 3D forward-difference bound of 12)
    # so the two blocks of A^T A + rho D^T D are BALANCED at rho ~ 5.04e5/11.69 ~ 4.3e4. That is
    # the default. (An earlier default of 1e-3 was 2e-9 x ||A^T A||: the TV block would have been
    # invisible in the CG operator -- exactly the kind of scale slip that looks like "ADMM does
    # nothing" rather than like a bug.)
    ap.add_argument("--admm_rho", type=float, default=4.3e4,
                    help="augmented-Lagrangian penalty on ||Dz - d||^2; conditions the CG "
                         "operator A^T A + rho D^T D. Default balances the two blocks on THIS "
                         "geometry (||A^T A||=5.04e5, ||D^T D||=11.7).")
    ap.add_argument("--admm_thresh", type=float, default=3e-4,
                    help="soft threshold lam/rho, IN MU UNITS (our volumes are [0,0.06], not "
                         "[0,1]: DDS's published 4e-3 is ~7%% of our whole dynamic range and "
                         "would flatten it). 3e-4 = 0.5%% of the range.")
    ap.add_argument("--admm_dual", type=int, default=1, choices=[0, 1],
                    help="1 = ADMM. 0 = freeze the dual at zero, i.e. HALF-QUADRATIC SPLITTING "
                         "(the free ablation; HQS needs a rising rho we do not schedule).")
    # ---- ASD-POCS adaptive TV coupling (Sidky & Pan; TIGRE ASD_POCS.m) -------------------
    # The TV step is NOT an independent constant there: it is tied to how much the DATA step just
    # moved the image. dtvg = asd_alpha * dp on the first step, and whenever the TV-induced change
    # dg exceeds asd_rmax * dp the TV step is shrunk by asd_red. That is exactly the alpha-vs-kappa
    # balance we were hand-tuning (too much TV -> blur, too much data -> oscillation).
    ap.add_argument("--asd", action="store_true",
                    help="use ASD-POCS adaptive TV coupling instead of the fixed kappa blend")
    ap.add_argument("--asd_alpha", type=float, default=0.2,   # Sidky & Pan's classic value
                    help="dtvg = asd_alpha * dp at the first step")
    ap.add_argument("--asd_rmax", type=float, default=0.95,
                    help="shrink the TV step when dg > asd_rmax * dp")
    ap.add_argument("--asd_red", type=float, default=0.95, help="TV step reduction factor")
    ap.add_argument("--asd_ng", type=int, default=0,
                    help="TV gradient-descent iterations per step; 0 = use --tv_iters")
    ap.add_argument("--no_prior", action="store_true",
                    help="BASELINE: skip the FM prior step (x_prior = x), leaving estimator + "
                         "data step + TV = classical blind joint motion estimation with "
                         "iterative reconstruction. This is the comparison baseline, not an "
                         "accounting of the prior's step size. The ODE schedule still advances, "
                         "so t-dependent knobs (fdk_eta ramp, alpha_p) behave as in the real "
                         "loop.")
    # ---- readout -------------------------------------------------------------------------
    ap.add_argument("--theta_oracle", action="store_true",
                    help="skip motion estimation entirely and use theta_true every step. NOT a "
                         "method -- it is the ceiling the carried x_t could reach if the "
                         "estimator were perfect, and the only way to price theta's error on the "
                         "MAIN deliverable (exp_theta_transfer.py prices it on FDK(theta_hat) "
                         "only). Also ~PER iterations/step cheaper, so it runs faster.")
    ap.add_argument("--theta_avg", type=int, default=1,
                    help="output FDK(mean of the last K thetas) instead of FDK(theta_final). "
                         "K=2 was the 2D project's winning readout (period-2 landing lottery; "
                         "measured 9/10 tail alternation on cg/val0 here too) -- but the DEFAULT "
                         "IS 1 (off): the user excluded the averaged readout from the current "
                         "sweeps (2026-07-24) so every A/B grades the raw last-step theta. "
                         "theta_hist is saved in result.pt regardless, so any K can be evaluated "
                         "OFFLINE (scripts/cmp_cg_three.py) without re-running.")
    # P3 of the 2026-07-25 acceleration pass: default 1 -> 5. The metric block (full FDK +
    # four warm-started gauge fits + montage) measured ~17 s on a clean GPU -- pure readout:
    # nothing downstream consumes it, the final step is ALWAYS evaluated, and the montage
    # cadence stays plenty for the by-eye judgement.
    ap.add_argument("--metric_every", type=int, default=5,
                    help="metric/montage (or snapshot, see --metric_mode) every k steps "
                         "(the last step always)")
    # ---- DEFERRED READOUT (user, 2026-07-25): the metric block is ~17 s of GPU work per hit
    # (FDK + 4 x 150-iter gauge fits + PNG) that the LOOP never consumes -- so in `defer` mode
    # the loop only drops a snapshot (theta + fp16 x_t, ~0.2 s, atomic rename) and keeps going
    # at pure-inference speed; scripts/render_posterior3d.py turns the snapshots into the same
    # metrics + montages later (or concurrently -- run it with --watch on the OTHER GPU, since
    # rigid_align at 256^3 is a GPU optimization; "render on CPU" is minutes per gauge fit and
    # is supported but only as a last resort). motion_error (rot/obs vs truth) is CHEAP, so the
    # convergence diagnostic stays inline and on the console either way. The END-of-run FINAL
    # evaluation always stays inline: result.pt must be complete for the cmp_* scripts.
    ap.add_argument("--metric_mode", default="defer", choices=["defer", "inline"],
                    help="defer (DEFAULT) = snapshot per metric step, render later/elsewhere; "
                         "inline = the old behaviour (metrics + montage inside the loop)")
    ap.add_argument("--n_ctrl", type=int, default=30,
                    help="control points for --estimator basis. 30 = Thies' estimation model "
                         "(30 spline nodes x 6 dof = 180 dof). MEASURED optimum on the akima55 "
                         "bench: none 7.93 deg (direct) > 60 ctrl 1.19 > 30 ctrl 0.165 > "
                         "10 ctrl 1.90 -- his choice reproduces as the sweet spot")
    ap.add_argument("--est_coarse", type=int, default=2,
                    help="estimate motion on a 1/N grid (Thies uses 128^3 @ 2 mm and reconstructs "
                         "at 256^3 @ 1 mm). Volume avg-pooled N, panel binned N, ray samples //N: "
                         "~N^3 cheaper per iteration, which is what makes a large --per affordable")
    ap.add_argument("--passes", type=int, default=1,
                    help="re-integrate the whole ODE this many times, resetting the image to the "
                         "cold FDK each time but CARRYING THE ESTIMATOR OVER. The loop is "
                         "path-dependent: theta and x_t bootstrap each other, so the early steps "
                         "bake a badly-wrong geometry into x_t that later steps cannot undo. A "
                         "second pass re-runs those early steps with the theta we ended up with")
    # c2f IS FREE SPEED, measured 3v3 blind on the 500k prior (2026-08-03), x_t aligned vs GT:
    #     fine the whole way (--est_coarse 1)   1321 s   38.52 dB / SSIM 0.9858 / rot 0.122 deg
    #     c2f (this default)                     948 s   38.53 dB / SSIM 0.9869 / rot 0.100 deg
    # So running fine throughout costs +39% wall clock and buys NOTHING -- c2f is in fact ahead
    # on SSIM (3/3 patients, though by less than the 0.0018 rerun bar) and on rot. The reason is
    # visible in the step profile: c2f's saving is entirely in steps 0-24 (10.7 vs 25.3 s/step),
    # which is exactly the IMAGE-LIMITED stretch where theta falls 2.5 -> 0.16 deg and where this
    # file's own `per_at` note measured every estimator config plateauing at ~2 deg regardless of
    # lr/loss/bandwidth. 1 mm precision cannot be cashed in against a reference that bad; the
    # steps where it can are already fine. Coarse ALL the way (--est_coarse_until 1.0) is another
    # 2x cheaper again at equal x_t but 1.7x worse rot -- available, not deployed.
    ap.add_argument("--est_coarse_until", type=float, default=0.5,
                    help="ODE time at which --est_coarse switches back to the full grid. 1.0 = "
                         "coarse the whole way. TRUE COARSE-TO-FINE: the coarse grid converges "
                         "faster early but plateaus higher (~0.30 deg vs the fine run's 0.176), "
                         "so buy the descent cheaply and the endpoint precisely")
    # `ramp` IS REFUTED (2026-08-03, blind val0, c2f, PER 200): x_t 39.30 -> 36.76 dB and rot
    # 0.16 -> 0.32 deg against `const` at the same iteration count. Its premise -- that early
    # iterations are wasted because the cold reference is image-limited -- ignores that the
    # estimator WARM-STARTS across ODE steps and the loop is PATH-DEPENDENT, so the early fit is
    # what the whole trajectory is built on. It also costs MORE WALL CLOCK, not the same: "SAME
    # TOTAL" is true of ITERATIONS, but ramp moves them onto the t>0.5 steps, which run on the
    # fine grid at ~2.5x the price per iteration (557 s const -> 649 s ramp). Keep `const`.
    # If the tail ever does need trimming, the schedule to write is the OPPOSITE one (decay:
    # big early, small late) -- PER 400 plateaus at ~0.16 deg by step 20 and then spends ~60% of
    # the runtime buying 0.16 -> 0.08 deg, which PER 200's result says x_t does not cash in.
    ap.add_argument("--per_sched", default="const", choices=["const", "ramp"],
                    help="how --per is spent across the ODE. `const` is DEPLOYED. `ramp` is "
                         "linear in t at the same ITERATION count (2*per*(k+0.5)/N), back-loading "
                         "the work onto the late steps -- MEASURED WORSE on both axes, see above; "
                         "kept only so the refutation stays reproducible")
    ap.add_argument("--context", default="auto", choices=["auto", "global", "none"],
                    help="auto reads in_ch off the checkpoint's in_conv weight")
    ap.add_argument("--blend", default="uniform", choices=["uniform", "hann"],
                    help="patch->volume scheme. uniform = archive paper's non-overlapping random "
                         "tilings (arXiv:2512.18161), needs --patch_offsets>=2; the deployed "
                         "default (A/B: = hann quality, 2.5x cheaper, data/blend_cmp/). hann = "
                         "overlapping stride patch/2 + Hann window.")
    ap.add_argument("--patch_offsets", type=int, default=2,
                    help="tile grids blended per ODE step = the archive paper's K (K=2 optimal). "
                         "For blend=uniform this MUST be >=2 (a single non-overlapping pass leaves "
                         "seams, -0.4 dB); for blend=hann, >1 adds randomly SHIFTED grids.")
    # ---- prior evaluation cost knobs (P2 of the 2026-07-25 acceleration pass) -----------
    # The deploy config tiles 256^3 into 1024 32^3 patches per step; at the library's old
    # batch=8 that is 128 UNet launches of a net far too small to fill the GPU. Raising the
    # batch changes NOTHING numerically (GroupNorm statistics are per-sample) -- it only fills
    # the device. 64 saturates it here; measured on the deploy loop, fm_predict 13.4 -> ~4 s.
    ap.add_argument("--prior_batch", type=int, default=64,
                    help="tiles per UNet forward inside predict_x1_patched (numerically "
                         "identical at any value; GroupNorm is per-sample)")
    # ON by default (user, 2026-07-25), and in fp16, because that is TRAIN-MATCHING: the 500k
    # prior trained under fp16 autocast (ckpt args amp=True, amp_dtype=float16), so a fp16
    # forward is the regime the weights saw for 500k iterations, while the fp32 inference we ran
    # until now was itself the (harmless) mismatch. Perturbs v at rel ~5e-3 vs fp32 runs -- the
    # pre-2026-07-25 baselines (fair_cg, night_*) were fp32, so pass --no-prior_amp for a
    # bit-comparable rerun of those.
    ap.add_argument("--prior_amp", action="store_true", default=True,
                    help="run the prior UNet forward in fp16 autocast, as it was TRAINED "
                         "(blend stays fp32). --no-prior_amp for the fp32 forward the "
                         "pre-2026-07-25 baselines used.")
    ap.add_argument("--no-prior_amp", dest="prior_amp", action="store_false")
    # Same knob, same default and the same rationale as train_fm3d's --compile: the prior net is
    # 24% of an ODE step and torch.compile fuses its many small 3D-conv kernels. It compiles the
    # UNDERLYING module (the ckpt is loaded first), so nothing about the weights or the blend
    # changes -- only kernel scheduling. Measured 1.34x on top of --prior_amp.
    ap.add_argument("--compile", action="store_true", default=True,
                    help="torch.compile the prior net (~1.34x on the net forward, ~8%% of the "
                         "run). One graph, compiled once. --no-compile to disable.")
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    args = ap.parse_args()
    if args.blend == "uniform" and args.patch_offsets < 2:
        raise SystemExit("--blend uniform needs --patch_offsets >= 2 (K=1 leaves tile seams; "
                         "see data/blend_cmp/). Use K=2, or --blend hann for a single pass.")

    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)

    # SEED THE GLOBAL RNG, not just the two local generators. `--seed` used to reach only the
    # motion draw (`make_motion`) and the tile jitter (`gtile`) -- while the motion ESTIMATOR draws
    # from the global RNG twice, for its network init and for the `views_per_iter` random view
    # subset it takes EVERY iteration (fm3d/motion_estimation.py). So two runs of the same command
    # followed different estimator trajectories and were not comparable, which is fatal for an A/B
    # whose whole output is one number per run.
    #
    # The size of that noise, measured from the 5-way data-step comparison: step 0's theta is
    # fitted BEFORE the data step ever runs, so all five runs should report an IDENTICAL rot there
    # -- they reported 1.51 / 1.61 / 1.78 / 1.86 / 1.99 deg. Half a degree of pure RNG, on the
    # axis those runs were being ranked by.
    #
    # Still not bit-exact: the Triton kernels and cuDNN use atomics, so reruns differ in the last
    # digits. This removes the TRAJECTORY divergence, which is the part that was worth degrees.
    torch.manual_seed(args.seed)

    world = build_world(ckpt=args.ckpt, dev=dev, root=args.root, split=args.split,
                        data=args.data, run=args.run, z0=args.z0,
                        motion_kind=args.motion_kind, seed=args.seed,
                        trans_mm=args.trans_mm, rot_deg=args.rot_deg)
    ck, ca, cfg, gen = world["ck"], world["ca"], world["cfg"], world["gen"]

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
          f"blend={args.blend} x {args.patch_offsets} tile grid(s)/step")

    model = UNet3D(in_ch=in_ch, base=ca["base"]).to(dev)
    model.load_state_dict(ck["ema"])
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    # torch.compile, for the SAME reason the trainer does it (train_fm3d --compile): the prior is
    # a stack of many small 3D convs and the win is kernel fusion, not precision. MEASURED here,
    # 64 tiles of 32^3, in_ch=5 base=32, A6000:
    #     fp32 eager 369.3 ms | fp16 autocast 230.7 ms (the deployed --prior_amp) | +compile 172.3 ms
    # i.e. 1.34x on top of amp, which is ~24% of an ODE step (1024 tiles / 64 = 16 net calls per
    # step at the deployed uniform K=2). The tile SHAPE is constant for a whole run, so exactly
    # one graph is compiled and the ~1 min warmup is paid once. channels_last_3d was measured and
    # is SLOWER here (287 ms) -- do not add it.
    if args.compile:
        model = torch.compile(model)
        print("torch.compile: ON for the prior (~1.34x on the net forward). --no-compile to disable.")

    spacing, meas = world["spacing"], world["meas"]
    gt3, theta_true = world["gt3"], world["theta_true"]
    y, static_fdk = world["y"], world["static_fdk"]

    est_kw = dict(dx=gen.dx, dy=gen.dy, dz=gen.dz, loss=args.loss, lncc_win=args.lncc_win,
                  views_per_iter=args.views_per_iter)
    # bandwidth and its matched lr travel together -- see --est_band. Only `net` has an encoder;
    # direct/basis have no bandwidth to set, so they keep their own defaults.
    if args.estimator.lower() == "basis":
        est_kw["n_ctrl"] = args.n_ctrl
    if args.estimator.lower() in ("net", "mlp", "hashbl"):
        band = dict(fullband=dict(n_levels=16, base_resolution=16, per_level_scale=1.5),
                    hashbl=dict(n_levels=4, base_resolution=2, per_level_scale=2.0))[args.est_band]
        est_kw.update(band)
        # fullband's 3e-3 replaced 1e-3 on 2026-07-27: the akima55 bench put it at rot 0.093 deg
        # against 1e-3's 0.159 at equal cost, and it transferred in-loop (x_t 34.25 -> 35.85 dB,
        # rot 0.456 -> 0.259, ALL FOUR metric cells improved). hashbl keeps its own pairing.
        est_kw["lr"] = {"fullband": 3e-3, "hashbl": 1e-2}[args.est_band]
    # THE MATCHED lr IS TUNED FOR L2 (and equivalently for l2si: their gradients differ by a
    # measured factor of 1.00 +- 0.01 in norm, and Adam normalises per parameter anyway).
    # lncc/ncc/ramp are a different story -- their gradient MAGNITUDES differ by orders, so this
    # lr does not transfer: measured 2026-07-25, `--loss lncc --est_band fullband` left theta
    # FROZEN at its init (rot 2.08 deg for all 50 steps, fit stuck at ~0.71) -- lncc's gradient is
    # too small for 1e-3 to move the estimator. Changing to one of THOSE losses without re-tuning
    # lr is a confound, not an ablation; pass an explicit --lr with them.
    if args.lr is not None:                    # explicit --lr overrides the matched pair
        est_kw["lr"] = args.lr
    print(f"estimator: {args.estimator} band={args.est_band} lr={est_kw.get('lr')} "
          f"views/iter={args.views_per_iter} loss={args.loss}")
    # ---- the grid the ESTIMATOR works on, which need not be the reconstruction grid ------------
    # Thies fits motion on 128^3 @ 2 mm and reconstructs at 256^3 @ 1 mm. A rigid 6-DoF trajectory
    # has no high-frequency content to lose, but the per-iteration cost falls ~N^3 (volume pooled
    # N, panel binned N so N^2 fewer rays, ray sampling decimated N). Since the estimator is only
    # ~15% of a step (49.0 s blind vs 41.7 s with --theta_oracle => 7.3 s per PER=50) yet theta is
    # worth 6-10 dB on x_t, cheap iterations are exactly the currency this loop wants.
    # Geometry of the coarse path is gated by scripts/gate_coarse_est.py.
    def build_grid(n):
        """Everything the estimator needs to work on a 1/n grid."""
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
              f"panel {grids[ec]['cfg'].nv}x{grids[ec]['cfg'].nu}, "
              f"FP/FPV derived from the geometry"
              + (f", switching to 1/1 at t={args.est_coarse_until:g}"
                 if args.est_coarse_until < 1.0 else ""))
    g0 = grids[ec]
    est_kw.update(dx=g0["vox"], dy=g0["vox"], dz=g0["vox"])
    est = make_estimator(args.estimator, g0["cfg"], gen.P_nom, g0["u"], g0["v"], dev, **est_kw)

    def use_grid(t):
        """Pick the estimator's grid for ODE time t and RE-POINT the estimator at it.

        The estimator is a coordinate net over the VIEW INDEX -- it never sees the volume grid --
        so switching resolutions mid-ODE carries its weights and its Adam moments across intact.
        Nothing is re-fitted; only the images and rays it is scored against change. That is what
        makes true coarse-to-fine free here, unlike in Thies' setting where the grid is fixed for
        the whole optimization.

        MEASURED (val 0, 2026-07-27): the coarse grid converges FASTER early and plateaus HIGHER
        -- c2per400 beat per200 at every step to t=0.5 (0.36/0.29/0.34/0.30 deg at steps
        10/15/20/25 against 0.61/0.48/0.44/0.35) at 78% of its cost, but flattened around 0.30
        while the fine run kept descending to 0.176. So coarse buys the descent and fine buys the
        endpoint."""
        gg = grids[ec] if (ec > 1 and t < args.est_coarse_until) else grids[1]
        est.cfg, est.u, est.v = gg["cfg"], gg["u"], gg["v"]
        est.dx = est.dy = est.dz = gg["vox"]
        return gg

    def est_ref(img, n):
        """The reference image handed to the estimator, on ITS grid."""
        return img if n == 1 else F.avg_pool3d(img[None, None], n)[0, 0]

    def per_at(k):
        """Estimator iterations for ODE step k. `ramp` keeps the SAME TOTAL as `const` (the sum
        of 2*P*(k+0.5)/N over k is P*N) but spends it where it buys accuracy.

        WHY A SCHEDULE AT ALL. Measured 2026-07-26 with `exp_est_sweep.py`: given a COLD reference
        (the uncorrected FDK the loop starts from) every estimator config plateaus at ~2.0 deg and
        the whole config ranking collapses (1.99-2.12 across lr, loss, bandwidth, views), while on
        a GT reference the same budget reaches 0.087-0.265. The early loop is therefore
        IMAGE-limited, not estimator-limited -- iterations spent at t~0 are spent against a
        ceiling that the image, not the optimizer, sets. The late loop is the opposite."""
        if args.per_sched == "const":
            return args.per
        return max(1, int(round(2.0 * args.per * (k + 0.5) / args.n_steps)))

    with torch.no_grad():
        x = gen.fdk(y, gen.P_nom[None])[0]                               # cold start: uncorrected
    patch = ca["patch"]

    # EVERY volume in a montage is shown in the GT's frame. The blind problem has an exact SE(3)
    # gauge, so a reconstruction sits at an arbitrary pose (here ~3 mm / ~2 deg): at 1 mm voxels
    # that puts its z = D//2 slice several slices and a rotation away from the GT's, i.e. the raw
    # panels would compare DIFFERENT ANATOMICAL PLANES under an aligned-metric title. The input
    # FDK never changes, so its gauge is fitted once here and reused.
    m_cold, x0_input = aligned_metrics(x, gt3, spacing, mask=meas, iters=200, return_aligned=True)
    ms_cold = aligned_metrics(x, static_fdk, spacing, mask=meas, iters=200)
    print(f"cold start  " + json.dumps({k: round(v, 4) for k, v in m_cold.items()}))

    # The static FDK is fixed, so its vs-GT score is fitted ONCE. It stays the reference for the
    # vs-sFDK metric column (Thies' protocol), and aligning IT to the GT keeps every panel in the
    # GT frame; the vs-sFDK metrics below then align each recon to this same volume.
    m_sfdk = aligned_metrics(static_fdk, gt3, spacing, mask=meas, iters=200)
    def tv_l1(v):
        """The anisotropic TV seminorm ||D v||_1. `grad_forward_3d` returns the three
        one-voxel-shorter difference tensors, NOT a stacked one."""
        return sum(float(g.abs().sum()) for g in grad_forward_3d(v[None, None]))

    with torch.no_grad():
        tv_gt = tv_l1(gt3)                                              # tv_rel's normalizer

    # THE PANEL, though, is FDK(theta_TRUE): the same measured data reconstructed with the true
    # motion. That is the ceiling a motion-corrected FDK can actually REACH, whereas the static
    # FDK is a different scan entirely. It is computed once and costs one FDK.
    with torch.no_grad():
        ceil_fdk = gen.fdk(y, params_to_Pmot(theta_true, gen.P_nom)[None])[0]
    m_ceil, ceil_aligned = aligned_metrics(ceil_fdk, gt3, spacing, mask=meas, iters=200,
                                           return_aligned=True)
    print(f"static FDK (vs GT, Thies' reference)  "
          f"{m_sfdk['psnr_aligned']:.2f} dB / SSIM {m_sfdk['ssim_aligned']:.3f}")
    print(f"FDK(theta_true) = REACHABLE FDK CEILING (vs GT)  "
          f"{m_ceil['psnr_aligned']:.2f} dB / SSIM {m_ceil['ssim_aligned']:.3f}")

    snap_dir = None
    if args.metric_mode == "defer":
        snap_dir = os.path.join(args.out, "snaps")
        os.makedirs(snap_dir, exist_ok=True)
        # everything the renderer needs to rebuild the world (build_world) + label the montages
        torch.save({"args": {**vars(args), "amp_units": AMP_UNITS}},
                   os.path.join(snap_dir, "meta.pt"))
        print(f"metric_mode=defer: snapshots -> {snap_dir}/  "
              f"(render: python scripts/render_posterior3d.py --out {args.out} [--watch])")

    hist = []
    theta_hist = []                        # every step's theta, for the averaged readout
    gauge_th = None                                              # warm-start for the gauge fit
    xt_gauge = None                        # x_t carries its OWN gauge; fitted separately
    gtile = torch.Generator(device=dev).manual_seed(args.seed)   # reproducible tile jitter
    dtvg = None                            # ASD-POCS TV step; set from dp on the first step
    # ADMM's split variable d and scaled dual u, SHARED ACROSS OUTER STEPS (the "variable
    # sharing" of DiffusionMBIR / DDS): one sweep per ODE step only makes sense if the dual
    # accumulates. Lazily allocated inside admm_dc_step. NB our A_theta is NON-STATIONARY --
    # theta is refit every step -- so u accumulates residuals against a drifting operator, a
    # situation no source addresses. If it misbehaves, damp or reset u once theta has settled.
    admm_state = {}
    ng = args.asd_ng or args.tv_iters
    N = args.n_steps
    x_cold = x.clone()                     # the uncorrected FDK -- where every pass starts
    for k in range(N * args.passes):
        t_wall = time.time()
        kk = k % N
        t = kk / N
        dt = 1.0 / N

        # ---- OUTER PASS RESTART -------------------------------------------------------------
        # WHY A SECOND PASS AT ALL. theta and x_t bootstrap each other, so the EARLY steps of a
        # blind run apply a badly-wrong geometry to the data-consistency step and bake that error
        # into x_t; later steps refine theta but cannot fully undo what was baked in. The evidence
        # is that the loop is PATH-dependent, not theta-limited, by the end: per400 reaches rot
        # 0.125 deg yet its x_t is 38.16 dB, while --theta_oracle -- which starts from the SAME
        # cold FDK and differs only in having the right geometry from step 0 -- reaches 40.45.
        # A second pass reproduces the oracle's condition with the theta we actually have: reset
        # the image to the cold FDK, keep the estimator (net weights AND Adam moments, hence
        # theta), and re-integrate the ODE.
        if kk == 0 and k > 0:
            x = x_cold.clone()
            admm_state.clear()
            dtvg = None
            gtile = torch.Generator(device=dev).manual_seed(args.seed + k)
            print(f"---- pass {k // N + 1}/{args.passes}: ODE restarts from the cold FDK, "
                  f"estimator carries over (rot {motion_error(est.current_params(), theta_true, cfg=cfg)['rot_rmse_deg']:.3f} deg)",
                  flush=True)

        # 1. PREDICT -- the prior moves first (patch-blended; see fm_predict)
        # --no_prior turns this into the identity, leaving estimator + data step + TV: the
        # CLASSICAL BASELINE (blind joint motion estimation + iterative reconstruction) that a
        # reader will ask to see this method beaten against. It is a baseline, NOT a test of
        # whether the prior "matters" -- note in particular that with the prior off the ESTIMATOR
        # is handed a different image every step, and the Gauss-Seidel design ("fit the motion on
        # the improved image") is precisely what is being removed. Read the gap as
        # method-vs-baseline, not as an accounting of who moved x_t further.
        x_prior = x if args.no_prior else fm_predict(
            model, gen, x, t, dt, patch, context=args.context,
            n_offsets=args.patch_offsets, generator=gtile, blend=args.blend,
            batch=args.prior_batch, amp=args.prior_amp)

        # 2. ESTIMATE on the improved image (Gauss-Seidel, not simultaneous)
        if args.theta_oracle:
            # No estimation at all: the data step and the prior see the TRUE geometry. This is
            # the CEILING OF THE MAIN DELIVERABLE with respect to motion -- the gap between it and
            # a blind run is the total price x_t pays for theta being estimated, which is the one
            # thing `exp_theta_transfer.py` cannot measure (that script only re-runs FDK).
            loss = float("nan")
            theta = theta_true
        else:
            gg = use_grid(t)
            loss = est.refine_global(est_ref(x_prior, gg["n"]), gg["y"], iters=per_at(k))
            theta = est.current_params()
        theta_hist.append(theta.detach().clone())

        # 3. CORRECT -- PnP forward-backward, carrying z (a faithful PnP: the denoiser is TV,
        #    which cannot go out of distribution the way the learned denoiser does)
        a = args.alpha * ((1 - t) ** args.alpha_p if args.alpha_p > 0 else 1.0)
        z = x_prior
        z_post_data = None
        for _ in range(args.pnp_k):
            dc_v = None
            if 0 < args.dc_views < cfg.n_views:
                dc_v = torch.randperm(cfg.n_views, device=dev)[:args.dc_views]
            # ---- DATA STEP ------------------------------------------------------------
            z_pre = z
            if args.dc_op == "fdk":
                eta = args.fdk_eta_min + (args.fdk_eta - args.fdk_eta_min) * (t ** args.fdk_eta_p)
                z = fdk_dc_step(z, theta, y[0], gen, eta, meas, views=dc_v)
            elif args.dc_op == "cg":
                z = cg_dc_step(z, theta, y[0], gen, iters=args.cg_iters, lam=args.cg_lam,
                               views=dc_v)
            elif args.dc_op == "admm":
                z = admm_dc_step(z, theta, y[0], gen, admm_state, rho=args.admm_rho,
                                 thresh=args.admm_thresh, iters=args.cg_iters, views=dc_v,
                                 dual=bool(args.admm_dual))
            else:
                g = data_grad(z, theta, y[0], gen, views=dc_v)
                z = z - a * z.norm() * g / g.norm().clamp_min(1e-12)
            dp = float((z - z_pre).norm())          # ASD-POCS's dp: how far the DATA step moved us
            z_post_data = z                          # for the prior-vs-data budget below
            # ---- PRIOR (TV) STEP ------------------------------------------------------
            if args.asd:
                # Sidky & Pan: the TV step size is SLAVED to dp, not an independent constant.
                if dtvg is None:
                    dtvg = args.asd_alpha * dp
                z_tv = z
                for _ in range(ng):
                    gtv = sidky_dtv_grad_3d(z_tv[None, None])[0, 0]
                    z_tv = z_tv - dtvg * gtv / gtv.norm().clamp_min(1e-12)
                dg = float((z_tv - z).norm())
                z = z_tv
                if dg > args.asd_rmax * dp:
                    dtvg *= args.asd_red                     # TV outran the data step -> shrink
            elif args.kappa > 0 and args.dc_op != "admm":
                # ADMM already CONTAINS the TV term (as an exact prox + dual), so running the
                # kappa-blend on top would apply TV twice with two different, uncoordinated
                # strengths. Skipped rather than erroring so `--kappa` keeps its meaning for
                # every other dc_op.
                z = z + args.kappa * (sidky_dtv_denoise_3d(
                    z[None, None], args.tv_iters, args.tv_step)[0, 0] - z)
        # Per-step movement budget: how far the FM prior moved the volume, how far the data step
        # then moved it, how far TV did. `cg` is not a small nudge but a SOLVE (every iteration
        # takes the optimal step along its residual direction), so dc/fm grows with --cg_iters.
        #
        # WHAT THIS RATIO IS FOR, AND WHAT IT IS NOT. It is an OSCILLATION detector. In 2D the
        # same diagnostic read 14:1 on the adjoint step while 84% of the accumulated path length
        # CANCELLED -- the pathology was the cancellation, and that is what motivated alpha
        # 0.1 -> 0.02. A large ratio on its own is NOT evidence that the prior contributes little:
        # the two steps act in different subspaces (the data step fixes what y determines given
        # theta_hat; the prior supplies the null space and the unmeasured region), and the prior's
        # main job here is not to move x_t at all -- it is to hand the ESTIMATOR a plausible image
        # to fit on (the Gauss-Seidel order). Norms cannot rank those. Compare net displacement
        # against accumulated path if you want the cancellation number.
        d_fm = float((x_prior - x).norm())
        d_tv = float((z - z_post_data).norm()) if z_post_data is not None else 0.0
        # HOW SMOOTH THE RESULT ACTUALLY IS, relative to the ground truth's own total variation.
        # `d_tv` cannot compare TV COUPLING FORMS: with --dc_op admm the TV prox happens INSIDE
        # the data step, so z_post_data is already post-TV and d_tv reads 0. This measures the
        # OUTCOME instead -- 1.0 = as much gradient energy as the truth, < 1 = over-smoothed --
        # which is defined identically for the kappa blend, ASD-POCS and ADMM, and is therefore
        # the only way to check that an operator comparison is not secretly a TV-strength one.
        tv_rel = tv_l1(z) / tv_gt
        ratio = dp / max(d_fm, 1e-12)
        x = z.detach()

        do_metric = k % max(args.metric_every, 1) == 0 or k == N - 1
        if do_metric and args.metric_mode == "defer":
            # snapshot only: theta is tiny, x_t is 32 MB in fp16. Write-then-rename so a
            # concurrently watching renderer never reads a half-written file. rot/obs are cheap
            # (no gauge fit involved), so the convergence readout stays on the console.
            me = motion_error(theta, theta_true, cfg=cfg)
            sec = time.time() - t_wall
            hist.append({"step": k, "t": t, "loss": loss, "d_fm": d_fm, "d_dc": dp,
                         "d_tv": d_tv, "tv_rel": tv_rel, "dc_over_fm": ratio, "sec": sec, **me})
            tmp = os.path.join(snap_dir, f".step{k:03d}.tmp")
            torch.save({"step": k, "t": t, "theta": theta.detach().cpu(),
                        "x_t": x.half().cpu(), "loss": loss}, tmp)
            os.replace(tmp, os.path.join(snap_dir, f"step{k:03d}.pt"))
            print(f"step {k:3d} t={t:.2f} | fit {loss:.5f} | rot {me['rot_rmse_deg']:.2f} deg, "
                  f"obs {me['trans_obs_mm']:.2f} mm | dc/fm {ratio:.1f}x | tv {tv_rel:.3f} "
                  f"| {sec:.1f}s (snap)",
                  flush=True)
        elif do_metric:
            with torch.no_grad():
                x_fdk = gen.fdk(y, params_to_Pmot(theta, gen.P_nom)[None])[0]
            # warm-start the gauge fit from the last step's theta: the gauge moves slowly
            m, gauge_th, x_fdk_al = aligned_metrics(x_fdk, gt3, spacing, mask=meas, iters=150,
                                                    init=gauge_th, return_theta=True,
                                                    return_aligned=True)
            # x_t sits at its own pose (it is not FDK(theta)), so it needs its own gauge fit
            # before it can be shown beside the GT. Warm-started, so it is a cheap refit.
            # Its metrics are LOGGED too: the carried state is what the motion estimator actually
            # reads (via one FM step), so the x_t <-> FDK(theta) gap is the bottleneck diagnostic.
            # The 2D sibling's deploy eval reads x_t 23.57 dB / SSIM 0.861 vs FBP 26.98 / 0.910
            # (a 3.4 dB lag), which is the number to compare this against.
            mx, xt_gauge, x_al = aligned_metrics(x, gt3, spacing, mask=meas, iters=150,
                                                 init=xt_gauge, return_theta=True,
                                                 return_aligned=True)
            # second reference: the same two reconstructions scored vs the STATIC FDK (Thies).
            ms_out = aligned_metrics(x_fdk, static_fdk, spacing, mask=meas, iters=150)
            ms_xt = aligned_metrics(x, static_fdk, spacing, mask=meas, iters=150)
            me = motion_error(theta, theta_true, cfg=cfg)   # cfg -> beam-frame split
            sec = time.time() - t_wall
            hist.append({"step": k, "t": t, "loss": loss, "d_fm": d_fm, "d_dc": dp, "d_tv": d_tv,
                         "dc_over_fm": ratio, "sec": sec, **m, **me,
                         **{f"xt_{q}": v for q, v in mx.items()},
                         **{f"s_{q}": v for q, v in ms_out.items()},
                         **{f"xts_{q}": v for q, v in ms_xt.items()}})
            print(f"step {k:3d} t={t:.2f} | fit {loss:.5f} | OUT vsGT "
                  f"{m['psnr_aligned']:5.2f}/{m['ssim_aligned']:.3f} vsSFDK "
                  f"{ms_out['psnr_aligned']:5.2f}/{ms_out['ssim_aligned']:.3f} "
                  f"| x_t vsGT {mx['psnr_aligned']:5.2f}/{mx['ssim_aligned']:.3f} vsSFDK "
                  f"{ms_xt['psnr_aligned']:5.2f}/{ms_xt['ssim_aligned']:.3f} "
                  f"| rot {me['rot_rmse_deg']:.2f} deg, obs {me['trans_obs_mm']:.2f} mm "
                  f"| dc/fm {ratio:.1f}x | {sec:.1f}s", flush=True)
            montage(os.path.join(args.out, f"step{k:03d}.png"), gt3, x0_input, x_fdk_al, x_al,
                    ceil_aligned, k, t,
                    f"dc_op={args.dc_op} kappa={args.kappa:g} est={args.est_band} | theta rot "
                    f"{me['rot_rmse_deg']:.2f} deg, trans_obs {me['trans_obs_mm']:.2f} mm",
                    m_in=m_cold, m_out=m, m_xt=mx,
                    ms_in=ms_cold, ms_out=ms_out, ms_xt=ms_xt, m_ceil=m_ceil)
        else:
            print(f"step {k:3d} t={t:.2f} | fit {loss:.5f} | {time.time() - t_wall:.1f}s",
                  flush=True)

    # ---- THETA-AVERAGED READOUT (the 2D project's winning technique, ported 2026-07-24) --------
    # The output is FDK(theta_bar), theta_bar = mean of the last K thetas -- NOT FDK(theta_final).
    #
    # WHY. The estimator's tail is a near-period-2 limit cycle, so the LAST step lands on a random
    # phase of that cycle and the deliverable is a lottery ticket. Measured on this project's own
    # runs (tail of 12 steps, counting sign flips of successive rot changes): cg on val 0 alternates
    # 9/10 -- and happened to land on its single best step of the whole run (0.149 deg). Averaging
    # cancels the cycle instead of gambling on it. In 2D this was worth +0.015 aligned SSIM
    # (K8-K1 t=+2.75, significant, n=10), K=2 captured nearly all of it, and it costs NOTHING:
    # the thetas are already in memory.
    #
    # K=2 is the default for exactly the period-2 reason. Set --theta_avg 1 to get the old
    # last-step behaviour back.
    #
    # CAVEAT, and it is why K stays small: theta[:, 3:] is AXIS-ANGLE, and the arithmetic mean of
    # axis-angle vectors is not the geodesic mean on SO(3). At this project's tail amplitudes
    # (< 1 deg spread) the two agree to well under the estimator's own noise, but a large K over a
    # still-DESCENDING trajectory would average in stale, worse thetas -- which is a real risk here:
    # val 1's rot was still falling monotonically at step 49 (alternation 1/10), the opposite
    # regime from val 0. Averaging is a variance fix, not a convergence fix.
    K = max(1, min(args.theta_avg, len(theta_hist)))
    # With --theta_oracle the estimator was never stepped, so its OWN parameters are still zero;
    # the trajectory the loop actually used is the one in theta_hist.
    theta_last = theta_hist[-1] if args.theta_oracle else est.current_params()
    theta = torch.stack(theta_hist[-K:], 0).mean(0) if K > 1 else theta_last
    if K > 1:
        me_last = motion_error(theta_last, theta_true, cfg=cfg)
        me_avg = motion_error(theta, theta_true, cfg=cfg)
        print(f"theta readout: last-step rot {me_last['rot_rmse_deg']:.3f} deg / obs "
              f"{me_last['trans_obs_mm']:.3f} mm  ->  mean of last K={K} rot "
              f"{me_avg['rot_rmse_deg']:.3f} / obs {me_avg['trans_obs_mm']:.3f}")
    with torch.no_grad():
        x_final = gen.fdk(y, params_to_Pmot(theta, gen.P_nom)[None])[0]     # output = FDK(theta)
    fm, x_final_al = aligned_metrics(x_final, gt3, spacing, mask=meas, iters=300, init=gauge_th,
                                     return_aligned=True)
    _, x_al = aligned_metrics(x, gt3, spacing, mask=meas, iters=300, init=xt_gauge,
                              return_aligned=True)
    mx_final = aligned_metrics(x, gt3, spacing, mask=meas, iters=300, init=xt_gauge)
    # both deliverables scored vs BOTH references (user, 2026-07-25: x_t AND FDK must both be good)
    fm_s = aligned_metrics(x_final, static_fdk, spacing, mask=meas, iters=300)
    mx_final_s = aligned_metrics(x, static_fdk, spacing, mask=meas, iters=300)
    print("\nFINAL OUTPUT  vs GT  " + json.dumps({k: round(v, 4) for k, v in fm.items()}))
    print("FINAL OUTPUT  vs sFDK " + json.dumps({k: round(v, 4) for k, v in fm_s.items()}))
    print("FINAL x_t     vs GT  " + json.dumps({k: round(v, 4) for k, v in mx_final.items()}))
    print("FINAL x_t     vs sFDK " + json.dumps({k: round(v, 4) for k, v in mx_final_s.items()}))
    # x_t is SAVED, not just scored. FDK(theta_hat) is rebuildable from `theta` alone, but the
    # carried PnP state is not reproducible without re-running the whole loop -- and with a
    # spectral data step it can OVERTAKE the nominal output (fdk run, 2026-07-24: x_t 34.51 dB /
    # 0.929 vs FDK(theta_hat) 31.53 / 0.791), which makes "which volume is the deliverable" a live
    # question that any later analysis has to be able to re-open. ~64 MB per volume, fp16 on disk.
    torch.save({"theta": theta.cpu(), "theta_last": theta_last.cpu(),
                "theta_hist": torch.stack(theta_hist, 0).cpu(),   # (N,V,6): re-do any K offline
                "theta_true": theta_true.cpu(), "hist": hist, "theta_avg_K": K,
                "final": fm, "final_xt": mx_final,
                "final_s": fm_s, "final_xt_s": mx_final_s,       # vs the static-FDK reference
                "sfdk_vs_gt": m_sfdk,                             # the operator ceiling
                "x_final": x_final.half().cpu(), "x_t": x.half().cpu()},
               os.path.join(args.out, "result.pt"))
    montage(os.path.join(args.out, "final.png"), gt3, x0_input, x_final_al, x_al, ceil_aligned,
            N, 1.0, f"FINAL | dc_op={args.dc_op} kappa={args.kappa:g} est={args.est_band} "
            f"N={N} PER={args.per} {args.loss} | val {args.run} seed {args.seed}",
            m_in=m_cold, m_out=fm, m_xt=mx_final,
            ms_in=ms_cold, ms_out=fm_s, ms_xt=mx_final_s, m_ceil=m_ceil)
    print(f"montages -> {args.out}/  (judge by eye: streak-free, not the number)")


if __name__ == "__main__":
    main()
