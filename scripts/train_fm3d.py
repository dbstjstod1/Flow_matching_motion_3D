"""Train the 3D flow-matching prior on a MOTION-DECAY BRIDGE.

TWO BRIDGES, and `--bridge` picks one. Both decay the motion linearly to zero, so both are paths
of genuine reconstructions rather than a pixel-space interpolation through images no operator
produces. They differ in WHERE the motion is attenuated:

    data (DEFAULT since 2026-08-05)   x_t = FDK( A(x; P_nom @ T((1-t)*theta)), P_nom )
    geom (the original)               x_t = FDK( y, P_nom @ T(t*theta) ) + t*Delta

The `geom` bridge holds the measurement fixed and improves the GEOMETRY, which is literally what
the inference loop does as theta_hat converges -- that was the argument for it. Its cost is that
FDK's analytic inverse assumes a CIRCULAR orbit and P(theta) is not one, so its bare endpoint is
NOT a clean image and has to be pulled onto the static FDK by a linear detrend `Delta` (see
`bridge_pair`). The `data` bridge attenuates the motion in the MEASUREMENT instead; the geometry
stays nominal for every t, so the endpoint IS the static FDK by construction -- no anchor, no
detrend, and no angular-weight derivative in the tangent.

MEASURED BEFORE SWITCHING (scripts/diag_bridge_ab.py, 3 val patients, train amplitude
15 mm / 20 deg p2p; RMS as % of the net range):

    t      |B - A_anch|   |B - A_bare|   |A_anch - A_bare|
    0.25       0.79           0.79             0.43
    0.50       1.05           1.14             0.85
    0.75       1.06           1.42             1.28

  * the two targets agree to ~1% of range (38-41 dB), so the switch is not a change of regime;
  * ||Delta|| is 1.16 / 1.63 / 2.33% of range per patient -- the anchor was correcting a ONE
    PERCENT endpoint error, not a large one;
  * `geom`+anchor is in fact CLOSER to the bare geometry manifold than `data` is, at every t and
    every patient (0.43 vs 0.79, 0.85 vs 1.14, 1.28 vs 1.42). So the switch TRADES 0.3%p of
    proximity-to-the-inference-trajectory for an endpoint that is clean by construction, a path
    with no hand-tuned detrend in it, and a draw that no longer computes FDK(y, P(theta)).
    It is not a free win; it is a small, deliberate, user-approved trade (2026-08-05).

THE TANGENT IS EXACT ON BOTH. `geom` differentiates LEAP's VD backprojection
(`leap_vd_backproject_tangent`, 2026-07-30); `data` differentiates LEAP's pinned Joseph FORWARD
(`leap_forward_tangent`, 2026-08-05 -- written for this switch, because the central difference it
started with bottoms out at ~2% of the target and the project does not train on that; the kernel
is gated against a float64 autograd jvp at rel 2.8e-6, cos 1.00000000).

COST, measured back to back on one GPU: draw 3.4 s for `data` against 2.1-2.4 s for a `geom`
whose static-anchor memo has warmed (3.4 s while it is still cold, i.e. they are identical early
in a run). The data draw is LEAP's own projection (1.14 s, the value, kept for the bit-exact
endpoints) + the tangent kernel (1.52 s) + two FDKs; `geom` is one projection + one fused
backprojection tangent + the memoized anchor. Amortized over --refresh 12 that is about
+0.1 s per training step.

`t * theta` is a geodesic because the rotation is an axis-angle vector (`rigid_motion.so3_exp`);
with Euler angles it would not be, and the velocity target would quietly stop pointing where the
inference ODE travels. This holds for both bridges.

MEMORY. The network only ever sees `--patch`^3 PATCHES (32^3 in the deployed run -- the archive
paper's rule is patch = volume/8, i.e. 32 at our 256^3); the operator (one fused FDK+tangent pass per
sample, under `no_grad`) runs on the full slab. That split is the whole design -- it is what lets a 3D prior
train on a 24 GB card. Bridge draws are expensive, so a rolling cache of volume pairs is refreshed
every `--refresh` steps and each batch mixes patches from several cached draws (so `t` varies
within a batch). Both tricks are from the 4DCT project. `t` itself is sampled UNIFORMLY
(`sample_t`), matching arXiv:2512.18161 and the bridge-model default.

THE DEFAULTS ARE THE DEPLOYED RUN (2026-07-31), exactly -- nothing here is a toy setting any more.
This one line reproduces the training recipe: CQ500 256^3 @ 1 mm, patch 32 / batch 64 (the archive
paper's volume/8 rule), cache 8 / refresh 12, 500k iters, cosine lr 1e-4 -> 1e-6, EMA 0.999,
fp16 AMP, the DATA bridge, native simulation grid, Thies training amplitudes:

    setsid nohup python scripts/train_fm3d.py --out logs/fm3d_next </dev/null >> log 2>&1 &
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # for val_fm3d

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig, measured_region_mask
from fm3d.prior_patch import make_tile_inputs, volume_context
from fm3d.rigid_motion import (AMP_UNITS, bridge_P_and_dP, params_to_Pmot,
                               random_motion)
from fm3d.unet_3d import UNet3D
from val_fm3d import run_validation



def sample_t(n: int, device) -> torch.Tensor:
    """t ~ U[0,1], uniform.

    This is the STANDARD for this model family and matches our own references: the UNet
    paper we follow (arXiv:2512.18161) samples t ~ Uniform with no timestep weighting, and
    the bridge models this prior belongs to (I2SB; InDI, Delbracio & Milanfar 2023
    arXiv:2303.11435; Direct Diffusion Bridges) all default to uniform t.

    An earlier version pulled 30% of draws into [0,0.15] to over-sample the cold-start (t=0)
    endpoint where the inference ODE begins. That endpoint bias is a *validated* restoration
    trick (InDI reports biasing toward the degraded endpoint beats uniform), and the smooth,
    citable way to do it would be logit-normal sampling with negative location (SD3 / Esser
    et al. 2024, arXiv:2403.03206). It was dropped here in favour of the reference-faithful
    uniform sampler; restore it via a logit-normal draw if the cold-start region needs more
    capacity.
    """
    return torch.rand(n, device=device)


@torch.no_grad()
def bridge_pair_data(gen, idx: int, t: torch.Tensor, theta, delta: float = 0.005,
                     mode: str = "analytic"):
    """(x_t, dx_t) in NET space for THE DATA BRIDGE -- `--bridge data`, the default.

        x_t = FDK( A(x; P_nom @ T((1-t)*theta)), P_nom )

    The motion decays in the MEASUREMENT, not in the reconstruction geometry. Consequences, all
    of which are the reason this is now the default (see the module docstring for the numbers):

      * t = 0 -> the motion is full -> x_0 = FDK(y_theta, P_nom), the inference cold start,
        EXACTLY (measured: 142 dB against the `geom` bridge's t=0, i.e. float noise).
      * t = 1 -> the motion is gone -> x_1 = FDK(A(x; P_nom), P_nom) = THE STATIC FDK, by
        construction and to the bit. No anchor, no `Delta`, nothing to detrend. The `geom`
        bridge needed one because FDK's analytic inverse assumes a circular orbit and
        P(theta) is not one; here the orbit is P_nom for every t, so that error is identically
        zero along the whole path.
      * the reconstruction geometry does not depend on t, so neither does the Voronoi angular
        weight -- the tangent carries no `view_weight_dot` term (contrast `fdk_tangent`).

    THE TANGENT. FDK is LINEAR in the sinogram and its geometry is t-independent, so d/dt
    commutes straight through it:

        dx_t/dt = FDK( dy_s/dt, P_nom ),   y_s = A(x; P(s*theta)),  s = 1 - t,  d/dt = -d/ds

    which leaves dy_s/ds, a derivative of the FORWARD projector with respect to the geometry.
    mode="analytic" (the DEFAULT) takes it EXACTLY, from `gen.simulate_tangent` ->
    `triton_leap_grad.leap_forward_tangent`: the s-derivative of LEAP's own pinned Joseph
    kernel, in one fused pass, the forward-projection twin of what `leap_vd_backproject_tangent`
    already did for the geometry bridge. So both bridges now regress on an exact tangent of the
    operator they actually run, and nothing in production is a finite difference.

    mode="fd" is the central difference this started as, kept as the gate counterparty. It is
    NOT good enough to train on, which is why the kernel exists -- measured in
    scripts/diag_bridge_data_tangent.py (s0 = 0.5, 15 mm / 20 deg p2p), against a Richardson
    reference, in the IMAGE domain that the loss actually sees:

        delta        0.05    0.02    0.01    0.005   0.002   0.001
        rel vs ref   0.120   0.022   0.030   0.047   0.061   0.074

    a U-curve that BOTTOMS OUT AT ~2%: truncation above (the projection through the 612^3
    native volume is genuinely curved in s), fp32 cancellation below, amplified on the way
    through the ramp filter -- the sinogram-domain optimum is delta = 0.005 but the image-domain
    one is 0.02, and neither gets under 2%. Two Richardson pairs one octave apart still
    disagree by 4%. It is NOT grid ripple: the ripple probe in that script shows
    ||y(s+e)-y(s0)||/e converging smoothly and symmetrically (ratio 1.0004 at e = 5e-4). The
    fd path clamps to one-sided at s = 0 or 1, exactly as `bridge_pair` does.
    """
    tv = float(t)
    s = 1.0 - tv
    if mode == "analytic":
        # bridge_P_and_dP gives P(s) and dP/ds in closed form; the value comes from `simulate`
        # (the vendored library) so the endpoints stay bit-exact, the derivative from the
        # transcribed kernel -- the same value/derivative split `fdk_conebeam_3d_tangent` makes.
        P_s, Pdot_s = bridge_P_and_dP(theta, gen.P_nom, s)
        x_t = gen.to_net(gen.fdk(gen.simulate(idx, P_s[None]), gen.P_nom[None])[0])
        _, dy_ds = gen.simulate_tangent(idx, P_s[None], Pdot_s[None])
    elif mode == "fd":
        def sim(sv: float):
            return gen.simulate(idx, params_to_Pmot(sv * theta, gen.P_nom)[None])

        x_t = gen.to_net(gen.fdk(sim(s), gen.P_nom[None])[0])
        sp, sm = min(s + delta, 1.0), max(s - delta, 0.0)
        dy_ds = (sim(sp) - sim(sm)) / (sp - sm)
    else:
        raise ValueError(f"unknown data-bridge tangent mode: {mode!r}")
    # d/dt = -d/ds, and the FDK is linear, so ONE backprojection of the sinogram derivative.
    dx = -gen.to_net_tangent(gen.fdk(dy_ds, gen.P_nom[None])[0])
    return x_t, dx


def bridge_pair(gen, t: torch.Tensor, y, theta, dlt, delta: float = 0.02,
                mode: str = "analytic", filtered=None):
    """(x_t, dx_t) in NET space, for one volume. t: scalar tensor. `--bridge geom`.

    THE ANCHORED GEOMETRY BRIDGE:

        x_t = FDK(y, P_nom @ T(t*theta))  +  t * Delta,
        Delta = x_anchor - FDK(y, P_nom @ T(theta))                  [computed once, in `draw`]

    WHY THE ANCHOR EXISTS. The bare geometry bridge's endpoint is NOT a clean image. Handing FDK
    the TRUE theta does not reproduce a static scan: FDK is an analytic inverse derived for a
    CIRCULAR, EQUIANGULAR orbit, and per-view motion breaks that. Measured on CQ500 with the
    literature's own motion model (Akima, 10 nodes, 5 mm / 5 deg), FDK(y, P(theta_true)) sits
    **1-3 dB below the static scan** even after `view_angular_weights` recovers the gantry-axis
    part. So a prior trained on the bare bridge learns FDK's residual motion artefact AS ITS
    TARGET -- it would faithfully reproduce, at t=1, an image that is not clean.

    The anchor is a first-order detrend that costs nothing and pins the endpoint:
      * at t=0 the Delta term vanishes -> x_0 = FDK(y, P_nom), the inference cold start, to the bit
      * at t=1 -> x_1 = x_anchor, the clean image. NOT bit-exact in the analytic mode: Delta is
        computed with the batched FDK kernel (`gen.fdk`, in `draw`) while x_t comes from the fused
        Triton tangent kernel (`gen.fdk_tangent`), so x_1 = x_anchor holds to the two kernels'
        gated agreement (scripts/gate_fdk_tangent.py), not by construction. mode="fd" restores
        the single-kernel, exact-cancellation path.
      * in between the geometry term still dominates, so the path stays the manifold of
        partially-corrected reconstructions -- which is what the inference loop actually walks as
        theta_hat converges. This IS measurably true (`geom`+anchor sits 0.43/0.85/1.28% of range
        off the bare geometry manifold at t = 0.25/0.5/0.75, against 0.79/1.14/1.42% for the data
        bridge) -- but the margin is 0.3%p, and the two bridges' images agree to ~1% of range
        overall, which is why `--bridge data` is now the default anyway. The old text here
        dismissed the data bridge as "images of a patient who moved LESS, which inference never
        sees"; that is the right intuition but the wrong magnitude, and it was never measured
        until scripts/diag_bridge_ab.py (2026-08-05).
      * Delta is constant in t, so the velocity target is just  d/dt FDK(y,P(t*theta)) + Delta.
        No extra reconstructions.

    This is the image-domain twin of Flowmatching-4DCT's `t1_anchor` detrend (which pins its
    sinogram bridge to a REAL static scan). `dlt=None` restores the bare bridge.

    dx_t (mode="analytic", the default) is the EXACT forward-mode derivative w.r.t. Pmat:
    one fused reconstruction pass (`fdk_conebeam_3d_tangent` -- filter once, then a Triton
    kernel that accumulates FDK and d/ds FDK from the same detector taps), replacing the old
    central difference (mode="fd"): three full FDKs per sample that agreed with the exact
    derivative only to ~1e-3. The old comment here guessed forward-mode would cost ~3x; with
    the tangent fused into the backprojection it costs ~0.5x. Gated against float64 central
    differences in scripts/gate_fdk_tangent.py; "fd" is kept as the gate's counterparty and
    an escape hatch.
    """
    tv = float(t)
    if mode == "analytic":
        P, Pdot = bridge_P_and_dP(theta, gen.P_nom, tv)
        # `filtered` = the draw's already-filtered y (`gen.fdk_filtered`), shared with the
        # anchor's FDK(y, P(theta)). The ramp does not depend on the geometry, so filtering
        # twice per draw was pure waste; None keeps the self-contained path.
        x_mu, dx_mu = gen.fdk_tangent(y, P[None], Pdot[None], filtered=filtered)
        x_t = gen.to_net(x_mu[0])                                    # (D,H,W)
        dx = gen.to_net_tangent(dx_mu[0])                            # affine gain, no shift
    elif mode == "fd":
        def fdk_at(s: float):
            P = params_to_Pmot(s * theta, gen.P_nom)[None]
            return gen.to_net(gen.fdk(y, P)[0])                      # (D,H,W)

        sp, sm = min(tv + delta, 1.0), max(tv - delta, 0.0)
        x_t = fdk_at(tv)
        dx = (fdk_at(sp) - fdk_at(sm)) / (sp - sm)
    else:
        raise ValueError(f"unknown bridge tangent mode: {mode!r}")
    if dlt is not None:
        x_t = x_t + tv * dlt
        dx = dx + dlt
    return x_t, dx


def main():
    ap = argparse.ArgumentParser()
    # `--dataset` is kept (single choice) because checkpoints carry it and
    # run_posterior3d.build_world dispatches on it. The aapm stacked-slice stand-in was REMOVED
    # 2026-08-07 (user's call: only Thies/CQ500 will ever be used); its dataset_slab.py went
    # with it -- see git history if a second dataset ever returns.
    ap.add_argument("--dataset", default="cq500", choices=["cq500"],
                    help="cq500 = the literature's dataset in the literature's geometry "
                         "(SID 785 / SDD 1200)")
    ap.add_argument("--root", default="data/CQ500")
    ap.add_argument("--split", default="train")
    ap.add_argument("--shape", type=int, nargs=3, default=(256, 256, 256))   # @ 1 mm
    ap.add_argument("--out", default="logs/fm3d_a")
    # ---- THE DEFAULTS BELOW ARE THE DEPLOYED RUN (2026-07-31, user's call) ------------------
    # Everything here used to default to a toy configuration nobody ran, so the real recipe lived
    # only in a shell command that had to be copied correctly. `python scripts/train_fm3d.py
    # --out <dir>` now reproduces the deployed training EXACTLY -- no exceptions.
    ap.add_argument("--iters", type=int, default=500000)
    # patch 32 / batch 64 = the archive paper's recipe (arXiv:2512.18161), whose invariant is a
    # DOWNSAMPLE FACTOR OF 8: patch = volume/8, so 32 at Thies' 256^3, and batch 64 comes with it
    # (their 512 config is patch 64x64x32 / batch 16). Not a GPU-occupancy choice.
    ap.add_argument("--batch", type=int, default=64)         # patches per step
    ap.add_argument("--patch", type=int, default=32,
                    help="patch edge [voxels]. 32 = the archive paper's volume/8 rule at our "
                         "256^3 grid. Guarded by --resume, so resuming a 64-patch checkpoint "
                         "without passing --patch 64 fails loudly rather than silently.")
    # BRIDGE DRAWS HELD AT ONCE. Raised 6 -> 32 on 2026-07-31, and the reason is the RATIO to
    # --batch, not the number itself. The archive paper's recipe (arXiv:2512.18161, our source
    # for patch 32 / batch 64) means 64 INDEPENDENT (volume, t) conditions per step -- free for
    # them, since their x_t is just noise added to a volume. Ours is a 1.2 s FDK, so a batch of
    # 64 is assembled from `cache` cached draws and t is welded to the draw: at cache 8 the batch
    # is 8 distinct t x 8 crops each, NOT 64 i.i.d. samples.
    #
    # MEASURED (2026-07-31, 512 per-sample gradients at iter 218k, one-way random-effects
    # decomposition; scratch rig kept out of tree): the cost of that is SMALL, because gradient
    # variance is dominated by WHERE the patch sits, not by t --
    #     sigma_w^2 (patch position) = 4.139   sigma_b^2 (t, patient, motion) = 0.291
    #     intraclass correlation rho = 0.0658  (95% CI [0.013, 0.143], p = 0.006)
    # so batch-gradient variance vs the i.i.d. ideal is 1.46x at cache 8 (effective batch 43.8,
    # NOT 8) and 1.07x at 32. It is pure VARIANCE, no bias -- the t marginal stays uniform over
    # the run -- so 500k steps + cosine lr -> 1e-6 + EMA 0.999 already pay for it. Expect NO
    # val improvement from this; it is recipe hygiene, not a quality lever.
    #
    # SO 8 STAYS (user, 2026-07-31): 1.46x is a VARIANCE cost, not a bias, and 500k steps buy it
    # back. Raising it to 32 (1.07x, VRAM 4.2 GB, zero compute) was considered and declined for
    # that reason -- record the option, not a regret.
    #
    # AND CACHE CANNOT FIX THE OTHER AXIS ANYWAY. Bigger cache dilutes the shared between-draw
    # component (amplitude sigma_b^2/cache) but LENGTHENS its correlation time to cache*refresh
    # steps, so the INTEGRATED noise power is (sigma_b^2/C)*(C*refresh) = sigma_b^2*refresh --
    # cache CANCELS, only `refresh` sets it. Cache only moves the timescale: 96 steps at 8, 384
    # at 32, 768 at 64, against EMA 0.999's ~1000-step averaging window -- so 8 is also the
    # value the EMA averages over most thoroughly (10 full cache turnovers per window).
    #
    # NOT AFFECTED BY ANY OF THIS: per-draw coverage. A draw is sampled `batch * refresh` = 768
    # times over its life whatever the cache (batch/cache patches per step for cache*refresh
    # steps), covering 83.5% of the reachable region. Only --refresh and --batch move that.
    # COST: VRAM only, ~130 MB per draw (256^3 x_t + dx, fp32) -> ~1.0 GB at 8.
    ap.add_argument("--cache", type=int, default=8)          # bridge draws held at once
    # STEPS BETWEEN REFRESHING ONE DRAW. This is the knob that actually costs: a draw is 1.2 s
    # against a 0.47 s step, so 12 spends ~18% of the wall clock on data. It is also the ONLY
    # knob on the integrated gradient-noise power (see --cache) and on how much of each draw is
    # consumed: 12 steps x 64 patches x 32^3 = 25,165,824 voxel-samples against an 11.0 M-voxel
    # reachable region = 2.3x oversampling, i.e. 83.5% of the draw is seen before it is evicted.
    # Halving it to 6 would use only ~58% of each 1.2 s draw and push data to 26% of the run.
    # (The 4DCT sibling cites this same 25,165,824 figure for its own refresh 12 -- but it cites
    # OUR live value as the source, so treat "the sibling validated it" as circular: the voxel
    # arithmetic above is the non-circular part.)
    ap.add_argument("--refresh", type=int, default=12)       # steps between refreshing one draw
    ap.add_argument("--views", type=int, default=360)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--context", default="global", choices=["global", "none"],
                    help="global = the arXiv:2512.18161 conditioning (in_ch=5); "
                         "none = the bare-patch prior (in_ch=1)")
    ap.add_argument("--lr", type=float, default=1e-4)          # the 2D sibling's vanilla-FM lr
    ap.add_argument("--lr_final", type=float, default=1e-6,
                    help="COSINE-decay the lr from --lr (at it=0) to this value (at it=--iters). "
                         "1e-6 = the deployed run. Pass None for the constant lr that was the "
                         "pre-2026-07-31 default. The schedule is a pure closed-form function of "
                         "`it`, so --resume needs no scheduler state -- it just continues the "
                         "same curve. NB it is anchored to --iters: changing --iters moves the "
                         "whole curve, so a resume must keep both.")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--amp", action="store_true", default=True,
                    help="fp16 AMP, like the 2D sibling. --no-amp for fp32.")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--amp_dtype", default="float16", choices=["float16", "bfloat16"],
                    help="float16 matches the 2D sibling; bfloat16 is safer but not what it used")
    # ALL AMPLITUDES IN THIS FILE ARE PEAK-TO-PEAK (see fm3d/rigid_motion's module header):
    # the spline nodes are drawn from uniform(-amp/2, +amp/2), which is Thies' own convention.
    #
    # THE EVALUATION amplitude -- fed to `run_validation` only, so the val curve stays comparable
    # to our own history. Thies IV evaluates at 5 / 5; WE DELIBERATELY EVALUATE AT 10 / 10, i.e.
    # 2x his amplitude (uncompensated median RPE 6.10 mm against his 3.05), because every result
    # we have is on that harder point and we keep it as evidence the method holds there.
    ap.add_argument("--trans_mm", type=float, default=10.0,
                    help="EVALUATION translation amplitude, peak-to-peak [mm]")
    ap.add_argument("--rot_deg", type=float, default=10.0,
                    help="EVALUATION rotation amplitude, peak-to-peak [deg]")

    # THE TRAINING amplitude protocol -- Thies II-B: "10 nodes per spline ... with a maximal
    # amplitude of 10 mm for translation and 15 deg for rotation ... we include motion patterns
    # with unequal amplitude across the different motion parameters". In "thies" mode the numbers
    # below are per-DoF MAXIMA, not the amplitude every DoF gets: a_d = A_d * u_d, u_d ~ U(0,1).
    #
    # The defaults 15 / 20 reproduce his train/eval RELATIONSHIP at OUR (2x harder) eval point
    # rather than copying his absolute 10 / 15: P(a_dof >= eval amplitude) comes out 0.33 / 0.50
    # against his own 0.29 / 0.42. Copying 10 / 15 literally would drop translation coverage to
    # 0.035 -- training BELOW the test point, because a_d = A_d*u_d halves the mean.
    ap.add_argument("--motion_amp", default="thies", choices=["fixed", "thies"],
                    help="how training motion amplitudes are drawn. thies (DEFAULT) = per-DoF "
                         "u~U(0,1) fraction of the --train_* maxima; this is what the paper "
                         "trains on and it is the only way the prior ever sees the ANISOTROPIC "
                         "residual our posterior loop actually lives in (~92%% of it sits in the "
                         "unobservable beam-axis translation, so five DoFs are nearly right and "
                         "one is badly wrong). fixed = every DoF at --trans_mm/--rot_deg.")
    ap.add_argument("--train_trans_mm", type=float, default=15.0,
                    help="max TRAINING translation, peak-to-peak [mm]. motion_amp=thies only")
    ap.add_argument("--train_rot_deg", type=float, default=20.0,
                    help="max TRAINING rotation, peak-to-peak [deg]. motion_amp=thies only")

    ap.add_argument("--bridge", default="data", choices=["data", "geom"],
                    help="WHERE THE MOTION DECAYS. data (DEFAULT since 2026-08-05) = in the "
                         "MEASUREMENT: x_t = FDK(A(x; P((1-t)theta)), P_nom). The reconstruction "
                         "geometry is nominal for every t, so the endpoint IS the static FDK by "
                         "construction -- no anchor, no detrend, and no angular-weight derivative "
                         "in the tangent. geom = the original bridge, which holds y fixed and "
                         "improves the GEOMETRY, x_t = FDK(y, P(t*theta)) + t*Delta; that path is "
                         "0.3%% of range closer to what inference walks but needs the --anchor "
                         "detrend to reach a clean endpoint. The two agree to ~1%% of range at "
                         "every t (scripts/diag_bridge_ab.py); see the module docstring for the "
                         "table and for what the switch costs (3 native sims per draw, not 1). "
                         "--anchor applies to `geom` only; under `data` it selects nothing but "
                         "the VALIDATION reference, as it always did.")
    ap.add_argument("--data_tangent", default="analytic", choices=["analytic", "fd"],
                    help="how the DATA bridge gets dy/ds. analytic (DEFAULT) = the exact "
                         "s-derivative of LEAP's pinned Joseph forward, one fused pass "
                         "(triton_leap_grad.leap_forward_tangent). fd = the central difference "
                         "this started as -- kept as the gate counterparty ONLY: it bottoms out "
                         "at ~2%% of the target in the image domain whatever --bridge_delta is "
                         "(scripts/diag_bridge_data_tangent.py), which is why the kernel exists.")
    ap.add_argument("--bridge_delta", type=float, default=0.02,
                    help="central-difference step in s for --data_tangent fd. 0.02 is the "
                         "measured IMAGE-domain U-curve minimum (2.2%% from a Richardson "
                         "reference); the sinogram-domain optimum is 0.005, but the ramp filter "
                         "amplifies the fp32 cancellation and moves the image-domain one back "
                         "up. Ignored under --data_tangent analytic.")
    ap.add_argument("--anchor", default="static", choices=["static", "gt", "none"],
                    help="what the GEOM bridge's t=1 endpoint IS (--bridge geom; under --bridge "
                         "data the endpoint needs no anchor and this only picks the validation "
                         "reference). static (DEFAULT) = the MOTION-FREE "
                         "FDK -- the same reconstruction operator, the same scan, no motion. It "
                         "is what this scanner can actually produce of a still patient, and it is "
                         "where BOTH sibling projects anchor. Without it the endpoint is FDK(y, "
                         "P(theta_true)), which still sits 1-3 dB below a static scan, so the "
                         "prior would learn FDK's residual MOTION artefact as its target. "
                         "gt = the volume itself (also removes FDK's cone-beam floor, but asks "
                         "the net to invert the operator's own defect). none = the bare bridge.")
    ap.add_argument("--sim_grid", default="native", choices=["native", "coarse"],
                    help="WHICH GRID THE MEASUREMENT y IS SIMULATED ON (cq500 only). native "
                         "(DEFAULT) = the geometry's own voxel size du*SOD/SDD, LEAP's convention: "
                         "the truth is projected at 612^3/0.4187 mm while everything is still "
                         "INVERTED on the coarse grid, so y no longer carries the reconstruction "
                         "grid's own aliasing (static-FDK crosshatch 13-16 HU -> 0 HU, and the "
                         "prior stops being trained to reproduce it). coarse = simulate on the "
                         "reconstruction grid, i.e. the inverse crime -- pre-2026-07-29 behaviour, "
                         "kept for ablation. See dataset_cq500's simulation-grid note.")
    ap.add_argument("--tangent", default="analytic", choices=["analytic", "fd"],
                    help="how bridge_pair gets the velocity target dx_t, --bridge geom ONLY. "
                         "analytic (DEFAULT) = the exact d/ds FDK(y, P(s*theta)) in one fused "
                         "pass (fdk_conebeam_3d_tangent); fd = the old central difference, three "
                         "full FDKs, ~1e-3 off -- kept as an escape hatch and gate counterparty. "
                         "--bridge data has its own knob, --data_tangent (its analytic path is "
                         "the exact d/ds of LEAP's Joseph FORWARD, leap_forward_tangent).")
    ap.add_argument("--resume", default=None,
                    help="a ckpt .pth or a run dir (uses ckpt_last.pth): restore model+EMA+"
                         "optimizer+loss history+RNG and continue to --iters")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed torch+numpy at startup (0 = the deployed run; pass None for the "
                         "unseeded pre-2026-07-31 default). Also routes the motion draw through "
                         "a dedicated seeded generator so the Akima spline nodes become "
                         "reproducible too.")
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--keep_ckpts", type=int, default=0,
                    help="keep only the newest N numbered ckpt_iterNNNNNN.pth (0 = keep all, "
                         "the historical behavior). ckpt_last.pth is never pruned.")
    ap.add_argument("--val_every", type=int, default=10000)
    ap.add_argument("--val_patients", type=int, default=3)
    ap.add_argument("--val_ode_steps", type=int, default=50)   # the deploy loop's count
    ap.add_argument("--no_tb", action="store_true", help="disable the tensorboard writer")
    ap.add_argument("--compile", action="store_true", default=True,
                    help="torch.compile the velocity net (~1.30x end-to-end here: 0.54 -> 0.41 "
                         "s/it; the net fwd+bwd is 76%% of a step and compile fuses its many small "
                         "3D-conv kernels). Numerically faithful (rel-L2 1.6e-3 vs eager, within "
                         "fp16). Checkpoints save the UNDERLYING module, so resume/inference stay "
                         "plain-UNet3D compatible. --no-compile to disable.")
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    args = ap.parse_args()

    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)

    # --seed None keeps the historical unseeded behavior (and the historical RNG call sequence).
    # `motion_gen` stays None when unseeded, which is exactly what random_motion/sample_motion
    # received before it existed.
    motion_gen = None
    if args.seed is not None:
        torch.manual_seed(args.seed)                       # seeds CPU + all CUDA generators
        np.random.seed(args.seed)
        motion_gen = torch.Generator().manual_seed(args.seed + 1)
        print(f"seeded torch+numpy with {args.seed} (motion generator: {args.seed + 1})")

    cfg = ConeBeam3DConfig.thies(n_views=args.views)
    gen = CQ500Generator(args.root, cfg, device=dev, split=args.split,
                         shape=tuple(args.shape), voxel_mm=1.0,
                         sim_native=(args.sim_grid == "native"))
    print(f"CQ500 '{args.split}': {gen.n_slabs} patients | grid {gen.shape} @ 1 mm | "
          f"self-normalized FDK (SOD*SDD/2)")
    print(f"detector {cfg.nv}x{cfg.nu} @ {cfg.du:.3f} mm | FOV {cfg.fov_diameter_mm():.0f} mm | "
          f"{cfg.n_views} views")

    # Patches are drawn only from the MEASURED REGION (a barrel, not a cylinder -- it narrows
    # with radius). Training the prior on never-measured voxels teaches it to hallucinate exactly
    # where inference has no data to correct it.
    meas = measured_region_mask(gen.shape, (gen.dz, gen.dy, gen.dx), cfg, device=dev)
    p = args.patch
    D, H, W = gen.shape
    ok = []
    for z in range(0, D - p + 1, 8):
        for yy in range(0, H - p + 1, 16):
            for xx in range(0, W - p + 1, 16):
                if meas[z:z + p, yy:yy + p, xx:xx + p].float().mean() > 0.9:
                    ok.append((z, yy, xx))
    if not ok:
        raise RuntimeError("no patch fits inside the measured region; shrink --patch")
    print(f"valid patch origins: {len(ok)}")

    # GLOBAL CONTEXT (arXiv:2512.18161). A 32^3 patch of a head cannot tell whether it is
    # orbit or posterior fossa, nor what the rest of the slab looks like, so the bare-patch
    # prior can only learn LOCAL structure. Four conditioning channels close that: the whole
    # x_t resampled onto the patch grid, and the patch voxels' absolute (z,y,x) in the volume.
    # The velocity target is untouched -- the net still predicts one channel, for channel 0.
    in_ch = 5 if args.context == "global" else 1
    model = UNet3D(in_ch=in_ch, base=args.base).to(dev)
    ema = UNet3D(in_ch=in_ch, base=args.base).to(dev)
    ema.load_state_dict(model.state_dict())
    for q in ema.parameters():
        q.requires_grad_(False)
    n_par = sum(q.numel() for q in model.parameters())
    print(f"UNet3D in_ch={in_ch} (context={args.context}) base={args.base}: "
          f"{n_par / 1e6:.2f} M params")

    # torch.compile wraps the module but shares its parameters, so `model` stays the canonical
    # UNet3D for the optimizer, the EMA, and CHECKPOINTS (saving model.state_dict() keeps plain
    # keys -- no _orig_mod. prefix -- so resume and run_posterior3d/val load unchanged). `net` is
    # the compiled forward used ONLY in the training step. First few steps pay a one-off compile
    # warmup, so the printed s/it settles after ~iter 50.
    net = torch.compile(model) if args.compile else model
    if args.compile:
        print("torch.compile: ON (net fwd/bwd only; ~1.30x). --no-compile to disable.")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # Cosine LR as a PURE function of `it` (no scheduler object => nothing to save/restore, so
    # --resume from an older ckpt that predates the schedule just picks the curve back up).
    # lr_final=None reproduces the constant-lr path exactly (returns args.lr, never touches opt).
    def lr_at(it):
        if args.lr_final is None:
            return args.lr
        prog = min(it, args.iters) / max(args.iters, 1)
        return args.lr_final + 0.5 * (args.lr - args.lr_final) * (1 + math.cos(math.pi * prog))

    # ---- RESUME ---------------------------------------------------------------------------
    # Continue a finished/killed run: restore model + EMA + OPTIMIZER STATE + the loss history,
    # and carry on from the saved iter. The optimizer state is the part that matters -- AdamW's
    # moment estimates take thousands of steps to rebuild, and dropping them makes a "resumed"
    # run behave like a fresh one with a warm init (a silent, and silently worse, restart).
    start_it = 0
    if args.resume:
        ck_path = (os.path.join(args.resume, "ckpt_last.pth")
                   if os.path.isdir(args.resume) else args.resume)
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        prev = ck.get("args", {})

        # A resumed run MUST keep the architecture and the bridge it was trained under; anything
        # else silently changes what the weights mean. `_cmp` normalizes sequences because the
        # SAME --shape arrives as a tuple when defaulted and a list when typed on the CLI.
        def _cmp(v):
            return tuple(v) if isinstance(v, (list, tuple)) else v

        # Checkpoints written before 2026-08-05 have no 'bridge' key and were ALL trained on the
        # geometry bridge. The loop below only compares keys the checkpoint HAS, so without this
        # a resume of an old run would silently adopt the new `data` default and continue the
        # weights on a different manifold.
        prev.setdefault("bridge", "geom")

        for k in ("base", "patch", "context", "bridge", "anchor", "shape", "views", "dataset",
                  "tangent", "data_tangent", "trans_mm", "rot_deg", "sim_grid",
                  "motion_amp", "train_trans_mm", "train_rot_deg"):
            if k in prev and k in vars(args) and _cmp(prev[k]) != _cmp(vars(args)[k]):
                raise SystemExit(f"--resume mismatch on '{k}': checkpoint has {prev[k]!r}, "
                                 f"this run asks for {vars(args)[k]!r}")
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        if ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        # RNG: restore the sampling streams so a resumed run continues the data sequence the
        # interrupted one would have drawn. Old checkpoints predate these keys -- note and go on.
        rng = ck.get("rng")
        if rng is None:
            print("resume: checkpoint carries no RNG state (older format) -- continuing "
                  "with fresh RNG")
        else:
            torch.set_rng_state(rng["torch"].cpu())        # map_location may have moved these
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
            np.random.set_state(rng["numpy"])
            if rng.get("motion") is not None:
                if motion_gen is None:
                    motion_gen = torch.Generator()
                motion_gen.set_state(rng["motion"].cpu())
        loss_log_prev = ck.get("loss_log", [])
        start_it = int(ck.get("iter", 0))
        print(f"resumed {ck_path} at iter {start_it} "
              f"({len(loss_log_prev)} loss points carried over) -> training to {args.iters}")
        if start_it >= args.iters:
            raise SystemExit(f"--iters {args.iters} is not beyond the checkpoint's {start_it}")
    else:
        loss_log_prev = []

    # TRAIN vs EVAL motion amplitudes are deliberately different objects (Thies II-B vs IV).
    # `mot_*` feeds `draw()`; `args.trans_mm/rot_deg` feed `run_validation` untouched, so the
    # val curve keeps measuring the 5 mm / 5 deg evaluation protocol whatever training samples.
    mot_trans = args.train_trans_mm if args.motion_amp == "thies" else args.trans_mm
    mot_rot = args.train_rot_deg if args.motion_amp == "thies" else args.rot_deg
    # Print BOTH conventions: ours is the node half-range, Thies quotes peak-to-peak (2x).
    _mx = "max " if args.motion_amp == "thies" else ""
    print(f"motion (ALL PEAK-TO-PEAK): train={args.motion_amp} {_mx}{mot_trans:g} mm / "
          f"{mot_rot:g} deg  |  val=fixed {args.trans_mm:g} mm / {args.rot_deg:g} deg"
          f"   [Thies: train max 10/15, eval 5/5]")
    print("bridge: " + (f"DATA  x_t = FDK(A(x; P((1-t)theta)), P_nom)  -- endpoint = static FDK "
                        f"by construction, no anchor; tangent={args.data_tangent}"
                        + (f" (delta {args.bridge_delta:g})" if args.data_tangent == "fd" else
                           " (exact d/ds of LEAP's Joseph forward)")
                        if args.bridge == "data" else
                        f"GEOM  x_t = FDK(y, P(t*theta)) + t*Delta  -- anchor={args.anchor}, "
                        f"tangent={args.tangent}"))

    next_idx = [None]        # the NEXT cq500 patient, sampled one draw ahead so `prefetch_fine`
                             # can load its native-grid volume under the training steps
    prof_draw = os.environ.get("FM3D_DRAW_PROF", "0") != "0"   # per-stage draw timing to stdout

    def draw():
        """One bridge sample, held whole-volume: (x_t (1,1,D,H,W), dx (D,H,W), t, ctx).

        `ctx` is the global-context channel -- x_t on the patch grid -- and it is a property
        of the DRAW, not of the patch, so it is computed once here and shared by every patch
        cropped from this volume. At inference `predict_x1_patched` rebuilds it the same way
        from the evolving x_t, so train and infer see the same channel.

        Under `--bridge data` (the default) the draw does NOT simulate the full-motion y at all:
        every sinogram it needs is a partially-moved one at s = 1-t, and it needs three of them
        (the value plus a central difference) -- see `bridge_pair_data`. The anchor block below
        is skipped entirely; the endpoint is clean by construction.

        Under `--bridge geom` the ANCHOR (see `bridge_pair`) is a property of the draw. On cq500
        its motion-free static FDK is MEMOIZED per volume (`gen.static_anchor_net`), since it
        depends on neither the motion nor t -- that removes one forward projection (~0.9 s) and one
        FDK (~0.18 s) from every draw after a volume's first, and the cache refreshes a draw only
        every `--refresh` steps. To reach the memo we sample the volume INDEX ourselves here
        (cq500's `volume(idx)` is a clean per-patient lookup)."""
        _tm: dict[str, float] = {}
        _tk = [time.time()]

        def _tick(name):                       # FM3D_DRAW_PROF=1: wall time per draw stage
            if prof_draw:
                torch.cuda.synchronize()
                now = time.time(); _tm[name] = now - _tk[0]; _tk[0] = now

        if True:  # noqa: SIM115 -- kept one indent level so the diff to the two-dataset era stays readable
            # The patient for THIS draw was sampled by the PREVIOUS draw (and its native-grid
            # volume has been prefetching on a worker thread ever since -- see
            # `CQ500Generator.prefetch_fine` for the measured stall this hides). Sample the
            # NEXT draw's patient now and start its load. This shifts the global RNG stream by
            # one randint vs the pre-prefetch code (and a resume re-samples `next_idx[0]`, so
            # the first draw after a resume differs from the uninterrupted stream) -- both only
            # reshuffle which random patient is drawn when, never what a draw contains.
            idx = next_idx[0] if next_idx[0] is not None \
                else int(torch.randint(gen.n_slabs, (1,)).item())
            next_idx[0] = int(torch.randint(gen.n_slabs, (1,)).item())
            gen.prefetch_fine(next_idx[0])
            _tick("rng")
            # THE COARSE VOLUME IS ONLY THE `gt` ANCHOR'S TARGET, so it is loaded LAZILY. Under
            # the deployed --anchor static it is never read (the anchor comes from the memoized
            # `static_anchor_net`, the measurement from `simulate`'s NATIVE-grid volume), and
            # loading it anyway cost 0.11 s of a 1.16 s draw -- 9.6% of the draw for nothing.
            # `gen.volume` consumes no RNG, so deferring it is bit-identical.
            vol = None
            _tick("vol")
            th = random_motion(gen.cfg.n_views, trans_mm=mot_trans, rot_deg=mot_rot,
                               amp_mode=args.motion_amp, device=dev, generator=motion_gen)[None]
            _tick("motion")
            # y is simulated on the NATIVE grid (`gen.simulate`, see the simulation-grid note in
            # dataset_cq500.__init__); `vol` above is the coarse inversion-grid target/anchor.
            # The DATA bridge never needs the full-motion sinogram -- it simulates its own three
            # partially-moved ones at s = 1-t -- so skip this entirely there.
            y = None if args.bridge == "data" else \
                gen.simulate(idx, params_to_Pmot(th[0], gen.P_nom)[None])
            _tick("sim_y")

        if args.bridge == "data":
            # THE DATA BRIDGE: motion decays in the MEASUREMENT, geometry stays P_nom, endpoint
            # is the static FDK by construction. No anchor, no Delta, no shared ramp pass (each
            # of the three sinograms is its own).
            t = sample_t(1, dev)[0]
            x_t, dx = bridge_pair_data(gen, idx, t, th[0], delta=args.bridge_delta,
                                       mode=args.data_tangent)
            _tick("bridge")
            x_t = x_t[None, None]                                    # (1,1,D,H,W)
            ctx = volume_context(x_t, (p, p, p)) if in_ch == 5 else None
            _tick("ctx")
            if prof_draw:
                print("[draw]", " ".join(f"{k} {v:.3f}" for k, v in _tm.items()),
                      f"| total {sum(_tm.values()):.3f}", flush=True)
            return x_t, dx, t, ctx

        # ---- --bridge geom from here ----------------------------------------------------
        # ONE ramp pass per draw, shared by the anchor's FDK(y, P(theta)) and the bridge's
        # FDK(y, P(t*theta)) below. The filtering half of the FDK never reads Pmat (cosine /
        # Wang / Ohnesorge / ramp are detector-only), so this is EXACT, not an approximation --
        # see `projector_3d.fdk_backproject_filtered`. Costs ~0.08 s of a 1.16 s draw to run twice.
        filt = gen.fdk_filtered(y)
        _tick("filter")

        dlt = None
        if args.anchor != "none":
            if args.anchor == "gt":
                # THE VOLUME ITSELF. A motion-free FDK is NOT clean -- it still carries the
                # cone-beam artefact of a circular orbit (34.3 dB from the GT here; 44.3 dB if you
                # look only at the midplane, so it IS the cone and not sampling -- quadrupling the
                # views buys 0.1 dB). That artefact is a defect of the INVERSE, not a property of
                # the data: y is the projection of the true volume, so the image the data supports
                # is the GT. Anchoring at the static FDK would teach the prior to PAINT IN cone
                # artefacts it is supposed to remove.
                if vol is None:               # the lazy load above -- `gt` is its only consumer
                    vol = gen.volume(idx)
                x_anchor = gen.to_net(vol[0, 0])
            else:                             # "static": the memoized motion-free recon
                x_anchor = gen.static_anchor_net(idx)
            _tick("anchor")
            x1_geo = gen.to_net(gen.fdk(y, params_to_Pmot(th[0], gen.P_nom)[None],
                                        filtered=filt)[0])
            dlt = x_anchor - x1_geo
            _tick("x1_fdk")
        t = sample_t(1, dev)[0]
        x_t, dx = bridge_pair(gen, t, y, th[0], dlt, mode=args.tangent, filtered=filt)
        _tick("bridge")
        x_t = x_t[None, None]                                        # (1,1,D,H,W)
        ctx = volume_context(x_t, (p, p, p)) if in_ch == 5 else None
        _tick("ctx")
        if prof_draw:
            print("[draw]", " ".join(f"{k} {v:.3f}" for k, v in _tm.items()),
                  f"| total {sum(_tm.values()):.3f}", flush=True)
        return x_t, dx, t, ctx

    cache = [draw() for _ in range(args.cache)]
    print(f"bridge cache warm ({args.cache} draws)")

    # ---- validation: a held-out VAL generator + a tensorboard writer -----------------------
    # Same geometry, the VAL split (never trained on), evaluated inline every `val_every` steps.
    # Standalone `val_fm3d.py` runs the identical `run_validation`; the montages land in out/val.
    val_gen = None
    if args.dataset == "cq500" and args.val_every > 0:
        val_gen = CQ500Generator(args.root, cfg, device=dev, split="val",
                                 shape=tuple(args.shape), voxel_mm=1.0, verbose=False,
                                 sim_native=(args.sim_grid == "native"))
    val_dir = os.path.join(args.out, "val")
    os.makedirs(val_dir, exist_ok=True)
    writer = None
    if not args.no_tb:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.out, "tb"))
        print(f"tensorboard: tensorboard --logdir {os.path.join(args.out, 'tb')}")

    t0 = time.time()
    loss_log = list(loss_log_prev)          # (iter, loss) per step, saved into every checkpoint
    pending: list[tuple[int, torch.Tensor]] = []   # detached GPU losses awaiting ONE host sync
    n_nonfinite = 0

    def flush_losses():
        """Move pending losses into loss_log with a single device sync for the whole block.

        The old loop called float(loss) AND torch.isfinite(loss) every step -- two host syncs
        per iteration that serialized the CPU against the GPU. (GradScaler.step keeps its own
        internal found_inf sync; that one is inherent to fp16 AMP.) fp16 can overflow to
        inf/nan; GradScaler skips a non-finite step on its own, but a persistently non-finite
        LOSS means the run is dead and should say so, not spin silently -- the check now runs
        per flushed block instead of per step."""
        nonlocal n_nonfinite
        if not pending:
            return
        vals = torch.stack([v for _, v in pending]).cpu()          # the one sync
        nf = int((~torch.isfinite(vals)).sum())
        if nf:
            n_nonfinite += nf
            print(f"it {pending[-1][0]:6d} | WARN non-finite loss x{nf} in the last "
                  f"{len(pending)} steps ({n_nonfinite} total; fp16 overflow?)", flush=True)
        loss_log.extend((i, float(v)) for (i, _), v in zip(pending, vals))
        pending.clear()

    for it in range(start_it + 1, args.iters + 1):
        if args.lr_final is not None:
            for g in opt.param_groups:
                g["lr"] = lr_at(it)

        if it % args.refresh == 0:
            cache[torch.randint(len(cache), (1,)).item()] = draw()

        # Sample ALL (cache entry, origin) pairs first, in the exact per-patch RNG order the
        # old loop used (one randint over the cache, one over the origins, alternating), THEN
        # group by entry so make_tile_inputs runs once per distinct entry (<= cache size calls
        # instead of `batch`). The loss is a mean over the batch, so row order is free.
        picks = [(int(torch.randint(len(cache), (1,)).item()),
                  int(torch.randint(len(ok), (1,)).item())) for _ in range(args.batch)]
        groups: dict[int, list[int]] = {}
        for ci, oi in picks:
            groups.setdefault(ci, []).append(oi)
        xs, ds, ts = [], [], []
        for ci, ois in groups.items():
            x_t, dx, t, ctx = cache[ci]
            coords = [ok[oi] for oi in ois]
            xs.append(make_tile_inputs(x_t, coords, (p, p, p), ctx))   # (len(ois),C,p,p,p)
            ds += [dx[z:z + p, yy:yy + p, xx:xx + p] for (z, yy, xx) in coords]
            ts += [t] * len(ois)
        xb = torch.cat(xs, 0)                                   # (B,in_ch,p,p,p)
        db = torch.stack(ds)[:, None]                           # (B,1,p,p,p) -- target: ch 0 only
        tb = torch.stack(ts)

        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=getattr(torch, args.amp_dtype), enabled=args.amp):
            loss = ((net(xb, tb) - db) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            d = min(args.ema, (1 + it) / (10 + it))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm, alpha=1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)

        pending.append((it, loss.detach()))
        if it % 50 == 0:
            flush_losses()
            recent = [l for _, l in loss_log[-50:]]
            ma = sum(recent) / len(recent)
            # rate over THIS process's steps, not `it` -- after a resume `it` starts high and
            # dividing by it would report a nonsense s/it
            lr_now = opt.param_groups[0]["lr"]
            lr_str = f" | lr {lr_now:.2e}" if args.lr_final is not None else ""
            print(f"it {it:6d} | loss {loss_log[-1][1]:.5f} | ma50 {ma:.5f}{lr_str} | "
                  f"{(time.time() - t0) / max(it - start_it, 1):.2f}s/it", flush=True)
            if writer is not None:
                writer.add_scalar("train/fm_loss", loss_log[-1][1], it)
                writer.add_scalar("train/fm_loss_ma50", ma, it)
                writer.add_scalar("train/lr", lr_now, it)

        if val_gen is not None and it % args.val_every == 0:
            # PRIOR-ONLY ODE from the cold start on the val split -- what the loss cannot tell us.
            # EMA weights, eval mode, then straight back to training.
            ema.eval()
            # The patch->volume scheme is left at run_validation's DEFAULT on purpose: that
            # default IS the deployed inference scheme (uniform K=2, run_posterior3d.fm_predict),
            # so there is one place to change it and the inline curve can never drift away from
            # what the posterior loop runs.
            run_validation(ema, val_gen, meas, val_dir, it=it, patients=args.val_patients,
                           patch=args.patch, ode_steps=args.val_ode_steps, anchor=args.anchor,
                           trans_mm=args.trans_mm, rot_deg=args.rot_deg, writer=writer, dev=dev)
            ema.train()

        if it % args.save_every == 0 or it == args.iters:
            flush_losses()
            # loss_log rides along in the checkpoint, so the convergence curve survives even if
            # the text log is rotated or lost -- the whole per-step history, (iter, loss) pairs.
            # "rng" makes --resume continue the exact sampling streams (older ckpts lack it).
            ck = {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                  "scaler": scaler.state_dict() if args.amp else None,
                  "iter": it, "args": {**vars(args), "amp_units": AMP_UNITS},
                  "rng": {"torch": torch.get_rng_state(),
                          "cuda": torch.cuda.get_rng_state_all(),
                          "numpy": np.random.get_state(),
                          "motion": motion_gen.get_state() if motion_gen is not None else None},
                  "loss_log": loss_log}
            torch.save(ck, os.path.join(args.out, f"ckpt_iter{it:06d}.pth"))
            torch.save(ck, os.path.join(args.out, "ckpt_last.pth"))
            print(f"saved ckpt_iter{it:06d}.pth")
            if args.keep_ckpts > 0:
                numbered = sorted(glob.glob(os.path.join(args.out, "ckpt_iter*.pth")))
                for f in numbered[:-args.keep_ckpts]:
                    os.remove(f)
                    print(f"pruned {os.path.basename(f)}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
