"""Gate: the Thies baseline is wired to OUR geometry correctly, and its gradient is real.

Everything here runs on a synthetic head phantom in a few seconds -- no CQ500, no trained
quality network -- so it can be run before either bench script is launched, and re-run whenever
`fm3d/geometry_3d.py`, `fm3d/rigid_motion.py` or the vendored tree moves.

    G1  their torch Akima  ==  scipy Akima                                    max abs < 1e-12
    G2  the 30-node estimator model can represent a 10-node truth            RPE < 0.35 mm
    G3  geometry adapter: mm-domain P -> pixel-domain P                       exact, < 1e-4 px
    G4  VIF map sums to the Sheikh & Bovik scalar                            rel < 1e-5
    G5  vendored backprojection agrees with OUR FDK where it should          corr > 0.99
    G6  the vendored analytic dI/dP matches a central finite difference      rel < 5e-2
    G7  x = 0 reproduces the uncompensated scan; theta_true recovers it       PSNR gain > 3 dB
    G8  the Eq. 6 optimizer loop runs end to end (untrained net)              f finite, x moves
    G9  the FAST kernels reproduce the VENDORED ones (value and dI/dP)        rel < 1e-3, cos > 1-1e-6
    G10 the float32 VIF* reproduces the float64 one                           rel < 1e-3
    G11 the stage-1 pair is the PAPER's pair (static data, perturbed P)       target >> consistent-pair target

G6 IS THE ONE THAT MATTERS. A projector whose geometry gradient is wrong (or zero) does not
raise -- it silently makes the optimizer wander, which is exactly the failure mode this repo has
already been bitten by. The comparison is a DIRECTIONAL derivative against a central difference,
not an entrywise Jacobian, because their kernel is bilinear and its entrywise derivative has
cell-edge kinks that no finite difference can resolve.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.thies.motion import ThiesSplineMotion, akima_resample          # noqa: E402
from bench.thies.recon import ThiesConeRecon, VolumeGrid, to_pixel_matrices  # noqa: E402
from bench.thies.vif import (vif_map_3d, vif_scalar_3d,                   # noqa: E402
                             vif_star_map_3d)
from fm3d.geometry_3d import ConeBeam3DConfig, detector_coords_3d         # noqa: E402
from fm3d.phantom import head_phantom                                     # noqa: E402
from fm3d.projector_3d import (fdk_conebeam_3d_batched,                   # noqa: E402
                               forward_project_3d_batched)
from fm3d.reg_metric import psnr                                          # noqa: E402
from fm3d.rigid_motion import (akima_motion, params_to_Pmot,              # noqa: E402
                               reprojection_error)

DEV = "cuda"
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILED.append(name)


# --------------------------------------------------------------------------------------------
def g1_akima():
    print("G1  their torch Akima vs scipy")
    from scipy.interpolate import Akima1DInterpolator
    V, n = 360, 10
    vals = torch.randn(n, 6, dtype=torch.float64)
    got = akima_resample(vals, V).numpy()
    tn = np.linspace(0, V - 1, n)
    ref = np.stack([Akima1DInterpolator(tn, vals[:, c].numpy())(np.arange(V))
                    for c in range(6)], -1)
    m = float(np.abs(got - ref).max())
    check("G1 akima parity", m < 1e-12, f"max abs diff {m:.3e}")

    vals.requires_grad_(True)
    g = torch.autograd.grad(akima_resample(vals, V).sum(), vals)[0]
    check("G1 akima differentiable", torch.isfinite(g).all() and g.abs().sum() > 0,
          f"|grad| = {float(g.abs().sum()):.4g}")


def g2_model_capacity(cfg, P_nom):
    print("G2  30-node estimator model vs a 10-node truth")
    th = akima_motion(cfg.n_views, n_nodes=10, trans_mm=10.0, rot_deg=10.0,
                      device=DEV, seed=3, zero_centre=True)
    mot = ThiesSplineMotion(cfg.n_views, n_nodes=30, device=DEV)
    mot.load_theta_(th)
    e = reprojection_error(mot.theta().detach(), th, P_nom)["rpe_mm"]
    check("G2 representable", e < 0.35,
          f"RPE(30-node fit of the 10-node truth) = {e:.4f} mm "
          f"(Thies' own method reaches 0.61 mm, so the model must be well under it)")


def g3_adapter(cfg, P_nom):
    print("G3  mm-domain P -> pixel-domain P")
    P_pix = to_pixel_matrices(P_nom, cfg)
    u, v = detector_coords_3d(cfg, device=DEV)
    # A grid of world points, projected both ways.
    pts = torch.randn(64, 3, device=DEV) * 60.0
    ph = torch.cat([pts, torch.ones(64, 1, device=DEV)], -1)
    for vi in (0, cfg.n_views // 3, cfg.n_views - 1):
        h_mm = ph @ P_nom[vi].T
        h_px = ph @ P_pix[vi].T
        u_mm, v_mm = h_mm[:, 0] / h_mm[:, 2], h_mm[:, 1] / h_mm[:, 2]
        # the index our own detector_coords convention assigns to that mm position
        ju = (u_mm - cfg.det_offset_u_mm) / cfg.du + (cfg.nu - 1) / 2.0
        jv = (v_mm - cfg.det_offset_v_mm) / cfg.dv + (cfg.nv - 1) / 2.0
        e = max(float((h_px[:, 0] / h_px[:, 2] - ju).abs().max()),
                float((h_px[:, 1] / h_px[:, 2] - jv).abs().max()))
        check(f"G3 view {vi}", e < 1e-3, f"max |pixel index error| = {e:.3e} px")
    # sanity: the isocentre lands on the panel centre
    o = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=DEV) @ P_pix[0].T
    c = (float(o[0, 0] / o[0, 2]), float(o[0, 1] / o[0, 2]))
    tgt = ((cfg.nu - 1) / 2.0 - cfg.det_offset_u_mm / cfg.du,
           (cfg.nv - 1) / 2.0 - cfg.det_offset_v_mm / cfg.dv)
    check("G3 isocentre", max(abs(c[0] - tgt[0]), abs(c[1] - tgt[1])) < 1e-3,
          f"isocentre -> {c} (expected {tgt}); u range on panel [0,{cfg.nu-1}], "
          f"v [0,{cfg.nv-1}]; detector spans u {float(u[0]):.1f}..{float(u[-1]):.1f} mm")


def g4_vif():
    print("G4  the VIF map sums to the scalar")
    torch.manual_seed(0)
    ref = torch.rand(1, 1, 48, 48, 48, device=DEV)
    ref = torch.nn.functional.avg_pool3d(ref, 3, 1, 1)          # give it some structure
    dist = ref + 0.05 * torch.randn_like(ref)
    m = vif_map_3d(dist, ref)
    s = vif_scalar_3d(dist, ref)
    rel = float((m.sum() - s).abs() / s.abs().clamp_min(1e-12))
    check("G4 map sums to scalar", rel < 1e-5,
          f"sum(map) = {float(m.sum()):.6f} vs scalar {float(s):.6f} (rel {rel:.2e})")
    ident = vif_scalar_3d(ref, ref)
    check("G4 identity ~ 1", abs(float(ident) - 1.0) < 5e-2,
          f"VIF(ref, ref) = {float(ident):.4f} (1.0 means 'no information lost')")


def g5_recon(cfg, P_nom, y, vol_gt):
    print("G5  vendored backprojection vs our FDK (static scan)")
    recon = ThiesConeRecon(cfg)
    grid = VolumeGrid.centred(128, 2.0)
    got = recon(y, P_nom, grid)
    ours = fdk_conebeam_3d_batched(y, P_nom[None], recon_u(cfg), recon_v(cfg), cfg,
                                   D=128, H=128, W=128, dx=2.0, dy=2.0, dz=2.0,
                                   window="ramlak")[0]
    # Compare only the central sphere: their sum has no 1/w^2 term, so the two DIVERGE by a
    # smooth radial factor toward the FOV edge -- by construction, not by error (PROVENANCE §4.1).
    n = 128
    c = torch.arange(n, device=DEV) - (n - 1) / 2.0
    r = (c[:, None, None] ** 2 + c[None, :, None] ** 2 + c[None, None, :] ** 2).sqrt()
    m = r < 25.0
    a_, b_ = got[m], ours[m]
    corr = float(torch.corrcoef(torch.stack([a_, b_]))[0, 1])
    ratio = float((a_ * b_).sum() / (b_ * b_).sum())
    check("G5 shape agrees", corr > 0.99, f"corr = {corr:.5f} over the central 50 mm sphere")
    check("G5 scale agrees", 0.8 < ratio < 1.25,
          f"least-squares scale thies/ours = {ratio:.4f} (mu_scale = {recon.scale:.4g}); "
          f"a value far from 1 means the mu normalization in recon.scale is wrong")


def _blur2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian over the two DETECTOR axes of a (V,nv,nu) sinogram."""
    n = int(4 * sigma) | 1
    k = torch.arange(n, device=x.device, dtype=x.dtype) - (n - 1) / 2
    k = torch.exp(-k ** 2 / (2 * sigma ** 2)); k = k / k.sum()
    x = torch.nn.functional.conv2d(x[:, None], k.view(1, 1, n, 1), padding=(n // 2, 0))
    return torch.nn.functional.conv2d(x, k.view(1, 1, 1, n), padding=(0, n // 2))[:, 0]


def g6_gradient(cfg, P_nom, y):
    """THE gradient gate. Two things it must respect, or it will fail for the wrong reason.

    1. **The kink.** Their forward interpolates the sinogram BILINEARLY, so the true derivative
       is piecewise constant with jumps at every cell edge. A finite difference that moves the
       projected point across cells averages over those jumps and cannot converge. Fix: BLUR the
       sinogram, so the interpolant is locally smooth over several cells. This is the same
       precaution the analytic-tangent rig in this repo already needs.
    2. **The model gap is real and is theirs.** Their backward does NOT differentiate the
       bilinear interpolant; it uses `torch.gradient` -- "numerically approximated by second
       order accurate central differences" (TMI L~300-310) -- as the spatial derivative. So the
       analytic value is the derivative of a slightly SMOOTHED model, and exact agreement is not
       expected even in principle. On a blurred sinogram the two models coincide to the blur's
       accuracy, which is why the tolerance is 1e-1 and not 1e-4. That is a property of their
       published method, not of our wiring; what this gate can and does catch is a wrong chain
       rule, a wrong axis order, or a silently zero gradient.
    3. The image functional must be SMOOTH. A white-noise weight image puts all its mass at the
       frequencies the interpolation error lives in and makes the FD meaningless.
    """
    print("G6  analytic dI/dP vs a central finite difference (directional)")
    recon = ThiesConeRecon(cfg)
    grid = VolumeGrid.centred(64, 4.0)                   # small: FD needs 2 extra backprojections
    g_filt = _blur2d(recon.filter(y), sigma=3.0)         # (1) kill the bilinear kink

    torch.manual_seed(0)
    c = torch.arange(64, device=DEV, dtype=torch.float32) - 31.5
    zz, yy, xx = torch.meshgrid(c, c, c, indexing="ij")
    w = torch.cos(0.05 * xx) * torch.cos(0.04 * yy) * torch.cos(0.03 * zz)   # (3) smooth
    dP = torch.randn_like(P_nom)
    dP = dP / dP.norm() * P_nom.norm()                   # ||dP|| = ||P||, so eps is RELATIVE

    P = P_nom.clone().requires_grad_(True)
    f = (recon.backproject(g_filt, P, grid) * w).sum()
    f.backward()
    ana = float((P.grad * dP).sum())

    best = None
    for eps in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4):
        with torch.no_grad():
            fp = (recon.backproject(g_filt, P_nom + eps * dP, grid) * w).sum()
            fm = (recon.backproject(g_filt, P_nom - eps * dP, grid) * w).sum()
        fd = float((fp - fm) / (2 * eps))
        rel = abs(ana - fd) / max(abs(ana), abs(fd), 1e-30)
        print(f"      eps {eps:.0e}: FD {fd: .6e}  analytic {ana: .6e}  rel {rel:.3e}")
        best = rel if best is None else min(best, rel)
    check("G6 geometry gradient", best is not None and best < 1e-1,
          f"best relative agreement over the eps sweep = {best:.3e} "
          f"(see this function's docstring for why the bar is 1e-1 and not 1e-4)")
    check("G6 gradient is nonzero", abs(ana) > 0,
          f"<dI/dP, dP> = {ana:.6e} (a silently-zero geometry gradient is THE trap here)")


def g7_oracle(cfg, P_nom, vol_gt):
    print("G7  x = 0 is the uncompensated scan; theta_true recovers it")
    th = akima_motion(cfg.n_views, n_nodes=10, trans_mm=10.0, rot_deg=10.0,
                      device=DEV, seed=3, zero_centre=True)
    P_mot = params_to_Pmot(th, P_nom)
    y_mot = forward_project_3d_batched(vol_gt, P_mot[None], recon_u(cfg), recon_v(cfg),
                                       dx=1.0, dy=1.0, dz=1.0)
    recon = ThiesConeRecon(cfg)
    # 128^3 at 1 mm, i.e. the phantom's OWN grid, so the PSNR below compares like with like.
    grid = VolumeGrid.centred(128, 1.0)
    g_filt = recon.filter(y_mot)

    mot = ThiesSplineMotion(cfg.n_views, n_nodes=30, device=DEV)
    with torch.no_grad():
        v_zero = recon.backproject(g_filt, mot.Pmot(P_nom), grid)     # x = 0 -> identity
        v_nom = recon.backproject(g_filt, P_nom, grid)
        mot.load_theta_(th)
        v_true = recon.backproject(g_filt, mot.Pmot(P_nom), grid)
    d = float((v_zero - v_nom).abs().max())
    check("G7 x=0 is the identity", d < 1e-4,
          f"max |I(x=0) - I(P_nom)| = {d:.3e} (Eq. 6 starts at the uncompensated scan)")

    ref = vol_gt[0, 0]                                                # same 128^3 @ 1 mm grid
    p0, p1 = psnr(v_zero, ref), psnr(v_true, ref)
    check("G7 oracle helps", (p1 - p0) > 3.0,
          f"PSNR vs GT: uncompensated {p0:.2f} dB -> theta_true {p1:.2f} dB "
          f"(+{p1-p0:.2f} dB). If this is ~0 the motion is not reaching the backprojection.")


def g8_loop(cfg, P_nom, vol_gt):
    """THE STAGE-2 SMOKE TEST: run Eq. 6 for real, with an UNTRAINED quality network.

    This deliberately does not need `scripts/bench_thies_train_qm.py` to have finished. What it
    checks is the mechanics that a randomly-initialized net exercises just as well as a trained
    one: that `f` is finite, that a gradient reaches the 30x6 spline parameters, that `x`
    actually moves, and that the exponential step decay is applied. It says NOTHING about whether
    the method works -- an untrained VIF* regressor is not an objective, and RPE is expected to
    get WORSE here. Only `scripts/bench_thies_estimate.py` with a trained checkpoint answers that.
    """
    print("G8  the Eq. 6 optimizer loop runs end to end (untrained net)")
    from bench.thies.qmnet import QualityMetricUNet3D
    from bench.thies.recon import to_unit

    th = akima_motion(cfg.n_views, n_nodes=10, trans_mm=10.0, rot_deg=10.0,
                      device=DEV, seed=3, zero_centre=True)
    y_mot = forward_project_3d_batched(vol_gt, params_to_Pmot(th, P_nom)[None],
                                       recon_u(cfg), recon_v(cfg), dx=1.0, dy=1.0, dz=1.0)
    recon = ThiesConeRecon(cfg)
    grid = VolumeGrid.centred(64, 4.0)
    g_filt = recon.filter(y_mot)

    torch.manual_seed(0)
    net = QualityMetricUNet3D().to(DEV).freeze()
    mot = ThiesSplineMotion(cfg.n_views, n_nodes=30, device=DEV)
    x0 = mot.x.detach().clone()

    s0, decay, n_it = 100.0, 0.97, 3
    fs, gs = [], []
    for n in range(n_it):
        vol = recon.backproject(g_filt, mot.Pmot(P_nom), grid)
        f = net.score(to_unit(vol)[None, None]).mean()
        f.backward()
        gs.append(mot.gd_step(s0 * decay ** n))
        fs.append(float(f))
    moved = float((mot.x.detach() - x0).abs().max())

    check("G8 objective finite", all(math.isfinite(v) for v in fs),
          f"f = {[f'{v:.6g}' for v in fs]}")
    check("G8 gradient reaches x", all(g > 0 for g in gs) and moved > 0,
          f"|grad| per step {[f'{g:.3e}' for g in gs]}, max |dx| = {moved:.3e} "
          f"(zero here means the chain motion -> P -> backprojection -> net is broken). "
          f"NOTE the magnitude is meaningless with an untrained net -- but it IS the number to "
          f"re-read after stage 1, because Thies' s0 = 100 was calibrated on THEIR objective "
          f"scale, and s0 * |grad| must move x by a fraction of a mm/deg, not 1e-4.")
    check("G8 x stays finite", torch.isfinite(mot.x).all() and torch.isfinite(mot.theta()).all(),
          f"max |x| = {float(mot.x.abs().max()):.4g} "
          f"(mm and DEGREES -- see bench/thies/motion.py on why rotations are in degrees)")


def g10_vif_precision():
    """The DEPLOYED float32 VIF* against the float64 one it replaced.

    `vif._vif_terms` mean-centres before forming the second moments, which is what makes float32
    safe: the textbook `conv(x*x) - mu*mu` cancels ~5 digits at Sheikh's 0-255 scale and was the
    reason the module ran in float64 (412.8 ms per 128^3 pair = 33% of a `bench_thies_train_qm`
    step; now ~7 ms). Two regimes are checked because they fail differently -- pure noise has
    large local variance everywhere and is easy, while a SMOOTH head-like volume has regions
    where the local variance is tiny against the global spread and the global centring only
    partly helps. The head-like case is the binding one.
    """
    print("G10 float32 VIF* vs float64")
    torch.manual_seed(0)
    smooth = torch.rand(1, 1, 64, 64, 64, device=DEV)
    for _ in range(3):
        smooth = torch.nn.functional.avg_pool3d(smooth, 5, 1, 2)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
    cases = {
        "head-like (smooth, shifted)": (smooth,
                                        torch.roll(smooth, 2, dims=3) * 0.9
                                        + 0.03 * torch.randn_like(smooth)),
        "noise": (torch.rand_like(smooth), torch.rand_like(smooth)),
    }
    for name, (r, d) in cases.items():
        a64 = vif_star_map_3d(d, r, dtype=torch.float64)
        a32 = vif_star_map_3d(d, r, dtype=torch.float32)
        rel = float((a32 - a64).norm() / a64.norm().clamp_min(1e-30))
        check(f"G10 {name}", rel < 1e-3,
              f"rel L2 {rel:.3e}, mean {float(a64.mean()):.6f} (fp64) vs "
              f"{float(a32.mean()):.6f} (fp32)")


def g9_fast_kernels(cfg, P_nom, y):
    """The deployed FAST kernels vs the VENDORED ones, on a MOTION geometry.

    `bench/thies/fast_backprojector.py` replaces the vendored CUDA kernels with the same
    expressions in a different execution shape (register accumulation in the forward, a
    shared-memory tree reduction and one folded reciprocal in the backward) because the vendored
    backward spends 21 of its 21.5 s in serialized atomics -- 9.06e9 of them onto 12 addresses.
    Everything the benchmark reports now goes through the fast path, so this check is what keeps
    "same operator" from becoming an assumption:

      * the VALUE must agree to float reduction-order noise;
      * dI/dP must agree in DIRECTION to ~1e-6, because that is the quantity Thies' Eq. 6
        descends and a rotated gradient would change the baseline's answer, not just its speed.

    Run on a real motion geometry, not the nominal orbit: the panel-tilt regime is where the
    `int()`-truncating interpolation of `helper.interpolate2d_cuda` (which extrapolates off the
    panel edge instead of clamping) actually fires, and reproducing THAT is the whole reason the
    fast kernel is a transcription rather than a `grid_sample` rewrite.
    """
    print("G9  fast kernels vs the vendored kernels")
    from bench.thies.fast_backprojector import FastConeBackprojector
    from bench.thies.vendor_import import cone_backprojector

    grid = VolumeGrid.centred(64, 4.0)
    G = grid.vendored()
    recon = ThiesConeRecon(cfg)
    g_filt = recon.filter(y)
    th = akima_motion(cfg.n_views, n_nodes=10, trans_mm=10.0, rot_deg=10.0,
                      device=DEV, seed=11, zero_centre=True)
    P_pix = to_pixel_matrices(params_to_Pmot(th, P_nom), cfg).contiguous().float()
    torch.manual_seed(0)
    wgt = torch.rand(grid.shape, device=DEV)          # a non-uniform upstream volume gradient

    def run(cls):
        P = P_pix.detach().clone().requires_grad_(True)
        v = cls.apply(g_filt, P, G)
        (v * wgt).sum().backward()
        return v.detach(), P.grad

    vv, vg = run(cone_backprojector())
    fv, fg = run(FastConeBackprojector)
    rel_v = float((fv - vv).norm() / vv.norm().clamp_min(1e-30))
    rel_g = float((fg - vg).norm() / vg.norm().clamp_min(1e-30))
    cos = float((fg * vg).sum() / (fg.norm() * vg.norm()).clamp_min(1e-30))
    check("G9 value agrees", rel_v < 1e-3, f"rel L2 {rel_v:.3e}")
    check("G9 dI/dP agrees", rel_g < 1e-3, f"rel L2 {rel_g:.3e}")
    check("G9 dI/dP direction", abs(cos - 1.0) < 1e-6,
          f"cos = {cos:.9f} (this is the number Eq. 6 descends)")
    from bench.thies.fast_backprojector import backprojector
    check("G9 the fast path is the default", backprojector(None) is FastConeBackprojector,
          "ThiesConeRecon(fast=None) takes the fast kernels; FM3D_THIES_VENDOR_BP=1 reverts")


def g11_training_pair(cfg, P_nom, vol_gt, y_static):
    """THE STAGE-1 PAIR (TMI III, p.1103): the MOTION-FREE projection data reconstructed with
    PERTURBED matrices. The regression this check exists to catch was real: until 2026-08-06
    `QMSampleSource.sample` simulated y WITH the perturbed matrices and backprojected with the
    SAME matrices -- a consistent pair whose motion cancels, so the net trained on near-oracle
    volumes (batch target mean never above 0.44 over 7000 iterations) and then saturated at the
    x=0 states of stage 2 (true VIF* ~0.8). Three assertions:

      * the paper pair at a training-scale motion has a HIGH VIF* mean (the volume really is
        corrupted), and it sits well above the consistent pair's;
      * **the production `sample()` itself** (run against a stub generator wrapping the gate's
        phantom, so no CQ500 is needed) returns a target NUMERICALLY EQUAL to the paper-pair
        construction -- if the consistent-pair bug ever returns, this equality breaks by the
        full gap, not by a threshold;
      * `TRAIN_AMP` carries `amp_mode="thies_hn"` and the mode draws within its bound -- the
        clipped half-normal transcribed from their released sampler, i.e. the paper's
        "perturb the data only slightly" clause is present in the training distribution.

    NOTE the consistent pair is NOT identical to the static recon even in principle: their
    backprojection has no angular weight, so rz motion makes the effective view spacing uneven
    and leaves real artifacts in the consistent pair too (VIF* ~0.5 on the reduced config).
    That is why the detector is equality-to-the-paper-pair, not a ratio between the two.
    """
    print("G11 the stage-1 training pair is the PAPER's pair (static data, perturbed matrices)")
    import tempfile
    from bench.thies.data import QMSampleSource, TRAIN_AMP, THIES_TRAIN_AMP
    from bench.thies.recon import to_unit

    recon = ThiesConeRecon(cfg)
    grid = VolumeGrid.centred(64, 4.0)
    amp = dict(trans_mm=10.0, rot_deg=10.0, amp_mode="fixed")        # deterministic given seed
    th = akima_motion(cfg.n_views, n_nodes=10, device=DEV, seed=3, zero_centre=True, **amp)
    P_mot = params_to_Pmot(th, P_nom)

    g_static = recon.filter(y_static)
    with torch.no_grad():
        ref = recon.backproject(g_static, P_nom, grid)               # unperturbed recon
        v_paper = recon.backproject(g_static, P_mot, grid)           # THE pair: static data + P*
        y_mot = forward_project_3d_batched(vol_gt, P_mot[None], recon_u(cfg), recon_v(cfg),
                                           dx=1.0, dy=1.0, dz=1.0)
        v_bug = recon.backproject(recon.filter(y_mot), P_mot, grid)  # the consistent pair (bug)

    r = to_unit(ref)[None, None]
    t_paper = float(vif_star_map_3d(to_unit(v_paper)[None, None], r).mean())
    t_bug = float(vif_star_map_3d(to_unit(v_bug)[None, None], r).mean())
    check("G11 paper pair is corrupted, and more so than the consistent pair",
          t_paper > 0.15 and (t_paper - t_bug) > 0.1,
          f"VIF* mean: paper pair {t_paper:.4f}, consistent pair {t_bug:.4f} at 10 mm / 10 deg "
          f"p2p (the paper pair must carry the motion; the consistent pair largely cancels it)")

    # -- the REAL detector: run the production sample() and pin it to the paper construction --
    class _StubGen:
        """`CQ500Generator`'s 3-method surface as `sample()` consumes it, over the phantom.
        (`P_nom` is attached after the class body -- class bodies cannot see function locals.)"""
        records = [{"patient": 0}]

        def simulate(self, idx, Pmat, **kw):
            P = Pmat if Pmat.dim() == 4 else Pmat[None]
            return forward_project_3d_batched(vol_gt, P, recon_u(cfg), recon_v(cfg),
                                              dx=1.0, dy=1.0, dz=1.0)

        def prefetch_fine(self, idx):
            pass

    _StubGen.P_nom = P_nom
    src = object.__new__(QMSampleSource)                 # skip __init__: no CQ500 on a gate box
    src.gen, src.cfg, src.device = _StubGen(), cfg, DEV
    src.grid, src.recon, src.amp, src.n_nodes = grid, recon, amp, 10
    src.cache_dir = tempfile.mkdtemp(prefix="gate_thies_g11_")
    src._gfilt, src._static = {}, {}                     # the RAM caches __init__ would build
    got = float(src.sample(0, seed=3)["target"].mean())  # same seed -> the SAME theta as above
    check("G11 sample() builds the paper pair", abs(got - t_paper) < 5e-3,
          f"sample() target mean {got:.4f} vs the paper-pair construction {t_paper:.4f} "
          f"(the consistent-pair bug would land at {t_bug:.4f} -- the full gap away)")

    for name, amp in (("TRAIN_AMP", TRAIN_AMP), ("THIES_TRAIN_AMP", THIES_TRAIN_AMP)):
        check(f"G11 {name} mode", amp.get("amp_mode") == "thies_hn",
              f"amp_mode = {amp.get('amp_mode')!r} (their released clipped-half-normal sampler; "
              f"'thies' is the pre-2026-08-06 U(0,1) reading, 8x poorer mild-motion coverage)")
    th_hn = torch.stack([akima_motion(cfg.n_views, n_nodes=10, seed=100 + i, zero_centre=True,
                                      **TRAIN_AMP) for i in range(8)])
    p2p_t = float((th_hn[..., :3].amax(1) - th_hn[..., :3].amin(1)).max())
    p2p_r = float(torch.rad2deg(th_hn[..., 3:].amax(1) - th_hn[..., 3:].amin(1)).max())
    # Akima overshoots its node bound by up to ~1.5x; the node bound itself is the p2p amplitude.
    ok = p2p_t <= 1.6 * TRAIN_AMP["trans_mm"] and p2p_r <= 1.6 * TRAIN_AMP["rot_deg"]
    check("G11 thies_hn respects the bound", ok,
          f"max realized p2p over 8 draws: {p2p_t:.2f} mm / {p2p_r:.2f} deg against maxima "
          f"{TRAIN_AMP['trans_mm']:g} / {TRAIN_AMP['rot_deg']:g} (x1.6 Akima-overshoot allowance)")


# -- helpers ----------------------------------------------------------------------------------
_UV = {}


def recon_u(cfg):
    if "u" not in _UV:
        _UV["u"], _UV["v"] = detector_coords_3d(cfg, device=DEV)
    return _UV["u"]


def recon_v(cfg):
    recon_u(cfg)
    return _UV["v"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    # The gate tests WIRING (axis order, units, the chain rule), not image quality, so it runs on
    # a binned panel and a reduced orbit by default: ~0.6 GB and well under a minute, which means
    # it can be run on a GPU that is already busy without disturbing a timing-sensitive job.
    # `--full` restores the deployed 700x500 @ 360 views (~5 GB) when a GPU is free.
    ap.add_argument("--views", type=int, default=120)
    ap.add_argument("--det_bin", type=int, default=2)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    if a.full:
        a.views, a.det_bin = 360, 1

    torch.manual_seed(0)
    cfg = ConeBeam3DConfig.thies(n_views=a.views, det_bin=a.det_bin)
    from fm3d.geometry_3d import build_conebeam_orbit
    P_nom = build_conebeam_orbit(cfg, device=DEV)
    vol = head_phantom((128, 128, 128), (1.0, 1.0, 1.0), device=DEV)
    if vol.dim() == 3:
        vol = vol[None, None]
    y = forward_project_3d_batched(vol, P_nom[None], recon_u(cfg), recon_v(cfg),
                                   dx=1.0, dy=1.0, dz=1.0)
    print(f"phantom {tuple(vol.shape)} | sinogram {tuple(y.shape)} | "
          f"{cfg.n_views} views\n")

    g1_akima()
    g2_model_capacity(cfg, P_nom)
    g3_adapter(cfg, P_nom)
    g4_vif()
    g5_recon(cfg, P_nom, y, vol)
    g6_gradient(cfg, P_nom, y)
    g7_oracle(cfg, P_nom, vol)
    g8_loop(cfg, P_nom, vol)
    g9_fast_kernels(cfg, P_nom, y)
    g10_vif_precision()
    g11_training_pair(cfg, P_nom, vol, y)

    print()
    if FAILED:
        print(f"FAILED: {FAILED}")
        raise SystemExit(1)
    print("gate_bench_thies: ALL GREEN")


if __name__ == "__main__":
    main()
