"""Gate for THE operator: LEAP modular-beam forward + backprojection, and its gradients
(volume grad = LEAP's VD backprojection; geometry grad = ours, of LEAP's own model).

The pair became LEAP's on 2026-07-29 (`fm3d/leap_projector.py`) and its modular forward was
PINNED TO JOSEPH on 2026-07-30 (`FORCE_JOSEPH`, via our patch to the vendored library --
refs/LEAP/FM3D_PATCH.md). Each consequence is a test here:

  T1  THE FORWARD IS A DIFFERENT MODEL -- and since the pin it is a RAY-DRIVEN one, the same
      model class as the `grid_sample` ray march, so T1's old expectation is INVERTED: LEAP
      must now sit closer to the ray march than to our retired SF kernel. Backwards would mean
      the pin is not in force or the geometry mapping into `set_modularbeam` is wrong.

  T1b THE PIN AND THE TRANSCRIPTION, neither of which FD can pin (secant noise ~1e-2).
      T1b-i is the counterfactual that catches a fresh env silently reverting the operator:
      on the NOMINAL orbit stock LEAP would run SF (`stock_kernel_kind`), so the deployed
      library matching our JOSEPH transcription and not our SF one proves the patch is live.
      T1b-ii then checks the transcribed VALUE against `leap_project` on nominal and moved
      geometry. Machine-precision validation of the gradient chains lives in
      `scripts/dev_leap_ref_autograd.py` (autograd referee; read its kink-trap note first).

  T2  THE PAIR IS NOT MATCHED. `ADJOINT_MODE = "VD"` is not the transpose of LEAP's forward,
      and since the Joseph pin neither backprojector is (the 'SF' one used to be, back when
      the forward was SF too). T2 MEASURES the defect instead of asserting it away, and fails
      only if it drifts far past what was measured, which would mean a geometry or units bug
      rather than the known model gap.

  T3/T5  THE GEOMETRY GRADIENT is `triton_leap_grad.leap_grad_P`, the EXACT gradient of the
      Joseph kernel -- one kernel, so one gradient, and no surrogate branch any more (the
      SF-ripple hazard that forced one is recorded in `triton_leap_grad`'s docstring). Both
      central-difference the LEAP loss itself in theta; T3 at a small tilt, T5 at a large one
      (where stock LEAP used to switch kernels mid-fit).

  T4  END TO END through `params_to_Pmot`: the same check at the level the estimator uses,
      plus the volume gradient against the reference pair's exact autograd adjoint.

    CUDA_VISIBLE_DEVICES=1 python scripts/gate_leap_projector.py       # ~60 s, no dataset
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.geometry_3d import ConeBeam3DConfig, build_conebeam_orbit, detector_coords_3d
from fm3d.leap_projector import (ADJOINT_MODE, kernel_kind, leap_backproject, leap_project,
                                 leap_project_3d_batched, sf_branch, stock_kernel_kind)
from fm3d.triton_leap_grad import leap_forward_model
from fm3d.phantom import head_phantom
from fm3d.projector_3d import (adjoint_project_3d_batched, forward_project_3d_batched,
                               reference_adjoint_3d_batched, reference_project_3d_batched)
from fm3d.rigid_motion import params_to_Pmot
from fm3d.triton_sf import sf_project_3d_batched

DEV = "cuda"
# THE PRODUCTION REGIME, shrunk only in view count and volume side. The panel pitch and the
# voxel size are the deployed ones on purpose: LEAP silently swaps its SF kernel for a Joseph
# ray-driven one when the voxel is finer than the native grid (`leap_projector.sf_branch`), so
# a gate on a coarse panel would measure a model that production never runs.
# The VOLUME SIZE is part of the regime too, not just the voxel: on a box small enough to clip
# the phantom the boundary voxels dominate the residual and the two models disagree 20x more
# (measured 2026-07-29: LEAP vs our SF kernel 5.8e-2 at 96^3, 5.1e-3 at 192^3, 2.9e-3 at 256^3,
# with motion changing nothing). 256^3 @ 1 mm is what the pipeline reconstructs on.
CFG = ConeBeam3DConfig.thies(n_views=12)
SHAPE = (256, 256, 256)
SPACING = (1.0, 1.0, 1.0)            # (dz, dy, dx); dx == dy is required by LEAP

_FAILED = []


def check(name, ok, detail=""):
    print(f"     [{'PASS' if ok else 'FAIL'}] {name}   {detail}", flush=True)
    if not ok:
        _FAILED.append(name)


def cos(a, b):
    return float((a.flatten() @ b.flatten()) / (a.norm() * b.norm() + 1e-30))


def gauss_blur3(x, sigma_vox):
    """Separable Gaussian blur. T3 needs a C1 phantom for the SAME reason gate_geometry's G4a
    does: trilinear/footprint interpolation makes the projection only C0 in the sample
    coordinates, so on a SHARP phantom a finite difference secants across voxel-face kinks and
    a single small component can miss the analytic gradient by 10% with nothing wrong."""
    r = int(3 * sigma_vox)
    k = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (k / sigma_vox) ** 2)
    k = k / k.sum()
    y = x[None, None]
    for d in range(3):
        shape = [1, 1, 1, 1, 1]; shape[2 + d] = k.numel()
        pad = tuple(k.numel() // 2 if i == d else 0 for i in range(3))
        y = torch.nn.functional.conv3d(y, k.view(shape), padding=pad)
    return y[0, 0]


def main():
    dz, dy, dx = SPACING
    D, H, W = SHAPE
    P_nom = build_conebeam_orbit(CFG, device=DEV)
    u, v = detector_coords_3d(CFG, device=DEV)
    vol = head_phantom(SHAPE, SPACING, device=DEV)
    kw = dict(dx=dx, dy=dy, dz=dz)
    rkw = dict(**kw, n_samples=384, view_chunk=1, row_chunk=25)
    torch.manual_seed(0)
    th_true = torch.randn(CFG.n_views, 6, device=DEV) * torch.tensor(
        [2., 2., 2., .02, .02, .02], device=DEV)
    P = params_to_Pmot(0.5 * th_true, P_nom)[None]           # a MOVED geometry throughout
    uses_sf, detail = sf_branch(P[0], du=float(u[1] - u[0]), dv=float(v[1] - v[0]),
                                dx=dx, dz=dz, D=D, H=H, W=W)
    print(f"geometry: {CFG.nv}x{CFG.nu} @ {CFG.du:g} mm, {CFG.n_views} views | "
          f"volume {SHAPE} @ {SPACING} mm | adjoint mode {ADJOINT_MODE}")
    print(f"voxel-size condition (the OTHER half of stock `useSF`): {uses_sf} -- {detail}")

    # ---- T1 -----------------------------------------------------------------------------
    # The deployed forward is JOSEPH everywhere now, i.e. a RAY-DRIVEN, trilinear-basis model
    # -- the same model CLASS as the grid_sample ray march. So the expectation that held while
    # LEAP ran SF is INVERTED on purpose: LEAP must now sit closer to the ray march than to
    # our (retired) SF kernel. Getting this backwards would mean the pin is not in force.
    print("\nT1  forward: LEAP vs the ray-march reference vs our retired SF kernel (moved)")
    with torch.no_grad():
        g_leap = forward_project_3d_batched(vol[None, None], P, u, v, **kw)
        g_sf = sf_project_3d_batched(vol[None, None], P, u, v, **kw)
        g_ray = reference_project_3d_batched(vol[None, None], P, u, v, **rkw)
    r_ls = float((g_leap - g_sf).norm() / g_sf.norm())
    r_lr = float((g_leap - g_ray).norm() / g_ray.norm())
    r_sr = float((g_sf - g_ray).norm() / g_ray.norm())
    check("LEAP vs the ray march", r_lr < 2.0e-2, f"rel = {r_lr:.2e}   (same model class)")
    check("LEAP vs our retired SF kernel", r_ls < 5.0e-2, f"rel = {r_ls:.2e}")
    check("LEAP now closer to the RAY MARCH than to SF", r_lr < r_ls,
          f"{r_lr:.2e} vs LEAP-SF {r_ls:.2e}  (SF-ray {r_sr:.2e}) -- inverted by the Joseph pin")

    # ---- T1b ----------------------------------------------------------------------------
    # The strongest transcription test: the VALUE of the re-implemented kernels against the
    # real LEAP forward, per branch. Bars are tex3D's 9-bit lerp quantization plus fp32
    # accumulation (measured 1.7e-5 SF / 4.3e-5 JOSEPH at adoption).
    print("\nT1b  transcription parity: triton_leap_grad's LEAP-model kernels vs LEAP itself")
    kwf = dict(nv=CFG.nv, nu=CFG.nu, dx=dx, dy=dy, dz=dz, du=float(u[1] - u[0]),
               dv=float(v[1] - v[0]), u0=float(u[0] + u[-1]) * 0.5,
               v_off=float(v[0] + v[-1]) * 0.5)
    # T1b-i THE PATCH IS ACTIVE. The nominal orbit is the counterfactual that proves it:
    # STOCK LEAP would run SF there (`stock_kernel_kind`), so if the deployed library's output
    # matches our JOSEPH transcription and not our SF one, the pin is in force. This is the
    # test that catches a fresh env / a reinstalled leapct silently reverting the operator.
    with torch.no_grad():
        g_nom = leap_project(vol[None], P_nom[None], **{k: kwf[k] for k in
                             ("nv", "nu", "dx", "dy", "dz", "du", "dv", "u0", "v_off")})
        m_sf = leap_forward_model(vol[None], P_nom[None], **kwf, kind="SF")
        m_jo = leap_forward_model(vol[None], P_nom[None], **kwf, kind="JOSEPH")
    r_s = float((m_sf - g_nom).norm() / g_nom.norm())
    r_j = float((m_jo - g_nom).norm() / g_nom.norm())
    check("stock LEAP would have picked SF on the nominal orbit",
          stock_kernel_kind(P_nom, nv=CFG.nv, du=kwf["du"], dv=kwf["dv"], dx=dx, dz=dz,
                            D=D, H=H, W=W) == "SF", "(the counterfactual this test needs)")
    check("FORCE_JOSEPH is ACTIVE (deployed lib runs Joseph there anyway)", r_j < r_s / 10,
          f"vs JOSEPH-model {r_j:.2e}  vs SF-model {r_s:.2e}  -- an unpatched .so flips these")

    # T1b-ii the transcription itself, on the geometry production actually runs (Joseph),
    # nominal AND moved.
    check("JOSEPH transcription, nominal orbit", r_j < 2e-4, f"rel = {r_j:.2e}")
    kk = kernel_kind(P[0], nv=CFG.nv, du=kwf["du"], dv=kwf["dv"], dx=dx, dz=dz,
                     D=D, H=H, W=W)
    check("moved geometry is Joseph too (one kernel, always)", kk == "JOSEPH", f"kind = {kk}")
    with torch.no_grad():
        g_model = leap_forward_model(vol[None], P, **kwf)
    r_mv = float((g_model - g_leap).norm() / g_leap.norm())
    check("JOSEPH transcription, moved geometry", r_mv < 2e-4, f"rel = {r_mv:.2e}")

    # ---- T2 -----------------------------------------------------------------------------
    # The adjointness DEFECT DEPENDS ON THE PROBE, so both probes are measured. A real sinogram
    # is what the CG data step feeds `A^T`; white noise is the worst case and sits an order of
    # magnitude higher, because the operators' voxel bases differ most at high frequency.
    # BARS RECALIBRATED 2026-07-30 with the Joseph pin. While the forward was SF, LEAP's 'SF'
    # BACKprojector was nearly its exact transpose (1.9e-7) and served as the "this is what
    # matched looks like" reference. Against a JOSEPH forward neither backprojector is the
    # transpose, so both arms now just measure a model gap and get the same generous bars --
    # 'SF' is no longer the tight reference, and asserting the old 1e-4/1e-3 on it was
    # asserting a relationship that no longer exists.
    print("\nT2  adjointness of the deployed pair (measured, not asserted)")
    bars = {("sinogram", "VD"): 2e-3, ("sinogram", "SF"): 2e-3,
            ("white noise", "VD"): 5e-2, ("white noise", "SF"): 5e-3}
    probes_s = {"sinogram": g_leap, "white noise": torch.randn_like(g_leap)}
    for pname, sp in probes_s.items():
        lhs = float((g_leap * sp).sum())
        for mode in ("VD", "SF"):
            with torch.no_grad():
                ATs = leap_backproject(sp, P, D=D, H=H, W=W, dx=dx, dy=dy, dz=dz,
                                       du=float(u[1] - u[0]), dv=float(v[1] - v[0]),
                                       u0=float(u[0] + u[-1]) * 0.5,
                                       v_off=float(v[0] + v[-1]) * 0.5, mode=mode)
            rel = abs(lhs - float((vol[None] * ATs).sum())) / max(abs(lhs), 1e-20)
            bar = bars[(pname, mode)]
            check(f"adjointness [{mode}] on a {pname}"
                  f"{' (DEPLOYED)' if mode == ADJOINT_MODE else ''}",
                  rel < bar, f"rel = {rel:.2e}   (bar {bar:.0e})")
    # Direction check on a REAL sinogram: white noise would compare the two voxel bases, not
    # the two backprojections.
    with torch.no_grad():
        bp_dep = adjoint_project_3d_batched(g_leap, P, u, v, D=D, H=H, W=W, **kw)[:, 0]
    # NOT under no_grad: the reference adjoint IS an autograd backward pass.
    bp_ref = reference_adjoint_3d_batched(g_leap, P, u, v, D=D, H=H, W=W, **rkw)[:, 0]
    check("backprojection vs the reference adjoint", cos(bp_dep, bp_ref) > 0.99,
          f"cos = {cos(bp_dep, bp_ref):.5f}  |dep|/|ref| = "
          f"{float(bp_dep.norm() / bp_ref.norm()):.4f}")

    # ---- T3 -----------------------------------------------------------------------------
    # THE POINT OF THIS GATE. The value is LEAP's, the theta gradient is our SF kernel's, and
    # the finite difference below is taken through LEAP's own forward -- so a passing T3 says
    # the borrowed gradient really is a descent direction for the deployed loss.
    print("\nT3  d(loss)/d(theta): our analytic dP against a finite difference OF THE LEAP LOSS")
    volc = gauss_blur3(vol, 2.0)                    # C1 phantom -- see gauss_blur3
    with torch.no_grad():
        y = leap_project(volc[None], params_to_Pmot(th_true, P_nom)[None],
                         nv=CFG.nv, nu=CFG.nu, dx=dx, dy=dy, dz=dz,
                         du=float(u[1] - u[0]), dv=float(v[1] - v[0]),
                         u0=float(u[0] + u[-1]) * 0.5, v_off=float(v[0] + v[-1]) * 0.5)

    def loss_at(th):
        pred = leap_project_3d_batched(volc[None, None], params_to_Pmot(th, P_nom)[None],
                                       u, v, **kw)
        return 0.5 * ((pred - y) ** 2).mean()

    th = (0.5 * th_true).detach().clone().requires_grad_(True)
    loss_at(th).backward()
    g_auto = th.grad.clone()
    probes = [(0, 0), (5, 1), (11, 3), (7, 5)]
    eps = {0: 5e-2, 1: 5e-2, 2: 5e-2, 3: 5e-3, 4: 5e-3, 5: 5e-3}
    g_fd = torch.zeros_like(g_auto)
    with torch.no_grad():
        for iv, d in probes:
            e = eps[d]
            tp = (0.5 * th_true).clone(); tp[iv, d] += e
            tm = (0.5 * th_true).clone(); tm[iv, d] -= e
            g_fd[iv, d] = (loss_at(tp) - loss_at(tm)) / (2 * e)
    for iv, d in probes:
        a, f = g_auto[iv, d].item(), g_fd[iv, d].item()
        rel = abs(a - f) / max(abs(f), 1e-12)
        check(f"grad view{iv} dof{d}", rel < 0.10,
              f"analytic {a:+.4e}  fd(LEAP) {f:+.4e}  rel {rel:.1e}")

    # ---- T4 -----------------------------------------------------------------------------
    print("\nT4  the same gradient as a DIRECTION, and the volume gradient")
    a = torch.stack([g_auto[iv, d] for iv, d in probes])
    f = torch.stack([g_fd[iv, d] for iv, d in probes])
    check("probe direction", cos(a, f) > 0.99, f"cos = {cos(a, f):.5f}")
    x = vol[None, None].detach().clone().requires_grad_(True)
    (forward_project_3d_batched(x, P, u, v, **kw) * g_leap).sum().backward()
    gv = x.grad[:, 0]
    check("volume gradient vs the reference adjoint", cos(gv, bp_ref) > 0.99,
          f"cos = {cos(gv, bp_ref):.5f}")

    # ---- T5 -----------------------------------------------------------------------------
    # The SAME FD-of-the-LEAP-loss check, but with the eval point INSIDE the Joseph regime
    # (one view tilted ~8.6 deg, well past the 5.06 deg flip so the +-eps probes stay on one
    # model). This is the regime the old borrowed gradient never covered.
    print("\nT5  d(loss)/d(theta) at a LARGE tilt (where stock LEAP used to switch kernels)")
    th_true5 = th_true.clone()
    th_true5[3, 3] += 0.30
    with torch.no_grad():
        y5 = leap_project(volc[None], params_to_Pmot(th_true5, P_nom)[None],
                          nv=CFG.nv, nu=CFG.nu, dx=dx, dy=dy, dz=dz,
                          du=kwf["du"], dv=kwf["dv"], u0=kwf["u0"], v_off=kwf["v_off"])

    def loss5_at(th):
        pred = leap_project_3d_batched(volc[None, None], params_to_Pmot(th, P_nom)[None],
                                       u, v, **kw)
        return 0.5 * ((pred - y5) ** 2).mean()

    th5_0 = (0.5 * th_true5).detach().clone()
    kk5 = kernel_kind(params_to_Pmot(th5_0, P_nom), nv=CFG.nv, du=kwf["du"], dv=kwf["dv"],
                      dx=dx, dz=dz, D=D, H=H, W=W)
    check("the T5 eval point is Joseph (as everything now is)", kk5 == "JOSEPH",
          f"kind = {kk5}")
    th5 = th5_0.clone().requires_grad_(True)
    loss5_at(th5).backward()
    g5 = th5.grad.clone()
    probes5 = [(3, 3), (3, 4), (0, 0), (7, 5)]
    with torch.no_grad():
        for iv, d in probes5:
            e = eps[d]
            tp = th5_0.clone(); tp[iv, d] += e
            tm = th5_0.clone(); tm[iv, d] -= e
            fd = float((loss5_at(tp) - loss5_at(tm)) / (2 * e))
            a = g5[iv, d].item()
            rel = abs(a - fd) / max(abs(fd), 1e-12)
            check(f"grad view{iv} dof{d} (large tilt)", rel < 0.10,
                  f"analytic {a:+.4e}  fd(LEAP) {fd:+.4e}  rel {rel:.1e}")

    print("\n" + ("ALL GATES PASS" if not _FAILED else f"FAILED: {_FAILED}"))
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
