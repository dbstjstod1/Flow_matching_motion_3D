"""Milestone-1 gates: does per-view rigid motion enter the 3D cone-beam geometry correctly?

Runs with no dataset and no trained model. Every gate is an INVARIANT, not a quality score --
each one fails loudly on a bug class that would otherwise stay silent:

  G1  so(3) algebra          exp/log roundtrip, orthonormality, finite gradient at w=0,
                             and the one-parameter-subgroup law exp((a+b)w) = exp(aw)exp(bw)
                             -- that law IS the geometry bridge's licence to interpolate s*theta.
  G2  identity               theta = 0 leaves P_nom bit-for-bit unchanged (zero-init estimator
                             must start exactly at the nominal orbit).
  G3  object-space semantics P_nom @ T(theta) projects the object MOVED BY T. Checked against
                             exact array operations (integer-voxel roll; 90-deg rotate by
                             transpose+flip), so a flipped translation sign or a left-handed
                             rotation cannot slip through. A sign error here does not crash --
                             it mirrors the reconstruction.
  G4  differentiability      autograd d/dtheta of the projection loss vs central differences.
                             Motion estimation is nothing but this gradient.
  G5  reconstruction         static FDK recovers the phantom; motion corrupts it; correcting
                             the geometry with the TRUE theta recovers it again. If the third
                             one does not come back, the forward model and the reconstruction
                             disagree and no amount of estimation will save it.
  G6  bridge monotonicity    FDK(y, P_nom @ T(t*theta)) improves monotonically in t. This is
                             the path the flow-matching prior will be trained to travel.

Writes a montage to data/gates/ -- read it. G5/G6 pass a numeric threshold, but the failure
mode that matters (a mirrored or sheared reconstruction that still scores well) is only
visible by eye.

    python scripts/gate_geometry.py
"""

from __future__ import annotations

import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.geometry_3d import (ConeBeam3DConfig, build_conebeam_orbit, detector_coords_3d,
                              measured_region_mask)
from fm3d.phantom import MU_WATER, head_phantom
from fm3d.projector_3d import fdk_conebeam_3d_batched, forward_project_3d_batched
from fm3d.filters import calibrate_scale
from fm3d.rigid_motion import (apply_rigid_motion, make_motion, params_to_Pmot,
                               rigid_motion_matrices, so3_exp, so3_log)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gates")

# A small but honest geometry: full-fan Varian OBI (head-and-neck protocol -- no half-fan,
# the head fits the 26 cm FOV), 4x4-binned panel and 180 views to keep the gate under a minute.
CFG = ConeBeam3DConfig(det_bin=4, n_views=180)
SHAPE = (96, 128, 128)          # (D,H,W) = (z,y,x)
SPACING = (1.5, 1.5, 1.5)       # mm -> 144 x 192 x 192 mm, inside the 262 mm FOV / 199 mm cone

_fails: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name:38s} {detail}")
    if not ok:
        _fails.append(name)


def psnr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, peak: float) -> float:
    mse = ((a - b) ** 2)[mask].mean()
    return float(10 * torch.log10(peak ** 2 / mse.clamp_min(1e-20)))


# ======================================================================================
# G1 -- so(3) algebra
# ======================================================================================
def gate_so3() -> None:
    print("\nG1  so(3) algebra")
    torch.manual_seed(0)
    w = torch.randn(64, 3, device=DEV) * 0.3
    R = so3_exp(w)

    eye = torch.eye(3, device=DEV).expand_as(R)
    orth = (R @ R.transpose(-1, -2) - eye).abs().max()
    det = torch.linalg.det(R)
    check("R orthonormal", orth < 1e-5, f"max|RR^T - I| = {orth:.2e}")
    check("det(R) = +1", (det - 1).abs().max() < 1e-5, f"max|det - 1| = {(det - 1).abs().max():.2e}")

    rt = (so3_log(R) - w).abs().max()
    check("log(exp(w)) = w", rt < 1e-5, f"max err = {rt:.2e}")

    e0 = (so3_exp(torch.zeros(1, 3, device=DEV)) - torch.eye(3, device=DEV)).abs().max()
    check("exp(0) = I exactly", e0 == 0, f"max|exp(0) - I| = {e0:.2e}")

    # gradient at the origin: the naive Rodrigues formula divides by |w| and yields nan here.
    # The estimator is zero-initialized, so a nan here means motion estimation never starts.
    w0 = torch.zeros(1, 3, device=DEV, requires_grad=True)
    so3_exp(w0).sum().backward()
    g0 = w0.grad
    check("grad finite at w = 0", bool(torch.isfinite(g0).all()), f"grad = {g0.detach().cpu().numpy().ravel()}")

    # one-parameter subgroup: exp((a+b)w) == exp(aw) @ exp(bw).  THE bridge's licence.
    a, b = 0.31, 0.44
    lhs = so3_exp((a + b) * w)
    rhs = so3_exp(a * w) @ so3_exp(b * w)
    e = (lhs - rhs).abs().max()
    check("exp((a+b)w) = exp(aw)exp(bw)", e < 1e-5, f"max err = {e:.2e}  (s*theta is a geodesic)")


# ======================================================================================
# G2 -- identity
# ======================================================================================
def gate_identity(P_nom: torch.Tensor) -> None:
    print("\nG2  identity at theta = 0")
    V = P_nom.shape[0]
    T0 = rigid_motion_matrices(torch.zeros(V, 6, device=DEV))
    e_T = (T0 - torch.eye(4, device=DEV)).abs().max()
    check("T(0) = I exactly", e_T == 0, f"max|T(0) - I| = {e_T:.2e}")

    e_P = (params_to_Pmot(torch.zeros(V, 6, device=DEV), P_nom) - P_nom).abs().max()
    check("P_nom @ T(0) = P_nom", e_P == 0, f"max|dP| = {e_P:.2e}")


# ======================================================================================
# G3 -- object-space semantics (the sign convention)
# ======================================================================================
def gate_semantics(P_nom, u, v, vol) -> None:
    print("\nG3  object-space semantics  (P @ T projects the MOVED object)")
    dz, dy, dx = SPACING
    V = P_nom.shape[0]
    kw = dict(dx=dx, dy=dy, dz=dz, n_samples=192, view_chunk=8)

    def fp(x, P):
        return forward_project_3d_batched(x[None, None], P[None], u, v, **kw)[0]

    # --- translation. Roll the ARRAY by an integer number of voxels (exact, no interpolation)
    # and ask the motion parameters to reproduce it. new[i] = old[i - k]  <=>  the object moved
    # by +k voxels, i.e. by +k*dx mm in world x. So theta_t = (+k*dx, 0, 0).
    k = 6
    vol_shift = torch.roll(vol, shifts=k, dims=2)            # dim 2 = x = W
    th = torch.zeros(V, 6, device=DEV)
    th[:, 0] = k * dx
    y_arr = fp(vol_shift, P_nom)
    y_geo = fp(vol, params_to_Pmot(th, P_nom))
    rel = ((y_arr - y_geo).norm() / y_arr.norm()).item()
    check("translate +x: array == geometry", rel < 2e-2,
          f"rel diff = {rel:.2e}   (sign flip would give ~{((y_arr - fp(vol, params_to_Pmot(-th, P_nom))).norm() / y_arr.norm()).item():.2e})")

    # --- rotation about z by +90 deg, done exactly by transpose+flip so there is no
    # interpolation error to hide behind (and the flip i -> N-1-i negates the coordinate
    # exactly, because voxel centres are (i - (N-1)/2)*d).
    #
    # A right-handed rotation by +90 about +z sends the POINT (x,y) -> (-y,x), so the rotated
    # VOLUME samples the original at the inverse: vol_rot(x,y) = vol(y,-x). In indices, with
    # dim1 = y = H and dim2 = x = W (and H == W here):  new[iy,ix] = old[H-1-ix, iy], i.e.
    # transpose y/x and then flip along x. Check it on a bead at +y: it must land at -x.
    vol_rot = torch.flip(vol.transpose(1, 2), dims=[2])
    th = torch.zeros(V, 6, device=DEV)
    th[:, 5] = math.pi / 2                                    # w = (0, 0, +pi/2)
    y_arr = fp(vol_rot, P_nom)
    y_pos = fp(vol, params_to_Pmot(th, P_nom))
    y_neg = fp(vol, params_to_Pmot(-th, P_nom))
    rel_p = ((y_arr - y_pos).norm() / y_arr.norm()).item()
    rel_n = ((y_arr - y_neg).norm() / y_arr.norm()).item()
    check("rotate +90 about z: handedness", rel_p < 2e-2 and rel_p < 0.2 * rel_n,
          f"rel diff = {rel_p:.2e}  (opposite sign: {rel_n:.2e})")


# ======================================================================================
# G4 -- differentiability in theta
# ======================================================================================
def _gauss_blur3(x: torch.Tensor, sigma_vox: float) -> torch.Tensor:
    """Separable Gaussian blur, used ONLY to build the C1 phantom that G4a needs."""
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


def gate_grad(P_nom, u, v, vol) -> None:
    """d(loss)/d(theta) -- the motion estimator IS this gradient, so it gets two gates.

    G4a  CORRECTNESS, on a Gaussian-smoothed phantom. Trilinear interpolation makes the
         projection only C0 in the sample coordinates: its derivative jumps at every voxel
         face. On a SHARP phantom the analytic gradient (which autograd computes exactly, for
         the DISCRETIZED operator) and a finite difference (which secants across many of those
         jumps, i.e. approximates the CONTINUOUS operator's derivative) legitimately differ,
         and the size of the gap depends on the ray sampling rate, not on the code being wrong.
         Smoothing the phantom removes the kinks and the two must then agree to well under 1%.
         That is the real invariant, and it is the one that would break if the chain
         theta -> T -> P -> A_inv -> rays -> grid_sample were mis-wired.

    G4b  ADEQUACY of the ray sampling, on the sharp phantom: does the descent DIRECTION stop
         moving when the rays are sampled twice as finely? MEASURED here, and the answer is
         reassuring: from n_samples 384 -> 768 the translation block moves 0.1% and the
         rotation block 0.6%, so 192 already resolves the direction the estimator descends.

         What is NOT stable is any single SMALL component. The (view 2, wz) entry shifts ~17%
         between n_samples 192 and 768 -- but it is only ~2% of the rotation block's norm, so
         it perturbs the direction by ~0.4% and does not matter. That is worth knowing mainly
         as a warning about how to test: probing one small component with finite differences on
         a sharp phantom will look like a failed gradient when nothing is wrong. Hence G4a.
    """
    print("\nG4  d(loss)/d(theta)")
    dz, dy, dx = SPACING
    Vs = 8                                     # a few views is enough and keeps FD cheap
    P = P_nom[:Vs]
    probes = [(0, 0), (3, 1), (5, 3), (2, 5)]                # (view, dof)
    eps = {0: 1e-2, 1: 1e-2, 2: 1e-2, 3: 1e-3, 4: 1e-3, 5: 1e-3}

    torch.manual_seed(0)
    th_true = torch.randn(Vs, 6, device=DEV) * torch.tensor([2., 2., 2., .02, .02, .02], device=DEV)

    def grad_and_fd(x, n_samples, do_fd=True):
        kw = dict(dx=dx, dy=dy, dz=dz, n_samples=n_samples, view_chunk=8, row_chunk=64)
        with torch.no_grad():
            y = forward_project_3d_batched(
                x[None, None], params_to_Pmot(th_true, P)[None], u, v, **kw)

        def loss_at(th):
            pred = forward_project_3d_batched(
                x[None, None], params_to_Pmot(th, P)[None], u, v, **kw)
            return 0.5 * ((pred - y) ** 2).mean()

        th = torch.zeros(Vs, 6, device=DEV, requires_grad=True)
        loss_at(th).backward()
        g_auto = th.grad.clone()
        if not do_fd:
            return g_auto, None
        g_fd = torch.zeros_like(g_auto)
        with torch.no_grad():
            for (iv, d) in probes:
                e = eps[d]
                tp = torch.zeros(Vs, 6, device=DEV); tp[iv, d] = e
                g_fd[iv, d] = (loss_at(tp) - loss_at(-tp)) / (2 * e)
        return g_auto, g_fd

    # --- G4a: correctness, kink-free phantom
    print("     G4a  autograd vs central differences on a C1 (smoothed) phantom")
    g_auto, g_fd = grad_and_fd(_gauss_blur3(vol, 2.0), 384)
    for (iv, d) in probes:
        a, f = g_auto[iv, d].item(), g_fd[iv, d].item()
        rel = abs(a - f) / max(abs(f), 1e-12)
        check(f"grad view{iv} dof{d}", rel < 2e-2, f"autograd {a:+.4e}  fd {f:+.4e}  rel {rel:.1e}")

    # --- G4b: ray-sampling adequacy on the sharp phantom
    # Translation and rotation are reported SEPARATELY because they carry different units
    # (1/mm vs 1/rad) and differ by ~an order of magnitude in size, so a single whole-tensor
    # norm just reports whichever block is bigger and tells you nothing about the other.
    print("     G4b  gradient stability vs ray sampling (sharp edges)")
    prev, drifts = None, {}
    for ns in [192, 384, 768]:
        g, _ = grad_and_fd(vol, ns, do_fd=False)
        if prev is not None:
            drifts[ns] = (
                float((g[:, :3] - prev[:, :3]).norm() / prev[:, :3].norm()),
                float((g[:, 3:] - prev[:, 3:]).norm() / prev[:, 3:].norm()),
            )
            d = f"trans {drifts[ns][0]:5.1%}  rot {drifts[ns][1]:5.1%}"
        else:
            d = "     -"
        print(f"          n_samples={ns:4d}   |g_trans| = {g[:, :3].norm().item():.3e}"
              f"   |g_rot| = {g[:, 3:].norm().item():.3e}   drift: {d}")
        prev = g
    # A converged descent direction is one that stops moving when the ray sampling is refined.
    # If 384 -> 768 still shifts the rotation gradient by >10%, the estimator is descending on
    # an aliasing artefact: raise n_samples (or coarsen the voxels).
    worst = max(drifts[768])
    check("ray sampling converged", worst < 0.10,
          f"384 -> 768: trans {drifts[768][0]:.1%}, rot {drifts[768][1]:.1%}"
          f"   (estimator n_samples: use >= 384 here)")


# ======================================================================================
# G5 / G6 -- reconstruction and the bridge path
# ======================================================================================
def gate_recon(P_nom, u, v, vol):
    print("\nG5  reconstruction: static / corrupted / corrected")
    D, H, W = SHAPE
    dz, dy, dx = SPACING
    fp_kw = dict(dx=dx, dy=dy, dz=dz, n_samples=256, view_chunk=4, row_chunk=64)
    fdk_kw = dict(D=D, H=H, W=W, dx=dx, dy=dy, dz=dz, view_chunk=8)

    mask = measured_region_mask(SHAPE, SPACING, CFG, device=DEV)
    peak = float(vol[mask].max())

    theta = make_motion("mixed", CFG.n_views, device=DEV, seed=3)
    tr = theta[:, :3].abs().max().item()
    ro = math.degrees(theta[:, 3:].norm(dim=-1).max().item())
    print(f"     motion: |t| <= {tr:.1f} mm, |rot| <= {ro:.2f} deg over {CFG.n_views} views")

    with torch.no_grad():
        y_static = forward_project_3d_batched(vol[None, None], P_nom[None], u, v, **fp_kw)
        y_moved = forward_project_3d_batched(
            vol[None, None], params_to_Pmot(theta, P_nom)[None], u, v, **fp_kw)

        # one scalar, calibrated once against the static case; it is an operator constant
        # (geometry + filter), not an image-dependent fudge -- reused for every recon below.
        raw = fdk_conebeam_3d_batched(y_static, P_nom[None], u, v, CFG, scale=1.0, **fdk_kw)[0]
        scale = calibrate_scale(raw, vol, mask)
        print(f"     FDK scale calibrated: {scale:.6g}")

        def fdk(y, P):
            return fdk_conebeam_3d_batched(y, P[None], u, v, CFG, scale=scale, **fdk_kw)[0]

        x_static = fdk(y_static, P_nom)                            # no motion at all
        x_uncorr = fdk(y_moved, P_nom)                             # motion, uncorrected  = t=0
        x_corr = fdk(y_moved, params_to_Pmot(theta, P_nom))        # motion, true theta    = t=1

    p_static = psnr(x_static, vol, mask, peak)
    p_uncorr = psnr(x_uncorr, vol, mask, peak)
    p_corr = psnr(x_corr, vol, mask, peak)
    print(f"     PSNR  static {p_static:5.2f} dB | uncorrected {p_uncorr:5.2f} dB | corrected {p_corr:5.2f} dB")

    check("static FDK recovers phantom", p_static > 25, f"{p_static:.2f} dB")
    check("motion actually corrupts", p_static - p_uncorr > 3, f"drop {p_static - p_uncorr:.2f} dB")
    check("true theta recovers it", p_static - p_corr < 1.0,
          f"gap to static {p_static - p_corr:+.2f} dB  (this is the bridge's t=1 endpoint)")

    print("\nG6  bridge path: FDK(y, P_nom @ T(t*theta)) monotone in t")
    ts = [0.0, 0.25, 0.5, 0.75, 1.0]
    path, ps = [], []
    with torch.no_grad():
        for t in ts:
            xt = fdk(y_moved, params_to_Pmot(t * theta, P_nom))
            path.append(xt)
            ps.append(psnr(xt, vol, mask, peak))
    print("     " + " | ".join(f"t={t:.2f}: {p:5.2f} dB" for t, p in zip(ts, ps)))
    mono = all(ps[i + 1] > ps[i] - 0.15 for i in range(len(ps) - 1))
    check("monotone along the bridge", mono and ps[-1] - ps[0] > 3,
          f"t=0 {ps[0]:.2f} -> t=1 {ps[-1]:.2f} dB")

    return vol, x_static, x_uncorr, x_corr, path, ts, ps


def montage(vol, x_static, x_uncorr, x_corr, path, ts, ps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT, exist_ok=True)
    D, H, W = SHAPE
    zc, yc = D // 2, H // 2
    lo, hi = 0.0, 1.4 * MU_WATER

    rows = [("ground truth", vol), ("static FDK", x_static),
            ("uncorrected  (bridge t=0)", x_uncorr), ("true theta   (bridge t=1)", x_corr)]
    fig, ax = plt.subplots(len(rows), 2, figsize=(7, 3.0 * len(rows)))
    for r, (name, x) in enumerate(rows):
        ax[r, 0].imshow(x[zc].cpu(), cmap="gray", vmin=lo, vmax=hi)
        ax[r, 0].set_ylabel(name, fontsize=9)
        ax[r, 1].imshow(x[:, yc].cpu(), cmap="gray", vmin=lo, vmax=hi, aspect="auto")
        for c in range(2):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    ax[0, 0].set_title("axial", fontsize=9); ax[0, 1].set_title("coronal", fontsize=9)
    fig.tight_layout()
    f1 = os.path.join(OUT, "gate_recon.png")
    fig.savefig(f1, dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, len(path), figsize=(3.0 * len(path), 3.3))
    for i, (t, x, p) in enumerate(zip(ts, path, ps)):
        ax[i].imshow(x[zc].cpu(), cmap="gray", vmin=lo, vmax=hi)
        ax[i].set_title(f"t = {t:.2f}\n{p:.2f} dB", fontsize=9)
        ax[i].set_xticks([]); ax[i].set_yticks([])
    fig.suptitle("geometry bridge: FDK(y, P_nom @ T(t * theta))", fontsize=10)
    fig.tight_layout()
    f2 = os.path.join(OUT, "gate_bridge.png")
    fig.savefig(f2, dpi=130); plt.close(fig)
    print(f"\n     montages: {f1}\n               {f2}")


def main() -> int:
    t0 = time.time()
    print(f"device {DEV} | {CFG.n_views} views | detector {CFG.nv}x{CFG.nu} @ {CFG.du:.3f} mm"
          f" | volume {SHAPE} @ {SPACING[0]} mm")
    print(f"FOV diameter {CFG.fov_diameter_mm():.1f} mm | axial coverage {CFG.axial_coverage_mm():.1f} mm")

    P_nom = build_conebeam_orbit(CFG, device=DEV)
    u, v = detector_coords_3d(CFG, device=DEV)
    vol = head_phantom(SHAPE, SPACING, device=DEV)

    gate_so3()
    gate_identity(P_nom)
    gate_semantics(P_nom, u, v, vol)
    gate_grad(P_nom, u, v, vol)
    out = gate_recon(P_nom, u, v, vol)
    montage(*out)

    print(f"\n{'=' * 72}")
    if _fails:
        print(f"FAILED {len(_fails)} gate(s): {', '.join(_fails)}")
    else:
        print(f"all gates passed  ({time.time() - t0:.1f}s)")
    print("=" * 72)
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
