"""Differentiable 3D cone-beam ray-marching forward projector + FDK (prompt 6).

Torch-native 3D extension of `forward_projector_2d.py` / `recon_2d.py` (the
allowed alternative to LEAP: same math, one codebase, and it shares the exact
pixel-center / align_corners=False conventions with warp.py, which is what
makes the MC-FBP "phi=Id == static FDK" identity hold to float precision).

Forward: for each detector element the ray is recovered from the projection
matrix (source C = -A^{-1} b, direction d = A^{-1} [u; v; 1]), then the volume
is line-integrated by ray-marching with trilinear `grid_sample`. Everything is
torch, so gradients flow back to the volume (and through warp.py to the DVF).

FDK (Feldkamp) reconstruction, kept geometrically consistent with the forward:
  1. cosine pre-weight   g~ = g * SDD / sqrt(SDD^2 + u^2 + v^2)
  2. ramp filter along the detector u-axis (FFT, optional Hann apodization)
  3. distance-weighted backprojection  += g_filt(u*, v*) / w^2
using the SAME P matrices (u = u_h/w, v = v_h/w, w = along-axis source
distance) — the exact transpose of the forward model, no convention drift.
A single global scale is calibrated once by least squares (calibrate_scale)
against the clean recon and re-used for motion recons, exactly as in 2D.

Memory note (prompt 6, problem #1 of 2): everything is view-chunked
(`view_chunk`) and the FDK additionally voxel-chunked (`vox_chunk`); this is
the "view chunking" half of the 3D memory strategy. The FM prior's volume-size
problem is the OTHER half and is solved by 3D patches (prior_patch.py), not here.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .geometry_3d import ConeBeam3DConfig
from .filters import ramp_filter, calibrate_scale  # 1D-along-u filter + scalar fit; see filters.py

__all__ = [
    "forward_project_3d_batched",
    "fdk_conebeam_3d_batched",
    "wang_weight",
    "calibrate_scale",
]


def _ray_box_intersect_3d(o, d, box_min, box_max, eps_dir: float = 1e-8):
    """Slab-method ray/box intersection in 3D. o, d: (R, 3); box_*: (3,)."""
    d_safe = torch.where(d.abs() < eps_dir, torch.full_like(d, eps_dir), d)
    inv_d = 1.0 / d_safe
    t0 = (box_min[None, :] - o) * inv_d
    t1 = (box_max[None, :] - o) * inv_d
    t_small = torch.minimum(t0, t1)
    t_big = torch.maximum(t0, t1)
    tmin = torch.max(t_small, dim=-1).values
    tmax = torch.min(t_big, dim=-1).values
    hit = tmax > tmin
    return tmin, tmax, hit


def _world_to_grid_norm_3d(pts, *, W, H, D, dx, dy, dz, X0, Y0, Z0,
                           align_corners: bool):
    """World (x, y, z) -> 5D grid_sample normalized coords in [-1, 1].

    Returns (..., 3) ordered (x_norm, y_norm, z_norm) as grid_sample expects
    (x indexes W, y indexes H, z indexes D).

    REFERENCE implementation. The forward projector does NOT call this on its hot path --
    it uses `_grid_affine_3d` + one fused `addcmul` instead (22x faster, see below). Kept
    because it is the readable statement of the convention that `warp.py` and
    `forward_projector_2d.py` must agree with, and the fused path is checked against it."""
    ix = (pts[..., 0] - X0) / dx - 0.5
    iy = (pts[..., 1] - Y0) / dy - 0.5
    iz = (pts[..., 2] - Z0) / dz - 0.5
    if align_corners:
        x_norm = 2.0 * ix / max(W - 1, 1) - 1.0
        y_norm = 2.0 * iy / max(H - 1, 1) - 1.0
        z_norm = 2.0 * iz / max(D - 1, 1) - 1.0
    else:
        x_norm = (2.0 * ix + 1.0) / float(W) - 1.0
        y_norm = (2.0 * iy + 1.0) / float(H) - 1.0
        z_norm = (2.0 * iz + 1.0) / float(D) - 1.0
    return torch.stack([x_norm, y_norm, z_norm], dim=-1)


_WARNED_PGRAD = [False]


def _use_triton(backend: str, vol: torch.Tensor, align_corners: bool,
                pmat: torch.Tensor | None = None) -> bool:
    """Triton is used when available, on CUDA, in fp32, align_corners=False.

    The kernel hard-codes the align_corners=False voxel-centre convention (it is the one the
    whole project uses -- see `warp.py`), so align_corners=True falls back to grid_sample.

    IT IS ALSO REFUSED WHENEVER Pmat REQUIRES GRAD, and that refusal is load-bearing here in a
    way it was not in the 4DCT project this kernel came from. `_RayMarch.backward` returns
    `gvol, None, None, ...`: it is the exact adjoint with respect to the VOLUME and it drops
    the gradient with respect to the ray constants (A, Bk) -- which are precisely where Pmat
    enters. 4DCT never noticed because it put motion in a DVF that WARPS the volume and kept P
    fixed; this project puts motion in P itself (`rigid_motion.params_to_Pmot`), so d(loss)/dP
    IS the motion estimator.

    Dropping it does not raise. autograd reads a `None` from a Function as a ZERO gradient, so
    a graph that also differentiates the volume would return a perfectly healthy volume
    gradient alongside a silently zero d(loss)/d(theta), and the motion estimator would simply
    never move -- converged-looking, wrong, and with nothing in the log. Hence: fall back,
    loudly the first time.

    Making the Triton adjoint carry dA/dBk would put the estimator's inner loop back on the
    fast kernel; until then, motion estimation runs on grid_sample.
    """
    import os
    from .triton_raymarch import HAVE_TRITON
    backend = os.environ.get("FDCT_PROJECTOR", backend)
    if backend == "gridsample":
        return False

    if pmat is not None and pmat.requires_grad:
        if backend == "triton":
            raise RuntimeError(
                "backend='triton' cannot differentiate w.r.t. Pmat: _RayMarch.backward returns "
                "no gradient for the ray constants, so d(loss)/d(theta) would be silently zero. "
                "Use backend='gridsample' for motion estimation.")
        if not _WARNED_PGRAD[0] and HAVE_TRITON and vol.is_cuda:
            _WARNED_PGRAD[0] = True
            print("[projector_3d] Pmat.requires_grad -> falling back to the grid_sample backend "
                  "(the Triton kernel has no adjoint w.r.t. the projection matrices).")
        return False

    ok = HAVE_TRITON and vol.is_cuda and vol.dtype == torch.float32 and not align_corners
    if backend == "triton" and not ok:
        raise RuntimeError("backend='triton' needs triton + CUDA + fp32 + align_corners=False")
    return ok


def _grid_affine_3d(*, W, H, D, dx, dy, dz, X0, Y0, Z0, align_corners: bool, device):
    """(scale, shift) with  grid_norm = world * scale - shift,  per axis (x, y, z).

    `_world_to_grid_norm_3d` is AFFINE in the world point, so it collapses. With
    align_corners=False the -0.5 (voxel-centre) and the +1 (of (2i+1)/W) cancel exactly:

        x_norm = (2*((px - X0)/dx - 0.5) + 1)/W - 1 = px * 2/(dx*W) - (2*X0/(dx*W) + 1)

    Composing with the ray `p(t) = o + d*t` and `t = tmin + step*k` makes the whole sample
    grid one `addcmul` over per-RAY constants -- no (rays, n_samples, 3) `pts` tensor, no
    per-axis slicing, no `stack`. Measured on (2 views x 96 rows x 512 cols x 384 samples):
    grid construction 16.62 ms -> 0.76 ms (22x), peak 3053 -> 1616 MiB, and the whole block
    18.30 -> 2.44 ms (7.5x). The grid agrees with the reference to 2.1e-6 in [-1,1] units
    (= 2.7e-4 voxel) and the sampled line integrals to 2.1e-5 relative -- both far below the
    n_samples=384 quadrature error (5.1e-5), i.e. this is float reassociation, not a model
    change.

    WHY THIS IS THE BOTTLENECK AT ALL: profiling says `grid_sample` (the actual trilinear
    interpolation) is only 8% of a block; 92% went into materializing and re-reading the
    coordinate tensor. A torch-native ray marcher is memory-bandwidth-bound on coordinates,
    not compute-bound on interpolation."""
    if align_corners:
        sx, sy, sz = 2.0 / (dx * max(W - 1, 1)), 2.0 / (dy * max(H - 1, 1)), 2.0 / (dz * max(D - 1, 1))
        cx = 2.0 * (X0 / dx + 0.5) / max(W - 1, 1) + 1.0
        cy = 2.0 * (Y0 / dy + 0.5) / max(H - 1, 1) + 1.0
        cz = 2.0 * (Z0 / dz + 0.5) / max(D - 1, 1) + 1.0
    else:
        sx, sy, sz = 2.0 / (dx * W), 2.0 / (dy * H), 2.0 / (dz * D)
        cx, cy, cz = 2.0 * X0 / (dx * W) + 1.0, 2.0 * Y0 / (dy * H) + 1.0, 2.0 * Z0 / (dz * D) + 1.0
    scale = torch.tensor([sx, sy, sz], device=device, dtype=torch.float32)
    shift = torch.tensor([cx, cy, cz], device=device, dtype=torch.float32)
    return scale, shift


def forward_project_3d_batched(
    volumes: torch.Tensor,    # (B, 1, D, H, W)
    Pmat: torch.Tensor,       # (B, V, 3, 4)  per-volume, per-view projection matrices
    u_coords: torch.Tensor,   # (nu,) physical detector coords [mm], lateral
    v_coords: torch.Tensor,   # (nv,) physical detector coords [mm], axial (z)
    *,
    dx: float,
    dy: float,
    dz: float,
    X0: float | None = None,
    Y0: float | None = None,
    Z0: float | None = None,
    n_samples: int = 256,
    view_chunk: int = 4,
    row_chunk: int | None = None,
    align_corners: bool = False,
    reg: float = 1e-8,
    backend: str = "auto",
) -> torch.Tensor:
    """Batched cone-beam forward projection. Returns (B, V, nv, nu) [mm * mu].

    `backend`: "auto" (Triton if importable, on CUDA, align_corners=False), "triton", or
    "gridsample". The Triton path generates every sample coordinate in-register (no
    (rays, n_samples, 3) tensor at all), so autograd retains 2.4 MB of per-ray constants
    instead of a 432 MiB coordinate grid per block, and its backward is the exact matched
    adjoint. See `fdct/triton_raymarch.py`. Set FDCT_PROJECTOR=gridsample to force the
    reference path.

    3D twin of `forward_project_2d_batched`: vectorized over the batch and over
    chunks of views (one big 5D grid_sample per chunk). Differentiable in
    `volumes` (and, through warp.py upstream, in the DVF).

    `row_chunk` additionally splits the DETECTOR ROWS. At a realistic panel size
    (768 rows x 1024 cols x n_samples) a single view's sample-point tensor is
    ~3.6 GB in fp32, so row chunking (not just view chunking) is mandatory there.
    None = all rows at once (fine for small sanity detectors)."""
    if volumes.ndim != 5:
        raise ValueError("volumes must be (B, 1, D, H, W)")
    vol = volumes.to(torch.float32)
    device = vol.device
    B, _, D, H, W = vol.shape

    if X0 is None:
        X0 = -0.5 * W * dx
    if Y0 is None:
        Y0 = -0.5 * H * dy
    if Z0 is None:
        Z0 = -0.5 * D * dz

    Pmat = Pmat.to(device=device, dtype=torch.float32)
    u_coords = u_coords.to(device=device, dtype=torch.float32)
    v_coords = v_coords.to(device=device, dtype=torch.float32)
    V = Pmat.shape[1]
    nu = u_coords.shape[0]
    nv = v_coords.shape[0]

    box_min = torch.tensor([X0, Y0, Z0], device=device, dtype=torch.float32)
    box_max = torch.tensor([X0 + dx * W, Y0 + dy * H, Z0 + dz * D],
                           device=device, dtype=torch.float32)
    I3 = torch.eye(3, device=device, dtype=torch.float32)
    ks = torch.arange(n_samples, device=device, dtype=torch.float32) + 0.5

    # Homogeneous detector directions [u; v; 1] for every element (nv, nu, 3);
    # sinogram layout is (nv, nu): row = v (axial), column = u (lateral).
    uu = u_coords[None, :].expand(nv, nu)
    vv = v_coords[:, None].expand(nv, nu)
    uh = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)          # (nv, nu, 3)

    rc = nv if row_chunk is None else int(row_chunk)
    outs_v = []
    use_triton = _use_triton(backend, vol, align_corners, pmat=Pmat)
    if use_triton:
        from .triton_raymarch import raymarch
        # continuous VOXEL-INDEX coords: i = (p - P0)/d - 0.5  (align_corners=False centres)
        v_scale = torch.tensor([1.0 / dx, 1.0 / dy, 1.0 / dz], device=device)
        v_shift = torch.tensor([X0 / dx + 0.5, Y0 / dy + 0.5, Z0 / dz + 0.5], device=device)
        vol_t = vol[:, 0].contiguous()                        # (B,D,H,W)
    else:
        g_scale, g_shift = _grid_affine_3d(W=W, H=H, D=D, dx=dx, dy=dy, dz=dz,
                                           X0=X0, Y0=Y0, Z0=Z0,
                                           align_corners=align_corners, device=device)

    for v0 in range(0, V, view_chunk):
        v1 = min(v0 + view_chunk, V)
        vc = v1 - v0
        P = Pmat[:, v0:v1]                        # (B, vc, 3, 4)
        A = P[..., :3] + reg * I3                 # (B, vc, 3, 3)
        b = P[..., 3]                             # (B, vc, 3)
        A_inv = torch.linalg.inv(A)               # (B, vc, 3, 3)
        src = -(A_inv @ b.unsqueeze(-1)).squeeze(-1)                 # (B, vc, 3)

        rows = []
        for r0 in range(0, nv, rc):
            r1 = min(r0 + rc, nv)
            nr = r1 - r0
            dirs = torch.einsum("bvij,mnj->bvmni", A_inv, uh[r0:r1])  # (B,vc,nr,nu,3)
            dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-12)

            o = src[:, :, None, None, :].expand(B, vc, nr, nu, 3)
            tmin, tmax, hit = _ray_box_intersect_3d(
                o.reshape(-1, 3), dirs.reshape(-1, 3), box_min, box_max)
            tmin = tmin.view(B, vc, nr, nu)
            tmax = tmax.view(B, vc, nr, nu)
            hit = hit.view(B, vc, nr, nu)

            step = (tmax - tmin).clamp_min(0.0) / float(n_samples)

            if use_triton:
                # p_k = A + Bk*(k+0.5) in voxel index; coordinates never leave registers.
                # A misses the box => step == 0 => the integral is 0, so `hit` is implicit.
                Bd = dirs * v_scale                                   # (B,vc,nr,nu,3)
                A = torch.addcmul(o * v_scale - v_shift, Bd, tmin[..., None])
                Bk = Bd * step[..., None]
                R = B * vc * nr * nu
                out = raymarch(vol_t,
                               A.reshape(R, 3).t(), Bk.reshape(R, 3).t(),
                               step.reshape(R), vc * nr * nu, n_samples)
                rows.append(out.view(B, vc, nr, nu))
                continue

            # Fused sample grid: grid_norm(k) = A + Bk*k, both per-RAY (see _grid_affine_3d).
            # A = (o + d*tmin) * scale - shift ;  Bk = (d * scale) * step
            Bd = dirs * g_scale                                       # (B,vc,nr,nu,3)
            A = torch.addcmul(o * g_scale - g_shift, Bd, tmin[..., None])
            Bk = Bd * step[..., None]
            grid = torch.addcmul(A.unsqueeze(-2), Bk.unsqueeze(-2), ks[:, None])
            grid = grid.reshape(B, vc * nr, nu, n_samples, 3)

            samp = F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros",
                                 align_corners=align_corners)
            samp = samp[:, 0].view(B, vc, nr, nu, n_samples)

            integral = samp.sum(dim=-1) * step                       # (B, vc, nr, nu)
            rows.append(torch.where(hit, integral, torch.zeros_like(integral)))
        outs_v.append(torch.cat(rows, dim=2))                        # (B, vc, nv, nu)

    return torch.cat(outs_v, dim=1)                                  # (B, V, nv, nu)


def wang_weight(u_coords: torch.Tensor, cfg) -> torch.Tensor | None:
    """Displaced-detector redundancy weight (Wang, Med Phys 29(7):1634, 2002). (nu,) or None.

    In half-fan the panel is pushed laterally so each view images somewhat more than half the
    object; the opposite view supplies the rest. Over a FULL 360 deg orbit the rays with
    |u| < d (d = the panel's overhang past the central ray) are therefore acquired TWICE -- once
    at beta, once at its conjugate near beta+pi -- while everything beyond d is acquired once.
    Backprojecting all of it with a uniform 2*pi/V weight would double-count the centre.

        w(u) = 0                              t <= -1
             = sin^2( pi/4 * (1 + t) )        -1 < t < 1        t = sign(u0) * u / d
             = 1                              t >= 1

    `w(u) + w(-u) = 1` identically, so a conjugate pair sums to exactly one contribution, and
    `w` falls SMOOTHLY to 0 at the panel's SHORT edge.

    That last property used to be read as "half-fan therefore needs no truncation extrapolation".
    It does not follow, and it was WRONG on real data. Wang tapers the short edge; nothing tapers
    the LONG one, where a real patient (on a couch, against a projection that never decays to
    zero) is still cut off. MEASURED on SPARE-MC against their own RTK reconstruction of the same
    projections: turning the pad on moves rel-rmse 0.1030 -> 0.0947. See `ohnesorge_pad`.

    Returns None for a centred panel (full-fan), where the weight would be a no-op only if
    applied as identity -- so we skip it rather than multiply by a bump function.
    """
    if not getattr(cfg, "is_half_fan", False):
        return None
    d = cfg.overlap_half_width_mm()
    s = 1.0 if cfg.det_offset_u_mm > 0 else -1.0
    t = (s * u_coords / d).clamp(-1.0, 1.0)
    return torch.sin(0.25 * torch.pi * (1.0 + t)) ** 2


def ohnesorge_pad(g: torch.Tensor, npad: int, left: bool = True,
                  right: bool = True) -> torch.Tensor:
    """Lateral truncation extrapolation, TRANSCRIBED FROM RTK (`rtkFFTProjectionsConvolution
    ImageFilter.hxx`, `PadInputImageRegion` + `UpdateTruncationMirrorWeights`), which cites
    Ohnesorge et al., Med Phys 27(1):39, 2000, eq. 3a/3b. `g` is (..., nu); returns (..., nu+2*npad).

    The ramp is NON-LOCAL, so a projection that does not decay to zero at the panel edge presents
    a STEP to the |f| filter, and the ramp differentiates that step into a low-frequency
    cupping/halo spread across the whole reconstruction. Extrapolating the projection outward
    removes the step.

    THE EXTENSION IS A POINT REFLECTION THROUGH THE BORDER SAMPLE, not a copy of it:

        right:  g[nu-1 + d] = w[d] * ( 2*S_E - g[nu-1 - d] ),   S_E = g[nu-1]
        left:   g[   0 - d] = w[d] * ( 2*S_A - g[     0 + d] ), S_A = g[1]
        w[d]  = sin( (npad - d) * pi / (2*npad - 2) ) ** 0.75           d = 0..npad

    so the extension CONTINUES THE SLOPE of the data (2*S_E - g[edge-d] rises if g was falling
    into the edge) and then decays to zero under `w`. Our previous version copied the edge VALUE
    under a raised-cosine taper, which flat-lines the slope and leaves a first-order kink -- and
    it was disabled entirely for half-fan.

    `S_A = g[1]`, not `g[0]`, is RTK's own indexing (`iidx[0] = leftRegion.GetIndex(0) +
    leftRegion.GetSize(0) + 1`). It looks like an off-by-one against the right-hand branch, but it
    is what RTK ships and what SPARE's reference reconstructions were made with, so it is
    reproduced rather than "fixed". It moves one column of 512.

    `npad` is RTK's truncation extent in COLUMNS = `floor(TruncationCorrection * nu)`, capped by
    the available zero-pad (RTK's `m_ZeroPadFactors = 2` leaves nu/2 on each side).

    ONLY EXTRAPOLATE A BORDER THAT IS A REAL PANEL EDGE. `left`/`right` exist because a point
    reflection through a border whose value is an ARTIFICIAL ZERO evaluates to `-g[edge-d]`: it
    injects a negative mirror of the projection. That is what the half-fan symmetric enlargement
    leaves on one side, and padding it blindly made SPARE-MC monotonically WORSE (rel-rmse 0.1030
    -> 0.1249 at npad=256). The caller knows which side was enlarged; it must say so.

    `clamp_min(0)` IS WHAT MAKES THIS SAFE ON UNTRUNCATED DATA, and it is not a hack: `g` is a
    (weighted) LINE INTEGRAL, which cannot be negative. Ohnesorge's reflection presupposes
    truncation -- where the border sample S is large because the object was cut through. Where the
    object FITS the panel, S ~ 0 and the raw formula returns `-g[edge-d]`, a negative mirror, i.e.
    pure damage; that cost `gate_shepp_halffan` 10/10 -> 2/10. Clamping sends exactly that case to
    zero, so an untruncated projection is zero-extended (a no-op) and a truncated one is
    extrapolated. The alternative -- deciding per call from `g.abs().amax()` -- looks equivalent
    and is NOT: `moco._continuous_backproject_3d` filters the sinogram in VIEW CHUNKS, so a
    batch-dependent decision flips between chunks and breaks the `phi=Id MC-FDK == static FDK`
    invariant (measured: 2.3e-7 -> 2.3e-1). The extension must be an ELEMENTWISE function of the
    projection, and this one is.
    """
    if npad <= 0 or not (left or right):
        return g
    nu = g.shape[-1]
    dev, dt = g.device, g.dtype
    d = torch.arange(0, npad + 1, device=dev, dtype=torch.float64)
    denom = max(2 * npad - 2, 1)
    w = torch.sin((npad - d) * torch.pi / denom).clamp_min(0.0) ** 0.75
    w = w.to(dt)

    dd = torch.arange(1, npad + 1, device=dev)                     # distance from the border
    out = [g]
    if right:
        S_E = g[..., nu - 1:nu]                                    # the right border sample
        out.append(w[dd] * (2.0 * S_E - g[..., (nu - 1) - dd]).clamp_min(0.0))
    if left:
        S_A = g[..., 1:2]                                          # RTK's left reference (index 1)
        # d = 1..npad -> columns -1..-npad, so flip to store them inward-out
        out.insert(0, (w[dd] * (2.0 * S_A - g[..., dd]).clamp_min(0.0)).flip(-1))
    return torch.cat(out, dim=-1)


def fdk_conebeam_3d_batched(
    sino: torch.Tensor,       # (B, V, nv, nu)
    Pmat: torch.Tensor,       # (B, V, 3, 4) nominal geometry used for recon
    u_coords: torch.Tensor,   # (nu,)
    v_coords: torch.Tensor,   # (nv,)
    cfg: ConeBeam3DConfig,
    *,
    D: int,
    H: int,
    W: int,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    window: str = "ramlak",
    cutoff: float = 1.0,
    scale: float | None = None,
    view_chunk: int = 8,
    vox_chunk: int = 2_000_000,
    eps: float = 1e-8,
    disp: torch.Tensor | None = None,
    trunc_pad: int | None = None,
    trunc_thresh: float = 0.02,
) -> torch.Tensor:
    """Batched FDK (Feldkamp) cone-beam reconstruction. Returns (B, D, H, W).

    `window`/`cutoff` control the ramp apodization (see `recon_2d.ramp_filter`).
    Default `ramlak` (plain ramp) -- `hann` costs substantial resolution and was the
    reason the early 3D reconstructions looked blurred.

    IMPORTANT (V_axis trick contract, same as `fbp_fanbeam_2d_batched`): the
    angular weight is `angle_span / sino.shape[1]`. Calling this with the view
    axis = 1 (each view on the batch axis) yields angle_span * SVBP_v, so a
    caller that averages the batch elements over the true V reproduces the
    standard FDK weight — this is what the continuous 3D MC-FBP relies on.

    `disp` (B, V, 3, D, H, W) VOXEL displacements = MOTION-COMPENSATED backprojection.
    Each reference voxel q is backprojected from where its material actually WAS at that
    view's acquisition time:

        r(q) = q - phi(q, tau_v)        (1st-order inverse of the pull map r -> r + phi(r))

    i.e. the DVF moves the backprojection COORDINATES, and the ray geometry (u, v, and the
    1/w^2 distance weight) is evaluated at the moved point r, not at q. This is the correct
    MC-FDK and it costs NO extra interpolation: the only resampling is the detector
    `grid_sample` that plain FDK already performs.

    Do NOT go back to `backproject-then-warp-the-volume` (FDK on the reference grid followed
    by `warp_volume(svbp, -phi)`). It applies the SAME first-order map but pays one trilinear
    VOLUME resample per view, and at |phi| ~ 1 voxel a trilinear resample destroys ~40% of the
    Laplacian energy -- measured. The blur then survives the view average.

    `disp=None` (or all-zero) reduces EXACTLY to the uncorrected static FDK: the moved coords
    are the voxel centers bit-for-bit.
    """
    device = sino.device
    dtype = torch.float32
    sino = sino.to(device=device, dtype=dtype)
    Pmat = Pmat.to(device=device, dtype=dtype)
    u_coords = u_coords.to(device=device, dtype=dtype)
    v_coords = v_coords.to(device=device, dtype=dtype)
    B, V, nv, nu = sino.shape
    du = float(cfg.du)
    dv = float(cfg.dv)
    u0 = float(cfg.det_offset_u_mm)      # lateral panel offset (element coords are shifted)
    # NOT `v0`: that name is the view-chunk loop index further down, and shadowing it silently
    # replaces the detector's axial offset with a VIEW NUMBER inside the backprojection -- which
    # shifts the panel by tens of millimetres per chunk and smears the reconstruction into an
    # arc. It reads as a geometry bug and it cost an afternoon. (Caught by gate_shepp_halffan
    # dropping 10/10 -> 2/10 with an offset that was supposed to be a no-op at 0.)
    v_off = float(getattr(cfg, "det_offset_v_mm", 0.0))   # AXIAL panel offset (SPARE-MC: -2 mm)

    # 0) HALF-FAN: widen the projection to the SYMMETRIC range [-u_max, +u_max] with zeros.
    #
    # This is not cosmetic and it is not the same as `trunc_pad`. The ramp filter is NON-LOCAL,
    # so `ramp(cos * w * g)` has non-zero TAILS beyond the panel's short edge -- on the side the
    # displaced panel never covered. Those tails are real filtered signal and they backproject
    # into exactly the doubly-sampled core r < r_ov. Clipping them (which is what the
    # `|u - u0| <= half_u` mask does on the un-padded array) starves the core.
    # MEASURED on a uniform water cylinder, interior value relative to the periphery, Halcyon:
    #     overlap d (mm)      175     135      95      40
    #     un-padded         0.913   0.854   0.806   0.707     <- deficit grows as d shrinks
    #     symmetric pad     1.001   ...     ...     1.001     <- flat
    # This is what RTK's `rtkDisplacedDetectorImageFilter` does when it enlarges the projection.
    if cfg.is_half_fan:
        u_lo, u_hi = float(u_coords[0]), float(u_coords[-1])
        umax = max(abs(u_lo), abs(u_hi))
        nu_p = 2 * int(math.ceil(umax / du))
        u_p = (torch.arange(nu_p, device=device, dtype=dtype) - (nu_p - 1) / 2.0) * du
        j0 = int(round((u_lo - float(u_p[0])) / du))
        if j0 < 0 or j0 + nu > nu_p:
            raise RuntimeError(f"half-fan symmetric pad misaligned: j0={j0}, nu={nu}, "
                               f"nu_p={nu_p}")
        pad_l, pad_r = j0, nu_p - nu - j0
        sino = F.pad(sino, (pad_l, pad_r))
        u_coords, nu, u0 = u_p, nu_p, 0.0
        # Which borders of the WIDENED array are still REAL panel edges? A side that got zeros is
        # not one, and must never be extrapolated (see `ohnesorge_pad`). On SPARE-MC the panel is
        # offset so hard that j0 = 0: the LEFT border IS the panel's long edge, at full Wang
        # weight, and the enlargement lands entirely on the right.
        real_l, real_r = (pad_l == 0), (pad_r == 0)
    else:
        real_l = real_r = True                       # full-fan: both borders are the panel

    # 1) cosine pre-weight (flat panel): SDD / sqrt(SDD^2 + u^2 + v^2).
    cos_w = cfg.SDD / torch.sqrt(
        cfg.SDD ** 2 + u_coords[None, :] ** 2 + v_coords[:, None] ** 2 + eps
    )                                                                # (nv, nu)
    g = sino * cos_w[None, None]

    # 1a) HALF-FAN redundancy weight (Wang 2002), between the cosine weight and the ramp. It is
    # a projection-domain weight, so it must precede the filter.
    w_wang = wang_weight(u_coords, cfg)
    if w_wang is not None:
        g = g * w_wang[None, None, None, :]

    # 1b) LATERAL TRUNCATION EXTRAPOLATION, before the ramp (Ohnesorge et al., Med Phys
    # 27(1):39, 2000; Hsieh's water-cylinder variant). `ramp_filter` zero-pads to >= 2*nu, so a
    # projection that does not decay to zero at the panel edge presents a step to the |f|
    # filter: the ramp differentiates it into a DC/cupping error that is spread over the whole
    # reconstruction. Our full-fan FOV is 262.6 mm and DIR-Lab case2's thorax is 297 mm wide,
    # so its rows genuinely run off the panel. Extending each row by its edge value under a
    # raised-cosine taper removes the step.
    # Where the object DOES fit, the edge value is 0 and the appended columns are 0 -- but this
    # is NOT bit-for-bit a no-op, because `ramp_filter` sizes its FFT as the next power of two
    # above 2*nu: widening nu changes N (e.g. 256 -> 512) and hence the ramp's discretization
    # and its circular wraparound. Measured on a non-truncated Shepp-Logan: 6.5e-4 relative.
    # The longer transform is the more accurate one. It cannot perturb the
    # `phi=Id MC-FDK == static FDK` gate (verified 2.3e-7): the same filtered sinogram feeds
    # both paths and `disp` never enters here.
    # Half-fan needs NO extrapolation: the Wang weight already tapers the short edge to zero,
    # and the object fits inside the (much larger) offset FOV so the long edge sees nothing.
    # Extrapolating a half-fan projection would invent data the conjugate view already supplies.
    # LATERAL TRUNCATION EXTRAPOLATION -- and it applies to HALF-FAN TOO. The old code disabled it
    # whenever the Wang weight was on, reasoning that "Wang tapers the short edge to zero and the
    # object fits inside the offset FOV". The first half is true; the second is not. Wang only
    # touches the SHORT edge. On the LONG edge -- which after the symmetric enlargement is the
    # array's own border, at Wang weight 1.0 -- the patient and the COUCH are still cut off.
    # MEASURED on SPARE-MC (`Proj/`, cos x Wang applied, mean over views and rows):
    #     col   0 (u=-346.5, the long edge)  0.1124     <- should be 0 for an untruncated view
    #     col   1                            0.1189     <- and still climbing inward
    #     peak                               3.7715
    # 3% of the peak, and the ramp is NON-LOCAL, so that step becomes a low-frequency cupping /
    # halo across the whole reconstruction -- which is exactly what we measured against SPARE's own
    # reconstruction (see docs and data/diag/spare_operator_*.png).
    #
    # Extrapolate every REAL panel edge (`real_l`/`real_r`); `ohnesorge_pad`'s clamp makes it a
    # NO-OP wherever the projection was not actually truncated, so this needs no data-dependent
    # branch -- and it must not have one: `moco._continuous_backproject_3d` filters in view chunks,
    # so any decision taken from `g.amax()` flips between chunks and breaks the
    # `phi=Id MC-FDK == static FDK` invariant.
    npad = (nu // 4) if trunc_pad is None else int(trunc_pad)
    pad_l = 0                       # how many columns were prepended -- the crop offset below
    if npad > 0:
        g = ohnesorge_pad(g, npad, left=real_l, right=real_r)
        pad_l = npad if real_l else 0          # the sides are padded INDEPENDENTLY
    nu_pad = g.shape[-1]

    # 2) ramp filter along u (row-wise; FDK filters only the lateral axis).
    # CHUNKED OVER VIEWS. `ramp_filter` runs an rfft of length >= 2*nu_pad in complex64, so a
    # whole (B,V,nv,nu_pad) transform is ~4x the sinogram. Half-fan's symmetric padding nearly
    # doubles nu, which pushed det_bin=1 (nu 1280 -> 2320, N=8192) to a 27 GiB allocation.
    # Filtering is row-independent, so chunking is exact.
    out = torch.empty((B, V, nv, nu), device=device, dtype=dtype)
    fchunk = max(1, min(V, int(view_chunk)))
    for v0 in range(0, V, fchunk):
        v1 = min(v0 + fchunk, V)
        blk = ramp_filter(g[:, v0:v1].reshape(-1, nu_pad), du=du, window=window,
                          cutoff=cutoff).view(B, v1 - v0, nv, nu_pad)
        out[:, v0:v1] = blk[..., pad_l:pad_l + nu]      # pad_l, NOT npad: the sides are separate
    g = out

    # Voxel homogeneous world coords (centered at origin, matching projector).
    xs = (torch.arange(W, device=device, dtype=dtype) - (W - 1) / 2.0) * dx
    ys = (torch.arange(H, device=device, dtype=dtype) - (H - 1) / 2.0) * dy
    zs = (torch.arange(D, device=device, dtype=dtype) - (D - 1) / 2.0) * dz
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")           # (D, H, W)
    Xh = torch.stack(
        [xx.reshape(-1), yy.reshape(-1), zz.reshape(-1),
         torch.ones(D * H * W, device=device, dtype=dtype)], dim=0)  # (4, Npix)
    Npix = D * H * W
    if disp is not None:
        if disp.shape[-3:] != (D, H, W) or disp.shape[-4] != 3:
            raise ValueError(f"disp must be (B,V,3,{D},{H},{W}), got {tuple(disp.shape)}")
        # (B,V,3,Npix) voxels -> mm, per axis. Channel order is [dx, dy, dz] (warp.py).
        disp = disp.to(device=device, dtype=dtype).reshape(*disp.shape[:2], 3, Npix)
        spac = torch.tensor([dx, dy, dz], device=device, dtype=dtype)[None, None, :, None]
        disp = disp * spac                                           # (B,V,3,Npix) mm

    # detector extent, in the SHIFTED element coordinate system (u0 = panel offset)
    half_u = 0.5 * nu * du
    half_v = 0.5 * nv * dv
    recon = torch.zeros((B, Npix), device=device, dtype=dtype)

    # 3) distance-weighted backprojection, chunked over views AND voxels.
    for v0 in range(0, V, view_chunk):
        v1 = min(v0 + view_chunk, V)
        vc = v1 - v0
        P = Pmat[:, v0:v1]                                           # (B, vc, 3, 4)
        src = g[:, v0:v1].reshape(B * vc, 1, nv, nu)
        for p0 in range(0, Npix, vox_chunk):
            p1 = min(p0 + vox_chunk, Npix)
            if disp is None:
                uvw = torch.einsum("bvij,jp->bvip", P, Xh[:, p0:p1])  # (B, vc, 3, Np)
            else:
                # MOTION COMPENSATION: backproject reference voxel q from r = q - phi(q).
                d = disp[:, v0:v1, :, p0:p1]                          # (B, vc, 3, Np) mm
                Xc = Xh[None, None, :, p0:p1] - F.pad(d, (0, 0, 0, 1))
                uvw = torch.einsum("bvij,bvjp->bvip", P, Xc)          # (B, vc, 3, Np)
            u_h, v_h, w = uvw[:, :, 0], uvw[:, :, 1], uvw[:, :, 2]
            w_safe = torch.where(w.abs() < eps, torch.full_like(w, eps), w)
            u = u_h / w_safe
            v = v_h / w_safe

            # detector element index of the landing point (undo BOTH panel offsets)
            ju = (u - u0) / du + (nu - 1) / 2.0
            jv = (v - v_off) / dv + (nv - 1) / 2.0
            x_norm = (2.0 * ju + 1.0) / float(nu) - 1.0
            y_norm = (2.0 * jv + 1.0) / float(nv) - 1.0
            grid = torch.stack([x_norm, y_norm], dim=-1)             # (B, vc, Np, 2)

            val = F.grid_sample(
                src, grid.reshape(B * vc, 1, p1 - p0, 2), mode="bilinear",
                padding_mode="zeros", align_corners=False,
            )[:, 0, 0]                                               # (B*vc, Np)
            val = val.view(B, vc, p1 - p0)

            mask = (w > 0) & ((u - u0).abs() <= half_u) & ((v - v_off).abs() <= half_v)
            contrib = torch.where(mask, val / (w_safe ** 2), torch.zeros_like(val))
            recon[:, p0:p1] += contrib.sum(dim=1)

    recon = recon * (float(cfg.angle_span) / float(V))
    recon = recon.view(B, D, H, W)
    if scale is not None:
        recon = recon * scale
    return recon
