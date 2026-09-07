"""THE PROJECTOR PAIR: LEAP (LLNL leapct) modular-beam, driven from our per-view `P`.

User decision 2026-07-29, after the code-level cross-check in `scripts/diag_leap_crosscheck.py`
(read that file's docstring for the model comparison and the measured agreement). The forward
and the adjoint are LEAP's; the FDK and the geometry gradient `d/dP` stay ours, because LEAP has
neither an equivalent of our FDK (`disp` motion-compensated backprojection, Voronoi angular
weights, Wang half-fan, Ohnesorge padding) nor any geometry derivative at all. The `d/dP` is
`fm3d/triton_leap_grad.py`'s EXACT gradient of LEAP's own kernel maths, stacked on the
LEAP-form (src, moduleCenter, rowVec, colVec) parameters and chained back to P.

ONE FORWARD KERNEL, ALWAYS (2026-07-30, user decision). Stock LEAP picks between its
separable-footprint and Joseph modular forwards per GEOMETRY SET, and since reconstruction
lives in the OBJECT frame an object rotation IS a panel tilt -- so the switch fired silently
under motion (measured: 60/60 training draws Joseph, 54/60 eval, 0/60 at half amplitude).
That cost an ~11%-of-loss cliff wherever theta-hat crossed the 5.06 deg bar, put the bridge's
t=1 anchor (nominal orbit -> SF) on a DIFFERENT operator from its own y (motion -> Joseph),
and forced a second surrogate gradient to exist for the SF branch. We patched the vendored
LEAP with `set_forceJosephModular` and pin it in `_model` -- third_party/leap/FM3D_PATCH.md is the
provenance note, and an unpatched libleapct now raises instead of silently reverting.

WHY MODULAR AND NOT CONE. LEAP's `set_conebeam` parameterizes the orbit by ONE ANGLE PER VIEW,
which cannot express a per-view 6-DoF rigid motion. `set_modularbeam` takes the source position,
the detector module centre and the two detector axes per view -- exactly what our `P` decomposes
into, and the decomposition survives the motion right-multiply `P_nom @ T(theta)`. So the SAME
code path serves the nominal orbit and every motion geometry, and there is no second operator to
keep in sync. Measured (360 views x 500x700 @0.64 mm, A6000):

    256^3 @1 mm   forward  0.52 s   backproject 0.054 s (VD) / 0.151 s (SF)
    612^3 @0.42   forward  1.46 s   backproject 0.448 s (VD) / 0.902 s (SF)

against our retired SF pair's 0.33/0.35 and 2.92/3.12: LEAP is 2.6x on the round trip at the
simulation grid and 1.17x at the reconstruction grid, and its forward is the 2x win that pays
for the whole switch (`gen.simulate` runs at 612^3).

WHICH BACKPROJECTOR. `ADJOINT_MODE = "VD"` (voxel-driven), the user's choice, matching the 4DCT
sibling's deployed configuration where CG was verified stable on it. Note what that costs: LEAP's
'VD' is NOT the transpose of its forward -- measured self-adjointness 1.9e-4, against 1.9e-7 for
its 'SF' backprojector and 0.0 for the retired matched pair. Flip the constant to "SF" to buy
the tighter adjoint back at ~2.8x the backprojection cost.

TRAPS THIS MODULE EXISTS TO HIDE (all measured; see the crosscheck docstring):
  * LEAP's parameter object LEAKS STATE across geometry types -- so this module only ever sets
    MODULAR geometry, on its own cached instance, and nothing else in the repo may touch it.
  * `cudaSetDevice(whichGPU)` inside LEAP: `set_gpu` must track the tensor's device or LEAP
    dereferences our pointers on the wrong GPU. Handled per call.
  * LEAP syncs the device after every kernel (`cudaDeviceSynchronize`), so it is ordered with
    torch's default stream -- no explicit synchronization is needed here.
  * `set_diameterFOV` is forced huge: LEAP's default cylindrical mask would silently clip the
    volume corners, which our operator never did.

NO GEOMETRY CACHING. `set_modularbeam` + `set_volume` measures 0.09 ms and the `P` decomposition
~1 ms, against 300+ ms for the projection itself. Caching on tensor identity would be fragile
(motion `P` is rebuilt every step) for <0.5% -- so the geometry is pushed on every call.
"""

from __future__ import annotations

import torch

try:
    from leapctype import tomographicModels
    HAVE_LEAP = True
except ImportError:                                             # pragma: no cover
    HAVE_LEAP = False


ADJOINT_MODE = "VD"          # 'VD' voxel-driven (deployed) | 'SF' separable footprint
_FOV_MM = 1.0e5              # kill LEAP's cylindrical volume mask; our geometry has none

_MODELS: dict[int, "tomographicModels"] = {}


def _model(device: torch.device):
    """One LEAP instance per GPU, MODULAR geometry only (see the state-leak note above),
    with the FORWARD PINNED TO JOSEPH."""
    idx = 0 if device.index is None else int(device.index)
    m = _MODELS.get(idx)
    if m is None:
        if not HAVE_LEAP:
            raise RuntimeError("leapctype is not importable: the projector pair is LEAP's "
                               "(fm3d/leap_projector.py). Install LEAP or check the env.")
        m = tomographicModels()
        # ONE OPERATOR, EVERY GEOMETRY (2026-07-30). Stock LEAP swaps its modular forward
        # between the separable-footprint and Joseph kernels per GEOMETRY SET, and object
        # rotation IS panel tilt, so the swap fired silently under motion -- see the module
        # docstring. This pins it. The `set_` call is DELIBERATELY not guarded: on a stock
        # libleapct the symbol is missing and ctypes raises here, which is exactly what we
        # want. Silently falling back would retrain the prior against a different operator.
        try:
            m.set_forceJosephModular(True)
        except AttributeError as e:                              # pragma: no cover
            raise RuntimeError(
                "this libleapct.so is UNPATCHED: it has no `set_forceJosephModular`. The "
                "whole pipeline assumes one modular forward kernel (third_party/leap/FM3D_PATCH.md "
                "has the rationale and the rebuild recipe). Refusing to run on the stock "
                "library, which would silently switch operators under motion.") from e
        _MODELS[idx] = m
    m.set_gpu(idx)
    return m


def decompose_P(P: torch.Tensor):
    """(V,3,4) -> (C, e_u, e_v, e_n, sdd), world mm, float64 on P's device.

    `P = K [R | -R C]` with `K = diag(SDD, SDD, 1)` and principal point (0,0)
    (`geometry_3d.build_conebeam_orbit`), so `M := P[:, :3]` has rows `(SDD*e_u, SDD*e_v, e_n)`
    and `P[:, 3] = -M C`. A rigid motion right-multiplies an SE(3) into `P` and PRESERVES that
    form -- which is why the motion geometry needs no special case anywhere below.
    """
    Pm = P.detach().to(torch.float64)
    M, p4 = Pm[:, :, :3], Pm[:, :, 3]
    C = torch.linalg.solve(M, -p4[..., None])[..., 0]
    sdd = M[:, 0].norm(dim=-1)
    return C, M[:, 0] / sdd[:, None], M[:, 1] / sdd[:, None], M[:, 2], sdd


def modular_arrays(P: torch.Tensor, u0: float, v_off: float):
    """(V,3,4) -> the four float32 host arrays `set_modularbeam` wants.

    `u0`/`v_off` are the CENTRES of our detector coordinate axes, i.e. the panel offset (they
    are 0 on a centred panel and non-zero on a half-fan one). LEAP's `moduleCenters` is the
    world position of the centre of the detector ARRAY -- not of the central ray -- so the
    offsets belong here and NOT in a principal-point argument.
    """
    C, e_u, e_v, e_n, sdd = decompose_P(P)
    mod = C + sdd[:, None] * e_n + u0 * e_u + v_off * e_v
    out = torch.stack([C, mod, e_v, e_u]).to("cpu", torch.float32).numpy()
    return (out[0].copy(order="C"), out[1].copy(order="C"),
            out[2].copy(order="C"), out[3].copy(order="C"))


# ---------------------------------------------------------------------------------------
# GATE / DIAGNOSTIC ONLY, below: the COUNTERFACTUAL "which kernel would STOCK LEAP have run
# here?". Production never asks -- our patched libleapct sets `useSF = false` for modular
# geometry in `_model`, so the answer is unconditionally JOSEPH and there is no branch left
# to take. These two survive so `gate_leap_projector` can prove the pin is doing work: on the
# NOMINAL orbit stock LEAP would have picked SF, and the deployed library must not.


def stock_kernel_kind(P, *, nv, du, dv, dx, dz, D, H, W) -> str:
    """What UNPATCHED LEAP would pick for this geometry: 'SF' or 'JOSEPH'. Gates only.

    Replicated from `projectors_Joseph.cu`: SF runs only when
    `modularbeamIsAxiallyAligned() && useSF`. The first condition (`set_sourcesAndModules`) is
    the one `sf_branch` does NOT cover and the one MOTION trips: every view's unit rowVector
    must keep z >= 0.9961 (a 5.06-degree panel tilt) and the source z-span must stay under
    half the panel height. One view past the bar flipped the WHOLE geometry, silently and
    mid-estimation -- which is what the patch removes (third_party/leap/FM3D_PATCH.md).
    """
    C, _, e_v, _, _ = decompose_P(P)
    axial = bool((e_v[:, 2] >= 0.9961).all()) and \
        float(C[:, 2].max() - C[:, 2].min()) <= 0.5 * nv * dv
    if not axial:
        return "JOSEPH"
    ok, _ = sf_branch(P, du=du, dv=dv, dx=dx, dz=dz, D=D, H=H, W=W)
    return "SF" if ok else "JOSEPH"


def sf_branch(P, *, du, dv, dx, dz, D, H, W):
    """GATE-ONLY: would STOCK LEAP take its SEPARABLE-FOOTPRINT kernel here, or fall back?

    `projectors_Joseph.cu` picks the modular projector like this:

        useSF = true
        if (!voxelSizeWorksForFastSF() &&
            (voxelWidth < default_voxelWidth() || voxelHeight < default_voxelHeight()))
            useSF = false                       -> modularBeamJosephProjectorKernel

    with `default_voxel = sod/sdd * pixel` (the detector pitch back-projected to the isocentre)
    and `voxelSizeWorksForFastSF` demanding the voxel sit inside
    `[0.5 * pitch@nearest, 2 * pitch@furthest]`. So a volume sampled FINER than the native grid
    silently swaps the operator for a ray-driven Joseph one -- a different model, no warning,
    no error. Our deployed grids clear it (1 mm and 0.4187 mm against a 0.4187 mm native), but
    a gate or an experiment on a coarse panel can land in the fallback and read as a model
    disagreement. Returns (uses_sf, detail).
    """
    C, _, _, _, sdd = decompose_P(P)
    sod = float(C.norm(dim=-1).mean())
    sdd = float(sdd.mean())
    r = 0.5 * float(torch.tensor([float(W * dx), float(H * dx), float(D * dz)]).norm())
    big_u, small_u = (sod + r) / sdd * du, (sod - r) / sdd * du
    big_v, small_v = (sod + r) / sdd * dv, (sod - r) / sdd * dv
    works = (0.5 * big_u <= dx <= 2.0 * small_u) and (0.5 * big_v <= dz <= 2.0 * small_v)
    finer = dx < sod / sdd * du or dz < sod / sdd * dv
    uses_sf = works or not finer
    return uses_sf, (f"voxel ({dx:g}, {dz:g}) mm vs native ({sod / sdd * du:.4g}, "
                     f"{sod / sdd * dv:.4g}) mm, SF window u [{0.5 * big_u:.3g}, "
                     f"{2 * small_u:.3g}] v [{0.5 * big_v:.3g}, {2 * small_v:.3g}]")


def _set_geometry(leap, P, *, nv, nu, du, dv, u0, v_off, D, H, W, dx, dy, dz):
    if abs(dx - dy) > 1e-9:
        raise ValueError(f"LEAP wants a square in-plane voxel (dx == dy), got {dx} vs {dy}")
    # NO kernel-selection check here any more: `_model` pins the modular forward to Joseph, so
    # there is nothing to fall back to and nothing to warn about (the pre-pin version warned
    # per geometry key and cost a `decompose_P` to do it).
    src, mod, rowv, colv = modular_arrays(P, u0, v_off)
    if not leap.set_modularbeam(P.shape[0], nv, nu, dv, du, src, mod, rowv, colv):
        raise RuntimeError("LEAP rejected the modular geometry")
    if not leap.set_volume(W, H, D, dx, dz):
        raise RuntimeError("LEAP rejected the volume")
    leap.set_diameterFOV(_FOV_MM)


_ALLOC_NOTE = """WHY EVERY LEAP CALL IS CHECKED FOR ALL-ZERO VIEWS (added 2026-08-03).

LEAP allocates its own CUDA memory (`cudaMalloc3DArray` for the volume texture) OUTSIDE torch's
caching allocator. Torch does not return freed blocks to the driver, so on a memory-hungry
caller LEAP can fail to allocate on a GPU that torch reports as half empty -- observed exactly
once at iteration ~25 of the Thies quality-metric training, which holds sixteen 128^3 volumes
plus float64 VIF temporaries:

    cudaMalloc3DArray Error: out of memory

AND NOTHING RAISED. `leapctype.project_gpu` sets `restype = c_bool` and then **discards the
return value**, handing back the output tensor it was given -- which we pre-fill with
`torch.zeros`. So an allocation failure produces a SILENTLY ALL-ZERO sinogram. Downstream that
is a blank reconstruction, a VIF* target computed against a blank volume, and a training pair
that is pure noise, with no error anywhere. On a 56-hour run that is unacceptable.

A real head projection has no all-zero view, so the check below is unambiguous. On a hit we
`empty_cache()` (which returns torch's unused blocks to the driver, the actual fix for this
class of failure) and retry ONCE; a second failure raises rather than corrupting the caller.
This costs one `amax` reduction per call (~1 ms on a 504 MB sinogram)."""


def _zero_views(g: torch.Tensor) -> torch.Tensor:
    """(B,V) mask of views that came back identically zero."""
    return g.reshape(g.shape[0], g.shape[1], -1).abs().amax(-1) == 0


def leap_project(vol, P, *, nv, nu, dx, dy, dz, du, dv, u0=0.0, v_off=0.0):
    """Forward projection. vol (B,D,H,W) fp32 cuda -> sinogram (B,V,nv,nu). No autograd.

    Raises if LEAP silently returned zeros -- see `_ALLOC_NOTE`."""
    B, D, H, W = vol.shape
    V = P.shape[1]
    vol = vol.contiguous()
    leap = _model(vol.device)

    def _run():
        g = torch.zeros((B, V, nv, nu), device=vol.device, dtype=torch.float32)
        for b in range(B):
            _set_geometry(leap, P[b], nv=nv, nu=nu, du=du, dv=dv, u0=u0, v_off=v_off,
                          D=D, H=H, W=W, dx=dx, dy=dy, dz=dz)
            leap.project_gpu(g[b], vol[b])
        return g

    g = _run()
    bad = _zero_views(g)
    if bad.any() and vol.abs().amax() > 0:
        torch.cuda.empty_cache()
        g = _run()
        bad = _zero_views(g)
        if bad.any():
            raise RuntimeError(
                f"LEAP forward returned {int(bad.sum())}/{B * V} ALL-ZERO views even after "
                f"empty_cache(). Look for 'cudaMalloc3DArray Error: out of memory' above -- "
                f"leapctype discards LEAP's success flag, so this check is the only signal. "
                f"Volume {tuple(vol.shape)}, sinogram {(B, V, nv, nu)}. Reduce the caller's "
                f"resident torch memory or its batch size.")
    return g


def leap_backproject(g, P, *, D, H, W, dx, dy, dz, du, dv, u0=0.0, v_off=0.0,
                     mode: str | None = None):
    """Backprojection. g (B,V,nv,nu) -> (B,D,H,W). `mode` defaults to `ADJOINT_MODE`.

    This is LEAP's backprojector, NOT the exact transpose of `leap_project` (module docstring):
    it is the operator the CG data step and every `A^T` call in the loop now use.
    """
    B, V, nv, nu = g.shape
    g = g.contiguous()
    leap = _model(g.device)
    f = torch.zeros((B, D, H, W), device=g.device, dtype=torch.float32)
    for b in range(B):
        _set_geometry(leap, P[b], nv=nv, nu=nu, du=du, dv=dv, u0=u0, v_off=v_off,
                      D=D, H=H, W=W, dx=dx, dy=dy, dz=dz)
        leap.set_projector(mode or ADJOINT_MODE)
        leap.backproject_gpu(g[b], f[b])
    return f


def leap_fdk_backproject(g, Pmat, *, D, H, W, dx, dy, dz, du, dv, u0=0.0, v_off=0.0):
    """FDK distance-weighted backprojection THROUGH LEAP's modular VD backprojector.

    The FDK step our retired fused Triton kernel used to do: for each voxel,
    sum bilinear(g_filtered)(hit point) / w^2 over views. LEAP's VD kernel instead weights
    each contribution by  sdd * dist(hit) / w^2  (its geometric `backprojectionWeight`:
    pmcn * sqrt(D^2(ru^2+rv^2) + pmcn^2) / (r.n)^2, with |pmcn| = sdd and
    D*(r.u), D*(r.v) = the hit point's central-ray detector coordinates) and multiplies the
    result by dx*dy*dz/(du*dv). So dividing the FILTERED sinogram by sdd*dist(u,v) and the
    output by LEAP's voxel/detector scalar turns LEAP's backprojection into OUR FDK step --
    exactly, up to interpolating the folded weight together with the data (bilinear of a
    product vs product of bilinears; second-order in the detector cell) and LEAP's tex
    border handling of off-panel rays (zero, like our mask). Gated against the torch FDK
    reference by `gate_fdk_fast.py`.

    g (B,V,nv,nu) = the fully filtered sinogram in the FINAL panel coordinate system
    (u0/v_off = its centre offsets, half-fan enlargement included). Angular weights are the
    caller's business, exactly as with the retired kernel.
    """
    B, V, nv, nu = g.shape
    sdd = float(Pmat[..., 0, :3].to(torch.float64).norm(dim=-1).mean())
    uu = (torch.arange(nu, device=g.device, dtype=g.dtype) - (nu - 1) * 0.5) * du + u0
    vv = (torch.arange(nv, device=g.device, dtype=g.dtype) - (nv - 1) * 0.5) * dv + v_off
    dist = torch.sqrt(sdd * sdd + uu[None, :] ** 2 + vv[:, None] ** 2)
    g2 = (g / (sdd * dist)[None, None]).contiguous()
    out = leap_backproject(g2, Pmat, D=D, H=H, W=W, dx=dx, dy=dy, dz=dz, du=du, dv=dv,
                           u0=u0, v_off=v_off, mode="VD")
    return out * (du * dv / (dx * dy * dz))


def leap_fdk_backproject_tangent(g, Pmat, Pdot, wgt, dwgt, *, D, H, W, dx, dy, dz, du, dv,
                                 u0=0.0, v_off=0.0):
    """(x, dx/ds) of the DEPLOYED FDK backprojection -- the bridge's velocity target.

    The tangent twin of `leap_fdk_backproject`: same 1/(sdd*dist) weight fold, same VD
    backprojection model, and the derivative is the EXACT s-derivative of that very model
    (`triton_leap_grad.leap_vd_backproject_tangent`, chained through the modular
    decomposition's jvp). Until 2026-07-30 the tangent came from the retired kernel's model
    instead (~0.5% cross-model error in the FM velocity target -- measured in the
    tangent-is-the-training-target ledger); LEAP ships no derivative, so this closes the
    last non-LEAP operator in production. Per-view weights (wgt, dwgt) ride INSIDE the sum
    so the Voronoi share's own s-derivative lands in dx/ds.

    NOTE the VALUE output is this kernel's own bilinear (not LEAP's texture unit): identical
    maths, ~1e-4 apart (tex 9-bit lerp). Callers that must match the static anchor BIT-TIGHT
    keep taking the value from `leap_fdk_backproject` and only the derivative from here.
    """
    B, V, nv, nu = g.shape
    sdd = float(Pmat[..., 0, :3].to(torch.float64).norm(dim=-1).mean())
    uu = (torch.arange(nu, device=g.device, dtype=g.dtype) - (nu - 1) * 0.5) * du + u0
    vv = (torch.arange(nv, device=g.device, dtype=g.dtype) - (nv - 1) * 0.5) * dv + v_off
    dist = torch.sqrt(sdd * sdd + uu[None, :] ** 2 + vv[:, None] ** 2)
    g2 = (g / (sdd * dist)[None, None]).contiguous()
    from .triton_leap_grad import leap_vd_backproject_tangent
    return leap_vd_backproject_tangent(g2, Pmat, Pdot, wgt, dwgt, D=D, H=H, W=W,
                                       dx=dx, dy=dy, dz=dz, du=du, dv=dv,
                                       u0=u0, v_off=v_off)


class LEAPProject(torch.autograd.Function):
    """Differentiable `(vol, P) -> sinogram`.

    `grad_vol` is LEAP's backprojection. `grad_P` is `triton_leap_grad.leap_grad_P`: the
    EXACT analytic gradient of the Joseph kernel LEAP now always runs (the `_model` pin). The
    Joseph model is bilinear in CONTINUOUS coordinates, so its exact gradient is also the
    slope of the physical loss surface -- FD parity 4e-4 against the LEAP loss itself.

    WHY THERE IS NO LONGER A SECOND GRADIENT. Until 2026-07-30 the SF branch had to use a
    surrogate (`triton_sf.sf_grad_P`): LEAP's SF kernel projects ROUNDED voxel centres, which
    puts a ~1e-3 rad RIPPLE on the loss, and the exact gradient of that model is the ripple's
    local slope -- measured fp64, FD gives +4.1e-5 for eps <= 3e-4 rad but -3.4e-4 for
    eps >= 1e-3 rad, i.e. the SIGN FLIPS between the local and the trend scale, and the
    estimator's frontier (~0.1 deg = 1.7e-3 rad) sits right at the ripple. Pinning the kernel
    removed the branch, so this whole hazard is gone; the ledger is kept in
    `triton_leap_grad`'s docstring because it is the reason the patch exists.

    WHAT REMAINS APPROXIMATE: LEAP's 'VD' backprojection is not the transpose of LEAP's
    forward (1.9e-4) -- the volume gradient keeps that known defect (module docstring).
    """

    @staticmethod
    def forward(ctx, vol, P, nv, nu, dx, dy, dz, du, dv, u0, v_off):
        g = leap_project(vol, P, nv=nv, nu=nu, dx=dx, dy=dy, dz=dz, du=du, dv=dv,
                         u0=u0, v_off=v_off)
        ctx.save_for_backward(vol, P)
        ctx.meta = (dx, dy, dz, du, dv, u0, v_off)
        return g

    @staticmethod
    def backward(ctx, gout):
        vol, P = ctx.saved_tensors
        dx, dy, dz, du, dv, u0, v_off = ctx.meta
        B, D, H, W = vol.shape
        gout = gout.contiguous()
        gvol = gP = None
        if ctx.needs_input_grad[0]:
            gvol = leap_backproject(gout, P, D=D, H=H, W=W, dx=dx, dy=dy, dz=dz,
                                    du=du, dv=dv, u0=u0, v_off=v_off)
        if ctx.needs_input_grad[1]:
            from .triton_leap_grad import leap_grad_P
            gP = leap_grad_P(vol, gout, P, dx=dx, dy=dy, dz=dz, du=du, dv=dv,
                             u0=u0, v_off=v_off)
        return gvol, gP, None, None, None, None, None, None, None, None, None


def _detector_params(u_coords, v_coords):
    du = float(u_coords[1] - u_coords[0])
    dv = float(v_coords[1] - v_coords[0])
    u0 = float(u_coords[0] + u_coords[-1]) * 0.5
    v_off = float(v_coords[0] + v_coords[-1]) * 0.5
    return du, dv, u0, v_off


def leap_project_3d_batched(volumes, Pmat, u_coords, v_coords, *, dx, dy, dz):
    """(B,1,D,H,W), (B,V,3,4) -> (B,V,nv,nu), differentiable in BOTH inputs."""
    du, dv, u0, v_off = _detector_params(u_coords, v_coords)
    return LEAPProject.apply(volumes[:, 0].to(torch.float32), Pmat.to(torch.float32),
                             len(v_coords), len(u_coords), dx, dy, dz, du, dv, u0, v_off)


def leap_project_3d_tangent(volumes, Pmat, Pdot, u_coords, v_coords, *, dx, dy, dz):
    """(y, dy/ds) of the forward projection along Pdot. (B,1,D,H,W) -> two (B,V,nv,nu).

    The forward-projection twin of `leap_fdk_backproject_tangent`, and for the same reason: the
    DATA bridge parameterizes its MEASUREMENT as y(s) = A(x; P(s*theta)) and regresses on
    dy/ds, LEAP ships no derivative, so we differentiate LEAP's own pinned Joseph kernel
    (`triton_leap_grad.leap_forward_tangent`, chained through the modular decomposition's jvp).

    Raw kernel, no autograd Function -- taking it inside a live graph would silently return
    graph-free tensors. The bridge calls it under `no_grad`, like every other draw-time
    operator. The VALUE it returns is the transcription's, ~1e-4 from the texture unit; callers
    that need the endpoints bit-exact take the value from `leap_project_3d_batched`.
    """
    du, dv, u0, v_off = _detector_params(u_coords, v_coords)
    from .triton_leap_grad import leap_forward_tangent
    return leap_forward_tangent(volumes[:, 0].to(torch.float32),
                                Pmat.to(torch.float32), Pdot.to(torch.float32),
                                nv=len(v_coords), nu=len(u_coords),
                                dx=dx, dy=dy, dz=dz, du=du, dv=dv, u0=u0, v_off=v_off)


def leap_backproject_3d_batched(sino, Pmat, u_coords, v_coords, *, D, H, W, dx, dy, dz):
    """(B,V,nv,nu) -> (B,1,D,H,W). The adjoint call sites' single entry point."""
    du, dv, u0, v_off = _detector_params(u_coords, v_coords)
    return leap_backproject(sino.to(torch.float32), Pmat.to(sino.device, torch.float32),
                            D=D, H=H, W=W, dx=dx, dy=dy, dz=dz, du=du, dv=dv,
                            u0=u0, v_off=v_off)[:, None]
