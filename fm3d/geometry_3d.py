"""Cone-beam projection geometry for CQ500 motion simulation.

World coordinates and detector coordinates are in mm. The source orbits the
x-y plane; z is the rotation axis. Volumes use (B, 1, D, H, W) with W=x, H=y,
D=z. A 3x4 matrix maps homogeneous world coordinates to detector coordinates.
Rigid motion enters through P_moved = P_nominal @ T_object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class ConeBeam3DConfig:
    """Circular cone-beam geometry; the paper explicitly uses ``thies()``.

    Detector offsets and binning are expressed in physical mm. The evaluation
    mask is the region measured in every view, which narrows with radial distance.
    """

    # distances
    SOD: float = 1000.0         # source-to-isocenter (axis) [mm]
    SDD: float = 1500.0         # source-to-detector [mm]   (M = SDD/SOD = 1.5)

    # Generic detector defaults; thies() supplies the CQ500 protocol.
    det_nu: int = 1024          # columns (lateral, in the orbit plane)
    det_nv: int = 768           # rows (axial, parallel to +z = SI)
    det_pixel_mm: float = 0.388  # 2x2-binned pitch (native 0.194)
    det_offset_u_mm: float = 0.0  # lateral panel offset (half-fan); FDK supports it via the
                                  # Wang weight + symmetric enlargement (360 deg orbits only --
                                  # see class docstring and projector_3d.wang_weight)
    det_offset_v_mm: float = 0.0  # axial panel offset [mm]
    det_bin: int = 1            # extra binning applied on top (see class docstring)

    # acquisition
    n_views: int = 660         # faster CBCT system than the stock ~900-view OBI protocol
    scan_duration_sec: float = 60.0
    angular_range_deg: float = 360.0
    angle_start: float = 0.0    # first view angle [rad]
    clockwise: bool = False

    # ---- presets ------------------------------------------------------------------


    @classmethod
    def thies(cls, det_pixel_mm: float = 0.64, n_views: int = 360,
              det_bin: int = 1) -> "ConeBeam3DConfig":
        """CQ500 protocol: SOD 785 mm, SDD 1200 mm, 700x500 panel at 0.64 mm.

        The full circular scan has 360 views. The measured axial coverage is
        shorter than the 256 mm reconstruction box; use measured_region_mask
        for image evaluation. ``det_bin`` is used by the coarse pose fit.
        """
        return cls(SOD=785.0, SDD=1200.0, det_nu=700, det_nv=500, det_pixel_mm=det_pixel_mm,
                   det_bin=det_bin, n_views=n_views, angular_range_deg=360.0)



    # ---- derived (kept as properties so the old .nu/.du/.angle_span call sites work)
    @property
    def nu(self) -> int:
        return self.det_nu // self.det_bin

    @property
    def nv(self) -> int:
        return self.det_nv // self.det_bin

    @property
    def du(self) -> float:
        return self.det_pixel_mm * self.det_bin

    @property
    def dv(self) -> float:
        return self.det_pixel_mm * self.det_bin

    @property
    def angle_span(self) -> float:
        return self.angular_range_deg * 3.141592653589793 / 180.0

    @property
    def magnification(self) -> float:
        return self.SDD / self.SOD

    # ---- geometry summary (printed by the sanity/viz scripts) --------------------
    def fov_diameter_mm(self) -> float:
        import math
        half = 0.5 * self.nu * self.du + abs(self.det_offset_u_mm)
        return 2.0 * self.SOD * math.sin(math.atan(half / self.SDD))

    def axial_coverage_mm(self) -> float:
        return self.nv * self.dv / self.magnification

    def iso_pitch_mm(self) -> float:
        return self.du / self.magnification

    # ---- half-fan / displaced detector -------------------------------------------
    @property
    def is_half_fan(self) -> bool:
        return abs(self.det_offset_u_mm) > 1e-9

    def overlap_half_width_mm(self) -> float:
        """Half-width `d` of the doubly-sampled central region ON THE DETECTOR.

        The panel spans u in [u0 - half_u, u0 + half_u] around the piercing point u=0. The short
        side is |u0| - ... no: it is `half_u - |u0|`, the amount by which the panel still
        overhangs the piercing point on the near side. Rays with |u| < d are measured TWICE per
        360 deg (at beta and at its conjugate); rays beyond d, once. `d = 0` would mean the
        panel starts exactly at the central ray (a true half-scan with no overlap, which cannot
        be feathered)."""
        return 0.5 * self.nu * self.du - abs(self.det_offset_u_mm)

    def cone_half_angle_deg(self) -> float:
        import math
        return math.degrees(math.atan(0.5 * self.axial_coverage_mm() / self.SOD))

    def frame_rate_hz(self) -> float:
        return self.n_views / max(self.scan_duration_sec, 1e-9)

    def describe(self) -> str:
        return (f"cone-beam: SOD={self.SOD:.0f} SDD={self.SDD:.0f} (M={self.magnification:.2f}) | "
                f"det {self.nu}x{self.nv} @ {self.du:.3f}mm (bin={self.det_bin}, "
                f"offset_u={self.det_offset_u_mm:.0f}mm) | V={self.n_views} over "
                f"{self.angular_range_deg:.0f}deg in {self.scan_duration_sec:.0f}s "
                f"({self.frame_rate_hz():.1f} fps)\n"
                f"  -> FOV diam {self.fov_diameter_mm():.1f}mm | axial cov "
                f"{self.axial_coverage_mm():.1f}mm | det pitch @iso {self.iso_pitch_mm():.3f}mm")

def measured_region_mask(shape, spacing, cfg: ConeBeam3DConfig, *, fov_frac: float = 1.0,
                         axial_frac: float = 1.0, device="cpu") -> "torch.Tensor":
    """(D,H,W) bool: voxels that every view actually measures. A BARREL, not a cylinder.

        r  <= R_fov                                  (transaxial FOV; half-fan aware)
        |z| <= half_v * (SOD - r) / SDD              (axial, and it NARROWS with r)

    The second line is the one that is easy to get wrong. A voxel at radius r passes within
    (SOD - r) of the source once per rotation, and at that instant its axial magnification is
    largest: v = SDD*z/(SOD - r). If that exceeds the panel half-height it falls off the panel
    for those views. So the fully-sampled region is a barrel. On the OBI defaults the axial
    half-extent is 99.3 mm on the axis but only 86.3 mm at the FOV edge -- which is why vendors
    quote ~16 cm of usable axial FOV where `nv*dv/M` says 19.9 cm. The old cylinder-and-slab
    mask leaked a thin shell of never-fully-measured voxels into every metric.

    Voxel centres are `(i - (N-1)/2) * spacing`, matching `fdk_conebeam_3d_batched` and the
    forward projector's `X0 = -0.5*W*dx` box. Use it for every metric and for FM patch sampling.
    `fov_frac`/`axial_frac` shrink it further, for backing off the wall.
    """
    D, H, W = shape
    dx, dy, dz = spacing
    R = 0.5 * cfg.fov_diameter_mm() * float(fov_frac)
    # The panel spans v in [v0 - half_v, v0 + half_v]. An AXIAL offset (SPARE-MC: v0 = -2 mm)
    # makes the barrel asymmetric about z=0, so the two ends must be carried separately -- a
    # symmetric |z| <= hz test would silently include unmeasured voxels on one side and exclude
    # measured ones on the other. Reduces exactly to the symmetric case when v0 = 0.
    half_v = 0.5 * cfg.nv * cfg.dv
    v_lo = cfg.det_offset_v_mm - half_v
    v_hi = cfg.det_offset_v_mm + half_v
    zs = (torch.arange(D, device=device, dtype=torch.float32) - (D - 1) / 2.0) * dz
    ys = (torch.arange(H, device=device, dtype=torch.float32) - (H - 1) / 2.0) * dy
    xs = (torch.arange(W, device=device, dtype=torch.float32) - (W - 1) / 2.0) * dx
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    r = torch.sqrt(xx ** 2 + yy ** 2)
    # worst-case magnification: the voxel passes within (SOD - r) of the source once per turn
    mag = (cfg.SOD - r).clamp_min(1e-6) / cfg.SDD
    a = float(axial_frac)
    return (r <= R) & (zz >= a * v_lo * mag) & (zz <= a * v_hi * mag)


def build_conebeam_orbit(
    cfg: ConeBeam3DConfig,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build nominal cone-beam projection matrices for a circular orbit.

    Returns P_nominal: (V, 3, 4).

    For view angle beta the source sits at C = SOD*(cos b, sin b, 0); the camera
    axes are e_depth = -C/SOD (unit, toward isocenter), e_u = in-plane lateral
    (e_depth rotated by -90deg, matching the 2D convention so the central slice
    of this geometry IS the 2D fan-beam geometry), e_v = +z. P = K [R | -R C].
    """
    if cfg.is_half_fan:
        if abs(cfg.angular_range_deg - 360.0) > 1e-6:
            raise NotImplementedError(
                "half-fan (det_offset_u_mm != 0) on a SHORT scan needs Parker weighting "
                f"(angular_range_deg={cfg.angular_range_deg}). Only the 360 deg case is "
                "supported, where the Wang weight alone de-duplicates the overlap.")
        if cfg.overlap_half_width_mm() <= 0.0:
            raise ValueError(
                f"det_offset_u_mm={cfg.det_offset_u_mm} exceeds the panel half-width "
                f"{0.5 * cfg.nu * cfg.du:.1f} mm: the panel no longer reaches the central ray, "
                "so there is no overlap to feather and the reconstruction is undefined.")
    V = cfg.n_views
    device = torch.device(device)

    if V > 1:
        betas = cfg.angle_start + cfg.angle_span * torch.arange(
            V, device=device, dtype=dtype
        ) / float(V)
    else:
        betas = cfg.angle_start + torch.zeros(V, device=device, dtype=dtype)
    if cfg.clockwise:
        betas = -betas

    zeros = torch.zeros_like(betas)
    C = torch.stack(
        [cfg.SOD * torch.cos(betas), cfg.SOD * torch.sin(betas), zeros], dim=-1
    )  # (V, 3)

    e_depth = -C / cfg.SOD                                            # (V, 3)
    e_u = torch.stack([e_depth[:, 1], -e_depth[:, 0], zeros], dim=-1)  # (V, 3)
    e_v = torch.stack([zeros, zeros, torch.ones_like(betas)], dim=-1)  # (V, 3)

    # World->camera rotation: rows are the camera axes [e_u; e_v; e_depth].
    R = torch.stack([e_u, e_v, e_depth], dim=-2)                      # (V, 3, 3)

    K = torch.zeros((V, 3, 3), device=device, dtype=dtype)
    K[:, 0, 0] = cfg.SDD
    K[:, 1, 1] = cfg.SDD
    K[:, 2, 2] = 1.0
    # principal point (u0, v0) = (0, 0): detector centered on the central ray.

    A = torch.bmm(K, R)                                               # (V, 3, 3)
    b = -torch.bmm(A, C[:, :, None])[:, :, 0]                         # (V, 3)

    P = torch.cat([A, b[:, :, None]], dim=-1)                         # (V, 3, 4)
    return P


def source_positions(Pmat: torch.Tensor) -> torch.Tensor:
    """(..., V, 3) source positions, read straight out of the projection matrices.

    P maps a world point to a homogeneous detector coordinate, and the ONE world point that
    projects to the degenerate [0,0,0] is the source. So the source is the right null vector of
    P: take the last right-singular vector and dehomogenize.

    Divide by the homogeneous component AS IT IS. Clamping it is a trap: the sign of a singular
    vector is arbitrary and flips from view to view, so `clamp(min=eps)` maps a perfectly good
    negative w to +1e-12 and throws that view's source to ~1e12 mm with the wrong sign. (It did.
    The translation-only cases read -32 dB until this was fixed.)
    """
    _, _, Vh = torch.linalg.svd(Pmat.double())          # (..., V, 3, 4) -> Vh (..., V, 4, 4)
    n = Vh[..., -1, :]                                  # (..., V, 4)
    if bool((n[..., 3].abs() < 1e-9).any()):
        raise RuntimeError("a projection matrix puts its source at infinity (parallel beam?)")
    return (n[..., :3] / n[..., 3:4]).to(Pmat.dtype)


def view_angular_weights(Pmat: torch.Tensor) -> torch.Tensor:
    """(..., V) the angular share d_beta of each view, from the ACTUAL source trajectory in P.

    WHY THIS EXISTS. FDK ends with a single `angle_span / V` -- one weight for every view, i.e.
    "the views are equiangular". Rigid patient motion about the GANTRY AXIS breaks exactly that
    and nothing else: rotating the object about z maps the source circle onto ITSELF, so the
    trajectory stays a circle and the ramp stays along u -- the views merely stop being evenly
    spaced. That one wrong weight is worth **-2.01 dB** on a 5 deg rotation, against 0.00 dB for
    any translation (`scripts/diag_oracle_fdk.py`). Feeding these weights back gives **+1.14 dB**
    on the bridge's own motion, and it is an EXACT NO-OP on the nominal orbit (the weights come
    back uniform to 1 part in 1e5), so nothing static can regress.

    It does NOT close the whole gap: once the views really are unevenly spaced, that is a
    SAMPLING deficit, and no reweighting invents the missing angles. The rest of FDK's motion
    loss (the ramp is still filtered along u, which a tilt of the orbit plane by rx/ry
    invalidates) stays. Only an iterative reconstruction recovers that.

    THE WEIGHT IS A VORONOI PARTITION OF THE CIRCLE, NOT A DIFFERENCE ALONG THE VIEW INDEX, and
    that distinction is load-bearing. A central difference `|beta[v+1] - beta[v-1]| / 2` assumes
    the effective angle advances MONOTONICALLY with the view index. Real head motion contains
    steps: in the `mixed` profile one view rotates the patient 5 deg while the gantry advances
    1 deg, so the effective view angle jumps BACKWARDS by 9 deg. The central difference then
    hands that single view a 4x weight, and the reconstruction gets WORSE, not better (measured:
    -0.87 dB on the patient whose motion has the step, while the two smooth ones gained +1.1).

    So: sort the views by their effective angle, give each one half the gap to each of its
    neighbours ON THE CIRCLE, and scatter that back. Every weight is then non-negative by
    construction, the weights sum to exactly 2*pi whatever the motion does, and views that pile
    up at the same angle correctly split one share between them instead of each claiming a full
    one. On the nominal orbit the sort is the identity and every gap is angle_span/V -- an exact
    no-op.

    Full 360 deg orbits only: the circular closure assumes the trajectory wraps. A short scan
    needs Parker weighting, which is not implemented.

    The source comes from the closed form S = -M^{-1} p4 (P = [M | p4] full-rank), the same
    route `view_angular_weights_dot` uses -- NOT from `source_positions`' batched float64 SVD,
    which computes the identical point (agreement ~1e-12 relative, verified) but costs a
    cuSOLVER double SVD per view on EVERY weighted FDK call. `source_positions` itself is kept
    for callers that want the null-vector route.
    """
    M = Pmat[..., :3, :3].double()
    p4 = Pmat[..., :3, 3:].double()
    # float64 solve, then round to Pmat.dtype BEFORE atan2 -- exactly what `source_positions`
    # returned, so the weights this function hands out are unchanged.
    S = -torch.linalg.solve(M, p4)[..., 0].to(Pmat.dtype)   # (..., V, 3)
    beta = torch.atan2(S[..., 1], S[..., 0])            # (..., V) in (-pi, pi]
    V = beta.shape[-1]
    two_pi = 2.0 * math.pi

    order = torch.argsort(beta, dim=-1)                 # views, ordered around the circle
    bs = torch.gather(beta, -1, order)

    gap = torch.empty_like(bs)                          # gap[i] = bs[i+1] - bs[i], closed circularly
    gap[..., :-1] = bs[..., 1:] - bs[..., :-1]
    gap[..., -1] = bs[..., 0] + two_pi - bs[..., -1]

    share = 0.5 * (gap + torch.roll(gap, 1, dims=-1))   # half the gap on either side
    w = torch.empty_like(share)
    w.scatter_(-1, order, share)                        # back to view order
    return w


def view_angular_weights_dot(Pmat: torch.Tensor,
                             Pdot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(w, dw/ds): `view_angular_weights` AND its exact directional derivative along Pdot.

    For the analytic bridge tangent: the FDK the bridge draws weights each view by its Voronoi
    angular share computed from the ACTUAL orbit P(s) = P_nom @ T(s*theta), so the weight is
    s-dependent through the gantry-axis part of theta and its derivative belongs in
    d/ds FDK(y, P(s)).

    Everything is closed-form:
      * the source is the right null vector of P; for a full-rank P = [M | p4] that is
        S = -M^{-1} p4 (identical to `source_positions`' SVD route, without the SVD, whose
        derivative is a mess), so  Sdot = -M^{-1} (Mdot S + p4dot);
      * beta = atan2(Sy, Sx)  ->  betadot = (Sx Sydot - Sy Sxdot) / (Sx^2 + Sy^2);
      * the Voronoi share is a fixed linear map of the sorted betas (half the gap to each
        circular neighbour), so its derivative is the SAME map applied to betadot with the
        sort permutation FROZEN -- exact a.e.; a reordering/tie is measure-zero in s, the same
        argument that freezes the interpolation cell in the tangent backprojection kernel.

    Computed in float64 (matching `source_positions`), returned in Pmat.dtype. The value
    output matches `view_angular_weights(Pmat)` to float64-vs-SVD noise (~1e-12 relative);
    both outputs are gated in scripts/gate_fdk_tangent.py.
    """
    M = Pmat[..., :3, :3].double()
    p4 = Pmat[..., :3, 3].double()
    Minv = torch.linalg.inv(M)
    S = -(Minv @ p4[..., None])[..., 0]                             # (..., V, 3)
    dM = Pdot[..., :3, :3].double()
    dp4 = Pdot[..., :3, 3].double()
    Sdot = -(Minv @ ((dM @ S[..., None]) + dp4[..., None]))[..., 0]

    beta = torch.atan2(S[..., 1], S[..., 0])                        # (..., V)
    r2 = S[..., 0] ** 2 + S[..., 1] ** 2
    bdot = (S[..., 0] * Sdot[..., 1] - S[..., 1] * Sdot[..., 0]) / r2

    two_pi = 2.0 * math.pi
    order = torch.argsort(beta, dim=-1)                             # FROZEN permutation
    bs = torch.gather(beta, -1, order)
    bds = torch.gather(bdot, -1, order)

    gap = torch.empty_like(bs)
    gap[..., :-1] = bs[..., 1:] - bs[..., :-1]
    gap[..., -1] = bs[..., 0] + two_pi - bs[..., -1]
    dgap = torch.empty_like(bds)                                    # the +2*pi constant drops
    dgap[..., :-1] = bds[..., 1:] - bds[..., :-1]
    dgap[..., -1] = bds[..., 0] - bds[..., -1]

    share = 0.5 * (gap + torch.roll(gap, 1, dims=-1))
    dshare = 0.5 * (dgap + torch.roll(dgap, 1, dims=-1))
    w = torch.empty_like(share)
    w.scatter_(-1, order, share)
    dw = torch.empty_like(dshare)
    dw.scatter_(-1, order, dshare)
    return w.to(Pmat.dtype), dw.to(Pmat.dtype)


def detector_coords_3d(
    cfg: ConeBeam3DConfig, device="cpu", dtype=torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Physical detector element coordinates: u (nu,), v (nv,) [mm], measured from the
    central ray. The lateral panel offset (half-fan) shifts the ELEMENT positions; the
    intrinsic K keeps principal point (0,0), so `u` may be off-center by design."""
    ju = torch.arange(cfg.nu, device=device, dtype=dtype)
    jv = torch.arange(cfg.nv, device=device, dtype=dtype)
    u = (ju - (cfg.nu - 1) / 2.0) * cfg.du + cfg.det_offset_u_mm
    v = (jv - (cfg.nv - 1) / 2.0) * cfg.dv + cfg.det_offset_v_mm
    return u, v
