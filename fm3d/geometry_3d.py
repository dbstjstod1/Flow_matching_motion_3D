"""3D cone-beam geometry built on the projection-matrix formulation (prompt 6).

This is the full-3D version of `geometry_2d.py` (which was written as the 2D
reduction of this very pipeline). World points are (x, y, z), the detector is a
2D flat panel, and a projection matrix is 3x4 mapping a homogeneous world point
[x, y, z, 1]^T to a homogeneous detector coordinate [u*w, v*w, w]^T.

Geometry convention (documented once, used by projector/FDK/warp/estimator)
---------------------------------------------------------------------------
* World frame: isocenter at the origin. The source orbits in the x-y (axial)
  plane; **z is the rotation axis = SI (superior-inferior) direction**. This is
  the whole point of the 3D redesign: breathing motion is SI-dominant and z is
  now an explicit in-volume axis (never through-plane).
* Volume tensor layout: (B, 1, D, H, W) with W<->x, H<->y, D<->z. Voxel centers
  at x = (ix-(W-1)/2)*dx, y = (iy-(H-1)/2)*dy, z = (iz-(D-1)/2)*dz  [mm]
  (same centered pixel-center convention as the 2D pipeline).
* Detector: flat panel, u = lateral (in the orbit plane, nu columns), v = axial
  (parallel to +z, nv rows), both physical mm, centered (principal point 0).
* Intrinsic K = [[SDD, 0, u0], [0, SDD, v0], [0, 0, 1]] so the matrix output is
  the physical detector position in mm. Extrinsic rows are the camera axes
  [e_u; e_v; e_depth] with e_depth pointing source->isocenter, e_u the in-plane
  lateral axis (same -90deg rotation as the 2D `build_fanbeam_orbit`), e_v=+z.
* Per-view breathing motion NEVER enters P (Option-A invariant): the operator
  is always A(Warp(x, phi(., tau_v)); P_nom).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class ConeBeam3DConfig:
    """Circular cone-beam scan geometry (all distances in mm).

    Defaults model a **Varian OBI kV cone-beam** acquisition (2x2-binned panel):
    SOD=1000, SDD=1500 (magnification M=1.5), 1024x768 detector at 0.388mm pitch,
    660 views over 360 deg (a FASTER-than-stock OBI gantry; the stock protocol is ~900).
    Derived numbers for these defaults:

        transaxial FOV diameter = 2 * SOD * sin(atan(nu*du/2 / SDD)) = 262.6 mm
        axial coverage at iso   = nv*dv / M                          = 198.7 mm
        detector pitch at iso   = du / M                             = 0.259 mm

    So a DIR-Lab thorax (248 mm in-plane) fits transaxially, but its 235 mm of z
    does NOT fit the 198.7 mm axial coverage -- that truncation is PHYSICAL (it is
    why OBI needs multiple couch positions), so callers must crop z to the cone.

    `det_bin` sub-samples the panel (nu,nv //= det_bin; du,dv *= det_bin). This is
    NOT a shortcut when det_bin=2: the detector pitch at the isocenter (0.517 mm)
    still meets Nyquist for a 0.97 mm voxel (needs <= 0.485 mm), while det_bin=1 is
    4x oversampled relative to the reconstruction grid and only costs compute.

    `det_offset_u_mm` shifts the panel laterally = the HALF-FAN / displaced-detector body
    protocol that EVERY clinical linac CBCT uses for a thorax or pelvis, because no full-fan
    FOV (26-27.7 cm on any vendor) contains one. Each view then images a bit more than half the
    object and the opposite view supplies the rest. Supported here; the centrally overlapping
    rays are de-duplicated by the **Wang weight** (Wang, Med Phys 29(7):1634, 2002) in
    `projector_3d.fdk_conebeam_3d_batched`. It requires a FULL 360 deg orbit: on a short scan
    you would also need Parker weighting (Parker, Med Phys 9(2):254, 1982), which is NOT
    implemented -- `build_conebeam_orbit` raises for that combination.

    Use `ConeBeam3DConfig.halcyon()` for the half-fan preset. See its docstring for why a
    Halcyon-class panel is the only clinical geometry that contains a whole DIR-Lab thorax.

    CAUTION, `axial_coverage_mm()` is the ON-AXIS value. The region measured in EVERY view is
    not a cylinder but a BARREL that narrows with radius: a voxel at radius r comes within
    (SOD - r) of the source at some view, where its magnification is largest, so

        |z| <= half_v * (SOD - r) / SDD          (`measured_region_mask`)

    On the OBI defaults that is 99.3 mm on the axis but only 86.3 mm at the FOV edge. This is
    why vendors quote a smaller axial FOV (OBI: ~16 cm) than nv*dv/M would suggest.
    """

    # distances
    SOD: float = 1000.0         # source-to-isocenter (axis) [mm]
    SDD: float = 1500.0         # source-to-detector [mm]   (M = SDD/SOD = 1.5)

    # detector (native Varian OBI, 2x2 binned)
    det_nu: int = 1024          # columns (lateral, in the orbit plane)
    det_nv: int = 768           # rows (axial, parallel to +z = SI)
    det_pixel_mm: float = 0.388  # 2x2-binned pitch (native 0.194)
    det_offset_u_mm: float = 0.0  # lateral panel offset (half-fan); FDK unsupported
    det_offset_v_mm: float = 0.0  # AXIAL panel offset. Small but real: SPARE-MC ships
                                  # ProjectionOffsetY = -2 mm ("calibration for simulation"),
                                  # and ignoring it shifts the whole measured barrel by
                                  # 2*SOD/SDD = 1.33 mm at the isocentre -- more than a voxel on
                                  # their 1 mm grid, i.e. a systematic z registration error
                                  # between our reconstruction and their ground truth.
    det_bin: int = 1            # extra binning applied on top (see class docstring)

    # acquisition
    n_views: int = 660         # faster CBCT system than the stock ~900-view OBI protocol
    scan_duration_sec: float = 60.0   # gantry rotation time. THIS, with the patient's
                                      # respiratory rate, sets how many BREATHS the scan
                                      # contains -- see `pca_motion.n_cycles_from`. It is
                                      # unrelated to n_phases (=10, the motion model's
                                      # within-cycle resolution) and to n_views.
    angular_range_deg: float = 360.0
    angle_start: float = 0.0    # first view angle [rad]
    clockwise: bool = False

    # ---- presets ------------------------------------------------------------------
    @classmethod
    def halcyon(cls, det_bin: int = 2, n_views: int = 660,
                scan_duration_sec: float = 60.0) -> "ConeBeam3DConfig":
        """Varian Halcyon / Ethos kV-CBCT: the ONLY shipping clinical geometry whose single
        circular orbit contains a whole DIR-Lab thorax.

            SAD 1000, SDD 1540 (M=1.54) | 43x43 cm a-Si panel, 1280^2 @ 0.336 mm
            lateral offset 175 mm (half-fan) | 360 deg

        Derived: FOV diameter **491.2 mm**, axial coverage **279.3 mm** on-axis, cone half-angle
        **8.0 deg**. Compare the two DIR-Lab cases we have: case1 needs 335.2 x 235.0 mm, case2
        needs 418.4 x 280.0 mm. Both fit. Varian OBI (FOV 262.6, axial 198.7 -- and only ~16 cm
        of usable axial FOV in practice) fits neither, which is why OBI thorax protocols are
        half-fan and why long anatomy needs a couch-shift double orbit.

        8.0 deg is not an aggressive cone angle: it IS Halcyon's, and it sits beside Elekta XVI
        (7.6 deg) and HyperSight (7.0 deg). FDK's error grows only linearly with distance from
        the mid-plane (Wang & Lin); the regime where it visibly breaks (Defrise disks) is ~30 deg.

        `det_bin=2` -> 640^2 @ 0.672 mm -> detector pitch at iso **0.436 mm**, still inside the
        0.485 mm Nyquist limit of a 0.97 mm voxel, and only 0.52x the rays of the OBI default.

        DELIBERATE DEVIATION: Halcyon rotates at 4 RPM (~15 s). We keep `scan_duration_sec=60`
        and `n_views=660`, i.e. Halcyon's GEOMETRY with a SLOW gantry. Respiratory-correlated
        CBCT needs a slow rotation to sample the breathing cycle (Sonke et al. 2005 use ~4 min);
        `scan_duration_sec` is what turns a respiratory RATE into n_cycles (~15 breaths here).
        """
        return cls(SOD=1000.0, SDD=1540.0, det_nu=1280, det_nv=1280, det_pixel_mm=0.336,
                   det_offset_u_mm=175.0, det_bin=det_bin, n_views=n_views,
                   scan_duration_sec=scan_duration_sec, angular_range_deg=360.0)

    @classmethod
    def spare_mc(cls, det_bin: int = 1, n_views: int = 680,
                 scan_duration_sec: float = 60.0) -> "ConeBeam3DConfig":
        """The SPARE Monte-Carlo geometry -- **the only real inference target this project has**
        (SPARE is the sole public thoracic 4D-CBCT dataset with raw projections).

            SAD 1000, SDD 1500 (M=1.5) | 512x384 @ 0.776 mm | offset_u +148 (half-fan),
            offset_v -2 | 680 views / 360 deg

        Every number is from the dataset's own `README_DataInfo.txt` AND cross-checked against
        the `<Matrix>` entries in its `Geometry.xml` (solving `P [S;1] = 0` for the source puts
        it at +anterior at gantry 0, rotating toward the patient's LEFT -- which is what fixes
        `angle_start` and `clockwise` below). Derived: FOV diameter **462.2 mm**, axial coverage
        **198.7 mm** on-axis, cone half-angle 5.7 deg.

        NOTE THE AXIAL SHORTFALL. SPARE asks for a 220 mm (SI) reconstruction but the on-axis
        axial coverage is 198.7 mm, and the BARREL narrows with radius on top of that. The ends
        of the requested box are therefore NOT measured -- physical, not a bug (their own
        instructions warn that some scans do not even fill the panel vertically). Keep
        `measured_region_mask` in every metric and in FM patch sampling.

        ANGLE CONVENTION. IEC 0 deg = source anterior = our +y; our beta=0 puts the source at
        +x; and their gantry angle increases toward the patient's left, which is CLOCKWISE in
        our right-handed (L, A, I) frame. Hence `angle_start = -90 deg` with `clockwise=True`,
        which yields beta = 90 deg - gantry. Getting this backwards MIRRORS the reconstruction
        instead of failing loudly -- `scripts/smoke_spare.py` gates it against the reference FDK
        that ships with every scan. Prefer `spare.config_from_scan()`, which reads the scan's own
        Geometry.xml; this preset is the same numbers for when the archive is not mounted.

        NOTE `det_offset_u_mm = -148`, where SPARE's own files say +148: our detector u axis runs
        OPPOSITE to RTK's, so the panel's lateral displacement changes sign with it, and the
        loaded projections must be flipped in u to match (`spare.read_projections(flip_u=True)`).
        The two changes are one change. `det_offset_v_mm = -2` keeps its sign -- the v axes DO
        agree.
        """
        return cls(SOD=1000.0, SDD=1500.0, det_nu=512, det_nv=384, det_pixel_mm=0.776,
                   det_offset_u_mm=-148.0, det_offset_v_mm=-2.0, det_bin=det_bin,
                   n_views=n_views, scan_duration_sec=scan_duration_sec,
                   angular_range_deg=360.0, angle_start=-1.5707963267948966, clockwise=True)

    @classmethod
    def thies(cls, det_pixel_mm: float = 0.64, n_views: int = 360,
              det_bin: int = 1) -> "ConeBeam3DConfig":
        """**The head-motion-compensation literature's de-facto standard simulation geometry.**

            SID 785 (source-to-ISOCENTRE = our SOD), SDD 1200 (M = 1.529)
            500 x 700 panel @ 0.64 mm | 360 views over a full 2*pi

        Shared by Thies et al. (TMI 2025, arXiv:2401.09283; arXiv:2405.19079; MICCAI 2024) and by
        JRM-ADM (arXiv:2504.14033), all of which forward-project CQ500 head CT volumes through it.
        Matching it is what makes our numbers comparable to theirs. JRM-ADM differs on two knobs
        (0.5 mm pitch, 120 views) -- hence the arguments; see `ConeBeam3DConfig.jrm_adm()`.

        Derived (gated in `scripts/gate_cq500.py`):
            FOV diameter    288.4 mm       axial coverage  209.3 mm on-axis
            iso pitch       0.419 mm       cone half-angle 7.6 deg

        **"500 x 700" does not say which axis is which, and the papers never do.** It MUST be
        nu (lateral) = 700: that is the 288 mm FOV above, and a head fits. The other assignment
        gives a 207 mm FOV, which truncates a head at every view. Derived, not quoted -- gated.

        "SID" is source-to-ISOCENTRE here (Thies' text says so explicitly). Elsewhere in the
        literature SID often means source-to-IMAGE, i.e. the detector -- reading it that way would
        silently shrink the geometry by 415 mm.

        Note the axial shortfall: a 256 mm (256^3 @ 1 mm) reconstruction box, which is what Thies
        evaluates on, is TALLER than the 209 mm the panel sees on-axis (and the measured region is
        a barrel that narrows further with radius). The ends of the box are not measured. This is
        physical -- keep `measured_region_mask` in every metric and in FM patch sampling.
        """
        return cls(SOD=785.0, SDD=1200.0, det_nu=700, det_nv=500, det_pixel_mm=det_pixel_mm,
                   det_bin=det_bin, n_views=n_views, angular_range_deg=360.0)

    @classmethod
    def jrm_adm(cls, n_views: int = 120, det_bin: int = 1) -> "ConeBeam3DConfig":
        """JRM-ADM's variant of the standard geometry: same SID/SDD/panel, 0.5 mm pitch, and only
        120 full-scan views (they then SUBSAMPLE to 20/40/60 for the sparse-view experiments).
        Derived: FOV 225.9 mm, axial coverage 163.5 mm -- both notably tighter than Thies', and
        225.9 mm is only just wider than a head."""
        return cls.thies(det_pixel_mm=0.5, n_views=n_views, det_bin=det_bin)

    @classmethod
    def preset(cls, name: str, *, det_bin: int | None = None, n_views: int = 660,
               scan_duration_sec: float = 60.0) -> "ConeBeam3DConfig":
        """`"thies"` (**the CQ500 head-motion standard**), `"jrm_adm"` (its sparse-view variant),
        `"halcyon"` (half-fan, contains a whole DIR-Lab thorax), `"obi"` (full-fan Varian OBI,
        which contains neither case and therefore truncates), or `"spare_mc"` (the 4DCT inference
        target). One switch for every script."""
        if name in ("thies", "cq500"):
            return cls.thies(det_bin=1 if det_bin is None else det_bin,
                             n_views=360 if n_views == 660 else n_views)
        if name == "jrm_adm":
            return cls.jrm_adm(det_bin=1 if det_bin is None else det_bin,
                               n_views=120 if n_views == 660 else n_views)
        if name == "halcyon":
            return cls.halcyon(det_bin=2 if det_bin is None else det_bin, n_views=n_views,
                               scan_duration_sec=scan_duration_sec)
        if name == "obi":
            return cls(det_bin=1 if det_bin is None else det_bin, n_views=n_views,
                       scan_duration_sec=scan_duration_sec)
        if name == "spare_mc":
            return cls.spare_mc(det_bin=1 if det_bin is None else det_bin,
                                n_views=680 if n_views == 660 else n_views,
                                scan_duration_sec=scan_duration_sec)
        raise ValueError(f"unknown geometry preset {name!r} "
                         f"(thies|jrm_adm|halcyon|obi|spare_mc)")

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

    # ---- where FDK is actually defined -------------------------------------------
    def valid_recon_shape(self, spacing, *, fov_frac: float = 1.0,
                          axial_frac: float = 1.0, max_shape=None) -> tuple[int, int, int]:
        """Largest (D,H,W) box, centred on the isocentre at `spacing` mm, that stays inside
        the measured region: the transaxial FOV cylinder and the axial cone coverage.

        The RECON grid and the OBJECT grid are DIFFERENT choices and want opposite things.
        The object fed to the forward projector must hold the WHOLE patient, or its line
        integrals are wrong (a diverging cone still sees tissue past the nominal coverage --
        223.3 mm of z at the exit surface vs 198.7 mm at the isocentre). The reconstruction
        grid must stay INSIDE the measured region, because outside it FDK has no data and
        produces cone-angle and truncation artifacts that have nothing to do with motion.
        Conflating the two (one `--grid` for both) is what inflated the recon floor to 0.12-0.44
        NET RMSE and made a laterally-cropped patient's motion compensation diverge.

        The box CIRCUMSCRIBES the FOV cylinder, so its corners still fall outside; pair it with
        `fov_cylinder_mask` for metrics and patch sampling. `fov_frac`/`axial_frac` < 1 back off
        from the exact boundary, where FDK is defined but Feldkamp's approximation is weakest.
        """
        dx, dy, dz = spacing
        r = 0.5 * self.fov_diameter_mm() * float(fov_frac)
        hz = 0.5 * self.axial_coverage_mm() * float(axial_frac)
        shp = (max(2, 2 * int(hz / dz)), max(2, 2 * int(r / dy)), max(2, 2 * int(r / dx)))
        if max_shape is not None:
            shp = tuple(min(a, b) for a, b in zip(shp, max_shape))
        return shp


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


# back-compat alias; the cylinder was wrong at the corners (see above)
fov_cylinder_mask = measured_region_mask


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
    """
    S = source_positions(Pmat)                          # (..., V, 3)
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
