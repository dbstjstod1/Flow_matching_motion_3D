"""Virtual slab volumes from the AAPM head CT slice archive.

The real head-and-neck CBCT data is not here yet. In the meantime the 2D project's AAPM archive
(1626 separate 512x512 HU slices) is stacked back into VOLUMES, because adjacent slices in it are
genuinely adjacent anatomy -- so a run of them is a real, if short, head volume.

Two things had to be MEASURED before that was true, and both are load-bearing:

1. THE ARCHIVE IS CONTAMINATED WITH INTERLOPER SLICES. Sorting by the `imgNNN` index does not
   give a clean stack: roughly every 13th slice belongs to a different level entirely (img1367 is
   near the vertex, img1368 is the skull base, img1369 is back at the vertex). Stacking naively
   puts a skull-base slice inside the brain and the coronal reslice comes out streaked. They are
   detected by the fact that DELETING them reconnects their neighbours --
   rms(x[i-1], x[i+1]) << min(rms(x[i-1], x[i]), rms(x[i], x[i+1])) -- which is a property no
   genuine slice has. 123 of 1626 slices go.

2. THE ARCHIVE IS MANY PATIENTS CONCATENATED. After the interlopers are gone the remaining
   sequence still jumps, at patient boundaries. Splitting where consecutive-slice RMS exceeds
   `break_hu` leaves 10 runs of ~100 slices (892 slices in runs of >= 64), which reslice
   cleanly in coronal and sagittal. That ~100 is almost certainly one patient's stack.

   The two numbers are not interchangeable: without step 1 no threshold works at all (a low one
   shatters the archive into 5-slice fragments, a high one admits the interlopers and streaks
   the volume). Step 1 first, then step 2.

GEOMETRY OF THE STACK. The slices are 512x512 and the head's bounding box measures 358 px across,
so the pixel size is ~0.5 mm (a head is ~180 mm wide) -- NOT the 1.0 mm the 2D project assumed,
which would have made the head 358 mm wide and unable to fit any real CBCT's 26 cm FOV. We
2x-downsample in plane to land on an ISOTROPIC 1 mm grid, 256x256 per slice = a 256 mm field
holding a 180 mm head.

`dz` IS AN ASSUMPTION. The archive carries no slice thickness; 1.0 mm is the default and is what
makes the grid isotropic. It only affects how much cone angle the slab subtends, not whether any
of the algebra is right. Change it here if the real value turns up.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import torch

from .geometry_3d import ConeBeam3DConfig, build_conebeam_orbit, detector_coords_3d
from .filters import calibrate_scale
from .projector_3d import (adjoint_project_3d_batched, fdk_conebeam_3d_batched,
                           fdk_conebeam_3d_tangent, forward_project_3d_batched)
from .rigid_motion import params_to_Pmot, random_motion

MU_WATER = 0.02


def _scan_runs(files: list[str], slices: np.ndarray, break_hu: float,
               interloper_hu: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """-> (kept indices into `slices`, list of (start, stop) runs over the KEPT array)."""
    def rms(a, b):
        return float(np.sqrt(((a - b) ** 2).mean()))

    n = len(slices)
    d = np.array([rms(slices[i], slices[i + 1]) for i in range(n - 1)])

    bad = []
    for i in range(1, n - 1):
        dl, dr = d[i - 1], d[i]
        if min(dl, dr) > interloper_hu and rms(slices[i - 1], slices[i + 1]) < 0.5 * min(dl, dr):
            bad.append(i)
    keep = np.setdiff1d(np.arange(n), np.array(bad, dtype=int))

    s = slices[keep]
    d2 = np.array([rms(s[i], s[i + 1]) for i in range(len(s) - 1)])
    brk = np.where(d2 > break_hu)[0]
    edges = [0] + [int(b) + 1 for b in brk] + [len(s)]
    runs = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    return keep, runs


class AAPMSlabGenerator:
    """Stacks contiguous AAPM head slices into slab volumes and projects them.

    The 3D twin of the 2D `AAPMPairGenerator`, and it keeps that class's normalization contract
    exactly: PHYSICS in mu [1/mm], NETWORK in [-1, 1] under a fixed affine with no clipping.

        mu       = (HU/1000 + 1) * mu_water
        to_net   = 2 (mu - mu_lo) / (mu_hi - mu_lo) - 1        over the HU window hu_norm
    """

    def __init__(self, data_dir: str, cfg: ConeBeam3DConfig | None = None, *,
                 device="cuda", slab: int = 64, in_plane: int = 256, dz: float = 1.0,
                 pixel_mm: float = 0.5,
                 mu_water: float = MU_WATER, hu_norm=(-1000.0, 2000.0),
                 min_run: int = 64, break_hu: float = 400.0, interloper_hu: float = 300.0,
                 cache: str | None = None):
        self.device = torch.device(device)
        self.slab, self.size = slab, in_plane
        self.mu_water = mu_water

        # 512 px at ~0.5 mm -> `in_plane` px; keep the physical extent, change only the sampling.
        self.dx = self.dy = pixel_mm * 512.0 / in_plane
        self.dz = dz

        lo, hi = hu_norm
        self.mu_lo = (lo / 1000.0 + 1.0) * mu_water
        self.mu_hi = (hi / 1000.0 + 1.0) * mu_water

        self.cfg = cfg or ConeBeam3DConfig(det_bin=2, n_views=360)
        self.P_nom = build_conebeam_orbit(self.cfg, device=self.device)
        self.u_coords, self.v_coords = detector_coords_3d(self.cfg, device=self.device)

        files = []
        for sub in sorted(os.listdir(data_dir)):
            files += glob.glob(os.path.join(data_dir, sub, "*.raw"))
        files += glob.glob(os.path.join(data_dir, "*.raw"))
        if not files:
            raise FileNotFoundError(f"no .raw slices under {data_dir}")
        files.sort(key=lambda f: int(re.search(r"img(\d+)_", os.path.basename(f)).group(1)))

        raw = np.stack([np.fromfile(f, dtype=np.float32).reshape(512, 512) for f in files])

        cache = cache or os.path.join(data_dir, "_slab_runs.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            keep, runs = z["keep"], [tuple(r) for r in z["runs"]]
        else:
            keep, runs = _scan_runs(files, raw, break_hu, interloper_hu)
            np.savez(cache, keep=keep, runs=np.array(runs))

        hu = raw[keep]
        # in-plane 2x box-downsample (0.5 -> 1.0 mm) on the CPU, once
        t = torch.from_numpy(hu)[:, None]
        if in_plane != 512:
            t = torch.nn.functional.avg_pool2d(t, 512 // in_plane)
        self.mu_all = ((t[:, 0] / 1000.0 + 1.0) * mu_water).contiguous()      # (N, h, w) CPU

        self.runs = [(a, b) for (a, b) in runs if b - a >= max(min_run, slab)]
        if not self.runs:
            raise RuntimeError(f"no contiguous run of >= {max(min_run, slab)} slices")
        self.n_slabs = sum((b - a) - slab + 1 for a, b in self.runs)

        # NO CALIBRATION (user, 2026-07-28; the 4DCT sibling deleted the same thing 2026-07-21).
        # The FDK is SELF-NORMALIZED by the pure geometry constant SOD*SDD/2
        # (`projector_3d._fdk_physical_norm`, phantom-verified to 0.2%), exactly like RTK --
        # SOD and SDD are GIVEN, so there is nothing to fit and no scale attribute at all. The old least-squares scale silently absorbed any operator or
        # level error instead of surfacing it.

    # -- volumes -----------------------------------------------------------------------
    def volume(self, run: int = 0, z0: int = 0) -> torch.Tensor:
        """(1, 1, D, H, W) mu volume: `slab` contiguous slices from run `run`, starting at z0."""
        a, b = self.runs[run % len(self.runs)]
        z0 = int(np.clip(z0, 0, (b - a) - self.slab))
        v = self.mu_all[a + z0: a + z0 + self.slab]
        return v[None, None].to(self.device)

    def sample_volumes(self, batch: int = 1, generator=None) -> torch.Tensor:
        """(B, 1, D, H, W) random slabs."""
        out = []
        for _ in range(batch):
            r = int(torch.randint(len(self.runs), (1,), generator=generator))
            a, b = self.runs[r]
            z0 = int(torch.randint((b - a) - self.slab + 1, (1,), generator=generator))
            out.append(self.volume(r, z0))
        return torch.cat(out, 0)

    # -- net <-> mu --------------------------------------------------------------------
    def to_net(self, mu):
        return 2.0 * (mu - self.mu_lo) / (self.mu_hi - self.mu_lo) - 1.0

    def to_net_tangent(self, dmu):
        """to_net is affine, so a DERIVATIVE maps with the gain only (no -1 shift)."""
        return 2.0 * dmu / (self.mu_hi - self.mu_lo)

    def from_net(self, x):
        return (x + 1.0) * 0.5 * (self.mu_hi - self.mu_lo) + self.mu_lo

    # -- operators ---------------------------------------------------------------------
    @property
    def shape(self):
        return (self.slab, self.size, self.size)

    def project(self, vols: torch.Tensor, Pmat: torch.Tensor, **kw) -> torch.Tensor:
        return forward_project_3d_batched(
            vols, Pmat, self.u_coords, self.v_coords,
            dx=self.dx, dy=self.dy, dz=self.dz, **kw)

    def adjoint(self, sino: torch.Tensor, Pmat: torch.Tensor, **kw) -> torch.Tensor:
        """A^T sino -- LEAP's backprojection, NOT the exact transpose of `project`
        (`leap_projector.ADJOINT_MODE`). (B,V,nv,nu) -> (B,1,D,H,W)."""
        D, H, W = self.shape
        return adjoint_project_3d_batched(
            sino, Pmat, self.u_coords, self.v_coords,
            D=D, H=H, W=W, dx=self.dx, dy=self.dy, dz=self.dz, **kw)


    def fdk(self, sino: torch.Tensor, Pmat: torch.Tensor, *, scale=None, **kw) -> torch.Tensor:
        D, H, W = self.shape
        return fdk_conebeam_3d_batched(
            sino, Pmat, self.u_coords, self.v_coords, self.cfg,
            D=D, H=H, W=W, dx=self.dx, dy=self.dy, dz=self.dz,
            scale=scale,
            view_chunk=kw.pop("view_chunk", 8), **kw)

    def fdk_tangent(self, sino: torch.Tensor, Pmat: torch.Tensor, Pdot: torch.Tensor, *,
                    scale=None, **kw):
        """(FDK, its exact directional derivative along Pdot); no angular weighting here
        (this generator never derives one -- see the cq500 twin for the weighted variant)."""
        D, H, W = self.shape
        return fdk_conebeam_3d_tangent(
            sino, Pmat, Pdot, self.u_coords, self.v_coords, self.cfg,
            D=D, H=H, W=W, dx=self.dx, dy=self.dy, dz=self.dz,
            scale=scale, **kw)


    def sample_motion(self, batch: int = 1, *, trans_mm: float = 10.0, rot_deg: float = 6.0,
                      amp_mode: str = "fixed", generator=None):
        """-> (y (B,V,nv,nu), theta (B,V,6), vols (B,1,D,H,W)) with independent random motion.

        Returns the SINOGRAM and the MOTION, not an image pair: the geometry bridge synthesizes
        every x_t on the fly as FDK(y, P_nom @ T(t*theta)), so a precomputed x0/x1 pair would be
        both redundant and wrong (it would fix the path).

        `amp_mode="thies"` switches to the paper's TRAINING amplitude protocol, where the
        amplitudes become per-DoF maxima -- see `rigid_motion.akima_motion`.
        """
        vols = self.sample_volumes(batch, generator=generator)
        th = torch.stack([
            random_motion(self.cfg.n_views, trans_mm=trans_mm, rot_deg=rot_deg,
                          amp_mode=amp_mode, device=self.device, generator=generator)
            for _ in range(batch)], 0)                                        # (B,V,6)
        P = torch.stack([params_to_Pmot(th[b], self.P_nom) for b in range(batch)], 0)
        with torch.no_grad():
            y = self.project(vols, P)
        return y, th, vols
