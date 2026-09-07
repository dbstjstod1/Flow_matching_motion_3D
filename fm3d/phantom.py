"""A synthetic head phantom, so the geometry gates run with no dataset mounted.

This is scaffolding for the gates and nothing more -- it is NOT a stand-in for the real
head-and-neck CBCT data, and no result measured on it means anything about reconstruction
quality. Its only jobs are (a) to have sharp edges, so a sign error in the geometry shows up
as a visible mirror/shear rather than a small number, and (b) to be dense where a head is
dense, so the FDK scale calibration lands in the right ballpark.

Values are LINEAR ATTENUATION COEFFICIENTS [1/mm] at ~60 keV effective (mu_water = 0.02),
matching the convention the 2D project uses end to end: mu = (HU/1000 + 1) * mu_water.
"""

from __future__ import annotations

import torch

MU_WATER = 0.02   # 1/mm


def hu_to_mu(hu: float, mu_water: float = MU_WATER) -> float:
    return (hu / 1000.0 + 1.0) * mu_water


def head_phantom(
    shape: tuple[int, int, int],          # (D, H, W) = (z, y, x)
    spacing: tuple[float, float, float],  # (dz, dy, dx) [mm]
    device="cpu",
    dtype=torch.float32,
) -> torch.Tensor:
    """(D, H, W) volume of mu [1/mm]. World origin at the volume centre (projector's box)."""
    D, H, W = shape
    dz, dy, dx = spacing
    z = (torch.arange(D, device=device, dtype=torch.float32) - (D - 1) / 2) * dz
    y = (torch.arange(H, device=device, dtype=torch.float32) - (H - 1) / 2) * dy
    x = (torch.arange(W, device=device, dtype=torch.float32) - (W - 1) / 2) * dx
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

    def ellipsoid(cx, cy, cz, rx, ry, rz):
        return (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 + ((zz - cz) / rz) ** 2) <= 1.0

    vol = torch.zeros(shape, device=device, dtype=torch.float32)

    # skull: a bone shell around a soft-tissue brain
    outer = ellipsoid(0, 0, 0, 82, 98, 62)
    inner = ellipsoid(0, 0, 0, 74, 90, 55)
    vol[outer] = hu_to_mu(900.0)      # cortical bone
    vol[inner] = hu_to_mu(35.0)       # brain

    # ventricles (low contrast, tests that motion blur is visible where it matters)
    vol[ellipsoid(-14, 6, 8, 9, 20, 12)] = hu_to_mu(5.0)
    vol[ellipsoid(14, 6, 8, 9, 20, 12)] = hu_to_mu(5.0)

    # air sinuses / nasal cavity -- the strong edges FDK streaks radiate from
    vol[ellipsoid(0, -62, -14, 20, 22, 16)] = hu_to_mu(-1000.0)
    vol[ellipsoid(-26, -48, -22, 10, 12, 10)] = hu_to_mu(-1000.0)
    vol[ellipsoid(26, -48, -22, 10, 12, 10)] = hu_to_mu(-1000.0)

    # petrous bone / dense inclusions: high-contrast, off-axis, near the cone edge
    vol[ellipsoid(-46, -8, -30, 12, 10, 8)] = hu_to_mu(1300.0)
    vol[ellipsoid(46, -8, -30, 12, 10, 8)] = hu_to_mu(1300.0)

    # a few small beads: sub-voxel motion is legible on these before anything else
    for (bx, by, bz) in [(0, 70, 30), (-55, 30, -40), (55, 30, -40), (0, -20, 45)]:
        vol[ellipsoid(bx, by, bz, 4, 4, 4)] = hu_to_mu(1800.0)

    vol[~outer] = 0.0                  # air outside the head
    return vol.to(dtype)
