"""3D-patch FM prior evaluation over a full volume (prompt 6, Phase 5 / P3).

Literature: DiffusionBlend++ (Song et al., NeurIPS 2024) — evaluate the score/FM
prior on overlapping 3D patches and BLEND the per-patch predictions so the full
volume stays consistent (no seam artifacts, z-consistency for free because the
patches are 3D). Here we use the simpler overlap-tile weighted average (separable
raised-Hann window, weights normalized to 1 everywhere), which is exact for the
identity map: if the per-patch prediction equals the patch content, the blended
volume equals the input to float precision (sanity 5a).

Everything stays in IMAGE/VOXEL space (NET normalization of the training bridge),
so the blended x1_hat plugs straight into the continuous 3D MoCo DC step — no
decode, no latent (the stated reason for patches over a latent prior).

FM parameterization is the project's clean endpoint: x1_hat = x_t + (1-t) * v.
"""

from __future__ import annotations

import torch


def _positions(n: int, patch: int, stride: int) -> list[int]:
    """Start offsets covering [0, n) with the last patch flush to the border."""
    if n <= patch:
        return [0]
    pos = list(range(0, n - patch, stride))
    if pos[-1] != n - patch:
        pos.append(n - patch)
    return pos


def _hann_window_3d(p: tuple[int, int, int], device, eps: float = 1e-2) -> torch.Tensor:
    """Separable raised-Hann weight (1,1,pd,ph,pw), strictly positive (eps floor so
    border voxels of the VOLUME, covered by only one patch edge, keep weight)."""
    ws = []
    for n in p:
        i = torch.arange(n, device=device, dtype=torch.float32)
        ws.append(0.5 - 0.5 * torch.cos(2 * torch.pi * (i + 0.5) / n) + eps)
    w = ws[0][:, None, None] * ws[1][None, :, None] * ws[2][None, None, :]
    return w[None, None]


@torch.no_grad()
def predict_x1_patched(model, x_t: torch.Tensor, t: float, *, patch: int = 64,
                       stride: int | None = None, batch: int = 8) -> torch.Tensor:
    """Blended clean-endpoint prediction of the 3D-patch FM prior over a volume.

    model : UNet3D velocity net (NET space)   x_t : (1,1,D,H,W) NET
    t     : FM ODE time in [0,1)              ->    x1_hat (1,1,D,H,W) NET

    Splits the volume into overlapping `patch`^3 tiles (default stride = patch/2),
    computes x1_hat = x_t + (1-t)*v per tile, and overlap-blends with a Hann
    window normalized by the accumulated weight (== 1 everywhere)."""
    assert x_t.ndim == 5 and x_t.shape[:2] == (1, 1)
    device = x_t.device
    _, _, D, H, W = x_t.shape
    pd = min(patch, D)
    ph = min(patch, H)
    pw = min(patch, W)
    stride = stride or max(1, patch // 2)
    pos = [(z, y, x)
           for z in _positions(D, pd, min(stride, pd))
           for y in _positions(H, ph, min(stride, ph))
           for x in _positions(W, pw, min(stride, pw))]

    win = _hann_window_3d((pd, ph, pw), device)                 # (1,1,pd,ph,pw)
    acc = torch.zeros_like(x_t)
    wacc = torch.zeros_like(x_t)
    t_t = torch.full((1,), float(t), device=device)

    for c0 in range(0, len(pos), batch):
        chunk = pos[c0:c0 + batch]
        tiles = torch.cat([x_t[:, :, z:z + pd, y:y + ph, x:x + pw]
                           for (z, y, x) in chunk], dim=0)      # (b,1,pd,ph,pw)
        v = model(tiles, t_t.expand(tiles.shape[0]))
        x1 = tiles + (1.0 - float(t)) * v                       # clean endpoint
        for i, (z, y, x) in enumerate(chunk):
            acc[:, :, z:z + pd, y:y + ph, x:x + pw] += x1[i:i + 1] * win
            wacc[:, :, z:z + pd, y:y + ph, x:x + pw] += win
    return acc / wacc.clamp_min(1e-8)


@torch.no_grad()
def sample_patch_coords(shape_dhw, patch: int, n: int, generator=None, device="cpu"):
    """n random aligned crop origins (z,y,x) for training patch extraction."""
    D, H, W = shape_dhw
    zs = torch.randint(0, max(D - patch, 0) + 1, (n,), generator=generator, device=device)
    ys = torch.randint(0, max(H - patch, 0) + 1, (n,), generator=generator, device=device)
    xs = torch.randint(0, max(W - patch, 0) + 1, (n,), generator=generator, device=device)
    return list(zip(zs.tolist(), ys.tolist(), xs.tolist()))


def crop_pairs(x_t: torch.Tensor, x1: torch.Tensor, coords, patch: int):
    """Aligned patch pairs from two (1,1,D,H,W) volumes -> two (n,1,p,p,p) tensors."""
    a = torch.cat([x_t[:, :, z:z + patch, y:y + patch, x:x + patch]
                   for (z, y, x) in coords], dim=0)
    b = torch.cat([x1[:, :, z:z + patch, y:y + patch, x:x + patch]
                   for (z, y, x) in coords], dim=0)
    return a, b
