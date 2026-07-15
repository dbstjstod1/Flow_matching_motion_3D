"""3D-patch FM prior evaluation over a full volume (prompt 6, Phase 5 / P3).

Literature: DiffusionBlend++ (Song et al., NeurIPS 2024) — evaluate the score/FM
prior on overlapping 3D patches and BLEND the per-patch predictions so the full
volume stays consistent (no seam artifacts, z-consistency for free because the
patches are 3D). Here we use the simpler overlap-tile weighted average (separable
raised-Hann window, weights normalized to 1 everywhere), which is exact for the
identity map: if the per-patch prediction equals the patch content, the blended
volume equals the input to float precision (sanity 5a).

Everything stays in IMAGE/VOXEL space (NET normalization of the geometry bridge),
so the blended x1_hat plugs straight into the PnP-TV posterior loop's predict step
— no decode, no latent (the stated reason for patches over a latent prior).

`x1_hat = x_t + (1-t) * v` IS NOT A CLEAN ENDPOINT IN THIS PROJECT, and the name is
inherited from Flowmatching-4DCT, where it is one (4DCT regresses the endpoint:
L = ||x_t + (1-t)v - x_clean||^2, so its v points AT x_clean by construction). Here
`train_fm3d.py` regresses v on the TRUE TANGENT of the geometry bridge, dx_t/dt of a
FDK path that is curved — so x_t + (1-t)v is a first-order EXTRAPOLATION along it,
not the endpoint.

That does not make the blend wrong, and the reason is worth writing down because it
looks wrong. v -> x1 is affine with a CONSTANT coefficient, and the blend is a weighted
average whose weights are normalized to 1, so the map passes straight through it:

    blend(x1)_j = sum_i w_ij (x_t|_i + (1-t) v_i)_j / sum_i w_ij
                = x_t,j + (1-t) * blend(v)_j        <- every tile's ch-0 crop is the
                                                       SAME x_t at voxel j
    => (blend(x1) - x_t) / (1-t) = blend(v)         EXACTLY

so the (1-t) cancels, x1 is a scratch variable that is never consumed as an endpoint,
and what actually gets blended is v — which is the only thing the net emits. MEASURED
against blending v directly: agreement to 6e-6 relative at worst (t = 29/30). Do not
"fix" this. Do NOT, however, start using x1_hat as if it were a clean image (to score
it, to feed it to a denoiser, to show it in a montage): for a tangent-trained v it is
not one. The one real cost of the round-trip is conditioning — the cancellation error
grows as 1/(1-t), 8e-7 at t=0 to 6e-6 at t=29/30, which is free in fp32 and would not
be under AMP.

GLOBAL CONTEXT (Local Patches Meet Global Context, arXiv:2512.18161): a patch
alone cannot know where it sits in the head nor what the rest of the volume looks
like, and the paper measures that gap at 2.7x in FID (40.8 -> 112.1 without it).
Their fix — and ours — is 4 extra input channels per patch, all conditioning, no
architecture change:
  ch 1     the WHOLE current volume x_t, trilinearly downsampled to the patch grid
           (recomputed from the evolving x_t at every ODE step, so train == infer)
  ch 2..4  the patch voxels' absolute position in the volume, per axis,
           normalized to (-1, 1)  (voxel centre convention: 2*(i+.5)/N - 1)
The prediction target is unchanged, and the x1_hat expression uses ch 0 only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _positions(n: int, patch: int, stride: int, offset: int = 0) -> list[int]:
    """Start offsets covering [0, n), first and last patch flush to the borders;
    interior starts at offset + k*stride (offset jitters the tile grid)."""
    if n <= patch:
        return [0]
    pos = {0, n - patch}
    p = max(int(offset), 0)
    if p == 0:
        p = stride
    while p < n - patch:
        pos.add(p)
        p += stride
    return sorted(pos)


def _hann_window_3d(p: tuple[int, int, int], device, eps: float = 1e-2) -> torch.Tensor:
    """Separable raised-Hann weight (1,1,pd,ph,pw), strictly positive (eps floor so
    border voxels of the VOLUME, covered by only one patch edge, keep weight)."""
    ws = []
    for n in p:
        i = torch.arange(n, device=device, dtype=torch.float32)
        ws.append(0.5 - 0.5 * torch.cos(2 * torch.pi * (i + 0.5) / n) + eps)
    w = ws[0][:, None, None] * ws[1][None, :, None] * ws[2][None, None, :]
    return w[None, None]


def volume_context(x_vol: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    """The global-context channel: the whole (1,1,D,H,W) volume trilinearly
    downsampled onto the patch grid. Axis scale factors differ on our anisotropic
    grids — the aspect distortion is absorbed by the coordinate channels, which
    tell the net where each patch voxel sits in absolute position."""
    return F.interpolate(x_vol, size=tuple(int(s) for s in size),
                         mode="trilinear", align_corners=False)


def tile_coords(coords, psize: tuple[int, int, int], vol_shape: tuple[int, int, int],
                device, dtype=torch.float32) -> torch.Tensor:
    """(n,3,pd,ph,pw) absolute-position channels for tiles with origins `coords`
    (z,y,x). Value = 2*(i+0.5)/N - 1 in (-1,1), i the VOLUME voxel index — so
    overlapping tiles agree exactly wherever they overlap."""
    pd, ph, pw = psize
    D, H, W = (int(n) for n in vol_shape)
    iz = torch.arange(pd, device=device, dtype=dtype) + 0.5
    iy = torch.arange(ph, device=device, dtype=dtype) + 0.5
    ix = torch.arange(pw, device=device, dtype=dtype) + 0.5
    out = []
    for (z, y, x) in coords:
        cz = (2.0 * (z + iz) / D - 1.0)[:, None, None].expand(pd, ph, pw)
        cy = (2.0 * (y + iy) / H - 1.0)[None, :, None].expand(pd, ph, pw)
        cx = (2.0 * (x + ix) / W - 1.0)[None, None, :].expand(pd, ph, pw)
        out.append(torch.stack([cz, cy, cx]))
    return torch.stack(out)                                     # (n,3,pd,ph,pw)


def model_in_channels(model) -> int:
    """Input channels the net was built with (1 = patch only, 5 = +context+coords)."""
    return int(getattr(getattr(model, "in_conv", None), "in_channels", 1))


def make_tile_inputs(x_t: torch.Tensor, coords, psize: tuple[int, int, int],
                     ctx: torch.Tensor | None) -> torch.Tensor:
    """Crop tiles at `coords` from (1,1,D,H,W) and, when `ctx` is given, append the
    shared global-context channel and the 3 coordinate channels -> (n,5,pd,ph,pw)."""
    pd, ph, pw = psize
    tiles = torch.cat([x_t[:, :, z:z + pd, y:y + ph, x:x + pw]
                       for (z, y, x) in coords], dim=0)         # (n,1,pd,ph,pw)
    if ctx is None:
        return tiles
    n = tiles.shape[0]
    cc = tile_coords(coords, psize, x_t.shape[-3:], x_t.device, x_t.dtype)
    return torch.cat([tiles, ctx.expand(n, -1, -1, -1, -1), cc], dim=1)


def _grid_offsets(psize, stride, n_offsets: int, generator=None):
    """Tile-grid phases: (0,0,0) first (deterministic), then random in [1, stride)."""
    dev = generator.device if generator is not None else "cpu"
    offs = [(0, 0, 0)]
    for _ in range(n_offsets - 1):
        o = []
        for p in psize:
            s = min(stride, p)
            o.append(int(torch.randint(1, s, (1,), generator=generator,
                                       device=dev).item()) if s > 1 else 0)
        offs.append(tuple(o))
    return offs


@torch.no_grad()
def predict_x1_patched(model, x_t: torch.Tensor, t: float, *, patch: int = 64,
                       stride: int | None = None, batch: int = 8,
                       context: str = "auto", n_offsets: int = 1,
                       generator=None) -> torch.Tensor:
    """Blended clean-endpoint prediction of the 3D-patch FM prior over a volume.

    model : UNet3D velocity net (NET space)   x_t : (1,1,D,H,W) NET
    t     : FM ODE time in [0,1)              ->    x1_hat (1,1,D,H,W) NET

    Splits the volume into overlapping `patch`^3 tiles (default stride = patch/2),
    computes x1_hat = x_t + (1-t)*v per tile, and overlap-blends with a Hann
    window normalized by the accumulated weight (== 1 everywhere).

    context   : "global" appends the downsampled-volume + coordinate channels
                (nets trained with in_ch=5); "none" is the bare-patch path;
                "auto" reads the net's in_conv width and picks accordingly.
    n_offsets : >1 additionally blends tiles from (n_offsets-1) randomly SHIFTED
                tile grids (the FM analogue of the paper's recurrent noising,
                K=2 optimal there) — kills any fixed-grid artifact. Offset 0 is
                always the deterministic flush grid; pass `generator` to make
                the extra offsets reproducible."""
    assert x_t.ndim == 5 and x_t.shape[:2] == (1, 1)
    if context == "auto":
        context = "global" if model_in_channels(model) >= 5 else "none"
    if context not in ("global", "none"):
        raise ValueError(f"context must be auto|global|none, got {context!r}")
    device = x_t.device
    _, _, D, H, W = x_t.shape
    pd = min(patch, D)
    ph = min(patch, H)
    pw = min(patch, W)
    stride = stride or max(1, patch // 2)
    ctx = volume_context(x_t, (pd, ph, pw)) if context == "global" else None

    win = _hann_window_3d((pd, ph, pw), device)                 # (1,1,pd,ph,pw)
    acc = torch.zeros_like(x_t)
    wacc = torch.zeros_like(x_t)
    t_t = torch.full((1,), float(t), device=device)

    for (oz, oy, ox) in _grid_offsets((pd, ph, pw), stride, n_offsets, generator):
        pos = [(z, y, x)
               for z in _positions(D, pd, min(stride, pd), oz)
               for y in _positions(H, ph, min(stride, ph), oy)
               for x in _positions(W, pw, min(stride, pw), ox)]
        for c0 in range(0, len(pos), batch):
            chunk = pos[c0:c0 + batch]
            tiles = make_tile_inputs(x_t, chunk, (pd, ph, pw), ctx)
            v = model(tiles, t_t.expand(tiles.shape[0]))
            x1 = tiles[:, :1] + (1.0 - float(t)) * v            # clean endpoint (ch 0)
            for i, (z, y, x) in enumerate(chunk):
                acc[:, :, z:z + pd, y:y + ph, x:x + pw] += x1[i:i + 1] * win
                wacc[:, :, z:z + pd, y:y + ph, x:x + pw] += win
    return acc / wacc.clamp_min(1e-8)


@torch.no_grad()
def prior_ode(model, x0: torch.Tensor, *, n_steps: int = 50, patch: int = 64,
              stride: int | None = None, context: str = "auto", n_offsets: int = 1,
              generator=None) -> torch.Tensor:
    """Prior-ONLY Euler integration of the FM ODE, t: 0 -> 1. No data consistency, no TV --
    "what does the prior ALONE make of the cold start". This is the validation metric, the same
    one the sibling 4DCT project renders every `val_every` steps.

    x0 : (1,1,D,H,W) NET-space cold start (the uncorrected FDK, in practice)   -> x1_hat, NET

    The velocity is recovered from the clean-endpoint predictor the training target defines:
    `predict_x1_patched` blends x1_hat = x_t + (1-t)*v, so v = (x1_hat - x_t)/(1-t), and the Euler
    step is x <- x + dt * v. The (1-t) never actually divides here -- we step
    x <- x + (dt/(1-t)) * (x1_hat - x_t) -- so the t -> 1 endpoint is well behaved. On an affine
    path an oracle net emitting the true constant velocity gives x_N = x_1 for any N; on this
    project's CURVED geometry bridge more steps genuinely help, hence the default 50 (the deploy
    loop's own count), against 4DCT's cheaper 10."""
    x = x0
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = k / n_steps
        x1 = predict_x1_patched(model, x, t, patch=patch, stride=stride, context=context,
                                n_offsets=n_offsets, generator=generator)
        x = x + (dt / max(1.0 - t, 1e-3)) * (x1 - x)          # == x + dt * v
    return x


@torch.no_grad()
def sample_patch_coords(shape_dhw, patch: int, n: int, generator=None, device="cpu"):
    """n random aligned crop origins (z,y,x) for training patch extraction."""
    D, H, W = shape_dhw
    zs = torch.randint(0, max(D - patch, 0) + 1, (n,), generator=generator, device=device)
    ys = torch.randint(0, max(H - patch, 0) + 1, (n,), generator=generator, device=device)
    xs = torch.randint(0, max(W - patch, 0) + 1, (n,), generator=generator, device=device)
    return list(zip(zs.tolist(), ys.tolist(), xs.tolist()))


def crop_pairs(x_t: torch.Tensor, x1: torch.Tensor, coords, patch: int,
               ctx: torch.Tensor | None = None):
    """Aligned patch pairs from two (1,1,D,H,W) volumes -> (n,C,p,p,p), (n,1,p,p,p).

    With `ctx` (the (1,1,p,p,p) `volume_context` of x_t) the INPUT patches carry the
    global-context + coordinate channels (C=5); the TARGET stays single-channel."""
    a = make_tile_inputs(x_t, coords, (patch, patch, patch), ctx)
    b = torch.cat([x1[:, :, z:z + patch, y:y + patch, x:x + patch]
                   for (z, y, x) in coords], dim=0)
    return a, b
