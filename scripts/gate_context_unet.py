"""Gate the global-context patch prior upgrade (arXiv:2512.18161). Synthetic, CPU, no data.

What changed (and what can silently break): patches now carry 4 conditioning channels
(downsampled full volume + 3 absolute-coordinate channels), `predict_x1_patched` slices
channel 0 for the FM endpoint, and the tile grid can be jitter-averaged (`n_offsets`).
The failure modes are all bookkeeping — a mis-sliced channel, a coordinate map built
from the TILE index instead of the VOLUME index, a jittered grid that stops covering
the borders — so every check here is exact-by-construction, not statistical.

  [1] back-compat: in_ch=1 net unchanged (shape, state_dict round-trip);
      zero-velocity blend == identity, jittered or not (the old sanity 5a)
  [2] tile-grid coverage: _positions with random offsets always covers [0, n)
      with both borders flush
  [3] 5-channel assembly: make_tile_inputs / crop_pairs(ctx) channels are exactly
      [crop, volume_context, tile_coords]; target stays 1-channel
  [4] coordinate channels: a probe net that RETURNS its coord channel makes the
      blended volume x_t + (the analytic coordinate map) EXACTLY — overlapping
      tiles must agree wherever they overlap, on a non-cubic volume, jittered too
  [5] channel-0 slicing + auto context: in_ch=5 net runs end-to-end through
      predict_x1_patched(context="auto"); zero-velocity == identity again
  [6] determinism: same generator seed -> bit-identical jittered prediction
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.unet_3d import UNet3D
from fm3d.prior_patch import (_positions, crop_pairs, make_tile_inputs,
                              model_in_channels, predict_x1_patched, tile_coords,
                              volume_context)

FAIL = []


def check(idx, name, ok, detail=""):
    print(f"[{idx}] {name:<58s} {'PASS' if ok else 'FAIL'}  {detail}")
    if not ok:
        FAIL.append(name)


class ZeroVel(nn.Module):
    """v = 0 -> x1_hat == x_t: blending must be the identity (sanity 5a)."""
    def __init__(self, in_ch=1):
        super().__init__()
        self.in_conv = nn.Conv3d(in_ch, 1, 1)   # only read by model_in_channels

    def forward(self, x, t):
        return torch.zeros_like(x[:, :1])


class CoordProbe(nn.Module):
    """v = the coord channel `axis` -> blended == x_t + analytic coordinate map."""
    def __init__(self, axis):
        super().__init__()
        self.in_conv = nn.Conv3d(5, 1, 1)
        self.axis = axis

    def forward(self, x, t):
        return x[:, 2 + self.axis:3 + self.axis]


def main():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)

    # ---- [1] back-compat ------------------------------------------------------------------
    net1 = UNet3D(in_ch=1, base=16)
    y = net1(torch.randn(2, 1, 32, 32, 32), torch.tensor([0.1, 0.7]))
    check(1, "in_ch=1 forward shape", tuple(y.shape) == (2, 1, 32, 32, 32))
    net1b = UNet3D(in_ch=1, base=16)
    net1b.load_state_dict(net1.state_dict())
    check(1, "in_ch=1 state_dict round-trip", True)
    check(1, "model_in_channels reads 1", model_in_channels(net1) == 1)

    x = torch.randn(1, 1, 40, 52, 64)
    out = predict_x1_patched(ZeroVel(1), x, 0.3, patch=32, stride=16, context="none")
    e = float((out - x).abs().max())
    check(1, "zero-velocity blend == identity (context=none)", e < 1e-5, f"max|d|={e:.1e}")
    out = predict_x1_patched(ZeroVel(1), x, 0.3, patch=32, stride=16, context="none",
                             n_offsets=3, generator=g)
    e = float((out - x).abs().max())
    check(1, "…with 3 jittered grids", e < 1e-5, f"max|d|={e:.1e}")

    # ---- [2] jittered grid coverage --------------------------------------------------------
    ok = True
    for _ in range(200):
        n = int(torch.randint(8, 200, (1,), generator=g))
        p = int(torch.randint(4, min(n, 65), (1,), generator=g))
        s = int(torch.randint(1, p + 1, (1,), generator=g))
        off = int(torch.randint(0, s, (1,), generator=g))
        pos = _positions(n, p, s, off)
        ok &= pos[0] == 0 and pos[-1] == n - p and all(0 <= q <= n - p for q in pos)
        ok &= all(b - a <= p for a, b in zip(pos, pos[1:]))     # union covers [0, n)
    check(2, "_positions covers [0,n) flush at both borders", ok)

    # ---- [3] 5-channel assembly ------------------------------------------------------------
    psz = (32, 32, 32)
    ctx = volume_context(x, psz)
    check(3, "volume_context shape", tuple(ctx.shape) == (1, 1, 32, 32, 32))
    coords = [(0, 0, 0), (8, 20, 32), (8, 20, 31)]
    tiles = make_tile_inputs(x, coords, psz, ctx)
    check(3, "make_tile_inputs shape", tuple(tiles.shape) == (3, 5, 32, 32, 32))
    e0 = max(float((tiles[i, 0] - x[0, 0, z:z + 32, y:y + 32, xx:xx + 32]).abs().max())
             for i, (z, y, xx) in enumerate(coords))
    check(3, "channel 0 == raw crop", e0 == 0.0)
    e1 = float((tiles[:, 1] - ctx[0, 0]).abs().max())
    check(3, "channel 1 == volume_context (shared)", e1 == 0.0)
    cc = tile_coords(coords, psz, x.shape[-3:], x.device)
    e2 = float((tiles[:, 2:5] - cc).abs().max())
    check(3, "channels 2:5 == tile_coords", e2 == 0.0)
    x1v = torch.randn_like(x)
    a5, b1 = crop_pairs(x, x1v, coords, 32, ctx=ctx)
    a1, b1r = crop_pairs(x, x1v, coords, 32)
    check(3, "crop_pairs(ctx): input C=5, target C=1, ch0/target aligned",
          a5.shape[1] == 5 and b1.shape[1] == 1
          and float((a5[:, :1] - a1).abs().max()) == 0.0
          and float((b1 - b1r).abs().max()) == 0.0)

    # ---- [4] coordinate channels are ABSOLUTE (blend of a coord probe is exact) ------------
    D, H, W = x.shape[-3:]
    grids = [(2.0 * (torch.arange(D) + 0.5) / D - 1.0)[:, None, None].expand(D, H, W),
             (2.0 * (torch.arange(H) + 0.5) / H - 1.0)[None, :, None].expand(D, H, W),
             (2.0 * (torch.arange(W) + 0.5) / W - 1.0)[None, None, :].expand(D, H, W)]
    for ax, name in enumerate("zyx"):
        want = x + grids[ax][None, None]
        got = predict_x1_patched(CoordProbe(ax), x, 0.0, patch=32, stride=16,
                                 context="global")
        e = float((got - want).abs().max())
        check(4, f"coord-{name} probe: blend == x_t + analytic map", e < 1e-5,
              f"max|d|={e:.1e}")
        got = predict_x1_patched(CoordProbe(ax), x, 0.0, patch=32, stride=16,
                                 context="global", n_offsets=3, generator=g)
        e = float((got - want).abs().max())
        check(4, f"…coord-{name}, 3 jittered grids", e < 1e-5, f"max|d|={e:.1e}")

    # ---- [5] channel-0 slicing + auto context through a REAL in_ch=5 net -------------------
    net5 = UNet3D(in_ch=5, base=16)
    y = net5(torch.randn(2, 5, 32, 32, 32), torch.tensor([0.1, 0.7]))
    check(5, "in_ch=5 forward shape (velocity stays 1-channel)",
          tuple(y.shape) == (2, 1, 32, 32, 32))
    check(5, "model_in_channels reads 5", model_in_channels(net5) == 5)
    out = predict_x1_patched(net5, x, 0.2, patch=32, stride=16, context="auto")
    check(5, "context=auto -> global path runs end-to-end",
          tuple(out.shape) == tuple(x.shape) and torch.isfinite(out).all())
    out = predict_x1_patched(ZeroVel(5), x, 0.2, patch=32, stride=16, context="auto")
    e = float((out - x).abs().max())
    check(5, "zero-velocity == identity through the 5-ch path", e < 1e-5,
          f"max|d|={e:.1e}")

    # ---- [6] determinism of the jittered grids ---------------------------------------------
    ga = torch.Generator().manual_seed(7)
    gb = torch.Generator().manual_seed(7)
    oa = predict_x1_patched(net5, x, 0.2, patch=32, stride=16, n_offsets=3, generator=ga)
    ob = predict_x1_patched(net5, x, 0.2, patch=32, stride=16, n_offsets=3, generator=gb)
    check(6, "same seed -> bit-identical jittered prediction",
          float((oa - ob).abs().max()) == 0.0)

    n = len(FAIL)
    print(f"\n{'ALL PASS' if n == 0 else f'{n} FAILURE(S): ' + ', '.join(FAIL)}")
    sys.exit(1 if n else 0)


if __name__ == "__main__":
    main()
