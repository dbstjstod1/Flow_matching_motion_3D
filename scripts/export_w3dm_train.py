"""Export OUR CQ500 train split to JRM-ADM's training format (A2, 2026-08-13).

Their W3DM prior trains on per-patient .npy volumes storing RAW HU at 160x192x192 @ ~1 mm
(verified against their shipped demo: head.npy is HU, min -1282 / max +3347; their CTDataset
clips to [-1000, 2000] and normalizes at LOAD time). We therefore export HU, not mu, which
also removes the mu_water mismatch (ours 0.02, theirs 0.0193) -- their pipeline applies its
own constant downstream.

Crop: our volumes are 256^3 @ 1 mm; theirs 160(z) x 192 x 192. The paper does not publish its
crop rule, so this is OUR convention, stated openly and chosen to match their shipped demo's
framing (vertex intact, inferior neck cut, head filling the frame):

  z    : VERTEX-ANCHORED -- the window ends 4 voxels above the highest foreground slice
         (HU > -500), so the top of the skull is never cut; what falls off is neck/skull base.
         Measured on our train volumes the head's z-extent is 150-160 mm, i.e. the 160 window
         is tight and a center-of-mass crop clips the vertex (probed 2026-08-13).
  y, x : centered on the BONE center of mass (HU > 300) -- the cranium's center, immune to
         shoulders/headrest that stretch the soft-tissue bounding box past 192 mm.

The exported volume is z-FLIPPED: their convention runs superior -> inferior (their head.npy:
z=10 is the vertex, z=150 the sinuses; probed 2026-08-13), ours the opposite. Exporting in
THEIR orientation lets their released weights run on our volumes and keeps the retrained
prior drop-in for their solver stack. Residual domain gap vs their data, stated openly: our
volumes contain the CT table/headrest, theirs are patient-only -- self-consistent for the
retrained arm, a mismatch only if their released weights are scored on our exports.

HU comes back out of our stored mu via hu = (mu / MU_WATER - 1) * 1000 with OUR MU_WATER=0.02
(dataset_cq500 stores mu made with that constant, window already clipped to [-1000, 2000] --
identical to their clip, so the round trip is exact on the window).

    python scripts/export_w3dm_train.py [--out refs/jrm-adm/data/train_volumes] [--limit N]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_cq500 import MU_WATER, CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig

OUT_SHAPE = (160, 192, 192)          # their (D, H, W)


def crop_origin(vol_hu: torch.Tensor, out_shape) -> tuple:
    """Vertex-anchored z, bone-COM-centered in-plane; clamped to bounds (see module doc)."""
    D, H, W = vol_hu.shape
    fg = vol_hu > -500.0
    bone = (vol_hu > 300.0).float()
    if fg.sum() < 1 or bone.sum() < 1:                   # degenerate: fall back to center
        return tuple((s - o) // 2 for s, o in zip(vol_hu.shape, out_shape))
    z_top = int(fg.any(dim=1).any(dim=1).nonzero().max())
    oz = max(0, min(z_top + 4 - (out_shape[0] - 1), D - out_shape[0]))
    idx = [torch.arange(s, device=vol_hu.device, dtype=torch.float32) for s in vol_hu.shape]
    com = [float((bone.sum(dim=[d for d in range(3) if d != a]) * idx[a]).sum() / bone.sum())
           for a in range(3)]
    oy = int(max(0, min(round(com[1] - out_shape[1] / 2), H - out_shape[1])))
    ox = int(max(0, min(round(com[2] - out_shape[2] / 2), W - out_shape[2])))
    return oz, oy, ox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/CQ500")
    ap.add_argument("--out", default="refs/jrm-adm/data/train_volumes")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0, help="export only the first N (0 = all)")
    args = ap.parse_args()

    cfg = ConeBeam3DConfig.thies(n_views=360)
    gen = CQ500Generator(args.root, cfg, device="cuda", split=args.split,
                         shape=(256, 256, 256), voxel_mm=1.0, sim_native=True, verbose=False)
    os.makedirs(args.out, exist_ok=True)

    n = len(gen.records) if args.limit == 0 else min(args.limit, len(gen.records))
    for i in range(n):
        pid = gen.records[i]["patient"]
        with torch.no_grad():
            mu = gen.volume(i)[0, 0]                              # (256,256,256) mu
            hu = (mu / MU_WATER - 1.0) * 1000.0
            oz, oy, ox = crop_origin(hu, OUT_SHAPE)
            crop = hu[oz:oz + OUT_SHAPE[0], oy:oy + OUT_SHAPE[1], ox:ox + OUT_SHAPE[2]]
            crop = torch.flip(crop, dims=(0,))            # -> their superior-first z order
        path = os.path.join(args.out, f"p{pid:04d}.npy")
        np.save(path, crop.cpu().numpy().astype(np.float32))
        print(f"[{i + 1:3d}/{n}] p{pid:04d} origin ({oz},{oy},{ox}) "
              f"hu [{float(crop.min()):7.1f}, {float(crop.max()):7.1f}] -> {path}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
