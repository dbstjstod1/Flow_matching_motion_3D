"""Is the FM prior WEAK, or just BLIND? -- the decisive separation.

At inference the prior travels 85 % of the distance to the target but only cos = 0.68 in the right
DIRECTION (scripts/exp_pure_ode.py), so the carried x_t lags FDK(theta) badly. Two very different
explanations, and they call for opposite fixes:

  (A) TRAINING/CAPACITY. The net never learned the tangent field well. Fix = bigger net / more
      training / better target.
  (B) THE BLIND CONDITION. The net learned its target fine, but at inference it is shown a volume
      whose motion it does NOT know, and the true tangent DEPENDS on that unknown motion
      (dx/dt = d/dt FDK(y, P_nom @ T(t*theta))), so it can only regress the conditional mean over
      all motions consistent with what it sees. Fix = feed the loop's KNOWN theta back into the
      image update; more capacity cannot help.

This script separates them by evaluating the net exactly ON its training distribution: build the
bridge with the TRUE theta (the same `bridge_pair` the trainer calls, same anchor), then ask the
net for the velocity at that x_t and compare it with the EXACT analytic tangent.

    cos = 1, ratio = 1   -> the net learned the field; the inference deficit is (B), blindness.
    cos low ON-BRIDGE    -> (A), a real training/capacity deficit.

The comparison is also run against a MISMATCHED tangent (the same x_t but the tangent of a
DIFFERENT motion) to show what chance looks like on this metric.

    CUDA_VISIBLE_DEVICES=0 python scripts/exp_oracle_tangent.py
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig
from fm3d.prior_patch import predict_x1_patched
from fm3d.rigid_motion import amp_from_run_args, params_to_Pmot, random_motion
from fm3d.unet_3d import UNet3D
from train_fm3d import bridge_pair


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.flatten()[None], b.flatten()[None]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/fm3d_cq500/ckpt_iter500000.pth")
    ap.add_argument("--split", default="val")
    ap.add_argument("--patients", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--blend", default="uniform")
    ap.add_argument("--patch_offsets", type=int, default=2)
    args = ap.parse_args()

    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    ca = ck["args"]
    cfg = ConeBeam3DConfig.thies(n_views=ca["views"])
    gen = CQ500Generator(ca["root"], cfg=cfg, device=dev, split=args.split,
                         shape=tuple(ca["shape"]), voxel_mm=1.0)
    model = UNet3D(in_ch=int(ck["ema"]["in_conv.weight"].shape[1]), base=ca["base"]).to(dev).eval()
    model.load_state_dict(ck["ema"])
    for q in model.parameters():
        q.requires_grad_(False)
    patch = ca["patch"]

    print(f"ON-BRIDGE oracle tangent test | ckpt iter {ck.get('iter')} | anchor={ca.get('anchor')} "
          f"| tangent={ca.get('tangent')} | {args.patients} patients")
    print(f"{'pt':>3} {'t':>5} {'cos(v_pred,v_true)':>19} {'|v_pred|/|v_true|':>18} "
          f"{'cos vs WRONG motion':>20}")
    mg = torch.Generator().manual_seed(args.seed)      # CPU: random_motion draws its seed on CPU
    rows = []
    for p in range(args.patients):
        vol = gen.volume(p)
        _t, _r = amp_from_run_args(ca)              # p2p, doubling pre-2026-07-28 runs
        _t, _r = (_t or 10.0), (_r or 10.0)
        th = random_motion(cfg.n_views, trans_mm=_t, rot_deg=_r, device=dev, generator=mg)[None]
        th2 = random_motion(cfg.n_views, trans_mm=_t, rot_deg=_r, device=dev, generator=mg)[None]
        with torch.no_grad():
            y = gen.simulate(p, params_to_Pmot(th[0], gen.P_nom)[None])
            # the trainer's anchor: Delta = static-FDK - FDK(y, P(theta)), both in NET space
            x_anchor = gen.static_anchor_net(p)
            x1_geo = gen.to_net(gen.fdk(y, params_to_Pmot(th[0], gen.P_nom)[None])[0])
            dlt = x_anchor - x1_geo
        for tv in (0.1, 0.3, 0.5, 0.7, 0.9):
            t = torch.tensor(float(tv), device=dev)
            with torch.no_grad():
                x_t, dx_true = bridge_pair(gen, t, y, th[0], dlt, mode=ca.get("tangent", "analytic"))
                _, dx_wrong = bridge_pair(gen, t, y, th2[0], dlt,
                                          mode=ca.get("tangent", "analytic"))
                xin = x_t[None, None]
                gt_ = torch.Generator(device=dev).manual_seed(args.seed)
                x1 = predict_x1_patched(model, xin, float(tv), patch=patch, stride=patch // 2,
                                        context="auto", blend=args.blend,
                                        n_offsets=args.patch_offsets, generator=gt_)
                v_pred = ((x1 - xin) / max(1.0 - float(tv), 1e-3))[0, 0]
            c = cos(v_pred, dx_true)
            r = float(v_pred.norm() / dx_true.norm())
            cw = cos(v_pred, dx_wrong)
            rows.append((c, r, cw))
            print(f"{p:>3} {tv:>5.2f} {c:>19.4f} {r:>18.4f} {cw:>20.4f}", flush=True)
    import statistics as st
    print(f"\nMEAN  cos(on-bridge) = {st.mean(r[0] for r in rows):.4f} | "
          f"|v| ratio = {st.mean(r[1] for r in rows):.4f} | "
          f"cos vs wrong motion = {st.mean(r[2] for r in rows):.4f}")
    print("cos ~1 on-bridge  => the net LEARNED the field; the inference deficit is BLINDNESS,\n"
          "                     so feed theta back into the image update (capacity will not help).\n"
          "cos low on-bridge => a real TRAINING/CAPACITY deficit.")


if __name__ == "__main__":
    main()
