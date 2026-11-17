"""Common measured-FOV and supplementary body-ROI scores for CQ500 outputs.

    python scripts/roi_body_metrics.py [--tags p16,p14] [--n 30] [--out data/roi_body_metrics.json]

Motivation: the standing convention scores the full measured region, most of which is air --
that both dilutes differences and lets background behaviour leak into PSNR. Here the ROI is a
BODY MASK built from the GT volume: mu > 0.01 (about -500 HU), binary closing, hole fill (so
enclosed sinuses/mastoids stay inside), largest connected component (drops the CT table),
intersected with the measured region.

The ALIGNMENT is unchanged (mask=measured region, iters=200 -- the loop's own convention), so
this changes only WHERE the score is read, not the gauge fit. Volumes scored: the four
x_t-class deliverables (geometry / image-linear / autofocus / JRM-ADM).
The manuscript uses psnr_meas and ssim_meas, not the supplementary body scores.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.reg_metric import aligned_metrics, psnr, ssim   # noqa: E402

CKPT = "logs/fm3d_databridge/ckpt_iter500000.pth"
JRM_MU_RATIO = 0.02 / 0.0193          # their mu_water -> ours, as in score_jrm_native.py


def load_arm(name: str, d: str, tag: str, dev: str) -> torch.Tensor:
    """The x_t-class deliverable of one arm, in OUR frame (256^3, mu 1/mm)."""
    if name == "jrm":
        r = torch.load(os.path.join(d, f"{tag}_result.pt"), map_location=dev, weights_only=False)
        v = r["x_est"][0, 0].float().to(dev) * JRM_MU_RATIO          # their 224^3 grid
        out = torch.zeros((256, 256, 256), dtype=v.dtype, device=dev)
        o = [(256 - n) // 2 for n in v.shape]
        out[o[0]:o[0] + v.shape[0], o[1]:o[1] + v.shape[1], o[2]:o[2] + v.shape[2]] = v
        return out
    r = torch.load(os.path.join(d, tag, "result.pt"), map_location=dev, weights_only=False)
    return (r["out_vol"] if "out_vol" in r else r["x_t"]).float().to(dev)


def body_mask(gt: torch.Tensor, meas: torch.Tensor) -> torch.Tensor:
    b = (gt > 0.01).cpu().numpy()                       # about -500 HU
    b = ndi.binary_closing(b, structure=np.ones((3, 3, 3)), iterations=2)
    b = ndi.binary_fill_holes(b)
    lab, nl = ndi.label(b)
    if nl > 1:
        b = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    return torch.from_numpy(np.ascontiguousarray(b)).to(gt.device) & meas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=None, help="comma list; default p00..p29")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="data/roi_body_metrics.json")
    ap.add_argument("--root", default=None)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--arm", action="append", required=True,
                    help="NAME=DIRECTORY; repeat to select output cohorts")
    a = ap.parse_args()
    arms = [tuple(item.split("=", 1)) for item in a.arm]
    if any(len(item) != 2 for item in arms):
        ap.error("--arm requires NAME=DIRECTORY")
    tags = a.tags.split(",") if a.tags else [f"p{i:02d}" for i in range(a.n)]
    dev = "cuda"
    from run_posterior3d import build_world

    rows = []
    for tag in tags:
        i = int(tag[1:])
        w = build_world(ckpt=a.ckpt, root=a.root, dev=dev, split="test", run=i, seed=1000 + i,
                        motion_kind="akima", trans_mm=10.0, rot_deg=10.0)
        gt, sp, meas = w["gt3"], w["spacing"], w["meas"]
        body = body_mask(gt, meas)
        peak = float(gt[body].max())
        row = {"tag": tag, "body_voxels_M": float(body.sum()) / 1e6}
        for name, d in arms:
            vol = load_arm(name, d, tag, dev)
            m, al = aligned_metrics(vol, gt, sp, mask=meas, iters=200, return_aligned=True)
            row[name] = {
                "psnr_meas": m["psnr_aligned"], "ssim_meas": m["ssim_aligned"],
                "psnr_body": psnr(al, gt, mask=body, peak=peak),
                "ssim_body": ssim(al, gt, data_range=peak, mask=body),
            }
        rows.append(row)
        print(f"{tag}  body {row['body_voxels_M']:.1f}M | " + " | ".join(
            f"{n} {row[n]['psnr_body']:.2f}/{row[n]['ssim_body']:.4f}" for n, _ in arms),
            flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    print("wrote", a.out)

    if len(rows) > 1:
        print("\n== paper cohort summary (measured FOV, aligned; population SD) ==")
        for name, _ in arms:
            p = np.array([r[name]["psnr_meas"] for r in rows])
            s = np.array([r[name]["ssim_meas"] for r in rows])
            print(f"  {name:7s} PSNR {p.mean():6.2f} +- {p.std(ddof=0):.2f}   "
                  f"SSIM {s.mean():.4f} +- {s.std(ddof=0):.4f}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
