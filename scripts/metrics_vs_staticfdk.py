"""Re-score finished runs against the STATIC-FDK reference (Thies' protocol), offline.

WHY THIS EXISTS. Every number this project has printed so far is measured against the GT VOLUME
(the CQ500 CT itself). Thies (arXiv:2401.09283) -- whose numbers are our stated target -- scores
against the reconstruction of the MOTION-FREE scan instead: FDK(A_static(gt)), the same operator
fed an uncorrupted orbit. That reference quotients out what the operator itself cannot do (finite
views, cone-beam sampling, the FDK approximation), so it grades ONLY the motion damage and its
repair. Consequences of the difference, measured 2026-07-24:

  * vs GT, the ideal FDK itself only scores ~32.3-33.7 dB / SSIM ~0.79-0.80 -- so vs-GT numbers
    UNDERSTATE motion-compensation quality, and are NOT comparable to Thies' 0.83 -> 0.94 SSIM.
  * The carried x_t EXCEEDS the FDK ceiling vs GT (prior-restored detail the operator cannot
    render). Under the static-FDK reference that same restored detail counts as DEVIATION from
    the reference's own artifacts -- so expect x_t's edge to SHRINK under this protocol, and the
    output FDK(theta_hat)'s to GROW. Both readings are informative; report BOTH, never silently
    switch.

This script recomputes from the volumes result.pt already stores (x_final, x_t) -- nothing is
re-run. Runs older than the volume-saving change get their x_final rebuilt from the saved theta;
their x_t is gone and is reported as missing.

    python scripts/metrics_vs_staticfdk.py --runs cg_v0 cg_v1 cg_v2 cg_v0_tv --patients 0 1 2 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig, measured_region_mask
from fm3d.reg_metric import aligned_metrics
from fm3d.rigid_motion import params_to_Pmot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/fm3d_cq500/ckpt_iter500000.pth")
    ap.add_argument("--runs", nargs="+", default=["cg_v0", "cg_v1", "cg_v2"])
    ap.add_argument("--patients", nargs="+", type=int, default=[0, 1, 2],
                    help="val-patient index per run, same order")
    ap.add_argument("--dir", default="data")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    if len(args.patients) != len(args.runs):
        raise SystemExit("--patients must pair 1:1 with --runs")

    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    ca = ck["args"]
    cfg = ConeBeam3DConfig.thies(n_views=ca["views"])
    gen = CQ500Generator(ca["root"], cfg=cfg, device=dev, split=args.split,
                         shape=tuple(ca["shape"]), voxel_mm=1.0)
    spacing = (gen.dz, gen.dy, gen.dx)
    meas = measured_region_mask(gen.shape, spacing, cfg, device=dev)

    print(f"{'run':10} {'vol':>10} | {'vs GT (current)':>18} | {'vs static FDK (Thies)':>21}")
    print("-" * 70)
    statics = {}
    for tag, pid in zip(args.runs, args.patients):
        p = os.path.join(args.dir, tag, "result.pt")
        if not os.path.isfile(p):
            print(f"{tag:10} (not finished)")
            continue
        r = torch.load(p, map_location="cpu", weights_only=False)
        gt = gen.volume(pid)
        gt3 = gt[0, 0]
        if pid not in statics:                     # one static-FDK reference per patient
            with torch.no_grad():
                y0 = gen.simulate(pid, gen.P_nom[None])
                statics[pid] = gen.fdk(y0, gen.P_nom[None])[0]
        ref = statics[pid]

        vols = {}
        if "x_final" in r:
            vols["OUTPUT"] = r["x_final"].to(dev).float()
        else:                                      # older run: rebuild from theta
            th = r["theta"].to(dev)
            tt = r["theta_true"].to(dev)
            with torch.no_grad():
                y = gen.simulate(pid, params_to_Pmot(tt, gen.P_nom)[None])
                vols["OUTPUT"] = gen.fdk(y, params_to_Pmot(th, gen.P_nom)[None])[0]
        if "x_t" in r:
            vols["x_t"] = r["x_t"].to(dev).float()

        out = {}
        for name, v in vols.items():
            m_gt = aligned_metrics(v, gt3, spacing, mask=meas, iters=300)
            m_st = aligned_metrics(v, ref, spacing, mask=meas, iters=300)
            out[name] = {"vs_gt": m_gt, "vs_static": m_st}
            print(f"{tag:10} {name:>10} | {m_gt['psnr_aligned']:8.2f} "
                  f"{m_gt['ssim_aligned']:.3f}    | {m_st['psnr_aligned']:8.2f} "
                  f"{m_st['ssim_aligned']:.3f}", flush=True)
        if "x_t" not in vols:
            print(f"{tag:10} {'x_t':>10} | (volume not saved -- run predates the change)")
        # the reference's own vs-GT score = the operator ceiling, for context
        m_ref = aligned_metrics(ref, gt3, spacing, mask=meas, iters=300)
        print(f"{tag:10} {'(static)':>10} | {m_ref['psnr_aligned']:8.2f} "
              f"{m_ref['ssim_aligned']:.3f}    |    (reference)")
        with open(os.path.join(args.dir, tag, "metrics_vs_static.json"), "w") as f:
            json.dump({k: {kk: {m: float(x) for m, x in vv.items()} for kk, vv in v.items()}
                       for k, v in out.items()}, f, indent=1)


if __name__ == "__main__":
    main()
