"""Summarize final-iterate, estimated-pose FDK and motion metrics on CPU.

Image scores use the stored full-precision evaluation, not the fp16 display volumes.
RPE and component MAE use the ground-truth-independent zero-mean pose frame.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fm3d.geometry_3d import ConeBeam3DConfig, build_conebeam_orbit
from fm3d.rigid_motion import reprojection_error, zero_centre_gauge


def score(path):
    r = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    theta, truth = r["theta"].double(), r["theta_true"].double()
    if theta.shape != (360, 6) or truth.shape != theta.shape:
        raise ValueError(f"Expected 360 six-component poses: {path}")
    if not torch.isfinite(theta).all() or not torch.isfinite(truth).all():
        raise ValueError(f"Nonfinite poses: {path}")
    cfg = ConeBeam3DConfig.thies(n_views=360)
    orbit = build_conebeam_orbit(cfg, device="cpu", dtype=torch.float64)
    centered = zero_centre_gauge(theta)
    mae = (centered - truth).abs().mean(0)
    mae[3:] = torch.rad2deg(mae[3:])
    values = dict(psnr=r["final_xt"]["psnr_aligned"], ssim=r["final_xt"]["ssim_aligned"],
                  fdk_psnr=r["final"]["psnr_aligned"], fdk_ssim=r["final"]["ssim_aligned"],
                  rpe_mm=reprojection_error(centered, truth, orbit)["rpe_mm"])
    values.update(zip(["tx_mae_mm", "ty_mae_mm", "tz_mae_mm",
                       "rx_mae_deg", "ry_mae_deg", "rz_mae_deg"], mae.tolist()))
    values = {k: float(v) for k, v in values.items()}
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError(f"Nonfinite scores: {path}")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows = []
    common_settings = None
    for i in range(30):
        case = args.cohort / f"p{i:02d}"
        result = case / "result.pt"
        if not result.is_file():
            raise SystemExit(f"Incomplete 30-patient cohort: missing {result}")
        meta = torch.load(case / "snaps/meta.pt", map_location="cpu", weights_only=False)["args"]
        if (meta["split"], meta["run"], meta["seed"]) != ("test", i, 1000 + i):
            raise ValueError(f"Wrong patient/seed pairing: {case}")
        settings = {key: meta.get(key) for key in
                    ("estimator", "lr", "ckpt", "n_steps", "per", "loss", "views_per_iter",
                     "est_coarse", "est_coarse_until", "cg_iters", "kappa", "tv_iters",
                     "tv_step", "blend", "patch_offsets", "trans_mm", "rot_deg")}
        if common_settings is None:
            common_settings = settings
        elif settings != common_settings:
            raise ValueError(f"Mixed inference settings/checkpoints in cohort: {case}")
        rows.append(dict(patient=i, **score(result)))
    summary = {key: dict(mean=float(np.mean([row[key] for row in rows])),
                         sd=float(np.std([row[key] for row in rows], ddof=0)))
               for key in rows[0] if key != "patient"}
    report = dict(n=30, sd_ddof=0, reference="ground-truth CT", settings=common_settings,
                  rows=rows, summary=summary)
    output = args.out or args.cohort / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
