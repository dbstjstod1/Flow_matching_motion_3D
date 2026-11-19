"""Summarize one complete CQ500 spline cohort on CPU (population SD).

Scores use saved full-precision evaluations, not fp16 display volumes. Pose errors
use the ground-truth-independent zero-mean pose frame. Method identity is checked
from native metadata or the archived spline wrapper's explicit scheme metadata;
the prior bridge comes from saved checkpoint metadata, never its filename.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fm3d.geometry_3d import ConeBeam3DConfig, build_conebeam_orbit
from fm3d.rigid_motion import reprojection_error, zero_centre_gauge

PAPER_SETTINGS = dict(
    n_steps=50, per=200, loss="l2", views_per_iter=24, est_coarse=2,
    est_coarse_until=0.5, cg_iters=5, kappa=0.3, tv_iters=5,
    tv_step=0.015, blend="uniform", patch_offsets=2, context="global",
    trans_mm=10.0, rot_deg=10.0, motion_kind="akima", amp_units="p2p",
)
# These were configurable in archived runs and are fixed in the public runner.
LEGACY_FIXED = dict(cg_lam=0.0, pnp_k=1, dc_views=0, dc_op="cg", theta_avg=1,
                    passes=1, per_sched="const", prior="fm", no_prior=False,
                    theta_oracle=False)
SCHEMES = {
    "akima_gd": dict(control_points=30, optimizer="SGD", lr=1000.0),
    "bspline_rmsprop": dict(control_points=20, optimizer="RMSprop", lr=0.001),
}


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def scheme_identity(case, metadata):
    """Resolve wrapper overrides before interpreting the generic CLI estimator."""
    args = metadata["args"]
    configs = []
    for owner in (metadata, args):
        if "scheme_config" in owner:
            config = owner["scheme_config"]
            if not isinstance(config, dict):
                raise ValueError(f"Invalid scheme_config: {case}")
            configs.append(config)
    for name in ("scheme_config.json", "scheme.json"):
        path = case / name
        if path.is_file():
            configs.append(json.loads(path.read_text()))
    names = []
    for config in configs:
        named = [config[k] for k in ("scheme", "estimator", "name") if k in config]
        if not named or any(name not in SCHEMES for name in named):
            raise ValueError(f"Invalid spline scheme metadata: {case}: {config}")
        names.extend(named)
    native = args.get("estimator")
    if native in SCHEMES:
        names.append(native)
    elif native != "net" or not names:
        raise ValueError(f"Missing explicit spline estimator identity: {case}")
    if len(set(names)) != 1:
        raise ValueError(f"Conflicting spline estimator metadata: {case}: {names}")
    name = names[0]
    identity = dict(estimator=name, **SCHEMES[name])
    lr = args.get("lr")
    if lr is not None and lr != identity["lr"]:
        raise ValueError(f"Wrong {name} learning rate: {case}: {lr}")
    for config in configs:
        for key in ("n_ctrl", "control_points", "nodes"):
            if key in config and config[key] != identity["control_points"]:
                raise ValueError(f"Wrong spline control count: {case}: {config}")
        if "lr" in config and config["lr"] != identity["lr"]:
            raise ValueError(f"Wrong spline learning rate: {case}: {config}")
        if "optimizer" in config:
            optimizer = str(config["optimizer"]).lower()
            optimizer = {"gd": "sgd"}.get(optimizer, optimizer)
            expected = identity["optimizer"].lower()
            if optimizer != expected:
                raise ValueError(f"Wrong spline optimizer: {case}: {config}")
    # The archived wrapper hardcoded 30/20 controls. Its generic args.n_ctrl=30
    # was unused for B-splines, so only scheme-specific control counts are read.
    return identity


@lru_cache(maxsize=8)
def checkpoint_prior(checkpoint):
    path = Path(checkpoint).expanduser()
    if not path.is_file() and not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise ValueError(f"Cannot verify prior bridge: {checkpoint}. Use --checkpoint with the original prior.")
    saved = load(path)
    return dict(bridge=saved["args"]["bridge"], iteration=saved["iter"])


def settings_for_case(case, index, checkpoint=None):
    metadata = load(case / "snaps/meta.pt")
    args = metadata["args"]
    if (args.get("split"), args.get("run"), args.get("seed")) != ("test", index, 1000 + index):
        raise ValueError(f"Wrong patient/seed pairing: {case}")
    mismatches = {key: (args.get(key), value) for key, value in PAPER_SETTINGS.items()
                  if args.get(key) != value}
    mismatches.update({key: (args[key], value) for key, value in LEGACY_FIXED.items()
                       if key in args and args[key] != value})
    if mismatches:
        raise ValueError(f"Not the paper inference protocol: {case}: {mismatches}")
    identity = scheme_identity(case, metadata)
    prior = metadata.get("prior")
    if prior is None or checkpoint is not None:
        verified = checkpoint_prior(str(checkpoint or args["ckpt"]))
        if prior is not None and any(prior.get(k) != verified[k] for k in verified):
            raise ValueError(f"Checkpoint conflicts with saved prior metadata: {case}")
        prior = verified
    if prior.get("bridge") not in ("data", "linear") or prior.get("iteration") != 500000:
        raise ValueError(f"Not a paper bridge checkpoint at iteration 500000: {case}: {prior}")
    identity["bridge"] = "geometry" if prior["bridge"] == "data" else "image_linear"
    identity["checkpoint_iteration"] = prior["iteration"]
    settings = dict(PAPER_SETTINGS, **identity, checkpoint=args["ckpt"])
    settings.update({key: args.get(key) for key in ("prior_batch", "prior_amp", "compile")})
    return settings


def score(path):
    result = load(path)
    theta, truth = result["theta"].double(), result["theta_true"].double()
    if theta.shape != (360, 6) or truth.shape != theta.shape:
        raise ValueError(f"Expected 360 six-component poses: {path}")
    if not torch.isfinite(theta).all() or not torch.isfinite(truth).all():
        raise ValueError(f"Nonfinite poses: {path}")
    history = result["theta_hist"]
    if history.shape != (50, 360, 6) or not torch.equal(history[-1], result["theta"]):
        raise ValueError(f"Expected the last of 50 pose updates: {path}")
    orbit = build_conebeam_orbit(ConeBeam3DConfig.thies(n_views=360),
                                device="cpu", dtype=torch.float64)
    centered = zero_centre_gauge(theta)
    mae = (centered - truth).abs().mean(0)
    mae[3:] = torch.rad2deg(mae[3:])
    values = dict(psnr=result["final_xt"]["psnr_aligned"], ssim=result["final_xt"]["ssim_aligned"],
                  fdk_psnr=result["final"]["psnr_aligned"], fdk_ssim=result["final"]["ssim_aligned"],
                  rpe_mm=reprojection_error(centered, truth, orbit)["rpe_mm"])
    values.update(zip(["tx_mae_mm", "ty_mae_mm", "tz_mae_mm",
                       "rx_mae_deg", "ry_mae_deg", "rz_mae_deg"], mae.tolist()))
    values = {key: float(value) for key, value in values.items()}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError(f"Nonfinite scores: {path}")
    return values


def summarize(cohort, checkpoint=None):
    rows = []
    common = None
    for index in range(30):
        case = Path(cohort) / f"p{index:02d}"
        for name in ("result.pt", "snaps/meta.pt"):
            if not (case / name).is_file():
                raise ValueError(f"Incomplete 30-patient cohort: missing {case / name}")
        settings = settings_for_case(case, index, checkpoint)
        if common is None:
            common = settings
        elif settings != common:
            raise ValueError(f"Mixed estimators, bridges, checkpoints or settings: {case}")
        rows.append(dict(patient=index, seed=1000 + index, **score(case / "result.pt")))
    summary = {key: dict(mean=float(np.mean([row[key] for row in rows])),
                         sd=float(np.std([row[key] for row in rows], ddof=0)))
               for key in rows[0] if key not in ("patient", "seed")}
    identity = {key: common[key] for key in
                ("estimator", "control_points", "optimizer", "lr", "bridge", "checkpoint_iteration")}
    return dict(n=30, sd_ddof=0, reference="ground-truth CT", identity=identity,
                settings=common, rows=rows, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--checkpoint", type=Path,
                        help="original prior at a new location, for archived metadata")
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = summarize(args.cohort, args.checkpoint)
    output = args.out or args.cohort / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(identity=report["identity"], summary=report["summary"]), indent=2))


if __name__ == "__main__":
    main()
