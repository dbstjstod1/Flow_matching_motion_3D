"""Run the paper's training, single-case inference or paired test cohort.

Settings come from configs/paper.json. --dry-run needs only Python's standard library.
Existing outputs are reused only when a successful run has the same launch manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/paper.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def flags(settings):
    result = []
    for key, value in settings.items():
        if value is None or value is False:
            continue
        result.append("--" + key)
        if value is not True:
            result.extend(map(str, value if isinstance(value, list) else [value]))
    return result


def commands(args, config):
    """Build portable commands without reading an archived job or importing CUDA."""
    if args.mode == "train":
        settings = dict(config["training"], bridge=args.bridge,
                        root=str(args.root), out=str(args.out))
        return [(args.out, [sys.executable, "-u", str(ROOT / "scripts/train_fm3d.py"),
                            *flags(settings)])]
    settings = dict(config["inference"], estimator=args.estimator,
                    lr=config["validation"]["selected_lr"][args.estimator])
    patients = config["cohort"]["patients"] if args.mode == "cohort" else [args.patient]
    jobs = []
    for patient in patients:
        out = args.out / f"p{patient:02d}" if args.mode == "cohort" else args.out
        case = dict(settings, ckpt=str(args.ckpt), root=str(args.root),
                    split="test", run=patient, seed=1000 + patient, out=str(out))
        jobs.append((out, [sys.executable, "-u", str(ROOT / "scripts/run_posterior3d.py"),
                           *flags(case)]))
    return jobs


def source_hashes():
    files = [CONFIG, *sorted((ROOT / "fm3d").glob("*.py")),
             *sorted((ROOT / "fm3d/assets").glob("*")),
             *sorted((ROOT / "bench").rglob("*.py")),
             *sorted((ROOT / "scripts").glob("*.py"))]
    return {str(p.relative_to(ROOT)): digest(p) for p in files if p.is_file()}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate_checkpoint(path, bridge, config):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    saved = checkpoint["args"]
    expected = {key: config["training"][key] for key in
                ("dataset", "shape", "views", "base", "context", "patch", "sim_grid")}
    expected.update(bridge=bridge)
    for key, value in expected.items():
        actual = saved.get(key)
        if isinstance(value, list):
            actual = list(actual) if actual is not None else None
        if actual != value:
            raise ValueError(f"Checkpoint {key}: expected {value!r}, found {actual!r}")
    if saved.get("target", "v") != "v":
        raise ValueError("The paper uses velocity prediction, not an endpoint-prediction model.")
    if int(checkpoint.get("iter", -1)) != config["training"]["iters"]:
        raise ValueError("The paper evaluates the 500,000-iteration checkpoint.")
    if checkpoint["ema"]["in_conv.weight"].shape[1] != 5:
        raise ValueError("Expected the five-channel global-context EMA network.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "infer", "cohort"])
    parser.add_argument("--root", type=Path, default=Path("data/CQ500"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--bridge", choices=["data", "linear"], default="data")
    parser.add_argument("--estimator", choices=["akima_gd", "bspline_rmsprop"], default="akima_gd")
    parser.add_argument("--patient", type=int, choices=range(30), default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.root, args.out = args.root.resolve(), args.out.resolve()
    if args.mode != "train" and args.ckpt is None:
        parser.error("--ckpt is required for infer/cohort; train a prior first.")
    if args.ckpt is not None:
        args.ckpt = args.ckpt.resolve()
    config = json.loads(CONFIG.read_text())
    jobs = commands(args, config)
    if args.dry_run:
        for _, command in jobs:
            print(shlex.join(command))
        return
    if not args.root.is_dir():
        parser.error(f"CQ500 directory does not exist: {args.root}")
    sys.path.insert(0, str(ROOT))
    from fm3d.paper_protocol import check_dataset
    check_dataset(args.root)
    checkpoint_sha = None
    if args.mode != "train":
        validate_checkpoint(args.ckpt, args.bridge, config)
        checkpoint_sha = digest(args.ckpt)
    sources = source_hashes()
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "8")
    environment.setdefault("MKL_NUM_THREADS", "8")
    for out, command in jobs:
        identity = dict(protocol=config["protocol"], command=command,
                        checkpoint_sha256=checkpoint_sha, source_sha256=sources,
                        execution_environment={key: environment.get(key) for key in
                            ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS")})
        record = out / "launch.json"
        if out.exists() and any(out.iterdir()):
            previous = json.loads(record.read_text()) if record.exists() else {}
            artifact = out / ("ckpt_iter500000.pth" if args.mode == "train" else "result.pt")
            if previous.get("identity") == identity and previous.get("returncode") == 0 and artifact.exists():
                print(f"Completed, same protocol: {out}", flush=True)
                continue
            raise SystemExit(f"Existing/incomplete or different run: {out}. Choose a fresh output directory.")
        out.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents two launchers from claiming an empty case directory.
        with record.open("x") as stream:
            json.dump(dict(identity=identity, started=time.time(), returncode=None), stream, indent=2)
        started = time.time()
        print(shlex.join(command), flush=True)
        with (out / "run.log").open("w") as log:
            process = subprocess.run(command, cwd=ROOT, env=environment,
                                     stdout=log, stderr=subprocess.STDOUT)
        artifact = out / ("ckpt_iter500000.pth" if args.mode == "train" else "result.pt")
        returncode = process.returncode or (0 if artifact.is_file() else 1)
        write_json(record, dict(identity=identity, started=started, finished=time.time(),
                                elapsed_seconds=time.time() - started, returncode=returncode))
        if returncode:
            raise SystemExit(f"Run failed or final artifact missing; see {out / 'run.log'}")
        print(f"Completed: {out}", flush=True)


if __name__ == "__main__":
    main()
