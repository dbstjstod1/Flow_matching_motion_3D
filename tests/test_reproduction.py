"""CPU checks for paper command selection and prevention of mixed/overwritten runs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import reproduce
import run_posterior3d
import train_fm3d


class Parsed(Exception):
    pass


def parse_runner(module, argv):
    original = argparse.ArgumentParser.parse_args
    parsed = []

    def capture(parser):
        parsed.append(original(parser, argv))
        raise Parsed()

    with patch.object(argparse.ArgumentParser, "parse_args", capture):
        try:
            module.main()
        except Parsed:
            return parsed[0]
    raise AssertionError("Runner did not parse its command")


class PaperCommands(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(reproduce.CONFIG.read_text())

    def args(self, mode="cohort", bridge="data"):
        return argparse.Namespace(mode=mode, bridge=bridge,
                                  root=Path("/data/CQ500"), out=Path("/out/cohort"),
                                  ckpt=Path("/weights/prior.pth"), patient=0)

    def test_all_cohort_commands_parse_with_fixed_pairing(self):
        jobs = reproduce.commands(self.args(), self.config)
        self.assertEqual(len(jobs), 30)
        for i, (out, command) in enumerate(jobs):
            args = parse_runner(run_posterior3d, command[3:])
            self.assertEqual((args.split, args.run, args.seed), ("test", i, 1000 + i))
            self.assertEqual((args.estimator, args.lr, args.per), ("net", .003, 200))
            self.assertEqual(out.name, f"p{i:02d}")

    def test_bridge_ablation_changes_only_prior_checkpoint(self):
        base_args = self.args("infer")
        linear_args = self.args("infer", bridge="linear")
        linear_args.ckpt = Path("/weights/linear.pth")
        base = parse_runner(run_posterior3d, reproduce.commands(base_args, self.config)[0][1][3:])
        other = parse_runner(run_posterior3d, reproduce.commands(linear_args, self.config)[0][1][3:])
        self.assertEqual({k for k in vars(base) if getattr(base, k) != getattr(other, k)}, {"ckpt"})
        self.assertEqual((base.estimator, base.loss, base.lr), ("net", "l2", .003))

    def test_both_training_configs_parse(self):
        cases = []
        for bridge in ("data", "linear"):
            command = reproduce.commands(self.args("train", bridge), self.config)[0][1]
            cases.append(parse_runner(train_fm3d, command[3:]))
        self.assertEqual({k for k in vars(cases[0]) if getattr(cases[0], k) != getattr(cases[1], k)},
                         {"bridge"})
        self.assertEqual((cases[0].iters, cases[0].seed), (500000, 0))

    def test_default_inference_uses_paper_estimator(self):
        args = parse_runner(run_posterior3d, ["--ckpt", "model.pth"])
        self.assertEqual((args.estimator, args.loss), ("net", "l2"))

    def test_checkpoint_rejects_wrong_bridge_or_iteration(self):
        saved = {key: self.config["training"][key] for key in
                 ("dataset", "shape", "views", "base", "context", "patch", "sim_grid")}
        saved.update(bridge="data")
        checkpoint = dict(args=saved, iter=500000,
                          ema={"in_conv.weight": torch.empty(1, 5, 1, 1, 1)})
        with patch("torch.load", return_value=checkpoint):
            reproduce.validate_checkpoint("model.pth", "data", self.config)
            with self.assertRaisesRegex(ValueError, "bridge"):
                reproduce.validate_checkpoint("model.pth", "linear", self.config)
            checkpoint["iter"] = 499999
            with self.assertRaisesRegex(ValueError, "500,000"):
                reproduce.validate_checkpoint("model.pth", "data", self.config)

    def test_hash_pose_fit_preserves_adam_state_and_zero_initialization(self):
        from fm3d import motion_net
        from fm3d.motion_estimation import make_estimator
        # Use the CPU encoder only for this optimizer/shape check. Paper inference
        # requires tiny-cuda-nn and rejects this fallback before loading data.
        with patch.object(motion_net, "HAVE_TCNN", False):
            estimator = make_estimator("net", SimpleNamespace(n_views=4, du=1.),
                                       torch.zeros(4, 3, 4), torch.zeros(1), torch.zeros(1),
                                       "cpu", loss="l2", views_per_iter=4)
        self.assertEqual(estimator.net.enc.n_levels, 16)
        self.assertEqual(estimator.net.enc.n_output_dims, 32)
        self.assertEqual((estimator.net.trans_max_mm, estimator.net.rot_max_deg), (15., 8.))
        self.assertEqual((estimator.lr0, estimator.smooth_w), (.003, 0.))
        torch.testing.assert_close(estimator.current_params(), torch.zeros(4, 6))
        optimizer = estimator._opt()
        estimator._project = lambda image, theta, views=None: theta[:, :1, None]
        estimator.refine_global(torch.zeros(1, 1, 1), torch.ones(4, 1, 1), iters=2)
        head = estimator.net.mlp[-1].bias
        self.assertEqual(float(optimizer.state[head]["step"]), 2.)
        estimator.dx = estimator.dy = estimator.dz = 2.
        estimator.refine_global(torch.zeros(1, 1, 1), torch.ones(4, 1, 1), iters=1)
        self.assertIs(estimator._opt(), optimizer)
        self.assertEqual(float(optimizer.state[head]["step"]), 3.)
        self.assertGreater(float(estimator.current_params()[:, 0].mean()), 0.)

    def test_completed_launch_skips_but_changed_launch_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "case"
            argv = ["reproduce.py", "infer", "--ckpt", "model.pth", "--out", str(out),
                    "--root", directory]

            def completed(*args, **kwargs):
                (out / "result.pt").write_bytes(b"test artifact")
                return subprocess.CompletedProcess(args[0], 0)

            with patch.object(sys, "argv", argv), \
                 patch("fm3d.paper_protocol.check_dataset"), \
                 patch.object(reproduce, "validate_checkpoint"), \
                 patch.object(reproduce, "digest", return_value="checkpoint"), \
                 patch.object(reproduce, "source_hashes", return_value={"source": "v1"}), \
                 patch.object(reproduce.subprocess, "run", side_effect=completed) as run:
                reproduce.main()
                reproduce.main()
                self.assertEqual(run.call_count, 1)
                with patch.object(reproduce, "source_hashes", return_value={"source": "v2"}):
                    with self.assertRaises(SystemExit):
                        reproduce.main()
                self.assertEqual(run.call_count, 1)

    def test_nonempty_output_without_manifest_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "case"
            out.mkdir()
            (out / "valuable.txt").write_text("keep")
            with patch.object(sys, "argv", ["reproduce.py", "train", "--root", directory,
                                            "--out", str(out)]), \
                 patch("fm3d.paper_protocol.check_dataset"), \
                 patch.object(reproduce, "source_hashes", return_value={}), \
                 patch.object(reproduce.subprocess, "run") as run:
                with self.assertRaises(SystemExit):
                    reproduce.main()
                run.assert_not_called()
            self.assertEqual((out / "valuable.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
