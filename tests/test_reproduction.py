"""CPU checks for paper command selection and prevention of mixed/overwritten runs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
import math

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

    def args(self, mode="cohort", bridge="data", estimator=None):
        return argparse.Namespace(mode=mode, bridge=bridge, estimator=estimator,
                                  root=Path("/data/CQ500"), out=Path("/out/cohort"),
                                  ckpt=Path("/weights/prior.pth"), patient=0)

    def test_all_cohort_commands_parse_with_fixed_pairing(self):
        jobs = reproduce.commands(self.args(), self.config)
        self.assertEqual(len(jobs), 30)
        for i, (out, command) in enumerate(jobs):
            args = parse_runner(run_posterior3d, command[3:])
            self.assertEqual((args.split, args.run, args.seed), ("test", i, 1000 + i))
            self.assertEqual((args.estimator, args.lr, args.per), ("akima_gd", 1000., 200))
            self.assertEqual(out.name, f"p{i:02d}")

    def test_bridge_ablation_changes_only_prior_checkpoint(self):
        base_args = self.args("infer")
        linear_args = self.args("infer", bridge="linear")
        linear_args.ckpt = Path("/weights/linear.pth")
        base = parse_runner(run_posterior3d, reproduce.commands(base_args, self.config)[0][1][3:])
        other = parse_runner(run_posterior3d, reproduce.commands(linear_args, self.config)[0][1][3:])
        self.assertEqual({k for k in vars(base) if getattr(base, k) != getattr(other, k)}, {"ckpt"})
        self.assertEqual((base.estimator, base.loss, base.lr), ("akima_gd", "l2", 1000.))

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
        self.assertEqual((args.estimator, args.loss), ("akima_gd", "l2"))

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

    def test_bspline_command_changes_only_estimator_and_step_size(self):
        base = parse_runner(run_posterior3d, reproduce.commands(self.args("infer"), self.config)[0][1][3:])
        alternate = parse_runner(run_posterior3d, reproduce.commands(
            self.args("infer", estimator="bspline_rmsprop"), self.config)[0][1][3:])
        self.assertEqual({k for k in vars(base) if getattr(base, k) != getattr(alternate, k)},
                         {"estimator", "lr"})
        self.assertEqual((alternate.estimator, alternate.lr, alternate.motion_kind),
                         ("bspline_rmsprop", .001, "akima"))

    def test_bspline_asset_is_fingerprinted_and_tampering_is_rejected(self):
        from fm3d import spline_motion
        hashes = reproduce.source_hashes()
        self.assertEqual(hashes["fm3d/assets/jrm_cubic_bspline_V360_C20.npy"], spline_motion.BASIS_SHA256)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "changed.npy"
            path.write_bytes(b"not the published interpolation matrix")
            with patch.object(spline_motion, "BASIS", path), self.assertRaisesRegex(ValueError, "published basis"):
                spline_motion.SplineSchemeEstimator(SimpleNamespace(n_views=360), None, None, None,
                                                    "cpu", scheme="bspline_rmsprop", lr=.001)

    def test_spline_coefficients_persist_but_optimizer_state_resets(self):
        from fm3d.motion_estimation import make_estimator
        for scheme, control_points in [("akima_gd", 30), ("bspline_rmsprop", 20)]:
            with self.subTest(scheme=scheme):
                estimator = make_estimator(scheme, SimpleNamespace(n_views=360, du=1.),
                                           None, None, None, "cpu", views_per_iter=24)
                self.assertEqual(tuple(estimator.coefficients.shape), (control_points, 6))
                self.assertEqual(estimator.smooth_w, 0.)
                torch.testing.assert_close(estimator.current_params(), torch.zeros(360, 6))
                observed = []
                def project(image, theta, views=None):
                    observed.append(theta.detach().clone())
                    values = theta / estimator.units
                    return .01 * (values if views is None else values[views]).reshape(-1, 2, 3)
                estimator._project = project
                target = torch.ones(360, 2, 3) * .004
                estimator.refine_global(None, target, iters=2)
                after_first = estimator.current_params().clone()
                self.assertGreater(float(after_first.abs().max()), 0.)
                first_optimizer = estimator._opt()
                if scheme == "bspline_rmsprop":
                    self.assertEqual(float(first_optimizer.state[estimator.coefficients]["step"]), 2.)
                else:
                    self.assertAlmostEqual(first_optimizer.param_groups[0]["lr"], 1000. * .97)
                estimator.dx = estimator.dy = estimator.dz = 2.
                estimator.refine_global(None, target, iters=1)
                torch.testing.assert_close(observed[2], after_first, rtol=0, atol=0)
                if scheme == "bspline_rmsprop":
                    self.assertIsNot(first_optimizer, estimator._opt())
                    self.assertEqual(float(estimator._opt().state[estimator.coefficients]["step"]), 1.)
                    defaults = estimator._opt().defaults
                    self.assertEqual((defaults["lr"], defaults["alpha"], defaults["eps"],
                                      defaults["momentum"], defaults["centered"]),
                                     (.001, .99, 1e-8, 0, False))
                else:
                    self.assertIs(first_optimizer, estimator._opt())
                    self.assertEqual(estimator._opt().param_groups[0]["lr"], 1000.)

    def test_spline_units_constant_trajectory_and_gauge(self):
        from fm3d.motion_estimation import make_estimator
        from fm3d.rigid_motion import zero_centre_gauge
        for scheme in ("akima_gd", "bspline_rmsprop"):
            with self.subTest(scheme=scheme):
                estimator = make_estimator(scheme, SimpleNamespace(n_views=360, du=1.),
                                           None, None, None, "cpu")
                with torch.no_grad():
                    estimator.coefficients.copy_(torch.tensor([3., -2., 1., 2., -3., 4.]))
                expected = torch.tensor([3., -2., 1., 2*math.pi/180, -3*math.pi/180, 4*math.pi/180])
                poses = estimator.current_params()
                torch.testing.assert_close(poses, expected.expand(360, 6), rtol=1e-6, atol=1e-6)
                centered = zero_centre_gauge(poses.double())
                torch.testing.assert_close(centered, torch.zeros_like(centered), rtol=0, atol=1e-6)

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
