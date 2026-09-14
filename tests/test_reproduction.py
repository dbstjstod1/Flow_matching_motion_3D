"""CPU checks for paper command selection and prevention of mixed/overwritten runs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
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

    def args(self, mode="cohort", bridge="data", estimator="akima_gd"):
        return argparse.Namespace(mode=mode, bridge=bridge, estimator=estimator,
                                  root=Path("/data/CQ500"), out=Path("/out/cohort"),
                                  ckpt=Path("/weights/prior.pth"), patient=0)

    def test_all_cohort_commands_parse_with_fixed_pairing(self):
        jobs = reproduce.commands(self.args(), self.config)
        self.assertEqual(len(jobs), 30)
        for i, (out, command) in enumerate(jobs):
            args = parse_runner(run_posterior3d, command[3:])
            self.assertEqual((args.split, args.run, args.seed), ("test", i, 1000 + i))
            self.assertEqual((args.estimator, args.lr, args.per), ("akima_gd", 1000, 200))
            self.assertEqual(out.name, f"p{i:02d}")

    def test_ablation_changes_only_estimator_and_step_size(self):
        base = parse_runner(run_posterior3d, reproduce.commands(self.args("infer"), self.config)[0][1][3:])
        other = parse_runner(run_posterior3d, reproduce.commands(
            self.args("infer", estimator="bspline_rmsprop"), self.config)[0][1][3:])
        self.assertEqual({k for k in vars(base) if getattr(base, k) != getattr(other, k)},
                         {"estimator", "lr"})
        self.assertEqual(other.lr, .001)

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
        self.assertEqual(args.estimator, "akima_gd")

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
