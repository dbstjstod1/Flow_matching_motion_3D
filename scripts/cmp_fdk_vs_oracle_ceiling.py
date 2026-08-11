"""Score both methods' FDK-class outputs against the TRUE FDK ceiling: FDK(theta_true), per operator.

USER'S CALL (2026-08-11): the static FDK is the wrong ceiling for judging a motion-compensated
FDK -- even a PERFECT theta cannot reproduce a static scan (per-view motion breaks the circular
equiangular assumptions the analytic inverse is derived under; the repo has measured the three
distinct ceilings static FDK > FDK(theta_true) > loop output). The honest ceiling is the SAME
reconstruction operator handed the TRUE motion: FDK_ours(y, P(theta_true)) for us,
BP_thies(g_filt, P(theta_true)) for the baseline. This script re-references the 10/10 paired
cohort to exactly that, per method, and asks: WHICH METHOD'S OUTPUT IS CLOSER TO ITS OWN
OPERATOR'S TRUE CEILING -- and how high is each ceiling vs GT.

Everything is rebuilt deterministically from (split=test, run=i, seed=1000+i): `build_world`
resimulates the byte-identical sinogram, both oracles are one reconstruction each, and the
outputs come off the cohorts' result.pt (ours: x_final fp16; theirs: out_vol fp32). The pairing
is asserted on theta_true before anything is scored.

    CUDA_VISIBLE_DEVICES=0 python scripts/cmp_fdk_vs_oracle_ceiling.py \
        [--ours data/fm3d_test30_databridge] [--theirs data/bench_thies_test30] [--n 30]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.thies.recon import ThiesConeRecon, VolumeGrid                    # noqa: E402
from fm3d.reg_metric import aligned_metrics                                 # noqa: E402
from fm3d.rigid_motion import params_to_Pmot                                # noqa: E402
from scripts.run_posterior3d import build_world                             # noqa: E402

DEFAULT_CKPT = "logs/fm3d_databridge/ckpt_iter500000.pth"


def wilcoxon(d):
    try:
        from scipy.stats import wilcoxon as w
        s, p = w(d)
        return float(p)
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                    help="rebuilds the world only; the prior itself is never evaluated")
    ap.add_argument("--ours", default="data/fm3d_test30_databridge")
    ap.add_argument("--theirs", default="data/bench_thies_test30")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default=None, help="json out; default <theirs>/vs_oracle_fdk.json")
    a = ap.parse_args()
    dev = "cuda"

    rows = []
    recon_t = None
    grid256 = VolumeGrid.centred(256, 1.0)
    for i in range(a.n):
        tag = f"p{i:02d}"
        o = torch.load(os.path.join(a.ours, tag, "result.pt"), map_location="cpu",
                       weights_only=False)
        t = torch.load(os.path.join(a.theirs, tag, "result.pt"), map_location="cpu",
                       weights_only=False)
        W = build_world(ckpt=a.ckpt, dev=dev, split="test", run=i, seed=1000 + i)
        gen, cfg, gt, th_true, y = W["gen"], W["cfg"], W["gt3"], W["theta_true"], W["y"]
        # THE PAIRING ASSERT: all three worlds (ours, theirs, this rebuild) must carry the same
        # motion, or the oracle below belongs to a different scan than the outputs.
        for name, saved in (("ours", o["theta_true"]), ("theirs", t["theta_true"])):
            dv = float((saved.to(dev) - th_true).abs().max())
            if dv > 1e-5:
                raise SystemExit(f"{tag}: {name} theta_true mismatch {dv:.3e} -- unpaired")
        if recon_t is None:
            recon_t = ThiesConeRecon(cfg)

        P_true = params_to_Pmot(th_true, gen.P_nom)
        with torch.no_grad():
            oracle_ours = gen.fdk(y, P_true[None])[0]                      # our FDK, true theta
            oracle_thies = recon_t(y, P_true, grid256)                     # their BP, true theta
        sp = (1.0, 1.0, 1.0)
        out_ours = o["x_final"].to(dev).float()
        out_thies = t["out_vol"].to(dev).float()
        r = dict(tag=tag)
        # each output vs ITS OWN operator's true-theta ceiling
        r["ours_vs_ceiling"] = aligned_metrics(out_ours, oracle_ours, sp)
        r["thies_vs_ceiling"] = aligned_metrics(out_thies, oracle_thies, sp)
        # the ceiling heights themselves, vs GT (context: the operators' ceilings differ)
        r["oracle_ours_vs_gt"] = aligned_metrics(oracle_ours, gt, sp)
        r["oracle_thies_vs_gt"] = aligned_metrics(oracle_thies, gt, sp)
        rows.append(r)
        print(f"{tag}  ours->ceiling {r['ours_vs_ceiling']['ssim_aligned']:.4f} "
              f"({r['ours_vs_ceiling']['psnr_aligned']:.2f} dB) | "
              f"thies->ceiling {r['thies_vs_ceiling']['ssim_aligned']:.4f} "
              f"({r['thies_vs_ceiling']['psnr_aligned']:.2f} dB) | ceilings vs GT: "
              f"ours {r['oracle_ours_vs_gt']['ssim_aligned']:.4f} / "
              f"thies {r['oracle_thies_vs_gt']['ssim_aligned']:.4f}", flush=True)
        del W, gen, y, oracle_ours, oracle_thies, out_ours, out_thies
        torch.cuda.empty_cache()

    def col(key, m="ssim_aligned"):
        return np.array([r[key][m] for r in rows])

    print("\n" + "=" * 78)
    print(f"PAIRED over {len(rows)} patients -- output vs ITS OWN FDK(theta_true) ceiling")
    print("=" * 78)
    for key, name in (("ours_vs_ceiling", "ours   FDK(theta_hat) -> ceiling"),
                      ("thies_vs_ceiling", "Thies  output        -> ceiling")):
        s, p_ = col(key), col(key, "psnr_aligned")
        print(f"  {name:34s} SSIM {s.mean():.4f} +- {s.std(ddof=1):.4f}   "
              f"PSNR {p_.mean():6.2f} +- {p_.std(ddof=1):.2f} dB")
    d = col("ours_vs_ceiling") - col("thies_vs_ceiling")
    print(f"  paired diff (ours - thies)         {d.mean():+.4f} SSIM  "
          f"(ours closer on {int((d > 0).sum())}/{len(rows)})  Wilcoxon p = {wilcoxon(d):.2e}")
    print("\n  ceiling heights (context -- the two operators do NOT share a ceiling):")
    for key, name in (("oracle_ours_vs_gt", "FDK_ours(theta_true)  vs GT"),
                      ("oracle_thies_vs_gt", "BP_thies(theta_true)  vs GT")):
        s, p_ = col(key), col(key, "psnr_aligned")
        print(f"  {name:34s} SSIM {s.mean():.4f} +- {s.std(ddof=1):.4f}   "
              f"PSNR {p_.mean():6.2f} +- {p_.std(ddof=1):.2f} dB")

    out = a.out or os.path.join(a.theirs, "vs_oracle_fdk.json")
    json.dump([{k: (v if isinstance(v, str) else {m: float(x) for m, x in v.items()})
                for k, v in r.items()} for r in rows], open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
