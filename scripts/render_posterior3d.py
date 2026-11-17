"""Render deferred CQ500 reconstruction snapshots and per-step image metrics.

Rebuilds the scan from snaps/meta.pt using the same code, data and checkpoint.
Image alignment requires CUDA for practical runtimes. The main result's stored
final image scores remain the source for cohort evaluation.

    python scripts/render_posterior3d.py --out data/test30/p00

Use --watch on another GPU to render while inference is running. This updates
result.pt history and writes render_hist.json plus montage images.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_posterior3d import build_world, montage

from fm3d.reg_metric import aligned_metrics
from fm3d.rigid_motion import amp_from_run_args, motion_error, params_to_Pmot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the run dir (contains snaps/)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--watch", action="store_true",
                    help="poll snaps/ and render as snapshots appear (for running beside a "
                         "live loop, ideally on the other GPU); exits when every expected "
                         "step is rendered. Without it: render what exists, then exit.")
    ap.add_argument("--poll", type=float, default=10.0, help="watch poll interval [s]")
    args = ap.parse_args()

    snap_dir = os.path.join(args.out, "snaps")
    meta_path = os.path.join(snap_dir, "meta.pt")
    if not os.path.isfile(meta_path):
        raise SystemExit(f"{meta_path} not found -- was the run started with "
                         f"--metric_mode defer?")
    a = torch.load(meta_path, map_location="cpu", weights_only=False)["args"]

    dev = args.device
    world = build_world(ckpt=a["ckpt"], dev=dev, root=a.get("root"), split=a.get("split", "val"),
                        run=a["run"],
                        motion_kind=a["motion_kind"], seed=a["seed"],
                        # .get, not [...]: snapshots written before the amplitude flags existed
                        # have no such key, and None reproduces make_motion's old defaults --
                        # which is exactly the world those runs were simulated in.
                        trans_mm=amp_from_run_args(a)[0], rot_deg=amp_from_run_args(a)[1])
    cfg, gen = world["cfg"], world["gen"]
    spacing, meas = world["spacing"], world["meas"]
    gt3, theta_true = world["gt3"], world["theta_true"]
    y, static_fdk = world["y"], world["static_fdk"]

    # the fixed panels/references, fitted once -- identical to the loop's own startup
    with torch.no_grad():
        x_cold = gen.fdk(y, gen.P_nom[None])[0]
    m_cold, x0_input = aligned_metrics(x_cold, gt3, spacing, mask=meas, iters=200,
                                       return_aligned=True)
    ms_cold = aligned_metrics(x_cold, static_fdk, spacing, mask=meas, iters=200)
    m_sfdk = aligned_metrics(static_fdk, gt3, spacing, mask=meas, iters=200)
    # The montage's reference panel is FDK(theta_TRUE) -- the ceiling this data can actually reach
    # with an FDK -- not the static FDK, which is a different (motion-free) scan. See
    # run_posterior3d.montage. The static FDK remains the vs-sFDK metric reference.
    with torch.no_grad():
        ceil_fdk = gen.fdk(y, params_to_Pmot(theta_true, gen.P_nom)[None])[0]
    m_ceil, ceil_aligned = aligned_metrics(ceil_fdk, gt3, spacing, mask=meas, iters=200,
                                           return_aligned=True)
    print(f"static FDK (vs GT, Thies' reference)  "
          f"{m_sfdk['psnr_aligned']:.2f} dB / SSIM {m_sfdk['ssim_aligned']:.3f}")
    print(f"FDK(theta_true) = REACHABLE FDK CEILING (vs GT)  "
          f"{m_ceil['psnr_aligned']:.2f} dB / SSIM {m_ceil['ssim_aligned']:.3f}")

    N, me_every = int(a["n_steps"]), max(int(a.get("metric_every", 1)), 1)
    expected = sorted({k for k in range(N) if k % me_every == 0 or k == N - 1})

    gauge_th = xt_gauge = None
    rows: dict[int, dict] = {}
    while True:
        have = {int(os.path.basename(p)[4:7]): p
                for p in glob.glob(os.path.join(snap_dir, "step???.pt"))}
        todo = [k for k in expected if k not in rows and k in have]
        # STRICTLY IN STEP ORDER, even if later snapshots already exist: the gauge warm-start
        # chain assumes it. A later snapshot with an earlier one missing waits for it.
        for k in todo:
            if any(k2 not in rows and k2 not in have for k2 in expected if k2 < k):
                break
            d = torch.load(have[k], map_location=dev, weights_only=False)
            t, theta = float(d["t"]), d["theta"].to(dev)
            x = d["x_t"].to(dev).float()
            with torch.no_grad():
                x_fdk = gen.fdk(y, params_to_Pmot(theta, gen.P_nom)[None])[0]
            m, gauge_th, x_fdk_al = aligned_metrics(x_fdk, gt3, spacing, mask=meas, iters=150,
                                                    init=gauge_th, return_theta=True,
                                                    return_aligned=True)
            mx, xt_gauge, x_al = aligned_metrics(x, gt3, spacing, mask=meas, iters=150,
                                                 init=xt_gauge, return_theta=True,
                                                 return_aligned=True)
            ms_out = aligned_metrics(x_fdk, static_fdk, spacing, mask=meas, iters=150)
            ms_xt = aligned_metrics(x, static_fdk, spacing, mask=meas, iters=150)
            me = motion_error(theta, theta_true, cfg=cfg)
            rows[k] = {**m, **me,
                       **{f"xt_{q}": v for q, v in mx.items()},
                       **{f"s_{q}": v for q, v in ms_out.items()},
                       **{f"xts_{q}": v for q, v in ms_xt.items()}}
            print(f"step {k:3d} t={t:.2f} | OUT vsGT "
                  f"{m['psnr_aligned']:5.2f}/{m['ssim_aligned']:.3f} vsSFDK "
                  f"{ms_out['psnr_aligned']:5.2f}/{ms_out['ssim_aligned']:.3f} "
                  f"| x_t vsGT {mx['psnr_aligned']:5.2f}/{mx['ssim_aligned']:.3f} vsSFDK "
                  f"{ms_xt['psnr_aligned']:5.2f}/{ms_xt['ssim_aligned']:.3f} "
                  f"| rot {me['rot_rmse_deg']:.2f} deg", flush=True)
            montage(os.path.join(args.out, f"step{k:03d}.png"), gt3, x0_input, x_fdk_al, x_al,
                    ceil_aligned, k, t,
                    f"N={a['n_steps']} PER={a['per']} kappa={a['kappa']:g} | theta rot "
                    f"{me['rot_rmse_deg']:.2f} deg, trans_obs {me['trans_obs_mm']:.2f} mm",
                    m_in=m_cold, m_out=m, m_xt=mx,
                    ms_in=ms_cold, ms_out=ms_out, ms_xt=ms_xt, m_ceil=m_ceil)
        if all(k in rows for k in expected):
            break
        if not args.watch:
            missing = [k for k in expected if k not in rows]
            print(f"not watching and steps {missing} have no snapshot yet -- exiting "
                  f"(re-run later, the gauge chain restarts cleanly)")
            break
        time.sleep(args.poll)

    with open(os.path.join(args.out, "render_hist.json"), "w") as f:
        json.dump({str(k): v for k, v in rows.items()}, f, indent=1)

    # merge into result.pt so the cmp_* scripts see the inline-mode schema
    res_path = os.path.join(args.out, "result.pt")
    if os.path.isfile(res_path) and rows:
        r = torch.load(res_path, map_location="cpu", weights_only=False)
        n = 0
        for row in r.get("hist", []):
            if row.get("step") in rows:
                row.update(rows[row["step"]])
                n += 1
        torch.save(r, res_path)
        print(f"merged {n} metric rows into {res_path}")
    elif rows:
        print(f"{res_path} not there (run still going?) -- rows kept in render_hist.json; "
              f"re-run this script after the run to merge")
    print(f"montages -> {args.out}/  (judge by eye: streak-free, not the number)")


if __name__ == "__main__":
    main()
