"""Score the NATIVE JRM-ADM cohort (refs/jrm-adm/data/recon_ours_v2/pNN_result.pt) under the
manuscript's unified convention, so its numbers can go straight into Tables 2 and 3.

Per patient (skipped if already in the json):
  * x_est (their 224^3 grid, their mu constant) -> our frame: mu * 0.02/0.0193, zero-padded to
    256^3 at the centre (exactly what fig_jrm_cmp.py does); scored against the GT volume with
    `aligned_metrics(mask=meas, iters=200)` = the loop's own stored-metric convention;
  * theta: their (V,3,4) affines -> our (V,6) via jrm_theta_convert (gated), then the SAME
    fp64 RPE as cmp_thies_vs_ours (zero-centred gauge) and the per-DoF MAE of
    fig_motion_arxiv.py (zero-mean gauge, translations mm / rotations deg);
  * FDK(theta_hat_JRM) with OUR vanilla operator, scored the same way (the FDK-class witness);
  * runtime from result.pt.
Pairing is asserted on theta_true against our cohort run, as every cmp script does.

Summary: cohort mean +- std, paired Wilcoxon vs our final iterate / FDK(theta_hat), win counts.

    CUDA_VISIBLE_DEVICES=1 python scripts/score_jrm_native.py          # -> data/jrm_native_test30/scores.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fm3d.geometry_3d import ConeBeam3DConfig, build_conebeam_orbit   # noqa: E402
from fm3d.reg_metric import aligned_metrics                            # noqa: E402
from fm3d.rigid_motion import params_to_Pmot, reprojection_error, zero_centre_gauge  # noqa: E402
from jrm_theta_convert import jrm_thetas_to_ours                       # noqa: E402

JRM = "refs/jrm-adm/data/recon_ours_v2"
OURS = "data/test30"
DATA_ROOT = None
CKPT = "logs/fm3d_databridge/ckpt_iter500000.pth"
MU_RATIO = 0.02 / 0.0193


def pad_center(v, shape=(256, 256, 256)):
    out = torch.zeros(shape, dtype=v.dtype, device=v.device)
    o = [(a - b) // 2 for a, b in zip(shape, v.shape)]
    out[o[0]:o[0] + v.shape[0], o[1]:o[1] + v.shape[1], o[2]:o[2] + v.shape[2]] = v
    return out


def score_one(i: int, dev: str) -> dict:
    from run_posterior3d import build_world
    tag = f"p{i:02d}"
    jrm = torch.load(os.path.join(JRM, f"{tag}_result.pt"), map_location="cpu", weights_only=False)
    ours = torch.load(os.path.join(OURS, tag, "result.pt"), map_location="cpu", weights_only=False)
    tt = ours["theta_true"].double()
    devi = float((jrm["theta_true"].double() - tt).abs().max())
    assert devi < 1e-5, f"{tag}: JRM and ours saw different motion (|dtheta| {devi:.2e})"

    w = build_world(ckpt=CKPT, root=DATA_ROOT, dev=dev, split="test", run=i, seed=1000 + i,
                    motion_kind="akima", trans_mm=10.0, rot_deg=10.0)
    gen, gt3, meas, spacing = w["gen"], w["gt3"], w["meas"], w["spacing"]
    assert (w["theta_true"].double().cpu() - tt).abs().max() < 1e-5, f"{tag}: world pairing"

    th = jrm_thetas_to_ours(jrm["thetas_est"].float().to(dev))            # (V,6) our convention
    with torch.no_grad():
        xe = pad_center(jrm["x_est"][0, 0].float().to(dev) * MU_RATIO)
        fdk = gen.fdk(w["y"], params_to_Pmot(th, gen.P_nom)[None])[0]
        fdk = fdk.reshape(256, 256, 256) if fdk.ndim > 3 else fdk
    m_out = aligned_metrics(xe, gt3, spacing, mask=meas, iters=200)
    m_fdk = aligned_metrics(fdk, gt3, spacing, mask=meas, iters=200)

    P = build_conebeam_orbit(ConeBeam3DConfig.thies(n_views=tt.shape[0]), device="cpu").double()
    th64 = th.double().cpu()
    th_zc = zero_centre_gauge(th64)
    rpe_zc = reprojection_error(th_zc, tt, P)["rpe_mm"]
    rpe_raw = reprojection_error(th64, tt, P)["rpe_mm"]
    mae = (th_zc - tt).abs().mean(0).numpy()
    return dict(tag=tag,
                out_psnr=float(m_out["psnr_aligned"]), out_ssim=float(m_out["ssim_aligned"]),
                fdk_psnr=float(m_fdk["psnr_aligned"]), fdk_ssim=float(m_fdk["ssim_aligned"]),
                rpe_zc=float(rpe_zc), rpe_raw=float(rpe_raw),
                mae_t_mm=[float(x) for x in mae[:3]],
                mae_r_deg=[float(np.degrees(x)) for x in mae[3:]],
                runtime_min=float(jrm["runtime_sec"]) / 60.0,
                ours_xt_psnr=float(ours["final_xt"]["psnr_aligned"]),
                ours_xt_ssim=float(ours["final_xt"]["ssim_aligned"]),
                ours_fdk_psnr=float(ours["final"]["psnr_aligned"]),
                ours_fdk_ssim=float(ours["final"]["ssim_aligned"]))


def summary(rows: list[dict]):
    from scipy.stats import wilcoxon
    n = len(rows)
    g = lambda k: np.array([r[k] for r in rows])  # noqa: E731
    print(f"\n=== native JRM-ADM, {n} patients (unified convention: aligned, mask=meas) ===")
    for k, lab in (("out_psnr", "JRM-ADM output   PSNR"), ("out_ssim", "JRM-ADM output   SSIM"),
                   ("fdk_psnr", "FDK(theta_JRM)   PSNR"), ("fdk_ssim", "FDK(theta_JRM)   SSIM"),
                   ("rpe_zc", "RPE zero-centred (mm)"), ("rpe_raw", "RPE raw (mm)"),
                   ("runtime_min", "runtime (min)")):
        v = g(k); print(f"  {lab:24s} {v.mean():7.3f} +- {v.std():.3f}")
    mt = np.array([r["mae_t_mm"] for r in rows]).mean(0)
    mr = np.array([r["mae_r_deg"] for r in rows]).mean(0)
    print(f"  per-DoF MAE  t(mm) {mt[0]:.2f} {mt[1]:.2f} {mt[2]:.2f} | r(deg) {mr[0]:.2f} {mr[1]:.2f} {mr[2]:.2f}")
    if n >= 6:
        for a, b, lab in (("ours_xt_psnr", "out_psnr", "our x_t - JRM output, PSNR"),
                          ("ours_xt_ssim", "out_ssim", "our x_t - JRM output, SSIM"),
                          ("ours_fdk_psnr", "fdk_psnr", "our FDK(th) - FDK(th_JRM), PSNR"),
                          ("ours_fdk_ssim", "fdk_ssim", "our FDK(th) - FDK(th_JRM), SSIM")):
            d = g(a) - g(b)
            print(f"  {lab:34s} {d.mean():+7.3f}  wins {(d > 0).sum()}/{n}  p={wilcoxon(d).pvalue:.1e}")


def main():
    global JRM, OURS, CKPT, DATA_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/jrm_native_test30/scores.json")
    ap.add_argument("--only", type=int, nargs="*", default=None)
    ap.add_argument("--jrm", default=JRM)
    ap.add_argument("--ours", default=OURS)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    JRM, OURS, CKPT, DATA_ROOT = a.jrm, a.ours, a.ckpt, a.root
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    have = {r["tag"] for r in rows}
    done = sorted(int(re.search(r"p(\d+)_result", p).group(1))
                  for p in glob.glob(os.path.join(JRM, "p*_result.pt")))
    if a.only is not None:
        done = [i for i in done if i in a.only]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for i in done:
        if f"p{i:02d}" in have:
            continue
        r = score_one(i, dev)
        rows.append(r); rows.sort(key=lambda r: r["tag"])
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"{r['tag']}: out {r['out_psnr']:.2f}/{r['out_ssim']:.3f}  fdk {r['fdk_psnr']:.2f}/"
              f"{r['fdk_ssim']:.3f}  rpe_zc {r['rpe_zc']:.2f}  ({r['runtime_min']:.0f} min)", flush=True)
    summary(rows)


if __name__ == "__main__":
    main()
