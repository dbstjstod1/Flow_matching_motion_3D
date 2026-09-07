"""arXiv manuscript motion-trajectory figure + per-DoF MAE table (user request, 2026-09-03).

Six panels (tx, ty, tz in mm; rx, ry, rz in degrees) against the view index for ONE test
patient: the true trajectory, the uncorrected state (theta = 0, i.e. before correction) and
the trajectory each compared method recovered. Every estimate is moved onto the zero-mean
gauge with `zero_centre_gauge` (no ground truth involved), the convention the simulator
writes theta_true in, so the unobservable global rigid transform is removed before plotting.
The same gauge is used for the cohort MAE printed at the end (30 patients, per DoF).

    python scripts/fig_motion_arxiv.py --tag p14        # -> docs/arxiv/fig_motion.{pdf,png}
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm3d.rigid_motion import zero_centre_gauge   # noqa: E402

TW = 6.5
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"          # the validated categorical slots
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a85", "#e3e3df"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "figure.dpi": 400, "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})

# (label, run dir, theta key, colour) -- method NAMES, fixed order = fixed hue
ARMS = [
    ("Learned autofocus", "data/bench_thies_test30", "theta_hat", C2),
    ("JRM-ADM", "refs/jrm-adm/data/recon_ours_v2", "thetas_est", "#9a5bd2"),
    ("Proposed, pixel-linear bridge", "data/linbridge_test30", "theta", C3),
    ("Proposed", "data/fm3d_test30_databridge", "theta", C1),
]
DOF = [("$t_x$ (mm)", 0, 1.0), ("$t_y$ (mm)", 1, 1.0), ("$t_z$ (mm)", 2, 1.0),
       ("$r_x$ (deg)", 3, 180 / np.pi), ("$r_y$ (deg)", 4, 180 / np.pi),
       ("$r_z$ (deg)", 5, 180 / np.pi)]


def load(tag):
    tt = torch.load(f"data/fm3d_test30_databridge/{tag}/result.pt", map_location="cpu",
                    weights_only=False)["theta_true"].double()
    est = {}
    for name, d, key, _ in ARMS:
        if key == "thetas_est":                 # native JRM-ADM: (V,3,4) affines -> our (V,6)
            from jrm_theta_convert import jrm_thetas_to_ours
            r = torch.load(f"{d}/{tag}_result.pt", map_location="cpu", weights_only=False)
            th = jrm_thetas_to_ours(r[key].float()).double()
        else:
            r = torch.load(f"{d}/{tag}/result.pt", map_location="cpu", weights_only=False)
            th = r[key].double()
        assert (r["theta_true"].double() - tt).abs().max() < 1e-5, (name, tag)
        est[name] = zero_centre_gauge(th).numpy()
    return tt.numpy(), est


def figure(tag, outdir):
    tt, est = load(tag)
    V = np.arange(tt.shape[0])
    fig, axes = plt.subplots(2, 3, figsize=(TW, 3.3), sharex=True)
    for ax, (lab, j, scale) in zip(axes.flat, DOF):
        ax.axhline(0.0, color=MUTED, lw=1.0, ls=(0, (3, 2)), zorder=1)
        ax.plot(V, tt[:, j] * scale, color=INK, lw=1.6, zorder=2)
        for name, d, key, col in ARMS:
            ax.plot(V, est[name][:, j] * scale, color=col, lw=1.1, zorder=3)
        ax.set_ylabel(lab, labelpad=2)
        ax.grid(True, color=GRID, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=2.5, width=0.6)
    for ax in axes[1]:
        ax.set_xlabel("view index")
        ax.set_xlim(0, V[-1])
    # one legend for the whole figure, above the panels
    handles = [plt.Line2D([], [], color=INK, lw=1.6, label="True motion"),
               plt.Line2D([], [], color=MUTED, lw=1.0, ls=(0, (3, 2)),
                          label="Before correction ($\\theta=0$)")]
    handles += [plt.Line2D([], [], color=col, lw=1.1, label=name) for name, _, _, col in ARMS]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.02), handlelength=2.2, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.6, w_pad=0.8)
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig_motion.{ext}"))
    plt.close(fig)
    print("wrote", os.path.join(outdir, "fig_motion.{png,pdf}"), "for", tag)


def cohort_mae(n=30):
    rows = {"No compensation": []}
    for name, *_ in ARMS:
        rows[name] = []
    for i in range(n):
        tag = f"p{i:02d}"
        tt, est = load(tag)
        rows["No compensation"].append(np.abs(tt).mean(0))
        for name in est:
            rows[name].append(np.abs(est[name] - tt).mean(0))
    print(f"\nper-DoF MAE over {n} patients (zero-mean gauge): tx ty tz [mm] | rx ry rz [deg]")
    for name, v in rows.items():
        m = np.array(v).mean(0)
        print(f"  {name:30s} {m[0]:.2f} {m[1]:.2f} {m[2]:.2f} | "
              f"{np.degrees(m[3]):.2f} {np.degrees(m[4]):.2f} {np.degrees(m[5]):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p14")
    ap.add_argument("--outdir", default="docs/arxiv")
    ap.add_argument("--no_table", action="store_true")
    a = ap.parse_args()
    figure(a.tag, a.outdir)
    if not a.no_table:
        cohort_mae()
