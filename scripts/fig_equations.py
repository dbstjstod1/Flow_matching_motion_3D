"""Equation sheets for talks: the blind inverse problem, and the scheme that breaks it.

Typeset with matplotlib's STIX math engine (there is no LaTeX on this machine), one numbered
equation per row with its annotation in the right margin -- a paper's method section, not a
box-and-arrow diagram. The companion diagrams are scripts/fig_workflow.py.

    python scripts/fig_equations.py                 # -> figs/eq_{problem,scheme,sheet}.{png,pdf,svg}

Numbers quoted in the annotations come from this repo: geometry from
fm3d/geometry_3d.ConeBeam3DConfig.thies, loop settings from scripts/run_posterior3d.py,
bridge/loss from scripts/train_fm3d.py.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

FIG_W, FIG_H = 160.0, 90.0

EQ_X = 50.0          # equations are centred here
NOTE_X = 100.0       # annotations start here
NUM_X = 157.0        # equation numbers, right-aligned
INK = "#0f172a"
GREY = "#475569"


def frame(ax):
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")


def title(ax, text, sub=""):
    ax.text(FIG_W / 2, 85.0, text, ha="center", va="center", fontsize=16.5, color=INK)
    if sub:
        ax.text(FIG_W / 2, 79.8, sub, ha="center", va="center", fontsize=9.6, color=GREY,
                fontstyle="italic")


def head(ax, y, text):
    ax.text(4.0, y, text, ha="left", va="center", fontsize=11.0, color=INK)
    ax.plot([4.0, 156.0], [y - 2.6, y - 2.6], lw=0.9, color="#cbd5e1")


def eq(ax, y, expr, note="", num=None, fs=14.5, note_fs=8.6, color=INK):
    """One numbered display equation with a right-margin annotation."""
    ax.text(EQ_X, y, expr, ha="center", va="center", fontsize=fs, color=color)
    if note:
        ax.text(NOTE_X, y, note, ha="left", va="center", fontsize=note_fs, color=GREY,
                linespacing=1.5)
    if num is not None:
        ax.text(NUM_X, y, f"({num})", ha="right", va="center", fontsize=10.0, color=GREY)


def rule(ax, y):
    ax.plot([4.0, 156.0], [y, y], lw=0.9, color="#e2e8f0")


# ==========================================================================================
def draw_problem(ax):
    frame(ax)
    title(ax, "Blind rigid-motion CBCT — the problem",
          "one measurement, two unknowns: the geometry cannot be fitted without the image, "
          "and the image cannot be solved without the geometry")

    eq(ax, 71.0,
       "$y \\;=\\; A_{P(\\theta)}\\,x \\;+\\; n,\\qquad "
       "P(\\theta) \\;=\\; P_{nom}\\,T(\\theta)$",
       "the cone-beam forward model.  $A_{P}$ is the projector for the\n"
       "$V$ per-view matrices $P$;  $T(\\theta)$ is the rigid motion.", num=1)
    eq(ax, 64.5,
       "$y\\in\\mathbb{R}^{V\\times n_v\\times n_u},\\quad "
       "x\\in\\mathbb{R}^{N},\\quad \\theta\\in\\mathbb{R}^{V\\times 6}$",
       "$V=360$ views,  $n_v\\times n_u = 500\\times700$,  $N=256^3$:\n"
       "$126$ M measurements,  $16.8$ M $+$ $2\\,160$ unknowns.", num=2, fs=12.5)

    rule(ax, 58.5)
    eq(ax, 53.0,
       "$(\\hat x,\\hat\\theta) \\;=\\; \\mathrm{arg\\,min}_{\\,x,\\,\\theta}\\;"
       "\\frac{1}{2}\\Vert A_{P(\\theta)}x-y\\Vert ^2 \\;+\\; \\lambda\\,\\mathrm{TV}(x) "
       "\\;+\\; \\mathcal{R}_\\psi(x)$",
       "the joint (blind) problem.  It is BILINEAR in $(x,\\theta)$,\n"
       "hence non-convex: many minima explain the same $y$.", num=3)

    eq(ax, 44.0,
       "$\\Theta(x) \\;\\equiv\\; \\mathrm{arg\\,min}_{\\,\\theta}\\;"
       "\\frac{1}{2}\\Vert A_{P(\\theta)}x-y\\Vert ^2$",
       "GEOMETRY given the image.  Measured here: from the cold FDK\n"
       "every estimator config plateaus at $\\approx\\!2.0\\degree$;  from a clean\n"
       "reference the same budget reaches $0.09$–$0.27\\degree$.", num=4)
    eq(ax, 35.0,
       "$X(\\theta) \\;\\equiv\\; \\mathrm{arg\\,min}_{\\,x}\\;"
       "\\frac{1}{2}\\Vert A_{P(\\theta)}x-y\\Vert ^2 + \\lambda\\,\\mathrm{TV}(x)$",
       "IMAGE given the geometry.  Measured here: the same loop reaches\n"
       "$40.45$ dB with the true $\\theta$ from step 0,  $38.16$ dB blind.", num=5)

    eq(ax, 26.0,
       "$\\hat\\theta \\;=\\; \\Theta(X(\\hat\\theta)),\\qquad "
       "\\hat x \\;=\\; X(\\Theta(\\hat x))$",
       "THE CIRCULARITY, stated exactly: each unknown is defined through\n"
       "the other, so neither map can be evaluated on its own.", num=6)

    rule(ax, 20.0)
    eq(ax, 14.5,
       "$A_{P(\\theta\\circ g^{-1})}\\,(g\\!\\cdot\\!x) \\;=\\; A_{P(\\theta)}\\,x"
       "\\qquad \\forall\\, g\\in SE(3)$",
       "an EXACT gauge: a global rigid pose is unobservable, so the\n"
       "solution is an ORBIT $\\{(g\\!\\cdot\\!\\hat x,\\;\\hat\\theta\\circ g^{-1})\\}$, "
       "not a point.", num=7)
    ax.text(EQ_X, 7.5, "depth along the beam:  $\\partial y/\\partial t_{\\parallel} \\approx 0$",
            ha="center", va="center", fontsize=11.0, color=INK)
    ax.text(NOTE_X, 7.5, "and beyond the gauge, translation ALONG THE BEAM is nearly\n"
            "unobservable — 92% of a converged estimator's residual lives there.",
            ha="left", va="center", fontsize=8.6, color=GREY, linespacing=1.5)
    ax.text(NUM_X, 7.5, "(8)", ha="right", va="center", fontsize=10.0, color=GREY)


# ==========================================================================================
def draw_scheme(ax):
    frame(ax)
    title(ax, "The scheme — a prior on the geometry bridge, then alternating minimization",
          "the prior supplies the information the data cannot; the alternation is what makes "
          "each unknown computable from the other")

    head(ax, 73.0, "Training  (offline, on simulated motion)")
    eq(ax, 66.5,
       "$x_t \\;=\\; \\mathrm{FDK}(y,\\,P_{nom}T(t\\theta)) \\;+\\; t\\,\\Delta,\\qquad "
       "\\Delta = x_{static}-\\mathrm{FDK}(y,\\,P_{nom}T(\\theta))$",
       "the geometry bridge:  $t\\!=\\!0$ is the uncorrected recon (the\n"
       "inference cold start),  $t\\!=\\!1$ the motion-free scan.", num=9, fs=13.0)
    eq(ax, 58.5,
       "$\\psi^\\star \\;=\\; \\mathrm{arg\\,min}_{\\,\\psi}\\;"
       "\\mathbb{E}_{t,\\theta,x}\\,\\Vert v_\\psi(x_t,t)-\\dot x_t\\Vert ^2,\\qquad "
       "\\dot x_t=\\frac{d}{dt}x_t$",
       "flow matching against the EXACT tangent of the bridge, so $v_\\psi$\n"
       "points along the set of reachable reconstructions.", num=10, fs=13.0)

    head(ax, 50.0, "Inference  (blind;  $k=0,\\ldots,N\\!-\\!1$,   $t_k=k/N$,   "
                   "$\\Delta t=1/N$,   $N=50$)")
    eq(ax, 43.0,
       "$\\tilde x_k \\;=\\; x_k \\;+\\; \\Delta t\\; v_{\\psi^\\star}(x_k,\\,t_k)$",
       "PREDICT — the prior moves first, evaluated on $32^3$ tiles\n"
       "and blended back to the volume.", num=11)
    eq(ax, 34.5,
       "$\\theta_{k+1} \\;\\approx\\; \\mathrm{arg\\,min}_{\\,\\theta}\\;"
       "L\\,(A_{P(\\theta)}\\tilde x_k,\\; y)$",
       "ESTIMATE — 50 warm-started Adam steps on a coordinate net over\n"
       "the view index.  Note the argument: $\\tilde x_k$, not $x_k$ (Gauss–Seidel).", num=12)
    eq(ax, 26.0,
       "$z_k \\;=\\; \\mathrm{arg\\,min}_{\\,z}\\;\\Vert A_{P(\\theta_{k+1})}z-y\\Vert ^2$",
       "DATA — 5 CG iterations warm-started at $\\tilde x_k$; the constraint set\n"
       "itself has just been re-aimed by (12).", num=13)
    eq(ax, 17.5,
       "$x_{k+1} \\;=\\; z_k \\;+\\; \\kappa\\,(\\mathrm{TV}(z_k)-z_k)$",
       "PnP corrector, $\\kappa=0.3$ — an OOD-safe denoiser, unlike using\n"
       "$v_{\\psi^\\star}$ itself inside the splitting.", num=14)

    rule(ax, 12.0)
    eq(ax, 7.0,
       "$\\bar\\theta=\\frac{1}{K}\\sum_{j=N-K+1}^{N}\\theta_j,\\qquad "
       "x_{out}=\\mathrm{FDK}(y,\\,P_{nom}T(\\bar\\theta))$",
       "readout, $K=2$: the estimator's tail is a period-2 limit cycle,\n"
       "so the mean is taken rather than the last step.", num=15, fs=13.0)


# ==========================================================================================
def save(fig, path_noext, dpi, formats):
    for ext in formats:
        p = f"{path_noext}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.18, facecolor="white")
        print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--formats", nargs="+", default=["png", "pdf", "svg"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name, draw in (("eq_problem", draw_problem), ("eq_scheme", draw_scheme)):
        fig, ax = plt.subplots(figsize=(16, 9))
        draw(ax)
        fig.subplots_adjust(0, 0, 1, 1)
        save(fig, os.path.join(args.out, name), args.dpi, args.formats)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(16, 18))
    draw_problem(axes[0])
    draw_scheme(axes[1])
    fig.subplots_adjust(0, 0, 1, 1, hspace=0.04)
    save(fig, os.path.join(args.out, "eq_sheet"), args.dpi, args.formats)
    plt.close(fig)


if __name__ == "__main__":
    main()
