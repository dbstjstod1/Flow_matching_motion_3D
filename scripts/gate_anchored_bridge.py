"""Gate the ANCHORED geometry bridge. Phantom only -- no CQ500, no checkpoint. ~2 min.

The bare geometry bridge, x_t = FDK(y, P_nom @ T(t*theta)), has a contaminated endpoint: handing
FDK the TRUE theta does NOT reproduce a static scan, because FDK is an analytic inverse derived
for a circular EQUIANGULAR orbit and per-view motion breaks that. A prior trained on it learns
FDK's residual motion artefact as its target, and would faithfully reproduce a not-clean image at
t=1. (The 2D sibling has the same bug in its geometry bridge -- `train_fm.py` literally comments
"x_t(1) = clean recon (P_true)". Its LINEAR bridge is fine, because there x1 is the clean CT.)

The anchor detrends it:  x_t = FDK(y, P(t*theta)) + t * (x_anchor - FDK(y, P(theta))).

Both endpoints then hold BY CONSTRUCTION, so they are asserted to machine precision, not measured:

  [1] t=0 IS THE COLD START.  x_0 == FDK(y, P_nom) exactly -- whatever theta is, whatever the
      anchor is. This is the image inference actually starts from; if the bridge does not begin
      there, the ODE's first step is off-manifold and nothing downstream can tell you.
  [2] t=1 IS THE TARGET.      x_1 == x_anchor exactly. This is the whole point.
  [3] the velocity is the true tangent of the anchored path (finite-difference consistency)
  [4] anchor="none" restores the bare bridge bit-for-bit (no silent behaviour change)
  [5] and the thing that motivated all this: on a real cone-beam geometry the BARE bridge's
      endpoint is measurably NOT the static reconstruction, while the anchored one IS.

    python scripts/gate_anchored_bridge.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.filters import calibrate_scale
from fm3d.geometry_3d import (ConeBeam3DConfig, build_conebeam_orbit, detector_coords_3d,
                              measured_region_mask, view_angular_weights)
from fm3d.phantom import head_phantom
from fm3d.projector_3d import fdk_conebeam_3d_batched, forward_project_3d_batched
from fm3d.rigid_motion import akima_motion, params_to_Pmot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_fm3d import bridge_pair                                    # the REAL function

FAIL = []


def check(i, name, ok, detail=""):
    print(f"[{i}] {name:<54s} {'PASS' if ok else 'FAIL'}  {detail}")
    if not ok:
        FAIL.append(name)


class Gen:
    """The minimum `bridge_pair` needs: P_nom, fdk, to_net -- on a phantom, with no dataset."""

    def __init__(self, dev):
        self.V = 360
        self.cfg = ConeBeam3DConfig.thies(n_views=self.V, det_bin=2)
        self.P_nom = build_conebeam_orbit(self.cfg, device=dev)
        self.uc, self.vc = detector_coords_3d(self.cfg, device=dev)
        self.shape = (96, 128, 128)
        self.dz = self.dy = self.dx = 1.5
        self.mu_lo, self.mu_hi = 0.0, (2000 / 1000 + 1) * 0.02
        self.vol = head_phantom(self.shape, (self.dz, self.dy, self.dx), device=dev)[None, None]
        self.meas = measured_region_mask(self.shape, (self.dz,) * 3, self.cfg, device=dev)
        self.fbp_scale = 1.0
        y0 = self.project(self.vol, self.P_nom[None])
        self.fbp_scale = calibrate_scale(self.fdk(y0, self.P_nom[None])[0], self.vol[0, 0],
                                         self.meas)

    def to_net(self, mu):
        return 2.0 * (mu - self.mu_lo) / (self.mu_hi - self.mu_lo) - 1.0

    def project(self, v, P):
        return forward_project_3d_batched(v, P, self.uc, self.vc, dx=self.dx, dy=self.dy,
                                          dz=self.dz, n_samples=384, view_chunk=8)

    def fdk(self, y, P):
        return fdk_conebeam_3d_batched(y, P, self.uc, self.vc, self.cfg, D=self.shape[0],
                                       H=self.shape[1], W=self.shape[2], dx=self.dx, dy=self.dy,
                                       dz=self.dz, scale=self.fbp_scale, view_chunk=8,
                                       view_weight=view_angular_weights(P))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = Gen(dev)
    th = akima_motion(g.V, n_nodes=10, trans_mm=5.0, rot_deg=5.0, device=dev, seed=0)

    with torch.no_grad():
        y = g.project(g.vol, params_to_Pmot(th, g.P_nom)[None])       # the motion-corrupted scan
        cold = g.to_net(g.fdk(y, g.P_nom[None])[0])                   # the inference cold start
        static = g.to_net(g.fdk(g.project(g.vol, g.P_nom[None]), g.P_nom[None])[0])
        x1_geo = g.to_net(g.fdk(y, params_to_Pmot(th, g.P_nom)[None])[0])
        dlt = static - x1_geo                                         # the anchor's Delta

    T = lambda v: torch.tensor(float(v))

    # ---- [1] t = 0 is the cold start, exactly ------------------------------------------
    with torch.no_grad():
        x0, _ = bridge_pair(g, T(0.0), y, th, dlt)
    e = float((x0 - cold).abs().max())
    check(1, "t=0 == FDK(y, P_nom): the inference cold start", e < 1e-5, f"max|d| {e:.1e}")

    # ---- [2] t = 1 is the anchor, exactly -----------------------------------------------
    with torch.no_grad():
        x1, _ = bridge_pair(g, T(1.0), y, th, dlt)
    e = float((x1 - static).abs().max())
    check(2, "anchor=static: t=1 == the motion-free reconstruction", e < 1e-5, f"max|d| {e:.1e}")
    e_bare = float((x1_geo - static).abs().max())
    check(2, "... which the BARE bridge does NOT reach", e_bare > 20 * max(e, 1e-9),
          f"bare max|d| {e_bare:.1e}  vs anchored {e:.1e}")

    # and the DEFAULT anchor: the ground truth itself. A motion-free FDK is not clean either --
    # it carries the cone-beam artefact of a circular orbit, which is a defect of the INVERSE and
    # not of the data, so the prior must not learn to reproduce it.
    with torch.no_grad():
        gt_net = g.to_net(g.vol[0, 0])
        x1g, _ = bridge_pair(g, T(1.0), y, th, gt_net - x1_geo)
    e = float((x1g - gt_net).abs().max())
    check(2, "anchor=gt (DEFAULT): t=1 == the GROUND TRUTH", e < 1e-5, f"max|d| {e:.1e}")

    # ---- [3] the velocity is the tangent of the ANCHORED path ---------------------------
    # The anchor adds a CONSTANT to the path, so it must add exactly that constant to the
    # velocity: dx_anchored - dx_bare == Delta, to the bit. That is the bookkeeping claim, and it
    # is the one worth asserting.
    with torch.no_grad():
        _, d_anc = bridge_pair(g, T(0.5), y, th, dlt)
        _, d_bare = bridge_pair(g, T(0.5), y, th, None)
    e = float((d_anc - d_bare - dlt).abs().max())
    check(3, "dx_anchored - dx_bare == Delta (exactly)", e < 1e-5, f"max|d| {e:.1e}")

    # And the finite difference reproduces it AT THE SAME STEP the velocity is defined with
    # (delta = 0.02). Comparing against a COARSER step would fail, and legitimately so: the
    # geometry bridge is genuinely CURVED (cos 0.987 / 17% rel at h = 0.05), which is exactly why
    # the prior regresses the true tangent instead of the secant x1 - x0.
    h = 0.02
    with torch.no_grad():
        xa, _ = bridge_pair(g, T(0.5 - h), y, th, dlt)
        xb, _ = bridge_pair(g, T(0.5 + h), y, th, dlt)
    fd = (xb - xa) / (2 * h)
    rel = float((fd - d_anc).norm() / fd.norm())
    check(3, "... and matches the FD at the step it is defined with", rel < 1e-3,
          f"rel {rel:.1e} at h={h}")

    # ---- [4] anchor=none restores the bare bridge bit-for-bit ---------------------------
    with torch.no_grad():
        xn, dn = bridge_pair(g, T(0.7), y, th, None)
        xg, dg = bridge_pair(g, T(0.7), y, th, torch.zeros_like(dlt))
    check(4, "dlt=None == dlt=0 (bare bridge, bit-for-bit)",
          float((xn - xg).abs().max()) == 0.0 and float((dn - dg).abs().max()) == 0.0)

    # ---- [5] the measurement that motivated the anchor -----------------------------------
    def psnr(a):
        b = g.to_net(g.vol[0, 0])
        e = (a - b)[g.meas]
        rng = float(b[g.meas].max() - b[g.meas].min())
        return float(20 * np.log10(rng / (e.pow(2).mean().sqrt().item() + 1e-12)))
    p_st, p_geo = psnr(static), psnr(x1_geo)
    check(5, "the BARE endpoint really is below the static scan", p_st - p_geo > 0.5,
          f"static {p_st:.2f} dB  vs  FDK(y, P(theta_true)) {p_geo:.2f} dB "
          f"({p_geo - p_st:+.2f}) -- with the Voronoi weight ALREADY on")
    check(5, "the STATIC anchor is the static scan", abs(psnr(x1) - p_st) < 0.01,
          f"{psnr(x1):.2f} dB")

    # AND the static scan is not clean either: the motion-free FDK sits well below the GT, and
    # the deficit is a CONE effect -- it nearly vanishes at the midplane. This is why the default
    # anchor is the GT and not the static reconstruction.
    D = g.shape[0]
    mid = torch.zeros_like(g.meas)
    mid[D // 2 - 8:D // 2 + 8] = True
    mid &= g.meas
    b = g.to_net(g.vol[0, 0])
    e_all = (static - b)[g.meas]
    e_mid = (static - b)[mid]
    rng = float(b[g.meas].max() - b[g.meas].min())
    p_mid = float(20 * np.log10(rng / (e_mid.pow(2).mean().sqrt().item() + 1e-12)))
    check(5, "the STATIC scan is NOT the GT either (cone-beam floor)", p_st < 45.0,
          f"static {p_st:.2f} dB vs GT")
    # The threshold is loose because THIS PHANTOM UNDERSTATES THE EFFECT: it is smooth ellipsoids
    # over 144 mm of z, so its cone artefact is mild. On a real CQ500 head over 256 mm the gap is
    # +9.9 dB (midplane 44.26 vs whole-volume 34.34), and quadrupling the views buys 0.1 dB --
    # i.e. it is the cone, not angular sampling. The gate only has to see the SIGN.
    check(5, "... and that floor is a CONE effect (midplane is better)", p_mid - p_st > 0.5,
          f"midplane-only {p_mid:.2f} dB  vs  whole volume {p_st:.2f} dB "
          f"(+{p_mid - p_st:.2f}; on a real head this gap is +9.9)")
    check(5, "the GT anchor reaches the GT", psnr(x1g) > 100.0, f"{psnr(x1g):.0f} dB")

    n = len(FAIL)
    print(f"\n{'ALL PASS' if n == 0 else f'{n} FAILURE(S): ' + ', '.join(FAIL)}")
    sys.exit(1 if n else 0)


if __name__ == "__main__":
    main()
