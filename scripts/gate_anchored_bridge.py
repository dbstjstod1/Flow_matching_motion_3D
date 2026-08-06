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
                              measured_region_mask, view_angular_weights,
                              view_angular_weights_dot)
from fm3d.phantom import head_phantom
from fm3d.projector_3d import (fdk_conebeam_3d_batched, fdk_conebeam_3d_tangent,
                               forward_project_3d_batched)
from fm3d.rigid_motion import akima_motion, params_to_Pmot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_fm3d import bridge_pair                                    # the REAL function

FAIL = []


def check(i, name, ok, detail=""):
    print(f"[{i}] {name:<54s} {'PASS' if ok else 'FAIL'}  {detail}")
    if not ok:
        FAIL.append(name)


class Gen:
    """The minimum `bridge_pair` needs -- on a phantom, with no dataset. Since bridge_pair's
    default is mode="analytic", that minimum includes `fdk_tangent` and `to_net_tangent`,
    mirroring `CQ500Generator` (same Voronoi weight + weight-derivative plumbing)."""

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
        # no fitted scale: the FDK self-normalizes (projector_3d._fdk_physical_norm)

    def to_net(self, mu):
        return 2.0 * (mu - self.mu_lo) / (self.mu_hi - self.mu_lo) - 1.0

    def to_net_tangent(self, dmu):
        """to_net is affine, so a DERIVATIVE maps with the gain only (no -1 shift)."""
        return 2.0 * dmu / (self.mu_hi - self.mu_lo)

    def fdk_tangent(self, y, P, Pdot, filtered=None):
        # `filtered` is the trainer's one-ramp-per-draw reuse (dataset_cq500.Gen.fdk_filtered);
        # this stub only has to accept and forward it -- `bridge_pair` passes it through.
        vw, vwd = view_angular_weights_dot(P, Pdot)
        return fdk_conebeam_3d_tangent(
            y, P, Pdot, self.uc, self.vc, self.cfg, D=self.shape[0], H=self.shape[1],
            W=self.shape[2], dx=self.dx, dy=self.dy, dz=self.dz, scale=None,
            view_chunk=8, view_weight=vw, view_weight_dot=vwd, filtered=filtered)

    def project(self, v, P):
        return forward_project_3d_batched(v, P, self.uc, self.vc, dx=self.dx, dy=self.dy,
                                          dz=self.dz)

    def fdk(self, y, P, **kw):
        return fdk_conebeam_3d_batched(y, P, self.uc, self.vc, self.cfg, D=self.shape[0],
                                       H=self.shape[1], W=self.shape[2], dx=self.dx, dy=self.dy,
                                       dz=self.dz, scale=None, view_chunk=8,
                                       view_weight=view_angular_weights(P), **kw)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = Gen(dev)
    th = akima_motion(g.V, n_nodes=10, trans_mm=10.0, rot_deg=10.0, device=dev, seed=0)

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

    # mode="fd" is the escape hatch / the gate's counterparty: ITS velocity is literally the
    # central difference of the anchored path at its defining step (delta = 0.02), so a manual
    # FD of the fd-mode path must reproduce it exactly. This is the old exactness claim, kept
    # where it is still true -- it asserts the anchor bookkeeping (+t*Delta in x, +Delta in dx)
    # is consistent inside the fd path.
    h = 0.02
    with torch.no_grad():
        xa, _ = bridge_pair(g, T(0.5 - h), y, th, dlt, mode="fd")
        xb, _ = bridge_pair(g, T(0.5 + h), y, th, dlt, mode="fd")
        _, d_fd = bridge_pair(g, T(0.5), y, th, dlt, mode="fd")
    fd = (xb - xa) / (2 * h)
    rel = float((fd - d_fd).norm() / fd.norm())
    check(3, "... and the fd-mode velocity IS the FD of the anchored path", rel < 1e-3,
          f"rel {rel:.1e} at h={h}")

    # The ANALYTIC velocity (the default the trainer uses) cannot be checked against a FD at
    # ONE step size: the bilinear interpolant is C0, so a central difference converges only at
    # O(h) across cell-crossing voxels, and the size of that floor depends on how kinky the
    # sinogram is -- i.e. ON THE FORWARD OPERATOR. Fixing a threshold at h=0.02 therefore
    # measured the operator, not the tangent: the ray-march era read cos 0.9875 and SF reads
    # 0.9734 at the same h, with NOTHING wrong in either (user's diagnosis, 2026-07-28).
    #
    # So assert the property that actually defines "exact derivative": CONVERGENCE. Measured
    # h-sweep on SF (scratchpad hsweep, 2026-07-28) -- cos 0.863 / 0.944 / 0.973 / 0.987 /
    # 0.993 / 0.997 and rel 4.7e-2 -> 6.3e-3 as h goes 0.08 -> 0.0025, textbook O(h). A WRONG
    # analytic tangent (wrong dP, dropped weight derivative) plateaus instead, which is what
    # this now catches -- and it catches it independently of which projector is installed.
    hs = (0.04, 0.01)
    rels, coss = [], []
    for hh in hs:
        with torch.no_grad():
            xa2, _ = bridge_pair(g, T(0.5 - hh), y, th, dlt)
            xb2, _ = bridge_pair(g, T(0.5 + hh), y, th, dlt)
        fd2 = (xb2 - xa2) / (2 * hh)
        rels.append(float((d_anc - fd2).pow(2).mean().sqrt() / fd2.abs().amax().clamp_min(1e-30)))
        coss.append(float((d_anc * fd2).sum() / (d_anc.norm() * fd2.norm())))
    ratio = rels[0] / max(rels[1], 1e-30)                 # 4x smaller h should ~halve O(h) error
    check(3, "... and the ANALYTIC velocity is the EXACT derivative (FD converges to it)",
          ratio > 1.5 and coss[1] > coss[0] and rels[1] < 3e-2,
          f"rel {rels[0]:.2e}(h={hs[0]}) -> {rels[1]:.2e}(h={hs[1]}), x{ratio:.1f} smaller; "
          f"cos {coss[0]:.4f} -> {coss[1]:.4f}")

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
    # MEASURE THIS SUB-CHECK WITH AN UNAPODIZED RAMP. It is an ATTRIBUTION claim -- "the
    # static-vs-GT deficit is the cone" -- and since 2026-07-29 the pipeline's FDK apodizes
    # (filters.DEFAULT_RAMP_WINDOW, see projector_3d._RAMP_WINDOW_NOTE) to band-limit the SF
    # cube basis. That
    # apodization is a SECOND, deliberate contributor to the deficit, and it is spatially uniform,
    # so it swamps the midplane's cone advantage and the comparison stops measuring the cone
    # (measured: the gap flips from +2.13 dB under ramlak to -0.41 dB under hann). Isolating the
    # cone therefore requires the sharp filter, even though the pipeline no longer uses it.
    with torch.no_grad():
        st_sharp = g.to_net(g.fdk(g.project(g.vol, g.P_nom[None]), g.P_nom[None],
                                  window="ramlak")[0])
    e_all_s, e_mid_s = (st_sharp - b)[g.meas], (st_sharp - b)[mid]
    p_st_s = float(20 * np.log10(rng / (e_all_s.pow(2).mean().sqrt().item() + 1e-12)))
    p_mid_s = float(20 * np.log10(rng / (e_mid_s.pow(2).mean().sqrt().item() + 1e-12)))
    check(5, "... and that floor is a CONE effect (midplane is better, ramlak)",
          p_mid_s - p_st_s > 0.5,
          f"midplane-only {p_mid_s:.2f} dB  vs  whole volume {p_st_s:.2f} dB "
          f"({p_mid_s - p_st_s:+.2f}; on a real head this gap is +9.9)")
    check(5, "the GT anchor reaches the GT", psnr(x1g) > 100.0, f"{psnr(x1g):.0f} dB")

    n = len(FAIL)
    print(f"\n{'ALL PASS' if n == 0 else f'{n} FAILURE(S): ' + ', '.join(FAIL)}")
    sys.exit(1 if n else 0)


if __name__ == "__main__":
    main()
