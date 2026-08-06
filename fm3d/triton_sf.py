"""Separable-footprint (SF-TT) cone-beam projector pair for ARBITRARY per-view P, with the
geometry gradient nobody ships (v2, CORNER-PARAMETERIZED, 2026-07-28).

WHY THIS EXISTS. The loop's two heavy consumers of the projector are the CG data step (wants a
MATCHED A/A^T) and the motion estimator (wants d loss/dP). The ray-march kernel provides both
but pays a 3.9 s atomic-scatter transpose; the toolkits escape via unmatched gathers; LEAP's SF
kernels are matched and fast but hardwired to the ideal circular orbit and differentiate the
volume only. The 4DCT sibling (`fdct/sf_projector.py`) reimplemented LEAP's SF faithfully --
detector-driven, circular-orbit-only, no d/dP. This module is the arbitrary-P + d/dP variant.

WHY CORNERS (v2). v1 parameterized the footprint by half-base q and taper h = min(a, b) of
Jacobian-lumped widths. Its d/dP was verified autograd-exact against a torch reference -- and
the estimator still DIVERGED: the (q, h) coordinates manufacture sub-cell gradient components
(|J| lumping, the min() branch, a 1/h spike at the rect limit) that the PHYSICAL loss surface
does not have; finite differences average them away, the exact gradient chases them. See the
sf-projector memory's hypothesis ledger. v2 removes the coordinates instead of patching them:

  * u-footprint = the SORTED PROJECTIONS OF THE FOUR TRANSAXIAL CORNERS ju1<=..<=ju4 -- the
    exact (projective) trapezoid vertices, asymmetric tapers included, no linearization.
    Every d/dP channel is then "a physical corner moving on the panel":

        dL/dju_k -> chain through corner k's own (u_h, w):  dP_0j += g_k X_k[j],
        dP_2j += -(u_h/w)_k g_k X_k[j]      (X_k = that corner's homogeneous coords)

  * cell weight = trapezoid CDF difference; the four CDF partials are single-formula and
    BOUNDED (r, q = clamped taper coordinates):
        dG/du1 = -r + r^2/2   dG/du2 = -r^2/2   dG/du3 = q - q^2/2   dG/du4 = q^2/2
  * v-footprint = the two projections of z +- dz/2 (exact; SF-TR treatment of the axial axis,
    like LEAP and the sibling: the transaxial-induced v-spread is <= ~0.5% at our cone angles).
  * amplitude = dx dy dz / (mean_width_u * width_v), both in PHYSICAL mm at the voxel
    (detector-mm corner spans scaled by w/SDD; SDD per view = |P row0's 3x3 part|, invariant
    under rigid motion). Bounded below by the projected voxel extents -- no min(), no 1/h.
    Sanity anchors: axis-aligned view -> peak = the crossing side; 45 deg -> peak = the
    diagonal, exactly (the trapezoid degenerates to the Radon triangle of the square).

  * the pair is matched BY CONSTRUCTION: forward scatters val*peak*Wu*Wv into the per-view
    1.4 MB sino slice (L2-resident scatter), the transpose gathers with the identical code.

The dP -> dtheta chain stays outside: the autograd Function returns dL/dP and torch
differentiates params_to_Pmot on top. Reference implementation + 12-entry autograd parity rig:
scripts/dev_sf_ref_autograd.py. Descent repro: scripts/dev_sf_est_repro.py.
Gate: scripts/gate_sf_projector.py.

STATUS (2026-07-30): FULLY RETIRED FROM PRODUCTION. The operator pair is LEAP's
(`fm3d/leap_projector.py`) and its modular forward is now PINNED TO JOSEPH, so there is no SF
branch left for `sf_grad_P` to cover either -- the deployed geometry gradient is
`triton_leap_grad.leap_grad_P` everywhere. This module survives ONLY as the gates' independent
SF-class implementation (`gate_sf_projector.py`, `gate_leap_projector.py` T1, and the
`diag_leap_crosscheck` rig). Nothing in production imports it -- the `GRAD_MODE = "sf"` A/B
arm that used to reach `sf_grad_P` was deleted with the rest of the operator switches on
2026-08-04. It was briefly load-bearing: LEAP's SF kernel projects ROUNDED voxel centres, and this
module's continuous-corner gradient was the only one following the loss TREND rather than the
lattice ripple (ledger in `triton_leap_grad`'s docstring). Pinning the kernel removed the need.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                             # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _voxel_world(p, W, H, D, dx, dy, dz):
        """Voxel linear index -> world mm (centred grid, (z,y,x) row-major)."""
        iz = p // (H * W)
        rem = p - iz * (H * W)
        iy = rem // W
        ix = rem - iy * W
        x = (ix.to(tl.float32) - (W - 1) * 0.5) * dx
        y = (iy.to(tl.float32) - (H - 1) * 0.5) * dy
        z = (iz.to(tl.float32) - (D - 1) * 0.5) * dz
        return x, y, z

    @triton.jit
    def _sf_kernel(f_ptr, g_ptr, P_ptr, sdd_ptr, dP_ptr,
                   V, nv, nu, Npix, W, H, D,
                   dx, dy, dz, du, dv, u0, v_off, eps,
                   BLOCK: tl.constexpr, MODE: tl.constexpr,
                   FP: tl.constexpr, FPV: tl.constexpr):
        """MODE 0: forward | MODE 1: exact transpose | MODE 2: d<g, SF(f)>/dP."""
        pid_b = tl.program_id(1)
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = p < Npix
        x, y, z = _voxel_world(p, W, H, D, dx, dy, dz)
        val = tl.load(f_ptr + pid_b.to(tl.int64) * Npix + p, mask=mask, other=0.0)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        # SF is defined for the CUBE (piecewise-constant) voxel basis, and that is what these
        # half-extents encode. The retired ray-march/gridsample operator used the trilinear-TENT
        # basis instead, which is smoother and is why its FDK images had no voxel-grid texture.
        # Widening these to the tent's SUPPORT was tried and REFUTED (2026-07-29): it is the
        # tent's piecewise-CUBIC shape that matters, not its support. See the sf-projector memory;
        # the texture is handled at the FDK's ramp window instead (`projector_3d`).
        hx = 0.5 * dx
        hy = 0.5 * dy
        hz = 0.5 * dz
        off_u = (nu - 1) * 0.5
        off_v = (nv - 1) * 0.5

        for view in range(0, V):
            pb = P_ptr + (pid_b * V + view) * 12
            p00 = tl.load(pb + 0)
            p01 = tl.load(pb + 1)
            p02 = tl.load(pb + 2)
            p03 = tl.load(pb + 3)
            p10 = tl.load(pb + 4)
            p11 = tl.load(pb + 5)
            p12 = tl.load(pb + 6)
            p13 = tl.load(pb + 7)
            p20 = tl.load(pb + 8)
            p21 = tl.load(pb + 9)
            p22 = tl.load(pb + 10)
            p23 = tl.load(pb + 11)
            sdd = tl.load(sdd_ptr + pid_b * V + view)

            # ---- four transaxial corners, projected exactly. Payload (uh, w, cx, cy)
            # travels with each ju through the sorting network so the gradient can chain
            # back to the corner that actually sits at each sorted slot.
            zc = z
            base_u = p02 * zc + p03
            base_w = p22 * zc + p23
            xa = x - hx
            xb = x + hx
            ya = y - hy
            yb = y + hy
            u1h = p00 * xa + p01 * ya + base_u
            w1 = p20 * xa + p21 * ya + base_w
            u2h = p00 * xa + p01 * yb + base_u
            w2 = p20 * xa + p21 * yb + base_w
            u3h = p00 * xb + p01 * ya + base_u
            w3 = p20 * xb + p21 * ya + base_w
            u4h = p00 * xb + p01 * yb + base_u
            w4 = p20 * xb + p21 * yb + base_w
            ok_w = (w1 > eps) & (w2 > eps) & (w3 > eps) & (w4 > eps)
            w1s = tl.where(w1 > eps, w1, 1.0)
            w2s = tl.where(w2 > eps, w2, 1.0)
            w3s = tl.where(w3 > eps, w3, 1.0)
            w4s = tl.where(w4 > eps, w4, 1.0)
            j1 = (u1h / w1s - u0) / du + off_u
            j2 = (u2h / w2s - u0) / du + off_u
            j3 = (u3h / w3s - u0) / du + off_u
            j4 = (u4h / w4s - u0) / du + off_u
            c1x = xa
            c1y = ya
            c2x = xa
            c2y = yb
            c3x = xb
            c3y = ya
            c4x = xb
            c4y = yb
            # sorting network (1,2)(3,4)(1,3)(2,4)(2,3), payload swapped alongside
            sw = j1 > j2
            j1, j2 = tl.where(sw, j2, j1), tl.where(sw, j1, j2)
            u1h, u2h = tl.where(sw, u2h, u1h), tl.where(sw, u1h, u2h)
            w1s, w2s = tl.where(sw, w2s, w1s), tl.where(sw, w1s, w2s)
            c1x, c2x = tl.where(sw, c2x, c1x), tl.where(sw, c1x, c2x)
            c1y, c2y = tl.where(sw, c2y, c1y), tl.where(sw, c1y, c2y)
            sw = j3 > j4
            j3, j4 = tl.where(sw, j4, j3), tl.where(sw, j3, j4)
            u3h, u4h = tl.where(sw, u4h, u3h), tl.where(sw, u3h, u4h)
            w3s, w4s = tl.where(sw, w4s, w3s), tl.where(sw, w3s, w4s)
            c3x, c4x = tl.where(sw, c4x, c3x), tl.where(sw, c3x, c4x)
            c3y, c4y = tl.where(sw, c4y, c3y), tl.where(sw, c3y, c4y)
            sw = j1 > j3
            j1, j3 = tl.where(sw, j3, j1), tl.where(sw, j1, j3)
            u1h, u3h = tl.where(sw, u3h, u1h), tl.where(sw, u1h, u3h)
            w1s, w3s = tl.where(sw, w3s, w1s), tl.where(sw, w1s, w3s)
            c1x, c3x = tl.where(sw, c3x, c1x), tl.where(sw, c1x, c3x)
            c1y, c3y = tl.where(sw, c3y, c1y), tl.where(sw, c1y, c3y)
            sw = j2 > j4
            j2, j4 = tl.where(sw, j4, j2), tl.where(sw, j2, j4)
            u2h, u4h = tl.where(sw, u4h, u2h), tl.where(sw, u2h, u4h)
            w2s, w4s = tl.where(sw, w4s, w2s), tl.where(sw, w2s, w4s)
            c2x, c4x = tl.where(sw, c4x, c2x), tl.where(sw, c2x, c4x)
            c2y, c4y = tl.where(sw, c4y, c2y), tl.where(sw, c2y, c4y)
            sw = j2 > j3
            j2, j3 = tl.where(sw, j3, j2), tl.where(sw, j2, j3)
            u2h, u3h = tl.where(sw, u3h, u2h), tl.where(sw, u2h, u3h)
            w2s, w3s = tl.where(sw, w3s, w2s), tl.where(sw, w2s, w3s)
            c2x, c3x = tl.where(sw, c3x, c2x), tl.where(sw, c2x, c3x)
            c2y, c3y = tl.where(sw, c3y, c2y), tl.where(sw, c2y, c3y)

            # ---- the two axial corners
            vah = p10 * x + p11 * y + p12 * (z - hz) + p13
            wva = p20 * x + p21 * y + p22 * (z - hz) + p23
            vbh = p10 * x + p11 * y + p12 * (z + hz) + p13
            wvb = p20 * x + p21 * y + p22 * (z + hz) + p23
            ok_w = ok_w & (wva > eps) & (wvb > eps)
            wvas = tl.where(wva > eps, wva, 1.0)
            wvbs = tl.where(wvb > eps, wvb, 1.0)
            ma = (vah / wvas - v_off) / dv + off_v
            mb = (vbh / wvbs - v_off) / dv + off_v
            zea = z - hz
            zeb = z + hz
            sw = ma > mb
            ma, mb = tl.where(sw, mb, ma), tl.where(sw, ma, mb)
            vah, vbh = tl.where(sw, vbh, vah), tl.where(sw, vah, vbh)
            wvas, wvbs = tl.where(sw, wvbs, wvas), tl.where(sw, wvas, wvbs)
            zea, zeb = tl.where(sw, zeb, zea), tl.where(sw, zea, zeb)

            # ---- amplitude: physical mean width x physical axial width, at the voxel
            wc = p20 * x + p21 * y + p22 * z + p23          # centre depth (linear => exact)
            wc_s = tl.where(wc > eps, wc, 1.0)
            mw_e = 0.5 * ((j3 + j4) - (j1 + j2))            # mean u-width [elements]
            wv_e = mb - ma                                  # v-width [elements]
            mwp = tl.maximum(mw_e * du * wc_s / sdd, 0.05 * tl.minimum(dx, dy))
            wvp = tl.maximum(wv_e * dv * wc_s / sdd, 0.05 * dz)
            peak = dx * dy * dz / (mwp * wvp)
            amp = tl.where(mask & ok_w, val * peak, 0.0)

            d1 = tl.maximum(j2 - j1, 1e-6)                  # taper widths [elements]
            d2 = tl.maximum(j4 - j3, 1e-6)
            n0 = tl.floor(j1 + 0.5).to(tl.int32)
            m0 = tl.floor(ma + 0.5).to(tl.int32)
            base = ((pid_b * V + view) * nv).to(tl.int64) * nu

            s1 = tl.zeros((BLOCK,), dtype=tl.float32)       # MODE 2: dL/dju_k sums
            s2 = tl.zeros((BLOCK,), dtype=tl.float32)
            s3 = tl.zeros((BLOCK,), dtype=tl.float32)
            s4 = tl.zeros((BLOCK,), dtype=tl.float32)
            sva = tl.zeros((BLOCK,), dtype=tl.float32)      # dL/d(ma), dL/d(mb)
            svb = tl.zeros((BLOCK,), dtype=tl.float32)
            sm = tl.zeros((BLOCK,), dtype=tl.float32)       # mass (peak channel)
            for cm in tl.static_range(FPV):
                m = m0 + cm
                mf = m.to(tl.float32)
                lo = tl.maximum(ma, mf - 0.5)
                hi = tl.minimum(mb, mf + 0.5)
                Vm = tl.maximum(hi - lo, 0.0)
                ok_m = (m >= 0) & (m < nv)
                # d(Vm)/d(ma) = -1 iff ma is the binding lower edge, etc.
                dVa = -tl.where((ma > mf - 0.5) & (ma < mf + 0.5) & (Vm > 0.0), 1.0, 0.0)
                dVb = tl.where((mb > mf - 0.5) & (mb < mf + 0.5) & (Vm > 0.0), 1.0, 0.0)
                for cn in tl.static_range(FP):
                    n = n0 + cn
                    nf = n.to(tl.float32)
                    ok = mask & ok_w & ok_m & (n >= 0) & (n < nu)
                    off = base + m.to(tl.int64) * nu + n
                    # trapezoid CDF at both cell edges (t relative to nothing: absolute)
                    Wn = tl.zeros((BLOCK,), dtype=tl.float32)
                    g1 = tl.zeros((BLOCK,), dtype=tl.float32)
                    g2 = tl.zeros((BLOCK,), dtype=tl.float32)
                    g3 = tl.zeros((BLOCK,), dtype=tl.float32)
                    g4 = tl.zeros((BLOCK,), dtype=tl.float32)
                    for ee in tl.static_range(2):
                        t = nf - 0.5 + ee                    # cell edge
                        sgn = 2.0 * ee - 1.0                 # -1 for e1, +1 for e2
                        r = tl.minimum(tl.maximum((t - j1) / d1, 0.0), 1.0)
                        q = tl.minimum(tl.maximum((t - j3) / d2, 0.0), 1.0)
                        pl = tl.minimum(tl.maximum(t, j2), j3) - j2
                        G = 0.5 * d1 * r * r + pl + d2 * (q - 0.5 * q * q)
                        Wn += sgn * G
                        if MODE == 2:
                            g1 += sgn * (-r + 0.5 * r * r)
                            g2 += sgn * (-0.5 * r * r)
                            g3 += sgn * (q - 0.5 * q * q)
                            g4 += sgn * (0.5 * q * q)
                    if MODE == 0:
                        contrib = amp * Wn * Vm
                        tl.atomic_add(g_ptr + off, contrib, mask=ok & (contrib != 0.0),
                                      sem="relaxed")
                    if MODE == 1:
                        gval = tl.load(g_ptr + off, mask=ok, other=0.0)
                        acc += gval * peak * Wn * Vm
                    if MODE == 2:
                        gval = tl.load(g_ptr + off, mask=ok, other=0.0)
                        s1 += gval * g1 * Vm
                        s2 += gval * g2 * Vm
                        s3 += gval * g3 * Vm
                        s4 += gval * g4 * Vm
                        sva += gval * Wn * dVa
                        svb += gval * Wn * dVb
                        sm += gval * Wn * Vm
            if MODE == 2:
                z_ok = tl.where(mask & ok_w, val, 0.0)
                # peak channel: peak = dxdydz/(mwp*wvp); mwp ~ mw_e, wvp ~ wv_e, both ~ wc.
                # Gate at the width floors (a.e.): where floored, the width derivative is 0.
                mw_on = tl.where(mwp > 0.05 * tl.minimum(dx, dy), 1.0, 0.0)
                wv_on = tl.where(wvp > 0.05 * dz, 1.0, 0.0)
                dpk_mw = -peak / tl.maximum(mw_e, 1e-6) * mw_on   # d(peak)/d(mw_e)
                dpk_wv = -peak / tl.maximum(wv_e, 1e-6) * wv_on
                # dL/dju_k: footprint-shape channel (peak * s_k) + peak-width channel
                L1 = z_ok * (peak * s1 + sm * dpk_mw * (-0.5))
                L2 = z_ok * (peak * s2 + sm * dpk_mw * (-0.5))
                L3 = z_ok * (peak * s3 + sm * dpk_mw * 0.5)
                L4 = z_ok * (peak * s4 + sm * dpk_mw * 0.5)
                La = z_ok * (peak * sva + sm * dpk_wv * (-1.0))
                Lb = z_ok * (peak * svb + sm * dpk_wv * 1.0)
                # wc channel of peak: peak ~ 1/wc^2 (both widths carry wc/sdd)
                Lw = z_ok * sm * (-2.0 * peak / wc_s) * mw_on * wv_on
                dpb = dP_ptr + (pid_b * V + view) * 12
                # u-corners -> rows 0 and 2 (per-corner projective chain)
                a1 = L1 / (du * w1s)
                a2 = L2 / (du * w2s)
                a3 = L3 / (du * w3s)
                a4 = L4 / (du * w4s)
                b1 = -a1 * u1h / w1s
                b2 = -a2 * u2h / w2s
                b3 = -a3 * u3h / w3s
                b4 = -a4 * u4h / w4s
                tl.atomic_add(dpb + 0, tl.sum(a1 * c1x + a2 * c2x + a3 * c3x + a4 * c4x)
                              .to(tl.float64), sem="relaxed")
                tl.atomic_add(dpb + 1, tl.sum(a1 * c1y + a2 * c2y + a3 * c3y + a4 * c4y)
                              .to(tl.float64), sem="relaxed")
                tl.atomic_add(dpb + 2, tl.sum((a1 + a2 + a3 + a4) * z).to(tl.float64),
                              sem="relaxed")
                tl.atomic_add(dpb + 3, tl.sum(a1 + a2 + a3 + a4).to(tl.float64),
                              sem="relaxed")
                # v-corners -> rows 1 and 2
                aa = La / (dv * wvas)
                ab = Lb / (dv * wvbs)
                ba = -aa * vah / wvas
                bb = -ab * vbh / wvbs
                tl.atomic_add(dpb + 4, tl.sum((aa + ab) * x).to(tl.float64), sem="relaxed")
                tl.atomic_add(dpb + 5, tl.sum((aa + ab) * y).to(tl.float64), sem="relaxed")
                tl.atomic_add(dpb + 6, tl.sum(aa * zea + ab * zeb).to(tl.float64),
                              sem="relaxed")
                tl.atomic_add(dpb + 7, tl.sum(aa + ab).to(tl.float64), sem="relaxed")
                # row 2: u-corner terms + v-corner terms + the wc (peak) channel
                r2x = b1 * c1x + b2 * c2x + b3 * c3x + b4 * c4x + (ba + bb) * x + Lw * x
                r2y = b1 * c1y + b2 * c2y + b3 * c3y + b4 * c4y + (ba + bb) * y + Lw * y
                r2z = (b1 + b2 + b3 + b4) * z + ba * zea + bb * zeb + Lw * z
                r2c = b1 + b2 + b3 + b4 + ba + bb + Lw
                tl.atomic_add(dpb + 8, tl.sum(r2x).to(tl.float64), sem="relaxed")
                tl.atomic_add(dpb + 9, tl.sum(r2y).to(tl.float64), sem="relaxed")
                tl.atomic_add(dpb + 10, tl.sum(r2z).to(tl.float64), sem="relaxed")
                tl.atomic_add(dpb + 11, tl.sum(r2c).to(tl.float64), sem="relaxed")

        if MODE == 1:
            tl.store(f_ptr + pid_b.to(tl.int64) * Npix + p, acc, mask=mask)


def _sdd_of(P):
    """(B,V) per-view SDD = |P row0's 3x3 part| -- invariant under rigid T(theta)."""
    return P[..., 0, :3].norm(dim=-1).contiguous().to(torch.float32)


_WINDOW_CACHE: dict = {}


def _footprint_window(P, D, H, W, dx, dy, dz, du, dv):
    """(FP, FPV): detector cells the trapezoid footprint needs. DERIVED, not a magic constant.

    THIS IS THE PARAMETER THAT BIT US ONCE AND WAS STILL WRONG. A voxel of physical extent
    (dx, dy, dz) at depth w projects to a footprint of

        u-width = sqrt(dx^2 + dy^2) * M / du      (the sqrt(2)*dx worst case is 45 deg)
        v-width = dz * M / dv                     with M = SDD / w

    and the kernel scans cells `n0 .. n0+FP-1` where `n0 = floor(j1 + 0.5)`, i.e. the window
    covers `[n0 - 0.5, n0 + FP - 0.5]` while `n0` may sit as low as `j1 - 0.5`. So the window
    starts as early as `j1 - 1`, and holding a footprint of width `wu` needs

        FP >= wu + 1   ->   FP = ceil(wu) + 1     (tight; verified against a 14/12 reference)

    M is largest for the voxel NEAREST THE SOURCE, so the bound is evaluated at the minimum
    depth over the volume's eight corners and all views -- which is why it cannot be a constant:
    it moves with the voxel size, the detector pitch, det_bin, and the geometry. The old
    hardcoded `fp=6, fpv=4` was right in u and WRONG IN V for our own deployed configuration
    (256^3 @ 1 mm, det_bin 1: v-width 3.10 cells needs FPV 5, and FPV 4 truncated the footprints
    of the near-source voxels by up to 5.6e-4 relative -- a one-sided loss of mass, since
    truncation only ever removes it). Measured cost of the fix: +8%.

    FP is rounded UP TO EVEN: it is the innermost `static_range`, and an odd count measured ~3.5x
    slower (91.8 ms at 7/6 against 30.5 ms at 8/6). FPV is the outer loop and pays no such
    penalty, so it is left tight. Results are cached per (shape, spacing, geometry) key: FP/FPV
    are `tl.constexpr`, so a value that wobbled per call would thrash Triton's compile cache.
    """
    # THE KEY MUST NOT DEPEND ON P's VALUES. A first version keyed on their sum: under motion
    # every call is a fresh key, so the cache filled with junk (7 entries in a 5-call benchmark),
    # every call paid a float64 matmul plus a HOST SYNC (~15 ms measured), and FP/FPV -- which are
    # `tl.constexpr` -- could wobble and re-trigger Triton's JIT. Keyed on the geometry only, the
    # first call fills it and every later call is a dict hit. Motion is absorbed by MOTION_MARGIN
    # instead: theta shifts the corner depths by at most its translation amplitude, and 3% of our
    # SOD is ~24 mm, comfortably more than any amplitude we simulate.
    MOTION_MARGIN = 0.03
    key = (D, H, W, round(dx, 6), round(dy, 6), round(dz, 6), round(du, 6), round(dv, 6))
    hit = _WINDOW_CACHE.get(key)
    if hit is not None:
        return hit
    with torch.no_grad():
        Pv = P.reshape(-1, 3, 4).to(torch.float64)
        sx, sy, sz = 0.5 * W * dx, 0.5 * H * dy, 0.5 * D * dz     # origin at the volume centre
        c = torch.tensor([[a * sx, b * sy, d * sz, 1.0]
                          for a in (-1., 1.) for b in (-1., 1.) for d in (-1., 1.)],
                         device=P.device, dtype=torch.float64)     # (8,4)
        w = (Pv[:, 2, :] @ c.T)                                    # (VB,8) corner depths
        sdd = Pv[:, 0, :3].norm(dim=-1)[:, None]
        wmin = float(w.min())
        Mmax = float(sdd.max()) / max(wmin * (1.0 - MOTION_MARGIN), 1e-6)
    wu = math.hypot(dx, dy) * Mmax / du
    wv = dz * Mmax / dv
    fp = int(math.ceil(wu)) + 1
    fp += fp & 1                                                   # even: odd FP is ~3.5x slower
    fpv = int(math.ceil(wv)) + 1
    out = (max(4, fp), max(3, fpv))
    _WINDOW_CACHE[key] = out
    return out


def _launch(mode, f, g, P, dP, nv, nu, D, H, W, dx, dy, dz, du, dv, u0, v_off,
            eps=1e-8, block=128, fp=None, fpv=None):
    B = P.shape[0]
    V = P.shape[1]
    Npix = D * H * W
    if fp is None or fpv is None:
        afp, afpv = _footprint_window(P, D, H, W, dx, dy, dz, du, dv)
        fp = afp if fp is None else fp
        fpv = afpv if fpv is None else fpv
    Pc = P.reshape(B, V, 12).contiguous().to(torch.float32)
    sdd = _sdd_of(P.view(B, V, 3, 4))
    z1 = torch.zeros(1, device=P.device, dtype=torch.float64)
    grid = (triton.cdiv(Npix, block), B)
    _sf_kernel[grid](f, g, Pc, sdd, dP if dP is not None else z1,
                     V, nv, nu, Npix, W, H, D,
                     float(dx), float(dy), float(dz), float(du), float(dv),
                     float(u0), float(v_off), float(eps),
                     BLOCK=block, MODE=mode, FP=fp, FPV=fpv)


def sf_project(vol, P, *, nv, nu, dx, dy, dz, du, dv, u0=0.0, v_off=0.0):
    """SF forward: vol (B,D,H,W) fp32 -> sinogram (B,V,nv,nu). No autograd (see SFProject)."""
    B, D, H, W = vol.shape
    V = P.shape[1]
    g = torch.zeros((B, V, nv, nu), device=vol.device, dtype=torch.float32)
    _launch(0, vol.contiguous(), g, P, None, nv, nu, D, H, W, dx, dy, dz, du, dv, u0, v_off)
    return g


def sf_backproject(g, P, *, D, H, W, dx, dy, dz, du, dv, u0=0.0, v_off=0.0):
    """Exact transpose of `sf_project` (same weights, gather). g (B,V,nv,nu) -> (B,D,H,W)."""
    B = g.shape[0]
    f = torch.zeros((B, D, H, W), device=g.device, dtype=torch.float32)
    _launch(1, f, g.contiguous(), P, None, g.shape[2], g.shape[3], D, H, W,
            dx, dy, dz, du, dv, u0, v_off)
    return f


def sf_grad_P(vol, ghat, P, *, dx, dy, dz, du, dv, u0=0.0, v_off=0.0):
    """d <ghat, SF(vol; P)> / dP  -> (B,V,3,4). The geometry gradient (module docstring)."""
    B, D, H, W = vol.shape
    V = P.shape[1]
    dP = torch.zeros((B, V, 12), device=vol.device, dtype=torch.float64)
    _launch(2, vol.contiguous(), ghat.contiguous(), P, dP, ghat.shape[2], ghat.shape[3],
            D, H, W, dx, dy, dz, du, dv, u0, v_off)
    return dP.view(B, V, 3, 4).to(torch.float32)


class SFProject(torch.autograd.Function):
    """Differentiable SF forward: (vol, P) -> sinogram. grad_vol via the exact transpose,
    grad_P via the corner-parameterized analytic gradient. Either input may require grad."""

    @staticmethod
    def forward(ctx, vol, P, nv, nu, dx, dy, dz, du, dv, u0, v_off):
        g = sf_project(vol, P, nv=nv, nu=nu, dx=dx, dy=dy, dz=dz, du=du, dv=dv,
                       u0=u0, v_off=v_off)
        ctx.save_for_backward(vol, P)
        ctx.meta = (dx, dy, dz, du, dv, u0, v_off)
        return g

    @staticmethod
    def backward(ctx, gout):
        vol, P = ctx.saved_tensors
        dx, dy, dz, du, dv, u0, v_off = ctx.meta
        B, D, H, W = vol.shape
        gout = gout.contiguous()
        gvol = gP = None
        if ctx.needs_input_grad[0]:
            gvol = sf_backproject(gout, P, D=D, H=H, W=W, dx=dx, dy=dy, dz=dz,
                                  du=du, dv=dv, u0=u0, v_off=v_off)
        if ctx.needs_input_grad[1]:
            gP = sf_grad_P(vol, gout, P, dx=dx, dy=dy, dz=dz, du=du, dv=dv,
                           u0=u0, v_off=v_off)
        return gvol, gP, None, None, None, None, None, None, None, None, None


def sf_project_3d_batched(volumes, Pmat, u_coords, v_coords, *, dx, dy, dz):
    """Drop-in-shaped SF forward: (B,1,D,H,W), (B,V,3,4) -> (B,V,nv,nu), differentiable in
    BOTH the volume and Pmat. Mirrors forward_project_3d_batched's conventions."""
    B, _, D, H, W = volumes.shape
    nu, nv = len(u_coords), len(v_coords)
    du = float(u_coords[1] - u_coords[0])
    dv = float(v_coords[1] - v_coords[0])
    u0 = float(u_coords[0] + u_coords[-1]) * 0.5
    v_off = float(v_coords[0] + v_coords[-1]) * 0.5
    return SFProject.apply(volumes[:, 0].to(torch.float32), Pmat.to(torch.float32),
                           nv, nu, dx, dy, dz, du, dv, u0, v_off)
