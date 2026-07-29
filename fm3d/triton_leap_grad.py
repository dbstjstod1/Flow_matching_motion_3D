"""The geometry gradient OF LEAP'S OWN FORWARD MODELS, stacked on LEAP-form parameters
(sourcePositions, moduleCenters, rowVectors, colVectors), 2026-07-29.

WHY THIS EXISTS. Since `fm3d/leap_projector.py` the deployed forward is LEAP's modular-beam
projector, but LEAP ships no geometry derivative, so `LEAPProject.backward` borrowed
`triton_sf.sf_grad_P` -- the exact gradient of the RETIRED SF forward, a different operator
(2.9e-3 relative). This module removes the borrow: it differentiates the maths of LEAP's own
CUDA kernels, transcribed line-for-line from `refs/LEAP/src/projectors_Joseph.cu` (MIT), so
the estimator descends on the exact surface of the loss it evaluates.

TWO KERNELS, BECAUSE LEAP RUNS TWO. `project_Joseph_modular` picks its kernel per GEOMETRY SET
(the launcher, ibid. line ~2255):

    modularBeamProjectorKernel_SF      if modularbeamIsAxiallyAligned() && useSF
    modularBeamJosephProjectorKernel   otherwise

and `modularbeamIsAxiallyAligned()` is TRUE only while EVERY view's (unit) rowVector keeps
z >= 0.9961 -- a 5.06-degree panel tilt -- and the source z-span stays under half the panel
height (`parameters::set_sourcesAndModules`). Our nominal orbit has rowv_z = +1 exactly, but a
single view whose motion tilts the panel past ~5 degrees flips the WHOLE geometry to the
Joseph kernel, silently. At our eval amplitude (10 deg p2p nodes + Akima overshoot) both
branches are live production regimes, so both are differentiated here and
`leap_projector.kernel_kind` replicates the launcher's selection.

THE TWO MODELS (both: volume texture with BORDER addressing = zero outside, `loadTexture(...,
useExtrapolation=false, linear=true)`; sinogram (V, nv, nu); centred detector/volume grids):

  SF      detector-driven; for each pixel, march the dominant IN-PLANE axis one voxel-plane at
          a time; at each plane take the 3x3 (cross-axis x z) neighbours of the ray point,
          weighted by rect-footprint/pixel overlaps built from clamped min/max of the projected
          centre `iu_c, iv_c` and half-widths `hfw = 0.5*T*t*|u_flat|/du, vfw = 0.5*dz*t/dv`
          (t = magnification D/dist); scale by the separable length
          `T * sqrt((ra^2+rb^2)(ra^2+rz^2)) / ra^2`.
  JOSEPH  detector-driven ray march: bilinear sample at every dominant-axis voxel plane,
          entry-plane sample re-weighted by -0.5, length `T * |r| / |r_a|`.

THE GRADIENT. Reverse-mode by hand, accumulated per view into d/d(src, mod, rowv, colv) --
12 scalars per view, block-reduced then fp64 atomics. Piecewise-constant quantities (voxel
indices, min/max branch choices, the entry plane) get their LITERAL a.e. derivative.

THE RIPPLE LEDGER (2026-07-30) -- why the SF-branch gradient here is NOT the deployed
default, and must not silently become it. LEAP's SF kernel projects ROUNDED voxel centres
(x_c, z_c) into `iu_c`/`iv_c`, which superimposes a lattice-period RIPPLE on the loss
surface. Measured (fp64 FD of the transcribed model, gate_geometry G4a probe (view 2, rz),
theta = 0):

    eps <= 3e-4 rad:  FD -> +4.1e-5   == this kernel's exact gradient (it IS the local slope)
    eps >= 1e-3 rad:  FD -> -3.4e-4   == the TREND slope == the retired continuous-corner
                                         surrogate (`triton_sf.sf_grad_P`, reads -3.3e-4)

Opposite SIGNS. The estimator's convergence frontier (~0.1 deg = 1.7e-3 rad) sits at the
ripple scale, so descending the exact gradient means chasing ripple minima exactly where the
final accuracy is decided (observed: the exact-gradient smoke plateaued at obs 0.77 mm vs the
surrogate's 0.48 mm). Two failed alternatives, so nobody retries them: straight-through
rounded centres (d(x_c) := d(x_ray)) ZEROES the whole footprint-position channel, because the
continuous centre is a point on the pixel's own ray and its projection is the pixel itself;
and jittering the eval point does not help (the literal gradient is biased, not oscillating:
jittered mean +3.4e-5, std 3.3e-5). The JOSEPH kernel is bilinear in CONTINUOUS coordinates
-- no rounding, no ripple -- so ITS exact gradient is the trend (FD parity 4e-4, 100x tighter
than any SF-branch check). Hence `leap_projector.GRAD_MODE = "auto"`: JOSEPH branch -> this
module, SF branch -> the surrogate.

The P chain happens OUTSIDE, in torch: `modular_arrays_torch` is the differentiable
P -> (src, mod, rowv, colv) decomposition and `leap_grad_P` runs the vjp through it, so
`LEAPProject.backward` still returns dL/dP and `params_to_Pmot` still closes the theta chain
on top.

EXACTNESS LEDGER (what is NOT replicated, all measured to be below the gate bars):
  * tex3D linear interpolation quantizes lerp fractions to 9 bits; we compute them in fp32.
  * LEAP normalizes rowv/colv and re-orthogonalizes rowv against colv on ingest; our arrays
    are orthonormal by construction (rigid P), and on the rigid manifold those maps have
    identity Jacobian (|e_v| == 1 and e_u.e_v == 0 hold for ALL theta, so their derivatives
    stay tangent) -- skipping them is exact for the theta chain, approximate only for raw
    non-rigid dP probes.
  * Joseph's per-pixel prologue is double precision in LEAP; fp32 here (~7e-5 mm in r).
  * ties |r_x| == |r_y| (measure zero) and the z-dominant Joseph branch (cone angle < 45 deg
    keeps rays in-plane dominant) are not special-cased.
  * `rFOV` is not applied: `leap_projector` forces `set_diameterFOV(1e5)` precisely so the
    mask never bites.

Requires W == H and dx == dy (both deployed grids: 256^3 @ 1 mm, 612^3 @ 0.4187 mm); the
Joseph path additionally assumes dz == dx (LEAP's own `lineIntegral_Joseph_ZYX` states
"assumes T.x == T.y == T.z").

Self-checks: `leap_forward_model` re-implements the VALUE of both kernels so the gate can pin
the transcription against `leap_project` itself (`scripts/gate_leap_projector.py`), and the
theta gradient is finite-differenced through the actual LEAP loss in BOTH branch regimes.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                             # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _leap_sf_kernel(vol_ptr, gout_ptr, out_ptr,
                        src_ptr, mod_ptr, rowv_ptr, colv_ptr,
                        nv, nu, NP, D, HW,
                        dx, dz, du, dv, u0g, v0g, b0, z0,
                        BLOCK: tl.constexpr, MODE: tl.constexpr):
        """modularBeamProjectorKernel_SF, unified over the two in-plane branches.

        MODE 0: value -> out_ptr = sinogram (V, nv, nu) fp32, per-pixel stores.
        MODE 2: d<gout, g>/d(geometry) -> out_ptr = (V, 12) fp64 [src, mod, rowv, colv].
        One program = BLOCK detector pixels of one view. `NP` = W == H, `HW` = H*W.
        """
        view = tl.program_id(1)
        pix = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = pix < nv * nu
        m = pix // nu
        n = pix - m * nu

        px = tl.load(src_ptr + view * 3 + 0)
        py = tl.load(src_ptr + view * 3 + 1)
        pz = tl.load(src_ptr + view * 3 + 2)
        cx = tl.load(mod_ptr + view * 3 + 0)
        cy = tl.load(mod_ptr + view * 3 + 1)
        cz_ = tl.load(mod_ptr + view * 3 + 2)
        vx = tl.load(rowv_ptr + view * 3 + 0)   # v_vec = rowVectors (LEAP naming)
        vy = tl.load(rowv_ptr + view * 3 + 1)
        vz = tl.load(rowv_ptr + view * 3 + 2)
        ux = tl.load(colv_ptr + view * 3 + 0)   # u_vec = colVectors
        uy = tl.load(colv_ptr + view * 3 + 1)
        uz = tl.load(colv_ptr + view * 3 + 2)

        t_r = m.to(tl.float32) * dv + v0g       # row coordinate (LEAP `t`)
        s = n.to(tl.float32) * du + u0g         # column coordinate (LEAP `s`)

        detx = cx + ux * s + vx * t_r
        dety = cy + uy * s + vy * t_r
        detz = cz_ + uz * s + vz * t_r
        rx = detx - px
        ry = dety - py
        rz = detz - pz
        Dd = tl.sqrt(rx * rx + ry * ry + rz * rz)

        pmcx = px - cx
        pmcy = py - cy
        pmcz = pz - cz_
        pmc_u = pmcx * ux + pmcy * uy + pmcz * uz
        pmc_v = pmcx * vx + pmcy * vy + pmcz * vz

        rho = tl.sqrt(ux * ux + uy * uy)
        ufx = ux / rho
        ufy = uy / rho

        ydom = tl.abs(ry) > tl.abs(rx)
        r_a = tl.where(ydom, ry, rx)
        r_b = tl.where(ydom, rx, ry)
        p_a = tl.where(ydom, py, px)
        p_b = tl.where(ydom, px, py)
        uf_sel = tl.where(ydom, ufx, ufy)
        kh = 0.5 * dx * tl.abs(uf_sel) / du     # footprint half-width per unit magnification
        kv = 0.5 * dz / dv

        alpha = r_a * r_a + r_b * r_b
        beta = r_a * r_a + rz * rz
        sqab = tl.sqrt(alpha * beta)
        ra2 = r_a * r_a
        L = dx * sqab / ra2

        m_pos = m.to(tl.float32) + 0.5
        m_neg = m.to(tl.float32) - 0.5
        n_pos = n.to(tl.float32) + 0.5
        n_neg = n.to(tl.float32) - 0.5

        gout = tl.zeros((BLOCK,), dtype=tl.float32)
        if MODE == 2:
            gout = tl.load(gout_ptr + (view * nv * nu).to(tl.int64) + pix, mask=mask,
                           other=0.0)
        ghat = gout * L                          # dL/d(pre-length accumulator), per pixel

        G_acc = tl.zeros((BLOCK,), dtype=tl.float32)
        bp_x = tl.zeros((BLOCK,), dtype=tl.float32)   # adjoint accumulators: source
        bp_y = tl.zeros((BLOCK,), dtype=tl.float32)
        bp_z = tl.zeros((BLOCK,), dtype=tl.float32)
        bc_x = tl.zeros((BLOCK,), dtype=tl.float32)   # module centre
        bc_y = tl.zeros((BLOCK,), dtype=tl.float32)
        bc_z = tl.zeros((BLOCK,), dtype=tl.float32)
        bu_x = tl.zeros((BLOCK,), dtype=tl.float32)   # colVector (u)
        bu_y = tl.zeros((BLOCK,), dtype=tl.float32)
        bu_z = tl.zeros((BLOCK,), dtype=tl.float32)
        bv_x = tl.zeros((BLOCK,), dtype=tl.float32)   # rowVector (v)
        bv_y = tl.zeros((BLOCK,), dtype=tl.float32)
        bv_z = tl.zeros((BLOCK,), dtype=tl.float32)
        br_a = tl.zeros((BLOCK,), dtype=tl.float32)   # ray vector, axis-mapped
        br_b = tl.zeros((BLOCK,), dtype=tl.float32)
        br_z = tl.zeros((BLOCK,), dtype=tl.float32)
        bp_a = tl.zeros((BLOCK,), dtype=tl.float32)   # source, axis-mapped extras
        bp_b = tl.zeros((BLOCK,), dtype=tl.float32)
        bD = tl.zeros((BLOCK,), dtype=tl.float32)     # |r| channel of t = D/dist
        bufs = tl.zeros((BLOCK,), dtype=tl.float32)   # |u_flat| channel of kh

        inv_ra = 1.0 / r_a
        for j in range(0, NP):
            w_s = j * dx + b0
            dist_a = w_s - p_a
            bco = p_b + dist_a * r_b * inv_ra          # cross-axis world coordinate
            zpt = pz + dist_a * rz * inv_ra
            # LEAP: int(0.5f + (x - x0)/T) -- C truncation toward zero
            ib = (0.5 + (bco - b0) / dx).to(tl.int32)
            iz = (0.5 + (zpt - z0) / dz).to(tl.int32)
            b_c = ib.to(tl.float32) * dx + b0
            z_c = iz.to(tl.float32) * dz + z0
            qb = p_b - bco
            qz = pz - zpt
            dist = tl.sqrt(qb * qb + dist_a * dist_a + qz * qz)
            t_mag = Dd / dist
            dlx = tl.where(ydom, b_c - px, w_s - px)   # Delta: voxel-centre minus source
            dly = tl.where(ydom, w_s - py, b_c - py)
            dlz = z_c - pz
            U = dlx * ux + dly * uy + dlz * uz
            Vv = dlx * vx + dly * vy + dlz * vz
            iu_c = (pmc_u + t_mag * U - u0g) / du
            iv_c = (pmc_v + t_mag * Vv - v0g) / dv
            hfw = kh * t_mag
            vfw = kv * t_mag

            # -- footprint/pixel overlap weights, exactly LEAP's clamped expressions
            A1 = tl.minimum(n_pos, iu_c + hfw)
            B1 = tl.maximum(n_neg, iu_c - hfw)
            hW1 = tl.maximum(A1 - B1, 0.0)
            upos = uf_sel > 0.0
            A0 = tl.where(upos, tl.minimum(n_pos, iu_c - hfw),
                          tl.minimum(n_pos, iu_c + 2.0 * hfw))
            B0 = tl.where(upos, tl.maximum(n_neg, iu_c - 2.0 * hfw),
                          tl.maximum(n_neg, iu_c + hfw))
            hW0 = tl.maximum(A0 - B0, 0.0)
            hW2 = tl.maximum(1.0 - hW1 - hW0, 0.0)

            A1v = tl.minimum(m_pos, iv_c + vfw)
            B1v = tl.maximum(m_neg, iv_c - vfw)
            vW1 = tl.maximum(A1v - B1v, 0.0)
            A0v = tl.minimum(m_pos, iv_c - vfw)
            B0v = tl.maximum(m_neg, iv_c - 2.0 * vfw)
            vW0 = tl.maximum(A0v - B0v, 0.0)
            vW2 = tl.maximum(1.0 - vW1 - vW0, 0.0)

            # -- 3x3 taps (cross-axis a in {ib-1, ib, ib+1} x z in {iz-1, iz, iz+1}), border 0
            zrow = iz.to(tl.int64) * HW
            cross0 = (ib - 1).to(tl.int64)
            base_flat = tl.where(ydom, j * NP, j).to(tl.int64)   # + cross*(1 or NP) below
            cstep = tl.where(ydom, 1, NP).to(tl.int64)
            okz0 = (iz - 1 >= 0) & (iz - 1 < D)
            okz1 = (iz >= 0) & (iz < D)
            okz2 = (iz + 1 >= 0) & (iz + 1 < D)
            okc0 = (ib - 1 >= 0) & (ib - 1 < NP)
            okc1 = (ib >= 0) & (ib < NP)
            okc2 = (ib + 1 >= 0) & (ib + 1 < NP)
            fl00 = (zrow - HW) + base_flat + cross0 * cstep
            f00 = tl.load(vol_ptr + fl00, mask=mask & okz0 & okc0, other=0.0)
            f01 = tl.load(vol_ptr + fl00 + HW, mask=mask & okz1 & okc0, other=0.0)
            f02 = tl.load(vol_ptr + fl00 + 2 * HW, mask=mask & okz2 & okc0, other=0.0)
            f10 = tl.load(vol_ptr + fl00 + cstep, mask=mask & okz0 & okc1, other=0.0)
            f11 = tl.load(vol_ptr + fl00 + cstep + HW, mask=mask & okz1 & okc1, other=0.0)
            f12 = tl.load(vol_ptr + fl00 + cstep + 2 * HW, mask=mask & okz2 & okc1,
                          other=0.0)
            f20 = tl.load(vol_ptr + fl00 + 2 * cstep, mask=mask & okz0 & okc2, other=0.0)
            f21 = tl.load(vol_ptr + fl00 + 2 * cstep + HW, mask=mask & okz1 & okc2,
                          other=0.0)
            f22 = tl.load(vol_ptr + fl00 + 2 * cstep + 2 * HW, mask=mask & okz2 & okc2,
                          other=0.0)

            F0 = vW0 * f00 + vW1 * f01 + vW2 * f02       # per cross-tap, z-collapsed
            F1 = vW0 * f10 + vW1 * f11 + vW2 * f12
            F2 = vW0 * f20 + vW1 * f21 + vW2 * f22
            G_acc += hW0 * F0 + hW1 * F1 + hW2 * F2

            if MODE == 2:
                E0 = hW0 * f00 + hW1 * f10 + hW2 * f20   # per z-tap, cross-collapsed
                E1 = hW0 * f01 + hW1 * f11 + hW2 * f21
                E2 = hW0 * f02 + hW1 * f12 + hW2 * f22
                # hW2 = max(0, 1 - hW0 - hW1) feeds back into hW0/hW1
                a2h = tl.where(1.0 - hW1 - hW0 > 0.0, 1.0, 0.0)
                a2v = tl.where(1.0 - vW1 - vW0 > 0.0, 1.0, 0.0)
                hb0 = ghat * (F0 - a2h * F2)
                hb1 = ghat * (F1 - a2h * F2)
                vb0 = ghat * (E0 - a2v * E2)
                vb1 = ghat * (E1 - a2v * E2)
                # clamp gates: 1 where the min/max picked the moving argument
                g_A1 = tl.where((iu_c + hfw < n_pos) & (hW1 > 0.0), 1.0, 0.0)
                g_B1 = tl.where((iu_c - hfw > n_neg) & (hW1 > 0.0), 1.0, 0.0)
                edgeA0 = tl.where(upos, iu_c - hfw, iu_c + 2.0 * hfw)
                edgeB0 = tl.where(upos, iu_c - 2.0 * hfw, iu_c + hfw)
                g_A0 = tl.where((edgeA0 < n_pos) & (hW0 > 0.0), 1.0, 0.0)
                g_B0 = tl.where((edgeB0 > n_neg) & (hW0 > 0.0), 1.0, 0.0)
                biu = hb1 * (g_A1 - g_B1) + hb0 * (g_A0 - g_B0)
                bhf = hb1 * (g_A1 + g_B1) \
                    + hb0 * tl.where(upos, -g_A0 + 2.0 * g_B0, 2.0 * g_A0 - g_B0)
                g_A1v = tl.where((iv_c + vfw < m_pos) & (vW1 > 0.0), 1.0, 0.0)
                g_B1v = tl.where((iv_c - vfw > m_neg) & (vW1 > 0.0), 1.0, 0.0)
                g_A0v = tl.where((iv_c - vfw < m_pos) & (vW0 > 0.0), 1.0, 0.0)
                g_B0v = tl.where((iv_c - 2.0 * vfw > m_neg) & (vW0 > 0.0), 1.0, 0.0)
                biv = vb1 * (g_A1v - g_B1v) + vb0 * (g_A0v - g_B0v)
                bvf = vb1 * (g_A1v + g_B1v) + vb0 * (-g_A0v + 2.0 * g_B0v)

                # iu_c = (pmc_u + t*U - u0g)/du ; iv_c likewise ; hfw = kh*t ; vfw = kv*t
                biu_du = biu / du
                biv_dv = biv / dv
                bt = biu_du * U + biv_dv * Vv + bhf * kh + bvf * kv
                bU = biu_du * t_mag
                bV = biv_dv * t_mag
                bufs += bhf * t_mag                     # kh channel: 0.5*dx*|uf|/du
                # pmc_u/pmc_v channels
                bp_x += biu_du * ux + biv_dv * vx
                bp_y += biu_du * uy + biv_dv * vy
                bp_z += biu_du * uz + biv_dv * vz
                bc_x -= biu_du * ux + biv_dv * vx
                bc_y -= biu_du * uy + biv_dv * vy
                bc_z -= biu_du * uz + biv_dv * vz
                bu_x += biu_du * pmcx + bU * dlx
                bu_y += biu_du * pmcy + bU * dly
                bu_z += biu_du * pmcz + bU * dlz
                bv_x += biv_dv * pmcx + bV * dlx
                bv_y += biv_dv * pmcy + bV * dly
                bv_z += biv_dv * pmcz + bV * dlz
                # Delta channels: every component is (const - p)
                bp_x -= bU * ux + bV * vx
                bp_y -= bU * uy + bV * vy
                bp_z -= bU * uz + bV * vz
                # t = D/dist
                bD += bt / dist
                bdist = -bt * t_mag / dist
                inv_dist = 1.0 / dist
                bbco = bdist * (bco - p_b) * inv_dist
                bzpt = bdist * (zpt - pz) * inv_dist
                bp_b += bdist * qb * inv_dist
                bp_a += bdist * (p_a - w_s) * inv_dist
                bp_z += bdist * qz * inv_dist
                # bco = p_b + dist_a*r_b/r_a ; zpt = pz + dist_a*rz/r_a ; dist_a = w_s - p_a
                bp_b += bbco
                bp_a -= bbco * r_b * inv_ra
                br_b += bbco * dist_a * inv_ra
                br_a -= bbco * dist_a * r_b * inv_ra * inv_ra
                bp_z += bzpt
                bp_a -= bzpt * rz * inv_ra
                br_z += bzpt * dist_a * inv_ra
                br_a -= bzpt * dist_a * rz * inv_ra * inv_ra

        if MODE == 0:
            tl.store(out_ptr + (view * nv * nu).to(tl.int64) + pix, L * G_acc, mask=mask)
        if MODE == 2:
            # t's D channel: D = |r|
            br_x = bD * rx / Dd
            br_y = bD * ry / Dd
            brz2 = bD * rz / Dd
            # length factor L = dx*sqrt(alpha*beta)/r_a^2
            bL = gout * G_acc
            br_a += bL * dx * (r_a * (alpha + beta) / sqab - 2.0 * sqab * inv_ra) / ra2
            br_b += bL * dx * r_b * tl.sqrt(beta / alpha) / ra2
            brz2 += bL * dx * rz * tl.sqrt(alpha / beta) / ra2
            br_z += brz2
            # |u_flat| channel of kh (kh = 0.5*dx*|uf_sel|/du)
            bufk = bufs * 0.5 * dx / du * tl.where(uf_sel >= 0.0, 1.0, -1.0)
            inv_r3 = 1.0 / (rho * rho * rho)
            bu_x += bufk * tl.where(ydom, uy * uy * inv_r3, -ux * uy * inv_r3)
            bu_y += bufk * tl.where(ydom, -ux * uy * inv_r3, ux * ux * inv_r3)
            # unmap the in-plane axis split
            br_x += tl.where(ydom, br_b, br_a)
            br_y += tl.where(ydom, br_a, br_b)
            bp_x += tl.where(ydom, bp_b, bp_a)
            bp_y += tl.where(ydom, bp_a, bp_b)
            # r = (c + s*u + t_r*v) - p
            bp_x -= br_x
            bp_y -= br_y
            bp_z -= br_z
            bc_x += br_x
            bc_y += br_y
            bc_z += br_z
            bu_x += s * br_x
            bu_y += s * br_y
            bu_z += s * br_z
            bv_x += t_r * br_x
            bv_y += t_r * br_y
            bv_z += t_r * br_z
            ob = out_ptr + view * 12
            zero = tl.zeros((BLOCK,), dtype=tl.float32)
            tl.atomic_add(ob + 0, tl.sum(tl.where(mask, bp_x, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 1, tl.sum(tl.where(mask, bp_y, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 2, tl.sum(tl.where(mask, bp_z, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 3, tl.sum(tl.where(mask, bc_x, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 4, tl.sum(tl.where(mask, bc_y, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 5, tl.sum(tl.where(mask, bc_z, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 6, tl.sum(tl.where(mask, bv_x, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 7, tl.sum(tl.where(mask, bv_y, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 8, tl.sum(tl.where(mask, bv_z, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 9, tl.sum(tl.where(mask, bu_x, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 10, tl.sum(tl.where(mask, bu_y, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 11, tl.sum(tl.where(mask, bu_z, zero)).to(tl.float64), sem="relaxed")

    @triton.jit
    def _leap_joseph_kernel(vol_ptr, gout_ptr, out_ptr,
                            src_ptr, mod_ptr, rowv_ptr, colv_ptr,
                            nv, nu, NP, D, HW,
                            dx, dz, du, dv, u0g, v0g, b0, z0,
                            BLOCK: tl.constexpr, MODE: tl.constexpr):
        """modularBeamJosephProjectorKernel + lineIntegral_Joseph_ZYX (in-plane dominant).

        Bilinear sample at every dominant-axis voxel plane; the entry-plane sample carries an
        extra -0.5 weight (LEAP's start correction with the edge point ON the plane); length
        = dx * |r| / |r_a|. Same MODE/output conventions as `_leap_sf_kernel`.
        """
        view = tl.program_id(1)
        pix = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = pix < nv * nu
        m = pix // nu
        n = pix - m * nu

        px = tl.load(src_ptr + view * 3 + 0)
        py = tl.load(src_ptr + view * 3 + 1)
        pz = tl.load(src_ptr + view * 3 + 2)
        cx = tl.load(mod_ptr + view * 3 + 0)
        cy = tl.load(mod_ptr + view * 3 + 1)
        cz_ = tl.load(mod_ptr + view * 3 + 2)
        vx = tl.load(rowv_ptr + view * 3 + 0)
        vy = tl.load(rowv_ptr + view * 3 + 1)
        vz = tl.load(rowv_ptr + view * 3 + 2)
        ux = tl.load(colv_ptr + view * 3 + 0)
        uy = tl.load(colv_ptr + view * 3 + 1)
        uz = tl.load(colv_ptr + view * 3 + 2)

        t_r = m.to(tl.float32) * dv + v0g
        s = n.to(tl.float32) * du + u0g

        detx = cx + ux * s + vx * t_r
        dety = cy + uy * s + vy * t_r
        detz = cz_ + uz * s + vz * t_r
        rx = detx - px
        ry = dety - py
        rz = detz - pz

        ydom = tl.abs(ry) > tl.abs(rx)
        r_a = tl.where(ydom, ry, rx)
        r_b = tl.where(ydom, rx, ry)
        p_a = tl.where(ydom, py, px)
        p_b = tl.where(ydom, px, py)
        inv_ra = 1.0 / r_a
        j0 = tl.where(r_a > 0.0, 0, NP - 1)

        nrm = tl.sqrt(rx * rx + ry * ry + rz * rz)
        abs_ra = tl.abs(r_a)
        L = dx * nrm / abs_ra

        gout = tl.zeros((BLOCK,), dtype=tl.float32)
        if MODE == 2:
            gout = tl.load(gout_ptr + (view * nv * nu).to(tl.int64) + pix, mask=mask,
                           other=0.0)
        ghat = gout * L

        S_tot = tl.zeros((BLOCK,), dtype=tl.float32)
        bp_a = tl.zeros((BLOCK,), dtype=tl.float32)
        bp_b = tl.zeros((BLOCK,), dtype=tl.float32)
        bp_z = tl.zeros((BLOCK,), dtype=tl.float32)
        br_a = tl.zeros((BLOCK,), dtype=tl.float32)
        br_b = tl.zeros((BLOCK,), dtype=tl.float32)
        br_z = tl.zeros((BLOCK,), dtype=tl.float32)

        for j in range(0, NP):
            w_j = j * dx + b0
            lam = (w_j - p_a) * inv_ra
            cb = (p_b + lam * r_b - b0) / dx
            cz = (pz + lam * rz - z0) / dz
            ibf = tl.floor(cb)
            izf = tl.floor(cz)
            wb = cb - ibf
            wz = cz - izf
            ib = ibf.to(tl.int32)
            iz = izf.to(tl.int32)
            okc0 = (ib >= 0) & (ib < NP)
            okc1 = (ib + 1 >= 0) & (ib + 1 < NP)
            okz0 = (iz >= 0) & (iz < D)
            okz1 = (iz + 1 >= 0) & (iz + 1 < D)
            cstep = tl.where(ydom, 1, NP).to(tl.int64)
            fl00 = iz.to(tl.int64) * HW + tl.where(ydom, j * NP, j).to(tl.int64) \
                + ib.to(tl.int64) * cstep
            f00 = tl.load(vol_ptr + fl00, mask=mask & okz0 & okc0, other=0.0)
            f10 = tl.load(vol_ptr + fl00 + cstep, mask=mask & okz0 & okc1, other=0.0)
            f01 = tl.load(vol_ptr + fl00 + HW, mask=mask & okz1 & okc0, other=0.0)
            f11 = tl.load(vol_ptr + fl00 + cstep + HW, mask=mask & okz1 & okc1, other=0.0)
            S = (1.0 - wb) * (1.0 - wz) * f00 + wb * (1.0 - wz) * f10 \
                + (1.0 - wb) * wz * f01 + wb * wz * f11
            wgt = tl.where(j == j0, 0.5, 1.0)
            S_tot += wgt * S
            if MODE == 2:
                scale = ghat * wgt
                bcb = scale * ((1.0 - wz) * (f10 - f00) + wz * (f11 - f01))
                bcz = scale * ((1.0 - wb) * (f01 - f00) + wb * (f11 - f10))
                # cb = (p_b - b0 + lam*r_b)/dx ; cz = (pz - z0 + lam*rz)/dz
                bp_b += bcb / dx
                bp_z += bcz / dz
                blam = bcb * r_b / dx + bcz * rz / dz
                br_b += bcb * lam / dx
                br_z += bcz * lam / dz
                bp_a -= blam * inv_ra
                br_a -= blam * lam * inv_ra

        if MODE == 0:
            tl.store(out_ptr + (view * nv * nu).to(tl.int64) + pix, L * S_tot, mask=mask)
        if MODE == 2:
            # L = dx * |r| / |r_a|
            bL = gout * S_tot
            sgn_a = tl.where(r_a >= 0.0, 1.0, -1.0)
            br_a += bL * dx * (r_a / (abs_ra * nrm) - nrm * sgn_a / (r_a * r_a))
            br_b += bL * dx * r_b / (abs_ra * nrm)
            br_z += bL * dx * rz / (abs_ra * nrm)
            br_x = tl.where(ydom, br_b, br_a)
            br_y = tl.where(ydom, br_a, br_b)
            bp_x = tl.where(ydom, bp_b, bp_a)
            bp_y = tl.where(ydom, bp_a, bp_b)
            bp_x -= br_x
            bp_y -= br_y
            bp_z2 = bp_z - br_z
            bc_x = br_x
            bc_y = br_y
            bc_z = br_z
            bu_x = s * br_x
            bu_y = s * br_y
            bu_z = s * br_z
            bv_x = t_r * br_x
            bv_y = t_r * br_y
            bv_z = t_r * br_z
            ob = out_ptr + view * 12
            zero = tl.zeros((BLOCK,), dtype=tl.float32)
            tl.atomic_add(ob + 0, tl.sum(tl.where(mask, bp_x, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 1, tl.sum(tl.where(mask, bp_y, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 2, tl.sum(tl.where(mask, bp_z2, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 3, tl.sum(tl.where(mask, bc_x, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 4, tl.sum(tl.where(mask, bc_y, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 5, tl.sum(tl.where(mask, bc_z, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 6, tl.sum(tl.where(mask, bv_x, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 7, tl.sum(tl.where(mask, bv_y, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 8, tl.sum(tl.where(mask, bv_z, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 9, tl.sum(tl.where(mask, bu_x, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 10, tl.sum(tl.where(mask, bu_y, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 11, tl.sum(tl.where(mask, bu_z, zero)).to(tl.float64), sem="relaxed")


if HAVE_TRITON:

    @triton.jit
    def _vd_tangent_kernel(g_ptr, src_ptr, mod_ptr, rowv_ptr, colv_ptr,
                           dsrc_ptr, dmod_ptr, drowv_ptr, dcolv_ptr,
                           wgt_ptr, dwgt_ptr, out_ptr, dout_ptr,
                           V, nv, nu, Npix, W, H, D,
                           dx, dy, dz, du, dv, eps,
                           BLOCK: tl.constexpr):
        """VALUE and s-directional DERIVATIVE of LEAP's modular VD backprojection.

        The value replicates `modularBeamBackprojectorKernel_vox_stack` exactly (general
        branch; the |n.z|<=1e-5 branch is the same maths with D hoisted): per (voxel, view)

            n = u x v,  pmc = p - c,  r = x - p,  D = -(pmc.n)/(r.n)
            ju = (pmc.u + D r.u)/du + (nu-1)/2      jv likewise with v, dv
            Wt = (pmc.n) sqrt(D^2((r.u)^2+(r.v)^2) + (pmc.n)^2) / (r.n)^2
            out += wgt_view * Wt * bilerp(g2; ju, jv)

        (LEAP's dxdydz/(dudv) scalar and our FDK wrapper's inverse cancel, so none appears;
        `g2` arrives with the 1/(sdd*dist) fold already applied, exactly like the value path.)

        The derivative is the product rule through every geometry-dependent factor, with
        (dp, dc, du_vec, dv_vec) = d/ds of the modular arrays along Pdot (supplied by
        `modular_arrays_jvp`), plus the per-view weight derivative dwgt (the Voronoi share
        moves with the orbit). The bilinear cell and the border mask are FROZEN at the
        evaluation point -- the same a.e. convention as everywhere else in this module; the
        interpolant's own (d/dju, d/djv) come from the same four taps.

        Voxel-driven, no atomics, deterministic. One lane = one voxel, loop over views.
        """
        pid_b = tl.program_id(1)
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = p < Npix
        iz = p // (H * W)
        rem = p - iz * (H * W)
        iy = rem // W
        ixv = rem - iy * W
        x = (ixv.to(tl.float32) - (W - 1) * 0.5) * dx
        y = (iy.to(tl.float32) - (H - 1) * 0.5) * dy
        z = (iz.to(tl.float32) - (D - 1) * 0.5) * dz
        off_u = (nu - 1) * 0.5
        off_v = (nv - 1) * 0.5
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        dacc = tl.zeros((BLOCK,), dtype=tl.float32)
        for view in range(0, V):
            px = tl.load(src_ptr + view * 3 + 0)
            py = tl.load(src_ptr + view * 3 + 1)
            pz = tl.load(src_ptr + view * 3 + 2)
            cx = tl.load(mod_ptr + view * 3 + 0)
            cy = tl.load(mod_ptr + view * 3 + 1)
            cz = tl.load(mod_ptr + view * 3 + 2)
            vx = tl.load(rowv_ptr + view * 3 + 0)
            vy = tl.load(rowv_ptr + view * 3 + 1)
            vz = tl.load(rowv_ptr + view * 3 + 2)
            ux = tl.load(colv_ptr + view * 3 + 0)
            uy = tl.load(colv_ptr + view * 3 + 1)
            uz = tl.load(colv_ptr + view * 3 + 2)
            dpx = tl.load(dsrc_ptr + view * 3 + 0)
            dpy = tl.load(dsrc_ptr + view * 3 + 1)
            dpz = tl.load(dsrc_ptr + view * 3 + 2)
            dcx = tl.load(dmod_ptr + view * 3 + 0)
            dcy = tl.load(dmod_ptr + view * 3 + 1)
            dcz = tl.load(dmod_ptr + view * 3 + 2)
            dvx = tl.load(drowv_ptr + view * 3 + 0)
            dvy = tl.load(drowv_ptr + view * 3 + 1)
            dvz = tl.load(drowv_ptr + view * 3 + 2)
            dux = tl.load(dcolv_ptr + view * 3 + 0)
            duy = tl.load(dcolv_ptr + view * 3 + 1)
            duz = tl.load(dcolv_ptr + view * 3 + 2)
            wg = tl.load(wgt_ptr + pid_b * V + view)
            dwg = tl.load(dwgt_ptr + pid_b * V + view)

            nx = uy * vz - uz * vy
            ny = uz * vx - ux * vz
            nz = ux * vy - uy * vx
            dnx = duy * vz + uy * dvz - duz * vy - uz * dvy
            dny = duz * vx + uz * dvx - dux * vz - ux * dvz
            dnz = dux * vy + ux * dvy - duy * vx - uy * dvx
            pmcx = px - cx
            pmcy = py - cy
            pmcz = pz - cz
            dpmcx = dpx - dcx
            dpmcy = dpy - dcy
            dpmcz = dpz - dcz
            pmcn = pmcx * nx + pmcy * ny + pmcz * nz
            dpmcn = dpmcx * nx + dpmcy * ny + dpmcz * nz \
                + pmcx * dnx + pmcy * dny + pmcz * dnz
            pmcu = pmcx * ux + pmcy * uy + pmcz * uz
            dpmcu = dpmcx * ux + dpmcy * uy + dpmcz * uz \
                + pmcx * dux + pmcy * duy + pmcz * duz
            pmcv = pmcx * vx + pmcy * vy + pmcz * vz
            dpmcv = dpmcx * vx + dpmcy * vy + dpmcz * vz \
                + pmcx * dvx + pmcy * dvy + pmcz * dvz

            rx = x - px
            ry = y - py
            rz = z - pz
            rdn = rx * nx + ry * ny + rz * nz
            drdn = -dpx * nx - dpy * ny - dpz * nz + rx * dnx + ry * dny + rz * dnz
            ok = mask & (tl.abs(rdn) > eps)
            rdn_s = tl.where(tl.abs(rdn) > eps, rdn, 1.0)
            inv_rdn = 1.0 / rdn_s
            Dm = -pmcn * inv_rdn
            # d(-pmcn/rdn) = -(dpmcn - pmcn*drdn/rdn)/rdn, and pmcn/rdn = -Dm
            dDm = -(dpmcn + Dm * drdn) * inv_rdn
            ru = rx * ux + ry * uy + rz * uz
            dru = -dpx * ux - dpy * uy - dpz * uz + rx * dux + ry * duy + rz * duz
            rv = rx * vx + ry * vy + rz * vz
            drv = -dpx * vx - dpy * vy - dpz * vz + rx * dvx + ry * dvy + rz * dvz

            ju = (pmcu + Dm * ru) / du + off_u
            jv = (pmcv + Dm * rv) / dv + off_v
            dju = (dpmcu + dDm * ru + Dm * dru) / du
            djv = (dpmcv + dDm * rv + Dm * drv) / dv

            rho2 = ru * ru + rv * rv
            drho2 = 2.0 * (ru * dru + rv * drv)
            S2 = Dm * Dm * rho2 + pmcn * pmcn
            S = tl.sqrt(tl.maximum(S2, 1e-30))
            dS = (Dm * dDm * rho2 + 0.5 * Dm * Dm * drho2 + pmcn * dpmcn) / S
            Wt = pmcn * S * inv_rdn * inv_rdn
            dWt = (dpmcn * S + pmcn * dS) * inv_rdn * inv_rdn \
                - 2.0 * Wt * drdn * inv_rdn

            # -- bilinear taps, grid_sample(zeros) convention == tex border-zero
            x0 = tl.floor(ju)
            y0 = tl.floor(jv)
            fu = ju - x0
            fv = jv - y0
            iu = x0.to(tl.int32)
            ivx = y0.to(tl.int32)
            base = ((pid_b * V + view) * nv).to(tl.int64) * nu
            oku0 = (iu >= 0) & (iu < nu)
            oku1 = (iu + 1 >= 0) & (iu + 1 < nu)
            okv0 = (ivx >= 0) & (ivx < nv)
            okv1 = (ivx + 1 >= 0) & (ivx + 1 < nv)
            fl = base + ivx.to(tl.int64) * nu + iu
            t00 = tl.load(g_ptr + fl, mask=ok & okv0 & oku0, other=0.0)
            t10 = tl.load(g_ptr + fl + 1, mask=ok & okv0 & oku1, other=0.0)
            t01 = tl.load(g_ptr + fl + nu, mask=ok & okv1 & oku0, other=0.0)
            t11 = tl.load(g_ptr + fl + nu + 1, mask=ok & okv1 & oku1, other=0.0)
            G = (1.0 - fu) * (1.0 - fv) * t00 + fu * (1.0 - fv) * t10 \
                + (1.0 - fu) * fv * t01 + fu * fv * t11
            Gu = (1.0 - fv) * (t10 - t00) + fv * (t11 - t01)
            Gv = (1.0 - fu) * (t01 - t00) + fu * (t11 - t10)
            dG = Gu * dju + Gv * djv

            acc += tl.where(ok, wg * Wt * G, 0.0)
            dacc += tl.where(ok, wg * (Wt * dG + dWt * G) + dwg * Wt * G, 0.0)

        tl.store(out_ptr + pid_b.to(tl.int64) * Npix + p, acc, mask=mask)
        tl.store(dout_ptr + pid_b.to(tl.int64) * Npix + p, dacc, mask=mask)


def modular_arrays_jvp(P: torch.Tensor, dP: torch.Tensor, u0: float, v_off: float):
    """(arrays, d/ds arrays) of the modular decomposition along Pdot, fp64 -> fp32.

    The forward-mode twin of the vjp in `leap_grad_P`: the bridge parameterizes the orbit as
    P(s) and needs d(src, mod, rowv, colv)/ds for the tangent kernel above."""
    P64 = P.detach().to(torch.float64)
    dP64 = dP.detach().to(torch.float64)
    arrs, darrs = torch.autograd.functional.jvp(
        lambda Q: modular_arrays_torch(Q, u0, v_off), P64, dP64)
    to32 = lambda t: t.to(torch.float32).contiguous()
    return tuple(to32(a) for a in arrs), tuple(to32(d) for d in darrs)


def leap_vd_backproject_tangent(g2, P, dP, wgt, dwgt, *, D, H, W, dx, dy, dz, du, dv,
                                u0=0.0, v_off=0.0, block=256):
    """(x, dx/ds) of `sum_v wgt_v * VD-backprojection_v(g2; geometry(P))` along (dP, dwgt).

    `g2` (B,V,nv,nu) is the FILTERED sinogram with the 1/(sdd*dist) fold already applied --
    i.e. exactly what the deployed `leap_fdk_backproject` interpolates -- and s-independent.
    The caller applies the physical-norm scale, like every other backprojection here."""
    B, V, nv, nu = g2.shape
    x = torch.empty((B, D, H, W), device=g2.device, dtype=torch.float32)
    dxds = torch.empty_like(x)
    g2 = g2.contiguous().to(torch.float32)
    wgt = wgt.to(g2.device, torch.float32).reshape(B, V).contiguous()
    dwgt = dwgt.to(g2.device, torch.float32).reshape(B, V).contiguous()
    Npix = D * H * W
    grid = (triton.cdiv(Npix, block), B)
    for b in range(B):
        (src, mod, rowv, colv), (dsrc, dmod, drowv, dcolv) = \
            modular_arrays_jvp(P[b], dP[b], u0, v_off)
        _vd_tangent_kernel[grid[:1] + (1,)](
            g2[b:b + 1], src, mod, rowv, colv, dsrc, dmod, drowv, dcolv,
            wgt[b:b + 1], dwgt[b:b + 1], x[b:b + 1], dxds[b:b + 1],
            V, nv, nu, Npix, W, H, D,
            float(dx), float(dy), float(dz), float(du), float(dv), 1e-8,
            BLOCK=block)
    return x, dxds


def modular_arrays_torch(P: torch.Tensor, u0: float, v_off: float):
    """Differentiable P (V,3,4) -> (src, mod, rowv, colv), each (V,3), P's dtype/device.

    The same decomposition as `leap_projector.decompose_P`/`modular_arrays`, kept in torch so
    autograd can close the (src,mod,rowv,colv) -> P chain. LEAP's ingest normalization and
    re-orthogonalization are NOT replicated: on rigid P they are the identity map with
    identity tangent Jacobian (module docstring), and the estimator only ever probes rigidly.
    """
    M, p4 = P[..., :3], P[..., 3]
    C = torch.linalg.solve(M, -p4.unsqueeze(-1)).squeeze(-1)
    sdd = M[..., 0, :].norm(dim=-1)
    e_u = M[..., 0, :] / sdd[..., None]
    e_v = M[..., 1, :] / sdd[..., None]
    e_n = M[..., 2, :]
    mod = C + sdd[..., None] * e_n + u0 * e_u + v_off * e_v
    return C, mod, e_v, e_u


def _launch(kind, mode, vol, gout, out, arrs, *, nv, nu, D, H, W, dx, dz, du, dv, block=128):
    src, mod, rowv, colv = arrs
    if W != H:
        raise ValueError(f"the unified in-plane slab loop needs W == H, got {W}x{H}")
    if kind == "JOSEPH" and abs(dz - dx) > 1e-9:
        raise ValueError(f"LEAP's Joseph kernel assumes an isotropic voxel, got dx={dx} dz={dz}")
    V = src.shape[0]
    u0g = -0.5 * (nu - 1) * du
    v0g = -0.5 * (nv - 1) * dv
    b0 = -0.5 * (W - 1) * dx
    z0 = -0.5 * (D - 1) * dz
    kern = _leap_sf_kernel if kind == "SF" else _leap_joseph_kernel
    grid = (triton.cdiv(nv * nu, block), V)
    kern[grid](vol, gout if gout is not None else out, out,
               src, mod, rowv, colv,
               nv, nu, W, D, H * W,
               float(dx), float(dz), float(du), float(dv),
               float(u0g), float(v0g), float(b0), float(z0),
               BLOCK=block, MODE=mode)


def _arrays32(P_view, u0, v_off):
    with torch.no_grad():
        arrs = modular_arrays_torch(P_view.to(torch.float64), u0, v_off)
    return tuple(a.to(torch.float32).contiguous() for a in arrs)


def leap_forward_model(vol, P, *, nv, nu, dx, dy, dz, du, dv, u0=0.0, v_off=0.0, kind=None):
    """VALUE of the transcribed LEAP model: vol (B,D,H,W), P (B,V,3,4) -> (B,V,nv,nu).

    The parity rig for `gate_leap_projector`: this must match `leap_project` itself to tex-
    quantization precision, per branch. `kind` overrides the launcher-replica selection."""
    if abs(dx - dy) > 1e-9:
        raise ValueError(f"dx == dy required, got {dx} vs {dy}")
    B, D, H, W = vol.shape
    V = P.shape[1]
    from .leap_projector import kernel_kind
    g = torch.zeros((B, V, nv, nu), device=vol.device, dtype=torch.float32)
    vol = vol.contiguous().to(torch.float32)
    for b in range(B):
        k = kind or kernel_kind(P[b], nv=nv, du=du, dv=dv, dx=dx, dz=dz, D=D, H=H, W=W)
        arrs = _arrays32(P[b], u0, v_off)
        _launch(k, 0, vol[b], None, g[b], arrs, nv=nv, nu=nu, D=D, H=H, W=W,
                dx=dx, dz=dz, du=du, dv=dv)
    return g


def leap_grad_geom(vol, gout, P_view, *, nv, nu, dx, dz, du, dv, u0=0.0, v_off=0.0,
                   kind=None):
    """d<gout, LEAP(vol; geom)>/d(src, mod, rowv, colv) for ONE geometry set.

    vol (D,H,W), gout (V,nv,nu), P_view (V,3,4) -> four (V,3) fp64 tensors."""
    D, H, W = vol.shape
    from .leap_projector import kernel_kind
    k = kind or kernel_kind(P_view, nv=nv, du=du, dv=dv, dx=dx, dz=dz, D=D, H=H, W=W)
    arrs = _arrays32(P_view, u0, v_off)
    out = torch.zeros((P_view.shape[0], 12), device=vol.device, dtype=torch.float64)
    _launch(k, 2, vol.contiguous().to(torch.float32),
            gout.contiguous().to(torch.float32), out, arrs,
            nv=nv, nu=nu, D=D, H=H, W=W, dx=dx, dz=dz, du=du, dv=dv)
    return out[:, 0:3], out[:, 3:6], out[:, 6:9], out[:, 9:12]


def leap_grad_P(vol, gout, P, *, dx, dy, dz, du, dv, u0=0.0, v_off=0.0):
    """d <gout, LEAP(vol; P)> / dP -> (B,V,3,4) fp32. Drop-in for `triton_sf.sf_grad_P`,
    but the gradient of LEAP'S OWN forward (branch-matched), chained through the
    differentiable modular decomposition."""
    if abs(dx - dy) > 1e-9:
        raise ValueError(f"dx == dy required, got {dx} vs {dy}")
    B, D, H, W = vol.shape
    V = P.shape[1]
    nv, nu = gout.shape[2], gout.shape[3]
    gP = torch.zeros((B, V, 3, 4), device=vol.device, dtype=torch.float32)
    for b in range(B):
        gs = leap_grad_geom(vol[b], gout[b], P[b], nv=nv, nu=nu, dx=dx, dz=dz,
                            du=du, dv=dv, u0=u0, v_off=v_off)
        # this may run inside another backward pass (grad disabled there), so the vjp graph
        # through the modular decomposition needs grad re-enabled explicitly
        with torch.enable_grad():
            Pb = P[b].detach().to(torch.float64).requires_grad_(True)
            arrs = modular_arrays_torch(Pb, u0, v_off)
            gPb, = torch.autograd.grad(arrs, Pb, grad_outputs=gs)
        gP[b] = gPb.to(torch.float32)
    return gP
