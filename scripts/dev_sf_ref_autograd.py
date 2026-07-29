"""Pure-torch differentiable reference of the CORNER-parameterized SF forward (v2), tiny
problem. Autograd through it = the EXACT dL/dP of the model; sf_grad_P must match all 12
entries (the rig that caught v1's missing direct-Jacobian block and validated v2).

    python scripts/dev_sf_ref_autograd.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fm3d.triton_sf import sf_grad_P, sf_project

DEV = "cuda"
torch.manual_seed(0)
D = H = W = 16
nv = nu = 24
dx = dy = dz = 1.0
du = dv = 0.9
u0 = v_off = 0.0

vol = torch.rand(D, H, W, device=DEV)
ang = 0.7
R = 60.0
P = torch.tensor([[[[R * torch.cos(torch.tensor(ang)), R * torch.sin(torch.tensor(ang)), 3.0, 1.0],
                    [2.0, -1.5, R, 0.5],
                    [-torch.sin(torch.tensor(ang)), torch.cos(torch.tensor(ang)), 0.02, 70.0]]]],
                 device=DEV, dtype=torch.float32)


def ref_forward(P):
    p = P[0, 0]
    # DETACHED, matching the kernel: sdd = |P row0[:3]| is invariant under rigid T(theta)
    # (rotations preserve the norm), so d(sdd)/d(theta) = 0 exactly and the kernel treats it
    # as a per-view constant. Differentiating it here would add a d(sdd)/dP0j = P0j/sdd term
    # to ONLY the row-0 x/y/z entries -- the exact fingerprint that burned an hour before this
    # comment existed.
    sdd = p[0, :3].norm().detach()
    zz, yy, xx = torch.meshgrid(
        (torch.arange(D, device=DEV) - (D - 1) / 2) * dz,
        (torch.arange(H, device=DEV) - (H - 1) / 2) * dy,
        (torch.arange(W, device=DEV) - (W - 1) / 2) * dx, indexing="ij")
    x = xx.reshape(-1)
    y = yy.reshape(-1)
    z = zz.reshape(-1)
    hx, hy, hz = 0.5 * dx, 0.5 * dy, 0.5 * dz

    js = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = x + sx * hx, y + sy * hy
            uh = p[0, 0] * cx + p[0, 1] * cy + p[0, 2] * z + p[0, 3]
            w = p[2, 0] * cx + p[2, 1] * cy + p[2, 2] * z + p[2, 3]
            js.append((uh / w - u0) / du + (nu - 1) * 0.5)
    j, _ = torch.sort(torch.stack(js, 0), dim=0)          # ju1..ju4
    j1, j2, j3, j4 = j[0], j[1], j[2], j[3]

    def proj_v(cz):
        vh = p[1, 0] * x + p[1, 1] * y + p[1, 2] * cz + p[1, 3]
        w = p[2, 0] * x + p[2, 1] * y + p[2, 2] * cz + p[2, 3]
        return (vh / w - v_off) / dv + (nv - 1) * 0.5

    va = proj_v(z - hz)
    vb = proj_v(z + hz)
    ma = torch.minimum(va, vb)
    mb = torch.maximum(va, vb)

    wc = p[2, 0] * x + p[2, 1] * y + p[2, 2] * z + p[2, 3]
    mw_e = 0.5 * ((j3 + j4) - (j1 + j2))
    wv_e = mb - ma
    mwp = torch.clamp(mw_e * du * wc / sdd, min=0.05 * min(dx, dy))
    wvp = torch.clamp(wv_e * dv * wc / sdd, min=0.05 * dz)
    peak = dx * dy * dz / (mwp * wvp)
    d1 = torch.clamp(j2 - j1, min=1e-6)
    d2 = torch.clamp(j4 - j3, min=1e-6)

    def G(t):
        r = torch.clamp((t - j1) / d1, 0.0, 1.0)
        q = torch.clamp((t - j3) / d2, 0.0, 1.0)
        pl = torch.minimum(torch.maximum(t, j2), j3) - j2
        return 0.5 * d1 * r * r + pl + d2 * (q - 0.5 * q * q)

    g = torch.zeros(nv, nu, device=DEV)
    n0 = torch.floor(j1 + 0.5).long()
    m0 = torch.floor(ma + 0.5).long()
    val = vol.reshape(-1)
    for cm in range(3):
        m = m0 + cm
        mf = m.float()
        Vm = torch.clamp(torch.minimum(mb, mf + 0.5) - torch.maximum(ma, mf - 0.5), min=0.0)
        okm = (m >= 0) & (m < nv)
        for cn in range(4):
            n = n0 + cn
            nf = n.float()
            Wn = G(nf + 0.5) - G(nf - 0.5)
            ok = okm & (n >= 0) & (n < nu)
            contrib = torch.where(ok, val * peak * Wn * Vm, torch.zeros_like(val))
            idx = (m.clamp(0, nv - 1) * nu + n.clamp(0, nu - 1))
            g.view(-1).scatter_add_(0, idx, contrib)
    return g


with torch.no_grad():
    y = ref_forward(P) * 0.9
Pg = P.clone().requires_grad_(True)
g = ref_forward(Pg)
loss = ((g - y) ** 2).mean()
loss.backward()
dP_ref = Pg.grad[0, 0]

with torch.no_grad():
    g_k = sf_project(vol[None], P, nv=nv, nu=nu, dx=dx, dy=dy, dz=dz, du=du, dv=dv,
                     u0=u0, v_off=v_off)
    print("fwd parity kernel-vs-ref:", float((g_k[0, 0] - g).norm() / g.norm()))
    ghat = (2.0 * (g - y) / g.numel())[None, None]
    dP_k = sf_grad_P(vol[None], ghat, P, dx=dx, dy=dy, dz=dz, du=du, dv=dv,
                     u0=u0, v_off=v_off)[0, 0]
bad = 0
for r in range(3):
    for c in range(4):
        a, b = float(dP_k[r, c]), float(dP_ref[r, c])
        ratio = a / b if b != 0 else float("nan")
        if abs(ratio - 1.0) > 1e-3:
            bad += 1
        print(f"P{r}{c}: kernel {a:+.5e}  ref {b:+.5e}  ratio {ratio:+.4f}")
print("PARITY:", "PASS (12/12)" if bad == 0 else f"FAIL ({12-bad}/12)")
