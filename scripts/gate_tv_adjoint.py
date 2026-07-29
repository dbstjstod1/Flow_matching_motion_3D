"""Gate: `div_adjoint_3d` really is the adjoint of `grad_forward_3d`, and `shrink` really is the
prox of k*||.||_1.

WHY A GATE. A wrong boundary term in D^T does not crash and does not obviously distort the image
-- it makes A^T A + rho D^T D a slightly non-symmetric operator, which quietly breaks CG (whose
whole derivation assumes symmetry) and shows up only as "ADMM converges worse than the heuristic
it replaced". The two-line inner-product identity below catches it immediately.

    python scripts/gate_tv_adjoint.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.tv import div_adjoint_3d, grad_forward_3d, shrink


def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ok = True

    # ---- 1. adjointness: <D x, p> == <x, D^T p> for random x and random p ----------------
    for shape in [(1, 1, 7, 5, 6), (1, 1, 16, 16, 16), (1, 1, 32, 24, 20)]:
        x = torch.randn(shape, device=dev, dtype=torch.float64)
        dz, dy, dx = grad_forward_3d(x)
        pz, py, px = (torch.randn_like(dz), torch.randn_like(dy), torch.randn_like(dx))
        lhs = (dz * pz).sum() + (dy * py).sum() + (dx * px).sum()      # <D x, p>
        rhs = (x * div_adjoint_3d(pz, py, px, shape, dtype=torch.float64)).sum()  # <x, D^T p>
        rel = float((lhs - rhs).abs() / lhs.abs().clamp_min(1e-30))
        good = rel < 1e-12
        ok &= good
        print(f"{'PASS' if good else 'FAIL'} adjoint {str(shape):>18}  "
              f"<Dx,p>={float(lhs):+.6e}  <x,D^Tp>={float(rhs):+.6e}  rel={rel:.2e}")

    # ---- 2. D^T D is symmetric positive SEMI-definite (CG's precondition) ----------------
    shape = (1, 1, 10, 9, 8)
    a = torch.randn(shape, device=dev, dtype=torch.float64)
    b = torch.randn(shape, device=dev, dtype=torch.float64)

    def DtD(v):
        return div_adjoint_3d(*grad_forward_3d(v), shape, dtype=torch.float64)

    s1, s2 = float((DtD(a) * b).sum()), float((a * DtD(b)).sum())
    rel = abs(s1 - s2) / max(abs(s1), 1e-30)
    good = rel < 1e-12
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} D^T D symmetric   <DtD a,b>={s1:+.6e} "
          f"<a,DtD b>={s2:+.6e}  rel={rel:.2e}")
    q = float((a * DtD(a)).sum())
    good = q >= 0
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} D^T D PSD         <a,DtD a>={q:+.6e} (>= 0)")

    # constants are in the null space of D (TV of a constant is zero) -- the reason ADMM needs
    # the data term to pin the DC level, and the reason D^T D alone is only semi-definite.
    c = torch.ones(shape, device=dev, dtype=torch.float64)
    n = float(DtD(c).abs().max())
    good = n < 1e-12
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} D 1 = 0           max|D^T D 1|={n:.2e}")

    # ---- 3. shrink is the prox of k*||.||_1 ---------------------------------------------
    # brute-force check against argmin_z 0.5(z-a)^2 + k|z| on a grid
    k = 0.37
    a1 = torch.linspace(-2, 2, 401, dtype=torch.float64)
    got = shrink(a1, k)
    zs = torch.linspace(-3, 3, 60001, dtype=torch.float64)
    obj = 0.5 * (zs[None, :] - a1[:, None]) ** 2 + k * zs[None, :].abs()
    ref = zs[obj.argmin(dim=1)]
    err = float((got - ref).abs().max())
    good = err < 2e-4                      # grid resolution of the brute force
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} shrink == prox_l1  max|closed-form - brute|={err:.2e}")

    print("\nALL PASS" if ok else "\nSOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
