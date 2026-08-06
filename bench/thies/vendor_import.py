"""Import the VERBATIM upstream sources under `vendor/` without editing them.

Both vendored trees use flat, top-level imports (`from helper import ...`), which is why they
cannot simply be `from .vendor.geometry_gradients_CT import ...`. Rewriting those imports would
mean editing upstream code, and the whole value of `vendor/` is that it is byte-identical to what
Thies et al. released (see PROVENANCE.md). So we put the directories on `sys.path` instead.

`numba.cuda` is imported eagerly by `backprojector_cone`, so the first call here pays the numba
import (~1 s) and any missing-CUDA error surfaces immediately with a message that says what to
install, instead of a bare ModuleNotFoundError three frames deep.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
GGCT = os.path.join(_HERE, "vendor", "geometry_gradients_CT")
MDL = os.path.join(_HERE, "vendor", "moco_diff_likelihood")

_INSTALL_HINT = (
    "the vendored Thies backprojector needs numba-cuda. On this box:\n"
    '    pip install numba-cuda "cuda-bindings==12.9.*" "nvidia-cuda-nvcc-cu12==12.2.*"\n'
    "The nvcc pin is deliberate -- driver 535 (CUDA 12.2) will not JIT PTX from a 12.8 libnvvm. "
    "See bench/thies/PROVENANCE.md section 5."
)


def _ensure_path(p: str) -> None:
    if p not in sys.path:
        sys.path.insert(0, p)


def cone_backprojector():
    """`DifferentiableConeBeamBackprojector` (the class, not `.apply`)."""
    _ensure_path(GGCT)
    try:
        from backprojector_cone import DifferentiableConeBeamBackprojector
    except ImportError as e:                       # numba / numba-cuda missing
        raise ImportError(f"{e}\n\n{_INSTALL_HINT}") from e
    return DifferentiableConeBeamBackprojector


def vendored_geometry():
    """Their `Geometry` collector. The cone kernel reads only `volume_shape/spacing/origin`
    off it (`backprojector_cone.call_forward_kernel`); `detector_*` is carried for the fan-beam
    half and is unused here, because the whole detector mapping lives inside the projection
    matrices."""
    _ensure_path(GGCT)
    from geometry import Geometry
    return Geometry


def akima():
    """Their torch, differentiable Akima interpolator (`interpolate_akima_spline`).

    Matches `scipy.interpolate.Akima1DInterpolator` to 1.8e-15 on a 10-node / 360-view draw
    (gate check G1). NOTE its calling convention: `interpolation_points` must be an INTEGER
    tensor -- it is used directly as an index (`spline_piece = torch.zeros_like(...)`).
    `motion.ThiesSplineMotion` does that cast; do not call this raw with float sample points."""
    _ensure_path(MDL)
    from akima_spline import interpolate_akima_spline
    return interpolate_akima_spline
