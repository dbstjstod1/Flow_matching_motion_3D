"""Thies et al. (IEEE TMI 2025) rebuilt as a benchmark against our flow-matching posterior.

READ `bench/thies/PROVENANCE.md` FIRST. It states what is upstream code (verbatim, Apache-2.0),
what we reimplemented from the paper text, the six places we could not follow the paper, and --
most importantly -- that this baseline is run at 2x the paper's evaluation amplitude on purpose,
so its scores are NOT comparable to the published figures without `--thies_amp`.
"""

from .motion import ThiesSplineMotion, akima_resample
from .qmnet import QualityMetricUNet3D, THIES_F_MAPS
from .recon import ThiesConeRecon, VolumeGrid, to_unit, from_unit, MU_LO, MU_HI, HU_WINDOW
from .vif import vif_map_3d, vif_scalar_3d, vif_star_map_3d

__all__ = [
    "ThiesConeRecon", "VolumeGrid", "to_unit", "from_unit", "MU_LO", "MU_HI", "HU_WINDOW",
    "ThiesSplineMotion", "akima_resample",
    "QualityMetricUNet3D", "THIES_F_MAPS",
    "vif_map_3d", "vif_scalar_3d", "vif_star_map_3d",
]
