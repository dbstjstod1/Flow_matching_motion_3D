"""Score the demo JRM-ADM run against the paper's Table I convention (OURS, not upstream).

Their metric frame: the GT volume warped to the CENTRAL view's pose (the run script itself
plots exactly `warper.transform(x_true, thetas_gt[middle])`), because the blind problem has an
SE(3) gauge (their own NOTE at the end of run_jrm_adm.py). PSNR/SSIM are computed on the
[-1, 1]-normalized intensity ([-1000, 2000] HU window), data_range = 2.

Also reports the motion-corrupted FDK from the same views (their Table I "FDK" row is this) so
the single-volume smoke test has both ends of the gap: paper says 18.36/0.34 -> 30.71/0.94
(JRM-ADM, n_a=60, n=18 mean). A single demo volume landing in that gap's ballpark is the gate;
an exact match of an 18-patient mean is not expected.

    cd refs/jrm-adm && python score_demo.py
"""
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.physics.physics import RigidWarper
from src.utils.creator_utils import (create_cone_beam_projector, create_volume, extract_ts,
                                     extract_sino_ts)
from src.utils.dataloaders import get_minus_one_one_norm_hu_transform
from src.utils.yaml_config import load_yaml

cfg = load_yaml("config/adm_jrm.yaml")
acq = cfg["aquisition_params"]
dev = torch.device(cfg["common"]["device"])

data = torch.load("data/simulated/gts_and_measurements.pt", map_location=dev, weights_only=False)
rec = torch.load(f"data/recon/adm_jrm_{acq['n_angles']}_view.pt", map_location=dev,
                 weights_only=False)

x_true = data["x_true"]                                   # attenuation, (1,1,D,H,W)
thetas_gt = extract_ts(data["thetas_gt"], acq["n_angles"])
middle = int(thetas_gt.shape[0] // 2)

warper = RigidWarper()
gt_frame = warper.transform(x_true, thetas_gt[middle:middle + 1])   # GT in the eval frame

x_est = rec["x_est"]                                      # already attenuation (run script)


def to_norm(att):
    """attenuation -> the [-1,1] intensity the paper's metrics live in."""
    mu_water = 0.0193
    hu = (att / mu_water - 1.0) * 1000.0
    return get_minus_one_one_norm_hu_transform()(hu)


def fdk_motion_corrupted():
    """FDK of the motion-corrupted measurements, same subsampled views (Table I's FDK row).
    torch-radon fork API differs across versions; try the known spellings."""
    angles = torch.linspace(0, 2 * torch.pi, acq["n_base_angles"])[
        torch.arange(0, acq["n_base_angles"],
                     acq["n_base_angles"] // acq["n_angles"])].to(dev)
    volume = create_volume(size=acq["volume_size"])
    radon = create_cone_beam_projector(angles=angles, volume=volume,
                                       det_count_u=acq["det_count_u"],
                                       det_count_v=acq["det_count_v"],
                                       det_spacing_u=acq["det_spacing_u"],
                                       det_spacing_v=acq["det_spacing_v"],
                                       src_dist=acq["src_dist"], det_dist=acq["det_dist"])
    b = extract_sino_ts(data["b"], acq["n_angles"]).float()
    with torch.no_grad():
        filt = radon.filter_sinogram(b)
        for name in ("backprojection", "backward"):
            fn = getattr(radon, name, None)
            if fn is not None:
                return fn(filt)
    raise AttributeError("no backprojection method found on ConeBeam")


ref = to_norm(gt_frame)[0, 0].cpu().numpy()
rows = []
try:
    rows.append((f"FDK({acq['n_angles']}v, motion)", to_norm(fdk_motion_corrupted())[0, 0].cpu().numpy()))
except Exception as e:                                    # noqa: BLE001 -- the FDK row is optional
    print(f"[warn] FDK row skipped: {e}")
rows.append(("JRM-ADM x_est", to_norm(x_est)[0, 0].cpu().numpy()))

for name, vol in rows:
    psnr = peak_signal_noise_ratio(ref, vol, data_range=2.0)
    ssim = structural_similarity(ref, vol, data_range=2.0)
    print(f"{name:22s}  PSNR {psnr:6.2f} dB   SSIM {ssim:.3f}   [paper Table I n=18: "
          f"FDK 18.36/0.34, JRM-ADM 30.71/0.94]")
