"""Run the JRM-ADM solver on OUR cohort measurements (A2c stage 2, 2026-08-17). OURS, not
upstream -- a re-plumbed run_jrm_adm.py that swaps their simulated demo data for the exchange
files scripts/export_cohort_for_jrm.py writes, keeping their SOLVER STACK byte-identical
(sampler, optimizers, motion model, wavelet prior; the hardcoded lr/gamma schedule included).

Protocol deltas vs their paper, all stated:
  * measurements = OUR full-view protocol (360 views, 500x700 @ 0.64 mm detector, noiseless,
    Akima 10/10 p2p motion, test30 cohort) -- the whole point of the port;
  * angles come from the exchange file (= our betas + 90 deg, the measured operator bridge);
  * the prior = weights_retrain/model_state_dict.pth, retrained on OUR train split (config
    model_path, switched 2026-08-17);
  * recon grid stays THEIR 160x192x192 @ 1 mm (the retrained prior's native grid); our head
    fits the 192 mm in-plane FOV per the export-crop measurements;
  * angle_batch_size scales the WLS chunking to 360 views (20 -> 36 divides 360).

result.pt per patient: x_est (attenuation, their grid/frame), thetas_est, theta_true (copied
through for the cmp pairing assertion), runtime.

    cd refs/jrm-adm && python run_on_ours.py --case data/ours_cohort/p00.pt --out data/recon_ours
"""
import argparse
import os
import time

import torch

from src.optim.data_fidelity_grad import JRMDataFidelity
from src.optim.rmsprop_motion_estimation import RMSpropMotionEstimator
from src.optim.rmsprop_x_estimation import RMSpropXEstimator
from src.models.rigid_motion_theta import RigidMotionTheta
from src.physics.composed_physics import ExtractorRadonRigidWarper
from src.physics.physics import Extractor, RigidWarper, Wavelet
from src.sampler.adaptative_diffusion_sampler import AdaptativeDiffusionSampler
from src.sampler.diffusion_utils import OneStepDDIMSampler
from src.utils.creator_utils import (create_cone_beam_projector, create_initial_thetas,
                                     create_model, create_volume)
from src.utils.dataloaders import convert_m1_p1_to_attenuation
from src.utils.yaml_config import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="an exchange .pt from export_cohort_for_jrm")
    ap.add_argument("--out", default="data/recon_ours")
    ap.add_argument("--angle_batch", type=int, default=36, help="must divide the view count")
    ap.add_argument("--vol", type=int, nargs=3, default=None,
                    help="recon grid override (D H W). The p00 gate showed their 160x192x192 "
                         "TRUNCATES our protocol's measured region (our detector covers ~209 mm "
                         "axially at iso and the CT table sits outside 192 mm in-plane) -> "
                         "bright-rim artifacts. (208,224,224) covers the head+table barrel.")
    ap.add_argument("--gamma", type=float, default=None,
                    help="override config gamma. Their WLS data term SUMS per-chunk losses, so "
                         "at 360 views/angle_batch 36 it is 10 chunks vs their 3 -- the "
                         "data:prior ratio shifts by ~10/3. gamma 3.3e4 restores their tuning "
                         "balance (RMSprop self-normalizes the lr against grad scale, so only "
                         "the RATIO needs fixing).")
    ap.add_argument("--prior_zflip", action="store_true",
                    help="z-flip the volume around the prior call. Our retrained W3DM saw "
                         "superior-first heads (their orientation); the solver reconstructs in "
                         "our measurement frame (inferior-first). This aligns the prior's "
                         "anatomy without touching the data term.")
    ap.add_argument("--model_path", default=None,
                    help="override config model_path (control runs with their RELEASED weights)")
    args = ap.parse_args()

    cfg = load_yaml("config/adm_jrm.yaml")
    device = torch.device(cfg["common"]["device"])
    run_cfg, dcfg = cfg["adm_jrm"], cfg["diffusion"]
    acq = cfg["aquisition_params"]

    case = torch.load(args.case, map_location=device, weights_only=False)
    b, yi, angles = case["b"], case["yi"], case["angles"].to(device)
    V = angles.shape[0]
    assert V % args.angle_batch == 0, f"angle_batch {args.angle_batch} !| {V}"
    det = case["det"]
    tag = os.path.splitext(os.path.basename(args.case))[0]
    os.makedirs(args.out, exist_ok=True)

    model = create_model()
    model.to(device)
    if args.model_path:
        run_cfg["model_path"] = args.model_path
    model.load_state_dict(torch.load(run_cfg["model_path"], weights_only=True))
    model.eval()

    vol_size = list(args.vol) if args.vol else acq["volume_size"]
    gamma = float(args.gamma) if args.gamma is not None else run_cfg["gamma"]
    volume = create_volume(size=vol_size)
    conebeam = create_cone_beam_projector(
        angles=angles, volume=volume,
        det_count_u=det["nu"], det_count_v=det["nv"],
        det_spacing_u=det["du"], det_spacing_v=det["dv"],
        src_dist=det["src_dist"], det_dist=det["det_dist"])
    physics = ExtractorRadonRigidWarper(extractor=Extractor(), radon=conebeam,
                                        warper=RigidWarper())
    wavelet = Wavelet()

    data_fidelity = JRMDataFidelity(physics=physics, yi=yi, angles=angles,
                                    angle_batch_size=args.angle_batch)
    motion_gen = RigidMotionTheta(n_control_points=acq["n_control_points"],
                                  n_angles=V, n_base_angles=V)
    motion_gen.train()
    motion_solver = RMSpropMotionEstimator(physics=physics, motion_pattern_generator=motion_gen,
                                           angles=angles, lr=run_cfg["motion_lr"], yi=yi,
                                           angle_batch_size=args.angle_batch)
    x_solver = RMSpropXEstimator(data_fidelity=data_fidelity, lr=run_cfg["x_lr"])
    one_step = OneStepDDIMSampler(beta_start=dcfg["beta_start"], beta_end=dcfg["beta_end"],
                                  T=dcfg["diffusion_timesteps"], steps=dcfg["steps"],
                                  eta=dcfg["eta"], device=device)
    sampler = AdaptativeDiffusionSampler(model=model, operator=wavelet,
                                         data_fidelity=data_fidelity, x_solver=x_solver,
                                         motion_solver=motion_solver, one_step_sampler=one_step)
    if args.prior_zflip:
        # align the prior's learned orientation with the measurement frame: flip z entering
        # the net, flip back leaving it. Wavelet transform commutes with the flip only up to
        # band bookkeeping, so flip in IMAGE space on both sides of the whole inference.
        import types
        base_inference = sampler.model_inference

        def flipped(self, xt, t):
            img = self.operator.transposed_transform(xt)
            xt_f = self.operator.transform(torch.flip(img, dims=(2,)))
            x0_f = base_inference(xt_f, t)
            img0 = self.operator.transposed_transform(x0_f)
            return self.operator.transform(torch.flip(img0, dims=(2,)))

        sampler.model_inference = types.MethodType(flipped, sampler)

    D, H, W = vol_size
    xt = wavelet.transform(torch.randn(1, 1, D, H, W, device=device))
    thetas0 = create_initial_thetas(num_angles=V).to(device)

    t0 = time.time()
    x_est, thetas_est, cps_est, motion_full = sampler.sample(
        xt=xt, thetas=thetas0, b=b, gamma=gamma, root_plot=None)
    dt = time.time() - t0
    x_est = convert_m1_p1_to_attenuation()(x_est)

    out_p = os.path.join(args.out, f"{tag}_result.pt")
    torch.save({"x_est": x_est.cpu(), "thetas_est": thetas_est.cpu(),
                "cps_est": cps_est.cpu(), "motion_pattern_full_est": motion_full.cpu(),
                "theta_true": case["theta_true"], "runtime_sec": dt,
                "model_path": run_cfg["model_path"], "case": args.case,
                "port_args": {"vol": vol_size, "gamma": gamma,
                              "prior_zflip": bool(args.prior_zflip),
                              "angle_batch": args.angle_batch}}, out_p)
    print(f"{tag}: done in {dt / 60:.1f} min -> {out_p}")


if __name__ == "__main__":
    main()
