"""Gate the JRM-ADM -> ours motion conversion (scripts/jrm_theta_convert.py) END-TO-END.

The one check that pins the SEMANTICS (not just the algebra): their own data term and ours
must produce the same predicted sinogram views from the same volume under the same motion.

  stage theirs  (jrm_adm env, run from refs/jrm-adm): rebuild the port's physics exactly as
                run_on_ours.py does, take p00's final x_est and thetas_est, and compute their
                y_pred = Extractor(A_static,k(Warp(theta_k) x)) at PROBE views.
  stage ours    (flow_matching env, repo root): convert thetas_est with jrm_thetas_to_ours,
                project THE SAME volume with our LEAP forward at P_nom @ T(theta_conv) on the
                same views, and compare.

Bar: rel <= 2e-2 per view. Composition of the measured operator parity (1.6e-3) and their
warp's trilinear resampling (absent on our side -- we move geometry exactly), so a few
percent is expected; a SIGN or axis error is a >50% miss, which is what this gate exists to
catch.

    cd refs/jrm-adm && python ../../scripts/gate_jrm_theta_convert.py --stage theirs
    python scripts/gate_jrm_theta_convert.py --stage ours
"""
import argparse
import os
import sys

import numpy as np
import torch

SCRATCH = os.environ.get(
    "XCHECK_DIR",
    "/tmp/claude-1000/-home-mirlab-Desktop-Flow-matching-motion-3D/"
    "072ea6ff-0ebd-4544-af78-9c46bc79aeca/scratchpad/jrm_theta_gate")
PROBE = [0, 90, 180, 270]


def stage_theirs():
    sys.path.insert(0, os.getcwd())
    from src.physics.composed_physics import ExtractorRadonRigidWarper
    from src.physics.physics import Extractor, RigidWarper
    from src.utils.creator_utils import create_cone_beam_projector, create_volume

    os.makedirs(SCRATCH, exist_ok=True)
    case = torch.load("data/ours_cohort/p00.pt", map_location="cuda", weights_only=False)
    res = torch.load("data/recon_ours_v2/p00_result.pt", map_location="cuda",
                     weights_only=False)
    x = res["x_est"].cuda()                                  # (1,1,224,224,224) attenuation
    thetas = res["thetas_est"].cuda()                        # (V,3,4) raw affine
    angles = case["angles"].cuda()
    det = case["det"]
    ys = []
    for k in PROBE:
        radon = create_cone_beam_projector(
            angles=angles[k:k + 1], volume=create_volume(size=list(x.shape[2:])),
            det_count_u=det["nu"], det_count_v=det["nv"],
            det_spacing_u=det["du"], det_spacing_v=det["dv"],
            src_dist=det["src_dist"], det_dist=det["det_dist"])
        physics = ExtractorRadonRigidWarper(extractor=Extractor(), radon=radon,
                                            warper=RigidWarper())
        with torch.no_grad():
            ys.append(physics.transform(x, thetas[k:k + 1], angles[k:k + 1])[0].cpu())
    np.save(os.path.join(SCRATCH, "x.npy"), x[0, 0].cpu().numpy())
    np.save(os.path.join(SCRATCH, "thetas.npy"), thetas.cpu().numpy())
    np.save(os.path.join(SCRATCH, "y_theirs.npy"),
            torch.stack([y.reshape(det["nv"], det["nu"]) for y in ys]).numpy())
    print("stage theirs done ->", SCRATCH)


def stage_ours():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "scripts"))
    from fm3d.geometry_3d import (ConeBeam3DConfig, build_conebeam_orbit,
                                  detector_coords_3d)
    from fm3d.projector_3d import forward_project_3d_batched
    from fm3d.rigid_motion import params_to_Pmot
    from jrm_theta_convert import jrm_thetas_to_ours

    x = torch.from_numpy(np.load(os.path.join(SCRATCH, "x.npy"))).float().cuda()
    thetas = torch.from_numpy(np.load(os.path.join(SCRATCH, "thetas.npy"))).float().cuda()
    y_th = torch.from_numpy(np.load(os.path.join(SCRATCH, "y_theirs.npy"))).float().cuda()
    cfg = ConeBeam3DConfig.thies(n_views=360)
    P_nom = build_conebeam_orbit(cfg, device="cuda")
    u, v = detector_coords_3d(cfg, device="cuda")
    theta_conv = jrm_thetas_to_ours(thetas)
    P = params_to_Pmot(theta_conv, P_nom)
    n_fail = 0
    for j, k in enumerate(PROBE):
        with torch.no_grad():
            y = forward_project_3d_batched(x[None, None], P[k:k + 1][None], u, v,
                                           dx=1.0, dy=1.0, dz=1.0)[0, 0]
        rel = float(torch.linalg.vector_norm(y - y_th[j])
                    / (torch.linalg.vector_norm(y_th[j]) + 1e-12))
        ok = rel < 2e-2
        n_fail += (not ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] view {k:3d}: rel {rel:.4f}   "
              f"theta_conv t=({theta_conv[k,0]:+.2f},{theta_conv[k,1]:+.2f},"
              f"{theta_conv[k,2]:+.2f})mm")
    print("ALL GATES PASS" if n_fail == 0 else f"{n_fail} GATE(S) FAILED")
    return 1 if n_fail else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["theirs", "ours"])
    a = ap.parse_args()
    raise SystemExit(stage_theirs() if a.stage == "theirs" else stage_ours())
