"""Calibrate OUR LEAP forward against JRM-ADM's torch-radon ConeBeam (A2c stage 0).

WHY. To run De Paepe's solver on OUR cohort we feed OUR simulated y as their `b`. Their motion
model warps the VOLUME and projects with a STATIC cone-beam (composed_physics: warp -> radon),
so the per-view motion needs no conversion -- it is baked into y by our simulation. What must
match is the STATIC operator: angle origin/direction, detector u/v orientation, and the line-
integral scale. This probe measures the transform that maps our sinogram layout onto theirs.

TWO ENVS, TWO STAGES (LEAP lives in `flow_matching`, torch-radon in `jrm_adm`):

  stage ours    (flow_matching env): project a mu volume with our LEAP forward at THEIR
                geometry sizes (det 700u x 500v @ 0.5 mm -- keep their det so the probe isolates
                CONVENTIONS, not protocol), save volume + sinogram to the scratch dir.
  stage theirs  (jrm_adm env): project the SAME volume with their ConeBeam, then search
                (u-flip) x (v-flip) x (angle direction) x (angle offset 0..359) by correlation
                on view 0, and report the best-aligned relative error over all views + the
                fitted global scale.

    python scripts/diag_jrm_operator_xcheck.py --stage ours     # env: flow_matching
    cd refs/jrm-adm && python ../../scripts/diag_jrm_operator_xcheck.py --stage theirs
"""
import argparse
import os
import sys

import numpy as np
import torch

SCRATCH = os.environ.get(
    "XCHECK_DIR",
    "/tmp/claude-1000/-home-mirlab-Desktop-Flow-matching-motion-3D/"
    "072ea6ff-0ebd-4544-af78-9c46bc79aeca/scratchpad/jrm_xcheck")
N_VIEWS = 60                      # enough to identify offset/direction, cheap
DET_U, DET_V, DU = 700, 500, 0.5  # THEIR detector for the convention probe
SRC, DET = 785.0, 415.0
MU_WATER_OURS = 0.02


def load_mu_volume():
    """p0000 export (their orientation, HU) -> mu with OUR constant, so both stages project
    IDENTICAL values and any scale residual is the operator's, not the window's."""
    hu = np.load("refs/jrm-adm/data/train_volumes/p0000.npy")
    return (hu / 1000.0 + 1.0) * MU_WATER_OURS


def stage_ours():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fm3d.geometry_3d import (ConeBeam3DConfig, build_conebeam_orbit,
                                  detector_coords_3d)
    from fm3d.projector_3d import forward_project_3d_batched

    os.makedirs(SCRATCH, exist_ok=True)
    mu = torch.from_numpy(load_mu_volume()).float().cuda()
    cfg = ConeBeam3DConfig(SOD=SRC, SDD=SRC + DET, det_nu=DET_U, det_nv=DET_V,
                           det_pixel_mm=DU, n_views=N_VIEWS)
    P = build_conebeam_orbit(cfg, device="cuda")
    u, v = detector_coords_3d(cfg, device="cuda")
    y = forward_project_3d_batched(mu[None, None], P[None], u, v, dx=1.0, dy=1.0, dz=1.0)[0]
    np.save(os.path.join(SCRATCH, "mu.npy"), mu.cpu().numpy())
    np.save(os.path.join(SCRATCH, "y_ours.npy"), y.cpu().numpy())
    print("ours:", tuple(y.shape), "range", float(y.min()), float(y.max()),
          "->", SCRATCH)


def stage_theirs():
    sys.path.insert(0, os.getcwd())            # run from refs/jrm-adm
    from src.utils.creator_utils import create_cone_beam_projector, create_volume

    mu = torch.from_numpy(np.load(os.path.join(SCRATCH, "mu.npy"))).float().cuda()
    y_ours = torch.from_numpy(np.load(os.path.join(SCRATCH, "y_ours.npy"))).float().cuda()
    # UNIFORM 2pi/N spacing, matching build_conebeam_orbit. (Their own scripts use
    # linspace(0, 2pi, N) WITH the endpoint -- spacing 2pi/(N-1), first==last view. For the
    # port we pass our angles explicitly, so the probe calibrates the uniform grid.)
    angles = (torch.arange(N_VIEWS) * (2 * torch.pi / N_VIEWS)).cuda()
    radon = create_cone_beam_projector(
        angles=angles, volume=create_volume(size=list(mu.shape)),
        det_count_u=DET_U, det_count_v=DET_V, det_spacing_u=DU, det_spacing_v=DU,
        src_dist=SRC, det_dist=DET)
    with torch.no_grad():
        y_th = radon.transform(mu[None, None].expand(1, 1, *mu.shape))[0, 0]
    # torch-radon layout: (V, v, u) expected -- assert and normalize
    if y_th.shape != (N_VIEWS, DET_V, DET_U):
        y_th = y_th.reshape(N_VIEWS, DET_V, DET_U)
    print("theirs:", tuple(y_th.shape), "range", float(y_th.min()), float(y_th.max()))

    ref0 = y_th[0]
    best = None
    for uflip in (False, True):
        for vflip in (False, True):
            for rev in (False, True):
                cand = y_ours
                if uflip:
                    cand = torch.flip(cand, dims=(2,))
                if vflip:
                    cand = torch.flip(cand, dims=(1,))
                if rev:
                    cand = torch.flip(cand, dims=(0,))
                # correlate their view 0 against every candidate view
                c = torch.stack([torch.nn.functional.cosine_similarity(
                    ref0.flatten(), cand[k].flatten(), dim=0) for k in range(N_VIEWS)])
                k = int(c.argmax())
                score = float(c[k])
                if best is None or score > best["score"]:
                    best = {"uflip": uflip, "vflip": vflip, "rev": rev, "offset": k,
                            "score": score}
    cand = y_ours
    if best["uflip"]:
        cand = torch.flip(cand, dims=(2,))
    if best["vflip"]:
        cand = torch.flip(cand, dims=(1,))
    if best["rev"]:
        cand = torch.flip(cand, dims=(0,))
    cand = torch.roll(cand, -best["offset"], dims=0)
    scale = float((y_th * cand).sum() / (cand * cand).sum())
    rel = float(torch.linalg.vector_norm(y_th - scale * cand)
                / torch.linalg.vector_norm(y_th))
    print(f"BEST map ours->theirs: uflip={best['uflip']} vflip={best['vflip']} "
          f"rev={best['rev']} offset={best['offset']} (view-0 cos {best['score']:.5f})")
    print(f"fitted scale {scale:.5f} | all-view rel error {rel:.4f}")
    print("PASS" if rel < 0.05 and abs(scale - 1.0) < 0.05 else
          "INVESTIGATE (geometry parity not yet established)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["ours", "theirs"])
    a = ap.parse_args()
    stage_ours() if a.stage == "ours" else stage_theirs()
