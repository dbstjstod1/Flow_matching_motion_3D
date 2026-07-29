#!/usr/bin/env bash
# TV COUPLING FORM at akima 5 mm / 5 deg, operator only (user, 2026-07-28).
#
# WHAT ACTUALLY DIFFERS. All three run the SAME CG data step; what changes is how the TV prior is
# coupled to it:
#     cg  (deployed)  fixed kappa blend      z <- z + kappa (TV(z) - z),  kappa 0.3, step 0.015 x5
#     admm            exact TV prox + dual   the standard ADMM-TV form (Boyd 6.4.1 = DDS's own 3D
#                                            solver). It SUBSUMES the corrector, so --kappa is
#                                            ignored and the strength is set by --admm_thresh.
#     cg --asd        ASD-POCS adaptive      the TV step is slaved to the DATA step's magnitude
#                                            (Sidky & Pan), not an independent constant.
#
# THE CONFOUND THIS SCOUT EXISTS TO KILL. ADMM parameterizes TV strength completely differently, so
# picking a threshold blind would measure a TV-STRENGTH difference and call it an operator one --
# exactly what the user asked to avoid. `d_tv` cannot arbitrate: with ADMM the prox happens INSIDE
# the data step, so it reads 0. The new `tv_rel` = ||grad z||_1 / ||grad gt||_1 measures the
# OUTCOME (1.0 = as much gradient energy as the truth, < 1 = over-smoothed) and is defined
# identically for all three. This stage runs 6 steps of each and reports it; the full runs then use
# the ADMM threshold whose tv_rel matches the deployed kappa blend.
#
# Everything else is the recommended operating point (per 400, coarse->fine at t=0.5, lr 3e-3).
#
#   GPU=0 setsid nohup bash scripts/drivers/drive_dcop_55.sh > logs/dcop55_scout.log 2>&1 &
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/tmp/claude-1000/-home-mirlab-Desktop-Flow-matching-motion-3D/029dc143-f83f-43eb-ae42-ff33ba52d5c1/scratchpad
BASE="--ckpt $CK --split val --run 0 --seed 3 --n_steps 6 --est_band fullband --views_per_iter 24 \
      --theta_avg 1 --loss l2si --lr 3e-3 --per 400 --est_coarse 2 --est_coarse_until 0.5 \
      --cg_iters 5"

echo "=== TV-strength scout (6 steps each; read the 'tv' column)  $(date) ==="
echo "--- cg + kappa 0.3 (DEPLOYED -- the target tv_rel) ---"
$PY scripts/run_posterior3d.py $BASE --dc_op cg   --kappa 0.3 --out $S/sc_cg   2>&1 | grep -E "^step"
echo "--- cg + asd (adaptive) ---"
$PY scripts/run_posterior3d.py $BASE --dc_op cg   --asd       --out $S/sc_asd  2>&1 | grep -E "^step"
for th in 1e-4 3e-4 1e-3; do
    echo "--- admm thresh $th ---"
    $PY scripts/run_posterior3d.py $BASE --dc_op admm --admm_thresh $th --out $S/sc_adm$th 2>&1 | grep -E "^step"
done
echo "=== SCOUT DONE  $(date) ==="
