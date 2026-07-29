#!/usr/bin/env bash
# The three fronts the user asked to run (2026-07-26), in VALUE order so a truncated night still
# lands the important one first.
#
#   C  THIES-MATCHED EVALUATION  -- the honest headline. Every number we have compared to Thies'
#      SSIM 0.94 was produced on an EASIER problem: our default motion is `mixed` at
#      (3,3,2) mm / (1.5,1.5,2) deg, while Thies simulates AKIMA splines at 5 mm / 5 deg
#      (measured peaks: ours 3.00 mm / 2.00 deg, his 5.23 mm / 5.50 deg). Same deployed config,
#      only the simulated motion changes -> the first genuinely comparable number.
#
#   B  ADMM-TV  -- the standard form (Boyd Sec. 6.4.1 = DDS's own 3D solver), replacing the
#      kappa-blend whose KM/averaged-operator justification is vacuous for our denoiser.
#      Threshold is UNTUNED, so val 0 gets a 3-point sweep BEFORE spending three patients.
#      rho default 4.3e4 balances the CG blocks (||A^T A||=5.04e5, ||D^T D||=11.7, measured).
#
#   A  --asd  -- ASD-POCS's adaptive TV coupling, already implemented with Sidky & Pan's own
#      constants. Free, citable, and it restores exactly the adaptivity our default kappa path
#      deleted (their TV step is slaved to the DATA step's magnitude; ours is frozen).
#
# All runs: deployed config otherwise (cg/fullband/N50/PER50/l2si/theta_avg 1/fp16/defer),
# patients val 0/1/2 with seeds 3/7/11 as in every other sweep.
#   GPU=0 bash scripts/drivers/drive_three_fronts.sh
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE="--ckpt $CK --split val --est_band fullband --loss l2si --n_steps 50 --per 50 \
      --views_per_iter 24 --theta_avg 1"

run () {  # run <out> <args...>
    local out=$1; shift
    if [ -f "data/$out/result.pt" ]; then echo "SKIP $out (done)"; return; fi
    echo "=== $out  $(date) ==="
    $PY scripts/run_posterior3d.py $BASE --out "data/$out" "$@"
}

# ---- C: Thies-matched motion, deployed cg config -------------------------------------------
THIES="--motion_kind akima --trans_mm 5 --rot_deg 5 --dc_op cg --cg_iters 5 --kappa 0.3"
run thies_v0 $THIES --run 0 --seed 3
run thies_v1 $THIES --run 1 --seed 7
run thies_v2 $THIES --run 2 --seed 11

# ---- B1: ADMM threshold sweep on val 0 (our own motion, so it is comparable to tvv_v0_*) ----
ADMM="--dc_op admm --cg_iters 5 --admm_rho 4.3e4"
run admm_t1e4_v0 $ADMM --admm_thresh 1e-4 --run 0 --seed 3
run admm_t3e4_v0 $ADMM --admm_thresh 3e-4 --run 0 --seed 3
run admm_t1e3_v0 $ADMM --admm_thresh 1e-3 --run 0 --seed 3
# HQS ablation at the middle threshold (dual frozen at zero) -- free, one line
run admm_hqs_v0  $ADMM --admm_thresh 3e-4 --admm_dual 0 --run 0 --seed 3

# ---- A: ASD-POCS adaptive coupling, 3 patients ---------------------------------------------
ASD="--dc_op cg --cg_iters 5 --asd --asd_ng 5"
run asd_v0 $ASD --run 0 --seed 3
run asd_v1 $ASD --run 1 --seed 7
run asd_v2 $ASD --run 2 --seed 11

echo "=== THREE FRONTS DONE  $(date) ==="
