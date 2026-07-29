#!/usr/bin/env bash
# OVERNIGHT CHAIN (user, 2026-07-25): everything in the optimization plan except the baselines.
#
# BASE CONFIG, fixed for every run below unless the run is the one varying it:
#   --dc_op cg --cg_iters 5   the fair 3-way winner
#   --est_band fullband       (auto-pairs lr 1e-3; bandwidth and lr are ONE knob)
#   --kappa 0                 TV off = the configuration the data-step verdict was reached at
#   --loss l2si  --n_steps 50  --per 50  --views_per_iter 24
#   --theta_avg 1             <- STAGE 0 RESULT, and a CHANGE from the shipped default of 2.
#     With the fullband estimator the cg tail DESCENDS monotonically (2/10 sign flips; measured
#     offline on data/fair_cg) instead of oscillating the way it did under hashbl+1e-2 (9/10 on
#     data/cg_v0). Averaging a still-descending tail only mixes in stale, worse thetas: K=1 0.102
#     deg, K=2 0.105, K=5 0.112, K=8 0.120 -- monotonically WORSE. The better estimator removed
#     the landing lottery that theta-averaging existed to fix. K=1 is also the oracle best-step
#     here (0.102 @ step 49), so no selector could beat it either.
#
# ORDER IS BY VALUE, NOT BY PLAN NUMBER, because a night can be cut short:
#   [A] multi-patient validity   -- everything since noon rests on val 0 / seed 3 alone, and val 1
#                                   has ALREADY reversed a verdict once (cg's theta was worse than
#                                   adj's there under the old estimator). Highest value.
#   [B] ablations at fullband    -- kappa / loss / cg_iters were all decided under hashbl+1e-2 and
#                                   are formally unvalidated at the estimator we now ship.
#   [C] N sweep                  -- pure efficiency: theta is already < 0.5 deg by step 14, so the
#                                   last 36 steps may be buying little.
#
# [B] DELIBERATELY RUNS AT N=50 rather than at whatever [C] finds, so its arms stay directly
# comparable to data/fair_cg. Folding an unvalidated N into the ablations would confound them.
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE="--ckpt $CK --split val --loss l2si --kappa 0 --dc_op cg --cg_iters 5 \
      --est_band fullband --theta_avg 1"

run () {   # run <outdir> <extra args...>
    local out=$1; shift
    if [ -f "data/$out/result.pt" ]; then echo "SKIP $out (already done)"; return; fi
    echo "=== $out  $(date) ==="
    $PY scripts/run_posterior3d.py $BASE --out "data/$out" "$@"
}

# ---- [A] multi-patient validity of the shipped config -------------------------------------
# val 0 already exists as data/fair_cg (same settings bar --theta_avg, which only changes the
# READOUT and is re-derivable offline from theta_hist).
run night_A_v1  --run 1 --seed 7
run night_A_v2  --run 2 --seed 11

# ---- [B] ablations at the fullband estimator, val 0, one axis each ------------------------
run night_B_kappa03  --run 0 --seed 3 --kappa 0.3      # TV on: the 2D winning triple
run night_B_lncc     --run 0 --seed 3 --loss lncc      # AI_Geocal's loss (fullband is its design)
run night_B_cgi8     --run 0 --seed 3 --cg_iters 8     # is cg_iters=5 saturated?

# ---- [C] N sweep: is the second half of the ODE buying anything? --------------------------
run night_C_n30  --run 0 --seed 3 --n_steps 30
run night_C_n20  --run 0 --seed 3 --n_steps 20

echo "=== NIGHT CHAIN DONE  $(date) ==="
