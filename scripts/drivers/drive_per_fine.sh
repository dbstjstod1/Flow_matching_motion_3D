#!/bin/bash
# 2026-08-03, phase 4: PER convergence on the FINE grid ONLY (--est_coarse 1). User's call:
# the coarse grid is dropped entirely, the deliverable is x_t, and the goal is to remove WASTE
# without conceding quality -- so the question is the largest PER cut whose x_t is
# indistinguishable from per400's, not the best speed/quality trade.
# `fine400b` is an identical rerun of fine/per400: the loop is not bit-reproducible (atomics) and
# the phase-2 probe put the x_t spread at 0.51 dB, so a null result needs its own noise bar.
# Read rot-vs-step, not just the endpoint: on c2f, per100's rot was STILL DESCENDING at step 49
# (budget-limited) while per400 plateaued by step ~20.
cd /home/mirlab/Desktop/Flow_matching_motion_3D
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
export CUDA_VISIBLE_DEVICES=0
CK=logs/fm3d_cq500_leap/ckpt_iter500000.pth
run () { tag=$1; shift
  echo "##### $tag :: $* #####"
  $PY scripts/run_posterior3d.py --ckpt $CK --out data/fine_$tag --run 0 --est_coarse 1 "$@" \
      > logs/fine_$tag.log 2>&1
  echo "--- $tag done ---"; }
run per200 --per 200
run per150 --per 150
run per100 --per 100
run per075 --per 75
run per050 --per 50
run per400b --per 400
echo "PHASE4 COMPLETE"
