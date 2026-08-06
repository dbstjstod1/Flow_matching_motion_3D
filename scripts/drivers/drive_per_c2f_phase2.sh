#!/bin/bash
# Phase 2 of the 2026-08-03 PER/coarse ablation. GPU0 ONLY, sequential.
# Phase 1 showed the two levers are ORTHOGONAL and were only ever measured separately:
#   coarse-all (per400) 510s @ x_t 39.20   |   c2f per200 557s @ 39.30   |   c2f per100 389s @ 38.46
# so their product is the untested optimum. `noise` is an IDENTICAL rerun of coarse-all/per400 --
# the loop is not bit-reproducible (Triton/cuDNN atomics), so this arm sizes the run-to-run noise
# that every 0.2-0.3 dB gap in the phase-1 table has to be read against.
cd /home/mirlab/Desktop/Flow_matching_motion_3D
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
export CUDA_VISIBLE_DEVICES=0
CK=logs/fm3d_cq500_leap/ckpt_iter500000.pth
run () { tag=$1; shift
  echo "##### $tag :: $* #####"
  $PY scripts/run_posterior3d.py --ckpt $CK --out data/persweep_$tag --run 0 "$@" \
      > logs/persweep_$tag.log 2>&1
  echo "--- $tag done ---"; }
run ca_per200 --est_coarse_until 1.0 --per 200
run ca_per100 --est_coarse_until 1.0 --per 100
run ca_per150 --est_coarse_until 1.0 --per 150
run noise     --est_coarse_until 1.0 --per 400
echo "PHASE2 COMPLETE"
