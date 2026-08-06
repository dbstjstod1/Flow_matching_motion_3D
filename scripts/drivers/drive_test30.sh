#!/bin/bash
# THE EVALUATION COHORT: 30 patients from the TEST split, our deployed inference config.
# 2026-08-03. GPU0 only, strictly sequential.
#
# WHY 30 AND WHY test: this is Thies' own evaluation protocol, verbatim (TMI L500-503):
#   "All motion compensation experiments are performed on the same 30 patients from the test set.
#    A random motion pattern is sampled for each patient ... which is kept constant across
#    different methods and optimization algorithms."
# The test split is 120 patients; they score 30 of them. So do we.
#
# EVERYTHING IS THE DEPLOYED DEFAULT except ONE knob:
#   --seed $((1000 + i))   per-patient motion, the repo's standing convention (val_fm3d.py:81,
#                          "the fixed seed (1000 + i) means the SAME motion is scored at every
#                          checkpoint"). The script's own default --seed 3 would hand all 30
#                          patients the IDENTICAL trajectory, which is not a 30-patient cohort
#                          and is not what Thies does.
# Defaults inherited (do not re-specify, so this run tracks whatever the deployed config is):
#   --per 200 (adopted 2026-08-03), --est_coarse 2 (c2f), --dc_op cg, 10 mm / 10 deg p2p.
#
# THE SEEDS ARE THE CONTRACT. scripts/bench_thies_estimate.py must be run over the same
# (split, run, seed) triples so the two methods see byte-identical sinograms -- that is what
# `run_posterior3d.build_world` guarantees and what makes the comparison paired.
#
# Cost: ~9.3 min/patient at PER 200 -> ~4.7 h for 30. Output ~420 MB/patient -> ~13 GB.
cd /home/mirlab/Desktop/Flow_matching_motion_3D
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
export CUDA_VISIBLE_DEVICES=0
CK=logs/fm3d_cq500_leap/ckpt_iter500000.pth
mkdir -p logs/test30 data/test30

for i in $(seq 0 29); do
  tag=$(printf "p%02d" "$i")
  if [ -f "data/test30/$tag/result.pt" ]; then
    echo "##### $tag :: already done, skipping #####"
    continue
  fi
  echo "##### $tag :: split=test run=$i seed=$((1000 + i)) :: $(date +%H:%M:%S) #####"
  $PY scripts/run_posterior3d.py --ckpt $CK --split test --run "$i" \
      --seed $((1000 + i)) --out "data/test30/$tag" \
      > "logs/test30/$tag.log" 2>&1
  rc=$?
  echo "--- $tag exit $rc ---"
  grep -E "FINAL|^step  49" "logs/test30/$tag.log" | tail -4
done
echo "TEST30 COMPLETE :: $(date)"
