#!/bin/bash
# THE THIES BASELINE ON THE SAME 30 PATIENTS OUR METHOD WAS SCORED ON -- at BOTH amplitudes.
#
# GPU 0 ONLY, strictly sequential (user, 2026-08-05). GPU 1 belongs to the 4DCT project on this
# box, and a shard there also made every `est_seconds` a contended timing.
#
# PASS 1 -- 10 mm / 10 deg peak-to-peak, into data/bench_thies_test30/
#   THE PAIRED COMPARISON. The only thing that makes it paired is that the triples match
#   scripts/drivers/drive_test30.sh exactly:
#       --split test --run $i --seed $((1000 + i))          i = 0 .. 29
#   `bench_thies_estimate.build_world` IS `run_posterior3d.build_world`, so with the same triple
#   both methods get the byte-identical patient, geometry, motion draw and simulated sinogram.
#   Change a seed and the pairing is silently gone -- the runs still finish and still print
#   plausible numbers. `scripts/cmp_thies_vs_ours.py` asserts theta_true matches, and refuses.
#
# PASS 2 -- 5 mm / 5 deg peak-to-peak (--thies_amp), into data/bench_thies_test30_amp55/
#   NOT A COMPARISON WITH US, and it must never be reported as one: our own 30 runs are at 10/10,
#   and a different amplitude means a different motion draw, so there is nothing to pair against.
#   It answers a DIFFERENT and necessary question -- **is the baseline weak, or is our
#   reimplementation of it weak?** The paper reports RPE 0.61 mm and SSIM 0.94 at exactly this
#   operating point (TMI IV), so pass 2 is the closest this repo gets to a published number. Read
#   it as "does the reimplementation land in the right RANGE", NOT as a digit-for-digit
#   reproduction: our patient filtering admits 343 scans where theirs admits 320, and the split is
#   sequential by patient index, so these are not the same 30 patients (PROVENANCE section 4.8).
#   The simulation grid, by contrast, is NOT a difference -- see section 4.7 before blaming it.
#
#   (To close the loop the other way one would also need OUR method at 5/5, which has never been
#   run -- see the note in the amplitude ledger. That is a separate decision, not this script's.)
#
# Quality metric: qmnet_best.pth, NOT _last. The paper is silent on checkpoint selection, so we
# select on the validation split it defines (bench/thies/PROVENANCE.md section 3b).
#
# Cost: ~80 s of estimation + ~2 min of world-building / final 256^3 recons / gauge fits per run,
# so ~3.5 min each -> ~3.5 h for both passes on one GPU. Before the fast kernels (PROVENANCE
# section 4.5) the estimation alone was 32 min/patient, i.e. ~32 h for this table.
#
#     setsid nohup bash scripts/drivers/drive_thies_test30.sh </dev/null > log 2>&1 &
set -u
cd /home/mirlab/Desktop/Flow_matching_motion_3D
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
export CUDA_VISIBLE_DEVICES=0
QM=logs/bench_thies_qm2/qmnet_best.pth
mkdir -p logs/bench_thies_test30 logs/bench_thies_test30_amp55 \
         data/bench_thies_test30 data/bench_thies_test30_amp55

run_pass () {                      # $1 = out/log tag, $2.. = extra flags
  local tag_dir="$1"; shift
  echo "===== PASS $tag_dir :: $* :: $(date '+%F %T') ====="
  for i in $(seq 0 29); do
    local tag; tag=$(printf "p%02d" "$i")
    if [ -f "data/$tag_dir/$tag/result.json" ]; then
      echo "##### $tag_dir/$tag :: already done, skipping #####"
      continue
    fi
    echo "##### $tag_dir/$tag :: run=$i seed=$((1000 + i)) :: $(date +%H:%M:%S) #####"
    $PY scripts/bench_thies_estimate.py --qm "$QM" \
        --split test --run "$i" --seed $((1000 + i)) \
        --out "data/$tag_dir/$tag" "$@" \
        > "logs/$tag_dir/$tag.log" 2>&1 \
      || echo "##### $tag_dir/$tag :: FAILED (see logs/$tag_dir/$tag.log) #####"
  done
}

run_pass bench_thies_test30                       # 10/10 p2p -- paired with data/fm3d_test30

# PASS 2 IS OFF BY DEFAULT (user, 2026-08-06). It is NOT part of the head-to-head: our own 30
# runs are at 10/10, and a different amplitude means a different motion draw, so there is nothing
# to pair against. Its value is the separate question in the header -- "is the baseline weak, or
# is our reimplementation of it weak?" -- which costs ~1.7 h whenever it is wanted, and which the
# patient-set difference in PROVENANCE section 4.8 weakens without answering. Turn it on with:
#
#     RUN_AMP55=1 bash scripts/drivers/drive_thies_test30.sh
#
if [ "${RUN_AMP55:-0}" != "0" ]; then
  run_pass bench_thies_test30_amp55 --thies_amp   # 5/5 p2p -- vs the PAPER's own numbers
else
  echo "===== PASS bench_thies_test30_amp55 SKIPPED (RUN_AMP55=0; see the note above) ====="
fi
echo "===== all requested passes done :: $(date '+%F %T') ====="
