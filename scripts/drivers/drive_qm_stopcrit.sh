#!/bin/bash
# >>> 2026-08-06: EVERYTHING THIS SCRIPT ONCE MEASURED IS VOID (the training-pair bug,
# >>> PROVENANCE 4.9) and its results were deleted. The rationale below survives; the specific
# >>> checkpoints/val numbers in it do not. The corrected run (logs/bench_thies_qm2) now logs an
# >>> in-training RPE probe to tensorboard on the SAME (val patient i, seed 2000+i) instances
# >>> (train_qm.py --rpe_every), so run this only if you need a full-fidelity stage-2 check
# >>> (final 256^3 recon + image metrics) beyond the TB curve. Update the `it` list first.
# DOES THE QUALITY NET'S VALIDATION L1 STILL BUY RPE?  -- the stopping criterion, measured.
#
# WHY THIS EXISTS. The stage-1 validation curve cannot tell us when to stop. Every training
# sample is a FRESH motion draw (bench_thies_train_qm.py L201 passes a generator into
# akima_motion), so the training loss is already an online held-out estimate along the motion
# axis and only the 150-patient axis can overfit. Measured (2026-08-06): that gap opened once
# between iter 1500 and 3000 (0 -> +0.005) and has been statistically FLAT since (|t| <= 1.0 on
# every recent window). A constant gap means val descends in lock-step with train forever, so
# "val turned up" will never fire and the budget would be decided by nothing but the clock.
#
# So decide it on the metric that matters. Val L1 is a REGRESSION score on VIF*; what the
# benchmark actually consumes is the GRADIENT LANDSCAPE that L1 induces for the 100-step GD, and
# those two are not the same object -- a net can predict VIF* better on average while leaving the
# descent direction unchanged. RPE is the paper's own headline metric and is what an extra 3000
# iterations has to move to be worth 13 hours.
#
# THE DESIGN
#   checkpoints  2000 / 4000 / 7000     val L1 0.0854 / 0.0714 / 0.0658
#     2000 is the CONTROL: if RPE does not separate 2000 from 7000, then RPE is insensitive to
#     this net over this range and the experiment says nothing about stopping -- read that
#     outcome as "inconclusive", NOT as "converged".
#     4000 -> 7000 is the QUESTION: a 7.7% val-L1 gain, about half of what the remaining 3000
#     iterations are projected to buy. Flat there means flat ahead.
#   patients     the VAL split, run 0..4, seed 2000+i
#     THE VAL SPLIT ON PURPOSE. This chooses a training budget, so it must not touch test. The
#     seed base is 2000 to keep these runs from ever being confused with the 30-patient test
#     cohort, which is seed 1000+i.
#   PAIRED. Every checkpoint sees the identical (patient, seed), so the checkpoint-to-checkpoint
#     difference is free of the patient-to-patient RPE spread (~0.5-1 mm) that would otherwise
#     swamp it at n=5. Compare the paired differences, never the raw means.
#
# ORDER: patient-major, so a complete paired triple exists after ~15 min instead of ~60.
#
# COST: 15 runs. ~5 min each while stage 1 has the GPU (1.76 s/it measured under contention vs
# ~0.8 s/it alone), so ~75 min, and it slows stage 1 by roughly the same amount. That is the
# trade being made: ~1 h now to decide whether the remaining ~13 h are worth spending.
#
#     setsid nohup bash scripts/drivers/drive_qm_stopcrit.sh </dev/null > logs/qm_stopcrit/drive.log 2>&1 &
set -u
cd /home/mirlab/Desktop/Flow_matching_motion_3D
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
export CUDA_VISIBLE_DEVICES=0
mkdir -p data/qm_stopcrit logs/qm_stopcrit

for i in 0 1 2 3 4; do
  for it in 2000 4000 7000; do
    tag=$(printf "it%d_v%d" "$it" "$i")
    if [ -f "data/qm_stopcrit/$tag/result.json" ]; then
      echo "##### $tag :: already done, skipping #####"
      continue
    fi
    echo "##### $tag :: run=$i seed=$((2000 + i)) :: $(date '+%F %T') #####"
    $PY scripts/bench_thies_estimate.py \
        --qm "logs/bench_thies_qm2/qmnet_iter$(printf '%06d' "$it").pth" \
        --split val --run "$i" --seed $((2000 + i)) --montage_every 0 \
        --out "data/qm_stopcrit/$tag" \
        > "logs/qm_stopcrit/$tag.log" 2>&1 \
      || echo "##### $tag :: FAILED (see logs/qm_stopcrit/$tag.log) #####"
  done
done
echo "===== stop-criterion sweep done :: $(date '+%F %T') ====="
echo "   read it with: python scripts/cmp_qm_stopcrit.py"
