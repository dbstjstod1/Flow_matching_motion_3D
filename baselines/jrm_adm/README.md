# JRM-ADM on the shared CQ500 cohort

Use [JRM-ADM](https://github.com/antoinedepaepe/jrm-adm) commit
`4552c7123aaa38848d75071eb0e0028b0206e052` in its own environment, following the
upstream instructions for `carterbox-torch-radon` and Python dependencies.
This directory supplies our CQ500 export/training/solver adapters; the upstream
sampler, optimizers and network are obtained from that pinned checkout.

From this repository root:

```bash
git clone https://github.com/antoinedepaepe/jrm-adm refs/jrm-adm
git -C refs/jrm-adm checkout 4552c7123aaa38848d75071eb0e0028b0206e052
cp baselines/jrm_adm/{run_on_ours.py,run_cohort.sh,train_w3dm.py} refs/jrm-adm/
```

Export with the **flow-matching environment**:

```bash
python scripts/export_w3dm_train.py --root data/CQ500 --split train \
    --out refs/jrm-adm/data/train_volumes
python scripts/export_cohort_for_jrm.py --root data/CQ500 \
    --ckpt logs/fm3d_databridge/ckpt_iter500000.pth --n 30 \
    --out refs/jrm-adm/data/ours_cohort
```

Training exports are the original vertex-anchored 160×192×192 crops in raw HU,
with superior-first orientation. Test exports preserve the shared measurements and
motion trajectories; ground-truth poses are carried for scoring and are not passed
to the motion optimizer as estimates.

In the **JRM environment**, run from `refs/jrm-adm`:

```bash
python train_w3dm.py --data data/train_volumes --out weights_retrain \
    --iters 300000 --batch 1 --lr .0001 --ema .999 --opt adamw --seed 0
MODEL_PATH=weights_retrain/model_state_dict.pth CUDA_VISIBLE_DEVICES=0 bash run_cohort.sh
```

These commands train in fp32 without augmentation and run the 100-step sampler on
224³ volumes, with angle batches of 36, gamma 33000 and a z-flip around prior calls.
`MODEL_PATH` explicitly selects the retrained EMA weights. `PYTHON`, `CASE_DIR` and
`RESULT_DIR` override the interpreter or directories. The driver stops if a result
already exists; use a fresh `RESULT_DIR` for another run.

New runs record an inference seed of `1000+i` per case. The archived comparison did
not record its random image-initialization seed; matching the protocol does not
promise bitwise reproduction of those archived volumes.

Back at this repository root, in the **flow-matching environment**:

```bash
python scripts/score_jrm_native.py --root data/CQ500 \
    --ckpt logs/fm3d_databridge/ckpt_iter500000.pth --ours data/test30 \
    --jrm refs/jrm-adm/data/recon_ours_v2 --out data/jrm_scores.json
```

Set `--ours` to your proposed-method cohort directory. The scorer checks paired
trajectories, converts the attenuation scale and affine-motion convention, and
reports the native output and FDK at the recovered poses under the common
measured-FOV metric. [PROVENANCE.md](PROVENANCE.md) lists the adaptation details.
