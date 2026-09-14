# JRM-ADM baseline adapter

Use [the authors' JRM-ADM repository](https://github.com/antoinedepaepe/jrm-adm) at commit
`4552c7123aaa38848d75071eb0e0028b0206e052` and its separate environment/dependencies.
This directory contains our dataset/training adapters, not vendored upstream solver code.

The current manuscript's archived baseline is the retrained **300,000-iteration** prior
(AdamW, batch 1, learning rate 10⁻⁴, EMA 0.999, seed 0, fp32, no augmentation),
224³ reconstruction, 360 views, angle batch 36, gamma **33000**, with a z-flip around prior
calls. This identifies the evaluated baseline; it does not claim to reproduce every training
choice of the original JRM-ADM paper. Later fine-tuning and released-weight diagnostics are
separate experiments.

From this repository root, clone upstream, install its dependencies in a separate environment,
and copy our adapters:

```bash
git clone https://github.com/antoinedepaepe/jrm-adm refs/jrm-adm
git -C refs/jrm-adm checkout 4552c7123aaa38848d75071eb0e0028b0206e052
cp baselines/jrm_adm/{run_on_ours.py,run_cohort.sh,train_w3dm.py} refs/jrm-adm/
```

Export training volumes and the shared test measurements using the **flow-matching environment**:

```bash
python scripts/export_w3dm_train.py --root data/CQ500 --split train \
    --out refs/jrm-adm/data/train_volumes
python scripts/export_cohort_for_jrm.py --root data/CQ500 \
    --ckpt logs/fm3d_databridge/ckpt_iter500000.pth --n 30 \
    --out refs/jrm-adm/data/ours_cohort
```

The training exporter preserves our original vertex-anchored 160×192×192 crop and superior-first
orientation in raw HU. The measurement exporter carries the paired ground-truth poses for
verification; the solvers do not receive them as estimates.

In the **JRM environment**, from `refs/jrm-adm`:

```bash
python train_w3dm.py --data data/train_volumes --out weights_retrain \
    --iters 300000 --batch 1 --lr .0001 --ema .999 --opt adamw --seed 0
MODEL_PATH=weights_retrain/model_state_dict.pth CUDA_VISIBLE_DEVICES=0 bash run_cohort.sh
```

`MODEL_PATH` is required, so the run cannot silently fall back to the authors' released prior.
`PYTHON`, `CASE_DIR` and `RESULT_DIR` can override the interpreter and directories. Existing
case results are preserved; choose a fresh `RESULT_DIR` for another configuration.

Back in this repository root and the **flow-matching environment**, score with the shared
operator/metric convention:

```bash
python scripts/score_jrm_native.py --root data/CQ500 \
    --ckpt logs/fm3d_databridge/ckpt_iter500000.pth --ours data/test30 \
    --jrm refs/jrm-adm/data/recon_ours_v2 --out data/jrm_scores.json
```

This reports both the native refined output and FDK at the recovered poses. It converts the
baseline's attenuation scale and affine-motion convention, checks paired trajectories, and
scores with the same measured-FOV mask. Use a fresh scores filename for a different experiment.
[PROVENANCE.md](PROVENANCE.md) preserves earlier adaptation notes; this guide specifies the
current manuscript cohort.
