# D4RL MuJoCo

The default sweep covers nine datasets: HalfCheetah, Hopper and Walker2d, each in the medium, medium-replay and medium-expert settings. It uses three seeds per dataset.

Each run trains ES-SSM at `K_max=32` with embedding dimension 72 and three layers, then exports separate checkpoints at `K = 2, 4, 8, 16, 32`.

## Environment

Use Python 3.10 or later and run the commands from the repository root:

```bash
python -m pip install -r requirements-rl.txt
```

The script uses CUDA when available and otherwise runs on CPU. Dependency versions are specified in `requirements-rl.txt` and `requirements.txt`.

## Full sweep

```bash
python d4rl_mujoco_RL.py download-data \
    --hdf5-dir ./data

python d4rl_mujoco_RL.py sweep-train \
    --project-root ./project --hdf5-dir ./data --cache-dir ./cache_filters \
    --seeds 42,43,44 --num-steps 150000 --slots-per-gpu 3

python d4rl_mujoco_RL.py sweep-export \
    --project-root ./project --cache-dir ./cache_filters \
    --seeds 42,43,44 --Ks 2,4,8,16,32

python d4rl_mujoco_RL.py sweep-eval \
    --project-root ./project --cache-dir ./cache_filters \
    --seeds 42,43,44 --num-eval-episodes 100 --slots-per-gpu 3

python d4rl_mujoco_RL.py tabulate \
    --project-root ./project --seeds 42,43,44 --out-dir ./results
```

`reproduce_DL.sh` runs these five commands in sequence with the same defaults.

The data, filter cache and run directories are set by `--hdf5-dir`, `--cache-dir` and `--project-root`.

## Single run

```bash
python d4rl_mujoco_RL.py train \
    --env hopper --dataset medium --model esssm --embed-dim 72 --K-max 32 \
    --seed 42 --num-steps 150000 \
    --hdf5-dir ./data --cache-dir ./cache_filters \
    --out-dir ./project/runs/esssm_K32/hopper-medium

python d4rl_mujoco_RL.py export-compact \
    --ckpt ./project/runs/esssm_K32/hopper-medium/ckpt.pt \
    --out-dir ./project/runs/esssm_K32/hopper-medium/compact \
    --Ks 2,4,8,16,32 --cache-dir ./cache_filters

python d4rl_mujoco_RL.py eval \
    --ckpt ./project/runs/esssm_K32/hopper-medium/compact/esssm_K8.pt \
    --out-json ./project/runs/esssm_K32/hopper-medium/compact/eval_K8.json \
    --num-eval-episodes 100 --cache-dir ./cache_filters
```

## Outputs

For each training cell `runs/<tag>/<env>-<dataset>[/seed<N>]/`:

```
ckpt.pt
summary.json
progress.jsonl
train.log
eval.json
compact/
```

After `tabulate`:

```
results/per_seed.csv
results/per_env_mean.csv
results/by_K.csv
```

## Default settings

| Setting           | Value          |
|-------------------|----------------|
| Embedding dim     | 72 |
| Layers            | 3              |
| Context length    | 20             |
| Batch size        | 64             |
| Optimizer         | AdamW(β=(0.9, 0.999), wd=1e-4) |
| Learning rate     | 1e-4 with linear warmup over 10k steps then constant |
| Total steps       | 150,000        |
| Gradient clip     | 0.25           |
| Eval episodes     | 100 (final)    |
| Eval target return| Higher of (3600, 1800) for Hopper, (12000, 6000) for HalfCheetah, (5000, 2500) for Walker2d |

Budget dropout uses the full budget for the first 10% of training steps. Afterwards, it samples a log-scale target and selects the nearest budget in `{2, 3, 4, 5, 6, 8, 12, 16, 24, 32}`, with a full-budget anchor every eight steps.

## Compact deployment

`build_compact_esssm_from_prefix` creates an `RLSequenceModel` with `K_max = K`, copies the first `K` entries of the channel-indexed tensors (`sigma`, `phi_f_ri`, `M` and selector output rows), and copies the shared parameters. Each exported `.pt` file contains the complete compact model.
