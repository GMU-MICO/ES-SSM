# D4RL MuJoCo Code for ES-SSM

The offline RL entry point runs the D4RL MuJoCo experiment grid:
- 9 datasets (HalfCheetah / Hopper / Walker2d × medium / medium-replay / medium-expert)
- 3 seeds per cell
- ES-SSM at full capacity `K_max=32` (`esssm_K32`), with embedding dimension 72
  and three layers, followed by compact models at `K ∈ {2, 4, 8, 16, 32}`.

After training, each ES-SSM(K_max=32) cell is sliced into five independent
compact checkpoints at K ∈ {2, 4, 8, 16, 32}. Compact inference loads each
file in isolation; its on-disk size, runtime memory, and forward-pass latency
reflect the deployed K and not the source K_max.

## File layout

```
d4rl_mujoco_RL.py
runtime_utils.py
reproduce_DL.sh
requirements-rl.txt
docs/offline_rl.md
```

The script keeps three separate working roots, all configurable on the CLI:

```
--hdf5-dir   ./data
--cache-dir  ./cache_filters
--project-root ./project
```

## Environment

Run the commands below from the repository root. Install the dependencies with
`python -m pip install -r requirements-rl.txt`.

Python ≥ 3.10 with:

```
torch >= 2.4
numpy
h5py
gymnasium >= 1.0
mujoco >= 3.0
```

Atari is not used. CUDA is recommended for the full sweep but the script also
runs end-to-end on CPU (slowly).

## Reproducing the headline grid

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

## Running a single cell (debugging)

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

## What gets written

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

## Hyperparameters

Archived training settings used by this entry point:

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

ES-SSM uses budget-dropout: after 10% of training at full K_max=32, K_rt is
sampled log-uniformly from {2, 3, 4, 5, 6, 8, 12, 16, 24, 32}; every 8 steps
the full K_max is anchored. This is what makes prefix-truncated compact
checkpoints reliable at every K.

## Compact deployment

`build_compact_esssm_from_prefix` constructs a fresh `RLSequenceModel` whose
ES-SSM has `K_max = K`, copies the K-prefix of `sigma`, `phi_f_ri`, `M`, and the
selector output rows from the source checkpoint, and re-saves the model as
an independent `.pt`. Loading the compact file does not require the source.
