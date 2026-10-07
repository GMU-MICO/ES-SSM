# ES-SSM

**Elastic Spectral State Space Models for Train-Once Budgeted Inference**

[Paper: arXiv:2601.22488](https://arxiv.org/abs/2601.22488) | [Code](#code) | [Hybrid follow-up](https://arxiv.org/abs/2609.32486) | [Citation](#citation)

Modern sequence models are typically trained at a fixed computational capacity. However, real-world applications require AI deployment across heterogeneous computational platforms, from high-throughput cloud clusters to compute-limited endpoints and power-constrained edge devices.

ES-SSM studies elasticity from an operator-approximation perspective: it controls the approximation resolution of the SSM sequence operator, rather than only shrinking architectural components.

## Method

ES-SSM is a train-once, export-many sequence modeling framework that gains elasticity through spectral approximation of the SSM sequence operator. Fixed Hankel spectral channels define an operator-level approximation resolution. Input-adaptive channel-wise gates and budget dropout train the same spectral prefixes that compact models use at inference time.

At deployment, direct prefix truncation keeps channels 1 through K and removes the tail channels. One full-capacity checkpoint can therefore produce smaller models across runtime budgets, without budget-specific retraining, channel search, or reordering.

![Figure 1: ES-SSM architecture overview](ES-SSM-figure-1.png)

**Figure 1.** ES-SSM architecture overview. Fixed Hankel spectral channels and a tokenwise Gate MLP produce input-adaptive, channel-wise budgeted gates. Thus, one full-capacity training can be truncated into compact models at multiple runtime budgets.

## Results

Evaluations cover byte-level language modeling, Long Range Arena, Speech Commands V2, D4RL offline reinforcement learning, and a byte-level generality check.

On SlimPajama (1.3B family, 30B tokens), ES-SSM produces a smooth, usable budget curve from one full-capacity training run; even the smallest prefixes remain outside the collapse zone. Ten exported parameter budgets span roughly 330M to 1.3B.

![Figure 3: SlimPajama scaling results](ES-SSM-figure-3.png)

**Figure 3.** SlimPajama main-table visualization for the 1.3B-family, 30B-token experiment. [A] Test BPB vs runtime budget K. [B] Test BPB vs throughput at sequence length 2048, batch 1, on one B200 GPU. Dashed non-elastic baselines are independently trained oracles, not intended for direct comparison with ES-SSM.

On D4RL (9 datasets x 3 seeds, RTX 3090), ES-SSM obtains all five budgets from one run and reduces total training wall-clock from 107.84 h to 23.99 h (77.8%). Returns match the independently trained Transformer at K=2, 4, and 16, and improve at K=8 and 32.

![Figure 4: D4RL quality and training-cost comparison](ES-SSM-figure-4.png)

**Figure 4.** [A] Average normalized return vs runtime budget. [B] Total training wall-clock on the full D4RL suite (9 datasets x 3 seeds, RTX 3090).

## Future work

For hybrid architectures, see our follow-up paper, [Elastic Selective Spectral Hybrids for Train-Once, Export-Many Budgeted Inference](https://arxiv.org/abs/2609.32486).

We are also pursuing an attention-free state-space model with strong retrieval, recall, and general sequence-modeling capabilities.

## Code

This release contains the main language-modeling and D4RL experiments from
the September 22, 2026 submission package. It includes the ES-SSM
language-model driver, compact model evaluation, offline RL export, and shared data utilities.
The existing PG19 notebook remains available as a separate example.

| File | Purpose |
| --- | --- |
| `essm_LM.py` | ES-SSM training and evaluation across compute budgets |
| `d4rl_mujoco_RL.py` | Offline RL training, compact export and evaluation |
| `runtime_utils.py` | Hankel filters, dataset caches and checkpoint writes |
| `download_slimpajama.py` | Download local parquet data with a pinned revision |
| `smoke_test.py` | Small functional checks using synthetic data |

### Install and check

Use Python 3.10 or later and a PyTorch build appropriate for your hardware.
The full language-model experiments target CUDA; the functional checks run on CPU.

```bash
python -m pip install -r requirements.txt
python smoke_test.py
```

The check covers forward and backward passes, activation checkpointing, short
training runs, inference, causal behavior, and independent compact checkpoint
reloads. It uses temporary synthetic data and downloads no dataset.

### Language modeling

Prepare a local SlimPajama parquet mirror:

```bash
python download_slimpajama.py --out-dir ./datasets/slimpajama \
  --train-files 250 --val-files 30 --test-files 30
```

This requests the complete shard layout used by the package and is a large
download. The downloader records the dataset revision and file hashes in
`download_manifest.json`; pass that revision with `--revision` to reuse it.
Without the shard-count flags, it downloads only one shard per split.

```bash
torchrun --nproc_per_node=1 essm_LM.py \
  --data-dir ./datasets/slimpajama --output-dir ./runs/essm \
  --filter-cache-dir ./cache/filter
```

The archived ES-SSM defaults are the large model and a 30B **byte-token**
training target. The smoke test is the small check; this command runs the main
experiment. Architecture and training settings are in `CFG`.

Evaluate a trained checkpoint:

```bash
python essm_LM.py --inference-only \
  --checkpoint ./runs/essm/checkpoints/best.pt \
  --data-dir ./datasets/slimpajama --output-dir ./runs/essm \
  --filter-cache-dir ./cache/filter
```

### Compact model evaluation

The ES-SSM inference command constructs a physically smaller model for each
configured budget using `build_compact`. The D4RL entry point also saves
independent compact checkpoints with its existing `export-compact` command.

### Offline reinforcement learning

```bash
python -m pip install -r requirements-rl.txt
bash reproduce_DL.sh
```

This runs the ES-SSM D4RL MuJoCo grid, exports five independent budgets and
evaluates the full and compact models. For a single
experiment and the individual commands, see [the offline RL guide](docs/offline_rl.md).
Use trusted RL checkpoints: the archived format includes NumPy metadata.

### Release scope

Budget and precision ablations, supplementary diagnostics, scheduler files,
datasets and trained weights are omitted. LRA and Speech Commands training
code was not present in the supplied package. The checks validate the
included implementation; they do not constitute a rerun of the paper's
full training results. The original model and training defaults are retained.

## Citation

If you use this work, please cite:

```bibtex
@misc{song2026elasticspectralstatespace,
  title={Elastic Spectral State Space Models for Train-Once Budgeted Inference},
  author={Dachuan Song and Junyu Yin and Zechen Hu and Xuan Wang},
  year={2026},
  eprint={2601.22488},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2601.22488},
}

@misc{song2026elasticselectivespectralhybrids,
  title={Elastic Selective Spectral Hybrids for Train-Once, Export-Many Budgeted Inference},
  author={Dachuan Song and Chuchu Chen and Xuan Wang},
  year={2026},
  eprint={2609.32486},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2609.32486},
}
```
