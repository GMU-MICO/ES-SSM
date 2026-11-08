# ES-SSM

Elastic Spectral State Space Models for Train-Once Budgeted Inference

Dachuan Song, Junyu Yin, Zechen Hu, Xuan Wang

NeurIPS 2026, Main Track (Poster).

[Paper](https://arxiv.org/abs/2601.22488) · [Hugging Face](https://huggingface.co/Dachuan99/ES-SSM) · [Usage](#usage) · [ESSH follow-up](https://arxiv.org/abs/2609.32486) · [Citation](#citation)

## Overview

ES-SSM trains once and exports compact models for different compute budgets. Fixed Hankel spectral channels provide an ordered approximation of the sequence operator; the runtime budget `K` selects a prefix of these channels.

Input-adaptive gates and budget dropout train the prefixes used at inference. Export keeps the first `K` filters, channel projections and gate-output rows, together with the shared parameters. Each compact model runs without further training.

![ES-SSM architecture](ES-SSM-figure-1.png)

Figure 1. ES-SSM residual block and spectral sublayer.

## Experiments

The paper evaluates ES-SSM on byte-level language modeling, Long Range Arena, Speech Commands V2, and D4RL offline reinforcement learning.

![SlimPajama results](ES-SSM-figure-3.png)

Figure 3. SlimPajama results for the 1.3B-parameter family trained on 30B byte-tokens. Test bits per byte (BPB) is plotted against spectral budget and inference throughput (sequence length 2048, batch 1, one B200 GPU). Dashed curves are non-elastic references trained separately at each capacity.

![D4RL results](ES-SSM-figure-4.png)

Figure 4. D4RL normalized returns and total training time across nine datasets and three seeds on RTX 3090. ES-SSM trains once per dataset and seed and exports five budgets; the Transformer reference trains at each budget.

## Usage

The code covers ES-SSM training and evaluation on byte-level SlimPajama and D4RL MuJoCo.

### Installation

Use Python 3.10 or later and PyTorch 2.4 or later. Install a PyTorch build suitable for your hardware; the full language-modeling experiment uses CUDA.

```bash
git clone https://github.com/GMU-MICO/ES-SSM.git
cd ES-SSM
python -m pip install -r requirements.txt
python smoke_test.py
```

`smoke_test.py` checks training, inference, causality, compact-model equivalence and export error handling on CPU with synthetic inputs.

### Language modeling

Download the SlimPajama parquet files used by the scripts:

```bash
python download_slimpajama.py --out-dir ./datasets/slimpajama \
  --train-files 250 --val-files 30 --test-files 30
```

Omit the shard-count flags to download one shard per split; use `--num-workers 0` in the training command for a single training shard. `download_manifest.json` records the dataset revision and file hashes; `--revision` selects a specific revision.

Train ES-SSM:

```bash
torchrun --nproc_per_node=1 essm_LM.py \
  --data-dir ./datasets/slimpajama --output-dir ./runs/essm \
  --filter-cache-dir ./cache/filter
```

`CFG` in `essm_LM.py` sets the model, training schedule and evaluation budgets. The defaults are the 1.3B-family configuration and 30B byte-tokens.

Evaluate a checkpoint across the configured budgets:

```bash
python essm_LM.py --inference-only \
  --checkpoint ./runs/essm/checkpoints/best.pt \
  --data-dir ./datasets/slimpajama --output-dir ./runs/essm \
  --filter-cache-dir ./cache/filter
```

Evaluation calls `build_compact` for each budget in `CFG["eval_k"]`.

### Offline reinforcement learning

```bash
python -m pip install -r requirements-rl.txt
bash reproduce_DL.sh
```

This trains, exports and evaluates compact checkpoints at `K = 2, 4, 8, 16, 32`. Individual commands and settings are in the [D4RL guide](docs/offline_rl.md).

### Files

| File | Contents |
| --- | --- |
| `essm_LM.py` | Language-model training and compact-model evaluation |
| `d4rl_mujoco_RL.py` | D4RL training, checkpoint export, and evaluation |
| `runtime_utils.py` | Hankel filters, data caches, and checkpoint utilities |
| `download_slimpajama.py` | SlimPajama download and revision manifest |
| `smoke_test.py` | Functional checks with small models and synthetic data |

## Follow-up work

[Elastic Selective Spectral Hybrids for Train-Once, Export-Many Budgeted Inference (ESSH)](https://arxiv.org/abs/2609.32486) extends this direction to hybrid language models. It combines selective spectral mixers with input-dependent decay and write/read gates, sliding-window attention, and joint training of spectral-channel count and feed-forward width.

At 1.53B parameters on an NVIDIA A100, fused decoding matches the speed of Mamba-2. At matched parameter counts and under the same compilation policy for all models, ESSH achieves a 1.1–1.6× speedup over Mamba-3 and a 1.1–4.6× speedup over Transformer++.

## Citation

BibTeX entries for ES-SSM and the ESSH follow-up:

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

## License

The code is released under the [Apache License 2.0](LICENSE).
