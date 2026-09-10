<div align="center">

# Crucible

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/tests-75%20passing-2ea44f?style=flat-square)](tests)

A Transformer-free sequence model that pairs a linear-time state-space
backbone (Mamba / Mamba2) with fast in-context adaptation, concept-level
self-supervised learning (JEPA), a hyperdimensional (HDC) long-term memory,
and a formal verifier (Z3) the model can call as a tool to check and correct
its own outputs.

[Overview](#overview) • [Architecture](#architecture) • [Installation](#installation) • [Usage](#usage) • [Configuration](#configuration) • [Training on Kaggle](#training-on-kaggle) • [Troubleshooting](#troubleshooting)

*(and yes, this code is AI-assisted, lol)*

</div>

## Overview

Most code-generation models are pure next-token predictors: they have no
way to check whether what they just wrote is actually correct before
handing it to you. This project explores a different shape — a small,
trainable stack of adapters sitting on top of a frozen state-space
backbone, with an explicit path to a symbolic solver for verification.

- **Backbone-agnostic**: train a small Mamba-style backbone from scratch,
  or freeze a pretrained Hugging Face checkpoint (Mamba2, SmolLM2, Qwen2.5,
  ...) and train only the adapters on top.
- **Concept-level learning**: a JEPA objective predicts masked *hidden
  representations* rather than raw tokens, encouraging the model to learn
  structure instead of surface patterns.
- **Fast adaptation**: a lightweight test-time-training mechanism (Micro-LoRA
  fast weights) lets the model adapt within a single forward pass.
- **Formal verification loop**: the model can emit a symbolic specification,
  have it checked by Z3, and retry on failure — an explicit propose → verify
  → correct cycle instead of a single unchecked guess.
- **Persistent memory**: a hyperdimensional-computing (HDC) memory stores
  patterns Z3 has verified as valid, for fast associative recall later.

> [!IMPORTANT]
> This is a research prototype, not a chatbot out of the box. You train it
> yourself, on your own data, for your own purpose. See
> [Experimental status](#experimental-status) below.

## Architecture

| Component | Location | Role |
|---|---|---|
| Tokenizer | `src/tokenizer/` | Production BPE tokenization (`tiktoken`, Hugging Face `tokenizers`, or a custom-trained BPE) instead of raw bytes. |
| Backbone | `src/backbone/` | A from-scratch Mamba-style state-space model, or a frozen pretrained Hugging Face checkpoint used as a feature extractor. |
| Fast weights (TTT) | `src/ttt/` | Low-rank, test-time adaptation mechanism. |
| JEPA | `src/jepa/` | Masked concept prediction in representation space. |
| Symbolic engine | `src/symbolic/` | Z3-backed constraint checking plus an autonomous propose/verify/retry loop. |
| HDC memory | `src/hdc/` | Hyperdimensional associative memory for verified patterns. |
| Model | `src/model.py` | Wires the components above into a single `NeuroSymbolicModel`. |
| Entry points | `train.py`, `evaluate.py` | Training and evaluation scripts. |

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Depending on which backbone and tokenizer you use, install the matching
extras:

```bash
pip install transformers          # backbone.mode: hf_pretrained
pip install tiktoken               # tokenizer.mode: tiktoken
pip install tokenizers              # tokenizer.mode: hf_tokenizers
```

Confirm the environment works before training anything:

```bash
pytest tests/ -v
```

## Usage

Train with the default configuration:

```bash
python train.py --config configs/default.yaml
```

Evaluate a checkpoint:

```bash
python evaluate.py --ckpt checkpoints/final.pt
```

Resume training from a checkpoint (useful across interrupted or
time-limited sessions):

```bash
python train.py --config configs/default.yaml --resume_from checkpoints/final.pt
```

> [!TIP]
> Checkpoints only store trainable parameters. If you're using a frozen
> pretrained backbone, it's re-downloaded from the Hub on each run instead
> of being duplicated in every checkpoint file.

## Configuration

All training is driven by a single YAML file. The key sections:

```yaml
tokenizer:
  mode: hf_pretrained        # byte | tiktoken | hf_tokenizers | hf_pretrained
  hf_repo: AntonV/mamba2-2.7b-hf

backbone:
  mode: hf_pretrained        # custom_mamba | hf_pretrained
  repo_id: AntonV/mamba2-2.7b-hf
  freeze: true
  torch_dtype: float16       # bfloat16 requires Ampere (sm_80) or newer
  config_overrides:
    chunk_size: 32            # lower this on memory-constrained GPUs

train:
  train_file: path/to/corpus.txt
  seq_len: 256
  batch_size: 1
  steps: 15000
  lr: 0.0002
  ckpt_every: 250
  ckpt_dir: checkpoints
  device: cuda
```

See `configs/default.yaml` for the full set of options and their defaults.

## Training on Kaggle

`configs/kaggle.yaml` and the scripts under `kaggle/` are tuned for a free
Kaggle Notebook session (GPU T4, 16 GB VRAM):

```bash
python kaggle/download_backbone.py --repo_id AntonV/mamba2-2.7b-hf
python kaggle/build_corpus_logic.py --output kaggle/corpus.txt
python train.py --config configs/kaggle.yaml
```

Sessions are capped at 12 hours, so checkpoints need to move between runs.
`kaggle/checkpoint_sync.py` pushes to (and pulls from) a private Hugging
Face Hub repo:

```bash
python kaggle/checkpoint_sync.py --backend hf --push --local_dir checkpoints --repo_id <you>/checkpoints
python kaggle/checkpoint_sync.py --backend hf --pull --local_dir checkpoints --repo_id <you>/checkpoints
```

> [!NOTE]
> `docs/kaggle-lessons-learned.md` documents every non-obvious issue
> encountered getting this pipeline stable on Kaggle's free tier — wrong
> Hub repo IDs, an out-of-memory failure mode specific to Mamba2's
> training-time code path, disk-filling checkpoints, and more. Worth
> reading before you start.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ImportError: No module named 'transformers'` | `backbone.mode: hf_pretrained` is set but `transformers` isn't installed. |
| Garbage output, loss doesn't move | Tokenizer and backbone repo IDs don't match. |
| `CUDA out of memory` | Lower `batch_size`, `seq_len`, or `backbone.config_overrides.chunk_size`. |
| `UnsafeExpressionError` from the Z3 engine | The constraint string used something outside the safe subset (comparisons, `and`/`or`/`not`, arithmetic, variables, numbers). |
| Tests fail on a fresh clone | Run `pip install -r requirements.txt` and the relevant extras from [Installation](#installation) before `pytest`. |

## Experimental status

> [!WARNING]
> This project is experimental. The base backbone has no instruction
> tuning or RLHF, and the training corpus is filtered only for structural
> and logical density — not for harmful content. Treat outputs
> accordingly, and say so explicitly if you share trained weights beyond
> your own experimentation.
