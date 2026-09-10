#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

DEFAULT_REPO = "AntonV/mamba2-2.7b-hf"
FALLBACK_REPO = "state-spaces/mamba-1.4b-hf"
CACHE_DIR = "/kaggle/working/backbone_cache"


def sh(cmd: str) -> int:
    print(f"$ {cmd}")
    return subprocess.call(cmd, shell=True)


def detect_dtype_and_kernels():
    import torch

    if not torch.cuda.is_available():
        print("[WARNING] No GPU - falling back to CPU/float32 (smoke-test only).")
        return (torch.float32, False)
    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"[GPU] {name} (compute capability sm_{major}{minor})")
    if major < 8:
        print(
            "[INFO] Architecture < Ampere (sm_80): forcing torch_dtype=float16 instead of bfloat16 (T4/Turing has no HW acceleration for bf16 on tensor cores)."
        )
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    return (dtype, (major, minor))


def try_install_fast_kernels():
    print(
        "\n[INFO] Attempting to install fast CUDA kernels (mamba-ssm, causal-conv1d)..."
    )
    rc1 = sh(
        f"{sys.executable} -m pip install --break-system-packages -q causal-conv1d>=1.4.0"
    )
    rc2 = sh(
        f"{sys.executable} -m pip install --break-system-packages -q mamba-ssm>=2.2.0"
    )
    if rc1 != 0 or rc2 != 0:
        print(
            "[WARNING] Installing the fast kernels failed - transformers will use the pure-PyTorch fallback (slower, but correct). This does NOT block training, it just slows it down."
        )
    else:
        print("[OK] mamba-ssm + causal-conv1d ready.")


def download_and_verify(repo_id: str, dtype, cache_dir: str):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    os.makedirs(cache_dir, exist_ok=True)
    print(f"\n[INFO] Downloading config/tokenizer/weights: {repo_id} -> {cache_dir}")
    config = AutoConfig.from_pretrained(repo_id, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(
        repo_id,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
    )
    hidden_dim = getattr(config, "hidden_size", None) or getattr(
        config, "d_model", None
    )
    vocab_size = getattr(config, "vocab_size", None)
    n_params = sum((p.numel() for p in model.parameters()))
    vram_gb_fp16 = n_params * 2 / 1000000000.0
    print(f"[OK] repo_id={repo_id}")
    print(
        f"     hidden_dim={hidden_dim}  vocab_size={vocab_size}  params={n_params / 1000000000.0:.2f}B"
    )
    print(
        f"     estimated weight memory (fp16/bf16, frozen): ~{vram_gb_fp16:.2f} GB / GPU"
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
        ids = tokenizer(
            "assert n >= 0  # precondition", return_tensors="pt"
        ).input_ids.to("cuda")
        with torch.no_grad():
            out = model(input_ids=ids)
        print(
            f"[SMOKE-TEST OK] forward pass: hidden_states.shape={tuple(out.last_hidden_state.shape)}"
        )
    print(
        f"\n[NEXT STEP] In configs/default.yaml, set:\n  tokenizer:\n    mode: hf_pretrained\n    hf_repo: {repo_id}\n  backbone:\n    mode: hf_pretrained\n    repo_id: {repo_id}\n    freeze: true\n    torch_dtype: {('float16' if dtype == torch.float16 else 'bfloat16')}\n    gradient_checkpointing: false\n"
    )
    return (model, tokenizer, config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo_id",
        default=DEFAULT_REPO,
        help=f"HF backbone repo (default {DEFAULT_REPO}, fallback on VRAM/time: {FALLBACK_REPO})",
    )
    ap.add_argument("--cache_dir", default=CACHE_DIR)
    ap.add_argument(
        "--skip_kernel_install",
        action="store_true",
        help="Skip attempting to install mamba-ssm/causal-conv1d",
    )
    args = ap.parse_args()
    dtype, cc = detect_dtype_and_kernels()
    if not args.skip_kernel_install:
        try_install_fast_kernels()
    try:
        download_and_verify(args.repo_id, dtype, args.cache_dir)
    except Exception as ex:
        print(f"\n[ERROR] Failed to download/verify {args.repo_id}: {ex}")
        print(f"[FALLBACK] Trying a smaller backbone: {FALLBACK_REPO}")
        download_and_verify(FALLBACK_REPO, dtype, args.cache_dir)


if __name__ == "__main__":
    main()
