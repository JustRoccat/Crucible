import argparse
import os
import random
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.model import NeuroSymbolicModel
from src.backbone import build_backbone
from src.tokenizer import ProductionTokenizer
from src.data import TokenizedTextDataset, SyntheticDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def build_dataset(cfg: dict, tokenizer: ProductionTokenizer):
    train_file = cfg["train"].get("train_file")
    seq_len = cfg["train"]["seq_len"]
    if train_file:
        return TokenizedTextDataset(train_file, tokenizer=tokenizer, seq_len=seq_len)
    print("[train] No `train.train_file` in the config -> using synthetic data.")
    return SyntheticDataset(
        vocab_size=tokenizer.vocab_size, seq_len=seq_len, n_samples=2000
    )


def build_optimizer(model: NeuroSymbolicModel, cfg: dict) -> torch.optim.Optimizer:
    lr_main = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"]["weight_decay"])
    backbone_frozen = not any(
        (
            n.startswith("backbone.") and p.requires_grad
            for n, p in model.named_parameters()
        )
    )
    if "backbone_lr" not in cfg["train"] and (not backbone_frozen):
        print(
            "[train] WARNING: the backbone is unfrozen, but `train.backbone_lr` is not set in the config -> using the same lr as the rest of the model. This is a real risk of catastrophic forgetting. Add `train.backbone_lr` (e.g. 1-5e-5) to the config."
        )
    lr_backbone = float(cfg["train"].get("backbone_lr", lr_main))
    backbone_params, other_params = ([], [])
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if name.startswith("backbone.") else other_params).append(p)
    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": lr_main})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    n_backbone = sum((p.numel() for p in backbone_params))
    n_other = sum((p.numel() for p in other_params))
    print(
        f"[train] Optimizer groups: rest_of_model={n_other:,} params @ lr={lr_main}, backbone={n_backbone:,} params @ lr={lr_backbone}{(' (backbone frozen, 0 trainable)' if n_backbone == 0 else '')}"
    )
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def load_checkpoint_for_resume(path: str, model, optimizer, scaler, device):
    print(f"[train] Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if unexpected:
        raise RuntimeError(
            f"[train] Checkpoint contains {len(unexpected)} unexpected keys (mismatched architecture/config?): {unexpected[:5]}..."
        )
    if "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        print(
            "[train] WARNING: checkpoint has no optimizer state (old format) - continuing with zero Adam moments."
        )
    if "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])
    step = ckpt.get("step", 0)
    print(f"[train] Resumed from step {step}.")
    return step


def trainable_state_dict(model) -> dict:
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    full_sd = model.state_dict()
    return {k: v for k, v in full_sd.items() if k in trainable_names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a checkpoint (.pt) from which to resume training (weights + optimizer + scaler + step). Use together with kaggle/checkpoint_sync.py --pull at the start of each new Kaggle session.",
    )
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["train"]["seed"])
    device = resolve_device(cfg["train"]["device"])
    print(f"[train] Device: {device}")
    tokenizer = ProductionTokenizer.from_config(cfg["tokenizer"])
    print(
        f"[train] Tokenizer: {tokenizer.backend_name}, vocab_size={tokenizer.vocab_size}"
    )
    backbone_cfg = dict(cfg["backbone"])
    if backbone_cfg.get("mode", "custom_mamba") == "custom_mamba":
        backbone_cfg["vocab_size"] = tokenizer.vocab_size
    backbone = build_backbone(backbone_cfg).to(device)
    vocab_size_for_head = getattr(backbone, "vocab_size", tokenizer.vocab_size)
    dataset = build_dataset(cfg, tokenizer)
    loader = DataLoader(
        dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, drop_last=True
    )
    model = NeuroSymbolicModel(
        backbone=backbone,
        vocab_size=vocab_size_for_head,
        ttt_rank=cfg["model"]["ttt_rank"],
        ttt_inner_lr=cfg["model"]["ttt_inner_lr"],
        ttt_inner_steps=cfg["model"]["ttt_inner_steps"],
        jepa_hidden_mult=cfg["model"]["jepa_hidden_mult"],
        ema_decay=cfg["model"]["ema_decay"],
        hdc_dim=cfg["model"]["hdc_dim"],
        mask_ratio=cfg["model"]["mask_ratio"],
        max_self_correction_attempts=cfg["model"]["max_self_correction_attempts"],
    ).to(device)
    n_params = sum((p.numel() for p in model.parameters()))
    n_trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(
        f"[train] Parameters: {n_params:,} total, {n_trainable:,} trainable ({100 * n_trainable / max(n_params, 1):.1f}%)"
    )
    optimizer = build_optimizer(model, cfg)
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(0)
        default_amp_dtype = "bfloat16" if major >= 8 else "float16"
    else:
        default_amp_dtype = "float32"
    amp_dtype_str = cfg["train"].get("amp_dtype", default_amp_dtype)
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[amp_dtype_str]
    print(f"[train] AMP: enabled={use_amp}, dtype={amp_dtype_str}")
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype == torch.float16
    )
    ckpt_dir = cfg["train"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    lm_w = cfg["train"]["lm_loss_weight"]
    jepa_w = cfg["train"]["jepa_loss_weight"]
    step = 0
    if args.resume_from:
        step = load_checkpoint_for_resume(
            args.resume_from, model, optimizer, scaler, device
        )
    model.train()
    pbar = tqdm(total=cfg["train"]["steps"], desc="training")
    data_iter = iter(loader)
    while step < cfg["train"]["steps"]:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x, y = (x.to(device), y.to(device))
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            out = model(x, targets=y)
            loss = lm_w * out["lm_loss"] + jepa_w * out["jepa_loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()
        model.update_target_encoder()
        step += 1
        pbar.update(1)
        if step % cfg["train"]["log_every"] == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lm=f"{out['lm_loss'].item():.4f}",
                jepa=f"{out['jepa_loss'].item():.4f}",
            )
        if step % cfg["train"]["ckpt_every"] == 0:
            ckpt_path = os.path.join(ckpt_dir, f"step_{step}.pt")
            torch.save(
                {
                    "model": trainable_state_dict(model),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step,
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"\n[train] Saved checkpoint: {ckpt_path}")
            for old in os.listdir(ckpt_dir):
                if old.startswith("step_") and old != f"step_{step}.pt":
                    os.remove(os.path.join(ckpt_dir, old))
    pbar.close()
    final_path = os.path.join(ckpt_dir, "final.pt")
    torch.save(
        {
            "model": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "config": cfg,
        },
        final_path,
    )
    print(f"[train] Training finished. Saved: {final_path}")


if __name__ == "__main__":
    main()
