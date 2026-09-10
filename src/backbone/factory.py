from __future__ import annotations
from .custom_mamba import CustomMambaBackbone
from .hf_backbone import HFPretrainedBackbone


def build_backbone(cfg: dict):
    mode = cfg.get("mode", "custom_mamba")
    if mode == "custom_mamba":
        return CustomMambaBackbone(
            vocab_size=cfg["vocab_size"],
            dim=cfg.get("dim", 128),
            n_layers=cfg.get("n_layers", 4),
            state_dim=cfg.get("state_dim", 16),
            scan_chunk_size=cfg.get("scan_chunk_size", 64),
            use_compile=cfg.get("use_compile", False),
        )
    elif mode == "hf_pretrained":
        return HFPretrainedBackbone(
            repo_id=cfg["repo_id"],
            freeze=cfg.get("freeze", True),
            torch_dtype=cfg.get("torch_dtype", "bfloat16"),
            gradient_checkpointing=cfg.get("gradient_checkpointing", False),
            trust_remote_code=cfg.get("trust_remote_code", False),
            config_overrides=cfg.get("config_overrides"),
        )
    raise ValueError(f"Nieznany tryb backbone: {mode}")
