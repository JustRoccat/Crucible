from __future__ import annotations
import torch
import torch.nn as nn


class HFPretrainedBackbone(nn.Module):

    def __init__(
        self,
        repo_id: str,
        freeze: bool = True,
        torch_dtype: str = "bfloat16",
        gradient_checkpointing: bool = False,
        trust_remote_code: bool = False,
        config_overrides: dict | None = None,
    ):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(torch_dtype, torch.bfloat16)
        self.config = AutoConfig.from_pretrained(
            repo_id, trust_remote_code=trust_remote_code
        )
        if config_overrides:
            for key, value in config_overrides.items():
                if not hasattr(self.config, key):
                    raise ValueError(
                        f"config_overrides: field '{key}' does not exist in the {type(self.config).__name__} config - check for a typo."
                    )
                setattr(self.config, key, value)
        self.model = AutoModel.from_pretrained(
            repo_id,
            config=self.config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )
        self.hidden_dim = self._infer_hidden_dim(self.config)
        self.vocab_size = self._infer_vocab_size(self.config)
        if gradient_checkpointing and hasattr(
            self.model, "gradient_checkpointing_enable"
        ):
            self.model.gradient_checkpointing_enable()
        self.freeze = freeze
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

    @staticmethod
    def _infer_hidden_dim(config) -> int:
        for attr in ("hidden_size", "d_model", "n_embd"):
            if hasattr(config, attr):
                return int(getattr(config, attr))
        raise ValueError(
            "Failed to read the hidden dimension from the HF config — check `AutoConfig.from_pretrained(repo_id)` manually."
        )

    @staticmethod
    def _infer_vocab_size(config) -> int:
        if hasattr(config, "vocab_size"):
            return int(config.vocab_size)
        raise ValueError("The HF config has no `vocab_size` — unusual architecture.")

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state
        return hidden
