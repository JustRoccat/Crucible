from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ttt import MicroLoRAFastWeights
from .jepa import JEPAEncoder, EMATargetEncoder, JEPAPredictor
from .hdc import HDCMemory, HDCBridge
from .symbolic import SymbolicTrigger
from .symbolic.self_correction import (
    SelfCorrectionEngine,
    SelfCorrectionOutcome,
    ProposeFn,
)


class NeuroSymbolicModel(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        vocab_size: int,
        ttt_rank: int = 8,
        ttt_inner_lr: float = 0.1,
        ttt_inner_steps: int = 1,
        jepa_hidden_mult: int = 2,
        ema_decay: float = 0.996,
        hdc_dim: int = 10000,
        mask_ratio: float = 0.25,
        max_self_correction_attempts: int = 5,
    ):
        super().__init__()
        self.backbone = backbone
        dim = backbone.hidden_dim
        self.dim = dim
        self.mask_ratio = mask_ratio
        self.fast_weights = MicroLoRAFastWeights(
            dim, rank=ttt_rank, inner_lr=ttt_inner_lr, inner_steps=ttt_inner_steps
        )
        self.jepa_encoder = JEPAEncoder(dim, hidden_mult=jepa_hidden_mult)
        self.target_encoder = EMATargetEncoder(self.jepa_encoder, decay=ema_decay)
        self.predictor = JEPAPredictor(dim, hidden_mult=jepa_hidden_mult)
        self.symbolic_trigger = SymbolicTrigger(dim)
        self.hdc_bridge = HDCBridge(latent_dim=dim, hdc_dim=hdc_dim)
        self.lm_head = nn.Linear(dim, vocab_size)
        self.self_correction_engine = SelfCorrectionEngine(
            max_attempts=max_self_correction_attempts
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        hdc_memory: HDCMemory | None = None,
        mask_ratio: float | None = None,
    ) -> dict:
        mask_ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        x = self.backbone(input_ids)
        x = self.fast_weights(x)
        latent_concepts = self.jepa_encoder(x)
        with torch.no_grad():
            target_latents = self.target_encoder(x)
        x_masked, mask = self.predictor.apply_mask(latent_concepts, mask_ratio)
        pred = self.predictor(x_masked)
        jepa_loss = JEPAPredictor.loss(pred, target_latents, mask)
        lm_logits = self.lm_head(latent_concepts)
        lm_loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(
                lm_logits.view(-1, lm_logits.size(-1)), targets.reshape(-1)
            )
        if hdc_memory is not None:
            pooled = latent_concepts.mean(dim=1)
            for b in range(pooled.size(0)):
                hv = self.hdc_bridge.to_hdc(pooled[b])
                hdc_memory.write(key=f"seq_{b}", value_hv=hv.detach())
        return {
            "lm_logits": lm_logits,
            "lm_loss": lm_loss,
            "jepa_loss": jepa_loss,
            "latent_concepts": latent_concepts,
            "mask": mask,
        }

    @torch.no_grad()
    def update_target_encoder(self):
        self.target_encoder.update(self.jepa_encoder)

    def self_correct(
        self,
        propose_fn: ProposeFn,
        initial_context: str = "",
        hdc_memory: HDCMemory | None = None,
    ) -> SelfCorrectionOutcome:
        outcome = self.self_correction_engine.run(propose_fn, initial_context)
        if outcome.success and hdc_memory is not None:
            last = outcome.attempts[-1]
            hdc_memory.store_verified_pattern(
                source="\n".join(last.spec.constraints),
                z3_status=last.result.status,
                z3_message=last.result.message,
                model_values=last.result.model_values,
            )
        return outcome
