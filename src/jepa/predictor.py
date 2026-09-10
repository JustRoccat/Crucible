import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAPredictor(nn.Module):

    def __init__(self, dim: int, hidden_mult: int = 2, context_kernel: int = 4):
        super().__init__()
        hidden = dim * hidden_mult
        self.mask_token = nn.Parameter(torch.randn(dim) * 0.02)
        self.context_conv = nn.Conv1d(
            dim, dim, kernel_size=context_kernel, groups=dim, padding=context_kernel - 1
        )
        self.context_norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def apply_mask(self, x: torch.Tensor, mask_ratio: float):
        B, L, D = x.shape
        n_mask = 0 if mask_ratio <= 0.0 else max(1, int(L * mask_ratio))
        mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        if n_mask > 0:
            for b in range(B):
                idx = torch.randperm(L, device=x.device)[:n_mask]
                mask[b, idx] = True
        x_masked = x.clone()
        x_masked[mask] = self.mask_token.to(x.dtype)
        return (x_masked, mask)

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        B, L, D = x_masked.shape
        ctx = self.context_conv(x_masked.transpose(1, 2))[:, :, :L]
        ctx = F.silu(ctx.transpose(1, 2))
        ctx = self.context_norm(ctx + x_masked)
        return self.net(ctx)

    @staticmethod
    def loss(
        pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if mask.sum() == 0:
            return pred.new_zeros(())
        pred_masked = pred[mask]
        target_masked = target[mask].detach()
        return F.smooth_l1_loss(pred_masked, target_masked)
