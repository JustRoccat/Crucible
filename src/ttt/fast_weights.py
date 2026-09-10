from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroLoRAFastWeights(nn.Module):

    def __init__(
        self, dim: int, rank: int = 8, inner_lr: float = 0.1, inner_steps: int = 1
    ):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.base_A = nn.Parameter(torch.randn(dim, rank) * (1.0 / dim**0.5))
        self.base_B = nn.Parameter(torch.zeros(rank, dim))

    def _adapt(self, x_detached: torch.Tensor) -> torch.Tensor:
        A = self.base_A.detach().clone()
        Bm = self.base_B.detach().clone()
        for _ in range(self.inner_steps):
            with torch.enable_grad():
                A_step = A.clone().requires_grad_(True)
                B_step = Bm.clone().requires_grad_(True)
                recon = x_detached @ A_step @ B_step
                loss = F.mse_loss(recon, x_detached)
                g_A, g_B = torch.autograd.grad(loss, [A_step, B_step])
            A = (A - self.inner_lr * g_A).detach()
            Bm = (Bm - self.inner_lr * g_B).detach()
        return x_detached @ A @ Bm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_detached = x.detach()
        adapted_delta = self._adapt(x_detached)
        base_delta = x @ self.base_A @ self.base_B
        delta = adapted_delta + (base_delta - base_delta.detach())
        return x + delta

    def extra_repr(self) -> str:
        n_full = self.dim * self.dim
        n_lora = 2 * self.dim * self.rank
        return f"dim={self.dim}, rank={self.rank} ({n_lora}/{n_full} = {100 * n_lora / n_full:.1f}% parametrow pelnej macierzy)"
