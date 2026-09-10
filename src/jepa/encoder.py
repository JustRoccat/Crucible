import copy
import torch
import torch.nn as nn


class JEPAEncoder(nn.Module):

    def __init__(self, dim: int, hidden_mult: int = 2):
        super().__init__()
        hidden = dim * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim), nn.LayerNorm(dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EMATargetEncoder(nn.Module):

    def __init__(self, encoder: JEPAEncoder, decay: float = 0.996):
        super().__init__()
        self.decay = decay
        self.target = copy.deepcopy(encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, encoder: JEPAEncoder):
        for tp, sp in zip(self.target.parameters(), encoder.parameters(), strict=True):
            tp.data.mul_(self.decay).add_(sp.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.target(x)
