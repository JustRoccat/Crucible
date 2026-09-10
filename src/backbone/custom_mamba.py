from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def chunked_selective_scan(
    Abar: torch.Tensor, Bx: torch.Tensor, C: torch.Tensor, chunk_size: int = 64
) -> torch.Tensor:
    Bsz, L, D, N = Abar.shape
    eps = 1e-06
    h = Abar.new_zeros(Bsz, D, N)
    ys = []
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        Abar_c = Abar[:, start:end]
        Bx_c = Bx[:, start:end]
        C_c = C[:, start:end]
        log_Abar_c = torch.log(Abar_c.clamp_min(eps))
        cum_log = torch.cumsum(log_Abar_c, dim=1)
        decayed_h_prev = torch.exp(cum_log) * h.unsqueeze(1)
        c_len = cum_log.shape[1]
        h_intra = []
        h_t = torch.zeros_like(h)
        for i in range(c_len):
            Abar_t = Abar_c[:, i]
            Bx_t = Bx_c[:, i]
            h_t = Abar_t * h_t + Bx_t
            h_intra.append(h_t)
        intra_chunk = torch.stack(h_intra, dim=1)
        h_all = decayed_h_prev + intra_chunk
        y_c = torch.einsum("bcdn,bcn->bcd", h_all, C_c)
        ys.append(y_c)
        h = h_all[:, -1]
    return torch.cat(ys, dim=1)


class SelectiveSSMBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        state_dim: int = 16,
        conv_kernel: int = 4,
        scan_chunk_size: int = 64,
        use_compile: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.scan_chunk_size = scan_chunk_size
        self.in_proj = nn.Linear(dim, 2 * dim)
        self.conv1d = nn.Conv1d(
            dim, dim, kernel_size=conv_kernel, groups=dim, padding=conv_kernel - 1
        )
        self.delta_proj = nn.Linear(dim, dim)
        self.B_proj = nn.Linear(dim, state_dim)
        self.C_proj = nn.Linear(dim, state_dim)
        self.A_log = nn.Parameter(torch.log(torch.rand(dim, state_dim) + 0.001))
        self.D_skip = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim)
        self._scan_fn = chunked_selective_scan
        if use_compile:
            self._scan_fn = torch.compile(chunked_selective_scan, dynamic=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Bsz, L, D = x.shape
        N = self.state_dim
        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :L]
        x_conv = F.silu(x_conv.transpose(1, 2))
        delta = F.softplus(self.delta_proj(x_conv))
        B_ssm = self.B_proj(x_conv)
        C_ssm = self.C_proj(x_conv)
        A = -torch.exp(self.A_log)
        Abar = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, D, N))
        Bx = delta.unsqueeze(-1) * B_ssm.unsqueeze(2) * x_conv.unsqueeze(-1)
        y = self._scan_fn(Abar, Bx, C_ssm, chunk_size=self.scan_chunk_size)
        y = y + self.D_skip * x_conv
        y = y * F.silu(z_branch)
        return self.out_proj(y)


class CustomMambaBackbone(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        n_layers: int = 4,
        state_dim: int = 16,
        scan_chunk_size: int = 64,
        use_compile: bool = False,
    ):
        super().__init__()
        self.hidden_dim = dim
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(dim),
                        "mamba": SelectiveSSMBlock(
                            dim,
                            state_dim=state_dim,
                            scan_chunk_size=scan_chunk_size,
                            use_compile=use_compile,
                        ),
                    }
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = x + layer["mamba"](layer["norm"](x))
        return x
