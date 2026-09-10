from __future__ import annotations
import ast
import dataclasses
import hashlib
import time
import torch
import torch.nn as nn


def _seed_from_string(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % 2**32


def hash_ast(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    class _Anonymizer(ast.NodeTransformer):

        def visit_Name(self, node):
            node.id = "VAR"
            return node

        def visit_Constant(self, node):
            node.value = "CONST" if not isinstance(node.value, bool) else node.value
            return node

    anon = _Anonymizer().visit(tree)
    dumped = ast.dump(anon, annotate_fields=False)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


@dataclasses.dataclass
class VerifiedPattern:
    key: str
    source: str
    z3_status: str
    z3_message: str
    model_values: dict | None = None
    timestamp: float = dataclasses.field(default_factory=time.time)


class HDCMemory:

    def __init__(self, dim: int = 10000, device: str = "cpu"):
        self.dim = dim
        self.device = device
        self._item_memory: dict[str, torch.Tensor] = {}
        self.trace = torch.zeros(dim, device=device)
        self._verified_patterns: dict[str, VerifiedPattern] = {}
        self._pattern_trace = torch.zeros(dim, device=device)

    def random_hypervector(self, seed: int | None = None) -> torch.Tensor:
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        bits = torch.randint(0, 2, (self.dim,), generator=gen).float()
        return (bits * 2 - 1).to(self.device)

    def symbol(self, key: str) -> torch.Tensor:
        if key not in self._item_memory:
            self._item_memory[key] = self.random_hypervector(
                seed=_seed_from_string(key)
            )
        return self._item_memory[key]

    @staticmethod
    def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b

    @staticmethod
    def bundle(vectors: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(vectors, dim=0).sum(dim=0)
        return torch.sign(stacked + 1e-12)

    @staticmethod
    def permute(v: torch.Tensor, shift: int = 1) -> torch.Tensor:
        return torch.roll(v, shifts=shift, dims=-1)

    def write(self, key: str, value_hv: torch.Tensor):
        key_hv = self.symbol(key)
        self.trace = self.trace + self.bind(key_hv, torch.sign(value_hv + 1e-12))

    def read(self, key: str) -> torch.Tensor:
        key_hv = self.symbol(key)
        return torch.sign(self.bind(key_hv, torch.sign(self.trace + 1e-12)))

    def cleanup(
        self, noisy_hv: torch.Tensor, candidates: dict[str, torch.Tensor] | None = None
    ) -> tuple[str | None, float]:
        pool = candidates if candidates is not None else self._item_memory
        if not pool:
            return (None, 0.0)
        best_key, best_sim = (None, -1.0)
        for key, hv in pool.items():
            sim = torch.nn.functional.cosine_similarity(
                noisy_hv.unsqueeze(0), hv.unsqueeze(0)
            ).item()
            if sim > best_sim:
                best_key, best_sim = (key, sim)
        return (best_key, best_sim)

    def store_verified_pattern(
        self,
        source: str,
        z3_status: str,
        z3_message: str,
        model_values: dict | None = None,
        key: str | None = None,
    ) -> str:
        if z3_status != "sat":
            raise ValueError(
                "HDC Memory only accepts patterns verified as SAT (safe, usable rules). For UNSAT, use the SelfCorrectionEngine history instead."
            )
        resolved_key = (
            key
            or hash_ast(source)
            or hashlib.sha256(source.encode("utf-8")).hexdigest()
        )
        self._verified_patterns[resolved_key] = VerifiedPattern(
            key=resolved_key,
            source=source,
            z3_status=z3_status,
            z3_message=z3_message,
            model_values=model_values,
        )
        pattern_hv = self.symbol(f"pattern::{resolved_key}")
        self._pattern_trace = self._pattern_trace + pattern_hv
        return resolved_key

    def query_verified_pattern(
        self, source_or_key: str, fuzzy_threshold: float = 0.15
    ) -> VerifiedPattern | None:
        exact_key = hash_ast(source_or_key) or source_or_key
        if exact_key in self._verified_patterns:
            return self._verified_patterns[exact_key]
        if source_or_key in self._verified_patterns:
            return self._verified_patterns[source_or_key]
        if not self._verified_patterns:
            return None
        query_hv = self.symbol(f"pattern::{exact_key}")
        candidates = {k: self.symbol(f"pattern::{k}") for k in self._verified_patterns}
        best_key, best_sim = self.cleanup(query_hv, candidates=candidates)
        if best_key is not None and best_sim >= fuzzy_threshold:
            return self._verified_patterns[best_key]
        return None

    def n_verified_patterns(self) -> int:
        return len(self._verified_patterns)

    def save(self, path: str):
        torch.save(
            {
                "trace": self.trace,
                "item_memory": self._item_memory,
                "dim": self.dim,
                "pattern_trace": self._pattern_trace,
                "verified_patterns": self._verified_patterns,
            },
            path,
        )

    def load(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.trace = state["trace"].to(self.device)
        self._item_memory = {
            k: v.to(self.device) for k, v in state["item_memory"].items()
        }
        self.dim = state["dim"]
        self._pattern_trace = state.get(
            "pattern_trace", torch.zeros(self.dim, device=self.device)
        ).to(self.device)
        self._verified_patterns = state.get("verified_patterns", {})


class HDCBridge(nn.Module):

    def __init__(self, latent_dim: int, hdc_dim: int = 10000):
        super().__init__()
        self.proj_to_hdc = nn.Linear(latent_dim, hdc_dim)
        self.proj_from_hdc = nn.Linear(hdc_dim, latent_dim)

    def to_hdc(self, latent: torch.Tensor) -> torch.Tensor:
        projected = self.proj_to_hdc(latent)
        quantized = torch.sign(projected + 1e-12)
        return quantized.detach() + projected - projected.detach()

    def from_hdc(self, hv: torch.Tensor) -> torch.Tensor:
        return self.proj_from_hdc(hv)
