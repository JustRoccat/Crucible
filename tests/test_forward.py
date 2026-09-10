import torch
from src.model import NeuroSymbolicModel
from src.backbone import CustomMambaBackbone
from src.backbone.custom_mamba import chunked_selective_scan
from src.hdc import HDCMemory
from src.symbolic import SymbolicSpec
from src.symbolic.z3_engine import Z3LogicalEngine
from src.tokenizer import ProductionTokenizer


def _tiny_model():
    backbone = CustomMambaBackbone(
        vocab_size=64, dim=32, n_layers=2, state_dim=8, scan_chunk_size=8
    )
    return NeuroSymbolicModel(
        backbone=backbone,
        vocab_size=64,
        ttt_rank=4,
        jepa_hidden_mult=2,
        hdc_dim=512,
        mask_ratio=0.25,
        max_self_correction_attempts=3,
    )


def test_tokenizer_byte_fallback_roundtrip():
    tok = ProductionTokenizer.from_config({"mode": "byte_fallback"})
    text = "Zażółć gęślą jaźń"
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert tok.vocab_size == 256


def _reference_sequential_scan(Abar, Bx, C):
    Bsz, L, D, N = Abar.shape
    h = Abar.new_zeros(Bsz, D, N)
    ys = []
    for t in range(L):
        h = Abar[:, t] * h + Bx[:, t]
        ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]))
    return torch.stack(ys, dim=1)


def test_chunked_scan_matches_sequential_reference():
    torch.manual_seed(0)
    Bsz, L, D, N = (2, 37, 6, 5)
    Abar = torch.rand(Bsz, L, D, N) * 0.9 + 0.05
    Bx = torch.randn(Bsz, L, D, N) * 0.1
    C = torch.randn(Bsz, L, N)
    y_chunked = chunked_selective_scan(Abar, Bx, C, chunk_size=8)
    y_ref = _reference_sequential_scan(Abar, Bx, C)
    assert torch.allclose(y_chunked, y_ref, atol=0.0001)


def test_custom_mamba_backbone_forward_shape():
    backbone = CustomMambaBackbone(
        vocab_size=64, dim=32, n_layers=2, state_dim=8, scan_chunk_size=8
    )
    ids = torch.randint(0, 64, (2, 20))
    out = backbone(ids)
    assert out.shape == (2, 20, 32)


def test_fast_weights_low_rank_and_no_retained_graph():
    from src.ttt import MicroLoRAFastWeights

    dim, rank = (32, 4)
    ffw = MicroLoRAFastWeights(dim, rank=rank, inner_steps=2)
    n_full = dim * dim
    n_lora = sum((p.numel() for p in ffw.parameters()))
    assert n_lora < n_full
    assert n_lora == 2 * dim * rank
    x = torch.randn(2, 5, dim, requires_grad=True)
    y = ffw(x)
    loss = y.pow(2).mean()
    loss.backward()
    assert ffw.base_A.grad is not None
    assert ffw.base_B.grad is not None


def test_z3_check_spec_sat_and_unsat():
    sat_spec = SymbolicSpec(var_names=["n"], constraints=["n > 0", "n < 10"])
    result = Z3LogicalEngine.check_spec(sat_spec)
    assert result.status == "sat"
    assert result.model_values is not None
    unsat_spec = SymbolicSpec(var_names=["n"], constraints=["n > 0", "n < 0"])
    result = Z3LogicalEngine.check_spec(unsat_spec)
    assert result.status == "unsat"
    assert "n > 0" in result.unsat_core or "n < 0" in result.unsat_core


def test_self_correction_loop_recovers_from_unsat():
    model = _tiny_model()
    calls = {"n": 0}

    def propose(context, history):
        calls["n"] += 1
        if calls["n"] == 1:
            return SymbolicSpec(var_names=["n"], constraints=["n > 0", "n < 0"])
        return SymbolicSpec(var_names=["n"], constraints=["n > 0", "n < 10"])

    outcome = model.self_correct(propose, initial_context="test")
    assert outcome.success is True
    assert outcome.n_attempts == 2
    assert outcome.attempts[0].result.status == "unsat"
    assert outcome.attempts[1].result.status == "sat"


def test_self_correction_loop_exhausts_attempts_gracefully():
    model = _tiny_model()

    def always_unsat(context, history):
        return SymbolicSpec(var_names=["n"], constraints=["n > 0", "n < 0"])

    outcome = model.self_correct(always_unsat, initial_context="test")
    assert outcome.success is False
    assert outcome.n_attempts == model.self_correction_engine.max_attempts


def test_hdc_stores_and_recovers_verified_pattern():
    model = _tiny_model()
    mem = HDCMemory(dim=512)

    def propose(context, history):
        return SymbolicSpec(var_names=["n"], constraints=["n > 0", "n < 10"])

    outcome = model.self_correct(propose, hdc_memory=mem)
    assert outcome.success is True
    assert mem.n_verified_patterns() == 1
    pattern = mem.query_verified_pattern("n > 0\nn < 10")
    assert pattern is not None
    assert pattern.z3_status == "sat"


def test_hdc_bind_bundle_roundtrip():
    mem = HDCMemory(dim=256)
    key_hv = mem.symbol("k1")
    value_hv = mem.random_hypervector(seed=42)
    bound = mem.bind(key_hv, value_hv)
    recovered = mem.bind(key_hv, bound)
    assert torch.allclose(recovered, value_hv)


def test_forward_shapes():
    model = _tiny_model()
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    out = model(x, targets=y)
    assert out["lm_logits"].shape == (2, 16, 64)
    assert out["latent_concepts"].shape == (2, 16, 32)
    assert out["lm_loss"].item() > 0
    assert out["jepa_loss"].item() >= 0


def test_backward_and_step():
    model = _tiny_model()
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    out = model(x, targets=y)
    loss = out["lm_loss"] + out["jepa_loss"]
    optim.zero_grad()
    loss.backward()
    assert model.fast_weights.base_A.grad is not None
    assert model.backbone.layers[0]["mamba"].A_log.grad is not None
    assert model.jepa_encoder.net[0].weight.grad is not None
    optim.step()
    model.update_target_encoder()


def test_eval_mode_no_grad():
    model = _tiny_model()
    model.eval()
    x = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        out = model(x)
    assert out["lm_logits"].shape == (1, 8, 64)
