import pytest
import torch
from src.hdc.memory import HDCBridge, HDCMemory, hash_ast


def test_symbol_is_deterministic_for_same_key():
    mem = HDCMemory(dim=256)
    a = mem.symbol("hello")
    b = mem.symbol("hello")
    assert torch.equal(a, b)


def test_symbol_differs_for_different_keys():
    mem = HDCMemory(dim=1024)
    a = mem.symbol("hello")
    b = mem.symbol("world")
    assert not torch.equal(a, b)
    sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    assert abs(sim) < 0.2


def test_bundle_recovers_majority_of_bound_pairs():
    mem = HDCMemory(dim=2048)
    keys = ["k1", "k2", "k3"]
    values = {k: mem.random_hypervector(seed=i) for i, k in enumerate(keys)}
    bound_pairs = [mem.bind(mem.symbol(k), values[k]) for k in keys]
    bundled = mem.bundle(bound_pairs)
    for k in keys:
        recovered = mem.bind(mem.symbol(k), bundled)
        sim = torch.nn.functional.cosine_similarity(
            recovered.unsqueeze(0), values[k].unsqueeze(0)
        ).item()
        assert sim > 0.3


def test_write_read_roundtrip_single_key():
    mem = HDCMemory(dim=1024)
    value_hv = mem.random_hypervector(seed=7)
    mem.write("only_key", value_hv)
    recovered = mem.read("only_key")
    assert torch.equal(recovered, torch.sign(value_hv))


def test_cleanup_finds_nearest_symbol():
    mem = HDCMemory(dim=2048)
    mem.symbol("cat")
    mem.symbol("dog")
    mem.symbol("car")
    noisy = mem.symbol("cat").clone()
    flip_idx = torch.randperm(2048)[:100]
    noisy[flip_idx] *= -1
    best_key, best_sim = mem.cleanup(noisy)
    assert best_key == "cat"
    assert best_sim > 0.5


def test_cleanup_on_empty_pool_returns_none():
    mem = HDCMemory(dim=128)
    key, sim = mem.cleanup(mem.random_hypervector(seed=0), candidates={})
    assert key is None
    assert sim == 0.0


def test_hash_ast_ignores_variable_and_constant_names():
    h1 = hash_ast("x + 1 > 0")
    h2 = hash_ast("y + 999 > 0")
    assert h1 == h2


def test_hash_ast_distinguishes_different_shapes():
    h1 = hash_ast("x + 1 > 0")
    h2 = hash_ast("x - 1 > 0")
    assert h1 != h2


def test_hash_ast_returns_none_for_unparseable_code():
    assert hash_ast("n > 0 and and and") is None


def test_store_verified_pattern_rejects_non_sat_status():
    mem = HDCMemory(dim=256)
    with pytest.raises(ValueError):
        mem.store_verified_pattern(source="n > 0", z3_status="unsat", z3_message="nope")


def test_store_and_query_verified_pattern_exact_match():
    mem = HDCMemory(dim=512)
    key = mem.store_verified_pattern(
        source="n > 0\nn < 10", z3_status="sat", z3_message="SAT: n=5"
    )
    pattern = mem.query_verified_pattern("n > 0\nn < 10")
    assert pattern is not None
    assert pattern.key == key
    assert pattern.z3_status == "sat"


def test_query_verified_pattern_unknown_returns_none_when_store_empty():
    mem = HDCMemory(dim=256)
    assert mem.query_verified_pattern("anything at all") is None


def test_query_verified_pattern_fuzzy_match_requires_threshold():
    mem = HDCMemory(dim=512)
    mem.store_verified_pattern(source="n > 0\nn < 10", z3_status="sat", z3_message="ok")
    result = mem.query_verified_pattern(
        "completely_unrelated_query_string", fuzzy_threshold=0.5
    )
    assert result is None


def test_n_verified_patterns_counts_entries():
    mem = HDCMemory(dim=256)
    assert mem.n_verified_patterns() == 0
    mem.store_verified_pattern(
        source="a > 0", z3_status="sat", z3_message="ok", key="pattern_a"
    )
    mem.store_verified_pattern(
        source="b > 0", z3_status="sat", z3_message="ok", key="pattern_b"
    )
    assert mem.n_verified_patterns() == 2


def test_store_verified_pattern_with_same_ast_shape_deduplicates():
    mem = HDCMemory(dim=256)
    mem.store_verified_pattern(source="a > 0", z3_status="sat", z3_message="first")
    mem.store_verified_pattern(source="b > 0", z3_status="sat", z3_message="second")
    assert mem.n_verified_patterns() == 1
    pattern = mem.query_verified_pattern("a > 0")
    assert pattern.z3_message == "second"


def test_save_load_roundtrip_preserves_associative_and_pattern_layers(tmp_path):
    mem = HDCMemory(dim=256)
    mem.write("k", mem.random_hypervector(seed=1))
    mem.store_verified_pattern(
        source="n > 0", z3_status="sat", z3_message="ok", model_values={"n": 1}
    )
    path = tmp_path / "hdc.pt"
    mem.save(str(path))
    loaded = HDCMemory(dim=1)
    loaded.load(str(path))
    assert loaded.dim == 256
    assert torch.equal(loaded.trace, mem.trace)
    assert loaded.n_verified_patterns() == 1
    pattern = loaded.query_verified_pattern("n > 0")
    assert pattern is not None
    assert pattern.model_values == {"n": 1}


def test_hdc_bridge_to_hdc_is_bipolar():
    bridge = HDCBridge(latent_dim=16, hdc_dim=64)
    latent = torch.randn(4, 16)
    hv = bridge.to_hdc(latent)
    assert set(torch.unique(torch.sign(hv)).tolist()) <= {-1.0, 0.0, 1.0}
    assert hv.shape == (4, 64)


def test_hdc_bridge_straight_through_gradient_flows():
    bridge = HDCBridge(latent_dim=8, hdc_dim=32)
    latent = torch.randn(2, 8, requires_grad=True)
    hv = bridge.to_hdc(latent)
    loss = hv.sum()
    loss.backward()
    assert latent.grad is not None
    assert bridge.proj_to_hdc.weight.grad is not None
