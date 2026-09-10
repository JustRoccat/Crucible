import importlib.util
import pathlib
import torch
from src.backbone import CustomMambaBackbone
from src.model import NeuroSymbolicModel


def _load_evaluate_module():
    path = pathlib.Path(__file__).resolve().parents[1] / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate_module = _load_evaluate_module()


def test_wrap_prompt_format():
    wrapped = evaluate_module.wrap_prompt("Does Tom like milk?")
    assert wrapped == "### Instruction\nDoes Tom like milk?\n\n### Response\n"
    assert wrapped.endswith("### Response\n")


def test_parse_symbolic_spec_happy_path():
    text = "some noise before\nVARS: n, m\nCONSTRAINT: n > 0\nCONSTRAINT: n < m\ntrailing noise"
    var_names, constraints = evaluate_module._parse_symbolic_spec(text)
    assert var_names == ["n", "m"]
    assert constraints == ["n > 0", "n < m"]


def test_parse_symbolic_spec_ignores_non_identifier_var_names():
    text = "VARS: n, 3bad, m\nCONSTRAINT: n > 0"
    var_names, _ = evaluate_module._parse_symbolic_spec(text)
    assert var_names == ["n", "m"]


def test_parse_symbolic_spec_missing_format_returns_empty():
    var_names, constraints = evaluate_module._parse_symbolic_spec(
        "just some random model output"
    )
    assert var_names == []
    assert constraints == []


def test_parse_symbolic_spec_is_case_insensitive_for_labels():
    text = "vars: n\nconstraint: n > 0"
    var_names, constraints = evaluate_module._parse_symbolic_spec(text)
    assert var_names == ["n"]
    assert constraints == ["n > 0"]


def test_generate_greedy_is_deterministic_and_produces_requested_length():
    torch.manual_seed(0)
    backbone = CustomMambaBackbone(
        vocab_size=32, dim=16, n_layers=1, state_dim=4, scan_chunk_size=8
    )
    model = NeuroSymbolicModel(backbone=backbone, vocab_size=32, ttt_rank=4, hdc_dim=64)
    model.eval()
    prompt_ids = [1, 2, 3]
    out1 = evaluate_module.generate(
        model, prompt_ids, n_tokens=5, device=torch.device("cpu"), greedy=True
    )
    out2 = evaluate_module.generate(
        model, prompt_ids, n_tokens=5, device=torch.device("cpu"), greedy=True
    )
    assert len(out1) == len(prompt_ids) + 5
    assert out1[: len(prompt_ids)] == prompt_ids
    assert out1 == out2


def test_generate_sampling_respects_n_tokens():
    torch.manual_seed(0)
    backbone = CustomMambaBackbone(
        vocab_size=32, dim=16, n_layers=1, state_dim=4, scan_chunk_size=8
    )
    model = NeuroSymbolicModel(backbone=backbone, vocab_size=32, ttt_rank=4, hdc_dim=64)
    model.eval()
    out = evaluate_module.generate(
        model, [1, 2, 3], n_tokens=7, device=torch.device("cpu"), greedy=False
    )
    assert len(out) == 3 + 7
