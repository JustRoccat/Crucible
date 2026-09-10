import importlib.util
import pathlib
import torch
from src.backbone import CustomMambaBackbone
from src.model import NeuroSymbolicModel


def _load_train_module():
    path = pathlib.Path(__file__).resolve().parents[1] / "train.py"
    spec = importlib.util.spec_from_file_location("train_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_module = _load_train_module()


def _tiny_model():
    backbone = CustomMambaBackbone(
        vocab_size=32, dim=16, n_layers=1, state_dim=4, scan_chunk_size=8
    )
    return NeuroSymbolicModel(backbone=backbone, vocab_size=32, ttt_rank=4, hdc_dim=64)


def test_frozen_backbone_gets_single_group():
    model = _tiny_model()
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    cfg = {"train": {"lr": 0.001, "weight_decay": 0.01}}
    optimizer = train_module.build_optimizer(model, cfg)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 0.001


def test_unfrozen_backbone_without_explicit_lr_uses_shared_lr_and_warns(capsys):
    model = _tiny_model()
    cfg = {"train": {"lr": 0.001, "weight_decay": 0.01}}
    optimizer = train_module.build_optimizer(model, cfg)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    lrs = {g["lr"] for g in optimizer.param_groups}
    assert lrs == {0.001}


def test_unfrozen_backbone_with_explicit_backbone_lr_uses_two_groups():
    model = _tiny_model()
    cfg = {"train": {"lr": 0.001, "backbone_lr": 1e-05, "weight_decay": 0.01}}
    optimizer = train_module.build_optimizer(model, cfg)
    lrs = sorted((g["lr"] for g in optimizer.param_groups))
    assert lrs == [1e-05, 0.001]
    n_backbone = sum(
        (p.numel() for p in model.backbone.parameters() if p.requires_grad)
    )
    group_sizes = {
        g["lr"]: sum((p.numel() for p in g["params"])) for g in optimizer.param_groups
    }
    assert group_sizes[1e-05] == n_backbone


def test_optimizer_step_actually_updates_params_in_both_groups():
    model = _tiny_model()
    cfg = {"train": {"lr": 0.01, "backbone_lr": 0.01, "weight_decay": 0.0}}
    optimizer = train_module.build_optimizer(model, cfg)
    before = {
        n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
    }
    x = torch.randint(0, 32, (2, 10))
    y = torch.randint(0, 32, (2, 10))
    out = model(x, targets=y)
    loss = out["lm_loss"] + out["jepa_loss"]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    changed = any(
        (
            not torch.equal(before[n], p)
            for n, p in model.named_parameters()
            if p.requires_grad
        )
    )
    assert changed
