import torch
from src.jepa.predictor import JEPAPredictor
from src.model import NeuroSymbolicModel
from src.backbone import CustomMambaBackbone


def _tiny_model(mask_ratio=0.5):
    backbone = CustomMambaBackbone(
        vocab_size=64, dim=32, n_layers=2, state_dim=8, scan_chunk_size=8
    )
    return NeuroSymbolicModel(
        backbone=backbone,
        vocab_size=64,
        ttt_rank=4,
        jepa_hidden_mult=2,
        hdc_dim=512,
        mask_ratio=mask_ratio,
        max_self_correction_attempts=3,
    )


def test_predictor_predictions_depend_on_context_not_just_mask_token():
    torch.manual_seed(0)
    B, L, D = (1, 10, 16)
    predictor = JEPAPredictor(dim=D, hidden_mult=2)
    x = torch.randn(B, L, D)
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[0, [2, 8]] = True
    x_masked = x.clone()
    x_masked[mask] = predictor.mask_token.to(x.dtype)
    pred = predictor(x_masked)
    assert not torch.allclose(
        pred[0, 2], pred[0, 8]
    ), "Predictions at two masked positions with different context are identical - the predictor is probably not mixing context again (regression of the 'dead JEPA target' bug)."


def test_predictor_same_context_same_prediction_deterministic():
    torch.manual_seed(1)
    B, L, D = (1, 6, 16)
    predictor = JEPAPredictor(dim=D, hidden_mult=2)
    predictor.eval()
    x = torch.randn(B, L, D)
    x_masked = x.clone()
    x_masked[0, 3] = predictor.mask_token.to(x.dtype)
    with torch.no_grad():
        out1 = predictor(x_masked)
        out2 = predictor(x_masked.clone())
    assert torch.allclose(out1, out2)


def test_predictor_context_mixing_is_causal():
    torch.manual_seed(0)
    B, L, D = (2, 12, 16)
    predictor = JEPAPredictor(dim=D, hidden_mult=2, context_kernel=4)
    x = torch.randn(B, L, D)
    t = 5
    out_before = predictor(x)
    x_perturbed = x.clone()
    x_perturbed[:, t + 1 :, :] += 50.0
    out_after = predictor(x_perturbed)
    assert torch.equal(
        out_before[:, : t + 1], out_after[:, : t + 1]
    ), "A change at positions > t affected the predictor output at positions <= t - context mixing stopped being causal."
    assert not torch.allclose(out_before[:, t + 1 :], out_after[:, t + 1 :])


def test_lm_logits_do_not_depend_on_jepa_predictor_output():
    torch.manual_seed(0)
    model = _tiny_model(mask_ratio=0.5)
    model.eval()
    x = torch.randint(0, 64, (2, 16))
    real_apply_mask = model.predictor.apply_mask

    def mask_all(latent, ratio):
        _, mask = real_apply_mask(latent, ratio)
        return (latent, torch.ones_like(mask))

    def mask_none(latent, ratio):
        _, mask = real_apply_mask(latent, ratio)
        return (latent, torch.zeros_like(mask))

    with torch.no_grad():
        model.predictor.apply_mask = mask_all
        out_all_masked = model(x)
        model.predictor.apply_mask = mask_none
        out_none_masked = model(x)
    model.predictor.apply_mask = real_apply_mask
    assert torch.allclose(
        out_all_masked["lm_logits"], out_none_masked["lm_logits"], atol=1e-05
    ), "lm_logits changed depending on the JEPA mask - lm_head again depends (indirectly) on `pred`, not just on `latent_concepts`."


def test_lm_loss_gradient_does_not_flow_through_jepa_predictor_net():
    torch.manual_seed(0)
    model = _tiny_model(mask_ratio=0.5)
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    out = model(x, targets=y)
    out["lm_loss"].backward()
    for name, p in model.predictor.named_parameters():
        if p.grad is not None:
            assert torch.all(
                p.grad == 0
            ), f"predictor.{name} got a non-zero gradient from lm_loss alone - lm_head appears to depend on JEPAPredictor's output."


def test_jepa_loss_gradient_still_flows_through_predictor():
    torch.manual_seed(0)
    model = _tiny_model(mask_ratio=0.5)
    x = torch.randint(0, 64, (2, 16))
    out = model(x)
    out["jepa_loss"].backward()
    grads = [p.grad for p in model.predictor.net.parameters()]
    assert any(
        (g is not None and torch.any(g != 0) for g in grads)
    ), "jepa_loss is not training the predictor - the JEPA target is dead."


def test_mask_ratio_zero_means_no_masking():
    torch.manual_seed(0)
    model = _tiny_model(mask_ratio=0.0)
    x = torch.randint(0, 64, (2, 16))
    out = model(x)
    assert out["mask"].sum().item() == 0
    x_single = torch.randint(0, 64, (3, 1))
    out_single = model(x_single, mask_ratio=0.0)
    assert out_single["mask"].sum().item() == 0
