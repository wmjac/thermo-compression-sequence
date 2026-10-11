"""Compare to the supplied sources, including parameter gradients and an Adam step."""
import numpy as np
import pytest
import torch
from scipy.constants import Boltzmann
from mixture_models.models import fixed_length_transformer_mlp as current
from mixture_models.losses import pressure_loss
from mixture_models.pressure import total_pressure


@pytest.mark.parametrize("dtype,rtol,atol", [("float64", 2e-8, 2e-10), ("float32", 3e-4, 3e-6)])
def test_legacy_pressure_loss_gradients_and_adam(reference_module, dtype, rtol, atol):
    reference = reference_module("pressure_predictor_tf_linc_meanz_fe_fn.py")
    torch.manual_seed(51)
    legacy = reference.integrated_pressure_predictor(3).to(dtype=getattr(torch, dtype))
    model = current.build_model(3, dtype)
    model.load_state_dict(legacy.state_dict())
    tokens = torch.randint(0, 20, (4, 2, 20))
    concentrations = torch.rand((4, 2, 1), dtype=getattr(torch, dtype)) * 0.12 + 0.001
    target = torch.randn((4, 1), dtype=getattr(torch, dtype)) * 0.1
    inputs = {"tokens": tokens, "concentrations": concentrations, "ctot": concentrations.sum(dim=1)}
    legacy_x = torch.cat([tokens.to(dtype=getattr(torch, dtype)), concentrations * 10], dim=-1)
    atm_per_beta_p = Boltzmann * 300 * 1e27 / 101325
    legacy.train()
    model.train()
    torch.manual_seed(914)
    prediction_atm = legacy(legacy_x)
    legacy_loss = (((target * atm_per_beta_p - prediction_atm) /
                    (reference.convert * legacy_x[:, :, -1].sum(dim=1, keepdim=True))) ** 2).mean()
    torch.manual_seed(914)
    prediction = total_pressure(model, inputs)
    loss = pressure_loss(prediction, target, inputs["ctot"])
    torch.testing.assert_close(prediction, prediction_atm / atm_per_beta_p, rtol=rtol, atol=atol)
    torch.testing.assert_close(loss, legacy_loss, rtol=rtol, atol=atol)
    loss.backward()
    legacy_loss.backward()
    for left, right in zip(model.parameters(), legacy.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=rtol, atol=atol)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(legacy.parameters(), 1.0)
    optimizer = current.build_optimizer(model, 1e-3, current.training_protocol())
    legacy_optimizer = torch.optim.Adam(legacy.parameters(), lr=1e-3)
    optimizer.step()
    legacy_optimizer.step()
    for (name, left), (_, right) in zip(model.named_parameters(), legacy.named_parameters()):
        if dtype == "float32" and name.endswith("self_attn.in_proj_bias"):
            # Attention key bias adds the same constant to every softmax logit;
            # its exact gradient is zero. FP32 reduction noise is amplified by
            # Adam's epsilon in this unobservable parameter direction.
            torch.testing.assert_close(left[:16], right[:16], rtol=rtol, atol=atol)
            torch.testing.assert_close(left[32:], right[32:], rtol=rtol, atol=atol)
            assert torch.max(torch.abs(left[16:32] - right[16:32])) < 2e-5
        else:
            torch.testing.assert_close(left, right, rtol=rtol, atol=atol)
    model.eval()
    legacy.eval()
    post_step = total_pressure(model, inputs)
    legacy_post_step = legacy(legacy_x.requires_grad_(True)) / atm_per_beta_p
    torch.testing.assert_close(post_step, legacy_post_step, rtol=rtol, atol=atol)
    torch.testing.assert_close(pressure_loss(post_step, target, inputs["ctot"]),
                               pressure_loss(legacy_post_step, target, inputs["ctot"]), rtol=rtol, atol=atol)


def test_authoritative_decoder_derivatives(reference_module):
    reference = reference_module("fe_predictor_tf_linc_meanz.py")
    torch.manual_seed(45)
    original = reference.integrated_fe_predictor(4).double()
    model = current.build_model(4, "float64")
    model.load_state_dict(original.state_dict())
    x = torch.randn(7, 4, dtype=torch.float64) * 0.06
    original_terms = original.model3(x, grad_out=True, hessian_out=True)
    terms = model.thermodynamics(x, hessian=True)
    for name, expected in zip(("free_energy_excess", "pressure_excess", "gradient", "hessian"), original_terms):
        torch.testing.assert_close(terms[name], expected, rtol=1e-12, atol=1e-12)
    zero = model.thermodynamics(torch.zeros(1, 4, dtype=torch.float64))
    for name in ("free_energy_excess", "pressure_excess", "gradient"):
        torch.testing.assert_close(zero[name], torch.zeros_like(zero[name]), rtol=0, atol=0)


def test_model_owns_length_validation_and_reversal_symmetry():
    with pytest.raises(ValueError, match="requires 20"):
        current.validate_sequences(["A" * 19])
    with pytest.raises(ValueError, match="Unsupported residues"):
        current.validate_sequences(["A" * 19 + "X"])
    model = current.build_model(2)
    model.eval()
    sequence = "MGKTRADEYVLQWFSHNPCI"
    with torch.no_grad():
        latent = model.encode_sequences([sequence, sequence[::-1]])
    torch.testing.assert_close(latent[0], latent[1], rtol=1e-6, atol=1e-6)
