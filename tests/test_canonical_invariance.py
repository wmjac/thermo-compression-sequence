import numpy as np
import torch
from mixture_models.alignment import align_ensemble
from mixture_models.models import fixed_length_transformer_mlp as module


def test_generalized_alignment_preserves_thermodynamic_distances_for_arbitrary_k():
    rng = np.random.default_rng(51)
    native = rng.normal(size=(7, 25, 3))
    matrices = rng.normal(size=(7, 3, 3))
    roots = np.array([matrix.T @ matrix + np.eye(3) for matrix in matrices])
    aligned = align_ensemble(native, roots)
    mapped = np.einsum("knd,ked->kne", native, aligned["transforms"])
    np.testing.assert_allclose(mapped, aligned["aligned"], rtol=1e-12, atol=1e-12)
    for member in range(7):
        before = native[member] @ roots[member].T
        np.testing.assert_allclose(before @ before.T, mapped[member] @ mapped[member].T, rtol=1e-11, atol=1e-11)


def test_affine_coordinate_change_preserves_pressure_gradients_and_hessians():
    torch.manual_seed(10)
    model = module.build_model(3, "float64")
    model.eval()
    rng = np.random.default_rng(57)
    transform = rng.normal(size=(3, 3)) + np.eye(3) * 3
    inverse = np.linalg.inv(transform)
    x = rng.normal(size=(5, 3)) * 0.03
    canonical = x @ transform.T
    restored = canonical @ inverse.T
    native_terms = model.thermodynamics(torch.tensor(x), hessian=True)
    terms = model.thermodynamics(torch.tensor(restored), hessian=True)
    for name in ("free_energy_excess", "pressure_excess"):
        torch.testing.assert_close(terms[name], native_terms[name], rtol=1e-12, atol=1e-12)
    gradient_canonical = terms["gradient"].detach().numpy() @ inverse
    hessian_canonical = np.einsum("ij,bjl,lm->bim", inverse.T, terms["hessian"].detach().numpy(), inverse)
    # Chemical potentials and directional curvature must be coordinate-invariant.
    direction = rng.normal(size=(5, 3))
    direction_canonical = direction @ transform.T
    np.testing.assert_allclose(np.sum(gradient_canonical * direction_canonical, axis=1),
                               np.sum(native_terms["gradient"].detach().numpy() * direction, axis=1), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.einsum("bi,bij,bj->b", direction_canonical, hessian_canonical, direction_canonical),
                               np.einsum("bi,bij,bj->b", direction, native_terms["hessian"].detach().numpy(), direction), rtol=1e-12, atol=1e-12)
