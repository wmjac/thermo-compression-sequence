import sys
from pathlib import Path
import numpy as np
from mixture_models.canonical import sample_composition


def test_sparse_prior_matches_authoritative_sampler(reference_module):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference_sources"))
    reference = reference_module("mixture_distribution.py")
    for n in (2, 10000):
        original_rng = np.random.default_rng(77)
        current_rng = np.random.default_rng(77)
        for _ in range(100):
            indices, weights = sample_composition(n, 0.3651, 1e-6, 50, current_rng)
            expected_indices, expected_weights = reference.sample_sparse_dirichlet_composition(
                n, 0.3651, eps_mass=1e-6, max_components=50, rng=original_rng)
            np.testing.assert_array_equal(indices, expected_indices)
            np.testing.assert_array_equal(weights, expected_weights)


def test_alignment_matches_source_functions(reference_module):
    from mixture_models.alignment import align_ensemble
    reference = reference_module("align_latent_uncentered_pca.py")
    rng = np.random.default_rng(33)
    native = rng.normal(size=(5, 30, 3))
    matrices = rng.normal(size=(5, 3, 3))
    roots = np.array([matrix.T @ matrix + np.eye(3) for matrix in matrices])
    transformed = np.array([z @ root.T for z, root in zip(native, roots)])
    original_pca = [reference.pca_with_unique_signs(z) for z in transformed]
    aligned, rotations, mean, _ = reference.generalized_procrustes(
        [item[0] for item in original_pca], max_iter=200, tol=1e-8, verbose=False)
    final_rotation = reference.pca_with_unique_signs(mean)[1]
    result = align_ensemble(native, roots)
    np.testing.assert_allclose(result["aligned"], np.array(aligned) @ final_rotation, rtol=1e-11, atol=1e-11)
