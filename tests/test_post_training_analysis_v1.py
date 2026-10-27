"""Scientific checks for the additive post-training scripts; run with unittest."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from _geometry_v1 import basis, eigenblocks, gram, mixing_statistics, procrustes, subspace_comparison
from _post_training_common_v1 import chemical_weights, check_common_measures, prepare_output
from _responses_v1 import canonical_samples, mixture_latent, response_eigensystem, response_pair
from enumerate_sequences_v1 import enumerate_sequences, sequence_count


class SequenceTests(unittest.TestCase):
    def test_reversal_orbits_even_and_odd(self):
        for length in range(1, 7):
            oriented = list(enumerate_sequences("AF", length, False))
            unique = list(enumerate_sequences("AF", length, True))
            self.assertEqual(len(oriented), 2**length)
            self.assertEqual(len(unique), sequence_count(2, length, True))
            self.assertEqual(set(unique), {min(s, s[::-1]) for s in oriented})
            self.assertEqual(unique, sorted(unique))
        self.assertEqual(sequence_count(2, 20, True), 524800)
        self.assertEqual(sequence_count(2, 20, False), 1048576)

    def test_chemical_measure_fallback_and_explicit_weights(self):
        weights, metadata = chemical_weights({}, ["AF", "FA"])
        np.testing.assert_array_equal(weights, [.5, .5])
        self.assertIn("default", metadata["source"])
        weights, _ = chemical_weights({"rho_chem": {"weights": {"AF": .2, "FA": .8}}}, ["FA", "AF"])
        np.testing.assert_array_equal(weights, [.8, .2])
        with self.assertRaises(ValueError):
            chemical_weights({"rho_chem": {"distribution": "unknown"}}, ["AF", "FA"])
        with self.assertRaises(ValueError):
            chemical_weights({"rho_chem": {"weights": [1., 1.]}}, ["AF", "FA"])

    def test_same_measure_reorders_identities_and_rejects_prior_change(self):
        reference = SimpleNamespace(dimension=2, config={"rho_mix": {"ctot_max": .25}},
            sequences=["AF", "FA"], weights=np.array([.2, .8]),
            aligned=np.array([[[1.,2.],[3.,4.]]]), native=np.array([[[1.,2.],[3.,4.]]]))
        other = SimpleNamespace(dimension=1, config={"rho_mix": {"ctot_max": .25}},
            sequences=["FA", "AF"], weights=np.array([.8, .2]),
            aligned=np.array([[[9.],[7.]]]), native=np.array([[[9.],[7.]]]))
        check_common_measures([other],reference)
        np.testing.assert_array_equal(other.aligned,np.array([[[7.],[9.]]]))
        np.testing.assert_array_equal(other.weights,reference.weights)
        other.config["rho_mix"]["ctot_max"] = .5
        with self.assertRaises(ValueError):
            check_common_measures([other],reference)

    def test_arbitrary_input_directories_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"original_inputs";path.mkdir()
            (path/"important.json").write_text("{}")
            with self.assertRaises(ValueError):
                prepare_output(path,"yes")
            self.assertTrue((path/"important.json").exists())


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.weights = np.full(6, 1 / 6)
        self.reference = np.sqrt(6) * np.eye(6)[:, :3] @ np.diag([3., 2., 1.])

    def test_rotated_leading_subspace_and_known_missing_direction(self):
        rotation = np.array([[.6, -.8], [.8, .6]])
        low = self.reference[:, :2] @ rotation
        stats, angles, _ = subspace_comparison(low, self.reference[:, :2], self.weights, 1e-12, 0.)
        np.testing.assert_allclose(angles, 0, atol=2e-6)
        self.assertAlmostEqual(stats["overlap"], 2)
        mixed = self.reference[:, [0, 2]]
        _, angles, _ = subspace_comparison(mixed, self.reference[:, :2], self.weights, 1e-12, 0.)
        np.testing.assert_allclose(angles, [0, 90], atol=2e-6)

    def test_rectangular_procrustes_and_no_centering(self):
        low = self.reference[:, :2]
        _, after = procrustes(low, self.reference, self.weights)
        self.assertAlmostEqual(after, 1.)  # Exactly the discarded eigenvalue's square root.
        shifted = low + np.ones_like(low)
        stats, _, _ = subspace_comparison(shifted, low, self.weights, 1e-12, 0.)
        self.assertLess(stats["overlap"], 2.)

    def test_weighted_orthonormality_and_rank(self):
        weights = np.array([.1, .2, .3, .1, .15, .15])
        z = np.column_stack([self.reference[:, 0], 2 * self.reference[:, 0], self.reference[:, 1]])
        q, _, _, keep = basis(z, weights, 1e-10, 0.)
        self.assertEqual(keep.sum(), 2)
        np.testing.assert_allclose(q.T @ q, np.eye(2), atol=1e-12)

    def test_degenerate_block_mixing_is_not_physical_mode_mixing(self):
        blocks = eigenblocks(np.array([1., .999, .1]), .01)
        self.assertEqual(blocks, [[0, 1], [2]])
        rows = mixing_statistics(np.array([[1 / np.sqrt(2)], [1 / np.sqrt(2)], [0.]]), blocks, 2)
        self.assertAlmostEqual(rows[0]["effective_modes"], 2.)
        self.assertAlmostEqual(rows[0]["effective_blocks"], 1.)


class ThermodynamicFactorTests(unittest.TestCase):
    def test_low_rank_tm_loss_and_spectrum_against_dense_operator(self):
        rng = np.random.default_rng(37)
        zr, zl = rng.normal(size=(7, 3)), rng.normal(size=(7, 2))
        gr, gl = rng.normal(size=(13, 3)), rng.normal(size=(13, 2))
        wc, wm = rng.dirichlet(np.ones(7)), rng.dirichlet(np.ones(13))
        result, overlap, retention, values = response_pair(zr, zl, gr, gl, wc, wm, 1e-12, 0.)
        mu_ref, mu_low = gr @ zr.T, gl @ zl.T
        dense_loss = np.sum(wm[:, None] * wc[None, :] * (mu_ref - mu_low)**2)
        self.assertAlmostEqual(result["tm_loss"], dense_loss, places=12)
        singular = np.linalg.svd(np.sqrt(wm)[:, None] * mu_ref * np.sqrt(wc)[None, :], compute_uv=False)
        np.testing.assert_allclose(values, singular[:3]**2, atol=1e-12)
        self.assertGreaterEqual(result["tm_loss"] + 1e-12, result["function_projection_residual"])
        self.assertGreaterEqual(result["function_projection_residual"] + 1e-12, result["eym_tail_bound"])
        self.assertTrue(np.all(retention <= 1))
        vals, coefficients, *_ = response_eigensystem(zr, gr, wc, wm, 1e-12, 0.)
        functions = gr @ coefficients
        np.testing.assert_allclose(gram(functions, functions, wm), np.eye(3), atol=1e-12)
        exact, _, _, _ = response_pair(zr, zr, gr, gr, wc, wm, 1e-12, 0.)
        self.assertLess(exact["tm_loss"], 1e-12)
        self.assertLess(exact["function_projection_residual"], 1e-12)

    def test_archived_mixture_construction(self):
        native = np.array([[1., 2.], [3., 4.], [5., 6.]])
        samples = {"ctot": np.array([.1, .2]), "offsets": np.array([0, 2, 3]),
            "indices": np.array([0, 2, 1]), "weights": np.array([.25, .75, 1.])}
        expected = np.array([.1 * (.25 * native[0] + .75 * native[2]), .2 * native[1]])
        np.testing.assert_allclose(mixture_latent(native, samples, 0, 2), expected)
        np.testing.assert_allclose(mixture_latent(native, samples, 1, 2), expected[1:])

    def test_unequal_canonical_batches_preserve_original_quadrature(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            np.savez(path/"samples.npz",ctot=np.array([.1,.2,.3]),offsets=np.arange(4),
                     indices=np.zeros(3,dtype=int),weights=np.ones(3))
            np.savez(path/"metric.npz",batch_sizes=np.array([2,1]))
            reference=SimpleNamespace(canonical=path,sequences=["AF"],
                config={"arguments":{"metric_aggregation":"batch-mean"}})
            np.testing.assert_array_equal(canonical_samples(reference)["quadrature_weights"],[.25,.25,.5])
            reference.config["arguments"]["metric_aggregation"]="sample-mean"
            np.testing.assert_allclose(canonical_samples(reference)["quadrature_weights"],np.full(3,1/3))


if __name__ == "__main__":
    unittest.main()
