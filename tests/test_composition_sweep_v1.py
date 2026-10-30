"""Scientific checks independent of saved production checkpoints."""
from pathlib import Path
import importlib.util
import unittest
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "analysis/composition_sweep_v1.py"
SPEC = importlib.util.spec_from_file_location("composition_sweep_v1", SCRIPT)
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


class QuadraticEnsemble:
    """Known EOS beta f_ex = a*x^2/2, with different member encoders."""
    def __init__(self):
        import torch
        self.models = [torch.nn.Linear(1, 1, dtype=torch.float64) for _ in range(2)]

    def encode_sequences(self, sequences, batch_size):
        return np.array([[[1.0], [3.0]], [[2.0], [4.0]]])

    def thermodynamics_from_latent(self, x):
        f = 0.5 * np.array([2.0, 5.0])[:, None] * x[:, :, 0] ** 2
        return {"free_energy_excess": f, "pressure_excess": f.copy()}


class TestCompositionSweep(unittest.TestCase):
    def test_composition_validation(self):
        _, x = sweep.parse_components([["A", "0.25"], ["F", "0.75"]])
        np.testing.assert_array_equal(x, [0.25, 0.75])
        for invalid in ([["A", "0.9"]], [["A", "nan"]], [["A", "-1"], ["F", "2"]],
                        [["A", ".5"], ["A", ".5"]]):
            with self.assertRaises(ValueError):
                sweep.parse_components(invalid)

    def test_grid_and_zero_limit(self):
        c = sweep.concentration_grid(0, .25, 3, "linear")
        np.testing.assert_array_equal(c, [0, .125, .25])
        self.assertEqual(sweep.ideal_free_energy(c, np.array([1.0]))[0], 0)
        np.testing.assert_allclose(sweep.concentration_grid(1e-6, 1e-2, 3, "log"), [1e-6, 1e-4, 1e-2])
        with self.assertRaises(ValueError):
            sweep.concentration_grid(0, .25, 3, "log")

    def test_ideal_mixing_and_zero_species(self):
        c = np.array([0, .01, .1])
        pure = sweep.ideal_free_energy(c, np.array([1.0]))
        mixed = sweep.ideal_free_energy(c, np.array([.5, .5]))
        np.testing.assert_allclose(mixed - pure, -c * np.log(2), atol=1e-16)
        np.testing.assert_array_equal(pure, sweep.ideal_free_energy(c, np.array([1.0, 0])))

    def test_model_pairing_and_thermodynamic_identity(self):
        c = np.array([0, 1e-5, .01, .1])
        x = np.array([.25, .75])
        result, f_id, info = sweep.evaluate_sweep(QuadraticEnsemble(), ["A", "F"], x, c, 2, "float64")
        # Model-specific rays are 2.5 and 3.5: averaging these before decoding
        # would give an incorrect prediction for both members.
        expected = .5 * np.array([2, 5])[:, None] * (np.array([2.5, 3.5])[:, None] * c) ** 2
        np.testing.assert_allclose(result["f_ex"], expected)
        np.testing.assert_allclose(result["P_ex"], expected)
        np.testing.assert_allclose(result["f"], expected + f_id)
        np.testing.assert_allclose(result["P"], expected + c)
        np.testing.assert_allclose(result["P"].std(axis=0), expected.std(axis=0))
        np.testing.assert_array_equal(info["mixture_features_by_model"], [[2.5], [3.5]])
        # Independent numerical derivative along a composition ray:
        # P = c*d f/d c - f, including the ideal mixing term.
        point, step = .06, 1e-6
        cc = np.array([point - step, point, point + step])
        v, _, _ = sweep.evaluate_sweep(QuadraticEnsemble(), ["A", "F"], x, cc, 3, "float64")
        pressure_fd = point * (v["f"][:, 2] - v["f"][:, 0]) / (2 * step) - v["f"][:, 1]
        np.testing.assert_allclose(pressure_fd, v["P"][:, 1], rtol=1e-8, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
