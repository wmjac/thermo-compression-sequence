"""Meaningful numerical checks for population model spreads."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from _ensemble_statistics_v2 import feature_statistics, plot_test_losses, test_loss_statistics as loss_statistics


class EnsembleStatisticsTests(unittest.TestCase):
    def test_coordinate_axis_and_rmsd_relation_for_five_models(self):
        features = np.array([[[-2., 4., 8.]], [[-1., 4., 8.]], [[0., 4., 8.]],
                             [[1., 4., 8.]], [[2., 4., 8.]]])
        mean, rmsd, std = feature_statistics(features)
        np.testing.assert_array_equal(mean, [[0., 4., 8.]])
        np.testing.assert_allclose(std, [[np.sqrt(2), 0., 0.]], rtol=0, atol=1e-15)
        np.testing.assert_allclose(rmsd, [np.sqrt(2)], rtol=0, atol=1e-15)
        np.testing.assert_allclose(rmsd**2, (std**2).sum(axis=1))
        repeated = np.repeat(np.array([[[1., 4.], [100., 400.]]]), 5, axis=0)
        np.testing.assert_array_equal(feature_statistics(repeated)[2], np.zeros((2, 2)))

    def test_known_test_losses_and_distinct_ensemble_estimators(self):
        target, ctot = np.array([1., 3.]), np.array([1., 2.])
        predictions = np.array([[1., 3.], [2., 1.], [0., 5.], [1., 5.], [0., 3.]])
        row, losses, rmse = loss_statistics(target, predictions, ctot)
        np.testing.assert_array_equal(losses, [0., 1., 1., .5, .5])
        self.assertAlmostEqual(row["member_pressure_loss_mean"], .6)
        self.assertAlmostEqual(row["member_pressure_loss_std"], np.sqrt(.14))
        self.assertAlmostEqual(row["ensemble_pressure_loss"], .04)
        self.assertAlmostEqual(row["mean_normalized_prediction_variance"], .56)
        self.assertAlmostEqual(row["member_normalized_rmse_mean"], (2 + np.sqrt(2)) / 5)
        self.assertNotAlmostEqual(row["member_normalized_rmse_mean"], np.sqrt(.6))
        np.testing.assert_array_equal(rmse, np.sqrt([0., 1., 1., .5, .5]))

    def test_single_member_and_equal_members(self):
        row, _, _ = loss_statistics([1.], [[2.]], [.5])
        self.assertEqual(row["member_pressure_loss_std"], 0.)
        self.assertEqual(row["member_normalized_rmse_std"], 0.)
        self.assertEqual(row["ensemble_pressure_loss"], 4.)
        _, rmsd, std = feature_statistics(np.ones((1, 3, 2)))
        np.testing.assert_array_equal(rmsd, np.zeros(3))
        np.testing.assert_array_equal(std, np.zeros((3, 2)))

    def test_invalid_shapes_and_nonfinite_values_rejected(self):
        for target, pressure, ctot in [
            ([], np.zeros((5, 0)), []), ([1.], [[2.]], [0.]),
            ([1.], [[np.nan]], [1.]), ([1.], [[2., 3.]], [1.]),
        ]:
            with self.assertRaises(ValueError):
                loss_statistics(target, pressure, ctot)
        with self.assertRaises(ValueError):
            feature_statistics(np.array([[[np.inf]]]))

    def test_log_plots_handle_zero_losses_and_nonpositive_sd_endpoints(self):
        row, _, _ = loss_statistics([0.], [[0.], [0.], [0.], [0.], [1.]], [1.])
        self.assertGreater(row["member_pressure_loss_std"], row["member_pressure_loss_mean"])
        zero, _, _ = loss_statistics([0.], np.zeros((5, 1)), [1.])
        with tempfile.TemporaryDirectory() as directory:
            for rows in ([dict(dimension=1, **row)], [dict(dimension=1, **zero)]):
                plot_test_losses(rows, directory)
            for filename in ("loss_by_dimension_test.png", "loss_by_dimension_test_log.png",
                             "rmse_by_dimension_test_log.png"):
                self.assertGreater((Path(directory) / filename).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
