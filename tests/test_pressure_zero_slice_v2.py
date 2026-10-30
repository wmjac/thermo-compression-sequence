"""Known pressure branches, censoring semantics, and plot-domain checks."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from _pressure_zero_solver_v1 import SolverSettings, solve_ray as solve_ray_v1
from _pressure_zero_reporting_v2 import solve_ray, summarize_curve, reporting_arrays, render_reported_map
from _pressure_zero_geometry_v1 import construct_slice


class ReportingTests(unittest.TestCase):
    def test_negative_endpoint_overrides_small_ascending_root(self):
        # Two negative-pressure intervals; the larger ascending root is .4,
        # beyond the .25 scan. The spurious smaller ascending root is .06.
        a, b, d, e = .02, .06, .12, .4
        polynomial = np.polynomial.Polynomial.fromroots([0, a, b, d, e]) / (a*b*d*e)
        settings = SolverSettings()
        pressure = lambda c: polynomial(c)[None, :]
        slope = lambda c: np.array([polynomial.deriv()(c)])
        result = solve_ray(pressure, slope, settings)
        roots = result["roots"][-1]
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0]["c_tot"], b, places=11)
        row = summarize_curve(roots, result["member_beta_pressure_at_c_tot_max"].mean(), settings)
        self.assertEqual(row["reported_c_tot"], .25)
        self.assertEqual(row["upper_bound_censored"], 1)
        self.assertEqual(row["n_solutions"], 1)
        self.assertAlmostEqual(row["largest_root_in_range"], .06, places=11)
        legacy = summarize_curve(roots, polynomial(.25), settings, "roots-only")
        self.assertAlmostEqual(legacy["reported_c_tot"], .06, places=11)
        self.assertEqual(legacy["upper_bound_censored"], 0)

    def test_negative_endpoint_without_any_ascending_root(self):
        settings = SolverSettings()
        result = solve_ray(lambda c: (c*(1-c/.1))[None], lambda c: np.array([1-2*c/.1]), settings)
        self.assertEqual(result["roots"][-1], [])
        row = summarize_curve([], result["member_beta_pressure_at_c_tot_max"].mean(), settings)
        self.assertEqual(row["reported_status"], "capped_at_c_tot_max")
        self.assertEqual(row["n_solutions"], 0)
        self.assertIsNone(row["largest_root_in_range"])

    def test_positive_endpoint_preserves_actual_root_or_no_solution(self):
        settings = SolverSettings()
        root = {"c_tot": .18}
        self.assertEqual(summarize_curve([root], .01, settings)["reported_c_tot"], .18)
        row = summarize_curve([], .25, settings)
        self.assertIsNone(row["reported_c_tot"])
        self.assertEqual(row["reported_status"], "no_positive_slope_root")
        # An actual boundary root stays a root, without censoring.
        row = summarize_curve([{"c_tot": .25}], 0, settings)
        self.assertEqual(row["reported_status"], "root_in_range")
        self.assertEqual(row["upper_bound_censored"], 0)

    def test_endpoint_tolerance_and_configured_maximum(self):
        settings = replace(SolverSettings(), c_max=.5)
        tolerance = settings.pressure_atol + settings.pressure_rtol * settings.c_max
        self.assertEqual(summarize_curve([], -tolerance/2, settings)["upper_bound_censored"], 0)
        self.assertEqual(summarize_curve([], -2*tolerance, settings)["reported_c_tot"], .5)

    def test_capture_keeps_original_roots_and_call_count(self):
        settings = SolverSettings()
        calls = []
        def pressure(c):
            calls.append(c.copy())
            return np.array([c*(1-c/.04)*(1-c/.18), c])
        slope = lambda c: np.array([1-2*c*(1/.04+1/.18)+3*c*c/(.04*.18), 1])
        legacy = solve_ray_v1(pressure, slope, settings)
        n = len(calls)
        calls.clear()
        result = solve_ray(pressure, slope, settings)
        self.assertEqual(len(calls), n)
        self.assertEqual(result["roots"], legacy["roots"])
        self.assertEqual(result["candidates"], legacy["candidates"])
        np.testing.assert_allclose(result["member_beta_pressure_at_c_tot_max"],
                                   [.25*(1-.25/.04)*(1-.25/.18), .25])

    def test_capped_points_heatmap_and_markers_respect_hull(self):
        points = np.array([[0,0],[1,0],[0,1]])
        geometry = construct_slice(points, (0,1), [], .25, 0, 100, 10)
        retained = geometry["inside_projected"]
        ids = np.flatnonzero(retained)
        reports = {int(i): summarize_curve([], -.1 if j%2 else .1, SolverSettings())
                   for j,i in enumerate(ids)}
        values, capped, no_value = reporting_arrays(retained, reports)
        self.assertTrue(np.isnan(values[~retained]).all())
        self.assertFalse(np.any(capped & no_value))
        self.assertTrue(np.all(values[capped] == .25))
        self.assertTrue(np.all(no_value | capped | ~retained))
        from matplotlib.axes import Axes
        real_scatter = Axes.scatter
        expected = {tuple(p) for p in geometry["projected"][retained]}
        seen = []
        def scatter(ax,x,y,*args,**kw):
            self.assertTrue(set(zip(x,y)) <= expected)
            seen.append(kw.get("marker", "o"))
            return real_scatter(ax,x,y,*args,**kw)
        with tempfile.TemporaryDirectory() as tmp, patch.object(Axes, "scatter", scatter):
            name = render_reported_map(tmp, geometry, (0,1), retained, reports,
                {"voxel_size":.25, "c_tot_max":.25, "dimension":2, "fixed_coordinates":{}})
            self.assertGreater((Path(tmp)/name).stat().st_size, 1000)
        self.assertIn("^", seen)
        self.assertIn("o", seen)


if __name__ == "__main__":
    unittest.main()
