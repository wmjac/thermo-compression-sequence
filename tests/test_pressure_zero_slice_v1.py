"""Numerical, geometry, matching, provenance, and plot-domain checks."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from _pressure_zero_solver_v1 import SolverSettings, SolverFailure, solve_ray, match_member_roots
from _pressure_zero_geometry_v1 import construct_slice, check_full_hull, lp_indices
from _pressure_zero_io_v1 import render_slice_maps


class RootTests(unittest.TestCase):
    def test_five_models_known_roots_slopes_and_ensemble_root(self):
        a = .04 + .0005 * np.arange(5)
        b = .18 + .001 * np.arange(5)
        def pressure(c):
            return c[None] * (1 - c[None] / a[:, None]) * (1 - c[None] / b[:, None])
        def slope(c):
            return 1 - 2*c*(1/a + 1/b) + 3*c*c/(a*b)
        result = solve_ray(pressure, slope, SolverSettings())
        expected = np.roots([np.mean(1/(a*b)), -np.mean(1/a + 1/b), 1]).max()
        self.assertAlmostEqual(result["roots"][-1][0]["c_tot"], expected, places=10)
        for m in range(5):
            self.assertEqual(len(result["roots"][m]), 1)
            self.assertAlmostEqual(result["roots"][m][0]["c_tot"], b[m], places=10)
        self.assertTrue(any(r["status"] == "rejected_nonpositive_slope" for r in result["candidates"]))
        self.assertTrue(result["scan_converged"])
        summaries, _ = match_member_roots(result["roots"][-1], [result["roots"][m] for m in range(5)], 1e-4, .25)
        self.assertEqual(summaries[0]["n_models_matched"], 5)
        self.assertAlmostEqual(summaries[0]["member_root_mean"], .182, places=10)
        self.assertAlmostEqual(summaries[0]["member_root_std"], .001*np.sqrt(2), places=10)
        self.assertNotEqual(summaries[0]["member_root_mean"], summaries[0]["ensemble_c_tot"])

    def test_vacuum_and_no_root(self):
        result = solve_ray(lambda c: np.repeat(c[None], 5, axis=0),
                           lambda c: np.ones(5), SolverSettings())
        self.assertTrue(all(not rows for rows in result["roots"].values()))

    def test_upper_boundary_root_and_positive_slope(self):
        a, b = .07, .25
        result = solve_ray(lambda c: (c*(1-c/a)*(1-c/b))[None],
                           lambda c: np.array([1-2*c*(1/a+1/b)+3*c*c/(a*b)]), SolverSettings())
        self.assertEqual(result["roots"][-1][0]["c_tot"], b)
        self.assertGreater(result["roots"][-1][0]["slope"], 0)

    def test_refinement_detects_root_set_changes(self):
        roots = np.array([.03, .06, .13, .19])
        poly = np.polynomial.Polynomial([0., 1.])
        for r in roots:
            poly *= np.polynomial.Polynomial([1., -1/r])
        p = lambda c: poly(c)[None]
        slope = lambda c: np.array([poly.deriv()(c)])
        with self.assertRaises(SolverFailure):
            solve_ray(p, slope, replace(SolverSettings(), n_brackets=2, max_refinements=1))
        result = solve_ray(p, slope, replace(SolverSettings(), n_brackets=2, max_refinements=4))
        np.testing.assert_allclose([r["c_tot"] for r in result["roots"][-1]], [.06, .19], atol=1e-10)
        self.assertGreater(result["refinements"], 1)

    def test_discontinuity_is_not_reported_as_a_root(self):
        with self.assertRaisesRegex(SolverFailure, "residual"):
            solve_ray(lambda c: (c*np.where(c < .151234567, -1., 1.))[None],
                      lambda c: np.ones(1), SolverSettings())

    def test_nonfinite_and_invalid_tolerances(self):
        with self.assertRaises(SolverFailure):
            solve_ray(lambda c: np.full((1,len(c)),np.nan), lambda c: np.ones(1), SolverSettings())
        with self.assertRaises(ValueError):
            replace(SolverSettings(), root_rtol=1e-20).validate()

    def test_ambiguous_matching_does_not_fabricate_sd(self):
        ensemble = [{"c_tot": .10}, {"c_tot": .20}]
        members = [[{"c_tot": .15}], [{"c_tot": .1001}, {"c_tot": .1002}]]
        summary, diagnostics = match_member_roots(ensemble, members, .1, 0.)
        self.assertTrue(all(row["member_root_std"] is None for row in summary))
        self.assertTrue(any(row["status"] == "ambiguous_equal_distance" for row in diagnostics))
        self.assertTrue(any(row["status"] == "ambiguous_multiple_roots" for row in diagnostics))


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])

    def test_fixed_plane_full_hull_exclusion_and_origin_grid(self):
        g = construct_slice(self.points, (0,1), [], .25, 0., 100, 10)
        np.testing.assert_array_equal(g["coordinates"][:,2], np.full(25,.25))
        projected = g["inside_projected"]
        statuses = check_full_hull(self.points, g["coordinates"][projected], 0., 1e-9, 10., 1)
        queries = g["coordinates"][projected]
        self.assertTrue(np.array_equal(statuses=="inside", queries[:,0]+queries[:,1] <= .75))
        np.testing.assert_allclose(g["projected"]/.25, np.round(g["projected"]/.25))

    def test_offset_extrema_and_degenerate_projection(self):
        g = construct_slice(self.points, (0,1), [(2,.125)], .5, 0., 100, 10)
        np.testing.assert_array_equal(g["coordinates"][:,2], np.full(9,.375))
        with self.assertRaises(ValueError):
            lp_indices(self.points, 1, 0)
        repeated = np.r_[self.points,self.points]
        self.assertEqual(len(lp_indices(repeated,None,0)),4)
        with self.assertRaisesRegex(ValueError,"degenerate"):
            construct_slice(np.zeros((4,3)),(0,1),[],.1,0.,100,10)

    def test_plot_dots_and_heatmap_cells_never_include_excluded_centers(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.axes import Axes
        g = construct_slice(self.points, (0,1), [], .25, 0., 100, 10)
        retained = g["inside_projected"] & (g["coordinates"][:,0]+g["coordinates"][:,1] <= .75)
        ids = np.flatnonzero(retained)
        roots = {int(i): ([{"c_tot":.12}] if j%2 else []) for j,i in enumerate(ids)}
        summary = [{"point_id":i,"ensemble_c_tot":.12,"member_root_std":.01}
                   for i, rows in roots.items() if rows]
        allowed = {tuple(g["projected"][i]) for i in ids if not roots[int(i)]}
        calls = []
        scatter = Axes.scatter
        mesh = Axes.pcolormesh
        def checked_scatter(ax,x,y,*a,**kw):
            calls.append("dots")
            self.assertTrue(set(zip(x,y)) <= allowed)
            return scatter(ax,x,y,*a,**kw)
        def checked_mesh(ax,*a,**kw):
            if len(a)>=3 and np.shape(a[2]) == tuple(reversed(g["shape"])):
                mask = np.ma.getmaskarray(a[2])
                self.assertFalse(np.any((~mask) & ~retained.reshape(g["shape"]).T))
                calls.append("mesh")
            return mesh(ax,*a,**kw)
        with tempfile.TemporaryDirectory() as directory, patch.object(Axes,"scatter",checked_scatter), patch.object(Axes,"pcolormesh",checked_mesh):
            files = render_slice_maps(directory,g,(0,1),retained,roots,summary,
                {"voxel_size":.25,"fixed_coordinates":{"3":.25},"dimension":3,"c_tot_max":.25})
            self.assertIn("c_tot_largest_root.png",files)
            self.assertTrue(all((Path(directory)/name).stat().st_size>1000 for name in files))
        self.assertIn("dots",calls)
        self.assertIn("mesh",calls)


if __name__ == "__main__":
    unittest.main()
