"""Nonlinear fields, coordinate derivatives, ensemble spread, and plot masking."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"analysis"))
from calc_free_energy_slice_v1 import evaluate_fields
from _free_energy_slice_plots_v1 import plot_arrays, render_maps
from _pressure_zero_geometry_v1 import construct_slice


class NonlinearEnsemble:
    num_models, dimension = 5, 3

    def thermodynamics_from_latent(self, x):
        # Deliberately nonquadratic: cubic terms test the full gradient field.
        f, g = [], []
        for a in range(1, self.num_models+1):
            f.append(a*x[:,0]**2 + .4*x[:,1]**3 + .7*x[:,0]*x[:,2] + .2*x[:,2]**3 + .1*x[:,1])
            g.append(np.column_stack([2*a*x[:,0]+.7*x[:,2], 1.2*x[:,1]**2+.1,
                                      .7*x[:,0]+.6*x[:,2]**2]))
        return {"free_energy_excess": np.asarray(f), "gradient": np.asarray(g)}


class FieldTests(unittest.TestCase):
    def test_nonlinear_gradient_chain_rule_and_five_member_spread(self):
        z = np.array([[.5,-.2,.7], [1.2,.8,.7], [.2,.4,.7]])
        c = .13
        model = NonlinearEnsemble()
        result = evaluate_fields(model,z,c,2)
        np.testing.assert_allclose(result["grad_zbar"], c*result["grad_x"], rtol=0, atol=0)
        for axis in (0,1,2):
            h = 1e-5
            step = np.zeros(3); step[axis] = h
            plus = model.thermodynamics_from_latent(c*(z+step))["free_energy_excess"]
            minus = model.thermodynamics_from_latent(c*(z-step))["free_energy_excess"]
            np.testing.assert_allclose((plus-minus)/(2*h), result["grad_zbar"][:,:,axis], rtol=1e-8, atol=1e-11)
        # Only the a*x0^2 term differs across the five models.
        np.testing.assert_allclose(result["f_ex"].std(0), np.sqrt(2)*(c*z[:,0])**2)
        np.testing.assert_allclose(result["grad_x"].std(0)[:,0], 2*np.sqrt(2)*abs(c*z[:,0]))
        np.testing.assert_allclose(result["grad_x"].std(0)[:,1:], 0, atol=1e-15)

    def test_vacuum_empty_selection_and_single_member(self):
        model = NonlinearEnsemble()
        z = np.array([[.2,.4,.8]])
        zero = evaluate_fields(model,z,0.,5)
        np.testing.assert_array_equal(zero["f_ex"],np.zeros((5,1)))
        np.testing.assert_array_equal(zero["grad_zbar"],np.zeros((5,1,3)))
        empty = evaluate_fields(model,np.empty((0,3)),.1,5)
        self.assertEqual(empty["f_ex"].shape,(5,0))
        model.num_models = 1
        result = evaluate_fields(model,z,.1,5)
        np.testing.assert_array_equal(result["f_ex"].std(0),[0])

    def test_nonfinite_predictions_are_errors(self):
        model = NonlinearEnsemble()
        def bad(x):
            return {"free_energy_excess":np.full((5,len(x)),np.nan),
                    "gradient":np.zeros((5,len(x),3))}
        model.thermodynamics_from_latent = bad
        with self.assertRaises(FloatingPointError):
            evaluate_fields(model,np.ones((2,3)),.1,2)

    def test_reversed_axes_and_excluded_cells(self):
        # A 3D tetrahedron sliced with axes in the order (z3,z1).
        cloud = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
        geometry = construct_slice(cloud,(2,0),[],.25,0,100,10)
        retained = geometry["inside_projected"] & (geometry["coordinates"][:,0]+geometry["coordinates"][:,2] <= .75)
        n=len(retained)
        fields={"f_ex_mean":np.arange(n,dtype=float)-2,"f_ex_std":np.ones(n)*.2,
                "grad_zbar_3_mean":np.arange(n,dtype=float)+1,"grad_zbar_1_mean":np.ones(n)*2}
        e,s,u,v,arrows=plot_arrays(geometry,retained,fields,(2,0),2)
        self.assertTrue(np.isnan(e[~retained]).all())
        self.assertTrue(np.isnan(s[~retained]).all())
        self.assertFalse(np.any(arrows & ~retained))
        np.testing.assert_array_equal(u,fields["grad_zbar_3_mean"])
        from matplotlib.axes import Axes
        real_quiver,real_mesh=Axes.quiver,Axes.pcolormesh
        allowed={tuple(p) for p in geometry["projected"][retained]}
        seen=[]
        def quiver(ax,x,y,uu,vv,*args,**kw):
            self.assertTrue(set(zip(x,y)) <= allowed)
            np.testing.assert_array_equal(uu,u[arrows]);np.testing.assert_array_equal(vv,v[arrows])
            seen.append(True)
            return real_quiver(ax,x,y,uu,vv,*args,**kw)
        def mesh(ax,*args,**kw):
            if len(args)>=3 and np.shape(args[2])==tuple(reversed(geometry["shape"])):
                self.assertFalse(np.any(~np.ma.getmaskarray(args[2]) & ~retained.reshape(geometry["shape"]).T))
            return real_mesh(ax,*args,**kw)
        with tempfile.TemporaryDirectory() as tmp,patch.object(Axes,"quiver",quiver),patch.object(Axes,"pcolormesh",mesh):
            result=render_maps(tmp,geometry,(2,0),retained,fields,
                {"voxel_size":.25,"fixed_coordinates":{"2":.25},"dimension":3,"c_tot":.1},2,None)
            self.assertEqual(len(result["files"]),3)
            self.assertGreater(result["resolved_arrow_scale"],0)
            self.assertTrue(all((Path(tmp)/name).stat().st_size>1000 for name in result["files"]))
        self.assertTrue(seen)


if __name__=="__main__":
    unittest.main()
