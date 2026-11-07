"""Masked free-energy heatmaps and globally scaled positive gradient arrows."""
from pathlib import Path
import numpy as np


def plot_arrays(geometry, retained, fields, axes, stride):
    retained = np.asarray(retained, dtype=bool)
    energy = np.asarray(fields["f_ex_mean"], dtype=np.float64).copy()
    energy[~retained] = np.nan
    std = np.asarray(fields["f_ex_std"], dtype=np.float64).copy()
    std[~retained] = np.nan
    u = np.asarray(fields[f"grad_zbar_{axes[0]+1}_mean"], dtype=np.float64)
    v = np.asarray(fields[f"grad_zbar_{axes[1]+1}_mean"], dtype=np.float64)
    selected = np.zeros(geometry["shape"], dtype=bool)
    selected[::stride, ::stride] = True
    arrows = selected.ravel() & retained & np.isfinite(u) & np.isfinite(v) & ((u != 0) | (v != 0))
    return energy, std, u, v, arrows


def render_maps(directory, geometry, axes, retained, fields, metadata, stride, scale):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, Normalize
    directory = Path(directory)
    energy, std, u, v, arrows = plot_arrays(geometry, retained, fields, axes, stride)
    points = geometry["projected"]
    spacing = metadata["voxel_size"]
    xs, ys = np.unique(points[:,0]), np.unique(points[:,1])
    x_edges = np.r_[xs-spacing/2, xs[-1]+spacing/2]
    y_edges = np.r_[ys-spacing/2, ys[-1]+spacing/2]
    fixed = ", ".join(f"z_{j}={x:.6g}" for j,x in metadata["fixed_coordinates"].items()) or "none (d=2)"
    details = {"arrow_stride": stride, "requested_arrow_scale": "auto" if scale is None else scale,
               "arrow_direction": "positive grad_zbar of ensemble-mean f_ex; no per-arrow normalization",
               "arrow_count": int(arrows.sum()), "resolved_arrow_scale": None, "files": []}

    def draw(values, filename, title, colorbar_label, gradient=False, uncertainty=False):
        finite = values[np.isfinite(values)]
        if len(finite):
            low, high = float(finite.min()), float(finite.max())
        else:
            low, high = 0., 1.
        if low == high:
            pad = max(abs(low)*.05, 1e-12)
            low, high = (0., high+pad) if uncertainty or low == 0 else (low-pad, high+pad)
        cmap = plt.get_cmap("coolwarm" if low < 0 < high and not uncertainty else "viridis").copy()
        cmap.set_bad("white")
        norm = TwoSlopeNorm(vmin=low, vcenter=0, vmax=high) if low < 0 < high and not uncertainty else Normalize(low,high)
        fig, ax = plt.subplots(figsize=(8,6.8))
        mesh = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(values.reshape(geometry["shape"]).T),
                            shading="flat", cmap=cmap, norm=norm, rasterized=True)
        if len(finite):
            fig.colorbar(mesh, ax=ax, label=colorbar_label)
        else:
            fig.text(.5,.115,"No evaluated grid points",ha="center",fontsize=9)
        quiver = None
        if gradient and arrows.any():
            q = ax.quiver(points[arrows,0], points[arrows,1], u[arrows], v[arrows],
                          angles="xy", scale_units="xy", scale=scale, color="black", width=.0025)
            quiver = q
        elif gradient and len(finite):
            message = "Mean in-plane gradient is zero" if np.all(np.nan_to_num(u[retained])==0) and np.all(np.nan_to_num(v[retained])==0) else "No nonzero arrows selected; reduce --arrow-stride"
            fig.text(.5,.115,message,ha="center",fontsize=8)
        ax.set(xlabel=rf"$\overline{{z}}_{{{axes[0]+1}}}$", ylabel=rf"$\overline{{z}}_{{{axes[1]+1}}}$",
               title=f"d={metadata['dimension']}: {title}\n" + fr"$c_{{\rm tot}}={metadata['c_tot']:.6g}$ nm$^{{-3}}$")
        ax.set_aspect("equal", adjustable="box")
        caption = f"Fixed coordinates: {fixed}. Excluded grid points are blank."
        if gradient:
            caption += "\nArrows: positive gradient with respect to the two displayed coordinates at fixed c_tot."
            caption += "\nA single global arrow scale preserves relative magnitudes; exact values are in lattice.tsv."
        elif uncertainty:
            caption += "\nPopulation SD across selected models (ddof=0), not SEM."
        fig.text(.5,.025,caption,ha="center",va="bottom",fontsize=8)
        fig.tight_layout(rect=(0,.15,1,1))
        if quiver is not None:
            fig.canvas.draw()
            details["resolved_arrow_scale"] = float(quiver.scale)
            reference = float(np.percentile(np.hypot(u[arrows],v[arrows]),75))
            ax.quiverkey(quiver, X=.35, Y=.115, U=reference,
                         label=rf"$|\nabla_{{\bar z,2D}}f^{{\rm ex}}|={reference:.4g}$", labelpos="E", coordinates="figure")
        fig.savefig(directory/filename,dpi=180)
        plt.close(fig)
        details["files"].append(filename)

    label=r"$f^{\rm ex}$ ($kT$ nm$^{-3}$)"
    draw(energy,"free_energy_excess.png","excess free-energy density",label)
    draw(energy,"free_energy_excess_gradient.png","excess free energy and gradient",label,gradient=True)
    draw(std,"free_energy_excess_std.png","model spread in excess free energy",
         r"Model SD of $f^{\rm ex}$ ($kT$ nm$^{-3}$)",uncertainty=True)
    return details
