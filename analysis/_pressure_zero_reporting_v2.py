"""Explicit upper-bound censoring of reported P=0 concentrations.

The validated v1 root-finding procedure is reused without changing its grids,
tolerances, slope tests, or raw roots. Upper-bound pressure is captured from
that existing scan, avoiding an extra model evaluation.
"""
from pathlib import Path
import numpy as np
from _pressure_zero_solver_v1 import SolverFailure, solve_ray as solve_ray_v1


def solve_ray(pressure, slope, settings, trace=None):
    upper_pressure = None

    def capture(c):
        nonlocal upper_pressure
        c = np.atleast_1d(np.asarray(c, dtype=np.float64))
        result = np.asarray(pressure(c), dtype=np.float64)
        at_upper = np.flatnonzero(c == settings.c_max)
        if len(at_upper) and result.ndim == 2 and result.shape[1] == len(c):
            upper_pressure = result[:, at_upper[0]].copy()
        return result

    result = solve_ray_v1(capture, slope, settings, trace)
    if upper_pressure is None or not np.isfinite(upper_pressure).all():
        raise SolverFailure("The upper concentration boundary was not evaluated successfully.")
    result["member_beta_pressure_at_c_tot_max"] = upper_pressure
    return result


def summarize_curve(roots, upper_pressure, settings, policy="cap"):
    """Return a reporting value, without inserting a fictitious root.

Negative pressure at the upper boundary means that the last negative-pressure
interval is still open. Under policy='cap', report c_max even if smaller
ascending roots were found. This does not prove a finite root exists beyond
the scanned range, nor establish the physical origin of extra crossings.
"""
    if policy not in ("cap", "roots-only"):
        raise ValueError("Upper-bound policy must be cap or roots-only.")
    upper_pressure = float(upper_pressure)
    if not np.isfinite(upper_pressure):
        raise SolverFailure("Nonfinite upper-bound pressure cannot be classified as a cap.")
    tolerance = settings.pressure_atol + settings.pressure_rtol * settings.c_max
    upper_negative = upper_pressure < -tolerance
    largest = max((float(row["c_tot"]) for row in roots), default=None)
    capped = policy == "cap" and upper_negative
    value = settings.c_max if capped else largest
    status = ("capped_at_c_tot_max" if capped else "root_in_range" if largest is not None
              else "no_positive_slope_root")
    return {
        "reported_c_tot": value, "reported_status": status,
        "upper_bound_censored": int(capped), "upper_bound_pressure_negative": int(upper_negative),
        "beta_pressure_at_c_tot_max": upper_pressure,
        "upper_bound_beta_pressure_tolerance": tolerance,
        "largest_root_in_range": largest, "n_solutions": len(roots),
    }


def reporting_arrays(retained, reports):
    """Keep excluded points blank and capped points distinct from no-value dots."""
    retained = np.asarray(retained, dtype=bool)
    values = np.full(len(retained), np.nan)
    capped = np.zeros(len(retained), dtype=bool)
    for point_id, row in reports.items():
        if retained[point_id] and row["reported_c_tot"] is not None:
            values[point_id] = float(row["reported_c_tot"])
            capped[point_id] = bool(row["upper_bound_censored"])
    no_value = retained & ~np.isfinite(values)
    return values, capped, no_value


def render_reported_map(directory, geometry, axes, retained, reports, metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    values, capped, no_value = reporting_arrays(retained, reports)
    points = geometry["projected"]
    spacing = metadata["voxel_size"]
    xs, ys = np.unique(points[:, 0]), np.unique(points[:, 1])
    x_edges = np.r_[xs - spacing / 2, xs[-1] + spacing / 2]
    y_edges = np.r_[ys - spacing / 2, ys[-1] + spacing / 2]
    image = np.ma.masked_invalid(values.reshape(geometry["shape"]).T)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(8, 6.8))
    mesh = ax.pcolormesh(x_edges, y_edges, image, shading="flat", cmap=cmap,
                         vmin=0, vmax=metadata["c_tot_max"], rasterized=True)
    fig.colorbar(mesh, ax=ax, label=r"Reported $c_{\mathrm{tot}}$ (nm$^{-3}$)",
                 extend="max" if capped.any() else "neither")
    if no_value.any():
        ax.scatter(points[no_value, 0], points[no_value, 1], s=9, color="black", zorder=4)
    if capped.any():
        ax.scatter(points[capped, 0], points[capped, 1], marker="^", s=17,
                   facecolors="none", edgecolors="black", linewidths=.55, zorder=5)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=4, color="black",
                      label="No root and no cap"),
               Line2D([], [], marker="^", linestyle="", markersize=5, color="black",
                      markerfacecolor="none", label=r"Capped at $c_{\mathrm{tot,max}}$")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .09), ncol=2, fontsize=8)
    fixed = ", ".join(f"z_{j}={v:.6g}" for j, v in metadata["fixed_coordinates"].items()) or "none (d=2)"
    caption = (f"Fixed coordinates: {fixed}. Hull-excluded cells are blank.\n"
               "Triangles: upper-bound pressure is negative beyond tolerance; reported value is capped.\n"
               "A cap is not a solved root. See lattice.tsv and curve_summary.tsv for status and raw roots.")
    fig.text(.5, .02, caption, ha="center", va="bottom", fontsize=8)
    ax.set(xlabel=rf"$\overline{{z}}_{{{axes[0]+1}}}$",
           ylabel=rf"$\overline{{z}}_{{{axes[1]+1}}}$",
           title=f"d={metadata['dimension']}: 2D slice, reported concentration with upper-bound caps")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout(rect=(0, .15, 1, 1))
    filename = "c_tot_reported.png"
    fig.savefig(Path(directory) / filename, dpi=180)
    plt.close(fig)
    return filename
