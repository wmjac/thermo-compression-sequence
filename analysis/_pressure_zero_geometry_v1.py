"""Explicit 2D affine slices; root solving is implemented independently."""
import warnings
import numpy as np
from scipy.optimize import OptimizeWarning, linprog
from scipy.spatial import ConvexHull, QhullError


def lp_indices(points, limit, seed):
    n, d = points.shape
    if limit is None or limit >= n:
        selected = np.arange(n)
    else:
        extrema = np.unique(np.r_[np.argmin(points, axis=0), np.argmax(points, axis=0)])
        if limit < len(extrema):
            raise ValueError(f"LP point limit {limit} cannot retain all {len(extrema)} bounding-box extrema.")
        rest = np.setdiff1d(np.arange(n), extrema, assume_unique=True)
        chosen = np.random.default_rng(seed).choice(rest, limit - len(extrema), replace=False)
        selected = np.sort(np.r_[extrema, chosen])
    # Exact duplicate columns can be removed without changing the chosen hull.
    _, unique = np.unique(points[selected], axis=0, return_index=True)
    return np.sort(selected[unique])


def construct_slice(points, axes, offsets, spacing, epsilon, max_grid_points, chunk_size):
    points = np.asarray(points, dtype=np.float64)
    n, d = points.shape
    if n < 3 or d < 2 or len(axes) != 2 or len(set(axes)) != 2 or not all(0 <= a < d for a in axes):
        raise ValueError("A 2D slice requires d>=2, distinct valid axes, and at least three sequence points.")
    if not np.isfinite(points).all() or spacing <= 0 or epsilon < 0:
        raise ValueError("Invalid slice points, spacing, or hull tolerance.")
    fill = points.mean(axis=0)
    seen = set()
    for axis, shift in offsets:
        if axis in axes or not 0 <= axis < d or axis in seen or not np.isfinite(shift):
            raise ValueError("Offsets must select distinct fixed dimensions only.")
        fill[axis] += shift
        seen.add(axis)
    projected = points[:, axes]
    try:
        hull = ConvexHull(projected)
    except QhullError as exc:
        raise ValueError("Selected projection is degenerate; choose other axes. No artificial jitter was added.") from exc
    low = np.floor(projected.min(axis=0) / spacing).astype(np.int64)
    high = np.ceil(projected.max(axis=0) / spacing).astype(np.int64)
    counts = high - low + 1
    total = int(counts[0]) * int(counts[1])
    if total > max_grid_points:
        raise ValueError(f"Slice requests {total} grid points, above --max-grid-points={max_grid_points}.")
    grid_k, grid_l = np.meshgrid(np.arange(low[0], high[0] + 1),
                                np.arange(low[1], high[1] + 1), indexing="ij")
    coordinates = np.column_stack([grid_k.ravel(), grid_l.ravel()]) * spacing
    full = np.broadcast_to(fill, (total, d)).copy()
    full[:, axes] = coordinates
    inside = np.empty(total, dtype=bool)
    for start in range(0, total, chunk_size):
        inequalities = coordinates[start:start + chunk_size] @ hull.equations[:, :2].T + hull.equations[:, 2]
        inside[start:start + chunk_size] = np.all(inequalities <= epsilon, axis=1)
    return {
        "coordinates": full, "projected": coordinates, "inside_projected": inside,
        "grid_k": grid_k.ravel(), "grid_l": grid_l.ravel(), "shape": tuple(map(int, counts)),
        "fill": fill, "unshifted_mean": points.mean(axis=0), "hull_equations": hull.equations,
    }


def check_full_hull(points, queries, epsilon, feasibility_tolerance, time_limit, threads, progress=None):
    n, d = points.shape
    a_ub = np.vstack([points.T, -points.T])
    a_eq = np.ones((1, n))
    options = {"primal_feasibility_tolerance": feasibility_tolerance,
               "dual_feasibility_tolerance": feasibility_tolerance,
               "time_limit": time_limit, "threads": threads}
    statuses = []
    with warnings.catch_warnings():
        # SciPy forwards this supported HiGHS option verbatim.
        warnings.filterwarnings("ignore", message="Unrecognized options detected:.*threads", category=OptimizeWarning)
        for index, query in enumerate(queries):
            result = linprog(
                np.zeros(n), A_ub=a_ub, b_ub=np.r_[query + epsilon, -query + epsilon],
                A_eq=a_eq, b_eq=[1.], bounds=(0, None), method="highs", options=options,
            )
            if result.success:
                # A feasible solver return must also meet the requested numerical
                # accuracy. epsilon is geometric slack, separate from LP accuracy.
                residual = float(np.max(np.abs(points.T @ result.x - query)))
                slack = 10 * feasibility_tolerance * max(1., float(np.max(np.abs(query))))
                if (residual > epsilon + slack or abs(result.x.sum() - 1) > slack
                        or result.x.min() < -slack):
                    raise RuntimeError(f"LP reported feasibility with inconsistent residual {residual:.17g}.")
                statuses.append("inside")
            elif result.status == 2:
                statuses.append("outside")
            else:
                raise RuntimeError(f"LP membership unresolved (status {result.status}): {result.message}")
            if progress is not None:
                progress(index + 1, len(queries))
    return np.array(statuses, dtype=str)
