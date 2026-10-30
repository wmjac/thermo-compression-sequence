"""Dimension-independent positive-concentration pressure root solver."""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.optimize import root_scalar


class SolverFailure(RuntimeError):
    """A numerical failure must not be classified as absence of a root."""


@dataclass(frozen=True)
class SolverSettings:
    c_min: float = 1e-8
    c_max: float = 0.25
    n_brackets: int = 64
    min_refinements: int = 1
    max_refinements: int = 3
    root_xtol: float = 1e-12
    root_rtol: float = 1e-12
    pressure_atol: float = 1e-10
    pressure_rtol: float = 1e-8
    slope_threshold: float = 1e-6
    maxiter: int = 100

    def validate(self):
        if not all(np.isfinite(value) for value in asdict(self).values()):
            raise ValueError("Solver settings must be finite.")
        if not 0 < self.c_min < self.c_max:
            raise ValueError("Require 0 < c-min < c-tot-max.")
        if (self.n_brackets < 2 or self.min_refinements < 1
                or self.max_refinements < self.min_refinements or self.maxiter < 1):
            raise ValueError("Require at least two brackets, at least one refinement, and positive maxiter.")
        if (self.root_xtol <= 0 or self.root_rtol < 4 * np.finfo(float).eps
                or self.pressure_atol < 0 or self.pressure_rtol < 0
                or self.pressure_atol + self.pressure_rtol <= 0 or self.slope_threshold < 0):
            raise ValueError("Invalid root, pressure, or slope tolerances.")


def roots_agree(left, right, settings):
    if len(left) != len(right):
        return False
    return all(abs(a["c_tot"] - b["c_tot"]) <= 10 * (
        settings.root_xtol + settings.root_rtol * max(abs(a["c_tot"]), abs(b["c_tot"]))
    ) for a, b in zip(left, right))


def solve_ray(pressure, slope, settings, trace=None):
    """Solve all k member curves and their mean along one arbitrary latent ray.

    pressure(c_array) -> (k, len(c_array)) in beta-pressure units nm^-3.
    slope(c) -> (k,) derivatives d(beta P_m)/dc.
    The solver knows nothing about 2D grids or convex hulls.
    """
    settings.validate()
    cache = {}
    candidates = []
    evaluations = 0

    def evaluate(c):
        nonlocal evaluations
        cs = np.atleast_1d(np.asarray(c, dtype=np.float64))
        missing = np.unique([float(x) for x in cs if float(x) not in cache])
        if len(missing):
            values = np.asarray(pressure(missing), dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != len(missing) or not values.shape[0]:
                raise SolverFailure("pressure(c) must have shape (models,number_of_concentrations).")
            if not np.isfinite(values).all():
                raise SolverFailure("Nonfinite pressure evaluation.")
            if cache and values.shape[0] != len(next(iter(cache.values()))):
                raise SolverFailure("Pressure callback changed its model count.")
            for j, x in enumerate(missing):
                cache[float(x)] = values[:, j]
            evaluations += len(missing)
        return np.column_stack([cache[float(x)] for x in cs])

    def extract(values, member):
        return values.mean(axis=0) if member == -1 else values[member]

    previous = None
    for level in range(settings.max_refinements + 1):
        brackets = settings.n_brackets * 2**level
        grid = np.linspace(settings.c_min, settings.c_max, brackets + 1)
        member_values = evaluate(grid)
        k = member_values.shape[0]
        current = {}
        for member in [-1, *range(k)]:
            p = extract(member_values, member)
            q = p / grid
            if not np.isfinite(q).all():
                raise SolverFailure("Nonfinite normalized pressure; inspect c-min and pressure scale.")
            curve_roots = []
            guesses = []
            # Only exact floating-point zeros are taken as endpoint roots.
            # Approximate zeros are not accepted without a bracket.
            for j in np.flatnonzero(q == 0):
                guesses.append((float(grid[j]), float(grid[j]), float(grid[j]), 0, 0, "grid_zero"))
            for j in np.flatnonzero((q[:-1] != 0) & (q[1:] != 0)
                                    & (np.signbit(q[:-1]) != np.signbit(q[1:]))):
                left, right = float(grid[j]), float(grid[j + 1])

                def normalized_pressure(c):
                    return float(extract(evaluate([c]), member)[0] / c)

                try:
                    solution = root_scalar(
                        normalized_pressure, bracket=(left, right), method="brentq",
                        xtol=settings.root_xtol, rtol=settings.root_rtol,
                        maxiter=settings.maxiter,
                    )
                except (ValueError, RuntimeError) as exc:
                    raise SolverFailure(
                        f"Brent failure for member={member}, bracket=({left:.17g},{right:.17g}): {exc}"
                    ) from exc
                if not solution.converged:
                    raise SolverFailure(f"Brent did not converge for member={member}, bracket=({left},{right}).")
                guesses.append((float(solution.root), left, right, int(solution.iterations),
                                int(solution.function_calls), "brentq"))
            for c, left, right, iterations, calls, method in sorted(guesses):
                if any(abs(c - r["c_tot"]) <= 10 * (settings.root_xtol
                        + settings.root_rtol * abs(c)) for r in curve_roots):
                    continue
                all_p = evaluate([c])[:, 0]
                value = float(all_p.mean() if member == -1 else all_p[member])
                slopes = np.asarray(slope(c), dtype=np.float64)
                if slopes.shape != (k,) or not np.isfinite(slopes).all():
                    raise SolverFailure("Slope callback must return one finite autodifferentiated slope per model.")
                derivative = float(slopes.mean() if member == -1 else slopes[member])
                threshold = settings.pressure_atol + settings.pressure_rtol * c
                status = ("excluded_near_zero" if c <= settings.c_min else
                          "rejected_nonpositive_slope" if derivative <= settings.slope_threshold else "accepted")
                if abs(value) > threshold:
                    status = "failed_pressure_residual"
                record = {
                    "member": member, "refinement": level, "brackets": brackets,
                    "method": method, "bracket_left": left, "bracket_right": right,
                    "c_tot": c, "pressure_residual": value, "normalized_residual": value / c,
                    "slope": derivative, "iterations": iterations, "function_calls": calls,
                    "pressure_acceptance_threshold": threshold, "status": status,
                }
                candidates.append(record)
                if trace is not None:
                    trace(record)
                if abs(value) > threshold:
                    raise SolverFailure(
                        f"Root residual failed for member={member}: c={c:.17g}, betaP={value:.17g}, "
                        f"allowed={threshold:.17g}; use float64 evaluation and inspect tolerances."
                    )
                if status == "accepted":
                    curve_roots.append(record)
            current[member] = curve_roots
        stable = previous is not None and all(
            roots_agree(previous[m], current[m], settings) for m in current
        )
        if trace is not None:
            trace({"event": "scan", "refinement": level, "brackets": brackets,
                   "root_counts": {str(m): len(roots) for m, roots in current.items()}, "stable": stable})
        if level >= settings.min_refinements and stable:
            return {"roots": current, "candidates": candidates, "refinements": level,
                    "final_brackets": brackets, "pressure_points_evaluated": evaluations,
                    "scan_converged": True}
        previous = current
    raise SolverFailure(
        f"Root sets did not stabilize by {settings.n_brackets * 2**settings.max_refinements} brackets. "
        "Increase --max-scan-refinements and inspect the solver log."
    )


def match_member_roots(ensemble_roots, member_roots, atol, rtol):
    """Unique-nearest local matching; ambiguous branches are never averaged."""
    references = np.array([r["c_tot"] for r in ensemble_roots])
    assignments = []
    diagnostics = []
    for member, roots in enumerate(member_roots):
        proposed = {}
        for index, root in enumerate(roots):
            if not len(references):
                diagnostics.append({"member": member, "member_root": index + 1,
                                    "status": "no_ensemble_root", "ensemble_root": None})
                continue
            distances = np.abs(references - root["c_tot"])
            nearest = int(np.argmin(distances))
            close = np.flatnonzero(np.isclose(distances, distances[nearest], rtol=1e-12, atol=1e-15))
            if distances[nearest] > atol + rtol * abs(references[nearest]):
                status = "outside_matching_tolerance"
            elif len(close) != 1:
                status = "ambiguous_equal_distance"
            else:
                status = "proposed"
                proposed.setdefault(nearest, []).append(index)
            diagnostics.append({"member": member, "member_root": index + 1, "status": status,
                                "ensemble_root": nearest + 1 if status == "proposed" else None})
        accepted = {}
        for ref_index, indices in proposed.items():
            for index in indices:
                diag = next(row for row in diagnostics
                            if row["member"] == member and row["member_root"] == index + 1)
                if len(indices) == 1:
                    diag["status"] = "matched"
                    accepted[ref_index] = float(roots[index]["c_tot"])
                else:
                    diag["status"] = "ambiguous_multiple_roots"
        assignments.append(accepted)
    rows = []
    for index, reference in enumerate(ensemble_roots):
        values = [assignment[index] for assignment in assignments if index in assignment]
        n = len(values)
        rows.append({
            "ensemble_root_index": index + 1, "ensemble_c_tot": float(reference["c_tot"]),
            "member_root_mean": float(np.mean(values)) if n else None,
            "member_root_std": float(np.std(values, ddof=0)) if n >= 2 else None,
            "n_models_matched": n, "n_models_total": len(member_roots),
            "matching_status": "complete" if n == len(member_roots) and n >= 2
                               else "partial" if n >= 2 else "insufficient_matches",
        })
    return rows, diagnostics
