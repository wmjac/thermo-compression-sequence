#!/usr/bin/env python3
"""Compute P=0 roots and explicitly capped concentrations on TWO-DIMENSIONAL slices."""
import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys
import shlex
import time
import numpy as np
from _post_training_common_v1 import (
    dimensions_argument, load_dimension, prepare_output, provenance, runtime, sha256, tsv, write_json,
)
from _pressure_zero_geometry_v1 import construct_slice, lp_indices, check_full_hull
from _pressure_zero_io_v1 import (
    start_log, dimension_log, flush_logs, load_feature_cloud, render_slice_maps,
)
from _pressure_zero_solver_v1 import SolverSettings, SolverFailure, match_member_roots
from _pressure_zero_reporting_v2 import solve_ray, summarize_curve, render_reported_map

LEGACY_REFERENCE = {
    "repository": "wmjac/idpmixture",
    "commit": "c425574a9fb61b24f6a8f2db748e18ef50cdc28b",
    "root_source": "calc_P_zeros.py",
    "root_blob": "e3f57504fd8607d96db28f8ff1e3e7b2a13195c0",
    "geometry_source": "calc_fe_slice.py",
    "geometry_blob": "3931bd10306b81c5150145499a5de2df0eb16f54",
}


class PressureEvaluator:
    """Use the existing ensemble API, each member's decoder and saved transform."""
    def __init__(self, saved, device, dtype, batch_size):
        import torch
        from mixture_models.evaluation import Ensemble
        self.ensemble = Ensemble(saved.run, saved.dimension, device, basis="canonical")
        self.checkpoint_dtypes = [str(next(model.parameters()).dtype) for model in self.ensemble.models]
        if dtype == "float64":
            for model in self.ensemble.models:
                model.to(dtype=torch.float64)
        self.evaluation_dtypes = [str(next(model.parameters()).dtype) for model in self.ensemble.models]
        self.k = self.ensemble.num_models
        self.batch_size = batch_size
        self.calls = 0

    def evaluate(self, coordinates, hessian=False):
        results = {"pressure_excess": [], "hessian": []}
        for start in range(0, len(coordinates), self.batch_size):
            out = self.ensemble.thermodynamics_from_latent(
                coordinates[start:start + self.batch_size], hessian=hessian
            )
            results["pressure_excess"].append(out["pressure_excess"])
            if hessian:
                results["hessian"].append(out["hessian"])
            self.calls += 1
        return {key: np.concatenate(value, axis=1) for key, value in results.items() if value}

    def ray_pressure(self, ray, concentration):
        concentration = np.atleast_1d(np.asarray(concentration, dtype=np.float64))
        coordinates = concentration[:, None] * ray[None]
        out = self.evaluate(coordinates)
        return out["pressure_excess"] + concentration[None]

    def ray_slope(self, ray, concentration):
        out = self.evaluate(float(concentration) * ray[None], hessian=True)
        return 1. + float(concentration) * np.einsum("i,kij,j->k", ray, out["hessian"][:, 0], ray)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Canonical run or combined analysis01 directory.")
    parser.add_argument("--features-dir", required=True, help="Completed v1/v2 sequence-feature export directory.")
    parser.add_argument("--dimensions", nargs="+", required=True, help="Learned dimensions d>=2 to slice.")
    parser.add_argument("--axes", type=int, nargs=2, required=True, help="Two varied canonical axes, 1-based.")
    parser.add_argument("--offset", nargs=2, action="append", default=[], metavar=("DIM", "SHIFT"),
                        help="Add SHIFT to the mean of a fixed coordinate DIM (1-based); repeat as needed.")
    parser.add_argument("--output-dir", required=True, help="Root containing d_NNN/ slice outputs and solver.log.")
    parser.add_argument("--voxel-size", type=float, required=True, help="Origin-aligned slice grid spacing.")
    parser.add_argument("--hull-policy", choices=("full", "projected"), default="full",
                        help="full: retain both hull tests; projected: legacy projected-only domain, with full-hull flags.")
    parser.add_argument("--hull-tol-mult", type=float, default=1.)
    parser.add_argument("--lp-num-points", default="all", help="all, or positive LP subsample count.")
    parser.add_argument("--lp-seed", type=int, default=0)
    parser.add_argument("--lp-feasibility-tol", type=float, default=1e-9)
    parser.add_argument("--lp-time-limit", type=float, default=60., help="Seconds per LP; unresolved LPs fail explicitly.")
    parser.add_argument("--max-grid-points", type=int, default=250000)
    parser.add_argument("--c-min", type=float, default=1e-8, help="Exclude roots at/below this concentration, nm^-3.")
    parser.add_argument("--c-tot-max", type=float, default=.25, help="Upper concentration including boundary, nm^-3.")
    parser.add_argument("--upper-bound-policy", choices=("cap", "roots-only"), default="cap",
                        help="cap: report c-tot-max if upper-bound pressure is negative beyond tolerance; roots-only: v1 reporting.")
    parser.add_argument("--n-brackets", type=int, default=64)
    parser.add_argument("--min-scan-refinements", type=int, default=1)
    parser.add_argument("--max-scan-refinements", type=int, default=3)
    parser.add_argument("--root-xtol", type=float, default=1e-12, help="Brent concentration absolute tolerance, nm^-3.")
    parser.add_argument("--root-rtol", type=float, default=1e-12)
    parser.add_argument("--pressure-atol", type=float, default=1e-10, help="Residual absolute tolerance in betaP, nm^-3.")
    parser.add_argument("--pressure-rtol", type=float, default=1e-8, help="Residual relative to c_tot; dimensionless.")
    parser.add_argument("--slope-threshold", type=float, default=1e-6, help="Minimum d(beta P)/dc; dimensionless.")
    parser.add_argument("--root-maxiter", type=int, default=100)
    parser.add_argument("--match-atol", type=float, default=1e-4, help="Member-root matching distance, nm^-3.")
    parser.add_argument("--match-rtol", type=float, default=.25)
    parser.add_argument("--evaluation-dtype", choices=("float64", "checkpoint"), default="float64")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--overwrite", choices=("yes", "no"), default="no")
    return parser


def validate(args):
    for key, value in vars(args).items():
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"{key} must be finite.")
    dims = dimensions_argument(args.dimensions)
    axes = tuple(a - 1 for a in args.axes)
    if len(set(axes)) != 2 or any(d < 2 or min(axes) < 0 or max(axes) >= d for d in dims):
        raise ValueError("This is a 2D slice: require d>=2 and two distinct axes within every selected dimension.")
    offsets = []
    for dim, value in args.offset:
        try:
            axis, shift = int(dim) - 1, float(value)
        except ValueError as exc:
            raise ValueError("--offset needs an integer coordinate index and a numeric shift.") from exc
        if (axis in axes or any(not 0 <= axis < d for d in dims)
                or not np.isfinite(shift) or axis in [i for i, _ in offsets]):
            raise ValueError("Offsets must identify distinct fixed coordinates present in every requested dimension.")
        offsets.append((axis, shift))
    if args.lp_num_points == "all":
        limit = None
    else:
        try:
            limit = int(args.lp_num_points)
        except ValueError as exc:
            raise ValueError("--lp-num-points is all or a positive integer.") from exc
        if limit <= 0:
            raise ValueError("--lp-num-points must be positive.")
    if (args.voxel_size <= 0 or args.hull_tol_mult < 0 or args.lp_seed < 0
            or not 1e-10 <= args.lp_feasibility_tol <= 1e-4 or args.lp_time_limit <= 0
            or args.max_grid_points < 1 or args.progress_every < 1
            or args.match_atol < 0 or args.match_rtol < 0):
        raise ValueError("Invalid geometry, LP, progress, or matching settings.")
    settings = SolverSettings(
        args.c_min, args.c_tot_max, args.n_brackets, args.min_scan_refinements,
        args.max_scan_refinements, args.root_xtol, args.root_rtol, args.pressure_atol,
        args.pressure_rtol, args.slope_threshold, args.root_maxiter,
    )
    settings.validate()
    return dims, axes, offsets, limit, settings


def compute_dimension(args, saved, output, device, axes, offsets, limit, settings, parent_log):
    d = saved.dimension
    directory = output / f"d_{d:03d}"
    directory.mkdir()
    log = dimension_log(parent_log, directory, d)
    meta = {
        "status": "in_progress", "dimension": d, "scope": "two-dimensional affine slice only",
        "varied_axes_1based": list(args.axes), "source": saved.source(),
        "voxel_size": args.voxel_size, "hull_policy": args.hull_policy,
        "c_tot_max": settings.c_max, "solver": asdict(settings),
        "units": {"concentration": "nm^-3", "pressure": "kT nm^-3",
                  "solver_pressure": "beta P in nm^-3", "energy": "kT", "slope": "dimensionless"},
    }
    meta["reporting"] = {
        "upper_bound_policy": args.upper_bound_policy, "primary_plot": "c_tot_reported.png",
        "cap_rule": "beta P(c_tot_max) < -(pressure_atol + pressure_rtol*c_tot_max)",
        "upper_bound_beta_pressure_tolerance": settings.pressure_atol + settings.pressure_rtol*settings.c_max,
        "cap_meaning": "reporting cap; last negative-pressure interval extends to the upper boundary; a finite higher root is not proven",
        "root_counts": "actual accepted roots only; caps are never added to roots.tsv or uncertainty statistics",
        "extra_crossings": "all raw crossings retained; no smoothing or fluctuation-based filtering",
    }
    write_json(directory / "metadata.json", meta)
    try:
        log.info("Exact workflow invocation: %s", shlex.join([sys.executable, *sys.argv]))
        log.info("Working directory: %s", Path.cwd())
        cloud, global_rmsd, feature_source = load_feature_cloud(args.features_dir, saved)
        epsilon = args.hull_tol_mult * global_rmsd
        geometry = construct_slice(
            cloud, axes, offsets, args.voxel_size, epsilon, args.max_grid_points, args.batch_size
        )
        fixed = {str(j + 1): float(geometry["fill"][j]) for j in range(d) if j not in axes}
        definition = {
            "scope": "2D slice; this is not the full d-dimensional zero-pressure surface",
            "varied_axes_1based": list(args.axes), "fixed_coordinates": fixed,
            "unshifted_sequence_mean": geometry["unshifted_mean"],
            "offsets_1based": {str(j+1): shift for j, shift in offsets},
            "population": "uniform arithmetic mean over the feature TSV sequence rows",
            "shared_canonical_ray": "every model evaluates x=c_tot*zbar at the same canonical slice point",
        }
        write_json(directory / "slice_definition.json", definition)
        meta.update(
            feature_source=feature_source, slice_definition=definition, fixed_coordinates=fixed,
            hull_epsilon=epsilon, global_rmsd=global_rmsd, grid_shape=geometry["shape"],
        )
        log.info("Dimension %d: varying z_%d,z_%d; this is only a 2D slice.", d, args.axes[0], args.axes[1])
        log.info("Fixed coordinates (exact values): %s", json.dumps(fixed, sort_keys=True))
        log.info("Unshifted sequence-population mean: %s", json.dumps(geometry["unshifted_mean"].tolist()))
        log.info("Model/feature source: %s", json.dumps(feature_source, sort_keys=True))
        log.info("Hull tolerance epsilon=%.17g = tol_mult %.17g * global_RMSD %.17g",
                 epsilon, args.hull_tol_mult, global_rmsd)
        projected_ids = np.flatnonzero(geometry["inside_projected"])
        n_points = len(geometry["coordinates"])
        log.info("Grid: %d bounding-box centers; %d inside projected hull; spacing %.17g.",
                 n_points, len(projected_ids), args.voxel_size)
        selected = lp_indices(cloud, limit, args.lp_seed)
        np.save(directory / "lp_indices.npy", selected)
        lp_cloud = cloud[selected]
        meta["lp"] = {
            "num_exported_sequences": len(cloud), "num_points_used": len(selected),
            "subsampled": limit is not None and limit < len(cloud), "seed": args.lp_seed,
            "exact_duplicates_removed": True, "indices_file": "lp_indices.npy",
            "epsilon": epsilon, "feasibility_tolerance": args.lp_feasibility_tol,
            "time_limit_seconds": args.lp_time_limit, "threads": args.threads,
            "membership": "|Z_lp.T*w-query| <= epsilon componentwise; sum(w)=1, w>=0",
            "subsampling_note": "An outside result concerns the selected LP cloud; subsampling can exclude points inside the full sequence-pool hull.",
        }
        log.info("LP call: scipy.optimize.linprog(method='highs'); points=%d; threads=%d; "
                 "feasibility_tolerance=%.17g; time_limit=%.17g s; subsampled=%s",
                 len(selected), args.threads, args.lp_feasibility_tol, args.lp_time_limit, meta["lp"]["subsampled"])
        if meta["lp"]["subsampled"]:
            log.info("LP is SUBSAMPLED: outside selected-cloud hull does not prove outside full-pool hull.")
        last_progress = [time.monotonic()]

        def lp_progress(done, total):
            if done == total or time.monotonic() - last_progress[0] >= 10:
                log.info("Full-dimensional hull LP progress: %d/%d", done, total)
                last_progress[0] = time.monotonic()

        full_status = np.full(n_points, "not_tested_projected_exclusion", dtype=object)
        full_status[projected_ids] = check_full_hull(
            lp_cloud, geometry["coordinates"][projected_ids], epsilon,
            args.lp_feasibility_tol, args.lp_time_limit, args.threads, lp_progress,
        )
        retained = geometry["inside_projected"].copy()
        if args.hull_policy == "full":
            retained &= full_status == "inside"
        log.info("Hull policy %s: %d retained; excluded points get no markers in plots.",
                 args.hull_policy, int(retained.sum()))
        if not retained.any():
            log.info("This slice has no retained grid centers; output maps will be empty. Inspect fixed coordinates and spacing.")
        np.savez_compressed(directory / "grid.npz", coordinates=geometry["coordinates"],
                            grid_k=geometry["grid_k"], grid_l=geometry["grid_l"],
                            inside_projected=geometry["inside_projected"], full_hull_status=full_status.astype(str),
                            retained=retained, hull_equations=geometry["hull_equations"])
        del cloud, lp_cloud
        evaluator = PressureEvaluator(saved, device, args.evaluation_dtype, args.batch_size)
        meta["evaluation"] = {
            "requested_dtype": args.evaluation_dtype, "checkpoint_dtypes": evaluator.checkpoint_dtypes,
            "evaluation_dtypes": evaluator.evaluation_dtypes, "num_models": evaluator.k,
            "slope_method": "autodifferentiated Hessian: 1+c*zbar.T*H_ex*zbar",
            "scalar_objective": "Q(c)=mean_m(beta P_m(c))/c for mean curve; Q_m(c)=beta P_m(c)/c for members",
            "uncertainty": "member roots on the same shared canonical ray; population SD, ddof=0; not SEM",
            "matching": {"method": "unique-nearest local root; reject ambiguous or out-of-tolerance matches",
                         "atol_nm^-3": args.match_atol, "rtol": args.match_rtol,
                         "sd_minimum_matched_models": 2},
        }
        write_json(directory / "metadata.json", meta)
        log.info("Decoder evaluation dtypes: %s; checkpoint dtypes: %s.",
                 evaluator.evaluation_dtypes, evaluator.checkpoint_dtypes)
        log.info("Pressure root: beta P(c)=c+mean_m beta P_ex,m(c*zbar), shared canonical zbar.")
        log.info("Brent objective Q(c)=beta P(c)/c. Positive roots are unchanged; vacuum c=0 is excluded.")
        log.info("Solver called as scipy.optimize.root_scalar(Q, bracket=(c_left,c_right), method='brentq', "
                 "xtol=%.17g, rtol=%.17g, maxiter=%d).", settings.root_xtol, settings.root_rtol, settings.maxiter)
        log.info("All solver settings: %s", json.dumps(asdict(settings), sort_keys=True))
        log.info("Upper-bound reporting: %s", json.dumps(meta["reporting"], sort_keys=True))
        log.info("Accept residual |beta P| <= %.17g + %.17g*c; accept slope d(beta P)/dc > %.17g.",
                 settings.pressure_atol, settings.pressure_rtol, settings.slope_threshold)
        log.info("Scan refinement doubles initial %d brackets until root sets agree; min=%d, max=%d refinements. "
                 "Finite-grid agreement is a convergence diagnostic, not a proof of exhaustive roots.",
                 settings.n_brackets, settings.min_refinements, settings.max_refinements)
        all_roots, member_rows, candidate_rows, summary_rows, match_rows = [], [], [], [], []
        root_lists = {}
        curve_reports, curve_rows = {}, []
        point_results = {}
        retained_ids = np.flatnonzero(retained)
        last_progress[0] = time.monotonic()
        for done, point_id in enumerate(retained_ids, 1):
            ray = geometry["coordinates"][point_id]

            def trace(record):
                log.debug("point_id=%d solver_event=%s", point_id, json.dumps(record, sort_keys=True))

            result = solve_ray(
                lambda c: evaluator.ray_pressure(ray, c),
                lambda c: evaluator.ray_slope(ray, c), settings, trace,
            )
            point_id = int(point_id)
            roots = result["roots"][-1]
            root_lists[point_id] = roots
            point_results[point_id] = result
            upper = result["member_beta_pressure_at_c_tot_max"]
            for member in [-1, *range(evaluator.k)]:
                upper_pressure = float(upper.mean() if member == -1 else upper[member])
                report = summarize_curve(result["roots"][member], upper_pressure, settings, args.upper_bound_policy)
                curve_rows.append({"point_id": point_id, "member": member, **report})
                if member == -1:
                    curve_reports[point_id] = report
            candidate_rows.extend({"point_id": point_id, **row} for row in result["candidates"])
            all_roots.extend({"point_id": point_id, "ensemble_root_index": index+1, **root}
                             for index, root in enumerate(roots))
            per_member = [result["roots"][m] for m in range(evaluator.k)]
            member_rows.extend({"point_id": point_id, "member_root_index": index+1, **root}
                               for m, rows in enumerate(per_member) for index, root in enumerate(rows))
            summaries, matches = match_member_roots(roots, per_member, args.match_atol, args.match_rtol)
            for row in summaries:
                pressure = evaluator.ray_pressure(ray, [row["ensemble_c_tot"]])[:, 0]
                summary_rows.append({"point_id": point_id, **row,
                                     "pressure_model_std_at_ensemble_root": float(np.std(pressure, ddof=0))})
            match_rows.extend({"point_id": point_id, **row} for row in matches)
            if done == 1 or done == len(retained_ids) or done % args.progress_every == 0 or time.monotonic() - last_progress[0] >= 10:
                log.info("Root progress %d/%d; point_id=%d; ensemble roots=%d; final brackets=%d.",
                         done, len(retained_ids), point_id, len(roots), result["final_brackets"])
                last_progress[0] = time.monotonic()
        lattice = []
        coordinate_names = [f"z_{j+1}_{'varied' if j in axes else 'fixed'}" for j in range(d)]
        for i, ray in enumerate(geometry["coordinates"]):
            record = point_results.get(i)
            status = ("solved" if root_lists.get(i) else "no_positive_slope_root") if retained[i] else (
                "excluded_projected_hull" if not geometry["inside_projected"][i] else "excluded_full_hull"
            )
            report = curve_reports.get(i, {})
            lattice.append({
                "point_id": i, "grid_k": int(geometry["grid_k"][i]), "grid_l": int(geometry["grid_l"][i]),
                **{name: float(value) for name, value in zip(coordinate_names, ray)},
                "inside_projected_hull": int(geometry["inside_projected"][i]), "full_hull_status": full_status[i],
                "retained": int(retained[i]), "n_solutions": len(root_lists.get(i, [])) if retained[i] else None,
                "solver_status": status, "final_brackets": record["final_brackets"] if record else None,
                "refinements": record["refinements"] if record else None,
                "reported_c_tot": report.get("reported_c_tot"),
                "reported_status": report.get("reported_status", status),
                "upper_bound_censored": report.get("upper_bound_censored"),
                "upper_bound_pressure_negative": report.get("upper_bound_pressure_negative"),
                "beta_pressure_at_c_tot_max": report.get("beta_pressure_at_c_tot_max"),
                "upper_bound_beta_pressure_tolerance": report.get("upper_bound_beta_pressure_tolerance"),
                "largest_root_in_range": report.get("largest_root_in_range"),
            })
        tsv(directory / "lattice.tsv", lattice)
        tsv(directory / "curve_summary.tsv", curve_rows, [
            "point_id", "member", "reported_c_tot", "reported_status", "upper_bound_censored",
            "upper_bound_pressure_negative", "beta_pressure_at_c_tot_max",
            "upper_bound_beta_pressure_tolerance", "largest_root_in_range", "n_solutions",
        ])
        root_fields = [
            "member", "refinement", "brackets", "method", "bracket_left", "bracket_right",
            "c_tot", "pressure_residual", "normalized_residual", "slope", "iterations", "function_calls",
            "pressure_acceptance_threshold", "status",
        ]
        tsv(directory / "roots.tsv", all_roots, ["point_id", "ensemble_root_index", *root_fields])
        tsv(directory / "member_roots.tsv", member_rows, ["point_id", "member_root_index", *root_fields])
        tsv(directory / "root_candidates.tsv", candidate_rows, ["point_id", *root_fields])
        tsv(directory / "root_summary.tsv", summary_rows, [
            "point_id", "ensemble_root_index", "ensemble_c_tot", "member_root_mean", "member_root_std",
            "n_models_matched", "n_models_total", "matching_status", "pressure_model_std_at_ensemble_root",
        ])
        tsv(directory / "root_matching.tsv", match_rows,
            ["point_id", "member", "member_root", "status", "ensemble_root"])
        meta["plots"] = render_slice_maps(directory, geometry, axes, retained, root_lists, summary_rows, meta)
        meta["plots"].insert(0, render_reported_map(directory, geometry, axes, retained, curve_reports, meta))
        meta["counts"] = {
            "bbox_grid_points": n_points, "projected_hull_points": len(projected_ids),
            "retained_points": len(retained_ids), "points_with_roots": sum(bool(rows) for rows in root_lists.values()),
            "ensemble_roots": len(all_roots), "member_roots": len(member_rows),
            "ensemble_decoder_batch_calls": evaluator.calls,
            "points_capped_at_c_tot_max": sum(row["upper_bound_censored"] for row in curve_reports.values()),
            "points_with_reported_concentration": sum(row["reported_c_tot"] is not None for row in curve_reports.values()),
        }
        source_files = {
            saved.best / "selection.json": saved.config["selection_sha256"],
            saved.canonical / "config.json": saved.source()["canonical_config_sha256"],
            Path(feature_source["feature_tsv"]): feature_source["feature_tsv_sha256"],
            Path(feature_source["metadata_path"]): feature_source["metadata_sha256"],
            **{saved.best / row["checkpoint"]: row["checkpoint_sha256"] for row in saved.selection["models"]},
        }
        for path, digest in source_files.items():
            if sha256(path) != digest:
                raise ValueError(f"Input changed during calculation: {path}")
        log.info("Completed d=%d TWO-DIMENSIONAL SLICE: %s", d, json.dumps(meta["counts"], sort_keys=True))
        log.info("Outputs: %s", directory)
        flush_logs(log)
        # Logs remain open; their exact content can receive an exception later,
        # so immutable numeric/plot artifacts alone enter this hash manifest.
        meta["files_sha256"] = {
            path.name: sha256(path) for path in directory.iterdir()
            if path.is_file() and path.name not in ("metadata.json", "solver.log")
        }
        meta["status"] = "complete"
        write_json(directory / "metadata.json", meta)
        return meta
    except Exception:
        log.exception("SLICE CALCULATION FAILED; this is not classified as absence of a pressure root.")
        meta["status"] = "failed"
        write_json(directory / "metadata.json", meta)
        raise


def main():
    args = build_parser().parse_args()
    dimensions, axes, offsets, limit, settings = validate(args)
    device = runtime(args)
    saved = [load_dimension(args.run_dir, d, features=False) for d in dimensions]
    output = prepare_output(args.output_dir, args.overwrite,
                            [args.run_dir, args.features_dir, *[item.run for item in saved]])
    log = start_log(output, [sys.executable, *sys.argv])
    meta = {
        "status": "in_progress", "scope": "two-dimensional affine slices only; no full-dimensional surface reconstruction",
        "provenance": provenance(args), "legacy_reference": LEGACY_REFERENCE,
        "arguments": vars(args), "dimensions": {},
        "changes_from_legacy": [
            "arbitrary selected ensemble size; current canonical features and physical units",
            "default full-dimensional hull filtering; projected option preserves legacy projected domain",
            "normalized pressure scalar objective; float64 evaluation by default",
            "autodifferentiated slopes; residual checks; converged bracket-grid refinement",
            "per-model roots and guarded uncertainty matching; full-precision fixed-width tables",
            "exact root counts; fixed coordinates explicit; hull-excluded centers unmarked",
            "v2: explicit upper-bound caps in reported concentrations; actual raw roots retained",
        ],
    }
    meta["provenance"]["analysis_source_sha256"].update({
        name: sha256(Path(__file__).parent / name) for name in
        ("calc_pressure_zeros_v2.py", "_pressure_zero_reporting_v2.py")
    })
    write_json(output / "metadata.json", meta)
    log.info("All CLI values: %s", json.dumps(vars(args), sort_keys=True))
    log.info("Legacy baseline: %s", json.dumps(LEGACY_REFERENCE, sort_keys=True))
    try:
        for item in saved:
            record = compute_dimension(args, item, output, device, axes, offsets, limit, settings, log)
            meta["dimensions"][f"d_{item.dimension:03d}"] = {
                "status": record["status"], "counts": record["counts"],
                "fixed_coordinates": record["fixed_coordinates"],
                "metadata": f"d_{item.dimension:03d}/metadata.json",
            }
            write_json(output / "metadata.json", meta)
        log.info("All requested TWO-DIMENSIONAL SLICES complete: %s", output)
        meta["status"] = "complete"
        write_json(output / "metadata.json", meta)
    except Exception:
        log.exception("P=0 slice workflow failed. Inspect d_NNN/solver.log and metadata.json.")
        meta["status"] = "failed"
        write_json(output / "metadata.json", meta)
        raise
    finally:
        flush_logs(log)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FloatingPointError, SolverFailure, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}")
