#!/usr/bin/env python3
"""Excess free-energy density and gradient on a fixed-c_tot canonical 2D slice."""
import argparse
import csv
import json
import logging
from pathlib import Path
import shlex
import sys
import time
import numpy as np
from _post_training_common_v1 import (
    dimensions_argument, load_dimension, prepare_output, provenance, runtime, sha256, tsv, write_json,
)
from _pressure_zero_geometry_v1 import construct_slice, lp_indices, check_full_hull
from _pressure_zero_io_v1 import load_feature_cloud, flush_logs
from _free_energy_slice_plots_v1 import render_maps

LEGACY_REFERENCE = {
    "repository": "wmjac/idpmixture", "path": "calc_fe_slice.py",
    "git_blob_sha": "3931bd10306b81c5150145499a5de2df0eb16f54",
    "url": "https://github.com/wmjac/idpmixture/blob/main/calc_fe_slice.py",
}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="Canonical run or combined analysis01 directory.")
    p.add_argument("--features-dir", required=True, help="Completed v1/v2 sequence-feature export.")
    p.add_argument("--dimensions", nargs="+", required=True, help="Learned dimensions d>=2.")
    p.add_argument("--axes", type=int, nargs=2, required=True, help="Two varied canonical axes, 1-based.")
    p.add_argument("--offset", nargs=2, action="append", default=[], metavar=("DIM", "SHIFT"),
                   help="Add SHIFT to the mean of fixed coordinate DIM, 1-based; repeat as needed.")
    p.add_argument("--c-tot", type=float, required=True, help="Fixed total chain concentration in nm^-3; zero is allowed.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--voxel-size", type=float, required=True, help="Origin-aligned spacing in zbar coordinates.")
    p.add_argument("--hull-policy", choices=("full", "projected"), default="full")
    p.add_argument("--hull-tol-mult", type=float, default=1.)
    p.add_argument("--lp-num-points", default="all")
    p.add_argument("--lp-seed", type=int, default=0)
    p.add_argument("--lp-feasibility-tol", type=float, default=1e-9)
    p.add_argument("--lp-time-limit", type=float, default=60., help="Seconds per LP; unresolved LPs fail.")
    p.add_argument("--max-grid-points", type=int, default=250000)
    p.add_argument("--evaluation-dtype", choices=("float64", "checkpoint"), default="float64")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=20, help="Log every N thermodynamic evaluation batches.")
    p.add_argument("--arrow-stride", type=int, default=5, help="Show arrows at every Nth grid index along each axis.")
    p.add_argument("--arrow-scale", default="auto", help="auto, or a positive Matplotlib quiver scale; larger means shorter arrows.")
    p.add_argument("--save-member-values", choices=("yes", "no"), default="yes")
    p.add_argument("--overwrite", choices=("yes", "no"), default="no")
    return p


def validate(args):
    for key, value in vars(args).items():
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"{key} must be finite.")
    dims = dimensions_argument(args.dimensions)
    axes = tuple(a - 1 for a in args.axes)
    if len(set(axes)) != 2 or any(d < 2 or min(axes) < 0 or max(axes) >= d for d in dims):
        raise ValueError("Require d>=2 and two distinct axes present in every selected dimension.")
    offsets = []
    for dim, value in args.offset:
        try:
            axis, shift = int(dim)-1, float(value)
        except ValueError as exc:
            raise ValueError("--offset requires an integer coordinate and a numeric shift.") from exc
        if (axis in axes or any(not 0 <= axis < d for d in dims)
                or not np.isfinite(shift) or axis in [i for i, _ in offsets]):
            raise ValueError("Offsets must select distinct fixed coordinates present in every requested dimension.")
        offsets.append((axis, shift))
    try:
        limit = None if args.lp_num_points == "all" else int(args.lp_num_points)
    except ValueError as exc:
        raise ValueError("--lp-num-points must be all or a positive integer.") from exc
    if (args.c_tot < 0 or args.voxel_size <= 0 or args.hull_tol_mult < 0
            or limit is not None and limit <= 0 or args.lp_seed < 0
            or not 1e-10 <= args.lp_feasibility_tol <= 1e-4 or args.lp_time_limit <= 0
            or min(args.max_grid_points, args.progress_every, args.arrow_stride, args.batch_size, args.threads) < 1):
        raise ValueError("Invalid concentration, grid, LP, batch, progress, or arrow settings.")
    try:
        scale = None if args.arrow_scale == "auto" else float(args.arrow_scale)
    except ValueError as exc:
        raise ValueError("--arrow-scale must be auto or a positive number.") from exc
    if scale is not None and (not np.isfinite(scale) or scale <= 0):
        raise ValueError("--arrow-scale must be auto or a positive finite number.")
    return dims, axes, offsets, limit, scale


def evaluate_fields(ensemble, zbar, c_tot, batch_size, progress=None):
    """Shared canonical zbar; gradient is transformed by the existing API."""
    zbar = np.asarray(zbar, dtype=np.float64)
    k, n, d = ensemble.num_models, len(zbar), ensemble.dimension
    if (zbar.ndim != 2 or zbar.shape[1] != d or not np.isfinite(zbar).all()
            or not np.isfinite(c_tot) or c_tot < 0 or batch_size < 1 or k < 1):
        raise ValueError("Invalid canonical points, concentration, batch size, or ensemble size.")
    f_ex = np.empty((k, n), dtype=np.float64)
    grad_x = np.empty((k, n, d), dtype=np.float64)
    for batch, start in enumerate(range(0, n, batch_size), 1):
        stop = min(start + batch_size, n)
        terms = ensemble.thermodynamics_from_latent(c_tot * zbar[start:stop])
        f = np.asarray(terms["free_energy_excess"], dtype=np.float64)
        g = np.asarray(terms["gradient"], dtype=np.float64)
        if f.shape != (k, stop-start) or g.shape != (k, stop-start, d):
            raise ValueError("Thermodynamic API returned incompatible field shapes.")
        if not np.isfinite(f).all() or not np.isfinite(g).all():
            raise FloatingPointError("Nonfinite free energy or gradient; calculation failed.")
        f_ex[:, start:stop], grad_x[:, start:stop] = f, g
        if progress is not None:
            progress(stop, n, batch)
    with np.errstate(over="raise", invalid="raise"):
        grad_zbar = c_tot * grad_x
    return {"f_ex": f_ex, "grad_x": grad_x, "grad_zbar": grad_zbar}


def start_log(output):
    logger = logging.getLogger("free_energy_slice")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    formatter.converter = time.gmtime
    handlers = ((logging.StreamHandler(sys.stdout), logging.INFO),
                (logging.FileHandler(output/"analysis.log", mode="w", encoding="utf-8"), logging.DEBUG))
    for handler,level in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)
    logger.info("Excess free-energy density and gradient on TWO-DIMENSIONAL fixed-concentration slices.")
    logger.info("Exact invocation: %s", shlex.join([sys.executable, *sys.argv]))
    logger.info("Working directory: %s", Path.cwd())
    return logger


def dimension_log(parent, directory, d):
    logger = parent.getChild(f"d_{d:03d}")
    logger.setLevel(logging.DEBUG)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    handler = logging.FileHandler(directory/"analysis.log", mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(parent.handlers[0].formatter)
    logger.addHandler(handler)
    return logger


def compute_dimension(args, saved, output, device, axes, offsets, limit, scale, parent_log):
    import torch
    from mixture_models.evaluation import Ensemble
    directory = output/f"d_{saved.dimension:03d}"
    directory.mkdir()
    log = dimension_log(parent_log, directory, saved.dimension)
    meta = {"status": "in_progress", "dimension": saved.dimension, "scope": "two-dimensional affine slice only",
            "c_tot": args.c_tot, "varied_axes_1based": args.axes, "source": saved.source(),
            "voxel_size": args.voxel_size, "hull_policy": args.hull_policy,
            "units": {"concentration": "nm^-3", "f_ex": "kT nm^-3", "grad_x": "kT per unit canonical feature",
                      "grad_zbar": "kT nm^-3 per unit canonical feature"},
            "derivatives": {"grad_x": "d Psi / d x in canonical coordinates",
                            "grad_zbar": "c_tot * grad_x; fixed c_tot and other zbar coordinates",
                            "plot": "positive in-plane gradient of the ensemble-mean excess free-energy density"},
            "uncertainty": "population standard deviation across all selected models, ddof=0; not SEM"}
    write_json(directory/"metadata.json", meta)
    try:
        log.info("Exact invocation: %s", shlex.join([sys.executable, *sys.argv]))
        log.info("Fixed c_tot=%.17g nm^-3; x=c_tot*zbar. grad_zbar=c_tot*grad_x.", args.c_tot)
        cloud, rmsd, feature_source = load_feature_cloud(args.features_dir, saved)
        epsilon = args.hull_tol_mult*rmsd
        geometry = construct_slice(cloud, axes, offsets, args.voxel_size, epsilon, args.max_grid_points, args.batch_size)
        fixed = {str(j+1): float(geometry["fill"][j]) for j in range(saved.dimension) if j not in axes}
        definition = {"scope": "2D slice only; all unvaried coordinates are fixed",
                      "c_tot": args.c_tot, "varied_axes_1based": args.axes, "fixed_coordinates": fixed,
                      "unshifted_sequence_mean": geometry["unshifted_mean"],
                      "offsets_1based": {str(j+1): v for j,v in offsets},
                      "population": "uniform arithmetic mean over feature TSV sequence rows",
                      "shared_canonical_point": "all models evaluate the same x=c_tot*zbar; no sequence re-encoding"}
        write_json(directory/"slice_definition.json", definition)
        meta.update(fixed_coordinates=fixed, slice_definition=definition, feature_source=feature_source,
                    hull_epsilon=epsilon, global_rmsd=rmsd, grid_shape=geometry["shape"])
        log.info("Fixed zbar coordinates (exact): %s", json.dumps(fixed, sort_keys=True))
        log.info("Unshifted sequence mean: %s", json.dumps(geometry["unshifted_mean"].tolist()))
        log.info("Hull epsilon=%.17g = multiplier %.17g * global RMSD %.17g.", epsilon, args.hull_tol_mult, rmsd)
        selected = lp_indices(cloud, limit, args.lp_seed)
        np.save(directory/"lp_indices.npy", selected)
        projected_ids = np.flatnonzero(geometry["inside_projected"])
        full_status = np.full(len(geometry["coordinates"]), "not_tested_projected_exclusion", dtype=object)
        last_log = [time.monotonic()]
        def lp_progress(done, total):
            if done == total or time.monotonic()-last_log[0] >= 10:
                log.info("Full-dimensional hull LP progress: %d/%d", done, total)
                last_log[0] = time.monotonic()
        log.info("LP: scipy.optimize.linprog(method='highs'); points=%d; epsilon=%.17g; feasibility tolerance=%.17g; time limit=%.17g s; threads=%d.",
                 len(selected), epsilon, args.lp_feasibility_tol, args.lp_time_limit, args.threads)
        full_status[projected_ids] = check_full_hull(cloud[selected], geometry["coordinates"][projected_ids],
                    epsilon, args.lp_feasibility_tol, args.lp_time_limit, args.threads, lp_progress)
        retained = geometry["inside_projected"].copy()
        if args.hull_policy == "full":
            retained &= full_status == "inside"
        meta["lp"] = {"num_exported_sequences": len(cloud), "num_points_used": len(selected),
                      "subsampled": limit is not None and limit < len(cloud), "seed": args.lp_seed,
                      "exact_duplicates_removed": True, "indices_file": "lp_indices.npy",
                      "epsilon": epsilon, "feasibility_tolerance": args.lp_feasibility_tol,
                      "time_limit_seconds": args.lp_time_limit, "threads": args.threads,
                      "membership": "|Z_lp.T*w-query| <= epsilon componentwise; sum(w)=1, w>=0",
                      "subsampling_note": "Outside selected-cloud hull does not prove outside full-pool hull."}
        np.savez_compressed(directory/"grid.npz", coordinates=geometry["coordinates"],
             grid_k=geometry["grid_k"], grid_l=geometry["grid_l"], inside_projected=geometry["inside_projected"],
             full_hull_status=full_status.astype(str), retained=retained, hull_equations=geometry["hull_equations"])
        del cloud
        log.info("Hull policy=%s; projected centers=%d; retained/evaluated centers=%d.", args.hull_policy, len(projected_ids), int(retained.sum()))
        ensemble = Ensemble(saved.run, saved.dimension, device=device, basis="canonical")
        checkpoint_dtypes = [str(next(m.parameters()).dtype) for m in ensemble.models]
        if args.evaluation_dtype == "float64":
            for model in ensemble.models:
                model.to(dtype=torch.float64)
        meta["evaluation"] = {"requested_dtype": args.evaluation_dtype, "checkpoint_dtypes": checkpoint_dtypes,
                              "evaluation_dtypes": [str(next(m.parameters()).dtype) for m in ensemble.models],
                              "num_models": ensemble.num_models, "method": "existing thermodynamics_from_latent API; autodifferentiated gradient; no Hessian"}
        write_json(directory/"metadata.json", meta)
        log.info("Evaluation: %s", json.dumps(meta["evaluation"], sort_keys=True))
        ids = np.flatnonzero(retained)
        def progress(done, total, batch):
            if done == total or batch % args.progress_every == 0 or time.monotonic()-last_log[0] >= 10:
                log.info("Thermodynamic evaluation: %d/%d points; batch=%d", done, total, batch)
                last_log[0] = time.monotonic()
        fields = evaluate_fields(ensemble, geometry["coordinates"][ids], args.c_tot, args.batch_size, progress)
        means = {name: value.mean(axis=0) for name,value in fields.items()}
        stds = {name: value.std(axis=0, ddof=0) for name,value in fields.items()}
        coordinate_names = [f"z_{j+1}_{'varied' if j in axes else 'fixed'}" for j in range(saved.dimension)]
        value_names = ["f_ex_mean", "f_ex_std"] + [f"{kind}_{axis+1}_{stat}" for kind in ("grad_x", "grad_zbar") for axis in axes for stat in ("mean", "std")]
        summary = {name: np.full(len(retained), np.nan) for name in value_names}
        summary["f_ex_mean"][ids], summary["f_ex_std"][ids] = means["f_ex"], stds["f_ex"]
        for kind in ("grad_x", "grad_zbar"):
            for axis in axes:
                summary[f"{kind}_{axis+1}_mean"][ids] = means[kind][:,axis]
                summary[f"{kind}_{axis+1}_std"][ids] = stds[kind][:,axis]
        rows = []
        for i, coordinate in enumerate(geometry["coordinates"]):
            rows.append({"point_id": i, "grid_k": int(geometry["grid_k"][i]), "grid_l": int(geometry["grid_l"][i]),
                **{name: float(v) for name,v in zip(coordinate_names, coordinate)}, "c_tot": args.c_tot,
                "inside_projected_hull": int(geometry["inside_projected"][i]), "full_hull_status": full_status[i],
                "retained": int(retained[i]), "evaluated": int(retained[i]),
                "evaluation_status": "evaluated" if retained[i] else "excluded_projected_hull" if not geometry["inside_projected"][i] else "excluded_full_hull",
                **{name: float(v[i]) if retained[i] else None for name,v in summary.items()}})
        tsv(directory/"lattice.tsv", rows)
        if args.save_member_values == "yes":
            member_names = ["f_ex"] + [f"{kind}_{axis+1}" for kind in ("grad_x", "grad_zbar") for axis in axes]
            with (directory/"member_fields.tsv").open("w", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(["point_id", "model_index", "checkpoint", "fold", "member", "c_tot", *member_names])
                for m,item in enumerate(saved.selection["models"]):
                    for j,i in enumerate(ids):
                        values = [fields["f_ex"][m,j]] + [fields[kind][m,j,axis] for kind in ("grad_x", "grad_zbar") for axis in axes]
                        writer.writerow([int(i), m, item["checkpoint"], item["fold"], item["member"], format(args.c_tot,".17g"), *[format(float(v),".17g") for v in values]])
        meta["plotting"] = render_maps(directory, geometry, axes, retained, summary, meta, args.arrow_stride, scale)
        meta["counts"] = {"bbox_grid_points": len(retained), "projected_hull_points": len(projected_ids),
                          "retained_points": len(ids), "evaluated_points": len(ids), "num_models": ensemble.num_models,
                          "evaluation_batches": (len(ids)+args.batch_size-1)//args.batch_size}
        inputs = {saved.best/"selection.json": saved.config["selection_sha256"],
                  saved.canonical/"config.json": meta["source"]["canonical_config_sha256"],
                  **{saved.canonical/name: digest for name,digest in saved.config["artifact_sha256"].items()},
                  Path(feature_source["feature_tsv"]): feature_source["feature_tsv_sha256"],
                  Path(feature_source["metadata_path"]): feature_source["metadata_sha256"],
                  **{saved.best/r["checkpoint"]: r["checkpoint_sha256"] for r in saved.selection["models"]}}
        for path,digest in inputs.items():
            if sha256(path) != digest:
                raise ValueError(f"Input changed during evaluation: {path}")
        flush_logs(log)
        meta["files_sha256"] = {p.name: sha256(p) for p in directory.iterdir() if p.is_file() and p.name not in ("metadata.json", "analysis.log")}
        meta["status"] = "complete"
        write_json(directory/"metadata.json", meta)
        log.info("Completed d=%d fixed-c_tot 2D slice: %s", saved.dimension, json.dumps(meta["counts"], sort_keys=True))
        return meta
    except Exception:
        meta["status"] = "failed"
        write_json(directory/"metadata.json", meta)
        log.exception("FREE-ENERGY SLICE FAILED; incomplete values are not valid output.")
        raise
    finally:
        flush_logs(log)


def main():
    args = build_parser().parse_args()
    dimensions, axes, offsets, limit, scale = validate(args)
    device = runtime(args)
    saved = [load_dimension(args.run_dir, d, features=False) for d in dimensions]
    output = prepare_output(args.output_dir, args.overwrite, [args.run_dir, args.features_dir, *[s.run for s in saved]])
    log = start_log(output)
    meta = {"status": "in_progress", "scope": "two-dimensional fixed-concentration affine slices only",
            "arguments": vars(args), "legacy_reference": LEGACY_REFERENCE, "provenance": provenance(args), "dimensions": {}}
    meta["provenance"]["analysis_source_sha256"].update({name: sha256(Path(__file__).parent/name)
        for name in ("calc_free_energy_slice_v1.py", "_free_energy_slice_plots_v1.py")})
    write_json(output/"metadata.json", meta)
    log.info("All CLI values: %s", json.dumps(vars(args), sort_keys=True))
    log.info("Reference implementation: %s", json.dumps(LEGACY_REFERENCE, sort_keys=True))
    try:
        for item in saved:
            record = compute_dimension(args, item, output, device, axes, offsets, limit, scale, log)
            meta["dimensions"][f"d_{item.dimension:03d}"] = {"status": record["status"], "counts": record["counts"],
                "fixed_coordinates": record["fixed_coordinates"], "metadata": f"d_{item.dimension:03d}/metadata.json"}
            write_json(output/"metadata.json", meta)
        meta["status"] = "complete"
        write_json(output/"metadata.json", meta)
        log.info("All fixed-concentration 2D slices complete: %s", output)
    except Exception:
        meta["status"] = "failed"
        write_json(output/"metadata.json", meta)
        log.exception("Free-energy slice workflow failed.")
        raise
    finally:
        flush_logs(log)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FloatingPointError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}")
