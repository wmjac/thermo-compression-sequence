#!/usr/bin/env python3
"""Evaluate a fixed-composition concentration sweep using selected checkpoints.

Run directly from an unpacked thermo-compression-sequence checkout; no install
or other analysis overlay is required. See docs/composition_sweep_v1.md.
"""
from pathlib import Path
import argparse
import csv
import math
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

C_REFERENCE_NM_INV3 = 1.0
QUANTITIES = ("f_ex", "P_ex", "f", "P")


def parse_components(components):
    sequences = [item[0] for item in components]
    if len(set(sequences)) != len(sequences):
        raise ValueError("Repeated sequences: combine their mole fractions explicitly.")
    try:
        fractions = np.array([float(item[1]) for item in components], dtype=np.float64)
    except ValueError as error:
        raise ValueError("Every component needs a numeric mole fraction.") from error
    if not np.isfinite(fractions).all() or (fractions < 0).any():
        raise ValueError("Mole fractions must be finite and nonnegative.")
    if not math.isclose(math.fsum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Mole fractions must sum to one (absolute tolerance 1e-12); no renormalization is performed.")
    return sequences, fractions


def concentration_grid(c_min, c_max, num_points, spacing):
    if not np.isfinite([c_min, c_max]).all() or not 0 <= c_min < c_max:
        raise ValueError("Require finite 0 <= c-min < c-max, in nm^-3.")
    if num_points < 2:
        raise ValueError("num-points must be at least two.")
    if spacing == "log" and c_min == 0:
        raise ValueError("Logarithmic spacing requires c-min > 0.")
    return (np.geomspace if spacing == "log" else np.linspace)(c_min, c_max, num_points)


def ideal_free_energy(concentration, fractions):
    """beta f_id = sum_i c_i [ln(c_i/c_reference)-1], including its zero limit."""
    species_c = concentration[:, None] * fractions[None, :]
    logarithm = np.zeros_like(species_c)
    np.log(species_c / C_REFERENCE_NM_INV3, out=logarithm, where=species_c > 0)
    return np.sum(species_c * (logarithm - 1.0), axis=1)


def evaluate_sweep(ensemble, sequences, fractions, concentration, batch_size, evaluation_dtype):
    import torch
    # Preserve the encoder precision used for checkpoint inference and feature
    # export. Promote only after encoding, for subsequent decoder evaluation.
    checkpoint_dtypes = [str(next(m.parameters()).dtype) for m in ensemble.models]
    features = ensemble.encode_sequences(sequences, batch_size)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite sequence features.")
    rays = np.einsum("knd,n->kd", features, fractions)
    if evaluation_dtype == "float64":
        for model in ensemble.models:
            model.to(dtype=torch.float64)
    evaluation_dtypes = [str(next(m.parameters()).dtype) for m in ensemble.models]
    values = {name: [] for name in ("f_ex", "P_ex")}
    for start in range(0, len(concentration), batch_size):
        c = concentration[start:start + batch_size]
        # Each member uses its OWN composition-weighted features and decoder.
        x = rays[:, None, :] * c[None, :, None]
        result = ensemble.thermodynamics_from_latent(x)
        values["f_ex"].append(result["free_energy_excess"])
        values["P_ex"].append(result["pressure_excess"])
    values = {name: np.concatenate(parts, axis=1) for name, parts in values.items()}
    f_ideal = ideal_free_energy(concentration, fractions)
    values["f"] = values["f_ex"] + f_ideal[None, :]
    values["P"] = values["P_ex"] + concentration[None, :]
    if any(not np.isfinite(array).all() for array in values.values()):
        raise ValueError("Nonfinite thermodynamic predictions; no sweep files were written.")
    return values, f_ideal, {
        "encoder_dtypes": checkpoint_dtypes, "decoder_dtypes": evaluation_dtypes,
        "sequence_features_by_model": features.tolist(), "mixture_features_by_model": rays.tolist(),
    }


def write_tsv(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        for row in rows:
            writer.writerow([format(float(v), ".17g") if isinstance(v, (float, np.floating)) else v for v in row])


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="Saved run, or analysis01 directory containing run_views/.")
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--component", nargs=2, action="append", required=True,
                        metavar=("SEQUENCE", "MOLE_FRACTION"), help="Repeat once per distinct sequence.")
    parser.add_argument("--c-min", type=float, default=1e-8, help="Minimum total chain concentration, nm^-3.")
    parser.add_argument("--c-max", type=float, default=0.25, help="Maximum total chain concentration, nm^-3.")
    parser.add_argument("--num-points", type=int, default=1001)
    parser.add_argument("--spacing", choices=("linear", "log"), default="log")
    parser.add_argument("--output", required=True, help="Summary TSV; companions use .members.tsv and .metadata.json.")
    parser.add_argument("--basis", choices=("canonical", "native"), default="canonical")
    parser.add_argument("--evaluation-dtype", choices=("float64", "checkpoint"), default="float64",
                        help="Decoder precision; sequences are always encoded at checkpoint precision.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, or auto.")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", help="Replace these three output files if present.")
    return parser


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    try:
        if min(args.dimension, args.batch_size, args.threads) <= 0:
            raise ValueError("dimension, batch-size, and threads must be positive.")
        sequences, fractions = parse_components(args.component)
        concentration = concentration_grid(args.c_min, args.c_max, args.num_points, args.spacing)
        output = Path(args.output).expanduser().absolute()
        if output.suffix != ".tsv":
            raise ValueError("output must end in .tsv.")
        members_path = output.with_suffix(".members.tsv")
        metadata_path = output.with_suffix(".metadata.json")
        outputs = (output, members_path, metadata_path)
        for path in outputs:
            if path.is_symlink() or path.exists() and not path.is_file():
                raise ValueError(f"Output must be an ordinary file: {path}")
            if path.exists() and not args.overwrite:
                raise ValueError(f"Output exists: {path}; choose a new name or use --overwrite.")
        run_root = Path(args.run_dir).expanduser().resolve()
        view = run_root / "run_views" / f"d_{args.dimension:03d}"
        run = view if (run_root / "run_views").is_dir() else run_root
        # Never permit --overwrite to replace selected inputs or source files.
        for path in outputs:
            if path.resolve().is_relative_to((ROOT / "src").resolve()) or any(
                path.resolve().is_relative_to((run / name).resolve()) for name in ("best", "canonical", "inputs", "trials")
            ):
                raise ValueError(f"Output falls inside protected source/model inputs: {path}")

        import torch
        from mixture_models.data import sha256_file
        from mixture_models.evaluation import Ensemble
        from mixture_models.provenance import collect_provenance, read_json, resolve_device, timestamp, write_json

        torch.set_num_threads(args.threads)
        device = resolve_device(args.device)
        ensemble = Ensemble(run, args.dimension, device=device, basis=args.basis)
        if not ensemble.models or ensemble.selection["dimension"] != args.dimension:
            raise ValueError("Selection is empty or has the wrong dimension.")
        inputs = {ensemble.best / "selection.json": sha256_file(ensemble.best / "selection.json")}
        for item in ensemble.selection["models"]:
            path = ensemble.best / item["checkpoint"]
            inputs[path] = item["checkpoint_sha256"]
        canonical_config = None
        if args.basis == "canonical":
            directory = run / "canonical" / f"d_{args.dimension:03d}"
            for name in ("config.json", "diagnostics.json", "transforms.npz"):
                inputs[directory / name] = sha256_file(directory / name)
            canonical_config = read_json(directory / "config.json")

        values, f_ideal, evaluation = evaluate_sweep(ensemble, sequences, fractions, concentration,
                                                   args.batch_size, args.evaluation_dtype)
        for path, digest in inputs.items():
            if sha256_file(path) != digest:
                raise ValueError(f"Selected input changed during evaluation: {path}")
        summary_fields = ["c_tot_nm^-3", "num_models", "f_ideal", "P_ideal"]
        summary_fields += [column for q in QUANTITIES for column in (q + "_mean", q + "_std")]
        means = {q: values[q].mean(axis=0) for q in QUANTITIES}
        stds = {q: values[q].std(axis=0, ddof=0) for q in QUANTITIES}
        summary_rows = ([c, ensemble.num_models, f_ideal[j], c,
                         *[v for q in QUANTITIES for v in (means[q][j], stds[q][j])]]
                        for j, c in enumerate(concentration))
        member_fields = ["model_index", "checkpoint", "fold", "member", "c_tot_nm^-3", *QUANTITIES]
        member_rows = ([m, item["checkpoint"], item.get("fold", ""), item.get("member", ""), c,
                        *[values[q][m, j] for q in QUANTITIES]]
                       for m, item in enumerate(ensemble.selection["models"]) for j, c in enumerate(concentration))
        metadata = {
            "status": "complete", "created_at": timestamp(), "arguments": vars(args),
            "command": shlex.join([sys.executable, str(Path(__file__).resolve()),
                                   *(sys.argv[1:] if argv is None else argv)]),
            "working_directory": str(Path.cwd()), "resolved_run_dir": str(run.resolve()),
            "num_models": ensemble.num_models,
            "components": [{"sequence": seq, "mole_fraction": float(x)} for seq, x in zip(sequences, fractions)],
            "mole_fraction_sum": float(math.fsum(fractions)), "mole_fractions_renormalized": False,
            "units": {"c_tot": "nm^-3 (chain number density)",
                      "f_ex": "kT nm^-3", "P_ex": "kT nm^-3", "f": "kT nm^-3", "P": "kT nm^-3"},
            "ideal_reference_concentration_nm^-3": C_REFERENCE_NM_INV3,
            "definitions": {"f_ideal": "sum_i c_i*(ln(c_i/c_reference)-1); c_i=x_i*c_tot; 0*ln(0)=0",
                            "P_ideal": "c_tot", "f": "f_ex + f_ideal", "P": "P_ex + c_tot",
                            "P_ex": "x_latent dot grad(f_ex) - f_ex, from existing model API"},
            "aggregation": "arithmetic mean; population standard deviation across all selected models (ddof=0), not SEM",
            "mixture_evaluation": "each model's own encoded sequences, mixture features, and decoder; no feature averaging across models",
            "free_energy_reference": "checkpoint excess-free-energy convention; ideal reference sets the linear-in-concentration term",
            "evaluation": evaluation, "canonical_config": canonical_config,
            "selected_models": ensemble.selection["models"],
            "inputs_sha256": {str(path.resolve()): digest for path, digest in inputs.items()},
            "script_sha256": sha256_file(Path(__file__)), "provenance": collect_provenance(device),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        # Completion metadata is written last; remove its old completion marker
        # before replacing tables when --overwrite is explicitly requested.
        if metadata_path.exists():
            metadata_path.unlink()
        write_tsv(output, summary_fields, summary_rows)
        write_tsv(members_path, member_fields, member_rows)
        metadata["outputs_sha256"] = {p.name: sha256_file(p) for p in (output, members_path)}
        write_json(metadata_path, metadata)
        print(f"Wrote {len(concentration)} concentrations for {ensemble.num_models} selected models:")
        print(output)
        print(members_path)
        print(metadata_path)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
