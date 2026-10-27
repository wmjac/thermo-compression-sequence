#!/usr/bin/env python3
"""Export aligned ensemble feature means, RMSD, and dataset membership to TSV."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
from _post_training_common_v1 import (dimensions_argument, load_dimension, models_for, prepare_output,
    provenance, read_sequences, runtime, sha256, write_json)


def membership_key(sequence, symmetry):
    return min(sequence, sequence[::-1]) if symmetry == "reversal" else sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Canonical run or combined analysis01 directory.")
    parser.add_argument("--dimensions", nargs="+", required=True)
    parser.add_argument("--sequence-list", type=Path, required=True)
    parser.add_argument("--test-json", nargs="*", required=True,
                        help="Test datasets; explicitly empty means no test dataset exists.")
    parser.add_argument("--membership-symmetry", choices=("exact", "reversal"), default="exact")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--save-member-features", choices=("yes", "no"), default="no")
    parser.add_argument("--overwrite", choices=("yes", "no"), default="no")
    args = parser.parse_args()
    device = runtime(args)
    dimensions = dimensions_argument(args.dimensions)
    sequences = read_sequences(args.sequence_list)
    from mixture_models.data import load_json_files, load_run_records
    test_records, test_inputs = load_json_files(args.test_json) if args.test_json else ([], [])
    key = lambda s: membership_key(s, args.membership_symmetry)
    test_sequences = {key(s) for record in test_records for s in record.sequences}
    test_flags = np.asarray([key(s) in test_sequences for s in sequences], dtype=bool)
    del test_records
    saved_dimensions = [load_dimension(args.run_dir, d, features=False) for d in dimensions]
    output = prepare_output(args.output_dir, args.overwrite,
                            [args.run_dir, *[saved.run for saved in saved_dimensions]])
    manifest = {"status": "in_progress", "provenance": provenance(args),
        "sequence_list": {"path": str(args.sequence_list.expanduser().resolve()),
                          "sha256": sha256(args.sequence_list), "num_sequences": len(sequences), "ordering": "input order"},
        "test_inputs": test_inputs, "test_set_defined": bool(args.test_json),
        "membership": {"symmetry": args.membership_symmetry,
                       "training": "appears in any system in the archived pooled training JSONs, including validation folds",
                       "test": "appears in any system in the supplied test JSONs"},
        "feature_mean": "arithmetic mean of selected models after their saved canonical transformations",
        "feature_rmsd": "sqrt(mean_m(sum_j((z_mj - mean_m(z_mj))**2))); population spread; no division by sqrt(k) or d",
        "precision": "checkpoint encoder dtype; transforms/aggregation float64; TSV .17g; no artificial precision added to inference",
        "dimensions": []}
    write_json(output / "metadata.json", manifest)
    import torch
    for saved in saved_dimensions:
        training_records, training_inputs = load_run_records(saved.run)
        trained = {key(s) for record in training_records for s in record.sequences}
        train_flags = np.asarray([key(s) in trained for s in sequences], dtype=bool)
        del training_records
        # One dimension is resident at a time; full sequence-by-member arrays are
        # optional disk-backed files, never required for the TSV export.
        loaded = list(models_for(saved, device))
        for _, module, _ in loaded:
            module.validate_sequences(sequences)
        path = output / f"features_d_{saved.dimension:03d}.tsv"
        temporary = path.with_name(path.name + ".tmp")
        member_path = output / f"member_features_d_{saved.dimension:03d}.npy"
        member_features = None
        if args.save_member_features == "yes":
            member_features = np.lib.format.open_memmap(member_path, mode="w+", dtype=np.float64,
                shape=(len(loaded), len(sequences), saved.dimension))
        with temporary.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["sequence", "in_training", "in_test",
                             *[f"z_{j + 1}" for j in range(saved.dimension)], "feature_rmsd"])
            for start in range(0, len(sequences), args.batch_size):
                stop = min(start + args.batch_size, len(sequences))
                chunk = sequences[start:stop]
                features = []
                for member, (model, _, _) in enumerate(loaded):
                    with torch.no_grad():
                        native = model.encode_sequences(chunk).detach().cpu().numpy().astype(np.float64)
                    aligned = native @ saved.transforms[member].T
                    features.append(aligned)
                    if member_features is not None:
                        member_features[member, start:stop] = aligned
                features = np.asarray(features)
                if not np.all(np.isfinite(features)):
                    raise FloatingPointError("Nonfinite encoded features; export remains incomplete.")
                mean = features.mean(axis=0)
                rmsd = np.sqrt(np.mean(np.sum((features - mean[None])**2, axis=-1), axis=0))
                for offset, sequence in enumerate(chunk):
                    index = start + offset
                    writer.writerow([sequence, int(train_flags[index]), int(test_flags[index]),
                        *[format(value, ".17g") for value in mean[offset]], format(rmsd[offset], ".17g")])
                if start == 0 or stop == len(sequences) or start // 100000 != stop // 100000:
                    print(f"d={saved.dimension}: exported {stop}/{len(sequences)} sequences", flush=True)
        if member_features is not None:
            member_features.flush()
            del member_features
        temporary.replace(path)
        record = {**saved.source(), "training_inputs": training_inputs, "tsv": path.name,
            "tsv_sha256": sha256(path), "num_training_members": int(train_flags.sum()),
            "num_test_members": int(test_flags.sum()), "num_models": len(loaded),
            "member_features": member_path.name if args.save_member_features == "yes" else None,
            "member_features_sha256": sha256(member_path) if args.save_member_features == "yes" else None}
        manifest["dimensions"].append(record)
        write_json(output / "metadata.json", manifest)
        del loaded
        if device.type == "cuda":
            torch.cuda.empty_cache()
    manifest["status"] = "complete"
    write_json(output / "metadata.json", manifest)
    print(f"Feature export complete: {output}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FloatingPointError) as exc:
        raise SystemExit(f"Error: {exc}")
