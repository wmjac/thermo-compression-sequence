"""Mixture-prior metric estimation and canonicalization of selected ensembles."""
from pathlib import Path
import json
import shutil
import numpy as np
import torch
from .alignment import align_ensemble
from .checkpoints import load_model
from .data import load_json_files, load_run_records, sha256_file
from .provenance import collect_provenance, read_json, resolve_device, timestamp, write_json
from .selection import load_selection


def stick_breaking_weights(total, eps_mass, max_components, rng):
    remaining, weights = 1.0, []
    for _ in range(max_components):
        if remaining < eps_mass:
            break
        fraction = float(rng.beta(1.0, total))
        weight = fraction * remaining
        weights.append(weight)
        remaining -= weight
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    return weights


def sample_composition(n, total, eps_mass, max_components, rng):
    weights = stick_breaking_weights(total, eps_mass, max_components, rng)
    if len(weights) > n:
        weights = weights[:n]
        weights /= weights.sum()
    indices = rng.choice(n, size=len(weights), replace=False)
    return indices, weights


def sequence_pool(run, source):
    if source == "training":
        records, manifests = load_run_records(run)
        sequences = sorted({sequence for record in records for sequence in record.sequences})
        metadata = {"source": "training", "input_sha256": [item["sha256"] for item in manifests],
                    "ordering": "lexicographically sorted unique training sequences"}
    else:
        path = Path(source).expanduser().resolve()
        metadata = {"source": str(path), "sha256": sha256_file(path), "ordering": "file order"}
        if path.suffix == ".npy":
            values = np.load(path, allow_pickle=False)
            if values.ndim != 1 or values.dtype.kind not in ("U", "S"):
                raise ValueError("Sequence .npy must be a one-dimensional string array.")
            sequences = values.astype(str).tolist()
        elif path.suffix == ".json":
            with path.open() as handle:
                raw = json.load(handle)
            if isinstance(raw, dict) and "systems" in raw:
                records, _ = load_json_files([path])
                sequences = sorted({s for record in records for s in record.sequences})
                metadata["ordering"] = "lexicographically sorted unique dataset sequences"
            else:
                sequences = raw.get("sequences") if isinstance(raw, dict) else raw
        else:
            sequences = [line.strip() for line in path.read_text().splitlines()
                         if line.strip() and not line.lstrip().startswith("#")]
    if not isinstance(sequences, list) or not sequences or any(not isinstance(s, str) for s in sequences):
        raise ValueError("Sequence pool must contain a nonempty list of sequence strings.")
    if len(set(sequences)) != len(sequences):
        raise ValueError("The uniform sequence pool must contain distinct sequences.")
    metadata["num_sequences"] = len(sequences)
    metadata["sequence_weighting"] = "uniform over distinct sequences"
    return sequences, metadata


def encode_pool(models, sequences, batch_size):
    encoded = []
    for model in models:
        chunks = []
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                chunks.append(model.encode_sequences(sequences[start:start + batch_size]).cpu().numpy().astype(np.float64))
        encoded.append(np.concatenate(chunks))
    return np.asarray(encoded)


def estimate_metric(models, native, args):
    rng = np.random.default_rng(args.seed)
    initial_rng = rng.bit_generator.state
    k, n, dimension = native.shape
    batches, batch_sizes, concentration_samples, effective_counts, active_counts, top_weights = [], [], [], [], [], []
    offsets, all_indices, all_weights = [0], [], []
    running_frobenius, running_eigenvalues = [], []
    total_outer = np.zeros((k, dimension, dimension), dtype=np.float64)
    done = 0
    while done < args.num_samples:
        count = min(args.batch_size, args.num_samples - done)
        mixture_x = np.empty((k, count, dimension), dtype=np.float64)
        for sample in range(count):
            ctot = float(rng.random() * (args.ctot_max - args.ctot_min) + args.ctot_min)
            indices, weights = sample_composition(n, args.dirichlet_total, args.eps_mass, args.max_components, rng)
            mixture_x[:, sample] = ctot * np.einsum("kid,i->kd", native[:, indices], weights)
            concentration_samples.append(ctot)
            active_counts.append(len(weights))
            effective_counts.append(int((weights > args.eps_effective).sum()))
            ordered = np.sort(weights)[::-1]
            top_weights.append([float(ordered[0]), float(ordered[1]) if len(ordered) > 1 else 0.0])
            all_indices.extend(indices.tolist())
            all_weights.extend(weights.tolist())
            offsets.append(len(all_weights))
        outer = np.empty((k, dimension, dimension), dtype=np.float64)
        for member, model in enumerate(models):
            parameters = next(model.parameters())
            x = torch.as_tensor(mixture_x[member], dtype=parameters.dtype, device=parameters.device)
            with torch.enable_grad():
                gradients = model.thermodynamics(x)["gradient"].detach().cpu().numpy().astype(np.float64)
            outer[member] = gradients.T @ gradients
        batches.append(outer / count)
        batch_sizes.append(count)
        total_outer += outer
        done += count
        running = np.mean(batches, axis=0) if args.metric_aggregation == "batch-mean" else total_outer / done
        running_frobenius.append(np.linalg.norm(running, axis=(1, 2)))
        running_eigenvalues.append(np.linalg.eigvalsh(running))
        print(f"Canonical metric: {done}/{args.num_samples} mixtures", flush=True)
    metric_batches = np.asarray(batches)
    metric = metric_batches.mean(axis=0) if args.metric_aggregation == "batch-mean" else total_outer / done
    metric = 0.5 * (metric + metric.transpose(0, 2, 1))
    batch_frobenius = np.linalg.norm(metric_batches, axis=(2, 3))
    se = batch_frobenius.std(axis=0, ddof=1) / np.sqrt(len(batches)) if len(batches) > 1 else None
    statistics = {"num_samples": done, "active_components_mean": float(np.mean(active_counts)),
                  "effective_components_mean": float(np.mean(effective_counts)),
                  "effective_components_std": float(np.std(effective_counts, ddof=1)) if done > 1 else 0.0,
                  "effective_components_histogram": {str(value): int(effective_counts.count(value)) for value in sorted(set(effective_counts))},
                  "ctot_mean": float(np.mean(concentration_samples)), "ctot_min_observed": float(np.min(concentration_samples)),
                  "ctot_max_observed": float(np.max(concentration_samples)),
                  "mean_largest_weight": float(np.mean(np.array(top_weights)[:, 0])),
                  "mean_second_weight": float(np.mean(np.array(top_weights)[:, 1])),
                  "batch_frobenius_standard_error": None if se is None else se.tolist(),
                  "rng_initial": initial_rng, "rng_final": rng.bit_generator.state}
    samples = {"ctot": np.array(concentration_samples), "offsets": np.array(offsets, dtype=np.int64),
               "indices": np.array(all_indices, dtype=np.int64), "weights": np.array(all_weights),
               "active_components": np.array(active_counts), "effective_components": np.array(effective_counts)}
    convergence = {"batch_metrics": metric_batches, "batch_sizes": np.asarray(batch_sizes),
                   "running_frobenius": np.asarray(running_frobenius), "running_eigenvalues": np.asarray(running_eigenvalues)}
    return metric, statistics, samples, convergence


def metric_square_roots(metric, tolerance):
    roots, eigenvalues = [], []
    for member, matrix in enumerate(metric):
        values, vectors = np.linalg.eigh(matrix)
        maximum = float(values.max())
        if maximum <= 0 or values.min() <= tolerance * maximum:
            raise ValueError(f"Metric for member {member} is singular or ill-conditioned: eigenvalues={values}. "
                             "No dimension reduction or regularization was applied; inspect metric.npz.")
        root = (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T
        roots.append(0.5 * (root + root.T))
        eigenvalues.append(values)
    return np.asarray(roots), np.asarray(eigenvalues)


def run_canonicalization(args):
    run = Path(args.run_dir).expanduser().resolve()
    device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    sequences, pool_metadata = sequence_pool(run, args.sequence_pool)
    for dimension in args.dimensions:
        best, selection = load_selection(run, dimension)
        output = run / "canonical" / f"d_{dimension:03d}"
        if output.exists():
            if args.overwrite != "yes":
                raise ValueError(f"Canonical output exists: {output}; use --overwrite yes to replace it.")
            shutil.rmtree(output)
        output.mkdir(parents=True)
        models, modules = [], []
        for item in selection["models"]:
            model, module, _ = load_model(best / item["checkpoint"], device)
            module.validate_sequences(sequences)
            models.append(model)
            modules.append(module)
        config = {"created_at": timestamp(), "arguments": vars(args), "sequence_pool": pool_metadata,
                  "selection_sha256": sha256_file(best / "selection.json"),
                  "selected_checkpoints": [item["checkpoint_sha256"] for item in selection["models"]],
                  "dimension": dimension, "num_models": len(models),
                  "rho_mix": {"concentration_distribution": "uniform", "concentration_units": "nm^-3",
                      "ctot_min": args.ctot_min, "ctot_max": args.ctot_max, "composition_distribution": "truncated_GEM",
                      "approximates": "symmetric Dirichlet(A/n) at large n", "dirichlet_total": args.dirichlet_total,
                      "eps_mass": args.eps_mass, "max_components": args.max_components, "eps_effective": args.eps_effective,
                      "sequence_weighting": args.sequence_weighting},
                  "coordinate_convention": "z_canonical = z_native @ A.T"}
        write_json(output / "config.json", config)
        write_json(output / "provenance.json", collect_provenance(device))
        native = encode_pool(models, sequences, args.encoding_batch_size)
        metric, statistics, samples, convergence = estimate_metric(models, native, args)
        np.savez_compressed(output / "samples.npz", **samples)
        np.savez_compressed(output / "metric.npz", metric=metric, **convergence)
        write_json(output / "diagnostics.json", {"metric": statistics, "status": "metric_estimated"})
        roots, values = metric_square_roots(metric, args.metric_eigenvalue_tolerance)
        aligned = align_ensemble(native, roots, args.align_max_iter, args.align_tolerance)
        inverses = np.linalg.inv(aligned["transforms"])
        np.savez_compressed(output / "transforms.npz", metric=metric, square_roots=roots,
                            transforms=aligned["transforms"], inverses=inverses,
                            pca_rotations=aligned["pca_rotations"], procrustes_rotations=aligned["procrustes_rotations"],
                            final_rotation=aligned["final_rotation"])
        np.savez_compressed(output / "features.npz", sequences=np.asarray(sequences), native=native,
                            aligned=aligned["aligned"], mean=aligned["mean"])
        indices = np.sort(np.random.default_rng(args.seed).choice(len(sequences), size=min(args.diagnostic_points, len(sequences)), replace=False))
        np.savez(output / "alignment_samples.npz", indices=indices, sequences=np.asarray(sequences)[indices],
                 before=aligned["pca_features"][:, indices], after=aligned["aligned"][:, indices])
        rmsd = float(np.sqrt(np.mean(np.sum((aligned["aligned"] - aligned["mean"]) ** 2, axis=-1))))
        diagnostics = {"status": "complete", "metric": statistics, "metric_eigenvalues": values,
                       "alignment": {"iterations": aligned["iterations"], "converged": aligned["converged"],
                           "rotation_change": aligned["convergence"], "ensemble_rmsd": rmsd,
                           "per_model_pca_eigenvalues": aligned["pca_eigenvalues"], "per_model_signs": aligned["pca_signs"],
                           "mean_pca_eigenvalues": aligned["final_eigenvalues"], "mean_signs": aligned["final_signs"]}}
        write_json(output / "diagnostics.json", diagnostics)
        config["artifact_sha256"] = {name: sha256_file(output / name)
            for name in ("samples.npz", "metric.npz", "transforms.npz", "features.npz", "alignment_samples.npz")}
        write_json(output / "config.json", config)
        from .plotting import plot_canonical_diagnostics
        plot_canonical_diagnostics(convergence, aligned, output)
        print(f"Saved canonical basis for d={dimension}: {output}", flush=True)
