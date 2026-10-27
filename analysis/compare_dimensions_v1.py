#!/usr/bin/env python3
"""Compare pressure-trained models with reference EYM modes and ensemble controls."""
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
from _post_training_common_v1 import (check_common_measures, dimensions_argument, load_dimension,
    prepare_output, provenance, runtime, sha256, tsv, write_json)
from _geometry_v1 import (angles_from_overlap, basis, eigenblocks, eigensystem, gram,
    mixing_statistics, nonnegative, procrustes, subspace_comparison, within_spread)


def comparison(x, reference, weights, d, args):
    values, vectors, keep = eigensystem(reference, weights, args.rank_rtol, args.rank_atol)
    coordinates = reference @ vectors[:, keep]
    leading = coordinates[:, :d]
    stats, angles, singular = subspace_comparison(x, leading, weights, args.rank_rtol, args.rank_atol)
    all_stats, _, _ = subspace_comparison(x, coordinates, weights, args.rank_rtol, args.rank_atol)
    before, after = procrustes(x, reference, weights)
    _, after_top = procrustes(x, leading, weights)
    stats.update(full_reference_fraction=all_stats["left_fraction"], rmsd_before_padded=before,
        rmsd_after_padded_procrustes=after, rmsd_to_reference_leading_d=after_top,
        reference_truncation_rmsd=float(np.sqrt(values[d:].sum())))
    return stats, angles, singular


def pair_identifiers(kind, low, low_member, ref, ref_member):
    return {"comparison": kind, "dimension": low.dimension, "member": low_member,
            "reference_dimension": ref.dimension, "reference_member": ref_member}


def append_angles(rows, identity, angles, singular, domain="chemical", target="leading_d"):
    for index, (angle, cosine) in enumerate(zip(angles, singular)):
        rows.append({**identity, "domain": domain, "target": target, "angle_index": index + 1,
                     "angle_deg": float(angle), "cosine": float(cosine)})


def plots(output, summary, spectra, mixing_heatmaps, tm_rows, reference_dimension):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / name, dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for values in spectra:
        ax.plot(np.arange(1, len(values) + 1), np.ma.masked_less_equal(values, 0), "o-", alpha=0.5)
    ax.set_yscale("log")
    ax.set(xlabel="Reference EYM mode", ylabel="TM eigenvalue", title=f"Reference d={reference_dimension}; individual selected models")
    save(fig, "reference_spectrum.png")
    if summary:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        dimensions = [row["dimension"] for row in summary]
        axes[0].errorbar(dimensions, [row["mean_leading_overlap_fraction"] for row in summary],
            yerr=[row["std_leading_overlap_fraction"] for row in summary], marker="o", label="Selected-model pairs")
        for key, label, style in (("within_dimension_overlap", "Within-d model pairs", "d:"),
                                   ("reference_leading_overlap", "Within-reference leading-d pairs", "^-.")):
            valid = [row for row in summary if row.get(key) is not None]
            if valid:
                axes[0].plot([row["dimension"] for row in valid], [row[key] for row in valid], style, label=label)
        if any(row["mean_random_overlap_fraction"] is not None for row in summary):
            axes[0].plot(dimensions, [np.nan if row["mean_random_overlap_fraction"] is None else row["mean_random_overlap_fraction"]
                                     for row in summary], "s--", label="Random reference subspaces")
        axes[0].set(xlabel="Learned dimension d", ylabel="Leading reference subspace retained", ylim=(-0.02, 1.02))
        axes[0].legend(fontsize=7)
        defined_angles = [row for row in summary if row["mean_max_angle_deg"] is not None]
        if defined_angles:
            axes[1].errorbar([row["dimension"] for row in defined_angles],
                [row["mean_max_angle_deg"] for row in defined_angles],
                yerr=[row["std_max_angle_deg"] for row in defined_angles], marker="o")
        axes[1].set(xlabel="Learned dimension d", ylabel="Maximum principal angle (degrees)")
        save(fig, "subspace_comparison.png")
    for d, heat in mixing_heatmaps.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        plot = ax.imshow(heat, origin="lower", aspect="auto", vmin=0, vmax=1)
        ax.set(xlabel="Lower-model EYM mode", ylabel="Reference EYM mode",
               xticks=np.arange(heat.shape[1]), xticklabels=np.arange(1, heat.shape[1] + 1),
               yticks=np.arange(heat.shape[0]), yticklabels=np.arange(1, heat.shape[0] + 1),
               title=f"d={d}: mean squared mode overlaps")
        fig.colorbar(plot, ax=ax, label="Squared overlap")
        save(fig, f"mode_mixing_d_{d:03d}.png")
    cross = [row for row in tm_rows if row["comparison"] == "cross_dimension"]
    if cross:
        dims = sorted({row["dimension"] for row in cross})
        means, tails = [], []
        for d in dims:
            rows = [row for row in cross if row["dimension"] == d]
            means.append(np.mean([row["tm_loss"] for row in rows]))
            tails.append(np.mean([row["eym_tail_bound"] for row in rows]))
        within = [row["tm_loss"] for row in tm_rows if row["comparison"] == "within_dimension" and row["dimension"] == reference_dimension]
        for rmse, logarithmic, name in ((False, False, "tm_loss_vs_dimension.png"),
                (False, True, "tm_loss_vs_dimension_log.png"), (True, True, "tm_rmse_vs_dimension_log.png")):
            fig, ax = plt.subplots(figsize=(6, 4))
            def transform(values):
                values = np.sqrt(values) if rmse else np.asarray(values)
                return np.ma.masked_less_equal(values, 0) if logarithmic else values
            for d in dims:
                values = [row["tm_loss"] for row in cross if row["dimension"] == d]
                ax.scatter(np.full(len(values), d), transform(values), alpha=0.25, s=12, color="C0")
            ax.plot(dims, transform(means), "o-", label="Pressure-trained models vs. reference")
            ax.plot(dims, transform(tails), "s--", label="Reference EYM tail bound")
            if within and (not logarithmic or np.mean(within) > 0):
                value = np.sqrt(np.mean(within)) if rmse else np.mean(within)
                ax.axhline(value, color="gray", linestyle=":", label="Within-reference disagreement")
            if logarithmic:
                ax.set_yscale("log")
                if not any(value > 0 for value in means + tails + within):
                    ax.set_ylim(1, 10)
                if any(value == 0 for value in means + tails + within):
                    ax.text(.02, .98, "Zero values omitted on log scale", transform=ax.transAxes, va="top", fontsize="small")
            ax.set(xlabel="Learned dimension d", ylabel="Dimensionless chemical-potential TM RMSE" if rmse
                   else "Dimensionless chemical-potential TM loss")
            ax.legend(fontsize="small")
            save(fig, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Canonical run or combined analysis01 directory.")
    parser.add_argument("--dimensions", nargs="+", required=True)
    parser.add_argument("--reference-dimension", type=int, required=True,
                        help="Choose explicitly using spectra/stability; no automatic physical noise cutoff.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank-rtol", type=float, default=1e-10)
    parser.add_argument("--rank-atol", type=float, default=0.0)
    parser.add_argument("--degeneracy-rtol", type=float, default=0.01)
    parser.add_argument("--random-controls", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--response-analysis", choices=("yes", "no"), default="yes")
    parser.add_argument("--save-responses", choices=("yes", "no"), default="yes")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--overwrite", choices=("yes", "no"), default="no")
    args = parser.parse_args()
    device = runtime(args)
    for name in ("rank_rtol", "rank_atol", "degeneracy_rtol"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if args.seed < 0 or args.random_controls < 0:
        raise ValueError("Seed and random-control count must be nonnegative.")
    dimensions = sorted(dimensions_argument(args.dimensions))
    if args.reference_dimension not in dimensions or max(dimensions) > args.reference_dimension:
        raise ValueError("Reference dimension must be included and at least as large as all compared dimensions.")
    saved = {d: load_dimension(args.run_dir, d) for d in dimensions}
    reference = saved[args.reference_dimension]
    check_common_measures(list(saved.values()), reference)
    weights = reference.weights
    output = prepare_output(args.output_dir, args.overwrite, [args.run_dir, *[item.run for item in saved.values()]])
    manifest = {"status": "in_progress", "provenance": provenance(args),
        "sources": [item.source() for item in saved.values()], "rho_mix": reference.config["rho_mix"],
        "rho_chem": reference.chemical_measure, "chemical_pool": "chemical_measure.npz",
        "reference_dimension": args.reference_dimension,
        "reference_selection": "explicit CLI choice; no automatic physical-noise dimension removal",
        "chemical_geometry": "uncentered sqrt(rho_chem)-weighted feature-column subspaces; saved canonical TM metric",
        "eym": "per selected reference model; no latent averaging is treated as a physical ensemble model",
        "controls": {"within_dimension": "all unordered pairs of distinct selected models; empirical spread, not independent samples",
            "random": "Haar random rank-d subspaces within each numerically retained reference span; feature spectrum matched to reference leading d",
            "truncation": "reference EYM truncation at each compared dimension",
            "normalization": "sqrt(within_d_rmsd**2 + within_reference_rmsd**2); descriptive scale, not a z score"},
        "centering_translation_rescaling": "none; Procrustes permits rotations and reflections only",
        "rank_tolerance": "numerical identification only; removed directions and all eigenvalues are recorded",
        "uncertainty": "per-member and per-pair distributions; model pairs share data and are not IID confidence samples",
        "loss": "TM loss of pressure-trained lower models relative to the pressure-trained reference; no retraining",
        "pressure_loss_bound": False}
    write_json(output / "metadata.json", manifest)
    np.savez_compressed(output / "chemical_measure.npz", sequences=np.asarray(reference.sequences), weights=weights)
    rng = np.random.default_rng(args.seed)
    initial_rng = rng.bit_generator.state
    spectra, reference_cache, spectrum_rows, truncation_rows = [], [], [], []
    for member, z in enumerate(reference.aligned):
        values, vectors, keep = eigensystem(z, weights, args.rank_rtol, args.rank_atol)
        retained_values = values[keep]
        if not len(retained_values):
            raise ValueError(f"Reference member {member} has no numerically identifiable TM mode; inspect the reference and tolerances.")
        q = (np.sqrt(weights)[:, None] * z) @ (vectors[:, keep] / np.sqrt(retained_values))
        blocks = eigenblocks(retained_values, args.degeneracy_rtol)
        reference_cache.append((values, vectors, keep, q, blocks))
        spectra.append(values)
        for mode, value in enumerate(values):
            block = next((i + 1 for i, group in enumerate(blocks) if mode in np.flatnonzero(keep)[group]), None)
            spectrum_rows.append({"reference_member": member, "mode": mode + 1, "eigenvalue": float(value),
                "relative_eigenvalue": float(value / values[0]) if values[0] else None,
                "numerically_retained": int(keep[mode]), "degenerate_block": block})
        for d in dimensions:
            truncation_rows.append({"reference_member": member, "dimension": d,
                "eym_tail": float(values[d:].sum()), "truncation_rmsd": float(np.sqrt(values[d:].sum()))})
    np.savez_compressed(output / "reference_eigensystems.npz", eigenvalues=np.asarray(spectra),
        eigenvectors=np.asarray([item[1] for item in reference_cache]), retained=np.asarray([item[2] for item in reference_cache]))
    tsv(output / "reference_spectrum.tsv", spectrum_rows)
    tsv(output / "reference_truncation.tsv", truncation_rows)
    spectrum_summary = []
    for mode, values in enumerate(np.asarray(spectra).T):
        spectrum_summary.append({"mode": mode + 1, "mean_eigenvalue": float(values.mean()),
            "ensemble_std": float(values.std()), "minimum": float(values.min()), "maximum": float(values.max())})
    tsv(output / "reference_spectrum_summary.tsv", spectrum_summary)
    pair_rows, within_rows, angle_rows, mixing_rows, random_rows, mean_rows, summaries = [], [], [], [], [], [], []
    reference_leading_rows = []
    for d in dimensions:
        for left, right in combinations(range(len(reference.aligned)), 2):
            vl, kl = reference_cache[left][1:3]
            vr, kr = reference_cache[right][1:3]
            x = reference.aligned[left] @ vl[:, kl][:, :d]
            y = reference.aligned[right] @ vr[:, kr][:, :d]
            identity = {"comparison": "reference_leading_stability", "dimension": d,
                "member": left, "reference_dimension": reference.dimension, "reference_member": right}
            stats, angles, singular = subspace_comparison(x, y, weights, args.rank_rtol, args.rank_atol)
            reference_leading_rows.append({**identity, **stats})
            append_angles(angle_rows, identity, angles, singular, target="reference_leading_d")
    tsv(output / "reference_leading_subspace_controls.tsv", reference_leading_rows)
    heatmaps = {}
    mean_reference, spread_reference = within_spread(reference.aligned, weights)
    for d, low in saved.items():
        mean_low, spread_low = within_spread(low.aligned, weights)
        low_values, low_vectors, low_keeps, low_blocks = [], [], [], []
        for z in low.aligned:
            values, vectors, keep = eigensystem(z, weights, args.rank_rtol, args.rank_atol)
            low_values.append(values); low_vectors.append(vectors); low_keeps.append(keep)
            low_blocks.append(eigenblocks(values[keep], args.degeneracy_rtol))
        np.savez_compressed(output / f"eigensystems_d_{d:03d}.npz", eigenvalues=np.asarray(low_values),
                            eigenvectors=np.asarray(low_vectors), retained=np.asarray(low_keeps))
        for left, right in combinations(range(len(low.aligned)), 2):
            identity = pair_identifiers("within_dimension", low, left, low, right)
            stats, angles, singular = comparison(low.aligned[left], low.aligned[right], weights, d, args)
            within_rows.append({**identity, **stats})
            append_angles(angle_rows, identity, angles, singular)
        if d == args.reference_dimension:
            continue
        stats, _, _ = comparison(mean_low, mean_reference, weights, d, args)
        uncertainty_scale = float(np.sqrt(spread_low**2 + spread_reference**2))
        mean_rows.append({"dimension": d, **stats, "within_dimension_rmsd": spread_low,
            "within_reference_rmsd": spread_reference, "uncertainty_scale": uncertainty_scale,
            "leading_d_residual_over_spread": stats["rmsd_to_reference_leading_d"] / uncertainty_scale if uncertainty_scale else None})
        overlaps, heat = {}, []
        for low_member, z in enumerate(low.aligned):
            qlow, _, _, _ = basis(z, weights, args.rank_rtol, args.rank_atol)
            weighted_low = np.sqrt(weights)[:, None] * z
            norms = np.sqrt(np.sum(weighted_low**2, axis=0))
            normalized_axes = np.divide(weighted_low, norms, out=np.zeros_like(weighted_low), where=norms > 0)
            for ref_member, zr in enumerate(reference.aligned):
                identity = pair_identifiers("cross_dimension", low, low_member, reference, ref_member)
                stats, angles, singular = comparison(z, zr, weights, d, args)
                pair_rows.append({**identity, **stats})
                append_angles(angle_rows, identity, angles, singular)
                values, vectors, keep, qr, blocks = reference_cache[ref_member]
                overlap, raw_axes = qr.T @ qlow, qr.T @ normalized_axes
                # qlow is the model's own sorted EYM basis, not its arbitrary native frame.
                overlaps[f"ref_{ref_member:03d}_member_{low_member:03d}_orthonormal"] = overlap
                overlaps[f"ref_{ref_member:03d}_member_{low_member:03d}_canonical_axes"] = raw_axes
                for axis_type, loadings in (("eym", overlap), ("canonical", raw_axes)):
                    for row in mixing_statistics(loadings, blocks, min(d, qr.shape[1])):
                        group = next((index + 1 for index, block in enumerate(low_blocks[low_member])
                            if row["axis"] - 1 in block), None) if axis_type == "eym" else None
                        mixing_rows.append({**identity, "axis_type": axis_type, "low_eigenvalue_block": group, **row})
                fixed = np.zeros((args.reference_dimension, d))
                fixed[np.ix_(np.flatnonzero(keep), np.flatnonzero(low_keeps[low_member]))] = overlap**2
                heat.append(fixed)
                rank = qr.shape[1]
                target = min(d, rank)
                if target == 0:
                    continue
                # Cross-Gram permits controls without allocating new N-by-d arrays.
                cross = weighted_low.T @ qr
                total = float(np.sum(weighted_low**2) + values[keep][:target].sum())
                for draw in range(args.random_controls):
                    rotation, _ = np.linalg.qr(rng.normal(size=(rank, target)))
                    random_overlap = rotation.T @ (qr.T @ qlow)
                    a, _ = angles_from_overlap(random_overlap)
                    mass = float(np.sum(random_overlap**2))
                    scaled_cross = (cross @ rotation) * np.sqrt(values[keep][:target])[None]
                    residual = np.sqrt(nonnegative(total - 2 * np.linalg.svd(scaled_cross, compute_uv=False).sum(), total))
                    random_rows.append({**identity, "draw": draw, "overlap_fraction": mass / target,
                        "max_angle_deg": float(a.max()) if len(a) else None, "matched_spectrum_rmsd": float(residual)})
        np.savez_compressed(output / f"mode_overlaps_d_{d:03d}.npz", **overlaps)
        heatmaps[d] = np.mean(heat, axis=0)
        rows = [row for row in pair_rows if row["dimension"] == d]
        controls = [row["overlap_fraction"] for row in random_rows if row["dimension"] == d]
        overlap_values = [row["right_fraction"] for row in rows if row["right_fraction"] is not None]
        angle_values = [row["max_angle_deg"] for row in rows if row["max_angle_deg"] is not None]
        summaries.append({"dimension": d, "num_model_pairs": len(rows),
            "mean_leading_overlap_fraction": float(np.mean(overlap_values)) if overlap_values else None,
            "std_leading_overlap_fraction": float(np.std(overlap_values)) if overlap_values else None,
            "mean_max_angle_deg": float(np.mean(angle_values)) if angle_values else None,
            "std_max_angle_deg": float(np.std(angle_values)) if angle_values else None,
            "mean_random_overlap_fraction": float(np.mean(controls)) if controls else None,
            "random_overlap_q025": float(np.quantile(controls, .025)) if controls else None,
            "random_overlap_q975": float(np.quantile(controls, .975)) if controls else None})
        print(f"Completed chemical-space comparisons: d={d} against D={args.reference_dimension}", flush=True)
    for row in summaries:
        d = row["dimension"]
        for name, collection in (("within_dimension_overlap", within_rows),
                                 ("reference_leading_overlap", reference_leading_rows)):
            values = [item["right_fraction"] for item in collection if item["dimension"] == d and item["right_fraction"] is not None]
            row[name] = float(np.mean(values)) if values else None
    tsv(output / "cross_dimension_geometry.tsv", pair_rows)
    tsv(output / "within_dimension_geometry.tsv", within_rows)
    tsv(output / "ensemble_mean_geometry.tsv", mean_rows)
    tsv(output / "mode_mixing.tsv", mixing_rows)
    tsv(output / "random_subspace_controls.tsv", random_rows)
    tsv(output / "comparison_summary.tsv", summaries)
    tm_rows, response_modes, response_spectra = [], [], []
    if args.response_analysis == "yes":
        from _responses_v1 import canonical_samples, evaluate_gradients, response_pair, response_eigensystem
        samples = canonical_samples(reference)
        mixture_weights = samples["quadrature_weights"]
        np.savez_compressed(output / "shared_mixture_samples.npz", **samples)
        manifest["shared_mixture_sampling"] = {"source": str(reference.canonical / "samples.npz"),
            "source_sha256": sha256(reference.canonical / "samples.npz"), "num_samples": len(mixture_weights),
            "aggregation": reference.config["arguments"]["metric_aggregation"],
            "meaning": "same archived reference draws for every model; no new prior, concentration conversion, or random sampling"}
        responses = {}
        for d, low in saved.items():
            cache_path = output / f"responses_d_{d:03d}.npy" if args.save_responses == "yes" else None
            responses[d] = evaluate_gradients(low, samples, device, args.batch_size, cache_path)
        for member, z in enumerate(reference.aligned):
            values, *_ = response_eigensystem(z, responses[args.reference_dimension][member], weights,
                mixture_weights, args.rank_rtol, args.rank_atol)
            for mode, value in enumerate(values):
                response_spectra.append({"reference_member": member, "mode": mode + 1, "eigenvalue": float(value)})
        # EYM function modes themselves may vary even if the full response span
        # is reproducible. Compare matching leading subspaces at every rank.
        coefficients = [response_eigensystem(z, responses[reference.dimension][member], weights,
            mixture_weights, args.rank_rtol, args.rank_atol)[1]
            for member, z in enumerate(reference.aligned)]
        np.savez_compressed(output / "reference_function_coefficients.npz",
            **{f"member_{member:03d}": value for member, value in enumerate(coefficients)})
        function_controls = []
        for left, right in combinations(range(len(reference.aligned)), 2):
            cross = gram(responses[reference.dimension][left], responses[reference.dimension][right], mixture_weights)
            overlap = coefficients[left].T @ cross @ coefficients[right]
            for d in dimensions:
                subspace = overlap[:d, :d]
                angles, singular = angles_from_overlap(subspace)
                mass = float(np.sum(singular**2))
                identity = {"comparison": "reference_leading_stability", "dimension": d,
                    "member": left, "reference_dimension": reference.dimension, "reference_member": right}
                function_controls.append({**identity, "rank_left": subspace.shape[0], "rank_right": subspace.shape[1],
                    "overlap": mass, "retained_fraction": mass / subspace.shape[1] if subspace.shape[1] else None,
                    "max_angle_deg": float(angles.max()) if len(angles) else None})
                append_angles(angle_rows, identity, angles, singular, "mixture_response", "reference_leading_d")
        tsv(output / "reference_leading_function_controls.tsv", function_controls)
        for d, low in saved.items():
            pairs = [(a, b, "cross_dimension", reference) for a in range(len(low.aligned))
                for b in range(len(reference.aligned))] if d != args.reference_dimension else []
            pairs.extend((a, b, "within_dimension", low) for a, b in combinations(range(len(low.aligned)), 2))
            for low_member, ref_member, kind, ref in pairs:
                identity = pair_identifiers(kind, low, low_member, ref, ref_member)
                stats, overlap, retention, values = response_pair(ref.aligned[ref_member], low.aligned[low_member],
                    responses[ref.dimension][ref_member], responses[d][low_member], weights, mixture_weights,
                    args.rank_rtol, args.rank_atol)
                tm_rows.append({**identity, **stats})
                for target, subspace in (("leading_d", overlap[:d]), ("full_reference", overlap)):
                    angles, singular = angles_from_overlap(subspace)
                    append_angles(angle_rows, identity, angles, singular, "mixture_response", target)
                for mode, retained in enumerate(retention):
                    response_modes.append({**identity, "reference_mode": mode + 1,
                        "function_space_retention": float(retained), "reference_eigenvalue": float(values[mode])})
        tsv(output / "tm_losses.tsv", tm_rows)
        tsv(output / "response_mode_retention.tsv", response_modes)
        tsv(output / "common_quadrature_reference_spectrum.tsv", response_spectra)
        manifest["common_quadrature_note"] = (
            "TM losses and their EYM tails use re-evaluated reference decoder gradients on the archived reference draws. "
            "reference_spectrum.tsv uses the saved canonical metric. Small floating-point differences are recorded separately; "
            "lower models' canonicalization Monte Carlo draws may differ, but every TM comparison uses identical reference draws.")
    tsv(output / "principal_angles.tsv", angle_rows)
    plots(output, summaries, spectra, heatmaps, tm_rows, args.reference_dimension)
    manifest["rng_initial"], manifest["rng_final"] = initial_rng, rng.bit_generator.state
    manifest["files_sha256"] = {path.name: sha256(path) for path in output.iterdir() if path.is_file() and path.name != "metadata.json"}
    manifest["status"] = "complete"
    write_json(output / "metadata.json", manifest)
    print(f"Dimension analysis complete: {output}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FloatingPointError) as exc:
        raise SystemExit(f"Error: {exc}")
