#!/usr/bin/env python3
"""Evaluate test-model loss spread, or reuse already saved test predictions."""
import argparse
from pathlib import Path
import re
from types import SimpleNamespace
import numpy as np
from _post_training_common_v1 import (
    dimension_run, dimensions_argument, prepare_output, read_json, runtime,
    sha256, tsv, write_json,
)
from _ensemble_statistics_v2 import (
    analysis_provenance, plot_test_losses, test_loss_statistics,
)


def load_test_evaluation(run, dimension, name, basis):
    """Verify selection identity and recompute loss spread from cached predictions."""
    from mixture_models.selection import load_selection
    view = dimension_run(run, dimension)
    best, selection = load_selection(view, dimension)
    source = view / "evaluation" / f"d_{dimension:03d}" / name
    metrics = read_json(source / "metrics.json")
    config = read_json(source / "config.json")
    if metrics.get("split") != "test" or config["arguments"].get("split") != "test":
        raise ValueError(f"Expected independent-test evaluation, not validation: {source}")
    if metrics["dimension"] != dimension or config["dimension"] != dimension:
        raise ValueError(f"Evaluation dimension mismatch: {source}")
    if metrics["basis"] != basis or config["arguments"]["basis"] != basis:
        raise ValueError(f"Saved basis differs from --basis: {source}")
    if config["selection_sha256"] != sha256(best / "selection.json"):
        raise ValueError(f"Evaluation belongs to a different selected ensemble: {source}")
    if basis == "canonical":
        transforms = view / "canonical" / f"d_{dimension:03d}" / "transforms.npz"
        if config.get("canonical_transforms_sha256") != sha256(transforms):
            raise ValueError(f"Evaluation canonical transforms changed: {source}")
    with np.load(source / "predictions.npz", allow_pickle=False) as archive:
        target = archive["target"].astype(np.float64)
        pressure = archive["individual_pressure"].astype(np.float64)
        ctot = archive["ctot"].astype(np.float64)
        eligibility = archive["eligibility"]
        eligible_count = archive["eligible_model_count"]
        saved_mean = archive["ensemble_pressure"].astype(np.float64)
    statistics, member_losses, member_rmses = test_loss_statistics(target, pressure, ctot)
    k, n = pressure.shape
    if k != len(selection["models"]) or config["num_models"] != k:
        raise ValueError("Saved predictions do not match selected model count.")
    if (eligibility.shape != (k, n) or not np.all(eligibility)
            or eligible_count.shape != (n,) or not np.all(eligible_count == k)):
        raise ValueError("Every selected model must predict every test system.")
    if saved_mean.shape != (n,) or not np.allclose(
            saved_mean, pressure.mean(axis=0), rtol=1e-12, atol=1e-15):
        raise ValueError("Saved ensemble pressure differs from the member mean.")
    if len(metrics["individual_models"]) != k:
        raise ValueError("Saved individual-model metrics have an inconsistent count.")
    if metrics["ensemble"]["num_systems"] != n or not np.isclose(
            metrics["ensemble"]["pressure_loss"], statistics["ensemble_pressure_loss"],
            rtol=1e-12, atol=1e-15):
        raise ValueError("Saved ensemble metrics disagree with the prediction archive.")
    individual_rows = []
    for index, (item, selected) in enumerate(zip(metrics["individual_models"], selection["models"])):
        if item["checkpoint"] != selected["checkpoint"] or item["fold"] != selected["fold"]:
            raise ValueError("Saved model order/identity differs from the selection.")
        if item["num_systems"] != n or not np.isclose(
                item["pressure_loss"], member_losses[index], rtol=1e-12, atol=1e-15):
            raise ValueError("Saved member loss disagrees with the prediction archive.")
        individual_rows.append({
            "dimension": dimension, "member": index, **item,
        })
    identity = statistics["ensemble_pressure_loss"] + statistics["mean_normalized_prediction_variance"]
    if not np.isclose(statistics["member_pressure_loss_mean"], identity, rtol=1e-10, atol=1e-14):
        raise FloatingPointError("Member/ensemble squared-loss variance identity failed.")
    updated = {**metrics, "test_model_spread": {
        **statistics, "ddof": 0,
        "interpretation": "population SD across selected-model test losses, not an SEM or confidence interval",
        "rmse_definition": "mean and SD of individual-model sqrt(loss), not sqrt(mean(loss))",
    }}
    source_info = {
        "dimension": dimension, "run_view": str(view), "evaluation_directory": str(source),
        "selection_sha256": sha256(best / "selection.json"),
        "selected_models": selection["models"],
        "files_sha256": {
            str(source / filename): sha256(source / filename)
            for filename in ("metrics.json", "config.json", "predictions.npz")
        },
    }
    return updated, {"dimension": dimension, **statistics}, individual_rows, source_info


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Normal run or combined analysis01 directory.")
    parser.add_argument("--dimensions", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True, help="New directory for uncertainty summaries/plots.")
    parser.add_argument("--reuse-predictions", choices=("yes", "no"), default="yes",
                        help="yes: read existing test predictions; no: first run the original evaluation routine.")
    parser.add_argument("--data-json", nargs="*", default=[],
                        help="Independent test files for fresh evaluation only; leave empty when reusing predictions.")
    parser.add_argument("--name", default="test", help="Label of the original evaluation to read or create.")
    parser.add_argument("--basis", choices=("canonical", "native"), default="canonical")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--save-hessians", choices=("yes", "no"), default="no")
    parser.add_argument("--allow-training-overlap", choices=("yes", "no"), default="no")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--overwrite", choices=("yes", "no"), default="no",
                        help="Replace this script's outputs; in fresh mode also passed to original evaluation.")
    return parser


def main():
    args = build_parser().parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        raise ValueError("Evaluation name may contain letters, digits, underscores, and hyphens.")
    if args.reuse_predictions == "yes" and args.data_json:
        raise ValueError("Reuse reads the archived test dataset; omit --data-json paths.")
    if args.reuse_predictions == "no" and not args.data_json:
        raise ValueError("Fresh evaluation requires --data-json.")
    runtime(args)
    dimensions = dimensions_argument(args.dimensions)
    root = Path(args.run_dir).expanduser().resolve()
    views = [dimension_run(root, d) for d in dimensions]
    source_dirs = [view / "evaluation" / f"d_{d:03d}" / args.name
                   for d, view in zip(dimensions, views)]
    output = prepare_output(args.output_dir, args.overwrite, [root, *views, *source_dirs])
    manifest = {
        "status": "in_progress", "provenance": analysis_provenance(args),
        "split": "test", "dimensions": dimensions, "sources": [],
        "aggregation": "sample-mean pressure loss; all selected models use the same test systems",
        "uncertainty": {
            "definition": "population standard deviation of individual-model test metrics",
            "ddof": 0, "num_models": "actual selected count at each dimension; not hard-coded to five",
            "scope": "selected models share training data; no IID, SEM, or confidence-interval interpretation",
            "central_values": "member mean and ensemble-prediction loss are both reported separately",
            "validation": "not pooled into these test error bars; original held-out-fold evaluation is unchanged",
        },
        "reuse": "original evaluation files are read-only" if args.reuse_predictions == "yes"
                 else "predictions generated with the original mixture_models.evaluation.run_evaluation",
    }
    write_json(output / "metadata.json", manifest)
    summary, individual, detailed = [], [], []
    for dimension, view in zip(dimensions, views):
        if args.reuse_predictions == "no":
            from mixture_models.evaluation import run_evaluation
            original = dict(vars(args), command="evaluate", run_dir=str(view),
                            dimensions=[dimension], split="test")
            run_evaluation(SimpleNamespace(**original))
        metrics, row, members, source = load_test_evaluation(root, dimension, args.name, args.basis)
        detailed.append(metrics)
        summary.append(row)
        individual.extend(members)
        manifest["sources"].append(source)
        write_json(output / f"metrics_d_{dimension:03d}.json", metrics)
        write_json(output / "metadata.json", manifest)
        print(f"d={dimension}: member loss {row['member_pressure_loss_mean']:.17g} "
              f"+/- {row['member_pressure_loss_std']:.17g} SD; "
              f"ensemble prediction loss {row['ensemble_pressure_loss']:.17g}", flush=True)
    write_json(output / "summary_test.json", detailed)
    tsv(output / "loss_summary.tsv", summary)
    tsv(output / "individual_model_losses.tsv", individual)
    plot_test_losses(summary, output)
    for source in manifest["sources"]:
        for path, digest in source["files_sha256"].items():
            if sha256(path) != digest:
                raise ValueError(f"Evaluation source changed during uncertainty analysis: {path}")
    manifest["files_sha256"] = {
        path.name: sha256(path) for path in output.iterdir()
        if path.is_file() and path.name != "metadata.json"
    }
    manifest["status"] = "complete"
    write_json(output / "metadata.json", manifest)
    print(f"Test uncertainty analysis complete: {output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FloatingPointError) as exc:
        raise SystemExit(f"Error: {exc}")
