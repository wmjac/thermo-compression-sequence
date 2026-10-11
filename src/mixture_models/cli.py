"""CLI settings are resolved against the selected model's declared protocol."""
import argparse
import math
from dataclasses import replace
from pathlib import Path
import re
import sys
from .models import DEFAULT_MODEL, get_model_module


def dimensions(values):
    result = []
    for value in values:
        if re.fullmatch(r"\d+", value):
            result.append(int(value))
        elif re.fullmatch(r"\d+-\d+", value):
            start, stop = map(int, value.split("-"))
            if stop < start:
                raise ValueError("Dimension ranges require increasing endpoints.")
            result.extend(range(start, stop + 1))
        elif re.fullmatch(r"\d+:\d+(?::\d+)?", value):
            fields = list(map(int, value.split(":")))
            start, stop = fields[:2]
            step = fields[2] if len(fields) == 3 else 1
            if step <= 0 or stop < start:
                raise ValueError("Dimension ranges require a positive step and increasing endpoints.")
            result.extend(range(start, stop + 1, step))
        else:
            raise ValueError(f"Invalid dimension specification {value!r}.")
    if not result or min(result) <= 0 or len(set(result)) != len(result):
        raise ValueError("Dimensions must be distinct positive integers; ranges use inclusive endpoints.")
    return result


def common(parser):
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dimensions", nargs="+", required=True, help="Integers or inclusive ranges, e.g. 2:10:2.")


def runtime(parser):
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--threads", type=int, default=1)


def build_parser(defaults):
    parser = argparse.ArgumentParser(description="Thermodynamic mixture-model training, canonicalization, and evaluation.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Pool JSON batches, train the hyperparameter grid, and select the ensemble.")
    common(train)
    runtime(train)
    train.add_argument("--training-json", nargs="+", required=True)
    train.add_argument("--model", default=DEFAULT_MODEL, help="Built-in model name or importable model-module path.")
    train.add_argument("--num-folds", type=int, default=5)
    train.add_argument("--num-models", type=int, default=5)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--batch-sizes", type=int, nargs="+", default=list(defaults.batch_sizes))
    train.add_argument("--learning-rates", type=float, nargs="+", default=list(defaults.learning_rates))
    for option in ("restarts", "max_epochs", "warmup_epochs", "patience"):
        train.add_argument("--" + option.replace("_", "-"), type=int, default=getattr(defaults, option))
    for option in ("grad_clip", "min_delta", "adam_beta1", "adam_beta2", "adam_eps", "weight_decay"):
        train.add_argument("--" + option.replace("_", "-"), type=float, default=getattr(defaults, option))
    train.add_argument("--amsgrad", choices=("yes", "no"), default="yes" if defaults.amsgrad else "no")
    train.add_argument("--dtype", choices=("float32", "float64"), default=defaults.dtype)
    train.add_argument("--validation-aggregation", choices=("batch-mean", "sample-mean"), default=defaults.validation_aggregation)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--pin-memory", choices=("auto", "yes", "no"), default="auto")
    train.add_argument("--persistent-workers", choices=("yes", "no"), default="yes")
    train.add_argument("--prefetch-factor", type=int, default=4)
    train.add_argument("--deterministic", choices=("yes", "no"), default="no")
    train.add_argument("--checkpoint-every", type=int, default=1, help="Save last.pt at this epoch interval, and on stop/completion.")
    train.add_argument("--plot-every", type=int, default=200)
    train.add_argument("--resume", choices=("yes", "no"), default="no")
    selection = commands.add_parser("select", help="Select exactly k models from completed trials without retraining.")
    common(selection)
    selection.add_argument("--num-models", type=int, default=5)
    canonical = commands.add_parser("canonicalize", help="Determine thermodynamic metrics and align the ensemble.")
    common(canonical)
    runtime(canonical)
    canonical.add_argument("--sequence-pool", default="training", help="training, or a .txt/.npy/.json sequence pool.")
    canonical.add_argument("--sequence-weighting", choices=("uniform",), default="uniform")
    canonical.add_argument("--concentration-prior", choices=("uniform",), default="uniform")
    canonical.add_argument("--composition-prior", choices=("gem",), default="gem")
    canonical.add_argument("--ctot-min", type=float, default=0.0, help="nm^-3")
    canonical.add_argument("--ctot-max", type=float, default=0.25, help="nm^-3")
    canonical.add_argument("--dirichlet-total", type=float, required=True, help="GEM total concentration A; choose explicitly.")
    canonical.add_argument("--num-samples", type=int, default=50000)
    canonical.add_argument("--batch-size", type=int, default=2000)
    canonical.add_argument("--encoding-batch-size", type=int, default=2048)
    canonical.add_argument("--eps-mass", type=float, default=1e-6)
    canonical.add_argument("--max-components", type=int, default=50)
    canonical.add_argument("--eps-effective", type=float, default=0.01)
    canonical.add_argument("--metric-aggregation", choices=("batch-mean", "sample-mean"), default="batch-mean")
    canonical.add_argument("--metric-eigenvalue-tolerance", type=float, default=1e-12)
    canonical.add_argument("--align-max-iter", type=int, default=200)
    canonical.add_argument("--align-tolerance", type=float, default=1e-8)
    canonical.add_argument("--diagnostic-points", type=int, default=1000)
    canonical.add_argument("--seed", type=int, default=0)
    canonical.add_argument("--overwrite", choices=("yes", "no"), default="no")
    evaluate = commands.add_parser("evaluate", help="Evaluate test JSONs or the saved validation folds.")
    common(evaluate)
    runtime(evaluate)
    evaluate.add_argument("--split", choices=("test", "validation"), required=True)
    evaluate.add_argument("--data-json", nargs="*", default=[])
    evaluate.add_argument("--name", required=True, help="Output label, e.g. test or validation.")
    evaluate.add_argument("--basis", choices=("canonical", "native"), default="canonical")
    evaluate.add_argument("--batch-size", type=int, default=2048)
    evaluate.add_argument("--save-hessians", choices=("yes", "no"), default="no")
    evaluate.add_argument("--allow-training-overlap", choices=("yes", "no"), default="no")
    evaluate.add_argument("--overwrite", choices=("yes", "no"), default="no")
    return parser


def resolve_protocol(args, defaults):
    return replace(defaults, batch_sizes=tuple(args.batch_sizes), learning_rates=tuple(args.learning_rates),
                   **{key: getattr(args, key) for key in ("restarts", "max_epochs", "warmup_epochs", "patience",
                       "grad_clip", "min_delta", "adam_beta1", "adam_beta2", "adam_eps", "weight_decay", "dtype", "validation_aggregation")},
                   amsgrad=args.amsgrad == "yes")


def validate_args(args):
    for key, value in vars(args).items():
        numbers = value if isinstance(value, list) else [value]
        if any(isinstance(number, float) and not math.isfinite(number) for number in numbers):
            raise ValueError(f"{key} must contain finite values.")
    args.dimensions = dimensions(args.dimensions)
    args.run_dir = str(Path(args.run_dir).expanduser().resolve())
    for key in ("threads", "batch_size", "encoding_batch_size", "num_samples", "max_components", "align_max_iter",
                "diagnostic_points", "max_epochs", "patience", "restarts", "checkpoint_every", "plot_every", "prefetch_factor"):
        if hasattr(args, key) and getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive.")
    for key in ("seed", "num_workers", "warmup_epochs"):
        if hasattr(args, key) and getattr(args, key) < 0:
            raise ValueError(f"{key} must be nonnegative.")
    if args.command == "train":
        if args.num_folds < 2 or args.num_models <= 0 or args.num_models % args.num_folds:
            raise ValueError("At least two folds are required; num_models must be a positive multiple of num_folds.")
        if min(args.batch_sizes) <= 0 or min(args.learning_rates) <= 0:
            raise ValueError("Batch sizes and learning rates must be positive.")
        if len(set(args.batch_sizes)) != len(args.batch_sizes) or len(set(args.learning_rates)) != len(args.learning_rates):
            raise ValueError("The hyperparameter grid cannot contain duplicates.")
        if not (0 <= args.adam_beta1 < 1 and 0 <= args.adam_beta2 < 1):
            raise ValueError("Adam betas must lie in [0,1).")
        if args.adam_eps <= 0 or args.grad_clip <= 0 or args.min_delta < 0 or args.weight_decay < 0:
            raise ValueError("Invalid optimizer/clipping/improvement settings.")
    elif args.command == "canonicalize":
        if args.ctot_min < 0 or args.ctot_max <= args.ctot_min or args.dirichlet_total <= 0:
            raise ValueError("Invalid concentration range or Dirichlet total.")
        if not (0 < args.eps_mass < 1 and 0 < args.eps_effective < 1):
            raise ValueError("eps_mass and eps_effective must lie in (0,1).")
        if args.metric_eigenvalue_tolerance < 0 or args.align_tolerance <= 0:
            raise ValueError("Invalid metric/alignment tolerances.")
    elif args.command == "evaluate" and not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        raise ValueError("Evaluation name may contain letters, digits, underscores, and hyphens.")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--model", default=DEFAULT_MODEL)
    model_name = preliminary.parse_known_args(argv)[0].model
    module = get_model_module(model_name)
    defaults = module.training_protocol()
    parser = build_parser(defaults)
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        if args.command == "train":
            from .training import run_training
            if not run_training(args, resolve_protocol(args, defaults)):
                raise SystemExit(75)
        elif args.command == "select":
            from .selection import select_models
            select_models(args.run_dir, args.dimensions, args.num_models)
        elif args.command == "canonicalize":
            from .canonical import run_canonicalization
            run_canonicalization(args)
        elif args.command == "evaluate":
            from .evaluation import run_evaluation
            run_evaluation(args)
    except (ValueError, FileNotFoundError, FloatingPointError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
