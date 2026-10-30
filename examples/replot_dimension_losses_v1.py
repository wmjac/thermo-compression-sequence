#!/usr/bin/env python3
"""Regenerate dimension PNGs from saved metrics, without loading any models."""
import argparse
import json
from pathlib import Path
import re
import sys

# Run directly from an unpacked checkout; installation is unnecessary.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mixture_models.plotting import plot_dimension_comparison, plot_dimension_losses


def load_summary(directory, name):
    # Prefer per-dimension metrics: an interrupted workflow's summary JSON may
    # contain only the last dimension passed to an individual CLI invocation.
    files = sorted(directory.glob(f"d_[0-9][0-9][0-9]/{name}/metrics.json"))
    if files:
        summary = []
        for path in files:
            with path.open() as handle:
                item = json.load(handle)
            if item["dimension"] != int(path.parents[1].name[2:]):
                raise ValueError(f"Dimension disagrees with metrics directory: {path}")
            summary.append(item)
    else:
        with (directory / f"summary_{name}.json").open() as handle:
            summary = json.load(handle)
    if not summary or len({item["dimension"] for item in summary}) != len(summary):
        raise ValueError(f"Empty or repeated dimensions in {name} metrics.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path, help="Existing run/evaluation or analysis01/evaluation directory.")
    parser.add_argument("--names", nargs="+", help="Evaluation names; default: discover all saved evaluations.")
    args = parser.parse_args()
    directory = args.evaluation_dir.expanduser().resolve()
    if not directory.is_dir():
        parser.error(f"Missing evaluation directory: {directory}")
    names = args.names
    if names is None:
        discovered = ({path.stem[len("summary_"):] for path in directory.glob("summary_*.json")} |
                      {path.parent.name for path in directory.glob("d_[0-9][0-9][0-9]/*/metrics.json")})
        names = [name for name in ("validation", "test") if name in discovered]
        names.extend(sorted(discovered - {"validation", "test"}))
    if not names or len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names):
        parser.error("Provide distinct evaluation names, or a directory containing saved metrics.")
    # Read all numerical inputs before writing any figures. JSONs/CSVs/NPZs and
    # collection completion status are never modified by this plotting helper.
    summaries = {name: load_summary(directory, name) for name in names}
    for name, summary in summaries.items():
        plot_dimension_losses(summary, directory / f"loss_by_dimension_{name}.png")
    plot_dimension_comparison({name.title(): summary for name, summary in summaries.items()},
                              directory / "loss_by_dimension.png")
    print(f"Saved linear loss, log loss and log RMSE PNGs for {', '.join(names)}: {directory}")


if __name__ == "__main__":
    main()
