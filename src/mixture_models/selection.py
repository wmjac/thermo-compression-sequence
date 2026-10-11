"""Full-precision validation selection with equal representation of every fold."""
from pathlib import Path
import math
import shutil
from .data import sha256_file
from .provenance import read_json, timestamp, write_json


def select_models(run_dir, dimensions, num_models):
    run = Path(run_dir)
    run_config = read_json(run / "config.json")
    num_folds = run_config["num_folds"]
    if num_models <= 0 or num_models % num_folds:
        raise ValueError("num_models must be a positive multiple of the run's num_folds.")
    quota = num_models // num_folds
    for dimension in dimensions:
        chosen = []
        for fold in range(num_folds):
            candidates = []
            for metrics_path in sorted((run / "trials" / f"d_{dimension:03d}" / f"fold_{fold:02d}").glob("*/metrics.json")):
                metrics = read_json(metrics_path)
                if metrics["status"] != "complete" or not math.isfinite(metrics["best_validation_loss"]):
                    continue
                config = read_json(metrics_path.parent / "config.json")
                if not (metrics_path.parent / "best.pt").exists():
                    raise ValueError(f"Missing best checkpoint for {metrics_path.parent}.")
                candidates.append({**config, **metrics, "trial_path": str(metrics_path.parent.relative_to(run))})
            candidates.sort(key=lambda item: (item["best_validation_loss"], item["best_epoch"],
                                              item["restart"], item["trial_id"]))
            if len(candidates) < quota:
                raise ValueError(f"d={dimension}, fold={fold}: need {quota} complete candidates, found {len(candidates)}.")
            for member, candidate in enumerate(candidates[:quota]):
                filename = f"fold_{fold:02d}_member_{member:02d}.pt"
                candidate.update(member=member, checkpoint=filename)
                chosen.append(candidate)
        output = run / "best" / f"d_{dimension:03d}"
        output.mkdir(parents=True, exist_ok=True)
        desired = {item["checkpoint"] for item in chosen}
        for old in output.glob("*.pt"):
            if old.name not in desired:
                old.unlink()
        for item in chosen:
            shutil.copy2(run / item["trial_path"] / "best.pt", output / item["checkpoint"])
            item["checkpoint_sha256"] = sha256_file(output / item["checkpoint"])
        manifest = {"dimension": dimension, "num_folds": num_folds, "num_models": num_models,
                    "models_per_fold": quota,
                    "ranking": ["best_validation_loss", "best_epoch", "restart", "trial_id"], "models": chosen}
        previous = read_json(output / "selection.json") if (output / "selection.json").exists() else {}
        if {key: value for key, value in previous.items() if key != "created_at"} != manifest:
            write_json(output / "selection.json", {"created_at": timestamp(), **manifest})
        print(f"Selected exactly {num_models} models for d={dimension}: {output}", flush=True)


def load_selection(run_dir, dimension):
    root = Path(run_dir) / "best" / f"d_{dimension:03d}"
    selection = read_json(root / "selection.json")
    if len(selection["models"]) != selection["num_models"]:
        raise ValueError("Selection manifest does not contain the declared number of models.")
    for item in selection["models"]:
        if sha256_file(root / item["checkpoint"]) != item["checkpoint_sha256"]:
            raise ValueError(f"Selected checkpoint has changed: {root / item['checkpoint']}")
    return root, selection
