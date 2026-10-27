"""Read-only access to saved runs for the post-training analysis scripts."""
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    def convert(item):
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False, default=convert)
        handle.write("\n")
    temporary.replace(path)


def tsv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format(float(value), ".17g") if isinstance(value, (float, np.floating))
                             else value for key, value in row.items()})


def prepare_output(path, overwrite, protected=()):
    original = Path(path).expanduser()
    if original.is_symlink():
        raise ValueError(f"Output directory must not be a symlink: {original}")
    path = original.resolve()
    for root in protected:
        root = Path(root).expanduser().resolve()
        if path == root or path in root.parents:
            raise ValueError(f"Output must not replace an input run or its ancestor: {path}")
    if path.exists():
        if overwrite != "yes":
            raise ValueError(f"Output exists: {path}; choose a new output directory or --overwrite yes.")
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"Expected an ordinary output directory: {path}")
        marker = path / "metadata.json"
        if any(path.iterdir()) and (not marker.is_file()
                or "analysis_source_sha256" not in read_json(marker).get("provenance", {})):
            raise ValueError("Only a previous post-training output (or an empty directory) may be overwritten.")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def provenance(args):
    from mixture_models.provenance import collect_provenance, resolve_device
    result = collect_provenance(resolve_device(getattr(args, "device", "cpu")))
    result["analysis_source_sha256"] = {path.name: sha256(path)
        for path in sorted(Path(__file__).parent.glob("*_v1.py"))}
    result["arguments"] = vars(args)
    return result


def read_sequences(path):
    path = Path(path).expanduser().resolve()
    if path.suffix == ".npy":
        array = np.load(path, allow_pickle=False)
        if array.ndim != 1 or array.dtype.kind not in ("U", "S"):
            raise ValueError("Sequence NPY must be a one-dimensional string array.")
        sequences = array.astype(str).tolist()
    elif path.suffix == ".json":
        raw = read_json(path)
        sequences = raw.get("sequences") if isinstance(raw, dict) else raw
    else:
        sequences = [line.strip() for line in path.read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith("#")]
        if sequences and sequences[0] == "sequence":
            sequences = sequences[1:]
    if not isinstance(sequences, list) or not sequences or any(not isinstance(s, str) or not s for s in sequences):
        raise ValueError("Supply a nonempty sequence list: one sequence per line, JSON list, or string NPY.")
    if len(set(sequences)) != len(sequences):
        raise ValueError("The input list contains repeated sequences; no rows were silently discarded.")
    return sequences


def dimension_run(root, dimension):
    root = Path(root).expanduser().resolve()
    return root / "run_views" / f"d_{dimension:03d}" if (root / "run_views").is_dir() else root


def chemical_weights(config, sequences):
    """Only archived weights may define the chemical measure; never the export list."""
    raw = config.get("rho_chem")
    n = len(sequences)
    if raw is None:
        return np.full(n, 1.0 / n), {"distribution": "uniform", "source": "default: rho_chem not archived"}
    if raw == "uniform" or isinstance(raw, dict) and raw.get("distribution") == "uniform":
        return np.full(n, 1.0 / n), {"distribution": "uniform", "source": "archived rho_chem", "definition": raw}
    if isinstance(raw, dict) and "weights" in raw:
        values = raw["weights"]
        if isinstance(values, dict):
            if set(values) != set(sequences):
                raise ValueError("Archived rho_chem weights do not match the canonical sequence pool.")
            weights = np.array([values[s] for s in sequences], dtype=np.float64)
        else:
            if raw.get("sequences", sequences) != sequences:
                raise ValueError("Archived rho_chem sequence ordering is inconsistent.")
            weights = np.asarray(values, dtype=np.float64)
        if weights.shape != (n,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("Archived rho_chem weights must be finite and strictly positive.")
        if not np.isclose(weights.sum(), 1.0, rtol=1e-12, atol=1e-14):
            raise ValueError("Archived rho_chem weights must sum to one; weights were not renormalized.")
        return weights, {"distribution": "explicit", "source": "archived rho_chem", "definition": raw}
    raise ValueError("Unsupported archived rho_chem definition; refusing to replace it with uniform weights.")


@dataclass
class SavedDimension:
    dimension: int
    run: Path
    best: Path
    canonical: Path
    selection: dict
    config: dict
    transforms: np.ndarray
    inverses: np.ndarray
    sequences: list
    aligned: np.ndarray | None
    native: np.ndarray | None
    weights: np.ndarray | None
    chemical_measure: dict | None

    def source(self):
        return {"dimension": self.dimension, "run_dir": str(self.run),
                "best_directory": str(self.best.resolve()), "canonical_directory": str(self.canonical.resolve()),
                "selection_sha256": sha256(self.best / "selection.json"),
                "canonical_config_sha256": sha256(self.canonical / "config.json"),
                "canonical_config": self.config,
                "selected_models": [{key: item.get(key) for key in
                    ("checkpoint", "checkpoint_sha256", "fold", "member", "trial_path", "best_validation_loss")}
                    for item in self.selection["models"]]}


def load_dimension(root, dimension, features=True):
    from mixture_models.selection import load_selection
    run = dimension_run(root, dimension)
    best, selection = load_selection(run, dimension)
    canonical = run / "canonical" / f"d_{dimension:03d}"
    config = read_json(canonical / "config.json")
    if config.get("rho_mix", {}).get("concentration_units") != "nm^-3":
        raise ValueError("Canonical sample concentrations must already be archived in nm^-3.")
    if config["selection_sha256"] != sha256(best / "selection.json"):
        raise ValueError(f"Canonicalization belongs to a different selection: d={dimension}.")
    if config["dimension"] != dimension or config["num_models"] != len(selection["models"]):
        raise ValueError("Canonical and selected-model dimensions/counts disagree.")
    if read_json(canonical / "diagnostics.json")["status"] != "complete":
        raise ValueError(f"Canonicalization is incomplete: {canonical}")
    for name, digest in config["artifact_sha256"].items():
        if sha256(canonical / name) != digest:
            raise ValueError(f"Canonical artifact changed: {canonical / name}")
    with np.load(canonical / "transforms.npz", allow_pickle=False) as data:
        transforms, inverses = data["transforms"], data["inverses"]
    expected = (len(selection["models"]), dimension, dimension)
    if transforms.shape != expected or inverses.shape != expected:
        raise ValueError("Invalid canonical transform shapes.")
    np.testing.assert_allclose(transforms @ inverses, np.broadcast_to(np.eye(dimension), expected),
                               rtol=1e-9, atol=1e-9)
    sequences, aligned, native, weights, measure = [], None, None, None, None
    if features:
        with np.load(canonical / "features.npz", allow_pickle=False) as data:
            sequences = data["sequences"].astype(str).tolist()
            aligned, native = data["aligned"].astype(np.float64), data["native"].astype(np.float64)
        if not sequences or len(set(sequences)) != len(sequences):
            raise ValueError("Canonical sequence pool is empty or contains duplicates.")
        if aligned.shape != (expected[0], len(sequences), dimension) or native.shape != aligned.shape:
            raise ValueError("Canonical features have inconsistent shapes.")
        if not np.all(np.isfinite(aligned)) or not np.all(np.isfinite(native)):
            raise ValueError("Nonfinite canonical features.")
        weights, measure = chemical_weights(config, sequences)
    return SavedDimension(dimension, run, best, canonical, selection, config, transforms, inverses,
                          sequences, aligned, native, weights, measure)


def models_for(saved, device):
    from mixture_models.checkpoints import load_model
    for item in saved.selection["models"]:
        yield load_model(saved.best / item["checkpoint"], device)


def check_common_measures(saved_dimensions, reference):
    """Reorder existing features when necessary; do not change either measure."""
    for saved in saved_dimensions:
        if saved.config["rho_mix"] != reference.config["rho_mix"]:
            raise ValueError(f"rho_mix differs between d={saved.dimension} and reference d={reference.dimension}.")
        if set(saved.sequences) != set(reference.sequences):
            raise ValueError("Canonicalization sequence pools differ; a common measure is required.")
        index = {s: i for i, s in enumerate(saved.sequences)}
        order = np.array([index[s] for s in reference.sequences])
        if not np.allclose(saved.weights[order], reference.weights, rtol=1e-12, atol=1e-15):
            raise ValueError("Archived rho_chem differs across dimensions.")
        saved.aligned = saved.aligned[:, order]
        saved.native = saved.native[:, order]
        saved.weights = saved.weights[order]
        saved.sequences = reference.sequences.copy()


def dimensions_argument(values):
    from mixture_models.cli import dimensions
    return dimensions(values)


def runtime(args):
    import torch
    from mixture_models.provenance import resolve_device
    if args.threads <= 0 or args.batch_size <= 0:
        raise ValueError("Threads and batch size must be positive.")
    torch.set_num_threads(args.threads)
    return resolve_device(args.device)
