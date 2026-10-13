#!/usr/bin/env python3
"""Small CPU integration check using explicitly synthetic JSON data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def use_source_checkout():
    """Make this checkout importable here and in the workflow subprocesses."""
    source = Path(__file__).resolve().parents[1] / "src"
    if not (source / "mixture_models" / "__init__.py").is_file():
        raise SystemExit("Place this script in the project's examples/ directory, next to ../src/.")
    sys.path.insert(0, str(source))
    inherited = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(source) + (os.pathsep + inherited if inherited else "")


def invoke(command, options):
    subprocess.run([sys.executable, "-m", "mixture_models", command, *options], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="smoke-output")
    args = parser.parse_args()
    use_source_checkout()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit("Choose an empty output directory.")
    units = {"pressure": "kT nm^-3", "potential_energy": "kT", "temperature": "K",
             "concentration": "nm^-3", "nominal_temperature": 300.0}
    rng = np.random.default_rng(18)
    alphabet = np.array(list("MGKTRADEYVLQWFSHNPCI"))
    sequences = ["".join(rng.choice(alphabet, size=20)) for _ in range(24)]
    systems = []
    for index in range(40):
        n = 1 + index % 4
        selected = rng.choice(len(sequences), size=n, replace=False)
        counts = rng.multinomial(96, np.full(n, 1 / n)) + 1
        concentration = float(rng.uniform(0.02, 0.22))
        pressure = concentration + 0.4 * concentration**2
        systems.append({"directory_name": f"synthetic_{index:03d}", "input_data_header": {"mixture_properties": {
            "chains": [[sequences[i], int(count)] for i, count in zip(selected, counts)],
            "total chains": int(counts.sum()), "chain concentration": concentration,
            "monomer concentration": 20 * concentration}},
            "observables": {"pressure": {"mean": pressure, "stddev": 0.001, "sem": 0.0001},
                            "pe": {"mean": 0.0}, "temperature": {"mean": 300.0}}})
    for name, subset in (("train_a", systems[:15]), ("train_b", systems[15:30]), ("test", systems[30:])):
        (output / f"{name}.json").write_text(json.dumps({"units": units,
            "script": "synthetic smoke fixture; these are not simulation results", "systems": subset}, indent=2) + "\n")
    run = output / "run"
    common = ["--run-dir", str(run), "--dimensions", "2", "3", "--device", "cpu", "--threads", "1"]
    invoke("train", [*common, "--training-json", str(output / "train_a.json"), str(output / "train_b.json"),
        "--num-folds", "5", "--num-models", "10", "--seed", "731", "--batch-sizes", "8", "--learning-rates", "0.001",
        "--restarts", "2", "--max-epochs", "3", "--warmup-epochs", "1", "--num-workers", "0", "--resume", "no"])
    # Idempotent selection and complete-run resumption must retain the same manifest.
    selection_bytes = [(run / "best" / f"d_{d:03d}" / "selection.json").read_bytes() for d in (2, 3)]
    invoke("train", [*common, "--training-json", str(output / "train_a.json"), str(output / "train_b.json"),
        "--num-folds", "5", "--num-models", "10", "--seed", "731", "--batch-sizes", "8", "--learning-rates", "0.001",
        "--restarts", "2", "--max-epochs", "3", "--warmup-epochs", "1", "--num-workers", "0", "--resume", "yes"])
    for d, expected in zip((2, 3), selection_bytes):
        assert (run / "best" / f"d_{d:03d}" / "selection.json").read_bytes() == expected
    invoke("canonicalize", [*common, "--sequence-pool", "training", "--dirichlet-total", "0.3651",
        "--num-samples", "200", "--batch-size", "50", "--encoding-batch-size", "32", "--diagnostic-points", "20", "--seed", "43"])
    for split in ("validation", "test"):
        invoke("evaluate", [*common, "--split", split, "--name", split, "--batch-size", "8", "--save-hessians", "yes",
               *(["--data-json", str(output / "test.json")] if split == "test" else [])])
    from mixture_models.data import load_json_files
    from mixture_models.evaluation import Ensemble
    test_records, _ = load_json_files([output / "test.json"])
    for dimension in (2, 3):
        best = run / "best" / f"d_{dimension:03d}"
        selection = json.loads((best / "selection.json").read_text())
        assert len(list(best.glob("*.pt"))) == 10
        assert [sum(item["fold"] == fold for item in selection["models"]) for fold in range(5)] == [2] * 5
        with np.load(run / "evaluation" / f"d_{dimension:03d}" / "validation" / "predictions.npz") as predictions:
            np.testing.assert_array_equal(predictions["eligible_model_count"], np.full(30, 2))
        native = Ensemble(run, dimension, basis="native").evaluate_records(test_records, batch_size=8)
        canonical = Ensemble(run, dimension, basis="canonical").evaluate_records(test_records, batch_size=8)
        np.testing.assert_allclose(native["pressure"], canonical["pressure"], rtol=2e-5, atol=2e-7)
    print(f"Smoke check passed. Outputs: {run}")


if __name__ == "__main__":
    main()
