"""Thermodynamic ensemble API and independent-test / out-of-fold evaluation."""
from pathlib import Path
import math
import shutil
import numpy as np
import torch
from .checkpoints import load_model
from .data import load_json_files, load_run_records, save_physical_dataset, sha256_file, validate_with_model
from .provenance import collect_provenance, read_json, resolve_device, timestamp, write_json
from .selection import load_selection
from .training import move_inputs


class Ensemble:
    """Each member uses its own encoder and decoder, including after alignment."""
    def __init__(self, run_dir, dimension, device="cpu", basis="canonical"):
        self.run = Path(run_dir)
        self.device = torch.device(device)
        self.best, self.selection = load_selection(self.run, dimension)
        loaded = [load_model(self.best / item["checkpoint"], self.device) for item in self.selection["models"]]
        self.models = [item[0] for item in loaded]
        self.modules = [item[1] for item in loaded]
        self.configs = [item[2] for item in loaded]
        self.dimension, self.basis = dimension, basis
        self.num_models = len(self.models)
        if basis == "canonical":
            directory = self.run / "canonical" / f"d_{dimension:03d}"
            config = read_json(directory / "config.json")
            if config["selection_sha256"] != sha256_file(self.best / "selection.json"):
                raise ValueError("Canonical basis belongs to a different selection manifest; canonicalize the current selection.")
            if read_json(directory / "diagnostics.json")["status"] != "complete":
                raise ValueError("Canonicalization is incomplete.")
            if config["artifact_sha256"]["transforms.npz"] != sha256_file(directory / "transforms.npz"):
                raise ValueError("Canonical transformations have changed since they were recorded.")
            with np.load(directory / "transforms.npz", allow_pickle=False) as arrays:
                self.transforms = arrays["transforms"]
                self.inverses = arrays["inverses"]
            if self.transforms.shape != (self.num_models, dimension, dimension):
                raise ValueError("Canonical transformations have an incompatible shape.")
        elif basis == "native":
            self.transforms = np.repeat(np.eye(dimension)[None], self.num_models, axis=0)
            self.inverses = self.transforms.copy()
        else:
            raise ValueError("basis must be native or canonical.")

    def encode_sequences(self, sequences, batch_size=2048):
        from .canonical import encode_pool
        for module in self.modules:
            module.validate_sequences(sequences)
        native = encode_pool(self.models, sequences, batch_size)
        return np.einsum("knd,ked->kne", native, self.transforms)

    def thermodynamics_from_latent(self, x, hessian=False):
        """x is (k,b,d) or a shared (b,d) point in the selected basis."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:
            x = np.broadcast_to(x, (self.num_models, *x.shape))
        if x.ndim != 3 or x.shape[0] != self.num_models or x.shape[2] != self.dimension:
            raise ValueError("Expected latent concentrations of shape (k,b,d) or (b,d).")
        result = {"free_energy_excess": [], "pressure_excess": [], "gradient": []}
        if hessian:
            result["hessian"] = []
        for member, model in enumerate(self.models):
            parameter = next(model.parameters())
            native_x = x[member] @ self.inverses[member].T
            tensor = torch.as_tensor(native_x, dtype=parameter.dtype, device=self.device)
            with torch.enable_grad():
                terms = model.thermodynamics(tensor, hessian=hessian)
            terms = {name: value.detach().cpu().numpy().astype(np.float64) for name, value in terms.items()}
            result["free_energy_excess"].append(terms["free_energy_excess"][:, 0])
            result["pressure_excess"].append(terms["pressure_excess"][:, 0])
            result["gradient"].append(terms["gradient"] @ self.inverses[member])
            if hessian:
                result["hessian"].append(np.einsum("ij,bjl,lm->bim", self.inverses[member].T,
                                                   terms["hessian"], self.inverses[member]))
        return {name: np.asarray(values) for name, values in result.items()}

    def evaluate_records(self, records, batch_size=2048, hessian=False):
        predictions, features, all_terms = [], [], {}
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            x, totals = [], []
            for member, (model, module, config) in enumerate(zip(self.models, self.modules, self.configs)):
                inputs = move_inputs(module.collate_records(batch, config["protocol"]["dtype"]), self.device)
                with torch.no_grad():
                    native_x = model.mixture_latent(inputs).cpu().numpy().astype(np.float64)
                x.append(native_x @ self.transforms[member].T)
                totals.append(inputs["ctot"].cpu().numpy()[:, 0].astype(np.float64))
            x = np.asarray(x)
            terms = self.thermodynamics_from_latent(x, hessian=hessian)
            predictions.append(terms["pressure_excess"] + np.asarray(totals))
            features.append(x)
            for name, values in terms.items():
                all_terms.setdefault(name, []).append(values)
        return {"pressure": np.concatenate(predictions, axis=1),
                "mixture_latent": np.concatenate(features, axis=1),
                **{name: np.concatenate(values, axis=1) for name, values in all_terms.items()}}


def pressure_metrics(target, prediction, ctot):
    residual = (target - prediction) / ctot
    mse = float(np.mean(residual ** 2))
    pearson = None
    if len(target) > 1 and np.std(target) > 0 and np.std(prediction) > 0:
        pearson = float(np.corrcoef(target, prediction)[0, 1])
    return {"num_systems": len(target), "pressure_loss": mse, "normalized_rmse": math.sqrt(mse),
            "normalized_bias": float(np.mean((prediction - target) / ctot)),
            "pressure_rmse_kT_nm^-3": float(np.sqrt(np.mean((prediction - target) ** 2))),
            "pearson_pressure": pearson}


def reject_training_overlap(records, training_records):
    by_id = {}
    for record in training_records:
        by_id.setdefault(record.system_id, []).append(record)
    for record in records:
        for trained in by_id.get(record.system_id, []):
            if (record.sequences == trained.sequences and record.counts == trained.counts
                    and math.isclose(record.ctot, trained.ctot, rel_tol=1e-12)):
                raise ValueError(f"Test system {record.system_id} overlaps the training pool; "
                                 "use validation evaluation or explicitly --allow-training-overlap yes.")


def run_evaluation(args):
    run = Path(args.run_dir).expanduser().resolve()
    device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    training_records, training_manifest = load_run_records(run)
    if args.split == "validation":
        if args.data_json:
            raise ValueError("Validation evaluation uses the run's saved pooled data and fold assignments; omit --data-json.")
        records, manifests = training_records, training_manifest
        folds = read_json(run / "folds.json")
        if folds["uid"] != [record.uid for record in records]:
            raise ValueError("Saved validation fold identities disagree with the input snapshots.")
        assignments = np.asarray(folds["validation_fold"])
    else:
        if not args.data_json:
            raise ValueError("Test evaluation requires --data-json.")
        records, manifests = load_json_files(args.data_json)
        if args.allow_training_overlap != "yes":
            reject_training_overlap(records, training_records)
        assignments = None
    summary = []
    for dimension in args.dimensions:
        ensemble = Ensemble(run, dimension, device, args.basis)
        for module in ensemble.modules:
            validate_with_model(records, module)
        output = run / "evaluation" / f"d_{dimension:03d}" / args.name
        if output.exists():
            if args.overwrite != "yes":
                raise ValueError(f"Evaluation output exists: {output}; use --overwrite yes to replace it.")
            shutil.rmtree(output)
        output.mkdir(parents=True)
        config = {"created_at": timestamp(), "arguments": vars(args), "dimension": dimension,
                  "num_models": ensemble.num_models, "inputs": manifests,
                  "selection_sha256": sha256_file(ensemble.best / "selection.json"),
                  "aggregation": "mean of eligible individual-model predictions",
                  "validation_eligibility": "only members trained with this system's fold held out",
                  "spread": "population standard deviation of eligible members, not a SEM"}
        if args.basis == "canonical":
            config["canonical_transforms_sha256"] = sha256_file(run / "canonical" / f"d_{dimension:03d}" / "transforms.npz")
        write_json(output / "config.json", config)
        write_json(output / "provenance.json", collect_provenance(device))
        save_physical_dataset(output / "dataset.npz", records)
        if args.split == "test":
            (output / "inputs").mkdir()
            for index, item in enumerate(manifests):
                shutil.copy2(item["path"], output / "inputs" / f'{index:03d}_{Path(item["path"]).name}')
        results = ensemble.evaluate_records(records, args.batch_size, args.save_hessians == "yes")
        target = np.array([record.pressure for record in records], dtype=np.float64)
        ctot = np.array([record.ctot for record in records], dtype=np.float64)
        eligibility = np.ones((ensemble.num_models, len(records)), dtype=bool)
        if args.split == "validation":
            eligibility = np.array([assignments == item["fold"] for item in ensemble.selection["models"]])
        counts = eligibility.sum(axis=0)
        if np.any(counts == 0):
            raise ValueError("Some systems have no eligible held-out models.")
        prediction = np.sum(np.where(eligibility, results["pressure"], 0.0), axis=0) / counts
        spread = np.sqrt(np.sum(np.where(eligibility, (results["pressure"] - prediction) ** 2, 0.0), axis=0) / counts)
        member_metrics = []
        for member, item in enumerate(ensemble.selection["models"]):
            mask = eligibility[member]
            member_metrics.append({"checkpoint": item["checkpoint"], "fold": item["fold"],
                "checkpoint_validation_loss": item["best_validation_loss"],
                **pressure_metrics(target[mask], results["pressure"][member, mask], ctot[mask])})
        metrics = {"dimension": dimension, "split": args.split, "basis": args.basis,
                   "loss_aggregation": "sample-mean", "ensemble": pressure_metrics(target, prediction, ctot),
                   "individual_models": member_metrics}
        write_json(output / "metrics.json", metrics)
        np.savez_compressed(output / "predictions.npz", uid=np.array([r.uid for r in records]),
            system_id=np.array([r.system_id for r in records]), target=target, ctot=ctot,
            individual_pressure=results["pressure"], ensemble_pressure=prediction, ensemble_spread=spread,
            eligibility=eligibility, eligible_model_count=counts, normalized_residual=(target - prediction) / ctot,
            mixture_latent=results["mixture_latent"], free_energy_excess=results["free_energy_excess"],
            pressure_excess=results["pressure_excess"], gradient=results["gradient"],
            **({"hessian": results["hessian"]} if "hessian" in results else {}))
        from .plotting import plot_pressure_evaluation, plot_selected_histories
        plot_pressure_evaluation(target, prediction, ctot, output)
        plot_selected_histories(run, ensemble.selection, output)
        summary.append(metrics)
        print(f'd={dimension} {args.split}: pressure_loss={metrics["ensemble"]["pressure_loss"]:.17g}', flush=True)
    write_json(run / "evaluation" / f"summary_{args.name}.json", summary)
    from .plotting import plot_dimension_losses
    plot_dimension_losses(summary, run / "evaluation" / f"loss_by_dimension_{args.name}.png")
