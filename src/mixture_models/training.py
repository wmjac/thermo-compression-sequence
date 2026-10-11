"""Reusable training execution; the selected model supplies its protocol and hooks."""
from dataclasses import asdict
from functools import partial
from pathlib import Path
import shutil
import signal
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader
from .checkpoints import load_checkpoint, restore_rng, save_checkpoint, seed_everything
from .data import load_json_files, save_physical_dataset, sha256_file, validate_with_model
from .losses import pressure_loss
from .models import get_model_module
from .pressure import total_pressure
from .provenance import (collect_provenance, read_json, resolve_device, snapshot_source,
                         source_manifest, timestamp, write_json)
from .splits import make_folds


class StopRequested(Exception):
    pass


class EpochStop:
    def __init__(self):
        self.requested = False
        self.previous = {}

    def __enter__(self):
        def handler(signum, frame):
            self.requested = True
            print("Stop requested; finishing this epoch and saving continuation state.", flush=True)
        for signum in (signal.SIGTERM, signal.SIGINT):
            self.previous[signum] = signal.signal(signum, handler)
        return self

    def __exit__(self, *args):
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)


def move_inputs(inputs, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}


def aggregate_losses(losses, sizes, method):
    if method == "batch-mean":
        return float(np.mean(losses))
    if method == "sample-mean":
        return float(np.average(losses, weights=sizes))
    raise ValueError(f"Unsupported loss aggregation {method}.")


def make_loader(records, module, batch_size, protocol, args, shuffle, device):
    options = {"batch_size": batch_size, "shuffle": shuffle,
               "collate_fn": partial(module.collate_records, dtype=protocol.dtype),
               "num_workers": args.num_workers,
               "pin_memory": device.type == "cuda" if args.pin_memory == "auto" else args.pin_memory == "yes"}
    if args.num_workers:
        options.update(persistent_workers=args.persistent_workers == "yes", prefetch_factor=args.prefetch_factor)
    return DataLoader(records, **options)


def train_trial(directory, train_records, val_records, module, protocol, trial_config, args, device, stop):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", trial_config)
    seed_everything(trial_config["seed"])
    model = module.build_model(trial_config["dimension"], protocol.dtype).to(device)
    optimizer = module.build_optimizer(model, trial_config["learning_rate"], protocol)
    train_loader = make_loader(train_records, module, trial_config["batch_size"], protocol, args, True, device)
    val_loader = make_loader(val_records, module, trial_config["batch_size"], protocol, args, False, device)
    scheduler = module.build_scheduler(optimizer, len(train_loader), protocol)
    state = {"next_epoch": 0, "best_epoch": -1, "best_validation_loss": float("inf"),
             "patience_counter": 0, "train_history": [], "validation_history": [],
             "learning_rates": [], "stop_reason": None}
    if args.resume == "yes" and (directory / "last.pt").exists():
        saved = load_checkpoint(directory / "last.pt")
        if saved["config"] != trial_config:
            raise ValueError(f"Checkpoint configuration differs: {directory}")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        state = saved["training_state"]
        restore_rng(saved["rng"])
    for epoch in range(state["next_epoch"], protocol.max_epochs):
        model.train()
        losses, sizes = [], []
        state["learning_rates"].append(float(optimizer.param_groups[0]["lr"]))
        for inputs in train_loader:
            inputs = move_inputs(inputs, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = total_pressure(model, inputs)
            loss = pressure_loss(prediction, inputs["target"], inputs["ctot"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss in {directory}, epoch {epoch}.")
            losses.append(loss.item())
            sizes.append(len(inputs["target"]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=protocol.grad_clip)
            optimizer.step()
            scheduler.step()
        train_loss = aggregate_losses(losses, sizes, "batch-mean")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        losses, sizes = [], []
        try:
            with torch.set_grad_enabled(True):
                for inputs in val_loader:
                    inputs = move_inputs(inputs, device)
                    inputs["concentrations"].requires_grad_(True)
                    prediction = total_pressure(model, inputs)
                    loss = pressure_loss(prediction, inputs["target"], inputs["ctot"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite validation loss in {directory}, epoch {epoch}.")
                    losses.append(loss.item())
                    sizes.append(len(inputs["target"]))
        finally:
            for parameter in model.parameters():
                parameter.requires_grad_(True)
        validation_loss = aggregate_losses(losses, sizes, protocol.validation_aggregation)
        state["train_history"].append(train_loss)
        state["validation_history"].append(validation_loss)
        state["next_epoch"] = epoch + 1
        improved = validation_loss < state["best_validation_loss"] - protocol.min_delta
        if improved:
            state.update(best_epoch=epoch, best_validation_loss=validation_loss, patience_counter=0)
        else:
            state["patience_counter"] += 1
        finished = state["patience_counter"] >= protocol.patience or epoch + 1 >= protocol.max_epochs
        state["stop_reason"] = "patience" if state["patience_counter"] >= protocol.patience else (
            "max_epochs" if finished else None)
        if improved:
            save_checkpoint(directory / "best.pt", model, optimizer, scheduler, state, trial_config)
        if (epoch + 1) % args.checkpoint_every == 0 or finished or stop.requested:
            save_checkpoint(directory / "last.pt", model, optimizer, scheduler, state, trial_config)
            np.savez(directory / "history.npz", training=np.array(state["train_history"], dtype=np.float64),
                     validation=np.array(state["validation_history"], dtype=np.float64),
                     learning_rate=np.array(state["learning_rates"], dtype=np.float64))
        if epoch == 0 or (epoch + 1) % args.plot_every == 0 or finished or stop.requested:
            from .plotting import plot_loss_history
            plot_loss_history(state["train_history"], state["validation_history"], directory / "loss.png")
            print(f'{trial_config["trial_id"]}: epoch={epoch + 1} train={train_loss:.17g} '
                  f'validation={validation_loss:.17g} best={state["best_validation_loss"]:.17g}', flush=True)
        metrics = {"status": "complete" if finished else "running", "best_epoch": state["best_epoch"],
                   "best_validation_loss": state["best_validation_loss"], "epochs_completed": epoch + 1,
                   "stop_reason": state["stop_reason"], "updated_at": timestamp()}
        if stop.requested and not finished:
            metrics["status"] = "interrupted"
        write_json(directory / "metrics.json", metrics)
        if stop.requested:
            raise StopRequested()
        if finished:
            break


def run_training(args, protocol):
    module = get_model_module(args.model)
    device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(args.deterministic == "yes")
    records, manifests = load_json_files(args.training_json)
    validate_with_model(records, module)
    if args.num_models <= 0 or args.num_models % args.num_folds:
        raise ValueError("num_models must be a positive multiple of num_folds.")
    if len(protocol.batch_sizes) * len(protocol.learning_rates) * protocol.restarts < args.num_models // args.num_folds:
        raise ValueError("The hyperparameter/restart grid contains too few distinct candidates per fold.")
    run = Path(args.run_dir).expanduser().resolve()
    protocol_values = asdict(protocol)
    protocol_values["batch_sizes"] = list(protocol.batch_sizes)
    protocol_values["learning_rates"] = list(protocol.learning_rates)
    config = {"model_module": module.__name__, "model_metadata": module.model_metadata(),
              "dimensions": args.dimensions, "num_folds": args.num_folds, "num_models": args.num_models,
              "seed": args.seed, "protocol": protocol_values,
              "runtime": {key: getattr(args, key) for key in ("device", "num_workers", "pin_memory",
                  "persistent_workers", "prefetch_factor", "threads", "deterministic", "checkpoint_every", "plot_every")},
              "training_inputs": [{"sha256": item["sha256"], "num_systems": item["num_systems"]} for item in manifests]}
    if (run / "config.json").exists():
        if args.resume != "yes":
            raise ValueError(f"Run already exists: {run}; use --resume yes to continue it.")
        if read_json(run / "config.json") != config:
            raise ValueError("Resumption requires the same data, model, protocol, dimensions, and runtime configuration.")
        if read_json(run / "provenance.json")["source"]["sha256"] != source_manifest()["sha256"]:
            raise ValueError("Code differs from the recorded run; use its source_snapshot to resume.")
        folds = read_json(run / "folds.json")
        if folds["uid"] != [record.uid for record in records]:
            raise ValueError("Saved fold assignments disagree with the supplied dataset.")
    else:
        if args.resume == "yes":
            raise ValueError(f"Cannot resume a run without config.json: {run}")
        if run.exists() and any(run.iterdir()):
            raise ValueError(f"New run directory is not empty: {run}")
        run.mkdir(parents=True, exist_ok=True)
        folds = make_folds(records, args.num_folds, args.seed)
        write_json(run / "config.json", config)
        write_json(run / "provenance.json", collect_provenance(device))
        write_json(run / "folds.json", folds)
        snapshot_source(run / "source_snapshot" / "mixture_models")
        (run / "inputs").mkdir()
        for index, item in enumerate(manifests):
            target = run / "inputs" / f'{index:03d}_{Path(item["path"]).name}'
            shutil.copy2(item["path"], target)
            item["snapshot"] = str(target.relative_to(run))
        write_json(run / "inputs.json", manifests)
        save_physical_dataset(run / "dataset.npz", records)
    invocation = run / "invocations"
    invocation.mkdir(exist_ok=True)
    write_json(invocation / f'{len(list(invocation.glob("*.json"))):04d}.json',
               {"time": timestamp(), "arguments": vars(args), "provenance": collect_provenance(device)})
    assignments = np.asarray(folds["validation_fold"])
    with EpochStop() as stop:
        for dimension in args.dimensions:
            for fold in range(args.num_folds):
                train_records = [record for index, record in enumerate(records) if assignments[index] != fold]
                val_records = [record for index, record in enumerate(records) if assignments[index] == fold]
                for batch_index, batch_size in enumerate(protocol.batch_sizes):
                    for lr_index, learning_rate in enumerate(protocol.learning_rates):
                        for restart in range(protocol.restarts):
                            trial_id = f'd_{dimension:03d}/fold_{fold:02d}/b{batch_index:02d}_lr{lr_index:02d}_r{restart:03d}'
                            directory = run / "trials" / trial_id
                            if (directory / "metrics.json").exists() and read_json(directory / "metrics.json")["status"] == "complete":
                                continue
                            seed = int(np.random.SeedSequence([args.seed, dimension, fold, batch_index,
                                                              lr_index, restart]).generate_state(1)[0])
                            trial_config = {"trial_id": trial_id, "model_module": module.__name__,
                                            "model_metadata": module.model_metadata(), "dimension": dimension,
                                            "fold": fold, "batch_size": batch_size, "learning_rate": learning_rate,
                                            "restart": restart, "seed": seed, "protocol": protocol_values}
                            try:
                                train_trial(directory, train_records, val_records, module, protocol, trial_config, args, device, stop)
                            except StopRequested:
                                print(f"Saved continuation state under {directory}. Resume with --resume yes.", flush=True)
                                return False
                            except Exception as exc:
                                write_json(directory / "failure.json", {"time": timestamp(), "error": str(exc),
                                                                      "traceback": traceback.format_exc()})
                                raise
    from .selection import select_models
    select_models(run, args.dimensions, args.num_models)
    return True
