"""Population spreads of selected models; no IID/SEM assumption."""
from pathlib import Path
import numpy as np


def analysis_provenance(args):
    from _post_training_common_v1 import provenance, sha256
    result = provenance(args)
    result["analysis_source_sha256"].update({
        path.name: sha256(path)
        for path in sorted(Path(__file__).parent.glob("*_v2.py"))
    })
    return result


def feature_statistics(features):
    """Input (models, sequences, dimensions), in saved aligned coordinates."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("Expected finite nonempty (models,sequences,dimensions) features.")
    mean = values.mean(axis=0)
    squared = (values - mean[None]) ** 2
    std = np.sqrt(np.mean(squared, axis=0))
    # Preserve the v1 summation order and definition exactly.
    rmsd = np.sqrt(np.mean(np.sum(squared, axis=-1), axis=0))
    return mean, rmsd, std


def test_loss_statistics(target, individual_pressure, ctot):
    """Every member is evaluated on the same independent test systems."""
    target = np.asarray(target, dtype=np.float64)
    pressure = np.asarray(individual_pressure, dtype=np.float64)
    ctot = np.asarray(ctot, dtype=np.float64)
    if (target.ndim != 1 or not target.size or ctot.shape != target.shape
            or pressure.ndim != 2 or pressure.shape[1] != len(target)
            or pressure.shape[0] < 1):
        raise ValueError("Expected target/ctot (systems,) and pressure (models,systems).")
    if not (np.isfinite(target).all() and np.isfinite(pressure).all()
            and np.isfinite(ctot).all() and np.all(ctot > 0)):
        raise ValueError("Test data/predictions must be finite and concentrations positive.")
    member_losses = np.mean(((target[None] - pressure) / ctot[None]) ** 2, axis=1)
    member_rmses = np.sqrt(member_losses)
    ensemble = pressure.mean(axis=0)
    ensemble_loss = float(np.mean(((target - ensemble) / ctot) ** 2))
    normalized_variance = float(np.mean(np.var(pressure / ctot[None], axis=0, ddof=0)))
    row = {
        "num_models": int(pressure.shape[0]),
        "num_systems": int(target.size),
        "ensemble_pressure_loss": ensemble_loss,
        "ensemble_normalized_rmse": float(np.sqrt(ensemble_loss)),
        "member_pressure_loss_mean": float(member_losses.mean()),
        "member_pressure_loss_std": float(member_losses.std(ddof=0)),
        "member_normalized_rmse_mean": float(member_rmses.mean()),
        "member_normalized_rmse_std": float(member_rmses.std(ddof=0)),
        "mean_normalized_prediction_variance": normalized_variance,
    }
    if not all(np.isfinite(value) for value in row.values()):
        raise FloatingPointError("Nonfinite pressure-loss statistics.")
    return row, member_losses, member_rmses


def plot_test_losses(summary, output):
    """Member mean +/- SD and ensemble-prediction loss are distinct curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ordered = sorted(summary, key=lambda row: row["dimension"])
    dims = np.array([row["dimension"] for row in ordered])
    for rmse, logarithmic, name in (
        (False, False, "loss_by_dimension_test.png"),
        (False, True, "loss_by_dimension_test_log.png"),
        (True, True, "rmse_by_dimension_test_log.png"),
    ):
        stem = "normalized_rmse" if rmse else "pressure_loss"
        mean = np.array([row[f"member_{stem}_mean"] for row in ordered])
        std = np.array([row[f"member_{stem}_std"] for row in ordered])
        ensemble = np.array([row[f"ensemble_{stem}"] for row in ordered])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        valid = mean > 0 if logarithmic else np.ones(len(mean), dtype=bool)
        lower_omitted = (mean - std <= 0) & valid if logarithmic else np.zeros(len(mean), dtype=bool)
        lower_error = np.where(lower_omitted, 0.0, std)
        ax.errorbar(dims[valid], mean[valid], yerr=np.array([lower_error[valid], std[valid]]),
                    fmt="o-", capsize=3, label="Individual-model mean +/- population SD")
        ev = ensemble > 0 if logarithmic else np.ones(len(ensemble), dtype=bool)
        ax.plot(dims[ev], ensemble[ev], "s--", label="Loss of ensemble-mean prediction" if not rmse
                else "RMSE of ensemble-mean prediction")
        annotations = []
        if logarithmic:
            ax.set_yscale("log")
            if not (valid.any() or ev.any()):
                ax.set_ylim(0.1, 1.0)
            if (~valid).any() or (~ev).any():
                annotations.append("Zero values omitted on log scale")
            if lower_omitted.any():
                annotations.append("Nonpositive lower SD endpoints omitted")
        if annotations:
            ax.text(.02, .98, "\n".join(annotations), transform=ax.transAxes, va="top", fontsize=8)
        ax.set(xlabel="Latent dimension d",
               ylabel="Dimensionless normalized pressure RMSE" if rmse else "Dimensionless pressure loss",
               xticks=dims)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(Path(output) / name, dpi=180)
        plt.close(fig)
