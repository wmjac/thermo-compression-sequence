"""Plot formatting never changes precision of archived numerical results."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_loss_history(training, validation, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = np.arange(1, len(training) + 1)
    ax.plot(epochs, training, label="Training")
    ax.plot(epochs, validation, label="Validation")
    if min(min(training), min(validation)) > 0:
        ax.set_yscale("log")
    ax.set(xlabel="Epoch", ylabel="Dimensionless pressure loss")
    ax.legend()
    finish(fig, path)


def plot_selected_histories(run, selection, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, item in enumerate(selection["models"]):
        with np.load(Path(run) / item["trial_path"] / "history.npz") as history:
            for axis, name in zip(axes, ("training", "validation")):
                values = history[name]
                axis.plot(np.arange(1, len(values) + 1), values, label=f'Model {index}, fold {item["fold"]}')
    for axis, title in zip(axes, ("Training", "Validation")):
        axis.set(xlabel="Epoch", ylabel="Dimensionless pressure loss", title=title)
        axis.set_yscale("log")
    axes[1].legend(fontsize="small")
    finish(fig, output / "selected_loss_curves.png")


def plot_pressure_evaluation(target, prediction, ctot, output):
    fig, ax = plt.subplots(figsize=(5, 5))
    measured, predicted = target / ctot, prediction / ctot
    ax.scatter(measured, predicted, s=8, alpha=0.5, rasterized=True)
    limits = [float(min(measured.min(), predicted.min())), float(max(measured.max(), predicted.max()))]
    ax.plot(limits, limits, color="black", lw=1)
    ax.set(xlabel="Measured P / (c_tot kT)", ylabel="Predicted P / (c_tot kT)")
    finish(fig, output / "pressure_parity.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(ctot, (target - prediction) / ctot, s=8, alpha=0.5, rasterized=True)
    ax.axhline(0, color="black", lw=1)
    ax.set(xlabel="Total chain concentration (nm⁻³)", ylabel="(P − predicted P) / (c_tot kT)")
    finish(fig, output / "residual_vs_concentration.png")


def plot_dimension_losses(summary, path):
    """Save linear loss, log loss, and log normalized RMSE versus dimension."""
    _plot_dimension_series({None: summary}, path)


def plot_dimension_comparison(summaries, path):
    """Plot named series, e.g. validation and test, in the same three formats."""
    _plot_dimension_series(summaries, path)


def _plot_dimension_series(summaries, path):
    path = Path(path)
    series = []
    for label, summary in summaries.items():
        ordered = sorted(summary, key=lambda item: item["dimension"])
        dimensions = np.asarray([item["dimension"] for item in ordered])
        losses = np.asarray([item["ensemble"]["pressure_loss"] for item in ordered], dtype=np.float64)
        if not len(losses) or np.any(~np.isfinite(losses)) or np.any(losses < 0):
            raise ValueError("Dimension plots require nonempty, finite, nonnegative pressure losses.")
        series.append((label, dimensions, losses))
    if not series:
        raise ValueError("Dimension plots require at least one series.")
    rmse_stem = "rmse" + path.stem[4:] if path.stem.startswith("loss") else path.stem + "_rmse"
    formats = ((False, False, path),
               (False, True, path.with_name(path.stem + "_log" + path.suffix)),
               (True, True, path.with_name(rmse_stem + "_log" + path.suffix)))
    for rmse, logarithmic, output in formats:
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, dimensions, losses in series:
            values = np.sqrt(losses) if rmse else losses
            # Zero cannot appear on a log axis. Mask it without imposing a floor
            # or modifying archived values; masked points also break line segments.
            if logarithmic:
                values = np.ma.masked_less_equal(values, 0)
            ax.plot(dimensions, values, marker="o", label=label)
        if logarithmic:
            ax.set_yscale("log")
            if not any(np.any(losses > 0) for _, _, losses in series):
                ax.set_ylim(1, 10)
            if any(np.any(losses == 0) for _, _, losses in series):
                ax.text(0.02, 0.98, "Zero values omitted on log scale", transform=ax.transAxes,
                        ha="left", va="top", fontsize="small")
        ax.set(xlabel="Latent dimension d",
               ylabel="Dimensionless pressure RMSE" if rmse else "Dimensionless pressure loss")
        if any(label is not None for label, _, _ in series):
            ax.legend()
        finish(fig, output)


def plot_canonical_diagnostics(convergence, aligned, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for member in range(convergence["running_frobenius"].shape[1]):
        axes[0].plot(np.arange(1, len(convergence["running_frobenius"]) + 1),
                     convergence["running_frobenius"][:, member], label=f"Model {member}")
    axes[0].set(xlabel="Monte Carlo batch", ylabel="Metric Frobenius norm")
    axes[0].legend(fontsize="small")
    axes[1].plot(np.arange(1, len(aligned["convergence"]) + 1), aligned["convergence"])
    axes[1].set(xlabel="Procrustes iteration", ylabel="Maximum rotation change")
    axes[1].set_yscale("log")
    finish(fig, output / "convergence.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(aligned["final_eigenvalues"]) + 1), aligned["final_eigenvalues"], marker="o")
    ax.set(xlabel="Canonical feature index", ylabel="Uncentered mean-feature variance")
    finish(fig, output / "feature_spectrum.png")
    if aligned["mean"].shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(5, 5))
        mean = aligned["mean"]
        ax.scatter(mean[:, 0], mean[:, 1], s=7, alpha=0.5, rasterized=True)
        ax.set(xlabel="Canonical feature 1", ylabel="Canonical feature 2")
        finish(fig, output / "canonical_features.png")
