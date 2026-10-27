"""Common-mixture thermodynamic comparisons, without a species-square kernel."""
from pathlib import Path
import numpy as np
from _post_training_common_v1 import models_for
from _geometry_v1 import gram, eigensystem, nonnegative


def canonical_samples(reference):
    with np.load(reference.canonical / "samples.npz", allow_pickle=False) as data:
        samples = {name: data[name] for name in data.files}
    n = len(samples["ctot"])
    offsets, indices, masses = samples["offsets"], samples["indices"], samples["weights"]
    if (offsets.shape != (n + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) <= 0)
            or offsets[-1] != len(indices) or indices.shape != masses.shape
            or np.any(indices < 0) or np.any(indices >= len(reference.sequences))
            or not np.all(np.isfinite(masses)) or np.any(masses <= 0)
            or not np.all(np.isfinite(samples["ctot"])) or np.any(samples["ctot"] < 0)):
        raise ValueError("Invalid archived canonical mixture samples.")
    totals = np.add.reduceat(masses, offsets[:-1])
    if not np.allclose(totals, 1.0, rtol=1e-12, atol=1e-14):
        raise ValueError("Archived composition masses do not sum to one.")
    with np.load(reference.canonical / "metric.npz", allow_pickle=False) as metric:
        batch_sizes = metric["batch_sizes"].astype(int)
    if batch_sizes.sum() != n or np.any(batch_sizes <= 0):
        raise ValueError("Canonical metric batches disagree with samples.")
    aggregation = reference.config["arguments"]["metric_aggregation"]
    if aggregation == "batch-mean":
        weights = np.concatenate([np.full(size, 1.0 / (len(batch_sizes) * size)) for size in batch_sizes])
    elif aggregation == "sample-mean":
        weights = np.full(n, 1.0 / n)
    else:
        raise ValueError("Unsupported canonical mixture aggregation.")
    samples["quadrature_weights"] = weights
    samples["canonical_batch_sizes"] = batch_sizes
    return samples


def mixture_latent(native, samples, start, stop):
    offsets = samples["offsets"]
    left, right = int(offsets[start]), int(offsets[stop])
    row = np.repeat(np.arange(stop - start), np.diff(offsets[start:stop + 1]))
    out = np.zeros((stop - start, native.shape[1]), dtype=np.float64)
    np.add.at(out, row, native[samples["indices"][left:right]] * samples["weights"][left:right, None])
    return out * samples["ctot"][start:stop, None]


def evaluate_gradients(saved, samples, device, batch_size, output=None):
    import torch
    shape = (len(saved.selection["models"]), len(samples["ctot"]), saved.dimension)
    gradients = np.lib.format.open_memmap(output, mode="w+", dtype=np.float64, shape=shape) if output else np.empty(shape)
    for member, (model, _, _) in enumerate(models_for(saved, device)):
        parameter = next(model.parameters())
        for start in range(0, shape[1], batch_size):
            stop = min(start + batch_size, shape[1])
            x = mixture_latent(saved.native[member], samples, start, stop)
            tensor = torch.as_tensor(x, dtype=parameter.dtype, device=device)
            with torch.enable_grad():
                gradient = model.thermodynamics(tensor)["gradient"].detach().cpu().numpy().astype(np.float64)
            gradients[member, start:stop] = gradient @ saved.inverses[member]
        if not np.all(np.isfinite(gradients[member])):
            raise FloatingPointError("Nonfinite decoder gradients.")
        print(f"d={saved.dimension}: common-mixture responses for model {member + 1}/{shape[0]}", flush=True)
        del model
    if isinstance(gradients, np.memmap):
        gradients.flush()
    return gradients


def response_eigensystem(features, gradients, chemical_weights, mixture_weights, rtol, atol):
    """SVD of sqrt(Wmix) G Z.T sqrt(Wchem), using its low-rank factors."""
    covariance = gram(gradients, gradients, mixture_weights)
    values_g, vectors_g = np.linalg.eigh(0.5 * (covariance + covariance.T))
    root_g = (vectors_g * np.sqrt(np.maximum(values_g, 0.0))) @ vectors_g.T
    metric_features = features @ root_g
    values, vectors, keep = eigensystem(metric_features, chemical_weights, rtol, atol)
    chemical_gram = gram(features, features, chemical_weights)
    # h_alpha = G @ coefficients; valid also when covariance is rank deficient.
    coefficients = chemical_gram @ root_g @ vectors[:, keep] / values[keep]
    return values, coefficients, covariance, metric_features, vectors, keep


def response_pair(reference_features, low_features, reference_gradients, low_gradients,
                  chemical_weights, mixture_weights, rtol, atol):
    cr, cl = gram(reference_features, reference_features, chemical_weights), gram(low_features, low_features, chemical_weights)
    cross_z = gram(reference_features, low_features, chemical_weights)
    gr = gram(reference_gradients, reference_gradients, mixture_weights)
    gl = gram(low_gradients, low_gradients, mixture_weights)
    cross_g = gram(reference_gradients, low_gradients, mixture_weights)
    magnitude_r, magnitude_l = float(np.sum(cr * gr)), float(np.sum(cl * gl))
    raw_loss = magnitude_r + magnitude_l - 2 * float(np.sum(cross_z * cross_g))
    loss = nonnegative(raw_loss, magnitude_r + magnitude_l)
    values, coefficients, _, _, _, keep = response_eigensystem(reference_features, reference_gradients,
        chemical_weights, mixture_weights, rtol, atol)
    low_values, low_vectors, low_keep = eigensystem(low_gradients, mixture_weights, rtol, atol)
    whitening_low = low_vectors[:, low_keep] / np.sqrt(low_values[low_keep])
    overlap = coefficients.T @ cross_g @ whitening_low
    retention = np.clip(np.sum(overlap**2, axis=1), 0.0, 1.0)
    # Include all computed reference eigenvalues in the tail. Numerical-rank
    # truncation affects identifiable mode directions, not the saved spectrum.
    retained = float(np.dot(values[keep], retention))
    projection_residual = nonnegative(magnitude_r - retained, magnitude_r)
    nominal_d = low_features.shape[1]
    bound = float(values[nominal_d:].sum())
    if loss + 1e-7 * max(magnitude_r + magnitude_l, 1e-300) < projection_residual:
        raise FloatingPointError("TM loss is smaller than its function-subspace projection residual.")
    return {"tm_loss": loss, "tm_loss_raw": raw_loss, "tm_rmse": float(np.sqrt(loss)),
            "reference_magnitude": magnitude_r, "low_model_magnitude": magnitude_l,
            "relative_tm_loss": loss / magnitude_r if magnitude_r else None,
            "eym_tail_bound": bound, "function_projection_residual": projection_residual,
            "loss_minus_eym_bound": loss - bound, "response_rank_low": int(low_keep.sum())}, overlap, retention, values
