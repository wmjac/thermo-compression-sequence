"""Uncentered, rho_chem-weighted geometry; only latent-size square matrices."""
import numpy as np


def gram(x, y, weights):
    return x.T @ (weights[:, None] * y)


def eigensystem(x, weights, rtol, atol):
    matrix = gram(x, x, weights)
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    values, vectors = values[::-1], vectors[:, ::-1]
    if values.min(initial=0) < -max(atol, rtol * max(values.max(initial=0), 1e-300)):
        raise ValueError("Feature Gram matrix has a significant negative eigenvalue.")
    values = np.maximum(values, 0.0)
    keep = values > max(atol, rtol * values.max(initial=0))
    return values, vectors, keep


def basis(x, weights, rtol, atol):
    values, vectors, keep = eigensystem(x, weights, rtol, atol)
    return (np.sqrt(weights)[:, None] * x) @ (vectors[:, keep] / np.sqrt(values[keep])), values, vectors, keep


def angles_from_overlap(overlap):
    singular = np.clip(np.linalg.svd(overlap, compute_uv=False), 0.0, 1.0)
    return np.degrees(np.arccos(singular)), singular


def subspace_comparison(x, y, weights, rtol, atol):
    qx, *_ = basis(x, weights, rtol, atol)
    qy, *_ = basis(y, weights, rtol, atol)
    overlap = qx.T @ qy
    angles, singular = angles_from_overlap(overlap)
    mass = float(np.sum(singular**2))
    return {"rank_left": qx.shape[1], "rank_right": qy.shape[1],
            "overlap": mass, "left_fraction": mass / qx.shape[1] if qx.shape[1] else None,
            "right_fraction": mass / qy.shape[1] if qy.shape[1] else None,
            "max_angle_deg": float(angles.max()) if len(angles) else None,
            "mean_angle_deg": float(angles.mean()) if len(angles) else None}, angles, singular


def nonnegative(value, scale):
    if value < -1e-9 * max(scale, 1e-300):
        raise ValueError(f"Significantly negative squared residual: {value}")
    return float(max(value, 0.0))


def procrustes(x, y, weights):
    """Rectangular orthogonal embedding; permits reflections, no shift or scale."""
    xx, yy, xy = gram(x, x, weights), gram(y, y, weights), gram(x, y, weights)
    total = float(np.trace(xx) + np.trace(yy))
    nuclear = float(np.linalg.svd(xy, compute_uv=False).sum())
    after = np.sqrt(nonnegative(total - 2 * nuclear, total))
    before = np.sqrt(nonnegative(total - 2 * np.trace(xy), total))
    return float(before), float(after)


def within_spread(features, weights):
    mean = features.mean(axis=0)
    squared = np.sum((features - mean[None])**2, axis=-1)
    return mean, float(np.sqrt(np.mean(squared @ weights)))


def eigenblocks(values, rtol):
    groups, current = [], []
    for i, value in enumerate(values):
        if current and values[current[0]] - value > rtol * max(values[current[0]], 1e-300):
            groups.append(current)
            current = []
        current.append(i)
    if current:
        groups.append(current)
    return groups


def mixing_statistics(loadings, blocks, leading):
    """One row per low-model axis; both mode-level and block-level mixing."""
    rows = []
    for axis in range(loadings.shape[1]):
        squares = np.maximum(loadings[:, axis]**2, 0)
        total = float(squares.sum())
        probabilities = squares / total if total > 0 else squares
        block_weights = np.array([probabilities[group].sum() for group in blocks])
        rows.append({"axis": axis + 1, "reference_span_overlap": total,
            "unexplained_fraction": float(max(1 - total, 0)),
            "leading_d_overlap": float(squares[:leading].sum()),
            "tail_fraction_within_reference": float(squares[leading:].sum() / total) if total else None,
            "effective_modes": float(1 / np.sum(probabilities**2)) if total else None,
            "effective_blocks": float(1 / np.sum(block_weights**2)) if total else None,
            "mode_entropy": float(-np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0]))) if total else None})
    return rows
