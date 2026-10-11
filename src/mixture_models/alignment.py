"""Uncentered PCA and generalized orthogonal Procrustes, following idpmixture."""
import numpy as np


def pca_with_unique_signs(z):
    covariance = (z.T @ z) / len(z)
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    rotated = z @ vectors
    signs = np.where(rotated.mean(axis=0) >= 0.0, 1.0, -1.0)
    return rotated * signs, vectors * signs, values, signs


def align_ensemble(native, square_roots, max_iter=200, tolerance=1e-8):
    k, n, dimension = native.shape
    transformed = np.einsum("knd,ked->kne", native, square_roots)
    pca, pca_rotations, pca_values, pca_signs = [], [], [], []
    for z in transformed:
        rotated, vectors, values, signs = pca_with_unique_signs(z)
        pca.append(rotated)
        pca_rotations.append(vectors)
        pca_values.append(values)
        pca_signs.append(signs)
    pca = np.asarray(pca)
    rotations = np.repeat(np.eye(dimension)[None], k, axis=0)
    aligned = pca.copy()
    convergence = []
    converged = False
    for iteration in range(max_iter):
        mean = aligned.mean(axis=0)
        updated = np.empty_like(rotations)
        for member in range(k):
            u, _, vt = np.linalg.svd(pca[member].T @ mean, full_matrices=False)
            updated[member] = u @ vt
        delta = float(np.max(np.linalg.norm(updated - rotations, axis=(1, 2))))
        rotations = updated
        aligned = np.einsum("knd,kde->kne", pca, rotations)
        convergence.append(delta)
        if delta < tolerance:
            converged = True
            break
    mean = aligned.mean(axis=0)
    _, final_rotation, final_values, final_signs = pca_with_unique_signs(mean)
    final_features = aligned @ final_rotation
    total_rotations = np.asarray(pca_rotations) @ rotations @ final_rotation
    transforms = np.transpose(total_rotations, (0, 2, 1)) @ square_roots
    return {"aligned": final_features, "mean": final_features.mean(axis=0),
            "pca_features": pca, "transforms": transforms,
            "pca_rotations": np.asarray(pca_rotations), "procrustes_rotations": rotations,
            "final_rotation": final_rotation, "pca_eigenvalues": np.asarray(pca_values),
            "pca_signs": np.asarray(pca_signs), "final_eigenvalues": final_values,
            "final_signs": final_signs, "convergence": np.asarray(convergence),
            "iterations": len(convergence), "converged": converged}
