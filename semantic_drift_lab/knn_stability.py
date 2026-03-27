"""
kNN Stability — шаг 3 пайплайна DEMON.

Измеряет устойчивость окрестности каждой точки: если точка является
частью аттрактора, её k ближайших соседей остаются теми же при малых
возмущениях входных данных. Шумовые точки меняют соседей хаотично.

Возвращает stability score ∈ [0, 1] для каждой точки:
  1.0 — полностью устойчивый аттрактор
  0.0 — нестабильный шум / переходная зона
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_stability(
    X: np.ndarray,
    k: int | None = None,
    epsilon: float = 0.05,
    n_perturbations: int = 20,
    random_state: int = 42,
) -> np.ndarray:
    """
    Вычисляет stability score для каждой точки в X.

    Parameters
    ----------
    X               : (n_samples, n_features)
    k               : число соседей; по умолчанию min(5, n//3)
    epsilon         : амплитуда возмущения (доля от std данных)
    n_perturbations : число повторений для усреднения
    random_state    : воспроизводимость

    Returns
    -------
    stability : np.ndarray, shape (n_samples,), значения в [0, 1]
    """
    rng = np.random.default_rng(random_state)
    n = len(X)

    if k is None:
        k = max(2, min(5, n // 3))

    # k не может быть >= n
    k = min(k, n - 1)

    nn = NearestNeighbors(n_neighbors=k + 1)  # +1 потому что точка — сама себе сосед
    nn.fit(X)
    _, base_idx = nn.kneighbors(X)
    base_neighbors = [set(row[1:]) for row in base_idx]  # исключаем саму точку

    scale = np.std(X, axis=0)
    scale[scale == 0] = 1.0  # защита от нулевого std

    stability = np.zeros(n)

    for _ in range(n_perturbations):
        noise = rng.normal(0, epsilon, X.shape) * scale
        X_perturbed = X + noise

        _, pert_idx = nn.kneighbors(X_perturbed)
        pert_neighbors = [set(row[1:]) for row in pert_idx]

        for i in range(n):
            overlap = len(base_neighbors[i] & pert_neighbors[i])
            stability[i] += overlap / k

    return stability / n_perturbations


def attractor_mask(stability: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Булева маска точек-аттракторов с stability >= threshold."""
    return stability >= threshold
