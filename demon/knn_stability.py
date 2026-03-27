"""Совместимый shim для исторического модуля demon.knn_stability."""

from semantic_drift_lab.knn_stability import knn_stability, attractor_mask

__all__ = ["knn_stability", "attractor_mask"]
