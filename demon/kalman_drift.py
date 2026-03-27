"""Совместимый shim для исторического модуля demon.kalman_drift."""

from semantic_drift_lab.kalman_drift import KalmanResult, KalmanDrift

__all__ = ["KalmanResult", "KalmanDrift"]
