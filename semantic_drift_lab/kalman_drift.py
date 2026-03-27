"""
Kalman Drift Detector — шаг 4 пайплайна Semantic Drift Lab.

Реализует 1D Kalman filter на чистом numpy (без внешних зависимостей).
Сглаживает числовой ряд и детектирует дрейф (систематическое отклонение
от предсказания), позволяя отслеживать изменение сигнала во времени
с учётом неопределённости наблюдений.

Может использоваться для отслеживания аномальных отклонений в числовом
сигнале до того, как они становятся очевидными визуально.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class KalmanResult:
    smoothed: np.ndarray       # сглаженный ряд
    innovations: np.ndarray    # остатки (z_k - H*x_hat_k)
    variances: np.ndarray      # апостериорные дисперсии
    drift_flags: np.ndarray    # True там, где |innovation| > threshold
    drift_score: float         # доля точек с дрейфом


class KalmanDrift:
    """
    Скалярный Kalman filter для обнаружения дрейфа в 1D временном ряду.

    Модель:
      x_k = x_{k-1} + w_k,   w_k ~ N(0, Q)   — случайное блуждание
      z_k = x_k + v_k,       v_k ~ N(0, R)   — зашумлённое наблюдение
    """

    def __init__(
        self,
        process_noise: float = 0.1,   # Q — дисперсия процесса
        measurement_noise: float = 1.0,  # R — дисперсия измерений
        drift_sigma: float = 2.0,     # порог: N сигм для флага дрейфа
    ):
        self.Q = process_noise
        self.R = measurement_noise
        self.drift_sigma = drift_sigma

    def filter(self, observations: np.ndarray | list) -> KalmanResult:
        """
        Применяет фильтр Калмана к ряду наблюдений.

        Returns KalmanResult с полем drift_flags: True там, где
        инновация (реальное − предсказанное) превышает drift_sigma σ.
        """
        z = np.asarray(observations, dtype=float)
        n = len(z)

        x_hat = np.zeros(n)   # апостериорная оценка состояния
        P = np.zeros(n)       # апостериорная дисперсия
        innovations = np.zeros(n)

        # Инициализация
        x_hat[0] = z[0]
        P[0] = self.R

        for k in range(1, n):
            # --- Predict ---
            x_pred = x_hat[k - 1]
            P_pred = P[k - 1] + self.Q

            # --- Update ---
            innov = z[k] - x_pred
            S = P_pred + self.R                    # инновационная дисперсия
            K = P_pred / S                         # калмановское усиление

            x_hat[k] = x_pred + K * innov
            P[k] = (1 - K) * P_pred
            innovations[k] = innov

        # Порог дрейфа: среднее ± drift_sigma * std инноваций
        innov_std = float(np.std(innovations[1:]))  # первый элемент = 0 (инициализация)
        threshold = self.drift_sigma * innov_std if innov_std > 0 else 1e-9
        drift_flags = np.abs(innovations) > threshold
        drift_flags[0] = False  # первая точка — инициализация, не флагуем

        return KalmanResult(
            smoothed=x_hat,
            innovations=innovations,
            variances=P,
            drift_flags=drift_flags,
            drift_score=float(drift_flags.sum() / max(n - 1, 1)),
        )
