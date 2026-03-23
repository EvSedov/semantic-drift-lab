"""
Takens Embedding — шаг 2 пайплайна DEMON.

Реализует теорему Такенса (1981): из скалярного временного ряда x(t)
строит delay-векторы [x(t), x(t-τ), ..., x(t-(m-1)τ)], которые
топологически эквивалентны исходному фазовому пространству системы.

В контексте PAI: ряд качества сессий (implied_sentiment) раскрывает
скрытую динамику системы — аттракторы и нестабильные режимы.
"""
import numpy as np


def takens_embedding(
    time_series: np.ndarray | list,
    delay: int = 1,
    embedding_dim: int = 3,
) -> np.ndarray:
    """
    Строит delay embedding из 1D временного ряда.

    Parameters
    ----------
    time_series : array-like, shape (N,)
    delay       : τ — шаг задержки между координатами
    embedding_dim : m — размерность вложения

    Returns
    -------
    embedded : np.ndarray, shape (N - (m-1)*τ, m)
        Каждая строка — вектор состояния системы в момент t.
    """
    ts = np.asarray(time_series, dtype=float)
    n = len(ts)
    n_vectors = n - (embedding_dim - 1) * delay

    if n_vectors <= 0:
        raise ValueError(
            f"Временной ряд слишком короткий: n={n}, "
            f"embedding_dim={embedding_dim}, delay={delay}. "
            f"Нужно минимум {(embedding_dim - 1) * delay + 1} точек."
        )

    embedded = np.zeros((n_vectors, embedding_dim))
    for i in range(n_vectors):
        for j in range(embedding_dim):
            embedded[i, j] = ts[i + j * delay]

    return embedded


def optimal_delay(time_series: np.ndarray | list, max_delay: int = 10) -> int:
    """
    Эвристика выбора τ через первый минимум автокорреляции.
    При отсутствии явного минимума возвращает 1.
    """
    ts = np.asarray(time_series, dtype=float)
    ts = ts - ts.mean()

    for tau in range(1, min(max_delay + 1, len(ts) // 2)):
        acf = float(np.correlate(ts[:-tau], ts[tau:])[0] / (np.std(ts) ** 2 * len(ts)))
        if acf <= 0:
            return tau

    return 1
