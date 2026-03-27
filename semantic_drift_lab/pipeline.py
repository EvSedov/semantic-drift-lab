"""
Semantic Drift Pipeline — объединяет аналитические шаги в единый
детерминированный пайплайн.

Поток данных:
  records → [SVD embed] → [kNN stability] → похожие записи
          → [Takens embed] → [Kalman drift] → детектор дрейфа сигнала
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .svd_embed import SVDEmbedder
from .takens_embed import takens_embedding, optimal_delay
from .knn_stability import knn_stability, attractor_mask
from .kalman_drift import KalmanDrift, KalmanResult


@dataclass
class CorpusRecord:
    idx: int
    label: str
    text: str
    signal: float = 0.0
    kind: str = "generic"
    meta: dict = field(default_factory=dict)

    @property
    def task(self) -> str:
        return str(self.meta.get("task", self.label))

    @property
    def effort(self) -> str:
        return str(self.meta.get("effort", ""))

    @property
    def sentiment(self) -> float:
        return float(self.signal)

    @property
    def criteria_count(self) -> int:
        return int(self.meta.get("criteria_count", 0))

    @property
    def criteria_passed(self) -> int:
        return int(self.meta.get("criteria_passed", 0))

    @property
    def within_budget(self) -> bool:
        return bool(self.meta.get("within_budget", True))


# Совместимость со старым именем модели
SessionRecord = CorpusRecord


@dataclass
class SimilarSession:
    idx: int
    task: str
    cosine_sim: float
    stability: float


@dataclass
class PipelineResult:
    records: list[CorpusRecord]
    embeddings: np.ndarray              # (n, svd_components)
    stability_scores: np.ndarray        # (n,) — kNN stability
    attractor_indices: list[int]        # индексы устойчивых точек
    similar_sessions: dict[int, list[SimilarSession]]  # idx → топ-K похожих
    kalman: KalmanResult                # результат Kalman на ряду sentiment
    takens_embedded: np.ndarray         # delay embedding sentiment-ряда
    svd_explained_variance: float
    n_attractors: int
    embedder: SVDEmbedder = field(default=None, repr=False)  # для find_similar


class DemonPipeline:
    """Детерминированный пайплайн для анализа текстового корпуса и сигнала."""

    def __init__(
        self,
        svd_components: int = 8,
        knn_k: int | None = None,
        knn_epsilon: float = 0.05,
        takens_delay: int | None = None,
        takens_dim: int = 3,
        kalman_Q: float = 0.1,
        kalman_R: float = 1.0,
        top_k_similar: int = 3,
    ):
        self.svd_components = svd_components
        self.knn_k = knn_k
        self.knn_epsilon = knn_epsilon
        self.takens_delay = takens_delay
        self.takens_dim = takens_dim
        self.kalman_Q = kalman_Q
        self.kalman_R = kalman_R
        self.top_k_similar = top_k_similar

    def load_jsonl(self, path: str | Path) -> list[CorpusRecord]:
        """
        Совместимый загрузчик старого PAI JSONL.
        Новый код лучше строить вокруг adapters.load_pai_jsonl.
        """
        from .adapters.pai_jsonl import load_pai_jsonl

        return load_pai_jsonl(path)

    def run(self, jsonl_path: str | Path) -> PipelineResult:
        records = self.load_jsonl(jsonl_path)
        return self.run_records(records)

    def run_records(self, records: list[CorpusRecord]) -> PipelineResult:
        if not records:
            raise ValueError("Пустой корпус: для анализа нужна хотя бы одна запись.")

        texts = [r.text for r in records]
        signals = np.array([r.signal for r in records], dtype=float)

        # ── Шаг 1: SVD embedding ──
        embedder = SVDEmbedder(n_components=self.svd_components)
        embeddings = embedder.fit_transform(texts)

        # ── Шаг 2: kNN Stability ──
        stability = knn_stability(embeddings, k=self.knn_k, epsilon=self.knn_epsilon)
        attractors = [i for i, s in enumerate(stability) if attractor_mask(np.array([s]))[0]]

        # ── Шаг 3: Похожие сессии через косинусное сходство ──
        # embeddings уже L2-нормализованы → dot product = cosine similarity
        sim_matrix = embeddings @ embeddings.T
        np.fill_diagonal(sim_matrix, -1)  # исключаем самосходство

        similar: dict[int, list[SimilarSession]] = {}
        for i, _rec in enumerate(records):
            top_idx = np.argsort(sim_matrix[i])[::-1][: self.top_k_similar]
            similar[i] = [
                SimilarSession(
                    idx=int(j),
                    task=records[j].task,
                    cosine_sim=float(sim_matrix[i, j]),
                    stability=float(stability[j]),
                )
                for j in top_idx
            ]

        # ── Шаг 4: Takens Embedding на сигнальном ряду ──
        if len(signals) >= 4:
            tau = self.takens_delay or optimal_delay(signals)
            # Убеждаемся что dim/delay дают хотя бы 2 вектора
            max_dim = max(2, (len(signals) - 1) // max(tau, 1))
            dim = min(self.takens_dim, max_dim)
            takens = takens_embedding(signals, delay=tau, embedding_dim=dim)
        else:
            takens = signals.reshape(-1, 1)

        # ── Шаг 5: Kalman Drift Detection ──
        kalman = KalmanDrift(process_noise=self.kalman_Q, measurement_noise=self.kalman_R)
        kalman_result = kalman.filter(signals)

        return PipelineResult(
            records=records,
            embeddings=embeddings,
            stability_scores=stability,
            attractor_indices=attractors,
            similar_sessions=similar,
            kalman=kalman_result,
            takens_embedded=takens,
            svd_explained_variance=embedder.explained_variance_ratio_,
            n_attractors=len(attractors),
            embedder=embedder,
        )

    def find_similar(
        self,
        query: str,
        result: PipelineResult,
        top_k: int | None = None,
    ) -> list[SimilarSession]:
        """
        Найти сессии из result, наиболее похожие на произвольный текст query.
        Использует уже обученный embedder из PipelineResult.
        """
        k = top_k or self.top_k_similar
        query_vec = result.embedder.transform([query])  # (1, n_components)
        sims = (result.embeddings @ query_vec.T).flatten()  # cosine similarity

        top_idx = np.argsort(sims)[::-1][:k]
        return [
            SimilarSession(
                idx=int(i),
                task=result.records[i].task,
                cosine_sim=float(sims[i]),
                stability=float(result.stability_scores[i]),
            )
            for i in top_idx
        ]


SemanticDriftPipeline = DemonPipeline

__all__ = [
    "CorpusRecord",
    "SessionRecord",
    "SimilarSession",
    "PipelineResult",
    "DemonPipeline",
    "SemanticDriftPipeline",
]
