"""
DEMON Pipeline — объединяет все 4 шага в единый детерминированный пайплайн.

Поток данных:
  .jsonl → [SVD embed] → [kNN stability] → похожие сессии
         → [Takens embed] → [Kalman drift] → детектор дрейфа качества
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .svd_embed import SVDEmbedder
from .takens_embed import takens_embedding, optimal_delay
from .knn_stability import knn_stability, attractor_mask
from .kalman_drift import KalmanDrift, KalmanResult


@dataclass
class SessionRecord:
    idx: int
    task: str
    effort: str
    sentiment: float
    criteria_count: int
    criteria_passed: int
    within_budget: bool
    text: str  # объединённый текст для SVD


@dataclass
class SimilarSession:
    idx: int
    task: str
    cosine_sim: float
    stability: float


@dataclass
class PipelineResult:
    records: list[SessionRecord]
    embeddings: np.ndarray              # (n, svd_components)
    stability_scores: np.ndarray        # (n,) — kNN stability
    attractor_indices: list[int]        # индексы устойчивых точек
    similar_sessions: dict[int, list[SimilarSession]]  # idx → топ-K похожих
    kalman: KalmanResult                # результат Kalman на ряду sentiment
    takens_embedded: np.ndarray         # delay embedding sentiment-ряда
    svd_explained_variance: float
    n_attractors: int


class DemonPipeline:
    """Детерминированный пайплайн DEMON для анализа сессий PAI."""

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

    def load_jsonl(self, path: str | Path) -> list[SessionRecord]:
        records = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                text_parts = [
                    d.get("task_description", ""),
                    d.get("reflection_q1", ""),
                    d.get("reflection_q2", ""),
                    d.get("reflection_q3", ""),
                    d.get("effort_level", ""),
                ]
                records.append(SessionRecord(
                    idx=i,
                    task=d.get("task_description", ""),
                    effort=d.get("effort_level", ""),
                    sentiment=float(d.get("implied_sentiment", 5)),
                    criteria_count=int(d.get("criteria_count", 0)),
                    criteria_passed=int(d.get("criteria_passed", 0)),
                    within_budget=bool(d.get("within_budget", True)),
                    text=" ".join(p for p in text_parts if p),
                ))
        return records

    def run(self, jsonl_path: str | Path) -> PipelineResult:
        records = self.load_jsonl(jsonl_path)
        texts = [r.text for r in records]
        sentiments = np.array([r.sentiment for r in records])

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
        for i, rec in enumerate(records):
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

        # ── Шаг 4: Takens Embedding на ряду sentiment ──
        if len(sentiments) >= 4:
            tau = self.takens_delay or optimal_delay(sentiments)
            # Убеждаемся что dim/delay дают хотя бы 2 вектора
            max_dim = max(2, (len(sentiments) - 1) // max(tau, 1))
            dim = min(self.takens_dim, max_dim)
            takens = takens_embedding(sentiments, delay=tau, embedding_dim=dim)
        else:
            takens = sentiments.reshape(-1, 1)

        # ── Шаг 5: Kalman Drift Detection ──
        kalman = KalmanDrift(process_noise=self.kalman_Q, measurement_noise=self.kalman_R)
        kalman_result = kalman.filter(sentiments)

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
        )
