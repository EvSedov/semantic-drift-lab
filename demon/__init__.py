"""
DEMON-Manifold: Deterministic Embedding from Manifold Observation Neighbors
DEMON-inspired pipeline: SVD + Takens + kNN Stability + Kalman
"""
from semantic_drift_lab import (
    SVDEmbedder,
    takens_embedding,
    knn_stability,
    KalmanDrift,
    DemonPipeline,
    CorpusRecord,
    SessionRecord,
    MarkdownCorpusIndex,
    SearchResult,
    KBIndex,
    KBResult,
    load_pai_jsonl,
)

__all__ = [
    "SVDEmbedder",
    "takens_embedding",
    "knn_stability",
    "KalmanDrift",
    "DemonPipeline",
    "CorpusRecord",
    "SessionRecord",
    "MarkdownCorpusIndex",
    "SearchResult",
    "KBIndex",
    "KBResult",
    "load_pai_jsonl",
]
