"""
DEMON-Manifold: Deterministic Embedding from Manifold Observation Neighbors
DEMON-inspired pipeline: SVD + Takens + kNN Stability + Kalman
"""
from .svd_embed import SVDEmbedder
from .takens_embed import takens_embedding
from .knn_stability import knn_stability
from .kalman_drift import KalmanDrift
from .pipeline import DemonPipeline
from .adapters import MarkdownCorpusIndex, SearchResult, KBIndex, KBResult

__all__ = [
    "SVDEmbedder",
    "takens_embedding",
    "knn_stability",
    "KalmanDrift",
    "DemonPipeline",
    "MarkdownCorpusIndex",
    "SearchResult",
    "KBIndex",
    "KBResult",
]
