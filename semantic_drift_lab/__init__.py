"""
Semantic Drift Lab.

Новый канонический пакет проекта.
"""

from .svd_embed import SVDEmbedder
from .takens_embed import takens_embedding
from .knn_stability import knn_stability
from .kalman_drift import KalmanDrift
from .pipeline import SemanticDriftPipeline, CorpusRecord
from .adapters import MarkdownCorpusIndex, SearchResult, load_pai_jsonl

__all__ = [
    "SVDEmbedder",
    "takens_embedding",
    "knn_stability",
    "KalmanDrift",
    "SemanticDriftPipeline",
    "CorpusRecord",
    "MarkdownCorpusIndex",
    "SearchResult",
    "load_pai_jsonl",
]
