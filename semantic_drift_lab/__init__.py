"""
Semantic Drift Lab.

Новый канонический пакет проекта.
"""

from .svd_embed import SVDEmbedder
from .takens_embed import takens_embedding
from .knn_stability import knn_stability
from .kalman_drift import KalmanDrift
from .pipeline import SemanticDriftPipeline, DemonPipeline, CorpusRecord, SessionRecord
from .adapters import MarkdownCorpusIndex, SearchResult, KBIndex, KBResult, load_pai_jsonl

__all__ = [
    "SVDEmbedder",
    "takens_embedding",
    "knn_stability",
    "KalmanDrift",
    "SemanticDriftPipeline",
    "DemonPipeline",
    "CorpusRecord",
    "SessionRecord",
    "MarkdownCorpusIndex",
    "SearchResult",
    "KBIndex",
    "KBResult",
    "load_pai_jsonl",
]
