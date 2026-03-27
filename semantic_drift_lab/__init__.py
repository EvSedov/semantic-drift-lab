"""
Semantic Drift Lab.

Новый канонический пакет проекта.
"""

from .svd_embed import SVDEmbedder
from .takens_embed import takens_embedding
from .knn_stability import knn_stability
from .kalman_drift import KalmanDrift
from .pipeline import DemonPipeline, CorpusRecord, SessionRecord
from .adapters import MarkdownCorpusIndex, SearchResult, KBIndex, KBResult, load_pai_jsonl

# Новое каноническое имя пайплайна.
SemanticDriftPipeline = DemonPipeline

__all__ = [
    "SVDEmbedder",
    "takens_embedding",
    "knn_stability",
    "KalmanDrift",
    "DemonPipeline",
    "SemanticDriftPipeline",
    "CorpusRecord",
    "SessionRecord",
    "MarkdownCorpusIndex",
    "SearchResult",
    "KBIndex",
    "KBResult",
    "load_pai_jsonl",
]
