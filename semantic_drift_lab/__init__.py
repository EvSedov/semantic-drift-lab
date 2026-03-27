"""
Semantic Drift Lab.

Новый канонический пакет проекта. На текущем этапе он переиспользует
реализацию из исторического пакета `demon`, чтобы переход на новое имя
был постепенным и без ломающих изменений.
"""

from demon import (
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
