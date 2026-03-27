"""
Адаптеры источников данных для DEMON-Manifold.

Здесь живут необязательные интеграции поверх общего аналитического ядра.
"""

from semantic_drift_lab.adapters import (
    MarkdownCorpusIndex,
    SearchResult,
    KBIndex,
    KBResult,
    load_pai_jsonl,
)

__all__ = [
    "MarkdownCorpusIndex",
    "SearchResult",
    "KBIndex",
    "KBResult",
    "load_pai_jsonl",
]
