"""
Адаптеры источников данных для DEMON-Manifold.

Здесь живут необязательные интеграции поверх общего аналитического ядра.
"""

from .markdown_corpus import MarkdownCorpusIndex, SearchResult, KBIndex, KBResult

__all__ = [
    "MarkdownCorpusIndex",
    "SearchResult",
    "KBIndex",
    "KBResult",
]
