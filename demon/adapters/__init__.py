"""
Адаптеры источников данных для DEMON-Manifold.

Здесь живут необязательные интеграции поверх общего аналитического ядра.
"""

from .markdown_corpus import MarkdownCorpusIndex, SearchResult, KBIndex, KBResult
from .pai_jsonl import load_pai_jsonl

__all__ = [
    "MarkdownCorpusIndex",
    "SearchResult",
    "KBIndex",
    "KBResult",
    "load_pai_jsonl",
]
