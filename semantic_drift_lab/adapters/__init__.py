"""
Адаптеры источников данных под новым именем пакета.
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
