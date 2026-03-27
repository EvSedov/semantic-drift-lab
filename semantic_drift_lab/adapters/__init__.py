"""
Адаптеры источников данных под новым именем пакета.
"""

from .markdown_corpus import MarkdownCorpusIndex, SearchResult
from .pai_jsonl import load_pai_jsonl

__all__ = [
    "MarkdownCorpusIndex",
    "SearchResult",
    "load_pai_jsonl",
]
