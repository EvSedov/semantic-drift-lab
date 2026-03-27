"""
Адаптеры источников данных под новым именем пакета.
"""

from .markdown_corpus import MarkdownCorpusIndex, SearchResult
from .generic_jsonl import load_jsonl_records

__all__ = [
    "MarkdownCorpusIndex",
    "SearchResult",
    "load_jsonl_records",
]
