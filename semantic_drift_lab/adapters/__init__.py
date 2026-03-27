"""
Адаптеры источников данных под новым именем пакета.
"""

from demon.adapters import MarkdownCorpusIndex, SearchResult, KBIndex, KBResult, load_pai_jsonl

__all__ = [
    "MarkdownCorpusIndex",
    "SearchResult",
    "KBIndex",
    "KBResult",
    "load_pai_jsonl",
]
