"""
Совместимый shim для старого модуля поиска по knowledge base.

Новый адаптер находится в demon.adapters.markdown_corpus.
Этот файл сохранён, чтобы не ломать старые импорты.
"""

from .adapters.markdown_corpus import KBIndex, KBResult, MarkdownCorpusIndex, SearchResult

__all__ = [
    "MarkdownCorpusIndex",
    "SearchResult",
    "KBIndex",
    "KBResult",
]
