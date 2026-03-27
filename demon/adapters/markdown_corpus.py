"""Совместимый shim для исторического модуля demon.adapters.markdown_corpus."""

from semantic_drift_lab.adapters.markdown_corpus import (
    MarkdownCorpusIndex,
    SearchResult,
    KBIndex,
    KBResult,
)

__all__ = [
    "MarkdownCorpusIndex",
    "SearchResult",
    "KBIndex",
    "KBResult",
]
