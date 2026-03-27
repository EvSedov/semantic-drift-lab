"""
Совместимый shim для исторического модуля demon.pipeline.

Каноническая реализация теперь находится в semantic_drift_lab.pipeline.
"""

from semantic_drift_lab.pipeline import (
    CorpusRecord,
    SessionRecord,
    SimilarSession,
    PipelineResult,
    DemonPipeline,
    SemanticDriftPipeline,
)

__all__ = [
    "CorpusRecord",
    "SessionRecord",
    "SimilarSession",
    "PipelineResult",
    "DemonPipeline",
    "SemanticDriftPipeline",
]
