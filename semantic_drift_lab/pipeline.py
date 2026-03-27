"""
Совместимый модуль пайплайна под новым именем пакета.
"""

from demon.pipeline import CorpusRecord, SessionRecord, SimilarSession, PipelineResult, DemonPipeline

SemanticDriftPipeline = DemonPipeline

__all__ = [
    "CorpusRecord",
    "SessionRecord",
    "SimilarSession",
    "PipelineResult",
    "DemonPipeline",
    "SemanticDriftPipeline",
]
