"""
Адаптер для загрузки PAI-подобных JSONL записей в универсальный формат.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..pipeline import CorpusRecord


def load_pai_jsonl(path: str | Path) -> list[CorpusRecord]:
    """
    Загружает PAI-подобный JSONL и приводит его к универсальному CorpusRecord.
    """
    records: list[CorpusRecord] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            text_parts = [
                data.get("task_description", ""),
                data.get("reflection_q1", ""),
                data.get("reflection_q2", ""),
                data.get("reflection_q3", ""),
                data.get("effort_level", ""),
            ]
            records.append(
                CorpusRecord(
                    idx=i,
                    label=data.get("task_description", ""),
                    text=" ".join(part for part in text_parts if part),
                    signal=float(data.get("implied_sentiment", 5)),
                    kind="pai_session",
                    meta={
                        "task": data.get("task_description", ""),
                        "effort": data.get("effort_level", ""),
                        "criteria_count": int(data.get("criteria_count", 0)),
                        "criteria_passed": int(data.get("criteria_passed", 0)),
                        "within_budget": bool(data.get("within_budget", True)),
                    },
                )
            )
    return records
