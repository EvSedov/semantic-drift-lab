"""
Адаптер для загрузки JSONL-записей в универсальный формат.

Поддерживает простой документный формат с текстом, метками и числовым
сигналом, а также совместим с более богатыми JSON-объектами, где нужные
данные распределены по нескольким полям.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..pipeline import CorpusRecord

TEXT_CANDIDATE_KEYS = (
    "text",
    "content",
    "body",
    "description",
    "summary",
    "task_description",
)

TEXT_FRAGMENT_KEYS = (
    "text",
    "content",
    "body",
    "description",
    "summary",
    "notes",
    "comment",
    "reflection_q1",
    "reflection_q2",
    "reflection_q3",
    "effort_level",
)

LABEL_CANDIDATE_KEYS = (
    "label",
    "title",
    "name",
    "task",
    "task_description",
    "description",
)

SIGNAL_CANDIDATE_KEYS = (
    "signal",
    "score",
    "value",
    "sentiment",
    "quality",
    "implied_sentiment",
)

META_PASSTHROUGH_KEYS = (
    "task",
    "effort",
    "effort_level",
    "category",
    "tags",
    "source",
)


def _first_string(data: dict, keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _combined_text(data: dict) -> str:
    direct_text = _first_string(data, TEXT_CANDIDATE_KEYS)
    if direct_text:
        return direct_text

    fragments = []
    for key in TEXT_FRAGMENT_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            fragments.append(value.strip())
    return " ".join(fragments)


def _first_number(data: dict, keys: tuple[str, ...], default: float = 0.0) -> float:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def load_jsonl_records(path: str | Path) -> list[CorpusRecord]:
    """
    Загружает JSONL и приводит записи к универсальному CorpusRecord.
    """
    records: list[CorpusRecord] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            label = _first_string(data, LABEL_CANDIDATE_KEYS, default=f"record-{i}")
            text = _combined_text(data) or label
            signal = _first_number(data, SIGNAL_CANDIDATE_KEYS, default=0.0)

            meta = {key: data[key] for key in META_PASSTHROUGH_KEYS if key in data}
            if "task_description" in data and "task" not in meta:
                meta["task"] = data["task_description"]
            if "effort_level" in data and "effort" not in meta:
                meta["effort"] = data["effort_level"]

            records.append(
                CorpusRecord(
                    idx=i,
                    label=label,
                    text=text,
                    signal=signal,
                    kind=str(data.get("kind", "document")),
                    meta=meta,
                )
            )
    return records
