#!/usr/bin/env python3
"""
Собирает data.jsonl из CSV-файла.

Позволяет указать, какие колонки использовать как:
- label
- text
- signal
- category
- source

Пример:
    python scripts/build_jsonl_from_csv.py data.csv \
      --text-column description \
      --label-column title \
      --signal-column score \
      --output data.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Собрать JSONL-корпус из CSV-файла"
    )
    parser.add_argument("input_csv", type=Path, help="CSV-файл с исходными данными")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data.jsonl"),
        help="Куда сохранить JSONL (по умолчанию ./data.jsonl)",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Название колонки с основным текстом",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Название колонки с короткой меткой",
    )
    parser.add_argument(
        "--signal-column",
        default="signal",
        help="Название колонки с числовым сигналом",
    )
    parser.add_argument(
        "--category-column",
        default="category",
        help="Название колонки с категорией",
    )
    parser.add_argument(
        "--source-column",
        default="source",
        help="Название колонки с источником",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="Разделитель CSV (по умолчанию запятая)",
    )
    return parser.parse_args()


def _clean(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_signal(value: str | None) -> float | None:
    value = _clean(value)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def build_record(row: dict[str, str], row_idx: int, args: argparse.Namespace) -> dict:
    label = _clean(row.get(args.label_column)) or f"row-{row_idx}"
    text = _clean(row.get(args.text_column)) or label
    signal = _parse_signal(row.get(args.signal_column))
    category = _clean(row.get(args.category_column))
    source = _clean(row.get(args.source_column))

    record: dict[str, object] = {
        "label": label,
        "text": text,
    }
    if signal is not None:
        record["signal"] = signal
    if category:
        record["category"] = category
    if source:
        record["source"] = source

    extra_meta = {
        key: value
        for key, value in row.items()
        if key not in {
            args.label_column,
            args.text_column,
            args.signal_column,
            args.category_column,
            args.source_column,
        }
        and _clean(value)
    }
    if extra_meta:
        record["meta"] = extra_meta

    return record


def main() -> None:
    args = parse_args()

    if not args.input_csv.exists():
        raise SystemExit(f"CSV-файл не найден: {args.input_csv}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.input_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f_in:
        reader = csv.DictReader(f_in, delimiter=args.delimiter)
        if reader.fieldnames is None:
            raise SystemExit("CSV не содержит заголовков колонок")

        with args.output.open("w", encoding="utf-8") as f_out:
            for row_idx, row in enumerate(reader):
                record = build_record(row, row_idx, args)
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    print(f"Собрано записей: {count}")
    print(f"JSONL сохранён в: {args.output}")


if __name__ == "__main__":
    main()
