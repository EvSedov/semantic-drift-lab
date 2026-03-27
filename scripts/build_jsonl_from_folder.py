#!/usr/bin/env python3
"""
Собирает data.jsonl из папки с текстовыми файлами.

По умолчанию каждая запись строится по принципу:
- label    -> имя файла без расширения
- text     -> содержимое файла
- source   -> относительный путь
- category -> первая директория относительно корня, если есть

Пример:
    python scripts/build_jsonl_from_folder.py ./documents --output data.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_EXTENSIONS = (".md", ".txt", ".text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Собрать JSONL-корпус из директории с текстовыми файлами"
    )
    parser.add_argument("input_dir", type=Path, help="Папка с исходными файлами")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data.jsonl"),
        help="Куда сохранить JSONL (по умолчанию ./data.jsonl)",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help="Какие расширения включать, например: --extensions .md .txt",
    )
    parser.add_argument(
        "--strip-markdown",
        action="store_true",
        help="Грубо очистить markdown-разметку перед сохранением текста",
    )
    return parser.parse_args()


def strip_markdown(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s]*[-*+]\s+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def collect_files(root: Path, extensions: set[str]) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        files.append(path)
    return sorted(files)


def build_record(root: Path, path: Path, strip_md: bool) -> dict:
    relative = path.relative_to(root)
    text = path.read_text(encoding="utf-8", errors="ignore")
    if strip_md:
        text = strip_markdown(text)
    else:
        text = text.strip()

    parts = relative.parts
    category = parts[0] if len(parts) > 1 else "default"

    return {
        "label": path.stem,
        "text": text,
        "source": str(relative),
        "category": category,
    }


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Директория не найдена: {args.input_dir}")

    extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.extensions}
    files = collect_files(args.input_dir, extensions)

    if not files:
        raise SystemExit(
            f"Не найдено файлов с расширениями {sorted(extensions)} в {args.input_dir}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as out:
        for path in files:
            record = build_record(args.input_dir, path, strip_md=args.strip_markdown)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Собрано записей: {len(files)}")
    print(f"JSONL сохранён в: {args.output}")


if __name__ == "__main__":
    main()
