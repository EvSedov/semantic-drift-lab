"""
KBSearch — поиск по knowledge-base markdown файлам через SVD-пайплайн.

Индексирует все .md файлы, сохраняет SVD-индекс в pickle-кэш,
при повторных запросах загружает из кэша без пересчёта.

Кэш инвалидируется при изменении числа файлов или суммарного mtime.
"""
from __future__ import annotations

import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

from .svd_embed import SVDEmbedder

# Директории, которые исключаются из индексации
EXCLUDE_DIRS: set[str] = {
    "private",
    ".git",
    "__pycache__",
    ".venv",
    "node_modules",
}

CACHE_DIR = Path.home() / ".cache" / "demon-manifold"
CACHE_FILE = CACHE_DIR / "kb_index.pkl"
MIN_TEXT_LENGTH = 50  # игнорируем файлы короче N символов


@dataclass
class KBResult:
    path: Path
    relative: str           # путь относительно kb_root
    section: str            # верхний уровень раздела (первая директория)
    cosine_sim: float
    snippet: str            # первые 120 символов текста
    low_confidence: bool = False  # True если cos < min_cosine


@dataclass
class _CacheEntry:
    kb_root: str
    n_files: int
    mtime_sum: float
    file_paths: list[str]
    texts: list[str]
    embeddings: np.ndarray
    embedder: SVDEmbedder


def _strip_markdown(text: str) -> str:
    """Убирает markdown-разметку, оставляет читаемый текст."""
    # Убираем code blocks (``` ... ```)
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    # Убираем inline code
    text = re.sub(r'`[^`]+`', ' ', text)
    # Убираем HTML-теги
    text = re.sub(r'<[^>]+>', ' ', text)
    # Убираем markdown-ссылки, оставляем текст
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Убираем изображения
    text = re.sub(r'!\[[^\]]*\]\([^\)]+\)', ' ', text)
    # Убираем заголовки (но оставляем текст)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Убираем горизонтальные линии
    text = re.sub(r'^[-*_]{3,}$', ' ', text, flags=re.MULTILINE)
    # Убираем bullet points
    text = re.sub(r'^[\s]*[-*+]\s+', ' ', text, flags=re.MULTILINE)
    # Убираем жирный/курсив
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', text)
    # Схлопываем пробелы
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _collect_files(kb_root: Path) -> list[Path]:
    """Рекурсивно собирает .md файлы, исключая EXCLUDE_DIRS."""
    files = []
    for path in kb_root.rglob("*.md"):
        # Проверяем что ни одна из частей пути не в EXCLUDE_DIRS
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def _cache_key(files: list[Path]) -> tuple[int, float]:
    n = len(files)
    mtime_sum = sum(f.stat().st_mtime for f in files if f.exists())
    return n, mtime_sum


def _load_cache(kb_root: Path) -> _CacheEntry | None:
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE, "rb") as f:
            entry: _CacheEntry = pickle.load(f)
        if entry.kb_root != str(kb_root):
            return None
        files = _collect_files(kb_root)
        n, mts = _cache_key(files)
        if entry.n_files == n and abs(entry.mtime_sum - mts) < 1.0:
            return entry
    except Exception:
        pass
    return None


def _save_cache(entry: _CacheEntry) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(entry, f)


class KBIndex:
    """SVD-индекс knowledge-base markdown файлов."""

    def __init__(self, entry: _CacheEntry) -> None:
        self._entry = entry

    @classmethod
    def build(
        cls,
        kb_root: Path,
        n_components: int = 20,
        force_rebuild: bool = False,
        verbose: bool = True,
    ) -> "KBIndex":
        """
        Строит или загружает из кэша SVD-индекс для kb_root.

        Parameters
        ----------
        kb_root      : корневая директория knowledge-base
        n_components : размерность SVD-пространства
        force_rebuild: игнорировать кэш и пересчитать
        verbose      : печатать прогресс
        """
        if not force_rebuild:
            cached = _load_cache(kb_root)
            if cached is not None:
                if verbose:
                    print(f"[KBIndex] Загружен кэш: {cached.n_files} файлов", flush=True)
                return cls(cached)

        if verbose:
            print(f"[KBIndex] Индексация {kb_root} …", flush=True)

        t0 = time.time()
        files = _collect_files(kb_root)
        if verbose:
            print(f"[KBIndex] Найдено файлов: {len(files)}", flush=True)

        texts: list[str] = []
        valid_files: list[Path] = []
        for f in files:
            try:
                raw = f.read_text(encoding="utf-8", errors="ignore")
                clean = _strip_markdown(raw)
                if len(clean) >= MIN_TEXT_LENGTH:
                    texts.append(clean)
                    valid_files.append(f)
            except Exception:
                continue

        if verbose:
            print(f"[KBIndex] Валидных файлов: {len(valid_files)}, строим SVD …", flush=True)

        safe_k = min(n_components, len(valid_files) - 1)
        embedder = SVDEmbedder(n_components=safe_k)
        embeddings = embedder.fit_transform(texts)

        n, mts = _cache_key(valid_files)
        entry = _CacheEntry(
            kb_root=str(kb_root),
            n_files=n,
            mtime_sum=mts,
            file_paths=[str(f) for f in valid_files],
            texts=texts,
            embeddings=embeddings,
            embedder=embedder,
        )
        _save_cache(entry)

        elapsed = time.time() - t0
        if verbose:
            print(
                f"[KBIndex] Готово за {elapsed:.1f}с "
                f"| SVD={safe_k}D "
                f"| variance={embedder.explained_variance_ratio_:.1%}",
                flush=True,
            )

        return cls(entry)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_cosine: float = 0.80,
    ) -> list[KBResult]:
        """
        Ищет top_k наиболее похожих документов на query.

        Parameters
        ----------
        min_cosine : порог отсечения. Результаты ниже порога помечаются
                     флагом `low_confidence=True` и не возвращаются по умолчанию
                     (возвращаются только если all results are below threshold,
                     чтобы пользователь видел хоть что-то).
        """
        e = self._entry
        query_vec = e.embedder.transform([query])          # (1, k)
        sims = (e.embeddings @ query_vec.T).flatten()      # cosine similarity

        top_idx = np.argsort(sims)[::-1][:top_k]
        kb_root = Path(e.kb_root)

        results = []
        for i in top_idx:
            path = Path(e.file_paths[i])
            try:
                relative = str(path.relative_to(kb_root))
            except ValueError:
                relative = str(path)

            parts = relative.split("/")
            section = parts[0] if len(parts) > 1 else "."
            snippet = e.texts[i][:120].replace("\n", " ").strip()

            results.append(KBResult(
                path=path,
                relative=relative,
                section=section,
                cosine_sim=float(sims[i]),
                snippet=snippet,
                low_confidence=float(sims[i]) < min_cosine,
            ))

        # Если все результаты ниже порога — возвращаем с пометкой
        # (лучше показать что-то, чем молчать, но пользователь видит флаг)
        confident = [r for r in results if not r.low_confidence]
        return confident if confident else results

    @property
    def n_files(self) -> int:
        return self._entry.n_files

    @property
    def explained_variance(self) -> float:
        return self._entry.embedder.explained_variance_ratio_
