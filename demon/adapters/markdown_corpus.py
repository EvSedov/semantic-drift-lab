"""
Markdown corpus adapter.

Индексирует markdown-файлы из произвольной директории через TF-IDF + SVD
и позволяет искать наиболее похожие документы по текстовому запросу.

Это не часть базового ядра пайплайна, а дополнительный адаптер источника
данных. Для обратной совместимости экспортируются также старые имена
KBIndex и KBResult.
"""
from __future__ import annotations

import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..svd_embed import SVDEmbedder

# Директории, которые исключаются из индексации
EXCLUDE_DIRS: set[str] = {
    "private",
    ".git",
    "__pycache__",
    ".venv",
    "node_modules",
}

CACHE_DIR = Path.home() / ".cache" / "demon-manifold"
CACHE_FILE = CACHE_DIR / "markdown_corpus_index.pkl"
MIN_TEXT_LENGTH = 50  # игнорируем файлы короче N символов


@dataclass
class SearchResult:
    path: Path
    relative: str
    section: str
    cosine_sim: float
    snippet: str
    low_confidence: bool = False


@dataclass
class _CacheEntry:
    corpus_root: str
    n_files: int
    mtime_sum: float
    file_paths: list[str]
    texts: list[str]
    embeddings: np.ndarray
    embedder: SVDEmbedder


def _strip_markdown(text: str) -> str:
    """Убирает markdown-разметку, оставляет читаемый текст."""
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", " ", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*_]{3,}$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s]*[-*+]\s+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _collect_files(corpus_root: Path) -> list[Path]:
    """Рекурсивно собирает .md файлы, исключая EXCLUDE_DIRS."""
    files = []
    for path in corpus_root.rglob("*.md"):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def _cache_key(files: list[Path]) -> tuple[int, float]:
    n = len(files)
    mtime_sum = sum(f.stat().st_mtime for f in files if f.exists())
    return n, mtime_sum


def _load_cache(corpus_root: Path) -> _CacheEntry | None:
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE, "rb") as f:
            entry: _CacheEntry = pickle.load(f)
        if entry.corpus_root != str(corpus_root):
            return None
        files = _collect_files(corpus_root)
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


class MarkdownCorpusIndex:
    """SVD-индекс markdown-корпуса."""

    def __init__(self, entry: _CacheEntry) -> None:
        self._entry = entry

    @classmethod
    def build(
        cls,
        corpus_root: Path,
        n_components: int = 20,
        force_rebuild: bool = False,
        verbose: bool = True,
    ) -> "MarkdownCorpusIndex":
        if not force_rebuild:
            cached = _load_cache(corpus_root)
            if cached is not None:
                if verbose:
                    print(
                        f"[MarkdownCorpusIndex] Загружен кэш: {cached.n_files} файлов",
                        flush=True,
                    )
                return cls(cached)

        if verbose:
            print(f"[MarkdownCorpusIndex] Индексация {corpus_root} …", flush=True)

        t0 = time.time()
        files = _collect_files(corpus_root)
        if verbose:
            print(f"[MarkdownCorpusIndex] Найдено файлов: {len(files)}", flush=True)

        texts: list[str] = []
        valid_files: list[Path] = []
        for file_path in files:
            try:
                raw = file_path.read_text(encoding="utf-8", errors="ignore")
                clean = _strip_markdown(raw)
                if len(clean) >= MIN_TEXT_LENGTH:
                    texts.append(clean)
                    valid_files.append(file_path)
            except Exception:
                continue

        if verbose:
            print(
                f"[MarkdownCorpusIndex] Валидных файлов: {len(valid_files)}, строим SVD …",
                flush=True,
            )

        safe_k = min(n_components, len(valid_files) - 1)
        embedder = SVDEmbedder(n_components=safe_k)
        embeddings = embedder.fit_transform(texts)

        n, mts = _cache_key(valid_files)
        entry = _CacheEntry(
            corpus_root=str(corpus_root),
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
                f"[MarkdownCorpusIndex] Готово за {elapsed:.1f}с "
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
    ) -> list[SearchResult]:
        e = self._entry
        query_vec = e.embedder.transform([query])
        sims = (e.embeddings @ query_vec.T).flatten()

        top_idx = np.argsort(sims)[::-1][:top_k]
        corpus_root = Path(e.corpus_root)

        results = []
        for i in top_idx:
            path = Path(e.file_paths[i])
            try:
                relative = str(path.relative_to(corpus_root))
            except ValueError:
                relative = str(path)

            parts = relative.split("/")
            section = parts[0] if len(parts) > 1 else "."
            snippet = e.texts[i][:120].replace("\n", " ").strip()

            results.append(
                SearchResult(
                    path=path,
                    relative=relative,
                    section=section,
                    cosine_sim=float(sims[i]),
                    snippet=snippet,
                    low_confidence=float(sims[i]) < min_cosine,
                )
            )

        confident = [r for r in results if not r.low_confidence]
        return confident if confident else results

    @property
    def n_files(self) -> int:
        return self._entry.n_files

    @property
    def explained_variance(self) -> float:
        return self._entry.embedder.explained_variance_ratio_


# Совместимость со старыми именами
KBIndex = MarkdownCorpusIndex
KBResult = SearchResult
