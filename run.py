#!/usr/bin/env python3
"""
Semantic Drift Lab: точка входа.
Запускает универсальный аналитический пайплайн для текстового корпуса
и опционального числового ряда, а также поддерживает отдельный поиск
по markdown-корпусу.

Использование:
    .venv/bin/python run.py
    .venv/bin/python run.py --input /path/to/custom.jsonl
    .venv/bin/python run.py --input data.jsonl --top-k 5
    .venv/bin/python run.py --json          ← только JSON, без графиков
    .venv/bin/python run.py --json --pretty ← JSON с отступами
"""
import argparse
import json
import sys
from pathlib import Path

DEFAULT_MARKDOWN_DIR = Path.home() / "documents"
DEFAULT_JSONL = Path("data.jsonl")
OUTPUT_DIR = Path(__file__).parent / "output"


def get_pyplot():
    """
    Ленивая загрузка pyplot только в графических режимах.

    Это позволяет использовать `--help` и JSON-режимы даже там, где
    matplotlib не установлен в системном Python.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def get_numpy():
    """Ленивая загрузка numpy только там, где он реально нужен."""
    import numpy as np

    return np


def import_runtime_symbol(module_name: str, symbol_name: str):
    """
    Импортирует runtime-зависимости с понятной диагностикой вместо traceback.
    """
    try:
        module = __import__(module_name, fromlist=[symbol_name])
        return getattr(module, symbol_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or "dependency"
        print(
            "Не хватает Python-зависимостей для запуска аналитических режимов.\n"
            f"Отсутствует модуль: {missing}\n\n"
            "Установи зависимости проекта и запусти команду из venv:\n"
            "  python3 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  pip install -r requirements.txt\n"
            "  python run.py ...\n\n"
            "Или запусти сразу так:\n"
            "  .venv/bin/python run.py ...",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic Drift Lab: анализ текстового корпуса, похожих записей и drift-сигнала"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--svd-components", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="Вывести результат как JSON (без визуализаций)")
    parser.add_argument("--pretty", action="store_true", help="JSON с отступами (используется с --json)")
    parser.add_argument("--find-similar", metavar="QUERY", help="Найти записи корпуса, похожие на текст запроса")
    parser.add_argument("--doc-query", metavar="QUERY", help="Поиск по markdown-документам (дополнительный режим)")
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_MARKDOWN_DIR, help="Путь к markdown-документам (по умолчанию ~/documents)")
    parser.add_argument("--explain-search", action="store_true", help="Показать, за счёт чего документы поднялись в топ для --doc-query")
    parser.add_argument("--rebuild-index", action="store_true", help="Принудительно пересчитать индекс markdown-корпуса")
    parser.add_argument("--min-cosine", type=float, default=0.80, help="Минимальный порог cos для поиска по markdown-корпусу (по умолчанию 0.80)")
    return parser.parse_args()


def to_json(result, top_k: int) -> dict:
    """Сериализует PipelineResult в dict для JSON-вывода."""
    records = result.records
    kr = result.kalman

    return {
        "meta": {
            "n_records": len(records),
            "svd_explained_variance": round(result.svd_explained_variance, 4),
            "n_attractors": result.n_attractors,
            "attractor_indices": result.attractor_indices,
        },
        "stability": [
            {
                "idx": r.idx,
                "task": r.task,
                "effort": r.effort,
                "sentiment": r.sentiment,
                "stability": round(float(result.stability_scores[r.idx]), 4),
                "is_attractor": r.idx in result.attractor_indices,
            }
            for r in records
        ],
        "similar_records": {
            str(i): [
                {
                    "idx": s.idx,
                    "task": s.task,
                    "cosine_sim": round(s.cosine_sim, 4),
                    "stability": round(s.stability, 4),
                }
                for s in sims[:top_k]
            ]
            for i, sims in result.similar_records.items()
        },
        "drift": {
            "drift_score": round(kr.drift_score, 4),
            "has_drift": bool(kr.drift_flags.any()),
            "drift_indices": [int(i) for i, f in enumerate(kr.drift_flags) if f],
            "signal": [r.sentiment for r in records],
            "smoothed": [round(float(x), 3) for x in kr.smoothed],
            "innovations": [round(float(x), 3) for x in kr.innovations],
        },
        "takens": {
            "shape": list(result.takens_embedded.shape),
            "vectors": result.takens_embedded.tolist(),
        },
    }


def print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def run_report(result, top_k: int) -> None:
    np = get_numpy()
    records = result.records
    n = len(records)

    print_section(f"Semantic Drift Lab | {n} записей проанализировано")

    # SVD info
    print(f"\n[SVD] Объяснённая дисперсия: {result.svd_explained_variance:.1%}")
    print(f"[kNN] Устойчивых точек: {result.n_attractors}/{n}")

    # kNN Stability
    print_section("kNN Stability — устойчивые записи (аттракторы)")
    sorted_by_stability = sorted(
        enumerate(result.stability_scores), key=lambda x: x[1], reverse=True
    )
    for rank, (i, score) in enumerate(sorted_by_stability[:5], 1):
        marker = "★" if i in result.attractor_indices else " "
        task_short = records[i].task[:55] + "…" if len(records[i].task) > 55 else records[i].task
        print(f"  {rank}. {marker} [{score:.3f}] {task_short}")

    # Похожие записи
    print_section(f"Похожие записи (топ-{top_k} по косинусному сходству)")
    for i, rec in enumerate(records):
        task_short = rec.task[:50] + "…" if len(rec.task) > 50 else rec.task
        print(f"\n  [{i}] {task_short} (sentiment={rec.sentiment})")
        for sim in result.similar_records[i][:top_k]:
            sim_task = sim.task[:45] + "…" if len(sim.task) > 45 else sim.task
            print(f"       → [{sim.idx}] cos={sim.cosine_sim:.3f} | {sim_task}")

    # Kalman drift
    print_section("Kalman Drift Detection — динамика сигнала")
    kr = result.kalman
    sentiments = [r.sentiment for r in records]
    print(f"  Сигнал:        {[round(s, 1) for s in sentiments]}")
    print(f"  Сглажено:      {[round(float(x), 1) for x in kr.smoothed]}")
    print(f"  Drift score:   {kr.drift_score:.1%}")

    drift_indices = [i for i, f in enumerate(kr.drift_flags) if f]
    if drift_indices:
        print(f"\n  ⚠️  DRIFT ALERT — аномальные точки: {drift_indices}")
        for i in drift_indices:
            task_short = records[i].task[:55] + "…" if len(records[i].task) > 55 else records[i].task
            print(f"      [{i}] Δ={kr.innovations[i]:+.2f} | {task_short}")
    else:
        print("\n  ✅ Дрейфа не обнаружено — качество стабильно")

    # Takens
    print_section("Takens Embedding — фазовое пространство")
    te = result.takens_embedded
    print(f"  Shape: {te.shape} (из {len(sentiments)} наблюдений)")
    print(f"  Первые 3 вектора состояния:")
    for v in te[:3]:
        print(f"    {np.round(v, 2)}")


def plot_clusters(result, output_dir: Path) -> None:
    """Визуализация SVD-эмбеддингов с stability окраской."""
    plt = get_pyplot()
    embeddings = result.embeddings
    stability = result.stability_scores

    fig, ax = plt.subplots(figsize=(10, 7))

    # PCA-проекция первых двух компонент SVD
    x, y = embeddings[:, 0], embeddings[:, 1]
    scatter = ax.scatter(x, y, c=stability, cmap="RdYlGn", s=120,
                         vmin=0, vmax=1, edgecolors="gray", linewidths=0.5)

    # Подписи
    for i, rec in enumerate(result.records):
        label = f"[{i}] {rec.effort}"
        ax.annotate(label, (x[i], y[i]), textcoords="offset points",
                    xytext=(6, 4), fontsize=7, alpha=0.8)

    plt.colorbar(scatter, ax=ax, label="kNN Stability")
    ax.set_title("Semantic Drift Lab: SVD Embeddings + kNN Stability\n(зелёный = аттрактор, красный = шум)")
    ax.set_xlabel("SVD Component 1")
    ax.set_ylabel("SVD Component 2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = output_dir / "clusters.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n[plot] Кластеры сохранены: {path}")


def plot_drift(result, output_dir: Path) -> None:
    """Визуализация Kalman-сглаживания и дрейфа."""
    np = get_numpy()
    plt = get_pyplot()
    kr = result.kalman
    sentiments = [r.sentiment for r in result.records]
    n = len(sentiments)
    x = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Верхний график: sentiment + Kalman smoothing
    ax1.plot(x, sentiments, "o--", color="steelblue", label="observed", alpha=0.7)
    ax1.plot(x, kr.smoothed, "-", color="darkorange", linewidth=2, label="Kalman smoothed")

    drift_x = [i for i, f in enumerate(kr.drift_flags) if f]
    if drift_x:
        drift_y = [sentiments[i] for i in drift_x]
        ax1.scatter(drift_x, drift_y, color="red", zorder=5, s=100, label="drift detected")

    ax1.set_ylabel("signal")
    ax1.set_ylim(0, 11)
    ax1.legend()
    ax1.set_title("Kalman Drift Detection — динамика сигнала")
    ax1.grid(True, alpha=0.3)

    # Нижний график: innovations
    ax2.bar(x, kr.innovations, color=["red" if f else "steelblue" for f in kr.drift_flags],
            alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Session index")
    ax2.set_ylabel("Innovation (z - predicted)")
    ax2.set_title("Инновации (отклонение от прогноза)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "drift.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot] Drift chart сохранён: {path}")


def main() -> None:
    args = parse_args()

    # ── Дополнительный режим: поиск по markdown-корпусу ──
    if args.doc_query:
        MarkdownCorpusIndex = import_runtime_symbol(
            "semantic_drift_lab", "MarkdownCorpusIndex"
        )

        if not args.doc_path.exists():
            print(f"Директория не найдена: {args.doc_path}", file=sys.stderr)
            sys.exit(1)
        index = MarkdownCorpusIndex.build(
            args.doc_path,
            force_rebuild=args.rebuild_index,
            verbose=not args.json,
        )
        results = index.search(args.doc_query, top_k=args.top_k, min_cosine=args.min_cosine)

        top1_score = results[0].score if results else 0.0
        top1_conf = results[0].confidence if results else 0.0
        top1_cos = results[0].cosine_sim if results else 0.0
        if top1_conf >= 0.85:
            trust = "high"
            trust_label = "✅ Можно доверять"
        elif top1_conf >= 0.80:
            trust = "medium"
            trust_label = "⚠️  Доверяй с осторожностью"
        else:
            trust = "none"
            trust_label = "❌ Не доверяй — тема, вероятно, отсутствует в корпусе"

        if args.json:
            indent = 2 if args.pretty else None
            output = {
                "query": args.doc_query,
                "n_indexed": index.n_files,
                "min_cosine": args.min_cosine,
                "trust": trust,
                "top1_confidence": round(top1_conf, 4),
                "top1_score": round(top1_score, 4),
                "top1_cosine": round(top1_cos, 4),
                "results": [
                    {
                        "relative": r.relative,
                        "section": r.section,
                        "confidence": round(r.confidence, 4),
                        "score": round(r.score, 4),
                        "cosine_sim": round(r.cosine_sim, 4),
                        "low_confidence": r.low_confidence,
                        "snippet": r.snippet,
                        **(
                            {
                                "explain": {
                                    "token_overlap_path": round(float(r.explain["token_overlap_path"]), 4),
                                    "token_overlap_text": round(float(r.explain["token_overlap_text"]), 4),
                                    "name_bonus": round(float(r.explain["name_bonus"]), 4),
                                    "path_substring_bonus": round(float(r.explain["path_substring_bonus"]), 4),
                                    "path_bonus": round(float(r.explain["path_bonus"]), 4),
                                    "title_bonus": round(float(r.explain["title_bonus"]), 4),
                                    "early_text_bonus": round(float(r.explain["early_text_bonus"]), 4),
                                    "repetition_bonus": round(float(r.explain["repetition_bonus"]), 4),
                                    "occurrence_count": int(r.explain["occurrence_count"]),
                                    "first_occurrence": int(r.explain["first_occurrence"]),
                                }
                            }
                            if args.explain_search and r.explain is not None
                            else {}
                        ),
                    }
                    for r in results
                ],
            }
            print(json.dumps(output, ensure_ascii=False, indent=indent))
        else:
            print(
                f"\n[Docs] {trust_label} | top-1 confidence={top1_conf:.3f} "
                f"| score={top1_score:.3f} | cos={top1_cos:.3f}"
            )
            print(f'Результаты по запросу: "{args.doc_query}"\n')
            for rank, r in enumerate(results, 1):
                flag = " ⚠️" if r.low_confidence else ""
                print(
                    f"  {rank}. [conf={r.confidence:.3f} | score={r.score:.3f} "
                    f"| cos={r.cosine_sim:.3f}]{flag} {r.relative}"
                )
                print(f"       {r.snippet[:100]}…")
                if args.explain_search and r.explain is not None:
                    print("       explain:")
                    print(
                        "         "
                        f"path_overlap={float(r.explain['token_overlap_path']):.3f} "
                        "(совпадение токенов запроса с путём)"
                    )
                    print(
                        "         "
                        f"text_overlap={float(r.explain['token_overlap_text']):.3f} "
                        "(совпадение токенов запроса с текстом)"
                    )
                    print(
                        "         "
                        f"name_bonus={float(r.explain['name_bonus']):.3f} "
                        "(бонус за подстроку в имени файла)"
                    )
                    print(
                        "         "
                        f"path_substring_bonus={float(r.explain['path_substring_bonus']):.3f} "
                        "(бонус за подстроку в полном пути)"
                    )
                    print(
                        "         "
                        f"path_bonus={float(r.explain['path_bonus']):.3f} "
                        "(суммарный бонус пути = имя файла + полный путь)"
                    )
                    print(
                        "         "
                        f"title_bonus={float(r.explain['title_bonus']):.3f} "
                        "(бонус за совпадение в начале документа)"
                    )
                    print(
                        "         "
                        f"early_bonus={float(r.explain['early_text_bonus']):.3f} "
                        "(бонус за раннее появление в тексте)"
                    )
                    print(
                        "         "
                        f"repeat_bonus={float(r.explain['repetition_bonus']):.3f} "
                        "(бонус за повторяемость запроса)"
                    )
                    print(
                        "         "
                        f"occurrences={int(r.explain['occurrence_count'])} "
                        "(сколько раз запрос встретился)"
                    )
                    print(
                        "         "
                        f"first_pos={int(r.explain['first_occurrence'])} "
                        "(позиция первого вхождения в тексте)"
                    )
        return

    if not args.input.exists():
        print(f"Файл не найден: {args.input}", file=sys.stderr)
        sys.exit(1)

    SemanticDriftPipeline = import_runtime_symbol(
        "semantic_drift_lab", "SemanticDriftPipeline"
    )

    pipeline = SemanticDriftPipeline(
        svd_components=args.svd_components,
        top_k_similar=args.top_k,
    )
    result = pipeline.run(args.input)

    if args.find_similar:
        matches = pipeline.find_similar(args.find_similar, result, top_k=args.top_k)
        if args.json:
            indent = 2 if args.pretty else None
            output = {
                "query": args.find_similar,
                "results": [
                    {"idx": s.idx, "task": s.task,
                     "cosine_sim": round(s.cosine_sim, 4),
                     "stability": round(s.stability, 4)}
                    for s in matches
                ],
            }
            print(json.dumps(output, ensure_ascii=False, indent=indent))
        else:
            print(f'\nПохожие записи для: "{args.find_similar}"\n')
            for rank, s in enumerate(matches, 1):
                print(f"  {rank}. [{s.idx}] cos={s.cosine_sim:.3f} | {s.task}")
        return

    if args.json:
        indent = 2 if args.pretty else None
        print(json.dumps(to_json(result, top_k=args.top_k), ensure_ascii=False, indent=indent))
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Загрузка: {args.input}")
    run_report(result, top_k=args.top_k)
    plot_clusters(result, OUTPUT_DIR)
    plot_drift(result, OUTPUT_DIR)

    print(f"\n{'═' * 60}")
    print(f"  Готово. Визуализации в: {OUTPUT_DIR}/")
    print('═' * 60)


if __name__ == "__main__":
    main()
