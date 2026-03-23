# DEMON-Manifold

**Deterministic Embedding from Manifold Observation Neighbors** — Python-реализация детерминированного пайплайна, вдохновлённого алгоритмом DEMON.

## Что это

Система объединяет 4 классических математических метода в единый пайплайн без нейросетей и без обучения:

| Шаг | Метод | Что делает |
|-----|-------|-----------|
| 1 | **SVD Embedding** | TF-IDF + TruncatedSVD → компактные векторы текстов |
| 2 | **Takens Embedding** | Delay vectors → фазовое пространство из числового ряда |
| 3 | **kNN Stability** | Устойчивость окрестности → аттракторы vs шум |
| 4 | **Kalman Filter** | Сглаживание + детектирование дрейфа качества |

## Быстрый старт

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

## Результат

- Консольный отчёт: похожие сессии, аттракторы, drift-алерты
- `output/clusters.png` — SVD-пространство с kNN stability окраской
- `output/drift.png` — Kalman-сглаживание и инновации

## Применение в PAI

Предназначен для анализа `algorithm-reflections.jsonl`:
- находит похожие прошлые задачи
- детектирует падение качества работы
- выделяет устойчивые паттерны (аттракторы) из шума

## Связь с оригинальным DEMON

Оригинальный DEMON (barometech) — proprietary, без peer review.
Эта реализация — независимая, open, использует те же математические принципы
применительно к данным PAI.
