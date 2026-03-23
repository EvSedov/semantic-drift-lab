# demon-manifold

DEMON-inspired детерминированный пайплайн: SVD + Takens + kNN Stability + Kalman.

## Структура

```
demon/
  svd_embed.py      — TF-IDF + TruncatedSVD (шаг 1)
  takens_embed.py   — delay embedding временного ряда (шаг 2)
  knn_stability.py  — kNN stability scoring (шаг 3)
  kalman_drift.py   — Kalman filter drift detection (шаг 4)
  pipeline.py       — единый пайплайн
run.py              — точка входа
output/             — PNG визуализации (создаётся автоматически)
```

## Запуск

```bash
# Активировать venv
source .venv/bin/activate

# Запустить на данных PAI
python run.py

# С кастомным файлом
python run.py --input /path/to/data.jsonl --top-k 5
```

## Зависимости

```bash
# Установка
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Входной формат (.jsonl)

Каждая строка — JSON с полями:
- `task_description` — описание задачи
- `effort_level` — уровень усилия (standard/extended/advanced/deep)
- `implied_sentiment` — оценка качества [1-10]
- `criteria_count`, `criteria_passed` — метрики прогресса
- `within_budget` — bool
- `reflection_q1..q3` — рефлексии (текст для SVD)

## Интеграция с PAI

Из Bun/TypeScript:
```typescript
const result = await Bun.spawn([
  "/home/evsedov/develop/my_dev/demon-manifold/.venv/bin/python",
  "/home/evsedov/develop/my_dev/demon-manifold/run.py",
  "--input", reflectionsPath,
]).exited;
```
