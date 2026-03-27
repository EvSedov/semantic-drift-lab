# semantic-drift-lab

Semantic Drift Lab: детерминированный исследовательский пайплайн для анализа текстового корпуса, устойчивости и дрейфа.

## Структура

```
semantic_drift_lab/
  __init__.py
  svd_embed.py      — TF-IDF + TruncatedSVD (шаг 1)
  takens_embed.py   — delay embedding временного ряда (шаг 2)
  knn_stability.py  — kNN stability scoring (шаг 3)
  kalman_drift.py   — Kalman filter drift detection (шаг 4)
  pipeline.py
  adapters/
    __init__.py
run.py              — точка входа
output/             — PNG визуализации (создаётся автоматически)
```

## Запуск

```bash
# Активировать venv
source .venv/bin/activate

# Запустить на данных JSONL
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

Каждая строка — JSON с текстом, метками и опциональным числовым сигналом.

Поддерживаются простые поля вроде:
- `label`
- `text`
- `signal`
- `category`
- `source`

## Интеграция

Из Bun/TypeScript:
```typescript
const result = await Bun.spawn([
  "/home/evsedov/develop/my_dev/demon-manifold/.venv/bin/python",
  "/home/evsedov/develop/my_dev/demon-manifold/run.py",
  "--input", dataPath,
]).exited;
```
