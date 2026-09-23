# Byteford United — Beeline tariff campaign agent

Командный репозиторий решения хакатонного кейса по выбору тарифных
маркетинговых кампаний.

## Быстрый запуск

```bash
python local_eval.py
python local_eval.py --runs 10
python benchmark.py --runs 10
python make_submission.py
```

Финальный агент должен находиться в `agent.py` и предоставлять класс `Agent`
с методом `act(env)`.

## Исторические priors и кандидаты

Модуль `priors.py` готовит слабое априорное знание для разведки и не заменяет
результаты пилотов:

```python
import pandas as pd

from priors import build_default_candidates, select_pilot_candidates

profile = pd.read_csv("customer_profile.csv")
history = pd.read_csv("data/change_tariff.csv")
tariffs = pd.read_csv("data/dict_tariff.csv")

priors, candidates = build_default_candidates(profile, history, tariffs)
pilot_shortlist = select_pilot_candidates(candidates, top_k=20)
```

`priors` содержит сглаженную оценку для каждого перехода, включая отсутствующие
в истории. `candidates` добавляет текущий размер аудитории, канал, стоимость и
ожидаемый чистый эффект. Агент должен обновлять эти оценки результатами пилотов.

Проверка модуля:

```bash
python -m unittest tests.test_priors
```

## Безопасность

- API-ключи и другие секреты не хранятся в репозитории.
- Локальные `.env`, ключи и файлы учётных данных исключены через `.gitignore`.
- Если интеграция с LLM понадобится, ключ читается только из переменной среды
  `OPENAI_API_KEY`; числовая логика агента обязана иметь локальный fallback.

Официальные данные кейса синтетические и предназначены только для учебной
задачи хакатона.

## Архитектура

- `agent.py` — обязательная точка входа для среды.
- `strategy.py` — адаптивные пилоты, оценка неопределённости и выбор портфеля.
- `priors.py` — исторические priors, сглаживание, fallback для неизвестных
  переходов и таблица кандидатов; при ошибке модуль заменяется безопасным
  fallback внутри стратегии.
- `tests/test_agent_smoke.py` — проверка интерфейса и лимитов среды.
- `benchmark.py` — A/B-сравнение стратегии с historical priors и без них.
- `TESTING.md` — пошаговая инструкция по самостоятельной проверке.

Функция `priors.build_candidates(profile, tariffs, base_dir)` возвращает
`DataFrame` с обязательными колонками `current_tariff`, `arpu_segment`,
`target_tariff`. Поддерживаются опциональные `prior_mean`, `prior_std`,
`priority` и `source`. `prior_mean` задаётся как относительный эффект при
множителе канала 1.0.
