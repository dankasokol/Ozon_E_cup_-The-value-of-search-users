# Ozon E-Cup 2026: прогноз GMV

Репозиторий содержит выбранное итоговое решение. Оно предсказывает GMV каждого из 250 000 пользователей на 30 дней: с 14 февраля по 15 марта 2026 года.

Готовый файл: `submissions/submission_monetary_cadence_inactive_guard.csv`.

## Состав решения

Конвейерно состоит из трёх уровней:

1. базовые, годовые и ритмические признаки;
2. CatBoost с признаками денежного ритма и его логарифмическая смесь с контрольным прогнозом;
3. защитное правило: для пользователей без активных дней заказов используется только контрольный прогноз.

Подробное теоретическое описание будет вынесено в `README_Theory.md`.

## Структура

```text
data/README.md                     описание исходных данных
notebooks/final_solution.ipynb     сборка и проверка итогового CSV
src/features.py                    базовые признаки
src/seasonality.py                 сезонный профиль
src/annual_features.py             годовые признаки
src/cadence_features.py            ритм заказов
src/monetary_cadence.py            денежный ритм
src/final_solution.py              базовый слой моделей
src/monetary_experiment.py         обучение денежного слоя
src/inactive_guard.py              итоговое защитное правило и CSV
```

## Окружение

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name ozon-ecup --display-name ozon-ecup
```

## Запуск итогового слоя

```bash
python -m src.inactive_guard
```

Полное переобучение из исходного Parquet выполняется послойно и требует значительно больше времени и памяти.
