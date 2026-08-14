# Ноутбуки

- `00_original_baseline.ipynb` — исходный ноутбук организаторов без изменений логики.
- `01_baseline_ru.ipynb` — его русскоязычная версия.
- `04_first_catboost_ltv.ipynb` — запуск первой CatBoost-версии; рабочая логика находится в `src/`.
- `05_multifold_validation.ipynb` — проверка той же модели на четырёх временных holdout.
- `06_catboost_v2.ipynb` — расширенные временные и поведенческие признаки, OOF-проверка и второй submission.

Будущие ноутбуки:

- `02_eda.ipynb` — анализ данных и гипотезы.
- `03_features.ipynb` — исследование и сравнение групп признаков.
- `07_catboost_tuning.ipynb` — следующий эксперимент: подбор параметров CatBoost.
- `12_final_seasonal_submission.ipynb` — финальное обучение depth=6 и два submission: базовый и сезонно скорректированный.
