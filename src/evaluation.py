"""Сохранение метрик и диагностика важности признаков."""

import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor


def feature_importance(
    model: CatBoostRegressor,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Возвращает важности, отсортированные от большей к меньшей."""
    return pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)


def save_json(payload: object, path: Path) -> None:
    """Сохраняет небольшой отчёт в читаемом JSON-формате."""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
