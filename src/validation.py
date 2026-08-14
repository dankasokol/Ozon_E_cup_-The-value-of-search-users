"""Метрика и правила честной временной валидации."""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from .config import ANCHOR_STEP_DAYS, HORIZON_DAYS, N_HISTORY_ANCHORS


@dataclass(frozen=True)
class TemporalFold:
    """Один честный временной holdout."""

    validation_anchor: date
    train_anchors: tuple[date, ...]


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSLE с защитой от отрицательных предсказаний."""
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def make_historical_anchors(
    latest_anchor: date,
    *,
    step_days: int = ANCHOR_STEP_DAYS,
    n_anchors: int = N_HISTORY_ANCHORS,
) -> list[date]:
    """Создаёт отсортированный список исторических дат-срезов."""
    return sorted(
        latest_anchor - timedelta(days=step_days * index)
        for index in range(n_anchors)
    )


def anchors_available_for_holdout(
    anchors: list[date],
    validation_anchor: date,
    *,
    horizon_days: int = HORIZON_DAYS,
) -> list[date]:
    """Оставляет срезы, чьи целевые 30 дней заканчиваются до валидационного якоря."""
    return [
        anchor
        for anchor in anchors
        if anchor + timedelta(days=horizon_days) <= validation_anchor
    ]


def make_temporal_folds(
    anchors: list[date],
    validation_anchors: tuple[date, ...],
    *,
    horizon_days: int = HORIZON_DAYS,
) -> list[TemporalFold]:
    """Строит фолды без пересечения обучающих целей с датой валидации."""
    available_anchors = set(anchors)
    missing_anchors = set(validation_anchors) - available_anchors
    if missing_anchors:
        missing = ", ".join(str(anchor) for anchor in sorted(missing_anchors))
        raise ValueError(f"Не найдены сохранённые срезы для дат: {missing}")

    folds: list[TemporalFold] = []
    for validation_anchor in validation_anchors:
        train_anchors = tuple(
            anchors_available_for_holdout(
                anchors,
                validation_anchor,
                horizon_days=horizon_days,
            )
        )
        if not train_anchors:
            raise ValueError(
                f"Для holdout {validation_anchor} не осталось обучающих срезов."
            )
        folds.append(
            TemporalFold(
                validation_anchor=validation_anchor,
                train_anchors=train_anchors,
            )
        )
    return folds


def feature_columns(data: pl.DataFrame) -> list[str]:
    """Возвращает колонки, которые можно передавать модели."""
    excluded = {"user_id", "anchor_date", "target"}
    return [column for column in data.columns if column not in excluded]
