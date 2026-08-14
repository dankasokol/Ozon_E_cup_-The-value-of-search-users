"""Общий запуск сравнений моделей на одинаковых временных фолдах."""

import gc
from collections.abc import Callable

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostRegressor

from .models import predict_gmv
from .validation import TemporalFold, rmsle


def run_temporal_experiment(
    snapshots: dict,
    folds: list[TemporalFold],
    features: list[str],
    model_factory: Callable[[], CatBoostRegressor],
    *,
    keep_oof: bool = False,
    label: str = "experiment",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Обучает одну версию модели на каждом фолде и возвращает её метрики.

    На всех фолдах передаётся один и тот же набор ``features``. Это делает
    разность RMSLE между экспериментами сопоставимой внутри каждой даты.
    """
    metric_rows: list[dict[str, object]] = []
    oof_frames: list[pd.DataFrame] = []

    for fold in folds:
        print(f"[{label}] holdout {fold.validation_anchor}")
        train = pl.concat(
            [snapshots[anchor] for anchor in fold.train_anchors],
            how="vertical_relaxed",
        )
        valid = snapshots[fold.validation_anchor]

        x_train = train.select(features).to_pandas()
        y_train = np.log1p(train["target"].to_numpy())
        x_valid = valid.select(features).to_pandas()
        y_valid = valid["target"].to_numpy()

        model = model_factory()
        model.fit(
            x_train,
            y_train,
            eval_set=(x_valid, np.log1p(y_valid)),
            early_stopping_rounds=200,
            use_best_model=True,
        )
        prediction = predict_gmv(model, x_valid)
        baseline_prediction = valid["gmv_sum_30d"].to_numpy()
        score = rmsle(y_valid, prediction)
        baseline_score = rmsle(y_valid, baseline_prediction)
        print(f"  RMSLE={score:.6f}; baseline={baseline_score:.6f}")

        metric_rows.append(
            {
                "experiment": label,
                "validation_anchor": fold.validation_anchor.isoformat(),
                "n_features": len(features),
                "n_train_rows": train.height,
                "catboost_rmsle": score,
                "baseline_rmsle": baseline_score,
                "improvement": baseline_score - score,
                "best_iteration": model.get_best_iteration(),
            }
        )
        if keep_oof:
            oof_frames.append(
                pd.DataFrame(
                    {
                        "user_id": valid["user_id"].to_numpy(),
                        "validation_anchor": fold.validation_anchor,
                        "target": y_valid,
                        "prediction": prediction,
                    }
                )
            )
        del train, valid, x_train, y_train, x_valid, y_valid, model
        gc.collect()

    metrics = pd.DataFrame(metric_rows).sort_values("validation_anchor")
    oof = pd.concat(oof_frames, ignore_index=True) if oof_frames else None
    return metrics, oof


def fit_final_model(
    snapshots: dict,
    test_snapshot: pl.DataFrame,
    features: list[str],
    model_factory: Callable[[], CatBoostRegressor],
) -> tuple[CatBoostRegressor, np.ndarray]:
    """Обучает модель на всех размеченных срезах и прогнозирует финальный срез."""
    train = pl.concat(list(snapshots.values()), how="vertical_relaxed")
    x_train = train.select(features).to_pandas()
    y_train = np.log1p(train["target"].to_numpy())
    x_test = test_snapshot.select(features).to_pandas()

    model = model_factory()
    model.fit(x_train, y_train)
    prediction = predict_gmv(model, x_test)
    return model, prediction
