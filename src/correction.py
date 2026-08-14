"""Модели и преобразования для коррекции базового GMV-прогноза."""

import numpy as np
from catboost import CatBoostRegressor

from .config import RANDOM_SEED


def correction_target(target: np.ndarray, base_prediction: np.ndarray) -> np.ndarray:
    """Возвращает ошибку базового прогноза в пространстве log1p."""
    target = np.clip(np.asarray(target, dtype=float), 0, None)
    base_prediction = np.clip(np.asarray(base_prediction, dtype=float), 0, None)
    return np.log1p(target) - np.log1p(base_prediction)


def apply_log_correction(
    base_prediction: np.ndarray,
    predicted_correction: np.ndarray,
    *,
    correction_limit: float = 0.2,
    correction_scale: float = 1.0,
) -> np.ndarray:
    """Добавляет ограниченную поправку к log1p базового прогноза."""
    base_prediction = np.clip(np.asarray(base_prediction, dtype=float), 0, None)
    predicted_correction = np.asarray(predicted_correction, dtype=float)
    bounded = np.clip(
        correction_scale * predicted_correction,
        -correction_limit,
        correction_limit,
    )
    corrected_log = np.maximum(np.log1p(base_prediction) + bounded, 0.0)
    return np.expm1(corrected_log)


def make_correction_model(
    random_seed: int = RANDOM_SEED,
    **overrides,
) -> CatBoostRegressor:
    """Создаёт консервативный неглубокий CatBoost для residual correction."""
    params = {
        "loss_function": "RMSE",
        "iterations": 500,
        "learning_rate": 0.03,
        "depth": 4,
        "l2_leaf_reg": 20.0,
        "random_strength": 0.5,
        "random_seed": random_seed,
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": 100,
    }
    params.update(overrides)
    return CatBoostRegressor(**params)
