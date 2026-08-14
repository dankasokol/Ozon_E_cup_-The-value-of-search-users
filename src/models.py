"""Фабрики CatBoost и преобразование его прогнозов GMV."""

import numpy as np
from catboost import CatBoostRegressor

from .config import RANDOM_SEED


def make_validation_model(
    random_seed: int = RANDOM_SEED,
    **overrides,
) -> CatBoostRegressor:
    """Создаёт модель с early stopping для временной валидации."""
    params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 1800,
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 10.0,
        "random_seed": random_seed,
        "random_strength": 0.5,
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": 200,
    }
    params.update(overrides)
    return CatBoostRegressor(**params)


def make_final_model(
    iterations: int,
    random_seed: int = RANDOM_SEED,
    **overrides,
) -> CatBoostRegressor:
    """Создаёт финальную модель без validation set."""
    params = {
        "loss_function": "RMSE",
        "iterations": iterations,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 10.0,
        "random_seed": random_seed,
        "random_strength": 0.5,
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": 200,
    }
    params.update(overrides)
    return CatBoostRegressor(**params)


def final_iteration_count(best_iteration: int) -> int:
    """Выбирает число деревьев финальной модели по результату holdout."""
    return max(300, (best_iteration if best_iteration > 0 else 1200) + 100)


def predict_gmv(model: CatBoostRegressor, features) -> np.ndarray:
    """Преобразует прогноз log1p(GMV) в неотрицательный GMV."""
    log_prediction = np.maximum(model.predict(features), 0.0)
    return np.expm1(log_prediction)
