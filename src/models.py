"""Финальная модель CatBoost и обратное преобразование GMV."""

import numpy as np
from catboost import CatBoostRegressor

from .config import RANDOM_SEED


def make_final_model(
    iterations: int,
    random_seed: int = RANDOM_SEED,
    **overrides,
) -> CatBoostRegressor:
    """Создаёт зафиксированную финальную модель без ранней остановки."""
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


def predict_gmv(model: CatBoostRegressor, features) -> np.ndarray:
    """Преобразует прогноз log1p(GMV) в неотрицательный GMV."""
    log_prediction = np.maximum(model.predict(features), 0.0)
    return np.expm1(log_prediction)
