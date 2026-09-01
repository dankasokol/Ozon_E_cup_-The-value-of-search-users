"""Загрузка исходных данных."""

import polars as pl

from .config import TRAIN_PATH


def load_train(path=TRAIN_PATH) -> pl.DataFrame:
    """Читает исходные события и стабилизирует типы ключей."""
    if not path.is_file():
        raise FileNotFoundError(f"Не найден исходный файл: {path}")
    return (
        pl.read_parquet(path)
        .with_columns(
            pl.col("event_date").cast(pl.Date),
            pl.col("user_id").cast(pl.Int64),
        )
        .sort(["event_date", "user_id"])
    )


def all_users(data: pl.DataFrame) -> pl.DataFrame:
    """Возвращает всех пользователей, включая неактивных в отдельных окнах."""
    return data.select("user_id").unique().sort("user_id")
