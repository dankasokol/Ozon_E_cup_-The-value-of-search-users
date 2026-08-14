"""Загрузка и базовые проверки исходных данных."""

from datetime import date, timedelta

import polars as pl

from .config import HORIZON_DAYS, TRAIN_PATH


def load_train(path=TRAIN_PATH) -> pl.DataFrame:
    """Читает исходные данные и приводит ключевые колонки к стабильным типам."""
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


def summarize_data(data: pl.DataFrame) -> dict[str, object]:
    """Возвращает краткую сводку и два контроля согласованности воронки."""
    return data.select(
        pl.len().alias("rows"),
        pl.col("user_id").n_unique().alias("users"),
        pl.col("event_date").min().alias("min_date"),
        pl.col("event_date").max().alias("max_date"),
        pl.col("event_date").n_unique().alias("days"),
        (pl.col("gmv") > 0).mean().alias("positive_gmv_row_share"),
        (pl.col("gmv") - pl.col("gmv_search") - pl.col("gmv_cat"))
        .abs()
        .max()
        .alias("max_gmv_identity_error"),
        (pl.col("to_ord") - pl.col("search_to_ord") - pl.col("cat_to_ord"))
        .abs()
        .max()
        .alias("max_order_identity_error"),
    ).to_dicts()[0]


def duplicate_user_day_count(data: pl.DataFrame) -> int:
    """Считает дубли пар «пользователь — дата» до построения агрегатов."""
    return (
        data.group_by(["user_id", "event_date"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )


def validate_user_days(data: pl.DataFrame) -> None:
    """Останавливает расчёт, если одна пользовательская дата встречается дважды."""
    duplicate_count = duplicate_user_day_count(data)
    if duplicate_count:
        raise ValueError(
            "Найдены дубликаты user_id × event_date; их нужно обработать до обучения. "
            f"Число дублирующихся пар: {duplicate_count}."
        )


def all_users(data: pl.DataFrame) -> pl.DataFrame:
    """Возвращает всех пользователей, включая неактивных в отдельных окнах."""
    return data.select("user_id").unique().sort("user_id")


def latest_labeled_anchor(data: pl.DataFrame, horizon_days: int = HORIZON_DAYS) -> date:
    """Последний якорь, для которого в истории наблюдаются все дни будущей цели."""
    max_date = data.select(pl.col("event_date").max()).item()
    return max_date - timedelta(days=horizon_days)
