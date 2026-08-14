"""Year-over-year признаки для сезонного 30-дневного прогноза."""

from datetime import date, timedelta

import polars as pl


FULL_WINDOW_SUM_COLS = (
    "gmv",
    "to_ord",
    "to_cart",
    "searches",
    "search",
    "cat",
    "gmv_search",
    "gmv_cat",
    "search_to_cart",
    "search_to_ord",
    "cat_to_cart",
    "cat_to_ord",
)
SHORT_WINDOW_SUM_COLS = ("gmv", "to_ord", "to_cart", "searches")
SEGMENT_SUM_COLS = ("gmv", "to_ord", "searches")


def shift_year_back(value: date) -> date:
    """Сдвигает календарную дату на год назад с поддержкой 29 февраля."""
    try:
        return value.replace(year=value.year - 1)
    except ValueError:
        return value.replace(year=value.year - 1, day=28)


def seasonal_window_dates(
    anchor: date,
    *,
    horizon_days: int = 30,
    pre_days: int = 14,
) -> dict[str, tuple[date, date]]:
    """Возвращает все календарные окна year-over-year эксперимента."""
    forecast_start = anchor + timedelta(days=1)
    forecast_end = anchor + timedelta(days=horizon_days)
    ly_calendar_start = shift_year_back(forecast_start)
    ly_calendar_end = shift_year_back(forecast_end)
    ly_weekday_start = forecast_start - timedelta(days=364)
    ly_weekday_end = forecast_end - timedelta(days=364)
    current_pre_start = anchor - timedelta(days=pre_days - 1)
    ly_pre_start = shift_year_back(current_pre_start)
    ly_pre_end = shift_year_back(anchor)

    windows: dict[str, tuple[date, date]] = {
        "current_pre_14d": (current_pre_start, anchor),
        "ly_pre_14d": (ly_pre_start, ly_pre_end),
        "ly_calendar_horizon": (ly_calendar_start, ly_calendar_end),
        "ly_weekday_horizon": (ly_weekday_start, ly_weekday_end),
    }
    segment_lengths = (7, 8, 8, 7)
    segment_start = ly_calendar_start
    for index, length in enumerate(segment_lengths, start=1):
        segment_end = segment_start + timedelta(days=length - 1)
        windows[f"ly_segment_{index}"] = (segment_start, segment_end)
        segment_start = segment_end + timedelta(days=1)
    if segment_start - timedelta(days=1) != ly_calendar_end:
        raise AssertionError("Сегменты не покрывают 30-дневный горизонт.")
    return windows


def _sum_expressions(
    *,
    prefix: str,
    start: date,
    end: date,
    columns: tuple[str, ...],
) -> list[pl.Expr]:
    in_window = pl.col("event_date").is_between(start, end)
    return [
        pl.when(in_window)
        .then(pl.col(column))
        .otherwise(0.0)
        .sum()
        .alias(f"{prefix}_{column}")
        for column in columns
    ]


def build_year_over_year_features(
    data: pl.DataFrame,
    users: pl.DataFrame,
    anchor: date,
    *,
    horizon_days: int = 30,
    pre_days: int = 14,
) -> tuple[pl.DataFrame, dict[str, tuple[date, date]]]:
    """Строит прошлогодние аналоги горизонта и признаки изменения масштаба.

    Календарный горизонт сохраняет даты праздников. Горизонт со сдвигом на
    364 дня сохраняет дни недели. Текущие и прошлогодние 14 дней перед
    якорем используются для оценки изменения масштаба пользователя.
    """
    windows = seasonal_window_dates(
        anchor,
        horizon_days=horizon_days,
        pre_days=pre_days,
    )
    relevant = pl.lit(False)
    for start, end in windows.values():
        relevant = relevant | pl.col("event_date").is_between(start, end)
    history = data.filter(relevant)

    expressions: list[pl.Expr] = []
    for name in ("current_pre_14d", "ly_pre_14d"):
        start, end = windows[name]
        expressions.extend(
            _sum_expressions(
                prefix=name,
                start=start,
                end=end,
                columns=SHORT_WINDOW_SUM_COLS,
            )
        )

    start, end = windows["ly_calendar_horizon"]
    expressions.extend(
        _sum_expressions(
            prefix="ly_calendar_horizon",
            start=start,
            end=end,
            columns=FULL_WINDOW_SUM_COLS,
        )
    )
    calendar_mask = pl.col("event_date").is_between(start, end)
    expressions.extend(
        (
            pl.when(calendar_mask).then(1).otherwise(0).sum().alias(
                "ly_calendar_observed_days"
            ),
            pl.when(calendar_mask & (pl.col("gmv") > 0))
            .then(1).otherwise(0).sum().alias("ly_calendar_gmv_days"),
            pl.when(calendar_mask & (pl.col("to_ord") > 0))
            .then(1).otherwise(0).sum().alias("ly_calendar_order_days"),
            pl.when(calendar_mask & (pl.col("searches") > 0))
            .then(1).otherwise(0).sum().alias("ly_calendar_search_days"),
        )
    )

    start, end = windows["ly_weekday_horizon"]
    expressions.extend(
        _sum_expressions(
            prefix="ly_weekday_horizon",
            start=start,
            end=end,
            columns=SHORT_WINDOW_SUM_COLS,
        )
    )
    for index in range(1, 5):
        start, end = windows[f"ly_segment_{index}"]
        expressions.extend(
            _sum_expressions(
                prefix=f"ly_segment_{index}",
                start=start,
                end=end,
                columns=SEGMENT_SUM_COLS,
            )
        )

    grouped = history.group_by("user_id").agg(expressions)
    features = users.join(grouped, on="user_id", how="left")
    value_columns = [column for column in features.columns if column != "user_id"]
    features = features.with_columns(
        [
            pl.col(column).fill_null(0.0).cast(pl.Float32).alias(column)
            for column in value_columns
        ]
    )

    smoothing: dict[str, float] = {}
    for metric in SHORT_WINDOW_SUM_COLS:
        column = f"ly_pre_14d_{metric}"
        median = features.filter(pl.col(column) > 0).select(
            pl.col(column).median()
        ).item()
        smoothing[metric] = max(float(median or 0.0), 1.0)

    scale_expressions: list[pl.Expr] = []
    scaled_expressions: list[pl.Expr] = []
    for metric in SHORT_WINDOW_SUM_COLS:
        scale_name = f"yoy_{metric}_scale_14d"
        scale_expressions.append(
            (
                (pl.col(f"current_pre_14d_{metric}") + smoothing[metric])
                / (pl.col(f"ly_pre_14d_{metric}") + smoothing[metric])
            )
            .clip(0.25, 4.0)
            .cast(pl.Float32)
            .alias(scale_name)
        )
        scaled_expressions.extend(
            (
                (
                    pl.col(f"ly_calendar_horizon_{metric}")
                    * pl.col(scale_name)
                ).alias(f"scaled_ly_calendar_{metric}"),
                (
                    pl.col(f"ly_weekday_horizon_{metric}")
                    * pl.col(scale_name)
                ).alias(f"scaled_ly_weekday_{metric}"),
            )
        )
    features = features.with_columns(scale_expressions).with_columns(
        scaled_expressions
    )

    eps = 1.0
    features = features.with_columns(
        [
            (
                pl.col("ly_calendar_horizon_gmv_search")
                / (pl.col("ly_calendar_horizon_gmv") + eps)
            ).alias("ly_calendar_search_gmv_share"),
            (
                pl.col("ly_calendar_horizon_search_to_ord")
                / (pl.col("ly_calendar_horizon_to_ord") + eps)
            ).alias("ly_calendar_search_order_share"),
            (
                pl.col("ly_calendar_horizon_gmv")
                / (pl.col("ly_calendar_horizon_to_ord") + eps)
            ).alias("ly_calendar_avg_order_value"),
            (pl.col("ly_calendar_horizon_gmv") > 0)
            .cast(pl.Int8).alias("has_ly_horizon_gmv"),
            (pl.col("current_pre_14d_gmv") > 0)
            .cast(pl.Int8).alias("has_current_pre_gmv"),
            (
                (pl.col("current_pre_14d_gmv") > 0)
                & (pl.col("ly_pre_14d_gmv") == 0)
                & (pl.col("ly_calendar_horizon_gmv") == 0)
            ).cast(pl.Int8).alias("is_new_vs_last_year"),
            (
                (pl.col("current_pre_14d_gmv") == 0)
                & (
                    (pl.col("ly_pre_14d_gmv") > 0)
                    | (pl.col("ly_calendar_horizon_gmv") > 0)
                )
            ).cast(pl.Int8).alias("is_lapsed_vs_last_year"),
            pl.lit(anchor).cast(pl.Date).alias("anchor_date"),
        ]
    )
    return features, windows


def daily_market_series(
    data: pl.DataFrame,
    start: date,
    end: date,
) -> pl.DataFrame:
    """Агрегирует основные показатели маркетплейса по календарным дням."""
    return (
        data.filter(pl.col("event_date").is_between(start, end))
        .group_by("event_date")
        .agg(
            pl.col("gmv").sum().alias("gmv"),
            pl.col("to_ord").sum().alias("orders"),
            pl.col("to_cart").sum().alias("to_cart"),
            pl.col("searches").sum().alias("searches"),
            pl.col("gmv_search").sum().alias("gmv_search"),
            pl.col("gmv_cat").sum().alias("gmv_cat"),
        )
        .sort("event_date")
    )
