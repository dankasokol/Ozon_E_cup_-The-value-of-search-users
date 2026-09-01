"""Профиль поведения пользователя в аналогичном сезоне год назад."""

from datetime import date, timedelta
from math import log

import polars as pl


SEGMENT_SUM_COLS = ("gmv", "to_ord", "searches")
PROFILE_PRE_WINDOW_DAYS = (7, 14)
PROFILE_PRE_WINDOW_COLS = ("gmv", "to_ord", "searches")


def shift_year_back(value: date) -> date:
    """Сдвигает календарную дату на год назад с поддержкой 29 февраля."""
    try:
        return value.replace(year=value.year - 1)
    except ValueError:
        return value.replace(year=value.year - 1, day=28)


def seasonal_analog_profile_windows(
    anchor: date,
    *,
    horizon_days: int = 30,
) -> dict[str, tuple[date, date]]:
    """Возвращает окна для подробного профиля аналогичного сезона."""
    forecast_start = anchor + timedelta(days=1)
    forecast_end = anchor + timedelta(days=horizon_days)
    windows: dict[str, tuple[date, date]] = {
        "ly_horizon": (
            shift_year_back(forecast_start),
            shift_year_back(forecast_end),
        ),
    }
    for days in PROFILE_PRE_WINDOW_DAYS:
        current_start = anchor - timedelta(days=days - 1)
        windows[f"current_pre_{days}d"] = (current_start, anchor)
        windows[f"ly_pre_{days}d"] = (
            shift_year_back(current_start),
            shift_year_back(anchor),
        )
    return windows


def build_seasonal_analog_profile(
    data: pl.DataFrame,
    users: pl.DataFrame,
    anchor: date,
    *,
    horizon_days: int = 30,
) -> tuple[pl.DataFrame, dict[str, tuple[date, date]]]:
    """Описывает подробный профиль аналогичного горизонта год назад.

    Признаки описывают масштаб, концентрацию и временную форму GMV,
    а также текущие и прошлогодние периоды перед прогнозом. Все
    окна заканчиваются не позже ``anchor`` и поэтому доступны в момент
    прогноза.
    """
    windows = seasonal_analog_profile_windows(
        anchor,
        horizon_days=horizon_days,
    )
    ly_start, ly_end = windows["ly_horizon"]

    relevant_mask = pl.lit(False)
    for start, end in windows.values():
        relevant_mask = relevant_mask | pl.col("event_date").is_between(start, end)
    relevant = data.filter(relevant_mask)

    profile = (
        relevant.filter(pl.col("event_date").is_between(ly_start, ly_end))
        .with_columns(
            (
                pl.col("event_date") - pl.lit(ly_start)
            ).dt.total_days().cast(pl.Int16).alias("ly_day_offset"),
            pl.col("event_date").dt.weekday().is_in([6, 7]).alias("is_weekend"),
        )
    )
    global_daily = (
        profile.group_by("ly_day_offset")
        .agg(pl.col("gmv").sum().alias("global_day_gmv"))
        .sort("ly_day_offset")
        .with_columns(
            (
                pl.col("global_day_gmv") / pl.col("global_day_gmv").sum()
            ).alias("global_day_gmv_weight")
        )
    )
    global_gmv_norm_sq = float(
        global_daily.select(pl.col("global_day_gmv").pow(2).sum()).item() or 0.0
    )
    profile = profile.join(global_daily, on="ly_day_offset", how="left")

    segment_bounds = ((0, 6), (7, 14), (15, 22), (23, 29))
    segment_expressions: list[pl.Expr] = []
    for index, (left, right) in enumerate(segment_bounds, start=1):
        in_segment = pl.col("ly_day_offset").is_between(left, right)
        for metric in SEGMENT_SUM_COLS:
            segment_expressions.append(
                pl.when(in_segment)
                .then(pl.col(metric))
                .otherwise(0.0)
                .sum()
                .alias(f"ly_profile_segment_{index}_{metric}")
            )

    positive_gmv = pl.col("gmv") > 0
    profile_grouped = profile.group_by("user_id").agg(
        pl.len().alias("ly_profile_observed_days"),
        pl.col("gmv").sum().alias("ly_profile_gmv"),
        pl.col("gmv").pow(2).sum().alias("ly_profile_gmv_sq_sum"),
        pl.when(positive_gmv)
        .then(pl.col("gmv") * pl.col("gmv").log())
        .otherwise(0.0)
        .sum()
        .alias("ly_profile_gmv_xlogx_sum"),
        pl.col("gmv").max().alias("ly_profile_gmv_max_day"),
        pl.col("gmv").filter(positive_gmv).median().alias(
            "ly_profile_gmv_median_positive_day"
        ),
        pl.col("gmv").top_k(3).sum().alias("ly_profile_gmv_top3_sum"),
        positive_gmv.sum().alias("ly_profile_gmv_days"),
        (pl.col("to_ord") > 0).sum().alias("ly_profile_order_days"),
        (pl.col("searches") > 0).sum().alias("ly_profile_search_days"),
        pl.col("to_ord").sum().alias("ly_profile_orders"),
        pl.col("searches").sum().alias("ly_profile_searches"),
        pl.when(positive_gmv).then(pl.col("ly_day_offset")).min().alias(
            "ly_profile_first_gmv_offset"
        ),
        pl.when(positive_gmv).then(pl.col("ly_day_offset")).max().alias(
            "ly_profile_last_gmv_offset"
        ),
        pl.col("ly_day_offset").sort_by("gmv", descending=True).first().alias(
            "ly_profile_peak_gmv_offset"
        ),
        pl.when(pl.col("is_weekend"))
        .then(pl.col("gmv"))
        .otherwise(0.0)
        .sum()
        .alias("ly_profile_weekend_gmv"),
        (pl.col("gmv") * pl.col("global_day_gmv"))
        .sum()
        .alias("ly_profile_global_cosine_numerator"),
        (pl.col("gmv") * pl.col("global_day_gmv_weight"))
        .sum()
        .alias("ly_profile_global_weighted_gmv"),
        pl.when(pl.col("ly_day_offset") == 0)
        .then(pl.col("gmv"))
        .otherwise(0.0)
        .sum()
        .alias("ly_profile_day_0_gmv"),
        pl.when(pl.col("ly_day_offset").is_between(7, 11))
        .then(pl.col("gmv"))
        .otherwise(0.0)
        .sum()
        .alias("ly_profile_days_7_11_gmv"),
        pl.when(pl.col("ly_day_offset").is_between(20, 24))
        .then(pl.col("gmv"))
        .otherwise(0.0)
        .sum()
        .alias("ly_profile_days_20_24_gmv"),
        *segment_expressions,
    )

    pre_expressions: list[pl.Expr] = []
    for days in PROFILE_PRE_WINDOW_DAYS:
        for prefix in ("current", "ly"):
            start, end = windows[f"{prefix}_pre_{days}d"]
            in_window = pl.col("event_date").is_between(start, end)
            for metric in PROFILE_PRE_WINDOW_COLS:
                pre_expressions.append(
                    pl.when(in_window)
                    .then(pl.col(metric))
                    .otherwise(0.0)
                    .sum()
                    .alias(f"{prefix}_pre_{days}d_{metric}")
                )
    pre_grouped = relevant.group_by("user_id").agg(pre_expressions)

    features = (
        users.join(profile_grouped, on="user_id", how="left")
        .join(pre_grouped, on="user_id", how="left")
    )
    value_columns = [column for column in features.columns if column != "user_id"]
    features = features.with_columns(
        [
            pl.col(column).fill_null(-1.0 if column.endswith("_offset") else 0.0)
            .cast(pl.Float32)
            .alias(column)
            for column in value_columns
        ]
    )

    eps = 1e-6
    gmv_mean = pl.col("ly_profile_gmv") / float(horizon_days)
    gmv_variance = pl.max_horizontal(
        pl.col("ly_profile_gmv_sq_sum") / float(horizon_days) - gmv_mean.pow(2),
        pl.lit(0.0),
    )
    derived = [
        gmv_mean.alias("ly_profile_gmv_mean_day"),
        gmv_variance.sqrt().alias("ly_profile_gmv_std_day"),
        (gmv_variance.sqrt() / (gmv_mean + eps)).alias("ly_profile_gmv_cv"),
        (
            pl.col("ly_profile_gmv_sq_sum")
            / (pl.col("ly_profile_gmv").pow(2) + eps)
        ).alias("ly_profile_gmv_hhi"),
        pl.when(pl.col("ly_profile_gmv") > 0)
        .then(
            (
                pl.col("ly_profile_gmv").log()
                - pl.col("ly_profile_gmv_xlogx_sum")
                / pl.col("ly_profile_gmv")
            ) / float(log(horizon_days))
        )
        .otherwise(0.0)
        .clip(0.0, 1.0)
        .alias("ly_profile_gmv_entropy_normalized"),
        (
            pl.col("ly_profile_gmv_max_day")
            / (pl.col("ly_profile_gmv") + eps)
        ).alias("ly_profile_top1_gmv_share"),
        (
            pl.col("ly_profile_gmv_top3_sum")
            / (pl.col("ly_profile_gmv") + eps)
        ).alias("ly_profile_top3_gmv_share"),
        (
            pl.col("ly_profile_weekend_gmv")
            / (pl.col("ly_profile_gmv") + eps)
        ).alias("ly_profile_weekend_gmv_share"),
        (
            pl.col("ly_profile_gmv") / (pl.col("ly_profile_orders") + 1.0)
        ).alias("ly_profile_avg_order_value"),
        (
            pl.col("ly_profile_orders")
            / (pl.col("ly_profile_searches") + 1.0)
        ).alias("ly_profile_search_to_order"),
        pl.when(pl.col("ly_profile_gmv_days") > 0)
        .then(
            pl.col("ly_profile_last_gmv_offset")
            - pl.col("ly_profile_first_gmv_offset")
            + 1.0
        )
        .otherwise(0.0)
        .alias("ly_profile_active_span_days"),
        pl.when(pl.col("ly_profile_gmv_days") > 1)
        .then(
            (
                pl.col("ly_profile_last_gmv_offset")
                - pl.col("ly_profile_first_gmv_offset")
            ) / (pl.col("ly_profile_gmv_days") - 1.0)
        )
        .otherwise(0.0)
        .alias("ly_profile_mean_gap_days"),
        pl.when(pl.col("ly_profile_gmv_sq_sum") > 0)
        .then(
            pl.col("ly_profile_global_cosine_numerator")
            / (
                (
                    pl.col("ly_profile_gmv_sq_sum")
                    * global_gmv_norm_sq
                ).sqrt()
                + eps
            )
        )
        .otherwise(0.0)
        .alias("ly_profile_global_cosine"),
        pl.when(pl.col("ly_profile_gmv") > 0)
        .then(
            pl.col("ly_profile_global_weighted_gmv")
            / pl.col("ly_profile_gmv")
            * float(horizon_days)
        )
        .otherwise(0.0)
        .alias("ly_profile_global_intensity"),
    ]
    for index in range(1, 5):
        derived.append(
            (
                pl.col(f"ly_profile_segment_{index}_gmv")
                / (pl.col("ly_profile_gmv") + eps)
            ).alias(f"ly_profile_segment_{index}_gmv_share")
        )

    features = features.with_columns(derived).with_columns(
        pl.when(pl.col("ly_profile_gmv") <= 0)
        .then(pl.lit(0))
        .when(pl.col("ly_profile_gmv_days") <= 1)
        .then(pl.lit(1))
        .when(pl.col("ly_profile_gmv_days") <= 3)
        .then(pl.lit(2))
        .otherwise(pl.lit(3))
        .cast(pl.Int8)
        .alias("ly_profile_reliability_segment"),
        pl.lit(anchor).cast(pl.Date).alias("anchor_date"),
    )
    return features, windows
