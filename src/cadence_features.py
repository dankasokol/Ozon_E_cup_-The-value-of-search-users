"""Признаки индивидуального ритма покупок.

Все признаки строятся только по датам не позже якоря. Ритм оценивается
по дням с заказами: мы не восстанавливаем недоступное время внутри дня.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl


CADENCE_HISTORY_DAYS = 365

CADENCE_FEATURES = (
    "cadence_available_history_days",
    "order_cadence_active_days",
    "order_cadence_total_orders",
    "order_cadence_multiple_order_day_share",
    "order_cadence_orders_per_active_day_mean",
    "order_cadence_orders_per_active_day_std",
    "order_cadence_orders_per_active_day_max",
    "order_cadence_orders_on_last_active_day",
    "order_cadence_active_span_days",
    "order_cadence_gap_count",
    "order_cadence_gap_last",
    "order_cadence_gap_previous",
    "order_cadence_gap_third",
    "order_cadence_gap_mean",
    "order_cadence_gap_median",
    "order_cadence_gap_std",
    "order_cadence_gap_min",
    "order_cadence_gap_max",
    "order_cadence_gap_q25",
    "order_cadence_gap_q75",
    "order_cadence_gap_iqr",
    "order_cadence_gap_cv",
    "order_cadence_gap_recent3_mean",
    "order_cadence_gap_recent5_mean",
    "order_cadence_recent3_vs_all",
    "order_cadence_recency_days",
    "order_cadence_recency_over_median",
    "order_cadence_recency_minus_median",
    "order_cadence_days_until_expected",
    "order_cadence_missed_cycles",
    "order_cadence_regularity_score",
    "order_cadence_weekday_entropy",
    "order_cadence_weekday_max_share",
    "order_cadence_dominant_weekday",
)


def build_cadence_features(
    data: pl.DataFrame,
    users: pl.DataFrame,
    anchor: date,
    *,
    history_days: int = CADENCE_HISTORY_DAYS,
) -> pl.DataFrame:
    """Строит одну строку ритма заказов на пользователя."""
    if history_days != CADENCE_HISTORY_DAYS:
        raise ValueError(f"Опыт зафиксирован на {CADENCE_HISTORY_DAYS} днях.")
    required = {"event_date", "user_id", "to_ord"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"В исходных данных нет {sorted(missing)}.")

    data_min = data.select(pl.col("event_date").min()).item()
    requested_start = anchor - timedelta(days=history_days - 1)
    available_start = max(requested_start, data_min)
    available_days = (anchor - available_start).days + 1
    history = data.filter(pl.col("event_date").is_between(available_start, anchor))

    # В исходнике допускается несколько строк user-day. Сначала
    # сводим их к одному дню, затем считаем паузы между активными днями.
    order_days = (
        history.filter(pl.col("to_ord") > 0)
        .group_by(["user_id", "event_date"])
        .agg(pl.col("to_ord").sum().cast(pl.Float64).alias("_orders_day"))
        .sort(["user_id", "event_date"])
        .with_columns(
            pl.col("event_date")
            .diff()
            .over("user_id")
            .dt.total_days()
            .cast(pl.Float64)
            .alias("_gap")
        )
    )

    gap = pl.col("_gap").drop_nulls()
    weekday_counts = [
        pl.when(pl.col("event_date").dt.weekday() == weekday)
        .then(pl.col("_orders_day"))
        .otherwise(0.0)
        .sum()
        .alias(f"_weekday_{weekday}")
        for weekday in range(1, 8)
    ]
    grouped = order_days.group_by("user_id", maintain_order=True).agg(
        pl.len().cast(pl.Float64).alias("order_cadence_active_days"),
        pl.col("_orders_day").sum().alias("order_cadence_total_orders"),
        (pl.col("_orders_day") > 1)
        .mean()
        .alias("order_cadence_multiple_order_day_share"),
        pl.col("_orders_day")
        .mean()
        .alias("order_cadence_orders_per_active_day_mean"),
        pl.col("_orders_day")
        .std(ddof=0)
        .alias("order_cadence_orders_per_active_day_std"),
        pl.col("_orders_day")
        .max()
        .alias("order_cadence_orders_per_active_day_max"),
        pl.col("_orders_day")
        .last()
        .alias("order_cadence_orders_on_last_active_day"),
        (pl.col("event_date").last() - pl.col("event_date").first())
        .dt.total_days()
        .cast(pl.Float64)
        .alias("order_cadence_active_span_days"),
        gap.len().cast(pl.Float64).alias("order_cadence_gap_count"),
        gap.last().alias("order_cadence_gap_last"),
        gap.slice(-2, 1).first().alias("order_cadence_gap_previous"),
        gap.slice(-3, 1).first().alias("order_cadence_gap_third"),
        gap.mean().alias("order_cadence_gap_mean"),
        gap.median().alias("order_cadence_gap_median"),
        gap.std(ddof=0).alias("order_cadence_gap_std"),
        gap.min().alias("order_cadence_gap_min"),
        gap.max().alias("order_cadence_gap_max"),
        gap.quantile(0.25, interpolation="linear").alias("order_cadence_gap_q25"),
        gap.quantile(0.75, interpolation="linear").alias("order_cadence_gap_q75"),
        gap.tail(3).mean().alias("order_cadence_gap_recent3_mean"),
        gap.tail(5).mean().alias("order_cadence_gap_recent5_mean"),
        pl.col("event_date").last().alias("_last_order_date"),
        *weekday_counts,
    )

    result = users.join(grouped, on="user_id", how="left", validate="1:1")
    zero_fill = (
        "order_cadence_active_days",
        "order_cadence_total_orders",
        "order_cadence_multiple_order_day_share",
        "order_cadence_orders_per_active_day_mean",
        "order_cadence_orders_per_active_day_std",
        "order_cadence_orders_per_active_day_max",
        "order_cadence_orders_on_last_active_day",
        "order_cadence_active_span_days",
        "order_cadence_gap_count",
    )
    gap_fill = (
        "order_cadence_gap_last",
        "order_cadence_gap_previous",
        "order_cadence_gap_third",
        "order_cadence_gap_mean",
        "order_cadence_gap_median",
        "order_cadence_gap_min",
        "order_cadence_gap_max",
        "order_cadence_gap_q25",
        "order_cadence_gap_q75",
        "order_cadence_gap_recent3_mean",
        "order_cadence_gap_recent5_mean",
    )
    result = result.with_columns(
        *[pl.col(column).fill_null(0.0) for column in zero_fill],
        *[
            pl.col(column).fill_null(float(available_days))
            for column in gap_fill
        ],
        pl.col("order_cadence_gap_std").fill_null(0.0),
        *[
            pl.col(f"_weekday_{weekday}").fill_null(0.0)
            for weekday in range(1, 8)
        ],
    )

    median_denominator = pl.max_horizontal(
        pl.col("order_cadence_gap_median"), pl.lit(1.0)
    )
    mean_denominator = pl.max_horizontal(
        pl.col("order_cadence_gap_mean"), pl.lit(1.0)
    )
    recency = (
        pl.when(pl.col("_last_order_date").is_not_null())
        .then((pl.lit(anchor) - pl.col("_last_order_date")).dt.total_days())
        .otherwise(available_days)
        .cast(pl.Float64)
    )
    total_orders_denominator = pl.max_horizontal(
        pl.col("order_cadence_total_orders"), pl.lit(1.0)
    )
    weekday_shares = [
        pl.col(f"_weekday_{weekday}") / total_orders_denominator
        for weekday in range(1, 8)
    ]
    entropy_terms = [
        pl.when(share > 0).then(-share * share.log()).otherwise(0.0)
        for share in weekday_shares
    ]
    result = result.with_columns(
        pl.lit(float(available_days)).alias("cadence_available_history_days"),
        (
            pl.col("order_cadence_gap_q75")
            - pl.col("order_cadence_gap_q25")
        ).alias("order_cadence_gap_iqr"),
        (
            pl.col("order_cadence_gap_std") / mean_denominator
        ).alias("order_cadence_gap_cv"),
        (
            pl.col("order_cadence_gap_recent3_mean") / mean_denominator
        ).alias("order_cadence_recent3_vs_all"),
        recency.alias("order_cadence_recency_days"),
        (recency / median_denominator).alias(
            "order_cadence_recency_over_median"
        ),
        (recency - pl.col("order_cadence_gap_median")).alias(
            "order_cadence_recency_minus_median"
        ),
        (pl.col("order_cadence_gap_median") - recency).alias(
            "order_cadence_days_until_expected"
        ),
        (recency / median_denominator)
        .floor()
        .clip(0.0, 20.0)
        .alias("order_cadence_missed_cycles"),
        (1.0 / (1.0 + pl.col("order_cadence_gap_std") / mean_denominator)).alias(
            "order_cadence_regularity_score"
        ),
        (pl.sum_horizontal(entropy_terms) / math.log(7.0)).alias(
            "order_cadence_weekday_entropy"
        ),
        pl.max_horizontal(weekday_shares).alias(
            "order_cadence_weekday_max_share"
        ),
        pl.concat_list([pl.col(f"_weekday_{weekday}") for weekday in range(1, 8)])
        .list.arg_max()
        .fill_null(0)
        .cast(pl.Float64)
        .alias("order_cadence_dominant_weekday"),
    )

    result = result.select("user_id", *CADENCE_FEATURES).with_columns(
        [pl.col(column).cast(pl.Float32) for column in CADENCE_FEATURES]
    )
    if result.height != users.height or result["user_id"].n_unique() != users.height:
        raise AssertionError("Ритм-профиль изменил множество пользователей.")
    values = result.select(CADENCE_FEATURES).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("В ритм-профиле есть NaN или бесконечность.")
    return result


