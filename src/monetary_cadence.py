"""Признаки денежного ритма покупок.

Блок дополняет уже используемый календарный ритм: он отвечает не только на
вопрос «когда пользователь покупает», но и «какие суммы обычно возникают в
последовательности его покупок». Все расчёты используют даты не позже якоря.
"""

from __future__ import annotations

import gc
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl


MONETARY_HISTORY_DAYS = 365

MONETARY_CADENCE_FEATURES = (
    "money_gmv_day_mean",
    "money_gmv_day_std",
    "money_gmv_day_median",
    "money_gmv_day_max",
    "money_gmv_day_q25",
    "money_gmv_day_q75",
    "money_gmv_day_cv",
    "money_aov_day_mean",
    "money_aov_day_std",
    "money_aov_day_median",
    "money_aov_day_max",
    "money_aov_day_q25",
    "money_aov_day_q75",
    "money_aov_day_cv",
    "money_last_gmv",
    "money_previous_gmv",
    "money_third_gmv",
    "money_last_aov",
    "money_previous_aov",
    "money_third_aov",
    "money_last_orders",
    "money_previous_orders",
    "money_third_orders",
    "money_recent3_gmv_mean",
    "money_recent5_gmv_mean",
    "money_recent10_gmv_mean",
    "money_recent3_aov_mean",
    "money_recent5_aov_mean",
    "money_recent10_aov_mean",
    "money_recent3_orders_mean",
    "money_recent5_orders_mean",
    "money_last3_vs_previous3_gmv_log",
    "money_last3_vs_all_gmv_log",
    "money_last3_vs_previous3_aov_log",
    "money_last3_gmv_share",
    "money_decayed_gmv_7d",
    "money_decayed_gmv_14d",
    "money_decayed_gmv_30d",
    "money_decayed_gmv_60d",
    "money_decayed_gmv_90d",
    "money_decayed_orders_7d",
    "money_decayed_orders_14d",
    "money_decayed_orders_30d",
    "money_decayed_orders_60d",
    "money_decayed_orders_90d",
    "money_expected_cycles_30d",
    "money_expected_gmv_by_cycle_30d",
    "money_expected_gmv_by_rate_30d",
    "money_expected_orders_by_rate_30d",
    "money_recency_adjusted_gmv_30d",
    "money_recent5_search_gmv_share",
    "money_recent5_cat_gmv_share",
    "money_recent5_search_share_minus_all",
)

_DECAY_DAYS = (7, 14, 30, 60, 90)


def monetary_profile_path(directory: Path, anchor: date) -> Path:
    """Возвращает устойчивое имя кэшированного профиля одного якоря."""
    return directory / f"monetary_{anchor.isoformat()}.parquet"


def _rank_value(column: str, rank: int) -> pl.Expr:
    return (
        pl.col(column)
        .filter(pl.col("_reverse_rank") == rank)
        .first()
    )


def _recent_mean(column: str, count: int) -> pl.Expr:
    return pl.col(column).filter(pl.col("_reverse_rank") <= count).mean()


def build_monetary_cadence_features(
    data: pl.DataFrame,
    users: pl.DataFrame,
    anchor: date,
    *,
    history_days: int = MONETARY_HISTORY_DAYS,
) -> pl.DataFrame:
    """Строит одну строку денежных характеристик на пользователя.

    Сначала все события сводятся к уровню ``пользователь-день покупки``.
    Поэтому несколько исходных строк одного дня не изображают несколько
    независимых покупательских моментов.
    """
    if history_days != MONETARY_HISTORY_DAYS:
        raise ValueError(f"Опыт зафиксирован на {MONETARY_HISTORY_DAYS} днях.")
    required = {
        "event_date",
        "user_id",
        "to_ord",
        "gmv",
        "gmv_search",
        "gmv_cat",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"В исходных данных нет {sorted(missing)}.")

    data_min = data.select(pl.col("event_date").min()).item()
    requested_start = anchor - timedelta(days=history_days - 1)
    available_start = max(requested_start, data_min)
    available_days = float((anchor - available_start).days + 1)

    purchase_days = (
        data.filter(
            pl.col("event_date").is_between(available_start, anchor)
            & ((pl.col("to_ord") > 0) | (pl.col("gmv") > 0))
        )
        .group_by(["user_id", "event_date"])
        .agg(
            pl.col("to_ord").sum().cast(pl.Float64).alias("_orders"),
            pl.col("gmv").sum().cast(pl.Float64).alias("_gmv"),
            pl.col("gmv_search")
            .sum()
            .cast(pl.Float64)
            .alias("_search_gmv"),
            pl.col("gmv_cat").sum().cast(pl.Float64).alias("_cat_gmv"),
        )
        .sort(["user_id", "event_date"])
        .with_columns(
            (
                pl.col("_gmv")
                / pl.max_horizontal(pl.col("_orders"), pl.lit(1.0))
            ).alias("_aov"),
            (pl.lit(anchor) - pl.col("event_date"))
            .dt.total_days()
            .cast(pl.Float64)
            .alias("_age"),
            pl.col("event_date")
            .rank(method="ordinal", descending=True)
            .over("user_id")
            .alias("_reverse_rank"),
            pl.col("event_date")
            .diff()
            .over("user_id")
            .dt.total_days()
            .cast(pl.Float64)
            .alias("_gap"),
        )
    )

    gmv = pl.col("_gmv")
    aov = pl.col("_aov")
    last3_gmv = gmv.filter(pl.col("_reverse_rank") <= 3)
    previous3_gmv = gmv.filter(pl.col("_reverse_rank").is_between(4, 6))
    last3_aov = aov.filter(pl.col("_reverse_rank") <= 3)
    previous3_aov = aov.filter(pl.col("_reverse_rank").is_between(4, 6))

    aggregations: list[pl.Expr] = [
        gmv.mean().alias("money_gmv_day_mean"),
        gmv.std(ddof=0).alias("money_gmv_day_std"),
        gmv.median().alias("money_gmv_day_median"),
        gmv.max().alias("money_gmv_day_max"),
        gmv.quantile(0.25, interpolation="linear").alias("money_gmv_day_q25"),
        gmv.quantile(0.75, interpolation="linear").alias("money_gmv_day_q75"),
        aov.mean().alias("money_aov_day_mean"),
        aov.std(ddof=0).alias("money_aov_day_std"),
        aov.median().alias("money_aov_day_median"),
        aov.max().alias("money_aov_day_max"),
        aov.quantile(0.25, interpolation="linear").alias("money_aov_day_q25"),
        aov.quantile(0.75, interpolation="linear").alias("money_aov_day_q75"),
        _rank_value("_gmv", 1).alias("money_last_gmv"),
        _rank_value("_gmv", 2).alias("money_previous_gmv"),
        _rank_value("_gmv", 3).alias("money_third_gmv"),
        _rank_value("_aov", 1).alias("money_last_aov"),
        _rank_value("_aov", 2).alias("money_previous_aov"),
        _rank_value("_aov", 3).alias("money_third_aov"),
        _rank_value("_orders", 1).alias("money_last_orders"),
        _rank_value("_orders", 2).alias("money_previous_orders"),
        _rank_value("_orders", 3).alias("money_third_orders"),
        _recent_mean("_gmv", 3).alias("money_recent3_gmv_mean"),
        _recent_mean("_gmv", 5).alias("money_recent5_gmv_mean"),
        _recent_mean("_gmv", 10).alias("money_recent10_gmv_mean"),
        _recent_mean("_aov", 3).alias("money_recent3_aov_mean"),
        _recent_mean("_aov", 5).alias("money_recent5_aov_mean"),
        _recent_mean("_aov", 10).alias("money_recent10_aov_mean"),
        _recent_mean("_orders", 3).alias("money_recent3_orders_mean"),
        _recent_mean("_orders", 5).alias("money_recent5_orders_mean"),
        last3_gmv.sum().alias("_last3_gmv_sum"),
        previous3_gmv.sum().alias("_previous3_gmv_sum"),
        last3_aov.mean().alias("_last3_aov_mean"),
        previous3_aov.mean().alias("_previous3_aov_mean"),
        gmv.sum().alias("_all_gmv_sum"),
        pl.col("_search_gmv").sum().alias("_all_search_gmv_sum"),
        pl.col("_search_gmv")
        .filter(pl.col("_reverse_rank") <= 5)
        .sum()
        .alias("_recent5_search_gmv_sum"),
        pl.col("_cat_gmv")
        .filter(pl.col("_reverse_rank") <= 5)
        .sum()
        .alias("_recent5_cat_gmv_sum"),
        pl.col("_age").min().alias("_recency"),
        pl.col("_gap").drop_nulls().median().alias("_median_gap"),
        pl.len().cast(pl.Float64).alias("_active_days"),
    ]
    for decay_days in _DECAY_DAYS:
        weight = (-pl.col("_age") / float(decay_days)).exp()
        aggregations.extend(
            (
                (pl.col("_gmv") * weight)
                .sum()
                .alias(f"money_decayed_gmv_{decay_days}d"),
                (pl.col("_orders") * weight)
                .sum()
                .alias(f"money_decayed_orders_{decay_days}d"),
            )
        )

    grouped = purchase_days.group_by("user_id", maintain_order=True).agg(
        aggregations
    )
    result = users.join(grouped, on="user_id", how="left", validate="1:1")

    # Нули для отсутствующих покупок — содержательно корректное значение.
    numeric_columns = [column for column in result.columns if column != "user_id"]
    result = result.with_columns(
        [pl.col(column).fill_null(0.0) for column in numeric_columns]
    )

    gmv_mean_denominator = pl.max_horizontal(
        pl.col("money_gmv_day_mean"), pl.lit(1.0)
    )
    aov_mean_denominator = pl.max_horizontal(
        pl.col("money_aov_day_mean"), pl.lit(1.0)
    )
    rate_denominator = pl.max_horizontal(pl.lit(available_days), pl.lit(1.0))
    median_gap_denominator = pl.max_horizontal(pl.col("_median_gap"), pl.lit(1.0))
    recent5_gmv = pl.col("_recent5_search_gmv_sum") + pl.col(
        "_recent5_cat_gmv_sum"
    )
    result = result.with_columns(
        (pl.col("money_gmv_day_std") / gmv_mean_denominator).alias(
            "money_gmv_day_cv"
        ),
        (pl.col("money_aov_day_std") / aov_mean_denominator).alias(
            "money_aov_day_cv"
        ),
        (
            (pl.col("_last3_gmv_sum") + 1.0).log()
            - (pl.col("_previous3_gmv_sum") + 1.0).log()
        ).alias("money_last3_vs_previous3_gmv_log"),
        (
            (pl.col("money_recent3_gmv_mean") + 1.0).log()
            - (pl.col("money_gmv_day_mean") + 1.0).log()
        ).alias("money_last3_vs_all_gmv_log"),
        (
            (pl.col("_last3_aov_mean") + 1.0).log()
            - (pl.col("_previous3_aov_mean") + 1.0).log()
        ).alias("money_last3_vs_previous3_aov_log"),
        (
            pl.col("_last3_gmv_sum")
            / pl.max_horizontal(pl.col("_all_gmv_sum"), pl.lit(1.0))
        ).alias("money_last3_gmv_share"),
        pl.when(pl.col("_active_days") >= 2)
        .then((30.0 / median_gap_denominator).clip(0.0, 30.0))
        .otherwise(0.0)
        .alias("money_expected_cycles_30d"),
        pl.when(pl.col("_active_days") >= 2)
        .then(
            (30.0 / median_gap_denominator).clip(0.0, 30.0)
            * pl.col("money_recent5_gmv_mean")
        )
        .otherwise(0.0)
        .alias("money_expected_gmv_by_cycle_30d"),
        (
            30.0 * pl.col("_all_gmv_sum") / rate_denominator
        ).alias("money_expected_gmv_by_rate_30d"),
        (
            30.0 * pl.col("_active_days") / rate_denominator
        ).alias("money_expected_orders_by_rate_30d"),
        (
            pl.col("_recent5_search_gmv_sum")
            / pl.max_horizontal(recent5_gmv, pl.lit(1.0))
        ).alias("money_recent5_search_gmv_share"),
        (
            pl.col("_recent5_cat_gmv_sum")
            / pl.max_horizontal(recent5_gmv, pl.lit(1.0))
        ).alias("money_recent5_cat_gmv_share"),
        (
            pl.col("_recent5_search_gmv_sum")
            / pl.max_horizontal(recent5_gmv, pl.lit(1.0))
            - pl.col("_all_search_gmv_sum")
            / pl.max_horizontal(pl.col("_all_gmv_sum"), pl.lit(1.0))
        ).alias("money_recent5_search_share_minus_all"),
    )
    result = result.with_columns(
        (
            pl.col("money_expected_gmv_by_cycle_30d")
            * (-pl.col("_recency") / median_gap_denominator).exp()
        ).alias("money_recency_adjusted_gmv_30d")
    )

    result = result.select("user_id", *MONETARY_CADENCE_FEATURES).with_columns(
        [
            pl.col(column).cast(pl.Float32)
            for column in MONETARY_CADENCE_FEATURES
        ]
    )
    if result.height != users.height or result["user_id"].n_unique() != users.height:
        raise AssertionError("Денежный профиль изменил множество пользователей.")
    if not np.isfinite(result.select(MONETARY_CADENCE_FEATURES).to_numpy()).all():
        raise ValueError("В денежных признаках есть NaN или бесконечность.")
    return result


def prepare_monetary_profiles(
    *,
    data: pl.DataFrame,
    users: pl.DataFrame,
    anchors: Iterable[date],
    profile_dir: Path,
    rebuild: bool = False,
) -> dict[date, Path]:
    """Строит недостающие профили и не повторяет уже завершённую работу."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    paths = {anchor: monetary_profile_path(profile_dir, anchor) for anchor in anchors}
    for index, (anchor, path) in enumerate(paths.items(), start=1):
        if path.exists() and not rebuild:
            print(f"[{index}/{len(paths)}] {anchor}: уже готов")
            continue
        print(f"[{index}/{len(paths)}] {anchor}: строю денежный профиль")
        profile = build_monetary_cadence_features(data, users, anchor)
        profile.write_parquet(path, compression="zstd")
        del profile
        gc.collect()
    validate_monetary_profiles(paths, expected_users=users.height)
    return paths


def validate_monetary_profiles(
    paths: dict[date, Path], *, expected_users: int = 250_000
) -> None:
    """Проверяет схему, количество строк и числовую корректность кэша."""
    expected_schema = ["user_id", *MONETARY_CADENCE_FEATURES]
    for anchor, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Нет денежного профиля {anchor}: {path}")
        if list(pl.read_parquet_schema(path)) != expected_schema:
            raise AssertionError(f"Изменилась схема денежного профиля {anchor}.")
        profile = pl.read_parquet(path)
        if (
            profile.height != expected_users
            or profile["user_id"].n_unique() != expected_users
        ):
            raise AssertionError(f"Некорректная детализация профиля {anchor}.")
        values = profile.select(MONETARY_CADENCE_FEATURES).to_numpy()
        if not np.isfinite(values).all():
            raise AssertionError(f"Некорректные числа в профиле {anchor}.")
