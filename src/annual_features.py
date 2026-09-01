"""Годовая устойчивость поведения пользователя.

Признаки строятся только по датам не позже якоря. Они дополняют
180-дневный baseline и отдельный year-over-year профиль: модель получает
информацию о регулярности покупок на горизонте почти года.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl


ANNUAL_DAYS = 365
BLOCK_DAYS = 30
N_BLOCKS = 12
ANNUAL_SUM_COLS = (
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
ANNUAL_ACTIVE_COLS = ("gmv", "to_ord", "to_cart", "searches")
BLOCK_METRICS = ("gmv", "to_ord", "searches")
BLOCK_SUMMARY_NAMES = (
    "positive_share",
    "mean_log1p",
    "std_log1p",
    "median_log1p",
    "max_log1p",
    "slope_log1p",
    "recent3_minus_previous3_log1p",
    "recent3_minus_older9_log1p",
    "recent6_minus_older6_log1p",
)


def annual_feature_names() -> list[str]:
    """Возвращает стабильный порядок годовых признаков."""
    names = ["annual_profile_available", "available_history_days_365d"]
    names.extend(f"{column}_sum_365d" for column in ANNUAL_SUM_COLS)
    names.extend(f"{column}_sum_181_365d" for column in ANNUAL_SUM_COLS)
    names.extend(f"{column}_active_days_365d" for column in ANNUAL_ACTIVE_COLS)
    names.append("observed_sparse_rows_365d")
    for metric in BLOCK_METRICS:
        names.extend(
            f"{metric}_blocks_{summary_name}"
            for summary_name in BLOCK_SUMMARY_NAMES
        )
    names.extend(
        (
            "days_since_last_order_365d",
            "days_since_last_gmv_365d",
            "order_active_span_days_365d",
            "order_gap_mean_days_365d",
            "order_gap_std_days_365d",
            "avg_order_value_365d",
            "search_gmv_share_365d",
            "gmv_recent90_share_365d",
            "orders_recent90_share_365d",
        )
    )
    return names


ANNUAL_FEATURES = tuple(annual_feature_names())


def _mean(expressions: list[pl.Expr]) -> pl.Expr:
    return pl.sum_horizontal(expressions) / float(len(expressions))


def _block_summary_expressions(metric: str) -> list[pl.Expr]:
    """Сжимает 12 непересекающихся 30-дневных блоков в признаки формы."""
    raw = [pl.col(f"_{metric}_block_{index}") for index in range(N_BLOCKS)]
    logged = [value.log1p() for value in raw]
    mean_log = _mean(logged)
    mean_sq_log = _mean([value.pow(2) for value in logged])
    variance = pl.max_horizontal(mean_sq_log - mean_log.pow(2), pl.lit(0.0))
    recent3 = _mean(logged[0:3])
    previous3 = _mean(logged[3:6])
    older9 = _mean(logged[3:12])
    recent6 = _mean(logged[0:6])
    older6 = _mean(logged[6:12])

    # x = 0..11; denominator n*sum(x^2)-sum(x)^2 = 1716.
    weighted_sum = pl.sum_horizontal(
        [pl.lit(float(index)) * value for index, value in enumerate(logged)]
    )
    slope = (12.0 * weighted_sum - 66.0 * pl.sum_horizontal(logged)) / 1716.0

    return [
        (
            pl.sum_horizontal([(value > 0).cast(pl.Float64) for value in raw])
            / float(N_BLOCKS)
        ).alias(f"{metric}_blocks_positive_share"),
        mean_log.alias(f"{metric}_blocks_mean_log1p"),
        variance.sqrt().alias(f"{metric}_blocks_std_log1p"),
        pl.concat_list(logged)
        .list.median()
        .alias(f"{metric}_blocks_median_log1p"),
        pl.max_horizontal(logged).alias(f"{metric}_blocks_max_log1p"),
        slope.alias(f"{metric}_blocks_slope_log1p"),
        (recent3 - previous3).alias(
            f"{metric}_blocks_recent3_minus_previous3_log1p"
        ),
        (recent3 - older9).alias(
            f"{metric}_blocks_recent3_minus_older9_log1p"
        ),
        (recent6 - older6).alias(
            f"{metric}_blocks_recent6_minus_older6_log1p"
        ),
    ]


def build_annual_features(
    data: pl.DataFrame,
    users: pl.DataFrame,
    anchor: date,
    *,
    annual_days: int = ANNUAL_DAYS,
) -> pl.DataFrame:
    """Строит одну строку годовых признаков на пользователя."""
    if annual_days != ANNUAL_DAYS:
        raise ValueError(f"Эксперимент зафиксирован на {ANNUAL_DAYS} днях.")
    if "event_date" not in data.columns or "user_id" not in data.columns:
        raise ValueError("В data нет event_date или user_id.")

    history_start = anchor - timedelta(days=annual_days - 1)
    old_start = history_start
    old_end = anchor - timedelta(days=180)
    recent90_start = anchor - timedelta(days=89)
    data_min = data.select(pl.col("event_date").min()).item()
    available_start = max(history_start, data_min)
    available_history_days = (anchor - available_start).days + 1
    history = data.filter(pl.col("event_date").is_between(history_start, anchor))

    expressions: list[pl.Expr] = [
        pl.len().alias("observed_sparse_rows_365d"),
    ]
    for column in ANNUAL_SUM_COLS:
        expressions.extend(
            (
                pl.col(column).sum().alias(f"{column}_sum_365d"),
                pl.when(pl.col("event_date").is_between(old_start, old_end))
                .then(pl.col(column))
                .otherwise(0.0)
                .sum()
                .alias(f"{column}_sum_181_365d"),
            )
        )
    for column in ANNUAL_ACTIVE_COLS:
        expressions.append(
            (pl.col(column) > 0).sum().alias(f"{column}_active_days_365d")
        )

    for block_index in range(N_BLOCKS):
        block_end = anchor - timedelta(days=BLOCK_DAYS * block_index)
        block_start = block_end - timedelta(days=BLOCK_DAYS - 1)
        in_block = pl.col("event_date").is_between(block_start, block_end)
        for metric in BLOCK_METRICS:
            expressions.append(
                pl.when(in_block)
                .then(pl.col(metric))
                .otherwise(0.0)
                .sum()
                .alias(f"_{metric}_block_{block_index}")
            )

    order_date = pl.col("event_date").filter(pl.col("to_ord") > 0).sort()
    expressions.extend(
        (
            order_date.first().alias("_first_order_date_365d"),
            order_date.last().alias("_last_order_date_365d"),
            order_date.diff().dt.total_days().mean().alias("order_gap_mean_days_365d"),
            order_date.diff().dt.total_days().std(ddof=0).alias("order_gap_std_days_365d"),
            pl.col("event_date")
            .filter(pl.col("gmv") > 0)
            .max()
            .alias("_last_gmv_date_365d"),
            pl.when(pl.col("event_date") >= recent90_start)
            .then(pl.col("gmv"))
            .otherwise(0.0)
            .sum()
            .alias("_gmv_sum_recent90"),
            pl.when(pl.col("event_date") >= recent90_start)
            .then(pl.col("to_ord"))
            .otherwise(0.0)
            .sum()
            .alias("_orders_sum_recent90"),
        )
    )

    grouped = history.group_by("user_id").agg(expressions)
    result = users.join(grouped, on="user_id", how="left", validate="1:1")
    raw_numeric = [
        column
        for column in result.columns
        if column != "user_id" and not column.endswith("_date_365d")
    ]
    result = result.with_columns(
        [pl.col(column).fill_null(0.0) for column in raw_numeric]
    )

    derived: list[pl.Expr] = []
    for metric in BLOCK_METRICS:
        derived.extend(_block_summary_expressions(metric))
    eps = 1.0
    derived.extend(
        (
            pl.when(pl.col("_last_order_date_365d").is_not_null())
            .then((pl.lit(anchor) - pl.col("_last_order_date_365d")).dt.total_days())
            .otherwise(annual_days)
            .alias("days_since_last_order_365d"),
            pl.when(pl.col("_last_gmv_date_365d").is_not_null())
            .then((pl.lit(anchor) - pl.col("_last_gmv_date_365d")).dt.total_days())
            .otherwise(annual_days)
            .alias("days_since_last_gmv_365d"),
            pl.when(
                pl.col("_first_order_date_365d").is_not_null()
                & pl.col("_last_order_date_365d").is_not_null()
            )
            .then(
                (
                    pl.col("_last_order_date_365d")
                    - pl.col("_first_order_date_365d")
                ).dt.total_days()
            )
            .otherwise(0.0)
            .alias("order_active_span_days_365d"),
            (
                pl.col("gmv_sum_365d") / (pl.col("to_ord_sum_365d") + eps)
            ).alias("avg_order_value_365d"),
            (
                pl.col("gmv_search_sum_365d") / (pl.col("gmv_sum_365d") + eps)
            ).alias("search_gmv_share_365d"),
            (
                pl.col("_gmv_sum_recent90") / (pl.col("gmv_sum_365d") + eps)
            ).alias("gmv_recent90_share_365d"),
            (
                pl.col("_orders_sum_recent90")
                / (pl.col("to_ord_sum_365d") + eps)
            ).alias("orders_recent90_share_365d"),
            pl.lit(1.0).alias("annual_profile_available"),
            pl.lit(float(available_history_days)).alias(
                "available_history_days_365d"
            ),
        )
    )
    result = result.with_columns(derived)

    result = result.with_columns(
        pl.col("order_gap_mean_days_365d").fill_null(float(annual_days)),
        pl.col("order_gap_std_days_365d").fill_null(0.0),
    )
    selected = result.select("user_id", *ANNUAL_FEATURES).with_columns(
        [
            pl.col(column)
            .fill_nan(0.0)
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias(column)
            for column in ANNUAL_FEATURES
        ]
    )
    if selected.height != users.height:
        raise AssertionError("Годовой profile изменил число пользователей.")
    if selected["user_id"].n_unique() != selected.height:
        raise AssertionError("В годовом profile есть дубли user_id.")
    return selected
