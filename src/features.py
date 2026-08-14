"""Построение пользовательских временных срезов и признаков."""

from datetime import date, timedelta
from pathlib import Path

import polars as pl

from .config import HISTORY_DAYS, HORIZON_DAYS


WINDOWS = (("7d", 6), ("30d", 29), ("90d", 89))
SUM_COLS = (
    "search",
    "cat",
    "searches",
    "to_cart",
    "to_ord",
    "gmv",
    "gmv_search",
    "gmv_cat",
    "search_to_cart",
    "search_to_ord",
    "cat_to_cart",
    "cat_to_ord",
)
MOMENT_COLS = ("gmv", "searches", "to_cart", "to_ord")
ACTIVE_COLS = ("gmv", "searches", "to_cart", "to_ord")
RECENCY_COLS = ("gmv", "to_ord", "to_cart", "searches", "search", "cat")
NON_OVERLAPPING_WINDOWS = (
    ("8_30d", 7, 29),
    ("31_60d", 30, 59),
    ("61_90d", 60, 89),
    ("91_180d", 90, 179),
)
ENHANCED_SUM_COLS = (
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
EPOCH = date(1970, 1, 1)


def build_snapshot(
    data: pl.DataFrame,
    users: pl.DataFrame,
    anchor: date,
    *,
    with_target: bool = True,
    history_days: int = HISTORY_DAYS,
    horizon_days: int = HORIZON_DAYS,
    feature_pack: str = "baseline",
) -> pl.DataFrame:
    """Создаёт одну строку числовых признаков на пользователя для даты ``anchor``.

    В признаки попадают только даты не позже якоря. Если ``with_target=True``,
    добавляется GMV следующих ``horizon_days`` дней исключительно для обучения.
    """
    history_start = anchor - timedelta(days=history_days - 1)
    history = data.filter(pl.col("event_date").is_between(history_start, anchor))

    expressions: list[pl.Expr] = [
        pl.len().alias(f"observed_active_days_{history_days}d")
    ]
    for window_name, offset in WINDOWS:
        start = anchor - timedelta(days=offset)
        in_window = pl.col("event_date").is_between(start, anchor)

        for column in SUM_COLS:
            expressions.append(
                pl.when(in_window)
                .then(pl.col(column))
                .otherwise(0.0)
                .sum()
                .alias(f"{column}_sum_{window_name}")
            )

        for column in MOMENT_COLS:
            expressions.extend(
                (
                    pl.when(in_window)
                    .then(pl.col(column))
                    .otherwise(None)
                    .mean()
                    .alias(f"{column}_mean_active_{window_name}"),
                    pl.when(in_window)
                    .then(pl.col(column))
                    .otherwise(None)
                    .max()
                    .alias(f"{column}_max_{window_name}"),
                )
            )

        for column in ACTIVE_COLS:
            expressions.append(
                pl.when(in_window & (pl.col(column) > 0))
                .then(1)
                .otherwise(0)
                .sum()
                .alias(f"{column}_active_days_{window_name}")
            )

    for column in RECENCY_COLS:
        expressions.append(
            pl.when(pl.col(column) > 0)
            .then(pl.col("event_date").cast(pl.Int32))
            .otherwise(None)
            .max()
            .alias(f"_last_{column}_day")
        )

    grouped = history.group_by("user_id").agg(expressions)
    snapshot = users.join(grouped, on="user_id", how="left")

    raw_feature_cols = [column for column in snapshot.columns if column != "user_id"]
    snapshot = snapshot.with_columns(
        [
            pl.col(column).fill_null(0.0).cast(pl.Float32).alias(column)
            for column in raw_feature_cols
        ]
    )

    anchor_day = (anchor - EPOCH).days
    recency_expressions = [
        (pl.lit(anchor_day) - pl.col(f"_last_{column}_day"))
        .clip(0, history_days)
        .fill_null(history_days)
        .cast(pl.Float32)
        .alias(f"days_since_{column}")
        for column in RECENCY_COLS
    ]
    snapshot = snapshot.with_columns(recency_expressions).drop(
        [f"_last_{column}_day" for column in RECENCY_COLS]
    )

    epsilon = 1.0
    snapshot = snapshot.with_columns(
        [
            (pl.col("to_ord_sum_30d") / (pl.col("to_cart_sum_30d") + epsilon)).alias(
                "cart_to_order_rate_30d"
            ),
            (pl.col("to_cart_sum_30d") / (pl.col("searches_sum_30d") + epsilon)).alias(
                "search_to_cart_rate_30d"
            ),
            (pl.col("to_ord_sum_30d") / (pl.col("searches_sum_30d") + epsilon)).alias(
                "search_to_order_rate_30d"
            ),
            (pl.col("search_to_ord_sum_30d") / (pl.col("search_to_cart_sum_30d") + epsilon)).alias(
                "search_cart_to_order_rate_30d"
            ),
            (pl.col("cat_to_ord_sum_30d") / (pl.col("cat_to_cart_sum_30d") + epsilon)).alias(
                "catalog_cart_to_order_rate_30d"
            ),
            (pl.col("gmv_search_sum_30d") / (pl.col("gmv_sum_30d") + epsilon)).alias(
                "search_gmv_share_30d"
            ),
            (pl.col("search_to_ord_sum_30d") / (pl.col("to_ord_sum_30d") + epsilon)).alias(
                "search_order_share_30d"
            ),
            (pl.col("gmv_sum_30d") / (pl.col("to_ord_sum_30d") + epsilon)).alias(
                "avg_order_value_30d"
            ),
            (7.0 * pl.col("gmv_sum_7d") / (pl.col("gmv_sum_30d") + epsilon)).alias(
                "gmv_recent_intensity_7_vs_30"
            ),
            (30.0 * pl.col("gmv_sum_30d") / (pl.col("gmv_sum_90d") + epsilon)).alias(
                "gmv_recent_intensity_30_vs_90"
            ),
            (7.0 * pl.col("searches_sum_7d") / (pl.col("searches_sum_30d") + epsilon)).alias(
                "search_recent_intensity_7_vs_30"
            ),
            (30.0 * pl.col("searches_sum_30d") / (pl.col("searches_sum_90d") + epsilon)).alias(
                "search_recent_intensity_30_vs_90"
            ),
            (pl.col("gmv_sum_30d") - pl.col("gmv_sum_90d") / 3.0).alias(
                "gmv_trend_30_vs_90"
            ),
            (pl.col("to_ord_sum_30d") - pl.col("to_ord_sum_90d") / 3.0).alias(
                "orders_trend_30_vs_90"
            ),
        ]
    )

    global_30d = (
        history.filter(pl.col("event_date") >= anchor - timedelta(days=29))
        .select(
            pl.col("gmv").sum().alias("global_gmv_sum_30d"),
            pl.col("to_ord").sum().alias("global_orders_sum_30d"),
            pl.col("searches").sum().alias("global_searches_sum_30d"),
        )
        .to_dicts()[0]
    )
    snapshot = snapshot.with_columns(
        [
            pl.lit(float(global_30d["global_gmv_sum_30d"]))
            .cast(pl.Float32)
            .alias("global_gmv_sum_30d"),
            pl.lit(float(global_30d["global_orders_sum_30d"]))
            .cast(pl.Float32)
            .alias("global_orders_sum_30d"),
            pl.lit(float(global_30d["global_searches_sum_30d"]))
            .cast(pl.Float32)
            .alias("global_searches_sum_30d"),
            pl.lit(anchor.month).cast(pl.Int8).alias("anchor_month"),
            pl.lit(anchor.weekday()).cast(pl.Int8).alias("anchor_weekday"),
        ]
    )

    if feature_pack == "enhanced_v2":
        snapshot = _add_enhanced_v2_features(
            snapshot=snapshot,
            history=history,
            anchor=anchor,
            history_days=history_days,
        )
    elif feature_pack != "baseline":
        raise ValueError(
            "Неизвестный набор признаков: "
            f"{feature_pack}. Допустимы: baseline, enhanced_v2."
        )

    if with_target:
        target = (
            data.filter(
                pl.col("event_date").is_between(
                    anchor + timedelta(days=1),
                    anchor + timedelta(days=horizon_days),
                )
            )
            .group_by("user_id")
            .agg(pl.col("gmv").sum().alias("target"))
        )
        snapshot = snapshot.join(target, on="user_id", how="left").with_columns(
            pl.col("target").fill_null(0.0).cast(pl.Float32)
        )

    return snapshot.with_columns(pl.lit(anchor).cast(pl.Date).alias("anchor_date"))


def _add_enhanced_v2_features(
    *,
    snapshot: pl.DataFrame,
    history: pl.DataFrame,
    anchor: date,
    history_days: int,
) -> pl.DataFrame:
    """Добавляет признаки v2: непересекающиеся периоды, регулярность и динамику.

    Каждый признак использует только строки не позднее ``anchor``. Базовые
    признаки не меняются, поэтому v2 можно сравнивать с baseline на тех же
    временных фолдах.
    """
    expressions: list[pl.Expr] = [
        pl.col("event_date").max().cast(pl.Int32).alias("_last_observed_day"),
        pl.col("event_date").min().cast(pl.Int32).alias("_first_observed_day"),
    ]

    for window_name, start_offset, end_offset in NON_OVERLAPPING_WINDOWS:
        start = anchor - timedelta(days=end_offset)
        end = anchor - timedelta(days=start_offset)
        in_window = pl.col("event_date").is_between(start, end)

        expressions.append(
            pl.when(in_window)
            .then(1)
            .otherwise(0)
            .sum()
            .alias(f"observed_days_{window_name}")
        )
        for column in ENHANCED_SUM_COLS:
            expressions.append(
                pl.when(in_window)
                .then(pl.col(column))
                .otherwise(0.0)
                .sum()
                .alias(f"{column}_sum_{window_name}")
            )
        for column in ("gmv", "to_ord", "searches"):
            expressions.extend(
                (
                    pl.when(in_window & (pl.col(column) > 0))
                    .then(1)
                    .otherwise(0)
                    .sum()
                    .alias(f"{column}_active_days_{window_name}"),
                    pl.when(in_window & (pl.col(column) > 0))
                    .then(pl.col(column))
                    .otherwise(None)
                    .mean()
                    .alias(f"{column}_mean_positive_{window_name}"),
                )
            )

    grouped = history.group_by("user_id").agg(expressions)
    snapshot = snapshot.join(grouped, on="user_id", how="left")

    anchor_day = (anchor - EPOCH).days
    snapshot = snapshot.with_columns(
        [
            (pl.lit(anchor_day) - pl.col("_last_observed_day"))
            .clip(0, history_days)
            .fill_null(history_days)
            .cast(pl.Float32)
            .alias("days_since_any_observed_activity"),
            (pl.lit(anchor_day) - pl.col("_first_observed_day") + 1)
            .clip(0, history_days)
            .fill_null(0)
            .cast(pl.Float32)
            .alias("observed_history_span_days"),
        ]
    ).drop(["_last_observed_day", "_first_observed_day"])

    added_cols = [
        column
        for column in snapshot.columns
        if column.endswith(("8_30d", "31_60d", "61_90d", "91_180d"))
    ]
    snapshot = snapshot.with_columns(
        [pl.col(column).fill_null(0.0).cast(pl.Float32).alias(column) for column in added_cols]
    )

    eps = 1.0
    derived_exprs: list[pl.Expr] = []
    for window_name in ("7d", "30d", "90d"):
        derived_exprs.extend(
            (
                (pl.col(f"gmv_sum_{window_name}") / (pl.col(f"gmv_active_days_{window_name}") + eps)).alias(
                    f"gmv_per_gmv_day_{window_name}"
                ),
                (pl.col(f"to_ord_sum_{window_name}") / (pl.col(f"to_ord_active_days_{window_name}") + eps)).alias(
                    f"orders_per_order_day_{window_name}"
                ),
                (pl.col(f"to_ord_sum_{window_name}") / (pl.col(f"to_cart_sum_{window_name}") + eps)).alias(
                    f"cart_to_order_rate_{window_name}"
                ),
                (pl.col(f"search_to_cart_sum_{window_name}") / (pl.col(f"searches_sum_{window_name}") + eps)).alias(
                    f"search_channel_to_cart_rate_{window_name}"
                ),
                (pl.col(f"search_to_ord_sum_{window_name}") / (pl.col(f"searches_sum_{window_name}") + eps)).alias(
                    f"search_channel_to_order_rate_{window_name}"
                ),
                (pl.col(f"cat_to_ord_sum_{window_name}") / (pl.col(f"cat_to_cart_sum_{window_name}") + eps)).alias(
                    f"catalog_cart_to_order_rate_{window_name}"
                ),
                (pl.col(f"gmv_search_sum_{window_name}") / (pl.col(f"gmv_sum_{window_name}") + eps)).alias(
                    f"search_gmv_share_{window_name}"
                ),
                (pl.col(f"search_to_ord_sum_{window_name}") / (pl.col(f"to_ord_sum_{window_name}") + eps)).alias(
                    f"search_order_share_{window_name}"
                ),
                (pl.col(f"gmv_sum_{window_name}") / (pl.col(f"to_ord_sum_{window_name}") + eps)).alias(
                    f"avg_order_value_{window_name}"
                ),
            )
        )

    for metric in ("gmv", "to_ord", "to_cart", "searches"):
        derived_exprs.extend(
            (
                (pl.col(f"{metric}_sum_7d") / 7.0 - pl.col(f"{metric}_sum_8_30d") / 23.0).alias(
                    f"{metric}_daily_trend_7_vs_8_30"
                ),
                (pl.col(f"{metric}_sum_8_30d") / 23.0 - pl.col(f"{metric}_sum_31_60d") / 30.0).alias(
                    f"{metric}_daily_trend_8_30_vs_31_60"
                ),
                (pl.col(f"{metric}_sum_31_60d") / 30.0 - pl.col(f"{metric}_sum_61_90d") / 30.0).alias(
                    f"{metric}_daily_trend_31_60_vs_61_90"
                ),
            )
        )

    derived_exprs.extend(
        (
            (pl.col("gmv_active_days_90d") / (pl.col("observed_active_days_180d") + eps)).alias(
                "gmv_day_share_of_observed_180d"
            ),
            (pl.col("to_ord_active_days_90d") / (pl.col("observed_active_days_180d") + eps)).alias(
                "order_day_share_of_observed_180d"
            ),
            (pl.col("observed_days_8_30d") / 23.0).alias("observed_day_density_8_30d"),
            (pl.col("observed_days_31_60d") / 30.0).alias("observed_day_density_31_60d"),
            (pl.col("observed_days_61_90d") / 30.0).alias("observed_day_density_61_90d"),
            (pl.col("observed_days_91_180d") / 90.0).alias("observed_day_density_91_180d"),
        )
    )
    return snapshot.with_columns(derived_exprs)


def save_snapshot(snapshot: pl.DataFrame, directory: Path, anchor: date, kind: str) -> Path:
    """Сохраняет рассчитанный срез в формат Parquet для повторного использования."""
    path = directory / f"{kind}_{anchor.isoformat()}.parquet"
    snapshot.write_parquet(path, compression="zstd")
    return path


def load_snapshots(directory: Path, kind: str = "train") -> dict[date, pl.DataFrame]:
    """Загружает сохранённые пользовательские срезы, индексируя их по дате якоря."""
    prefix = f"{kind}_"
    paths = sorted(directory.glob(f"{kind}_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"В {directory} нет срезов с префиксом {prefix}")

    snapshots: dict[date, pl.DataFrame] = {}
    for path in paths:
        anchor = date.fromisoformat(path.stem.removeprefix(prefix))
        snapshots[anchor] = pl.read_parquet(path)
    return snapshots
