"""Самодостаточный конвейер текущего финального решения Ozon GMV.

Из исходного ``data/train.parquet`` модуль строит все необходимые временные
срезы и профили, обучает две модели CatBoost и сохраняет их логарифмическую
смесь. Никакие артефакты отклонённых экспериментов не требуются.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl
from catboost import Pool

from .annual_features import ANNUAL_FEATURES, build_annual_features
from .cadence_features import CADENCE_FEATURES, build_cadence_features
from .config import (
    FINAL_SOLUTION_CACHE_DIR,
    FINAL_SOLUTION_DIR,
    FINAL_SUBMISSION_PATH,
    TRAIN_PATH,
)
from .data import all_users, load_train
from .features import build_snapshot
from .models import make_final_model, predict_gmv
from .seasonality import build_seasonal_analog_profile


HISTORICAL_ANCHORS = (
    date(2025, 7, 2),
    date(2025, 7, 30),
    date(2025, 8, 27),
    date(2025, 9, 24),
    date(2025, 10, 22),
    date(2025, 11, 19),
    date(2025, 12, 17),
)
JANUARY_ANCHOR = date(2026, 1, 14)
TEST_ANCHOR = date(2026, 2, 13)
ALL_ANCHORS = (*HISTORICAL_ANCHORS, JANUARY_ANCHOR, TEST_ANCHOR)

BASE_FEATURE_COUNT = 216
SEASONAL_FEATURE_COUNT = 58
ANNUAL_FEATURE_COUNT = 67
CADENCE_FEATURE_COUNT = 34
ANNUAL_ITERATIONS = 1061
CADENCE_ITERATIONS = 1171
ANNUAL_WEIGHT = 0.25
CADENCE_WEIGHT = 0.75


@dataclass(frozen=True)
class FinalInputPaths:
    """Все производные входы финального обучения."""

    snapshots: dict[date, Path]
    seasonal_profiles: dict[date, Path]
    annual_profiles: dict[date, Path]
    cadence_profiles: dict[date, Path]


@dataclass
class SnapshotBlocks:
    """Числовые блоки одного пользовательского временного среза."""

    user_id: np.ndarray
    target: np.ndarray
    base: np.ndarray
    seasonal: np.ndarray
    annual: np.ndarray
    cadence: np.ndarray


def final_input_paths(cache_dir: Path = FINAL_SOLUTION_CACHE_DIR) -> FinalInputPaths:
    """Возвращает единый контракт путей производных входов."""
    snapshot_dir = cache_dir / "snapshots"
    seasonal_dir = cache_dir / "seasonal"
    annual_dir = cache_dir / "annual"
    cadence_dir = cache_dir / "cadence"
    snapshots = {
        anchor: snapshot_dir
        / (
            f"test_{anchor.isoformat()}.parquet"
            if anchor == TEST_ANCHOR
            else f"train_{anchor.isoformat()}.parquet"
        )
        for anchor in ALL_ANCHORS
    }
    return FinalInputPaths(
        snapshots=snapshots,
        seasonal_profiles={
            anchor: seasonal_dir / f"seasonal_{anchor.isoformat()}.parquet"
            for anchor in (JANUARY_ANCHOR, TEST_ANCHOR)
        },
        annual_profiles={
            anchor: annual_dir / f"annual_{anchor.isoformat()}.parquet"
            for anchor in (JANUARY_ANCHOR, TEST_ANCHOR)
        },
        cadence_profiles={
            anchor: cadence_dir / f"cadence_{anchor.isoformat()}.parquet"
            for anchor in ALL_ANCHORS
        },
    )


def seasonal_feature_columns(schema: dict[str, pl.DataType]) -> list[str]:
    """Выбирает 58 прошлогодних признаков, использованных финальной моделью."""
    columns = [
        column
        for column in schema
        if column not in {"user_id", "anchor_date"}
        and not column.startswith("current_pre_")
    ]
    if (
        len(columns) != SEASONAL_FEATURE_COUNT
        or "ly_profile_reliability_segment" not in columns
    ):
        raise AssertionError(
            f"Ожидалось {SEASONAL_FEATURE_COUNT} сезонных признаков, "
            f"получено {len(columns)}."
        )
    return columns


def _all_input_files(paths: FinalInputPaths) -> list[Path]:
    return [
        *paths.snapshots.values(),
        *paths.seasonal_profiles.values(),
        *paths.annual_profiles.values(),
        *paths.cadence_profiles.values(),
    ]


def _validate_input_contract(paths: FinalInputPaths) -> None:
    """Проверяет схемы и детализацию всех производных входов."""
    missing = [str(path) for path in _all_input_files(paths) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Не построены входы: {missing}")

    january_schema = pl.read_parquet_schema(paths.snapshots[JANUARY_ANCHOR])
    base_features = [
        column
        for column in january_schema
        if column not in {"user_id", "anchor_date", "target"}
    ]
    if len(base_features) != BASE_FEATURE_COUNT:
        raise AssertionError("Изменилась схема базовых признаков.")
    for anchor, path in paths.snapshots.items():
        schema = pl.read_parquet_schema(path)
        expected_tail = {"user_id", "anchor_date"}
        if anchor != TEST_ANCHOR:
            expected_tail.add("target")
        if not expected_tail.issubset(schema):
            raise AssertionError(f"Некорректная схема временного среза {anchor}.")
        check = pl.scan_parquet(path).select(
            pl.len().alias("rows"), pl.col("user_id").n_unique().alias("users")
        ).collect().row(0)
        if check != (250_000, 250_000):
            raise AssertionError(f"Некорректная детализация среза {anchor}: {check}")

    for anchor, path in paths.seasonal_profiles.items():
        columns = seasonal_feature_columns(pl.read_parquet_schema(path))
        if len(columns) != SEASONAL_FEATURE_COUNT:
            raise AssertionError(f"Некорректный сезонный профиль {anchor}.")
    for anchor, path in paths.annual_profiles.items():
        if list(pl.read_parquet_schema(path)) != ["user_id", *ANNUAL_FEATURES]:
            raise AssertionError(f"Некорректный годовой профиль {anchor}.")
    for anchor, path in paths.cadence_profiles.items():
        if list(pl.read_parquet_schema(path)) != ["user_id", *CADENCE_FEATURES]:
            raise AssertionError(f"Некорректный профиль ритма {anchor}.")


def prepare_final_inputs(
    *,
    train_path: Path = TRAIN_PATH,
    cache_dir: Path = FINAL_SOLUTION_CACHE_DIR,
    rebuild: bool = False,
    data: pl.DataFrame | None = None,
    users: pl.DataFrame | None = None,
) -> FinalInputPaths:
    """Строит все входы финального решения из исходных событий.

    Готовые ``data`` и ``users`` можно передать из внешнего опыта, чтобы не
    читать исходный Parquet повторно.
    """
    paths = final_input_paths(cache_dir)
    for path in _all_input_files(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
    missing = [path for path in _all_input_files(paths) if not path.exists()]
    if not missing and not rebuild:
        _validate_input_contract(paths)
        return paths

    print("[1/5] Получаю исходные события")
    if data is None:
        data = load_train(train_path)
    if users is None:
        users = all_users(data)
    if users.height != 250_000:
        raise AssertionError(f"Ожидалось 250 000 пользователей: {users.height}")

    print("[2/5] Строю девять базовых временных срезов")
    for index, anchor in enumerate(ALL_ANCHORS, start=1):
        path = paths.snapshots[anchor]
        if rebuild or not path.exists():
            print(f"  [{index}/{len(ALL_ANCHORS)}] {anchor}")
            snapshot = build_snapshot(
                data,
                users,
                anchor,
                with_target=anchor != TEST_ANCHOR,
                feature_pack="enhanced_v2",
            )
            snapshot.write_parquet(path, compression="zstd")
            del snapshot
            gc.collect()

    print("[3/5] Строю два сезонных профиля")
    for anchor, path in paths.seasonal_profiles.items():
        if rebuild or not path.exists():
            profile, _ = build_seasonal_analog_profile(data, users, anchor)
            profile.write_parquet(path, compression="zstd")
            del profile
            gc.collect()

    print("[4/5] Строю два годовых профиля")
    for anchor, path in paths.annual_profiles.items():
        if rebuild or not path.exists():
            profile = build_annual_features(data, users, anchor)
            profile.write_parquet(path, compression="zstd")
            del profile
            gc.collect()

    print("[5/5] Строю девять профилей ритма")
    for index, (anchor, path) in enumerate(paths.cadence_profiles.items(), start=1):
        if rebuild or not path.exists():
            print(f"  [{index}/{len(paths.cadence_profiles)}] {anchor}")
            profile = build_cadence_features(data, users, anchor)
            profile.write_parquet(path, compression="zstd")
            del profile
            gc.collect()
    del data, users
    gc.collect()
    _validate_input_contract(paths)
    return paths


def _join_profile(
    frame: pl.DataFrame,
    *,
    profile_path: Path,
    columns: Sequence[str],
) -> pl.DataFrame:
    profile = pl.read_parquet(profile_path, columns=["user_id", *columns])
    if profile.height != 250_000 or profile["user_id"].n_unique() != 250_000:
        raise AssertionError(f"Некорректный профиль: {profile_path}")
    joined = frame.join(profile, on="user_id", how="left", validate="1:1")
    if joined.select(columns).null_count().to_numpy().sum() != 0:
        raise AssertionError(f"Профиль не покрыл пользователей: {profile_path}")
    return joined


def load_snapshot_blocks(
    *,
    snapshot_path: Path,
    seasonal_profile_path: Path,
    annual_profile_path: Path,
    cadence_profile_path: Path,
    base_features: Sequence[str],
    seasonal_features: Sequence[str],
    require_target: bool,
) -> SnapshotBlocks:
    """Соединяет четыре блока признаков строго 1:1 и сохраняет порядок."""
    columns = ["user_id", *base_features]
    if require_target:
        columns.append("target")
    frame = pl.read_parquet(snapshot_path, columns=columns).with_row_index("_row")
    frame = _join_profile(
        frame, profile_path=seasonal_profile_path, columns=seasonal_features
    )
    frame = _join_profile(
        frame, profile_path=annual_profile_path, columns=ANNUAL_FEATURES
    )
    frame = _join_profile(
        frame, profile_path=cadence_profile_path, columns=CADENCE_FEATURES
    ).sort("_row")
    if frame.height != 250_000 or frame["user_id"].n_unique() != 250_000:
        raise AssertionError("Соединение изменило множество пользователей.")
    result = SnapshotBlocks(
        user_id=frame["user_id"].to_numpy().astype(np.int64, copy=False),
        target=(
            frame["target"].to_numpy().astype(np.float32, copy=False)
            if require_target
            else np.zeros(frame.height, dtype=np.float32)
        ),
        base=frame.select(base_features).to_numpy().astype(np.float32, copy=False),
        seasonal=frame.select(seasonal_features)
        .to_numpy()
        .astype(np.float32, copy=False),
        annual=frame.select(ANNUAL_FEATURES)
        .to_numpy()
        .astype(np.float32, copy=False),
        cadence=frame.select(CADENCE_FEATURES)
        .to_numpy()
        .astype(np.float32, copy=False),
    )
    values = np.concatenate(
        [result.base, result.seasonal, result.annual, result.cadence], axis=1
    )
    if not np.isfinite(values).all() or not np.isfinite(result.target).all():
        raise AssertionError("В соединённой матрице есть NaN или бесконечность.")
    return result


def _base_and_seasonal_features(
    paths: FinalInputPaths,
) -> tuple[list[str], list[str]]:
    schema = pl.read_parquet_schema(paths.snapshots[JANUARY_ANCHOR])
    base_features = [
        column
        for column in schema
        if column not in {"user_id", "anchor_date", "target"}
    ]
    seasonal_features = seasonal_feature_columns(
        pl.read_parquet_schema(paths.seasonal_profiles[JANUARY_ANCHOR])
    )
    if len(base_features) != BASE_FEATURE_COUNT:
        raise AssertionError("Ожидалось 216 базовых признаков.")
    return base_features, seasonal_features


def load_training_matrix(
    *,
    paths: FinalInputPaths,
    base_features: Sequence[str],
    seasonal_features: Sequence[str],
    include_cadence: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Собирает 2 млн строк для одной из двух финальных моделей."""
    feature_names = [*base_features, *seasonal_features, *ANNUAL_FEATURES]
    if include_cadence:
        feature_names.extend(CADENCE_FEATURES)
    width = len(feature_names)
    matrix = np.zeros((2_000_000, width), dtype=np.float32)
    target = np.empty(2_000_000, dtype=np.float32)
    base_width = len(base_features)
    cadence_start = base_width + len(seasonal_features) + len(ANNUAL_FEATURES)

    offset = 0
    for anchor in HISTORICAL_ANCHORS:
        snapshot = pl.read_parquet(
            paths.snapshots[anchor], columns=["user_id", *base_features, "target"]
        ).with_row_index("_row")
        if include_cadence:
            snapshot = _join_profile(
                snapshot,
                profile_path=paths.cadence_profiles[anchor],
                columns=CADENCE_FEATURES,
            ).sort("_row")
        end = offset + snapshot.height
        matrix[offset:end, :base_width] = snapshot.select(base_features).to_numpy()
        if include_cadence:
            matrix[offset:end, cadence_start:] = snapshot.select(
                CADENCE_FEATURES
            ).to_numpy()
        target[offset:end] = snapshot["target"].to_numpy()
        offset = end
        del snapshot
        gc.collect()

    january = load_snapshot_blocks(
        snapshot_path=paths.snapshots[JANUARY_ANCHOR],
        seasonal_profile_path=paths.seasonal_profiles[JANUARY_ANCHOR],
        annual_profile_path=paths.annual_profiles[JANUARY_ANCHOR],
        cadence_profile_path=paths.cadence_profiles[JANUARY_ANCHOR],
        base_features=base_features,
        seasonal_features=seasonal_features,
        require_target=True,
    )
    end = offset + len(january.target)
    january_parts = [january.base, january.seasonal, january.annual]
    if include_cadence:
        january_parts.append(january.cadence)
    matrix[offset:end] = np.concatenate(january_parts, axis=1)
    target[offset:end] = january.target
    del january
    gc.collect()
    if end != 2_000_000 or not np.isfinite(matrix).all():
        raise AssertionError("Финальная обучающая матрица заполнена некорректно.")
    return matrix, target, feature_names


def log_blend(
    annual_prediction: np.ndarray,
    cadence_prediction: np.ndarray,
) -> np.ndarray:
    """Смешивает 25% годовой модели и 75% модели ритма в log1p."""
    annual_log = np.log1p(np.clip(annual_prediction, 0.0, None))
    cadence_log = np.log1p(np.clip(cadence_prediction, 0.0, None))
    return np.expm1(ANNUAL_WEIGHT * annual_log + CADENCE_WEIGHT * cadence_log)


def _train_model(
    *,
    matrix: np.ndarray,
    target: np.ndarray,
    feature_names: Sequence[str],
    iterations: int,
    model_path: Path,
) -> tuple[object, float]:
    pool = Pool(matrix, label=np.log1p(target), feature_names=list(feature_names))
    target_mean = float(target.mean())
    del matrix, target
    gc.collect()
    model = make_final_model(
        iterations=iterations,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=10.0,
        random_strength=0.5,
        verbose=200,
    )
    model.fit(pool)
    del pool
    gc.collect()
    model.save_model(str(model_path))
    return model, target_mean


def validate_saved_solution(
    *,
    artifact_dir: Path = FINAL_SOLUTION_DIR,
    submission_path: Path = FINAL_SUBMISSION_PATH,
    deep: bool = True,
) -> dict[str, object]:
    """Проверяет файлы, а в глубоком режиме пересчитывает смесь."""
    prediction_path = artifact_dir / "predictions.parquet"
    required = [
        artifact_dir / "annual_model.cbm",
        artifact_dir / "cadence_model.cbm",
        prediction_path,
        artifact_dir / "summary.json",
        submission_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Нет финальных файлов: {missing}")
    if not deep:
        report_path = artifact_dir / "validation_report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Нет отчёта проверки: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "ready_for_upload":
            raise AssertionError("Сохранённый отчёт не подтверждает готовность.")
        return report
    predictions = pd.read_parquet(prediction_path)
    submission = pd.read_csv(submission_path)
    if list(submission.columns) != ["user_id", "predict"]:
        raise AssertionError("Некорректная схема CSV.")
    if len(submission) != 250_000 or not submission.user_id.is_unique:
        raise AssertionError("Некорректная детализация CSV.")
    if not np.array_equal(submission.user_id, predictions.user_id):
        raise AssertionError("Порядок пользователей не совпал.")
    recomputed = log_blend(
        predictions.annual_prediction.to_numpy(dtype=float),
        predictions.cadence_prediction.to_numpy(dtype=float),
    )
    if not np.allclose(submission.predict, recomputed, rtol=1e-12, atol=1e-12):
        raise AssertionError("CSV не совпал с формулой смеси.")
    if not np.isfinite(submission.predict).all() or (submission.predict < 0).any():
        raise AssertionError("В CSV есть некорректные значения.")
    return {
        "status": "ready_for_upload",
        "rows": len(submission),
        "unique_users": int(submission.user_id.nunique()),
        "all_predictions_finite": True,
        "all_predictions_nonnegative": True,
        "saved_values_match_log_blend": True,
    }


def train_final_solution(
    *,
    train_path: Path = TRAIN_PATH,
    cache_dir: Path = FINAL_SOLUTION_CACHE_DIR,
    artifact_dir: Path = FINAL_SOLUTION_DIR,
    submission_path: Path = FINAL_SUBMISSION_PATH,
    rebuild_inputs: bool = False,
) -> dict[str, object]:
    """Полностью переобучает текущее финальное решение и создаёт CSV."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    paths = prepare_final_inputs(
        train_path=train_path, cache_dir=cache_dir, rebuild=rebuild_inputs
    )
    base_features, seasonal_features = _base_and_seasonal_features(paths)

    print("[1/6] Загружаю финальный тестовый срез")
    test = load_snapshot_blocks(
        snapshot_path=paths.snapshots[TEST_ANCHOR],
        seasonal_profile_path=paths.seasonal_profiles[TEST_ANCHOR],
        annual_profile_path=paths.annual_profiles[TEST_ANCHOR],
        cadence_profile_path=paths.cadence_profiles[TEST_ANCHOR],
        base_features=base_features,
        seasonal_features=seasonal_features,
        require_target=False,
    )
    annual_test = np.concatenate([test.base, test.seasonal, test.annual], axis=1)
    cadence_test = np.concatenate(
        [test.base, test.seasonal, test.annual, test.cadence], axis=1
    )
    annual_features = [*base_features, *seasonal_features, *ANNUAL_FEATURES]
    cadence_features = [*annual_features, *CADENCE_FEATURES]

    print("[2/6] Обучаю годовую модель")
    matrix, target, names = load_training_matrix(
        paths=paths,
        base_features=base_features,
        seasonal_features=seasonal_features,
        include_cadence=False,
    )
    if names != annual_features:
        raise AssertionError("Изменился порядок признаков годовой модели.")
    annual_model, annual_target_mean = _train_model(
        matrix=matrix,
        target=target,
        feature_names=names,
        iterations=ANNUAL_ITERATIONS,
        model_path=artifact_dir / "annual_model.cbm",
    )
    annual_prediction = predict_gmv(
        annual_model, Pool(annual_test, feature_names=annual_features)
    )
    del annual_model, annual_test
    gc.collect()

    print("[3/6] Обучаю модель ритма")
    matrix, target, names = load_training_matrix(
        paths=paths,
        base_features=base_features,
        seasonal_features=seasonal_features,
        include_cadence=True,
    )
    if names != cadence_features:
        raise AssertionError("Изменился порядок признаков модели ритма.")
    cadence_model, cadence_target_mean = _train_model(
        matrix=matrix,
        target=target,
        feature_names=names,
        iterations=CADENCE_ITERATIONS,
        model_path=artifact_dir / "cadence_model.cbm",
    )
    cadence_prediction = predict_gmv(
        cadence_model, Pool(cadence_test, feature_names=cadence_features)
    )
    del cadence_test
    gc.collect()

    print("[4/6] Смешиваю прогнозы 25/75")
    blend_prediction = log_blend(annual_prediction, cadence_prediction)
    predictions = pd.DataFrame(
        {
            "user_id": test.user_id,
            "annual_prediction": annual_prediction,
            "cadence_prediction": cadence_prediction,
            "blend_prediction": blend_prediction,
        }
    )
    predictions.to_parquet(artifact_dir / "predictions.parquet", index=False)
    predictions[["user_id"]].assign(predict=blend_prediction).to_csv(
        submission_path, index=False
    )

    print("[5/6] Сохраняю важности и сводку")
    groups = (
        ["base"] * len(base_features)
        + ["seasonal"] * len(seasonal_features)
        + ["annual"] * len(ANNUAL_FEATURES)
        + ["cadence"] * len(CADENCE_FEATURES)
    )
    importance = pd.DataFrame(
        {
            "feature": cadence_features,
            "importance": cadence_model.get_feature_importance(),
            "feature_group": groups,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(artifact_dir / "feature_importance.csv", index=False)
    del cadence_model, test
    gc.collect()

    summary: dict[str, object] = {
        "method": "annual_cadence_log_blend",
        "historical_anchors": [anchor.isoformat() for anchor in HISTORICAL_ANCHORS],
        "january_anchor": JANUARY_ANCHOR.isoformat(),
        "test_anchor": TEST_ANCHOR.isoformat(),
        "target_window": "2026-02-14..2026-03-15",
        "training_rows": 2_000_000,
        "base_features": len(base_features),
        "seasonal_features": len(seasonal_features),
        "annual_features": len(ANNUAL_FEATURES),
        "cadence_features": len(CADENCE_FEATURES),
        "annual_model_features": len(annual_features),
        "cadence_model_features": len(cadence_features),
        "annual_iterations": ANNUAL_ITERATIONS,
        "cadence_iterations": CADENCE_ITERATIONS,
        "annual_weight": ANNUAL_WEIGHT,
        "cadence_weight": CADENCE_WEIGHT,
        "annual_train_target_mean": annual_target_mean,
        "cadence_train_target_mean": cadence_target_mean,
        "local_validation_rmsle_before_cadence": 1.6631013062110978,
        "local_validation_rmsle_final_blend": 1.6619489291229776,
        "local_validation_improvement": 0.0011523770881203266,
        "submission_rows": len(predictions),
        "prediction_sum": float(blend_prediction.sum()),
        "prediction_mean": float(blend_prediction.mean()),
        "prediction_median": float(np.median(blend_prediction)),
        "prediction_zero_share": float((blend_prediction == 0).mean()),
        "submission_path": str(submission_path),
    }
    (artifact_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[6/6] Независимо проверяю сохранённый файл")
    validation = validate_saved_solution(
        artifact_dir=artifact_dir, submission_path=submission_path, deep=True
    )
    (artifact_dir / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def run_final_solution(
    *,
    retrain: bool = False,
    train_path: Path = TRAIN_PATH,
    cache_dir: Path = FINAL_SOLUTION_CACHE_DIR,
    artifact_dir: Path = FINAL_SOLUTION_DIR,
    submission_path: Path = FINAL_SUBMISSION_PATH,
) -> dict[str, object]:
    """Возвращает готовый результат или полностью переобучает решение."""
    summary_path = artifact_dir / "summary.json"
    if not retrain and summary_path.exists() and submission_path.exists():
        validate_saved_solution(
            artifact_dir=artifact_dir, submission_path=submission_path, deep=False
        )
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return train_final_solution(
        train_path=train_path,
        cache_dir=cache_dir,
        artifact_dir=artifact_dir,
        submission_path=submission_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_final_solution(retrain=args.retrain, train_path=args.train_path)


if __name__ == "__main__":
    main()
