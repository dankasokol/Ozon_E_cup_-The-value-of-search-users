"""Эксперимент с денежным ритмом поверх текущего лучшего решения.

Модуль намеренно разделяет два дорогих действия:

1. проверку гипотезы на сохранённом январском разбиении;
2. финальное обучение и создание CSV только после успешной проверки.

Контрольная модель повторно не обучается: её январские прогнозы уже сохранены.
"""

from __future__ import annotations

import gc
import json
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl
from catboost import Pool

from .annual_features import ANNUAL_FEATURES
from .cadence_features import CADENCE_FEATURES
from .config import (
    FINAL_SOLUTION_DIR,
    MONETARY_CADENCE_BLEND_SUBMISSION_PATH,
    MONETARY_CADENCE_CONTROL_PATH,
    MONETARY_CADENCE_CONTROL_SOURCE_PATH,
    MONETARY_CADENCE_DIR,
    MONETARY_CADENCE_PROFILE_DIR,
    MONETARY_CADENCE_SUBMISSION_PATH,
    TRAIN_PATH,
)
from .data import all_users, load_train
from .final_solution import (
    ALL_ANCHORS,
    BASE_FEATURE_COUNT,
    HISTORICAL_ANCHORS,
    JANUARY_ANCHOR,
    SEASONAL_FEATURE_COUNT,
    TEST_ANCHOR,
    FinalInputPaths,
    final_input_paths,
    prepare_final_inputs,
    seasonal_feature_columns,
)
from .models import make_final_model, predict_gmv
from .monetary_cadence import (
    MONETARY_CADENCE_FEATURES,
    monetary_profile_path,
    prepare_monetary_profiles,
    validate_monetary_profiles,
)


SCREENING_HISTORICAL_ANCHORS = HISTORICAL_ANCHORS[:-1]
MAX_SCREENING_ITERATIONS = 1_600
EARLY_STOPPING_ROUNDS = 150
BLEND_WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
MINIMUM_CONFIRM_IMPROVEMENT = 0.0003
MINIMUM_BOOTSTRAP_POSITIVE_SHARE = 0.80


@dataclass
class ValidationData:
    """Матрицы январской проверки и соответствующий контроль."""

    train_features: np.ndarray
    train_target: np.ndarray
    tune_features: np.ndarray
    tune_target: np.ndarray
    tune_user_id: np.ndarray
    tune_control: np.ndarray
    confirm_features: np.ndarray
    confirm_target: np.ndarray
    confirm_user_id: np.ndarray
    confirm_control: np.ndarray
    feature_names: list[str]


def rmsle(target: np.ndarray, prediction: np.ndarray) -> float:
    """Вычисляет корень из средней квадратичной логарифмической ошибки."""
    target_log = np.log1p(np.clip(np.asarray(target, dtype=float), 0.0, None))
    prediction_log = np.log1p(
        np.clip(np.asarray(prediction, dtype=float), 0.0, None)
    )
    return float(np.sqrt(np.mean(np.square(target_log - prediction_log))))


def log_blend(
    control: np.ndarray, candidate: np.ndarray, candidate_weight: float
) -> np.ndarray:
    """Смешивает контроль и новую модель в шкале log(1 + GMV)."""
    control_log = np.log1p(np.clip(control, 0.0, None))
    candidate_log = np.log1p(np.clip(candidate, 0.0, None))
    return np.expm1(
        (1.0 - candidate_weight) * control_log
        + candidate_weight * candidate_log
    )


def validate_control_predictions(
    path: Path = MONETARY_CADENCE_CONTROL_PATH,
) -> pd.DataFrame:
    """Проверяет сохранённое разбиение и возвращает две контрольные ошибки."""
    if not path.exists() and MONETARY_CADENCE_CONTROL_SOURCE_PATH.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MONETARY_CADENCE_CONTROL_SOURCE_PATH, path)
    if not path.exists():
        raise FileNotFoundError(
            "Нет контрольных январских прогнозов. Ожидался файл: " f"{path}"
        )
    frame = pd.read_parquet(path)
    required = {"role", "user_id", "target", "selected_prediction"}
    if not required.issubset(frame.columns):
        raise AssertionError(f"В контроле нет столбцов {sorted(required)}.")
    if len(frame) != 50_000 or not frame.user_id.is_unique:
        raise AssertionError("Ожидалось 50 000 уникальных контрольных пользователей.")
    if set(frame.role) != {"tune", "confirm"}:
        raise AssertionError("Ожидались части tune и confirm.")
    counts = frame.groupby("role", observed=True).size().to_dict()
    if counts != {"confirm": 25_000, "tune": 25_000}:
        raise AssertionError(f"Изменился размер частей: {counts}")
    if not np.isfinite(
        frame[["target", "selected_prediction"]].to_numpy(dtype=float)
    ).all():
        raise AssertionError("В контрольных прогнозах есть некорректные числа.")
    rows = []
    for role in ("tune", "confirm"):
        part = frame.loc[frame.role == role]
        rows.append(
            {
                "role": role,
                "users": len(part),
                "current_solution_rmsle": rmsle(
                    part.target.to_numpy(), part.selected_prediction.to_numpy()
                ),
            }
        )
    return pd.DataFrame(rows)


def prepare_experiment_inputs(
    *,
    rebuild_base: bool = False,
    rebuild_monetary: bool = False,
) -> tuple[FinalInputPaths, dict[date, Path]]:
    """Готовит базовые кэши и девять денежных профилей.

    Каждый готовый Parquet пропускается, поэтому повторный запуск безопасен.
    """
    validate_control_predictions()
    base_paths = final_input_paths()
    money_paths = {
        anchor: monetary_profile_path(MONETARY_CADENCE_PROFILE_DIR, anchor)
        for anchor in ALL_ANCHORS
    }
    base_files = [
        *base_paths.snapshots.values(),
        *base_paths.seasonal_profiles.values(),
        *base_paths.annual_profiles.values(),
        *base_paths.cadence_profiles.values(),
    ]
    need_base = rebuild_base or any(not path.exists() for path in base_files)
    need_money = rebuild_monetary or any(
        not path.exists() for path in money_paths.values()
    )
    if need_base or need_money:
        print("Один раз читаю исходные события для всех недостающих входов")
        data = load_train(TRAIN_PATH)
        users = all_users(data)
        base_paths = prepare_final_inputs(
            rebuild=rebuild_base, data=data, users=users
        )
        if need_money:
            money_paths = prepare_monetary_profiles(
                data=data,
                users=users,
                anchors=ALL_ANCHORS,
                profile_dir=MONETARY_CADENCE_PROFILE_DIR,
                rebuild=rebuild_monetary,
            )
        else:
            validate_monetary_profiles(money_paths)
        del data, users
        gc.collect()
    else:
        base_paths = prepare_final_inputs(rebuild=False)
        validate_monetary_profiles(money_paths)
        print("Все входы уже готовы")
    return base_paths, money_paths


def monetary_profile_audit(
    money_paths: dict[date, Path] | None = None,
) -> pd.DataFrame:
    """Краткая проверка распределений перед обучением."""
    if money_paths is None:
        money_paths = {
            anchor: monetary_profile_path(MONETARY_CADENCE_PROFILE_DIR, anchor)
            for anchor in ALL_ANCHORS
        }
    validate_monetary_profiles(money_paths)
    rows = []
    for anchor, path in money_paths.items():
        profile = pl.scan_parquet(path).select(
            pl.len().alias("rows"),
            (pl.col("money_last_gmv") > 0).mean().alias("purchase_user_share"),
            pl.col("money_last_gmv").mean().alias("last_gmv_mean"),
            pl.col("money_expected_gmv_by_cycle_30d")
            .mean()
            .alias("expected_gmv_mean"),
        ).collect()
        rows.append({"anchor": anchor.isoformat(), **profile.row(0, named=True)})
    audit = pd.DataFrame(rows)
    audit.to_csv(MONETARY_CADENCE_DIR / "profile_audit.csv", index=False)
    return audit


def _feature_columns(paths: FinalInputPaths) -> tuple[list[str], list[str], list[str]]:
    snapshot_schema = pl.read_parquet_schema(paths.snapshots[JANUARY_ANCHOR])
    base = [
        column
        for column in snapshot_schema
        if column not in {"user_id", "anchor_date", "target"}
    ]
    seasonal = seasonal_feature_columns(
        pl.read_parquet_schema(paths.seasonal_profiles[JANUARY_ANCHOR])
    )
    names = [
        *base,
        *seasonal,
        *ANNUAL_FEATURES,
        *CADENCE_FEATURES,
        *MONETARY_CADENCE_FEATURES,
    ]
    if len(names) != len(set(names)):
        raise AssertionError("Названия признаков пересекаются.")
    return base, seasonal, names


def _join_profile(
    frame: pl.DataFrame, path: Path, columns: Sequence[str]
) -> pl.DataFrame:
    profile = pl.read_parquet(path, columns=["user_id", *columns])
    if profile.height != 250_000 or profile["user_id"].n_unique() != 250_000:
        raise AssertionError(f"Некорректный профиль: {path}")
    joined = frame.join(profile, on="user_id", how="left", validate="1:1")
    if joined.select(columns).null_count().to_numpy().sum() != 0:
        raise AssertionError(f"Профиль не покрыл всех пользователей: {path}")
    return joined


def _load_full_anchor(
    *,
    anchor: date,
    paths: FinalInputPaths,
    money_paths: dict[date, Path],
    base_features: Sequence[str],
    seasonal_features: Sequence[str],
    require_target: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = ["user_id", *base_features]
    if require_target:
        columns.append("target")
    frame = pl.read_parquet(paths.snapshots[anchor], columns=columns).with_row_index(
        "_row"
    )
    frame = _join_profile(
        frame, paths.seasonal_profiles[anchor], seasonal_features
    )
    frame = _join_profile(frame, paths.annual_profiles[anchor], ANNUAL_FEATURES)
    frame = _join_profile(frame, paths.cadence_profiles[anchor], CADENCE_FEATURES)
    frame = _join_profile(
        frame, money_paths[anchor], MONETARY_CADENCE_FEATURES
    ).sort("_row")
    feature_names = [
        *base_features,
        *seasonal_features,
        *ANNUAL_FEATURES,
        *CADENCE_FEATURES,
        *MONETARY_CADENCE_FEATURES,
    ]
    matrix = frame.select(feature_names).to_numpy().astype(np.float32, copy=False)
    user_id = frame["user_id"].to_numpy().astype(np.int64, copy=False)
    target = (
        frame["target"].to_numpy().astype(np.float32, copy=False)
        if require_target
        else np.zeros(frame.height, dtype=np.float32)
    )
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise AssertionError(f"Некорректные числа в матрице якоря {anchor}.")
    return user_id, target, matrix


def _build_training_matrix(
    *,
    paths: FinalInputPaths,
    money_paths: dict[date, Path],
    historical_anchors: Sequence[date],
    base_features: Sequence[str],
    seasonal_features: Sequence[str],
    january_matrix: np.ndarray,
    january_target: np.ndarray,
    january_train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    feature_names = [
        *base_features,
        *seasonal_features,
        *ANNUAL_FEATURES,
        *CADENCE_FEATURES,
        *MONETARY_CADENCE_FEATURES,
    ]
    row_count = 250_000 * len(historical_anchors) + int(january_train_mask.sum())
    matrix = np.zeros((row_count, len(feature_names)), dtype=np.float32)
    target = np.empty(row_count, dtype=np.float32)
    base_end = len(base_features)
    cadence_start = base_end + len(seasonal_features) + len(ANNUAL_FEATURES)
    money_start = cadence_start + len(CADENCE_FEATURES)

    offset = 0
    for index, anchor in enumerate(historical_anchors, start=1):
        print(f"  исторический якорь [{index}/{len(historical_anchors)}]: {anchor}")
        frame = pl.read_parquet(
            paths.snapshots[anchor],
            columns=["user_id", *base_features, "target"],
        ).with_row_index("_row")
        frame = _join_profile(frame, paths.cadence_profiles[anchor], CADENCE_FEATURES)
        frame = _join_profile(
            frame, money_paths[anchor], MONETARY_CADENCE_FEATURES
        ).sort("_row")
        end = offset + frame.height
        matrix[offset:end, :base_end] = frame.select(base_features).to_numpy()
        matrix[offset:end, cadence_start:money_start] = frame.select(
            CADENCE_FEATURES
        ).to_numpy()
        matrix[offset:end, money_start:] = frame.select(
            MONETARY_CADENCE_FEATURES
        ).to_numpy()
        target[offset:end] = frame["target"].to_numpy()
        offset = end
        del frame
        gc.collect()

    january_rows = int(january_train_mask.sum())
    end = offset + january_rows
    matrix[offset:end] = january_matrix[january_train_mask]
    target[offset:end] = january_target[january_train_mask]
    if end != row_count or not np.isfinite(matrix).all():
        raise AssertionError("Обучающая матрица заполнена некорректно.")
    return matrix, target


def build_validation_data(
    *,
    paths: FinalInputPaths | None = None,
    money_paths: dict[date, Path] | None = None,
) -> ValidationData:
    """Собирает 1,7 млн обучающих строк и две январские проверки."""
    if paths is None:
        paths = final_input_paths()
    if money_paths is None:
        money_paths = {
            anchor: monetary_profile_path(MONETARY_CADENCE_PROFILE_DIR, anchor)
            for anchor in ALL_ANCHORS
        }
    validate_monetary_profiles(money_paths)
    base, seasonal, names = _feature_columns(paths)
    january_user, january_target, january_matrix = _load_full_anchor(
        anchor=JANUARY_ANCHOR,
        paths=paths,
        money_paths=money_paths,
        base_features=base,
        seasonal_features=seasonal,
        require_target=True,
    )

    control = pd.read_parquet(MONETARY_CADENCE_CONTROL_PATH).set_index("user_id")
    tune_mask = np.isin(january_user, control.index[control.role == "tune"])
    confirm_mask = np.isin(january_user, control.index[control.role == "confirm"])
    train_mask = ~(tune_mask | confirm_mask)
    if (tune_mask.sum(), confirm_mask.sum(), train_mask.sum()) != (
        25_000,
        25_000,
        200_000,
    ):
        raise AssertionError("Не удалось восстановить исходное январское разбиение.")

    def ordered_control(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        part = control.loc[january_user[mask]]
        saved_target = part.target.to_numpy(dtype=np.float32)
        if not np.allclose(saved_target, january_target[mask], rtol=0.0, atol=1e-4):
            raise AssertionError("Цель в кэше не совпала с январским срезом.")
        return (
            part.selected_prediction.to_numpy(dtype=float),
            saved_target,
        )

    tune_control, tune_saved_target = ordered_control(tune_mask)
    confirm_control, confirm_saved_target = ordered_control(confirm_mask)
    matrix, target = _build_training_matrix(
        paths=paths,
        money_paths=money_paths,
        historical_anchors=SCREENING_HISTORICAL_ANCHORS,
        base_features=base,
        seasonal_features=seasonal,
        january_matrix=january_matrix,
        january_target=january_target,
        january_train_mask=train_mask,
    )
    result = ValidationData(
        train_features=matrix,
        train_target=target,
        tune_features=january_matrix[tune_mask],
        tune_target=tune_saved_target,
        tune_user_id=january_user[tune_mask],
        tune_control=tune_control,
        confirm_features=january_matrix[confirm_mask],
        confirm_target=confirm_saved_target,
        confirm_user_id=january_user[confirm_mask],
        confirm_control=confirm_control,
        feature_names=names,
    )
    del january_matrix, january_target, january_user, control
    gc.collect()
    return result


def _best_blend_weight(
    target: np.ndarray, control: np.ndarray, candidate: np.ndarray
) -> tuple[float, pd.DataFrame]:
    rows = []
    for weight in BLEND_WEIGHTS:
        prediction = log_blend(control, candidate, float(weight))
        rows.append({"candidate_weight": float(weight), "rmsle": rmsle(target, prediction)})
    grid = pd.DataFrame(rows)
    best = grid.sort_values(["rmsle", "candidate_weight"]).iloc[0]
    return float(best.candidate_weight), grid


def _bootstrap_improvement(
    target: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
    *,
    repeats: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    improvements = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = rng.integers(0, len(target), size=len(target))
        improvements[index] = rmsle(target[sample], control[sample]) - rmsle(
            target[sample], candidate[sample]
        )
    return {
        "bootstrap_ci_low": float(np.quantile(improvements, 0.025)),
        "bootstrap_ci_high": float(np.quantile(improvements, 0.975)),
        "bootstrap_positive_share": float((improvements > 0).mean()),
    }


def run_validation_experiment(
    *,
    paths: FinalInputPaths | None = None,
    money_paths: dict[date, Path] | None = None,
) -> dict[str, object]:
    """Обучает одну проверочную модель и принимает решение по confirm."""
    MONETARY_CADENCE_DIR.mkdir(parents=True, exist_ok=True)
    data = build_validation_data(paths=paths, money_paths=money_paths)
    train_pool = Pool(
        data.train_features,
        label=np.log1p(data.train_target),
        feature_names=data.feature_names,
    )
    tune_pool = Pool(
        data.tune_features,
        label=np.log1p(data.tune_target),
        feature_names=data.feature_names,
    )
    del data.train_features, data.train_target
    gc.collect()

    model = make_final_model(
        iterations=MAX_SCREENING_ITERATIONS,
        eval_metric="RMSE",
        use_best_model=True,
        od_type="Iter",
        od_wait=EARLY_STOPPING_ROUNDS,
        verbose=100,
    )
    model.fit(train_pool, eval_set=tune_pool)
    del train_pool, tune_pool
    gc.collect()
    model.save_model(str(MONETARY_CADENCE_DIR / "validation_model.cbm"))

    tune_candidate = predict_gmv(
        model, Pool(data.tune_features, feature_names=data.feature_names)
    )
    confirm_candidate = predict_gmv(
        model, Pool(data.confirm_features, feature_names=data.feature_names)
    )
    best_weight, blend_grid = _best_blend_weight(
        data.tune_target, data.tune_control, tune_candidate
    )
    blend_grid.to_csv(MONETARY_CADENCE_DIR / "blend_grid_tune.csv", index=False)
    tune_selected = log_blend(data.tune_control, tune_candidate, best_weight)
    confirm_selected = log_blend(
        data.confirm_control, confirm_candidate, best_weight
    )

    tune_control_score = rmsle(data.tune_target, data.tune_control)
    tune_candidate_score = rmsle(data.tune_target, tune_candidate)
    tune_selected_score = rmsle(data.tune_target, tune_selected)
    confirm_control_score = rmsle(data.confirm_target, data.confirm_control)
    confirm_candidate_score = rmsle(data.confirm_target, confirm_candidate)
    confirm_selected_score = rmsle(data.confirm_target, confirm_selected)
    bootstrap = _bootstrap_improvement(
        data.confirm_target, data.confirm_control, confirm_selected
    )
    tune_improvement = tune_control_score - tune_selected_score
    confirm_improvement = confirm_control_score - confirm_selected_score
    accepted = bool(
        best_weight > 0
        and tune_improvement > 0
        and confirm_improvement >= MINIMUM_CONFIRM_IMPROVEMENT
        and bootstrap["bootstrap_positive_share"]
        >= MINIMUM_BOOTSTRAP_POSITIVE_SHARE
    )

    predictions = pd.concat(
        [
            pd.DataFrame(
                {
                    "role": "tune",
                    "user_id": data.tune_user_id,
                    "target": data.tune_target,
                    "control_prediction": data.tune_control,
                    "candidate_prediction": tune_candidate,
                    "selected_prediction": tune_selected,
                }
            ),
            pd.DataFrame(
                {
                    "role": "confirm",
                    "user_id": data.confirm_user_id,
                    "target": data.confirm_target,
                    "control_prediction": data.confirm_control,
                    "candidate_prediction": confirm_candidate,
                    "selected_prediction": confirm_selected,
                }
            ),
        ],
        ignore_index=True,
    )
    predictions.to_parquet(
        MONETARY_CADENCE_DIR / "validation_predictions.parquet", index=False
    )
    metrics = pd.DataFrame(
        [
            {"role": "tune", "model": "control", "rmsle": tune_control_score},
            {"role": "tune", "model": "candidate", "rmsle": tune_candidate_score},
            {"role": "tune", "model": "selected", "rmsle": tune_selected_score},
            {"role": "confirm", "model": "control", "rmsle": confirm_control_score},
            {"role": "confirm", "model": "candidate", "rmsle": confirm_candidate_score},
            {"role": "confirm", "model": "selected", "rmsle": confirm_selected_score},
        ]
    )
    metrics.to_csv(MONETARY_CADENCE_DIR / "validation_metrics.csv", index=False)

    groups = (
        ["base"] * BASE_FEATURE_COUNT
        + ["seasonal"] * SEASONAL_FEATURE_COUNT
        + ["annual"] * len(ANNUAL_FEATURES)
        + ["cadence"] * len(CADENCE_FEATURES)
        + ["monetary_cadence"] * len(MONETARY_CADENCE_FEATURES)
    )
    importance = pd.DataFrame(
        {
            "feature": data.feature_names,
            "importance": model.get_feature_importance(),
            "feature_group": groups,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(MONETARY_CADENCE_DIR / "feature_importance.csv", index=False)

    summary: dict[str, object] = {
        "experiment": "monetary_cadence_v1",
        "validation_anchor": JANUARY_ANCHOR.isoformat(),
        "historical_anchors": [
            anchor.isoformat() for anchor in SCREENING_HISTORICAL_ANCHORS
        ],
        "training_rows": 1_700_000,
        "features": len(data.feature_names),
        "monetary_features": len(MONETARY_CADENCE_FEATURES),
        "best_iteration_zero_based": int(model.get_best_iteration()),
        "final_iterations_if_accepted": int(model.get_best_iteration()) + 1,
        "candidate_weight": best_weight,
        "tune_control_rmsle": tune_control_score,
        "tune_candidate_rmsle": tune_candidate_score,
        "tune_selected_rmsle": tune_selected_score,
        "tune_improvement": tune_improvement,
        "confirm_control_rmsle": confirm_control_score,
        "confirm_candidate_rmsle": confirm_candidate_score,
        "confirm_selected_rmsle": confirm_selected_score,
        "confirm_improvement": confirm_improvement,
        **bootstrap,
        "minimum_confirm_improvement": MINIMUM_CONFIRM_IMPROVEMENT,
        "minimum_bootstrap_positive_share": MINIMUM_BOOTSTRAP_POSITIVE_SHARE,
        "accepted": accepted,
        "validation_limitation": (
            "Разбиение по пользователям на одном январском якоре не является "
            "отдельной временной проверкой февраля–марта."
        ),
    }
    (MONETARY_CADENCE_DIR / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del data, model
    gc.collect()
    return summary


def load_validation_summary() -> dict[str, object]:
    """Читает решение проверочного этапа."""
    path = MONETARY_CADENCE_DIR / "validation_summary.json"
    if not path.exists():
        raise FileNotFoundError("Сначала выполните проверочное обучение.")
    return json.loads(path.read_text(encoding="utf-8"))


def run_final_training(
    *,
    paths: FinalInputPaths | None = None,
    money_paths: dict[date, Path] | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Переобучает принятого кандидата на 2 млн строк и создаёт два CSV."""
    summary = load_validation_summary()
    if not summary["accepted"] and not force:
        raise RuntimeError(
            "Гипотеза не прошла защитный порог. Финальное обучение остановлено."
        )
    if paths is None:
        paths = final_input_paths()
    if money_paths is None:
        money_paths = {
            anchor: monetary_profile_path(MONETARY_CADENCE_PROFILE_DIR, anchor)
            for anchor in ALL_ANCHORS
        }
    validate_monetary_profiles(money_paths)
    base, seasonal, names = _feature_columns(paths)
    january_user, january_target, january_matrix = _load_full_anchor(
        anchor=JANUARY_ANCHOR,
        paths=paths,
        money_paths=money_paths,
        base_features=base,
        seasonal_features=seasonal,
        require_target=True,
    )
    matrix, target = _build_training_matrix(
        paths=paths,
        money_paths=money_paths,
        historical_anchors=HISTORICAL_ANCHORS,
        base_features=base,
        seasonal_features=seasonal,
        january_matrix=january_matrix,
        january_target=january_target,
        january_train_mask=np.ones(len(january_user), dtype=bool),
    )
    del january_user, january_target, january_matrix
    gc.collect()

    iterations = int(summary["final_iterations_if_accepted"])
    pool = Pool(matrix, label=np.log1p(target), feature_names=names)
    del matrix, target
    gc.collect()
    model = make_final_model(iterations=iterations, verbose=100)
    model.fit(pool)
    del pool
    gc.collect()
    model.save_model(str(MONETARY_CADENCE_DIR / "final_model.cbm"))

    test_user, _, test_matrix = _load_full_anchor(
        anchor=TEST_ANCHOR,
        paths=paths,
        money_paths=money_paths,
        base_features=base,
        seasonal_features=seasonal,
        require_target=False,
    )
    candidate = predict_gmv(model, Pool(test_matrix, feature_names=names))
    del test_matrix
    gc.collect()

    control_path = FINAL_SOLUTION_DIR / "predictions.parquet"
    control_frame = pd.read_parquet(control_path).set_index("user_id")
    control = control_frame.loc[test_user, "blend_prediction"].to_numpy(dtype=float)
    candidate_weight = float(summary["candidate_weight"])
    selected = log_blend(control, candidate, candidate_weight)
    if (
        not np.isfinite(candidate).all()
        or not np.isfinite(selected).all()
        or (candidate < 0).any()
        or (selected < 0).any()
    ):
        raise AssertionError("Финальный прогноз содержит некорректные значения.")

    predictions = pd.DataFrame(
        {
            "user_id": test_user,
            "control_prediction": control,
            "candidate_prediction": candidate,
            "selected_prediction": selected,
        }
    )
    predictions.to_parquet(MONETARY_CADENCE_DIR / "predictions.parquet", index=False)
    predictions[["user_id"]].assign(predict=candidate).to_csv(
        MONETARY_CADENCE_SUBMISSION_PATH, index=False
    )
    predictions[["user_id"]].assign(predict=selected).to_csv(
        MONETARY_CADENCE_BLEND_SUBMISSION_PATH, index=False
    )
    final_summary = {
        **summary,
        "final_training_rows": 2_000_000,
        "final_iterations": iterations,
        "candidate_submission": str(MONETARY_CADENCE_SUBMISSION_PATH),
        "selected_submission": str(MONETARY_CADENCE_BLEND_SUBMISSION_PATH),
        "prediction_rows": len(predictions),
        "selected_prediction_sum": float(selected.sum()),
    }
    (MONETARY_CADENCE_DIR / "final_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return final_summary
