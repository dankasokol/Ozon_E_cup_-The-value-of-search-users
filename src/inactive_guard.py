"""Консервативная смесь для пользователей без активных дней заказов.

Гипотеза фиксируется до проверки на подтверждающей части: если за доступную
историю у пользователя не было ни одного дня с заказом, новая денежная модель
не используется. Для остальных остаётся ранее выбранная смесь 60/40.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from .config import (
    ARTIFACTS_DIR,
    FINAL_SOLUTION_CACHE_DIR,
    MONETARY_CADENCE_CONTROL_PATH,
    MONETARY_CADENCE_DIR,
    SUBMISSIONS_DIR,
)
from .final_solution import TEST_ANCHOR, final_input_paths


EXPERIMENT_DIR = ARTIFACTS_DIR / "inactive_guard"
SUBMISSION_PATH = (
    SUBMISSIONS_DIR / "submission_monetary_cadence_inactive_guard.csv"
)
ORDER_ACTIVITY_COLUMN = "order_cadence_active_days"


def rmsle(target: np.ndarray, prediction: np.ndarray) -> float:
    target_log = np.log1p(np.clip(np.asarray(target, dtype=float), 0.0, None))
    prediction_log = np.log1p(
        np.clip(np.asarray(prediction, dtype=float), 0.0, None)
    )
    return float(np.sqrt(np.mean(np.square(target_log - prediction_log))))


def log_blend(
    control: np.ndarray,
    candidate: np.ndarray,
    candidate_weight: np.ndarray | float,
) -> np.ndarray:
    control_log = np.log1p(np.clip(np.asarray(control, dtype=float), 0.0, None))
    candidate_log = np.log1p(
        np.clip(np.asarray(candidate, dtype=float), 0.0, None)
    )
    return np.expm1(
        (1.0 - candidate_weight) * control_log
        + candidate_weight * candidate_log
    )


def _joined_validation() -> pd.DataFrame:
    predictions = pd.read_parquet(
        MONETARY_CADENCE_DIR / "validation_predictions.parquet"
    )
    activity = pd.read_parquet(MONETARY_CADENCE_CONTROL_PATH)[
        ["role", "user_id", "order_active_days"]
    ]
    if activity.duplicated(["role", "user_id"]).any():
        raise AssertionError("В контрольном файле есть дубликаты пользователей.")
    joined = predictions.merge(
        activity, on=["role", "user_id"], how="left", validate="1:1"
    )
    if len(joined) != 50_000 or joined.order_active_days.isna().any():
        raise AssertionError("Соединение изменило контрольную выборку.")
    return joined


def evaluate_guard(*, bootstrap_repeats: int = 2_000) -> dict[str, object]:
    """Проверяет заранее заданное правило на tune, затем на confirm."""
    summary = json.loads(
        (MONETARY_CADENCE_DIR / "validation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    global_weight = float(summary["candidate_weight"])
    if global_weight != 0.4:
        raise AssertionError("Опыт рассчитан для зафиксированной смеси с весом 0.4.")

    frame = _joined_validation()
    metric_rows: list[dict[str, object]] = []
    role_predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for role in ("tune", "confirm"):
        part = frame.loc[frame.role == role].reset_index(drop=True)
        target = part.target.to_numpy(dtype=float)
        control = part.control_prediction.to_numpy(dtype=float)
        candidate = part.candidate_prediction.to_numpy(dtype=float)
        global_prediction = log_blend(control, candidate, global_weight)
        guard_weight = np.where(
            part.order_active_days.to_numpy(dtype=float) == 0.0,
            0.0,
            global_weight,
        )
        guard_prediction = log_blend(control, candidate, guard_weight)
        if not np.allclose(
            global_prediction,
            part.selected_prediction.to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise AssertionError("Не воспроизвелась исходная смесь 60/40.")
        role_predictions[role] = (target, global_prediction, guard_prediction)
        control_score = rmsle(target, control)
        global_score = rmsle(target, global_prediction)
        guard_score = rmsle(target, guard_prediction)
        metric_rows.append(
            {
                "role": role,
                "users": len(part),
                "inactive_users": int((guard_weight == 0.0).sum()),
                "control_rmsle": control_score,
                "global_blend_rmsle": global_score,
                "guard_rmsle": guard_score,
                "guard_gain_vs_control": control_score - guard_score,
                "guard_gain_vs_global_blend": global_score - guard_score,
            }
        )

    # Доля смеси выбрана только по tune. Confirm используется один раз для
    # проверки знака и устойчивости заранее зафиксированного правила.
    target, global_prediction, guard_prediction = role_predictions["confirm"]
    rng = np.random.default_rng(42)
    improvements = np.empty(bootstrap_repeats, dtype=float)
    for index in range(bootstrap_repeats):
        sample = rng.integers(0, len(target), size=len(target))
        improvements[index] = rmsle(
            target[sample], global_prediction[sample]
        ) - rmsle(target[sample], guard_prediction[sample])

    metrics = pd.DataFrame(metric_rows)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(EXPERIMENT_DIR / "validation_metrics.csv", index=False)
    tune_gain = float(
        metrics.loc[
            metrics.role == "tune", "guard_gain_vs_global_blend"
        ].item()
    )
    confirm_gain = float(
        metrics.loc[
            metrics.role == "confirm", "guard_gain_vs_global_blend"
        ].item()
    )
    result: dict[str, object] = {
        "experiment": "inactive_order_guard_v1",
        "rule": "candidate_weight=0 if order_active_days=0 else 0.4",
        "rule_fixed_on": "tune",
        "tune_gain_vs_global_blend": tune_gain,
        "confirm_gain_vs_global_blend": confirm_gain,
        "confirm_bootstrap_ci_low": float(np.quantile(improvements, 0.025)),
        "confirm_bootstrap_ci_high": float(np.quantile(improvements, 0.975)),
        "confirm_bootstrap_positive_share": float((improvements > 0).mean()),
        "accepted": bool(tune_gain > 0.0 and confirm_gain > 0.0),
    }
    (EXPERIMENT_DIR / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def build_submission() -> dict[str, object]:
    """Создаёт итоговый CSV с тем же правилом на тестовых пользователях."""
    result = evaluate_guard()
    if not result["accepted"]:
        raise RuntimeError("Правило не улучшило обе контрольные части.")

    summary = json.loads(
        (MONETARY_CADENCE_DIR / "validation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    global_weight = float(summary["candidate_weight"])
    predictions = pl.read_parquet(
        MONETARY_CADENCE_DIR / "predictions.parquet"
    ).with_row_index("_row")
    cadence_path = final_input_paths(FINAL_SOLUTION_CACHE_DIR).cadence_profiles[
        TEST_ANCHOR
    ]
    activity = pl.read_parquet(
        cadence_path, columns=["user_id", ORDER_ACTIVITY_COLUMN]
    )
    if activity.height != 250_000 or activity["user_id"].n_unique() != 250_000:
        raise AssertionError("Некорректный тестовый профиль активности.")
    joined = predictions.join(
        activity, on="user_id", how="left", validate="1:1"
    ).sort("_row")
    if joined.height != 250_000 or joined[ORDER_ACTIVITY_COLUMN].null_count() != 0:
        raise AssertionError("Соединение изменило тестовых пользователей.")

    control = joined["control_prediction"].to_numpy()
    candidate = joined["candidate_prediction"].to_numpy()
    active_days = joined[ORDER_ACTIVITY_COLUMN].to_numpy()
    weights = np.where(active_days == 0.0, 0.0, global_weight)
    prediction = log_blend(control, candidate, weights)
    if not np.allclose(
        prediction[active_days > 0],
        joined["selected_prediction"].to_numpy()[active_days > 0],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError("Для активных пользователей изменилась исходная смесь.")
    if not np.allclose(
        prediction[active_days == 0],
        control[active_days == 0],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError("Для неактивных пользователей не вернулся контроль.")
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise AssertionError("Итоговый прогноз содержит некорректные значения.")

    output = pd.DataFrame(
        {"user_id": joined["user_id"].to_numpy(), "predict": prediction}
    )
    if len(output) != 250_000 or not output.user_id.is_unique:
        raise AssertionError("Некорректная детализация итогового файла.")
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(SUBMISSION_PATH, index=False)
    final = {
        **result,
        "submission_path": str(SUBMISSION_PATH),
        "rows": len(output),
        "inactive_users": int((active_days == 0).sum()),
        "inactive_share": float((active_days == 0).mean()),
        "prediction_sum": float(prediction.sum()),
        "prediction_mean": float(prediction.mean()),
        "prediction_median": float(np.median(prediction)),
        "zero_share": float((prediction == 0).mean()),
    }
    (EXPERIMENT_DIR / "final_summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return final


if __name__ == "__main__":
    print(json.dumps(build_submission(), ensure_ascii=False, indent=2))
