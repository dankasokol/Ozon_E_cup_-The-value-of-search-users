"""Единые пути и параметры первого CatBoost-эксперимента."""

from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

TRAIN_PATH = DATA_DIR / "train.parquet"
FIRST_CATBOOST_ARTIFACT_DIR = ARTIFACTS_DIR / "first_catboost_ltv"
FIRST_CATBOOST_SNAPSHOT_DIR = FIRST_CATBOOST_ARTIFACT_DIR / "snapshots"
FIRST_CATBOOST_SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_first_catboost.csv"
MULTIFOLD_VALIDATION_DIR = ARTIFACTS_DIR / "multifold_validation"
CATBOOST_V2_ARTIFACT_DIR = ARTIFACTS_DIR / "catboost_v2"
CATBOOST_V2_SNAPSHOT_DIR = CATBOOST_V2_ARTIFACT_DIR / "snapshots"
CATBOOST_V2_SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_catboost_v2.csv"
CATBOOST_TUNING_DIR = ARTIFACTS_DIR / "catboost_tuning"
CATBOOST_REGULARIZATION_DIR = CATBOOST_TUNING_DIR / "regularization"
CATBOOST_RSM_DIR = CATBOOST_TUNING_DIR / "rsm"
CATBOOST_SEASONALITY_DIR = ARTIFACTS_DIR / "year_over_year_seasonality"
CATBOOST_CORRECTION_DIR = ARTIFACTS_DIR / "seasonal_correction"
FINAL_SUBMISSION_ARTIFACT_DIR = ARTIFACTS_DIR / "final_submission"
DEPTH6_SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_depth6_822.csv"
SEASONAL_SUBMISSION_PATH = (
    SUBMISSIONS_DIR / "submission_depth6_822_seasonal_correction.csv"
)

RANDOM_SEED = 42
HORIZON_DAYS = 30
ANCHOR_STEP_DAYS = 28
N_HISTORY_ANCHORS = 8
HISTORY_DAYS = 180
VALIDATION_ANCHORS = (
    date(2025, 10, 22),
    date(2025, 11, 19),
    date(2025, 12, 17),
    date(2026, 1, 14),
)


def ensure_output_dirs() -> None:
    """Создаёт каталоги для производных результатов, но не для исходных данных."""
    FIRST_CATBOOST_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FIRST_CATBOOST_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    MULTIFOLD_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    CATBOOST_V2_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    CATBOOST_V2_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    CATBOOST_TUNING_DIR.mkdir(parents=True, exist_ok=True)
    CATBOOST_REGULARIZATION_DIR.mkdir(parents=True, exist_ok=True)
    CATBOOST_RSM_DIR.mkdir(parents=True, exist_ok=True)
    CATBOOST_SEASONALITY_DIR.mkdir(parents=True, exist_ok=True)
    CATBOOST_CORRECTION_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_SUBMISSION_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
