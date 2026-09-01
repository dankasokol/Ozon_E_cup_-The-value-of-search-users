"""Единые пути и общие параметры итогового решения."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# Большие рабочие Parquet нельзя хранить на синхронизируемом Рабочем столе:
# при нехватке места macOS превращает их в облачные указатели без локального
# содержимого. Путь можно переопределить переменной OZON_GMV_RUNTIME_DIR.
LOCAL_RUNTIME_DIR = Path(
    os.environ.get(
        "OZON_GMV_RUNTIME_DIR",
        str(Path.home() / "Library" / "Caches" / "Ozon_GMV"),
    )
)

TRAIN_PATH = DATA_DIR / "train.parquet"
FINAL_SOLUTION_DIR = ARTIFACTS_DIR / "final_solution"
FINAL_SOLUTION_CACHE_DIR = LOCAL_RUNTIME_DIR / "final_solution_cache"
FINAL_SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_annual_cadence_blend.csv"

# Денежный слой выбранного итогового решения.
MONETARY_CADENCE_DIR = LOCAL_RUNTIME_DIR / "monetary_cadence"
MONETARY_CADENCE_PROFILE_DIR = MONETARY_CADENCE_DIR / "profiles"
MONETARY_CADENCE_CONTROL_SOURCE_PATH = (
    ARTIFACTS_DIR / "monetary_cadence" / "control_validation_predictions.parquet"
)
MONETARY_CADENCE_CONTROL_PATH = (
    MONETARY_CADENCE_DIR / "control_validation_predictions.parquet"
)
MONETARY_CADENCE_SUBMISSION_PATH = (
    SUBMISSIONS_DIR / "submission_monetary_cadence.csv"
)
MONETARY_CADENCE_BLEND_SUBMISSION_PATH = (
    SUBMISSIONS_DIR / "submission_monetary_cadence_blend.csv"
)

RANDOM_SEED = 42
HORIZON_DAYS = 30
HISTORY_DAYS = 180


def ensure_output_dirs() -> None:
    """Создаёт каталоги итогового решения и активного опыта."""
    FINAL_SOLUTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MONETARY_CADENCE_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
