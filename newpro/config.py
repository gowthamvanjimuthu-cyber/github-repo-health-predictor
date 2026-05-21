"""Project configuration and paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

DATASET_PATH = DATA_DIR / "repository_data.csv"
MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"

# Features used for training (must match dataset columns except status)
FEATURE_COLUMNS = [
    "commits_per_week",
    "pull_requests",
    "issues_opened",
    "issues_closed",
    "contribution_days",
    "contributors_count",
    "stars",
    "forks",
    "watchers",
    "last_commit_days_ago",
]

TARGET_COLUMN = "status"
