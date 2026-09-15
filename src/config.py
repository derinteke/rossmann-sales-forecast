"""Project configuration: paths, columns, split and model settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data" / "raw" / "train.csv"
STORE_CSV = ROOT / "data" / "raw" / "store.csv"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
for _d in (MODELS_DIR, FIGURES_DIR, ROOT / "data" / "processed"):
    _d.mkdir(parents=True, exist_ok=True)

TARGET = "Sales"
DATE = "Date"

# We forecast the last 6 weeks of the training period (mirrors the Kaggle test window).
VALIDATION_WEEKS = 6

RANDOM_STATE = 42

# LightGBM params — tuned with Optuna (src/tune.py). n_estimators chosen by an
# inner time-based early-stopping split, so the real-val window was never peeked.
# Tuned model: real-val RMSPE 0.1244 (vs 0.1280 for the earlier hand-set params).
LGBM_PARAMS = {
    "n_estimators": 697,
    "learning_rate": 0.06841,
    "num_leaves": 582,
    "min_child_samples": 36,
    "subsample": 0.9371,
    "subsample_freq": 1,
    "colsample_bytree": 0.5595,
    "reg_alpha": 0.006472,
    "reg_lambda": 0.03485,
}
