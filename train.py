"""End-to-end: load -> features -> time split -> baseline + LightGBM -> evaluate.

Usage: python train.py
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data import load_clean
from src.evaluate import (plot_pred_vs_actual, plot_store_forecast,
                          print_metrics)
from src.features import (apply_target_encodings, build_features,
                          fit_target_encodings)
from src.model import baseline_predict, train_lgbm


def time_split(df: pd.DataFrame):
    """Hold out the last VALIDATION_WEEKS as validation (no leakage)."""
    cutoff = df["Date"].max() - pd.Timedelta(weeks=config.VALIDATION_WEEKS)
    train = df[df["Date"] <= cutoff].copy()
    val = df[df["Date"] > cutoff].copy()
    return train, val, cutoff


def main() -> None:
    # 1. Load + engineer calendar/store features
    df = build_features(load_clean())
    print(f"Data: {len(df):,} rows | {df['Store'].nunique()} stores | "
          f"{df['Date'].min().date()} -> {df['Date'].max().date()}")

    # 2. Time-based split
    train_df, val_df, cutoff = time_split(df)
    print(f"Split at {cutoff.date()} | train {len(train_df):,} | val {len(val_df):,}")

    # 3. Fit historical encodings on TRAIN only, apply to both
    enc = fit_target_encodings(train_df)
    train_df = apply_target_encodings(train_df, enc)
    val_df = apply_target_encodings(val_df, enc)

    y_val = val_df[config.TARGET].to_numpy()

    # 4. Baseline + model
    print("\n=== Validation results (last 6 weeks) ===")
    results = {}
    results["Baseline (store x dow median)"] = print_metrics(
        "Baseline", y_val, baseline_predict(val_df))

    model, pred = train_lgbm(train_df, val_df)
    results["LightGBM"] = print_metrics("LightGBM", y_val, pred)

    # 5. Figures
    plot_pred_vs_actual(y_val, pred, "LightGBM")
    for store_id in [1, 262]:  # one ordinary + one big store
        if store_id in val_df["Store"].unique():
            plot_store_forecast(val_df, pred, store_id, "LightGBM")

    # 6. Persist
    joblib.dump(model, config.MODELS_DIR / "lgbm_model.joblib")
    joblib.dump(enc, config.MODELS_DIR / "encodings.joblib")
    with open(config.MODELS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    best = min(results, key=lambda k: results[k]["RMSPE"])
    print(f"\nBest: {best} (RMSPE {results[best]['RMSPE']:.4f})")
    print(f"Saved -> {config.MODELS_DIR / 'lgbm_model.joblib'}")


if __name__ == "__main__":
    main()
