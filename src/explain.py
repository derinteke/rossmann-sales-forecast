"""SHAP feature-importance summary for the LightGBM sales model."""
from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import config
from .data import load_clean
from .features import (apply_target_encodings, build_features,
                       fit_target_encodings, FEATURE_COLS)


def main():
    df = build_features(load_clean())
    cutoff = df["Date"].max() - pd.Timedelta(weeks=config.VALIDATION_WEEKS)
    train_df = df[df["Date"] <= cutoff].copy()
    enc = fit_target_encodings(train_df)
    train_df = apply_target_encodings(train_df, enc)

    model = joblib.load(config.MODELS_DIR / "lgbm_model.joblib")

    cat = ["StoreType", "Assortment", "StateHoliday", "DayOfWeek"]
    X = train_df[FEATURE_COLS].sample(2000, random_state=0).copy()
    for c in cat:
        X[c] = X[c].astype("category")

    sv = shap.TreeExplainer(model).shap_values(X)
    plt.figure()
    shap.summary_plot(sv, X, plot_type="bar", max_display=15, show=False)
    plt.title("SHAP feature importance (LightGBM)")
    plt.tight_layout()
    p = config.FIGURES_DIR / "shap_importance.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print("SHAP figure saved to", p)


if __name__ == "__main__":
    main()
