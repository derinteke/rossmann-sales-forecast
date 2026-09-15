"""Models: a naive historical baseline and a LightGBM regressor.

The target is log-transformed (log1p) before training because sales are
right-skewed and the metric (RMSPE) is relative — modelling log-sales makes the
errors roughly proportional, which is what RMSPE rewards.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np

from . import config
from .features import FEATURE_COLS


def baseline_predict(val_df) -> np.ndarray:
    """Naive baseline: historical median sales for this store & day-of-week.

    Uses the ``EncStoreDow`` feature (fitted on train), i.e. "on a typical
    <weekday> this store sells about X". A surprisingly strong, honest yardstick.
    """
    return val_df["EncStoreDow"].to_numpy()


def train_lgbm(train_df, val_df):
    """Train LightGBM on log-sales. Returns (model, val_predictions)."""
    y_train = np.log1p(train_df[config.TARGET].to_numpy())

    cat_features = ["StoreType", "Assortment", "StateHoliday", "DayOfWeek"]
    X_train = train_df[FEATURE_COLS].copy()
    X_val = val_df[FEATURE_COLS].copy()
    for c in cat_features:
        X_train[c] = X_train[c].astype("category")
        X_val[c] = X_val[c].astype("category")

    model = lgb.LGBMRegressor(
        random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1, **config.LGBM_PARAMS
    )
    model.fit(X_train, y_train, categorical_feature=cat_features)

    pred = np.expm1(model.predict(X_val))
    pred = np.clip(pred, 0, None)
    return model, pred
