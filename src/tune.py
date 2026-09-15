"""Optuna hyperparameter tuning for the Rossmann LightGBM model.

Rigorous, no-peeking setup:
  - REAL validation  = last 6 weeks (never touched during tuning)
  - TUNE validation  = the 6 weeks *before* that
We optimise RMSPE on TUNE-val (early stopping on it), then the winning params are
evaluated by train.py on the untouched REAL-val window.

For speed the search runs on a representative subset of stores (hyperparameter
ranking transfers); the final model in train.py uses all 1,115 stores.

Study is stored in SQLite so it resumes across runs:
    python -m src.tune --trials 15
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from . import config
from .data import load_clean
from .evaluate import rmspe
from .features import (apply_target_encodings, build_features,
                       fit_target_encodings, FEATURE_COLS)

optuna.logging.set_verbosity(optuna.logging.WARNING)
CAT = ["StoreType", "Assortment", "StateHoliday", "DayOfWeek"]
STORAGE = "sqlite:////tmp/rossmann_optuna.db"
STUDY = "rossmann_lgbm"
CACHE = Path("/tmp/rossmann_tune_cache.pkl")
N_TUNE_STORES = 400


def _prepare():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    df = build_features(load_clean())
    rng = np.random.RandomState(config.RANDOM_STATE)
    stores = rng.choice(df["Store"].unique(), N_TUNE_STORES, replace=False)
    df = df[df["Store"].isin(stores)].copy()

    real_cut = df["Date"].max() - pd.Timedelta(weeks=config.VALIDATION_WEEKS)
    tune_cut = real_cut - pd.Timedelta(weeks=config.VALIDATION_WEEKS)
    tr = df[df["Date"] <= tune_cut].copy()
    va = df[(df["Date"] > tune_cut) & (df["Date"] <= real_cut)].copy()
    enc = fit_target_encodings(tr)
    tr = apply_target_encodings(tr, enc); va = apply_target_encodings(va, enc)

    Xtr = tr[FEATURE_COLS].copy(); Xva = va[FEATURE_COLS].copy()
    for c in CAT:
        Xtr[c] = Xtr[c].astype("category"); Xva[c] = Xva[c].astype("category")
    ytr = np.log1p(tr[config.TARGET].to_numpy())
    yva = va[config.TARGET].to_numpy()
    out = (Xtr, ytr, Xva, yva)
    CACHE.write_bytes(pickle.dumps(out))
    return out


def make_objective(Xtr, ytr, Xva, yva):
    dtrain = lgb.Dataset(Xtr, label=ytr, categorical_feature=CAT, free_raw_data=False)
    dvalid = lgb.Dataset(Xva, label=np.log1p(yva), categorical_feature=CAT,
                         reference=dtrain, free_raw_data=False)

    def objective(trial):
        params = {
            "objective": "regression", "metric": "rmse", "verbosity": -1,
            "feature_pre_filter": False, "seed": config.RANDOM_STATE,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.09, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 63, 800),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
        model = lgb.train(params, dtrain, num_boost_round=1500, valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        pred = np.clip(np.expm1(model.predict(Xva, num_iteration=model.best_iteration)), 0, None)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        return rmspe(yva, pred)

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=15)
    args = ap.parse_args()
    Xtr, ytr, Xva, yva = _prepare()
    study = optuna.create_study(
        direction="minimize", study_name=STUDY, storage=STORAGE, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE))
    study.optimize(make_objective(Xtr, ytr, Xva, yva), n_trials=args.trials)
    print(f"Trials so far: {len(study.trials)} | Best TUNE-val RMSPE: {study.best_value:.4f}")
    best = dict(study.best_params)
    best["n_estimators"] = study.best_trial.user_attrs.get("best_iteration", 1000)
    print("Best params:", best)


if __name__ == "__main__":
    main()
