"""Feature engineering for a 6-week-ahead sales forecast.

Design note (why no short lags): in the real task we predict six weeks with no
recent sales available, so lag-1/lag-7 features would leak. Instead we use
features that ARE known at forecast time — calendar, promo, store metadata — plus
historical aggregates fitted ONLY on the training split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

_MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["Date"]
    df["Year"] = d.dt.year
    df["Month"] = d.dt.month
    df["Day"] = d.dt.day
    df["WeekOfYear"] = d.dt.isocalendar().week.astype(int)
    df["DayOfYear"] = d.dt.dayofyear
    df["IsWeekend"] = (df["DayOfWeek"] >= 6).astype(int)
    df["IsMonthStart"] = d.dt.is_month_start.astype(int)
    df["IsMonthEnd"] = d.dt.is_month_end.astype(int)
    return df


def add_store_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Months since a competitor opened (0 if none/unknown)
    comp_open = (df["Year"] - df["CompetitionOpenSinceYear"]) * 12 + \
                (df["Month"] - df["CompetitionOpenSinceMonth"])
    df["CompetitionOpenMonths"] = np.where(
        df["CompetitionOpenSinceYear"] > 0, comp_open.clip(lower=0), 0
    )

    # Is Promo2 active for this store in this month?
    def _promo2_active(row):
        if row["Promo2"] == 0 or not row["PromoInterval"]:
            return 0
        return int(_MONTH_ABBR[row["Month"] - 1] in row["PromoInterval"].split(","))

    df["IsPromo2Month"] = df.apply(_promo2_active, axis=1)

    # Encode categoricals as pandas 'category' (LightGBM handles these natively)
    for col in ["StoreType", "Assortment", "StateHoliday"]:
        df[col] = df[col].astype("category")

    df["CompetitionDistanceLog"] = np.log1p(df["CompetitionDistance"])
    return df


def fit_target_encodings(train_df: pd.DataFrame) -> dict:
    """Compute historical sales aggregates from the TRAINING split only.

    Returns tables that are later merged onto any split (train/val/future) — this
    is the safe, no-leakage way to give the model store-level history.
    """
    enc = {}
    enc["store"] = train_df.groupby("Store")["Sales"].median().rename("EncStore")
    enc["store_dow"] = (train_df.groupby(["Store", "DayOfWeek"])["Sales"]
                        .median().rename("EncStoreDow"))
    enc["store_promo"] = (train_df.groupby(["Store", "Promo"])["Sales"]
                          .median().rename("EncStorePromo"))
    enc["global"] = float(train_df["Sales"].median())
    return enc


def apply_target_encodings(df: pd.DataFrame, enc: dict) -> pd.DataFrame:
    df = df.copy()
    df = df.merge(enc["store"], on="Store", how="left")
    df = df.merge(enc["store_dow"], on=["Store", "DayOfWeek"], how="left")
    df = df.merge(enc["store_promo"], on=["Store", "Promo"], how="left")
    for col in ["EncStore", "EncStoreDow", "EncStorePromo"]:
        df[col] = df[col].fillna(enc["global"])
    return df


FEATURE_COLS = [
    # promo / holiday (known ahead of time)
    "Promo", "SchoolHoliday", "StateHoliday", "IsPromo2Month", "Promo2",
    # calendar
    "DayOfWeek", "Year", "Month", "Day", "WeekOfYear", "DayOfYear",
    "IsWeekend", "IsMonthStart", "IsMonthEnd",
    # store metadata
    "StoreType", "Assortment", "CompetitionDistanceLog", "CompetitionOpenMonths",
    # historical aggregates (fitted on train only)
    "EncStore", "EncStoreDow", "EncStorePromo",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar + store features (target encodings added separately after split)."""
    return add_store_features(add_calendar(df))
