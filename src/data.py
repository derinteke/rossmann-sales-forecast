"""Load, merge and clean the Rossmann data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_raw() -> pd.DataFrame:
    """Load train + store, merge, parse dates."""
    train = pd.read_csv(config.TRAIN_CSV, dtype={"StateHoliday": str}, parse_dates=["Date"])
    store = pd.read_csv(config.STORE_CSV)
    df = train.merge(store, on="Store", how="left")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the merged frame.

    - Keep only open days with positive sales (closed days have Sales = 0 and are
      excluded from the competition metric).
    - Normalise StateHoliday ('0' vs 0) and cast flags.
    - Fill store-level NaNs (competition / promo2 fields) sensibly.
    """
    df = df.copy()

    # Train on open days with real sales only
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()

    # StateHoliday: unify to {'0','a','b','c'}
    df["StateHoliday"] = df["StateHoliday"].replace(0, "0").astype(str)

    # CompetitionDistance: missing -> treat as "very far" (large value)
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
        df["CompetitionDistance"].max() * 2
    )

    # Competition / Promo2 date fields: missing -> 0 (meaning "no competitor / no promo2")
    for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                "Promo2SinceWeek", "Promo2SinceYear"]:
        df[col] = df[col].fillna(0).astype(int)

    df["PromoInterval"] = df["PromoInterval"].fillna("")
    return df


def load_clean() -> pd.DataFrame:
    return clean(load_raw())


if __name__ == "__main__":
    d = load_clean()
    print(d.shape)
    print(d["Date"].min().date(), "->", d["Date"].max().date())
    print(d[[config.TARGET, "Customers"]].describe().round(0))
