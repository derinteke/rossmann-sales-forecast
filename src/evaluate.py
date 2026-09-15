"""Metrics (RMSPE / RMSE / MAE) and forecast plots."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config


def rmspe(y_true, y_pred) -> float:
    """Root Mean Squared Percentage Error — the Rossmann competition metric.

    Only defined where y_true > 0 (closed days are excluded upstream).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)))


def metrics(y_true, y_pred) -> dict:
    return {
        "RMSPE": rmspe(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def print_metrics(name: str, y_true, y_pred) -> dict:
    m = metrics(y_true, y_pred)
    print(f"{name:22s} | RMSPE {m['RMSPE']:.4f} | RMSE {m['RMSE']:.1f} | MAE {m['MAE']:.1f}")
    return m


def plot_store_forecast(val_df: pd.DataFrame, y_pred, store_id: int, name: str):
    """Actual vs predicted daily sales for a single store over the validation window."""
    v = val_df.copy()
    v["Pred"] = y_pred
    v = v[v["Store"] == store_id].sort_values("Date")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(v["Date"], v["Sales"], label="Actual", color="#1f2937", lw=1.8)
    ax.plot(v["Date"], v["Pred"], label="Forecast", color="#ef4444", lw=1.8, ls="--")
    ax.set_title(f"Store {store_id} — actual vs forecast (validation)")
    ax.set_ylabel("Sales"); ax.legend()
    fig.tight_layout()
    path = config.FIGURES_DIR / f"forecast_store_{store_id}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def plot_pred_vs_actual(y_true, y_pred, name: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    idx = np.random.RandomState(0).choice(len(y_true), size=min(4000, len(y_true)), replace=False)
    ax.scatter(np.asarray(y_true)[idx], np.asarray(y_pred)[idx], s=6, alpha=0.25, color="#2563eb")
    lim = max(np.max(y_true), np.max(y_pred))
    ax.plot([0, lim], [0, lim], "k--", alpha=0.6)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title(f"Predicted vs actual — {name}")
    fig.tight_layout()
    path = config.FIGURES_DIR / "pred_vs_actual.png"
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path
