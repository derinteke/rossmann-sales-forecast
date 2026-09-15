"""Generate EDA figures for the report."""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from . import config
from .data import load_clean

sns.set_theme(style="whitegrid")
BLUE = "#2563eb"


def _save(fig, name):
    p = config.FIGURES_DIR / name
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); return p


def main():
    df = load_clean()
    df["Month"] = df["Date"].dt.month

    # 1. Weekly pattern
    fig, ax = plt.subplots(figsize=(7, 4))
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df.groupby("DayOfWeek")["Sales"].mean().plot.bar(ax=ax, color=BLUE)
    ax.set_xticklabels(days, rotation=0); ax.set_xlabel("")
    ax.set_title("Average sales by day of week"); ax.set_ylabel("Avg sales")
    _save(fig, "eda_weekly_pattern.png")

    # 2. Promo effect
    fig, ax = plt.subplots(figsize=(5, 4))
    m = df.groupby("Promo")["Sales"].mean()
    ax.bar(["No promo", "Promo"], m.values, color=["#94a3b8", BLUE])
    for i, v in enumerate(m.values):
        ax.text(i, v + 50, f"{v:,.0f}", ha="center")
    ax.set_title(f"Promo lifts sales by {m[1]/m[0]-1:.0%}"); ax.set_ylabel("Avg sales")
    _save(fig, "eda_promo_effect.png")

    # 3. Monthly seasonality
    fig, ax = plt.subplots(figsize=(7, 4))
    df.groupby("Month")["Sales"].mean().plot(ax=ax, marker="o", color=BLUE)
    ax.set_title("Monthly seasonality (Dec peak)"); ax.set_xlabel("Month")
    ax.set_ylabel("Avg sales"); ax.set_xticks(range(1, 13))
    _save(fig, "eda_seasonality.png")

    # 4. Sales distribution by store type
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="StoreType", y="Sales", ax=ax, color=BLUE, showfliers=False,
                order=sorted(df["StoreType"].unique()))
    ax.set_title("Sales distribution by store type")
    _save(fig, "eda_store_type.png")

    print("EDA figures saved to", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
