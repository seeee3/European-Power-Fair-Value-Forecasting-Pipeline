"""Figures for the submission package."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from cobblestone.models.forecaster import FoldResult

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})
PALETTE = ["#1a6faf", "#e06c00", "#2ca02c", "#d62728"]


def fig_price_and_renewables(df: pd.DataFrame, out: Path) -> None:
    """
    Figure 1: DA price time series with wind+solar overlay.
    Shows the negative correlation between renewable output and price.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Weekly resampled for readability
    price_w = df["price_da_eur_mwh"].resample("W").mean()
    axes[0].plot(price_w.index, price_w.values, color=PALETTE[0], lw=1.5, label="DA price (weekly avg)")
    axes[0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[0].set_ylabel("EUR/MWh")
    axes[0].set_title("German Day-Ahead Price  ·  2022–2025")
    axes[0].legend(loc="upper right", fontsize=9)

    wind_cols = [c for c in ["wind_onshore_mwh", "wind_offshore_mwh"] if c in df.columns]
    if wind_cols:
        wind_w = df[wind_cols].sum(axis=1).resample("W").mean() / 1_000
        axes[1].fill_between(wind_w.index, wind_w.values, alpha=0.6, color=PALETTE[1], label="Wind (GWh)")
    if "solar_mwh" in df.columns:
        solar_w = df["solar_mwh"].resample("W").mean() / 1_000
        axes[1].fill_between(solar_w.index, solar_w.values, alpha=0.5, color="#f7c241", label="Solar (GWh)")
    axes[1].set_ylabel("Generation (GWh hourly avg)")
    axes[1].set_title("Wind + Solar Generation")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_cv_performance(fold_results: "list[FoldResult]", out: Path) -> None:
    """
    Figure 2: Walk-forward CV — actual vs predicted prices by fold.
    """
    fig, axes = plt.subplots(
        len(fold_results), 1,
        figsize=(14, 3 * len(fold_results)),
        sharex=False,
    )
    if len(fold_results) == 1:
        axes = [axes]

    for ax, fold in zip(axes, fold_results):
        ax.plot(fold.y_true.index, fold.y_true.values, color=PALETTE[0], lw=1.2, label="Actual", alpha=0.9)
        ax.plot(fold.y_pred.index, fold.y_pred.values, color=PALETTE[1], lw=1.2, label="Forecast", ls="--", alpha=0.9)
        ax.set_title(
            f"Fold {fold.fold}: {fold.val_start} → {fold.val_end}  "
            f"  MAE={fold.mae:.1f}  RMSE={fold.rmse:.1f}",
            fontsize=9,
        )
        ax.set_ylabel("EUR/MWh", fontsize=8)
        ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    fig.suptitle("Walk-Forward CV: Actual vs Forecast (LightGBM)", y=1.01)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_feature_importance(importances: pd.Series, out: Path, top_n: int = 20) -> None:
    """Figure 3: Top-N feature importances from LightGBM."""
    top = importances.head(top_n)
    fig, ax = plt.subplots(figsize=(8, top_n * 0.35 + 1))
    bars = ax.barh(top.index[::-1], top.values[::-1], color=PALETTE[0], alpha=0.85)
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(f"LightGBM Top {top_n} Feature Importances")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_price_heatmap(df: pd.DataFrame, out: Path) -> None:
    """Figure 4: Heatmap of average DA price by hour and month."""
    local = df["price_da_eur_mwh"].copy()
    local.index = local.index.tz_convert("Europe/Berlin")
    pivot = pd.DataFrame({
        "hour": local.index.hour,
        "month": local.index.month,
        "price": local.values,
    })
    heat = pivot.groupby(["hour", "month"])["price"].mean().unstack(level="month")

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    heat.columns = [month_labels[m - 1] for m in heat.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        heat, ax=ax, cmap="RdYlGn_r", annot=False,
        cbar_kws={"label": "EUR/MWh"}, linewidths=0.3,
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Hour (CET/CEST)")
    ax.set_title("Average DA Price by Hour of Day × Month (2022–2025)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_model_comparison(
    df_actuals: pd.DataFrame,
    baseline_preds: pd.Series,
    lgbm_preds: pd.Series,
    out: Path,
    sample_days: int = 14,
) -> None:
    """Figure 5: Baseline vs LightGBM on a recent window."""
    # Take the last sample_days of overlapping predictions
    common = lgbm_preds.dropna().index.intersection(baseline_preds.dropna().index)
    common = common[-sample_days * 24:]
    if len(common) == 0:
        return

    y_true = df_actuals.loc[common, "price_da_eur_mwh"]
    y_base = baseline_preds.reindex(common)
    y_lgbm = lgbm_preds.reindex(common)

    mae_base = float(np.nanmean(np.abs(y_true.values - y_base.values)))
    mae_lgbm = float(np.nanmean(np.abs(y_true.values - y_lgbm.values)))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(common, y_true.values, color="black", lw=1.5, label="Actual", zorder=3)
    ax.plot(common, y_base.values, color=PALETTE[2], lw=1.2, ls=":", label=f"Seasonal Naive (MAE={mae_base:.1f})")
    ax.plot(common, y_lgbm.values, color=PALETTE[1], lw=1.5, ls="--", label=f"LightGBM (MAE={mae_lgbm:.1f})", alpha=0.9)
    ax.set_ylabel("EUR/MWh")
    ax.set_title(f"Model Comparison — last {sample_days} days")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
