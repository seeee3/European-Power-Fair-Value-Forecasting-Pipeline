"""
Hourly forecasts → delivery-block views.

Aggregates hourly next-day price forecasts into the standard block structure
used across European power markets. The blocks matter here because flexible
demand is scheduled in blocks, not hour by hour: a chiller plant or a
desalination train commits to a run window, so a block-level view of when the
grid is cheap and clean is what an operator can actually act on.

Definitions (Central European convention):
  - Baseload:  all 24 hours (arithmetic mean of hourly prices)
  - Peak:      hours 08–19 (CET/CEST), Mon–Fri only
  - Off-peak:  remaining hours

This module is deliberately decision-free. It reshapes the forecast and nothing
more; the load-shift recommendation built on top of it lives in `load_shift.py`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DeliveryView:
    period_label: str          # e.g. "2025-W01-Base", "2025-M01-Peak"
    start: pd.Timestamp
    end: pd.Timestamp
    forecast_mean: float
    forecast_p10: float        # 10th-percentile (lower tail)
    forecast_p90: float        # 90th-percentile (upper tail)
    n_hours: int
    product_type: str          # "base" | "peak" | "offpeak"


def _local_hour(idx: pd.DatetimeIndex) -> pd.Series:
    """Return local (CET/CEST) hour as integer Series."""
    return pd.Series(idx.tz_convert("Europe/Berlin").hour, index=idx)


def _is_peak(idx: pd.DatetimeIndex) -> pd.Series:
    """Peak = Mon–Fri, hours 08–19 (local CET/CEST)."""
    local = idx.tz_convert("Europe/Berlin")
    is_weekday = local.dayofweek < 5
    is_peak_hour = (local.hour >= 8) & (local.hour < 20)
    return pd.Series(is_weekday & is_peak_hour, index=idx)


def hourly_to_blocks(y_pred: pd.Series) -> pd.DataFrame:
    """
    Convert hourly forecast series into daily Base/Peak/Offpeak blocks.

    Returns a DataFrame with columns:
      date, base_avg, peak_avg, offpeak_avg, peak_spread
    """
    df = pd.DataFrame({"y_pred": y_pred})
    local_tz = y_pred.index.tz_convert("Europe/Berlin")
    df["date"] = local_tz.date
    df["is_peak"] = _is_peak(y_pred.index).values

    daily_base = df.groupby("date")["y_pred"].mean().rename("base_avg")
    daily_peak = df[df["is_peak"]].groupby("date")["y_pred"].mean().rename("peak_avg")
    daily_offpeak = df[~df["is_peak"]].groupby("date")["y_pred"].mean().rename("offpeak_avg")

    blocks = pd.concat([daily_base, daily_peak, daily_offpeak], axis=1)
    blocks["peak_spread"] = blocks["peak_avg"] - blocks["offpeak_avg"]
    blocks.index = pd.to_datetime(blocks.index)
    return blocks


def compute_delivery_views(
    y_pred: pd.Series,
    forecast_horizon: str = "week",
    y_lower: pd.Series | None = None,
    y_upper: pd.Series | None = None,
) -> list[DeliveryView]:
    """
    Aggregate hourly forecasts into DeliveryView objects.

    Args:
        y_pred:            Hourly point forecast Series (UTC DatetimeIndex)
        forecast_horizon:  'day' | 'week' | 'month'
        y_lower:           Hourly lower prediction interval (p10), or None
        y_upper:           Hourly upper prediction interval (p90), or None

    Prediction interval semantics:
        forecast_p10 / forecast_p90 are the average bounds over the delivery
        period, derived from empirical CV residual quantiles (calibrated by
        hour-of-day to capture intraday heteroskedasticity). They represent
        forecast *uncertainty*, not intraperiod price shape.
    """
    peak_mask = _is_peak(y_pred.index)
    views: list[DeliveryView] = []

    freq_map = {"day": "D", "week": "W-MON", "month": "ME"}
    freq = freq_map.get(forecast_horizon, "W-MON")

    for period, group in y_pred.groupby(pd.Grouper(freq=freq)):
        if len(group) == 0:
            continue
        label = period.strftime("%Y-W%W" if "W" in freq else "%Y-%m")
        for ptype, mask in [
            ("base", pd.Series(True, index=group.index)),
            ("peak", peak_mask.reindex(group.index, fill_value=False)),
            ("offpeak", ~peak_mask.reindex(group.index, fill_value=False)),
        ]:
            sub = group[mask.values]
            if len(sub) < 2:
                continue
            # Prediction interval: mean of the per-hour bounds over the period
            if y_lower is not None and y_upper is not None:
                sub_lo = y_lower.reindex(sub.index)
                sub_hi = y_upper.reindex(sub.index)
                p10 = round(float(sub_lo.mean()), 2)
                p90 = round(float(sub_hi.mean()), 2)
            else:
                p10 = round(float(np.percentile(sub, 10)), 2)
                p90 = round(float(np.percentile(sub, 90)), 2)
            views.append(
                DeliveryView(
                    period_label=f"{label}-{ptype}",
                    start=sub.index.min(),
                    end=sub.index.max(),
                    forecast_mean=round(float(sub.mean()), 2),
                    forecast_p10=p10,
                    forecast_p90=p90,
                    n_hours=len(sub),
                    product_type=ptype,
                )
            )
    return views


def views_to_dataframe(views: list[DeliveryView]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "period_label": v.period_label,
            "start": v.start,
            "end": v.end,
            "product_type": v.product_type,
            "forecast_mean_eur_mwh": v.forecast_mean,
            "forecast_p10": v.forecast_p10,
            "forecast_p90": v.forecast_p90,
            "n_hours": v.n_hours,
        }
        for v in views
    ])
