"""
DA → Prompt Curve Translation.

Converts hourly next-day price forecasts into delivery-period views
that can directly inform prompt month/quarter positioning.

Definitions (German market convention):
  - Baseload:  all 24 hours (MW average = arithmetic mean of hourly prices)
  - Peak:      hours 08–19 (CET/CEST), Mon–Fri only
  - Off-peak:  remaining hours
  - Prompt month:  nearest calendar month not yet expired (Cal+1)
  - Prompt quarter: nearest calendar quarter not yet expired

Trading signal framework:
  The desk uses the fair-value band to decide directional bias and
  whether to buy/sell the prompt month against the spot roll.
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


def generate_trading_signal(
    views: list[DeliveryView],
    model_mae: float,
    current_prompt_price: float | None = None,
) -> dict[str, Any]:
    """
    Translate delivery views into an actionable trading signal.

    Logic:
      - If forecast_mean > current_prompt + MAE → directional long bias (prompt month)
      - If forecast_mean < current_prompt - MAE → directional short bias
      - Otherwise → no edge, stand aside
      - peak_spread > 0 → shape: buy peak, sell base
      - Confidence band width (p90-p10) determines position sizing

    Invalidation conditions (must be stated explicitly per case study requirements):
      - Unexpected nuclear outage announcement → upside to base load
      - Extreme weather event (cold snap / wind drought) → override fundamentals
      - Cross-border congestion changes → price zone divergence
      - Model MAE > 30 EUR/MWh → signal unreliable, reduce size
    """
    base_views = [v for v in views if v.product_type == "base"]
    peak_views = [v for v in views if v.product_type == "peak"]

    if not base_views:
        return {"signal": "no_data", "reasoning": "No base delivery views available"}

    # Use next full delivery period as the actionable view
    primary = base_views[0]
    band_width = primary.forecast_p90 - primary.forecast_p10

    direction = "neutral"
    reasoning = []
    if current_prompt_price is not None:
        diff = primary.forecast_mean - current_prompt_price
        if diff > model_mae:
            direction = "long"
            reasoning.append(
                f"Fair value {primary.forecast_mean:.1f} exceeds prompt {current_prompt_price:.1f} "
                f"by {diff:.1f} EUR/MWh (>{model_mae:.1f} MAE threshold) → buy prompt base"
            )
        elif diff < -model_mae:
            direction = "short"
            reasoning.append(
                f"Fair value {primary.forecast_mean:.1f} below prompt {current_prompt_price:.1f} "
                f"by {abs(diff):.1f} EUR/MWh → sell prompt base"
            )
        else:
            reasoning.append(
                f"Forecast {primary.forecast_mean:.1f} within ±MAE of prompt {current_prompt_price:.1f} "
                "→ no directional edge"
            )
    else:
        reasoning.append("No current prompt price provided; cannot compute directional signal")

    peak_spread_signal = None
    if peak_views:
        pk = peak_views[0]
        spread = pk.forecast_mean - primary.forecast_mean
        peak_spread_signal = {
            "peak_vs_base_spread_eur_mwh": round(spread, 2),
            "action": "buy peak vs base" if spread > 5 else "flatten shape" if spread < 2 else "hold",
        }
        reasoning.append(
            f"Peak/base spread forecast: {spread:.1f} EUR/MWh → {peak_spread_signal['action']}"
        )

    return {
        "period": primary.period_label,
        "forecast_base_eur_mwh": primary.forecast_mean,
        "current_prompt_price_eur_mwh": round(current_prompt_price, 2) if current_prompt_price is not None else None,
        "confidence_band": [primary.forecast_p10, primary.forecast_p90],
        "band_width_eur_mwh": round(band_width, 2),
        "model_mae_threshold_eur_mwh": round(model_mae, 2),
        "direction": direction,
        "peak_spread": peak_spread_signal,
        "reasoning": reasoning,
        "invalidation_conditions": [
            "Unplanned nuclear outage (>1 GW) announcement within 4 hours of gate closure",
            "Extreme cold snap (load +15% vs forecast) not captured in fundamentals",
            "Wind drought: generation <30% of installed capacity for ≥3 consecutive days",
            "Cross-border congestion causing price zone split from NL/FR >€20/MWh",
            f"Model out-of-sample MAE > 2× training MAE ({2 * model_mae:.0f} EUR/MWh) → reduce position to 0",
        ],
        "desk_actions": [
            "Express directional bias via prompt month baseload futures (EEX)",
            "Express shape via peak/offpeak spread products (EEX Peak/Offpeak)",
            "Size position by band_width: narrower band → larger position",
        ],
    }


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
