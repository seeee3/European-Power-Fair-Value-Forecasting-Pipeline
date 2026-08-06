"""
Carbon-aware load shifting: turning a price forecast into a dispatch decision.

The premise
-----------
In a merit-order market the day-ahead price is set by the *marginal* generator.
When wind and solar are abundant they push the expensive thermal plants out of
the stack, and the clearing price falls; when renewables are scarce, gas and
coal set the price and it rises. Price and carbon intensity therefore move
together, and a 24h-ahead price forecast is implicitly a 24h-ahead forecast of
*when the grid is clean*.

That makes the price curve directly actionable for any operator with flexible
demand — district cooling, desalination, EV fleets, data centre batch workloads.
Shift consumption out of the dirtiest hours into the cleanest ones and you cut
both cost and emissions without reducing the service delivered.

`validate_price_carbon_link` tests that premise on real data rather than
assuming it. Everything downstream is conditional on it holding.

Emissions accounting
--------------------
Carbon intensity per hour is estimated from the observed renewable share:

    intensity(h) = RESIDUAL_INTENSITY * (1 - renewable_share(h))

where `renewable_share = (wind + solar) / load`, clipped to [0, 1], and
RESIDUAL_INTENSITY is the average intensity of the non-renewable residual
generation mix. This is a transparent approximation, not a metered figure —
see `docs` in the proposal for its limitations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Average CO2 intensity of the non-renewable residual generation mix (tCO2/MWh).
# Germany's residual stack is roughly half hard coal/lignite (~0.9-1.1) and half
# gas CCGT (~0.35-0.40), giving a blended figure near 0.6. Parameterised so it
# can be re-pointed at another grid's fuel mix.
RESIDUAL_INTENSITY_T_PER_MWH = 0.60

# Fraction of hours assigned to the clean and dirty bands within each day.
GREEN_QUANTILE = 0.33
RED_QUANTILE = 0.67


@dataclass
class ShiftWindow:
    """A contiguous or scattered set of hours sharing a carbon band."""

    band: str                      # "green" | "amber" | "red"
    hours_local: list[int]         # local clock hours in the band
    mean_price_eur_mwh: float
    mean_renewable_share: float
    mean_intensity_t_per_mwh: float
    n_hours: int


@dataclass
class ShiftPlan:
    period_label: str
    green: ShiftWindow
    red: ShiftWindow
    price_saving_eur_per_mwh: float
    co2_avoided_t_per_mwh: float
    confidence: str
    evidence: dict[str, Any] = field(default_factory=dict)
    override_conditions: list[str] = field(default_factory=list)


def renewable_share(df: pd.DataFrame) -> pd.Series:
    """
    Hourly renewable share of load, in [0, 1].

    Uses whichever of wind onshore/offshore/solar are present. Returns an empty
    Series if load or all renewable columns are unavailable.
    """
    if "load_mwh" not in df.columns:
        return pd.Series(dtype=float)

    ren_cols = [c for c in ("wind_onshore_mwh", "wind_offshore_mwh", "solar_mwh")
                if c in df.columns]
    if not ren_cols:
        return pd.Series(dtype=float)

    total_ren = df[ren_cols].sum(axis=1, min_count=1)
    share = (total_ren / df["load_mwh"]).clip(lower=0.0, upper=1.0)
    return share.dropna()


def carbon_intensity(
    share: pd.Series,
    residual_intensity: float = RESIDUAL_INTENSITY_T_PER_MWH,
) -> pd.Series:
    """Estimated grid carbon intensity (tCO2/MWh) from renewable share."""
    return residual_intensity * (1.0 - share)


def validate_price_carbon_link(df: pd.DataFrame) -> dict[str, Any]:
    """
    Test the premise that price tracks carbon intensity.

    Returns the Pearson and Spearman correlation between the day-ahead price and
    the estimated carbon intensity, plus the mean intensity in the cheapest and
    most expensive terciles of hours. A strong positive correlation is what makes
    the price forecast usable as a carbon signal; a weak one invalidates the
    whole approach and the caller should say so rather than proceed.
    """
    share = renewable_share(df)
    if share.empty or "price_da_eur_mwh" not in df.columns:
        return {"status": "unavailable", "reason": "missing load, renewable or price data"}

    intensity = carbon_intensity(share)
    price = df.loc[intensity.index, "price_da_eur_mwh"].dropna()
    intensity = intensity.loc[price.index]

    if len(price) < 24:
        return {"status": "unavailable", "reason": f"only {len(price)} usable hours"}

    pearson = float(price.corr(intensity, method="pearson"))
    spearman = float(price.corr(intensity, method="spearman"))

    cheap_cut, exp_cut = price.quantile([GREEN_QUANTILE, RED_QUANTILE])
    cheap_hours = intensity[price <= cheap_cut]
    exp_hours = intensity[price >= exp_cut]

    return {
        "status": "ok",
        "n_hours": int(len(price)),
        "pearson_r": round(pearson, 3),
        "spearman_r": round(spearman, 3),
        "mean_intensity_cheapest_tercile_t_per_mwh": round(float(cheap_hours.mean()), 3),
        "mean_intensity_priciest_tercile_t_per_mwh": round(float(exp_hours.mean()), 3),
        "intensity_gap_t_per_mwh": round(
            float(exp_hours.mean() - cheap_hours.mean()), 3
        ),
        "mean_renewable_share_cheapest_tercile": round(
            float(share.loc[cheap_hours.index].mean()), 3
        ),
        "mean_renewable_share_priciest_tercile": round(
            float(share.loc[exp_hours.index].mean()), 3
        ),
    }


def _band_within_days(y_pred: pd.Series, tz: str) -> pd.Series:
    """
    Label every hour green/amber/red *relative to the other hours of its own day*.

    Banding within the day rather than across the whole window is what makes the
    output actionable: an operator can move a chiller run from 19:00 to 13:00,
    but cannot move June demand into April. Pooling across months would mostly
    rank days against each other and produce a recommendation nobody can execute.
    """
    local_date = pd.Series(y_pred.index.tz_convert(tz).date, index=y_pred.index)

    def label_day(day: pd.Series) -> pd.Series:
        lo, hi = day.quantile([GREEN_QUANTILE, RED_QUANTILE])
        return pd.Series(
            np.where(day <= lo, "green", np.where(day >= hi, "red", "amber")),
            index=day.index,
        )

    return y_pred.groupby(local_date, group_keys=False).apply(label_day)


def _band_window(
    band: str,
    price: pd.Series,
    share: pd.Series,
    tz: str,
    min_frequency: float = 0.5,
) -> ShiftWindow:
    """
    Summarise one band, reporting the clock hours that land in it habitually.

    A clock hour is only listed if it falls in this band on at least
    `min_frequency` of days, so the recommendation reflects a stable daily
    pattern rather than every hour that qualified once.
    """
    all_hours = pd.Series(price.index.tz_convert(tz).hour, index=price.index)
    n_days = price.index.tz_convert(tz).normalize().nunique()
    counts = all_hours.value_counts()
    habitual = sorted(int(h) for h, c in counts.items() if c / max(n_days, 1) >= min_frequency)

    band_share = share.reindex(price.index).dropna()
    mean_share = float(band_share.mean()) if not band_share.empty else float("nan")
    return ShiftWindow(
        band=band,
        hours_local=habitual,
        mean_price_eur_mwh=round(float(price.mean()), 2),
        mean_renewable_share=round(mean_share, 3),
        mean_intensity_t_per_mwh=round(
            float(carbon_intensity(pd.Series([mean_share])).iloc[0]), 3
        ),
        n_hours=int(len(price)),
    )


def build_shift_plan(
    y_pred: pd.Series,
    df_actuals: pd.DataFrame,
    model_mae: float,
    tz: str = "Europe/Berlin",
    residual_intensity: float = RESIDUAL_INTENSITY_T_PER_MWH,
) -> dict[str, Any]:
    """
    Translate an hourly price forecast into a load-shift recommendation.

    Args:
        y_pred:       Hourly forecast price Series (UTC DatetimeIndex).
        df_actuals:   Raw market data supplying load and renewable generation,
                      used to estimate carbon intensity for the same hours.
        model_mae:    Hold-out MAE of the forecast, used to gate confidence.
        tz:           Local timezone for reporting clock hours.
        residual_intensity: CO2 intensity of the non-renewable residual mix.

    Returns a JSON-serialisable plan. If the price/carbon link cannot be
    validated the plan is returned with status "unvalidated" and no
    recommendation — the pipeline should not invent one.
    """
    if y_pred.empty:
        return {"status": "no_data", "reason": "empty forecast series"}

    window = df_actuals.loc[df_actuals.index.isin(y_pred.index)]
    link = validate_price_carbon_link(window)

    share = renewable_share(window)
    bands = _band_within_days(y_pred, tz)

    green = _band_window("green", y_pred[bands == "green"], share, tz)
    red = _band_window("red", y_pred[bands == "red"], share, tz)

    price_saving = round(red.mean_price_eur_mwh - green.mean_price_eur_mwh, 2)
    co2_avoided = round(
        float(residual_intensity * (green.mean_renewable_share - red.mean_renewable_share)),
        3,
    ) if np.isfinite(green.mean_renewable_share) and np.isfinite(red.mean_renewable_share) else None

    # Confidence is gated on both forecast accuracy and the strength of the
    # price/carbon relationship. A precise forecast of a signal that does not
    # track carbon is worthless for this use case.
    spread_vs_error = price_saving / model_mae if model_mae > 0 else 0.0
    r = link.get("pearson_r", 0.0) if link.get("status") == "ok" else 0.0
    if spread_vs_error >= 1.5 and r >= 0.4:
        confidence = "high"
    elif spread_vs_error >= 0.75 and r >= 0.25:
        confidence = "medium"
    else:
        confidence = "low"

    period_label = (
        f"{y_pred.index.min().tz_convert(tz).date()} → "
        f"{y_pred.index.max().tz_convert(tz).date()}"
    )

    plan: dict[str, Any] = {
        "status": "ok" if link.get("status") == "ok" else "unvalidated",
        "period": period_label,
        "price_carbon_link": link,
        "green_window": {
            "local_hours": green.hours_local,
            "n_hours": green.n_hours,
            "mean_price_eur_mwh": green.mean_price_eur_mwh,
            "mean_renewable_share": green.mean_renewable_share,
            "mean_intensity_t_per_mwh": green.mean_intensity_t_per_mwh,
        },
        "red_window": {
            "local_hours": red.hours_local,
            "n_hours": red.n_hours,
            "mean_price_eur_mwh": red.mean_price_eur_mwh,
            "mean_renewable_share": red.mean_renewable_share,
            "mean_intensity_t_per_mwh": red.mean_intensity_t_per_mwh,
        },
        "impact_per_mwh_shifted": {
            "cost_saving_eur": price_saving,
            "co2_avoided_t": co2_avoided,
            "assumed_residual_intensity_t_per_mwh": residual_intensity,
        },
        "forecast_mae_eur_mwh": round(model_mae, 2),
        "confidence": confidence,
        "recommendation": (
            f"Shift flexible load out of local hours {red.hours_local} "
            f"into local hours {green.hours_local}."
        ),
        # Written for the person who has to act on them, not for the model that
        # produced them. Each one names the situation and the action to take.
        "override_conditions": [
            "Wind and solar readings are more than six hours out of date. The estimate of "
            "which hours are cleaner cannot be trusted, so keep the current schedule.",
            f"Forecasts have been off by more than {2 * model_mae:.0f} euros per MWh over the past "
            "month, roughly double the expected error. Stop shifting automatically until it recovers.",
            "Cheap hours have stopped being cleaner hours over the past month. The reasoning behind "
            "the schedule no longer holds, so revert to a fixed one.",
            "Following the schedule would breach an operating limit, such as chilled-water storage "
            "capacity, a minimum desalination output, or a vehicle that must be charged by a set "
            "time. The operating limit always wins.",
            "The grid operator has issued its own demand instruction. That always overrides this "
            "recommendation.",
        ],
    }
    return plan


def scale_impact(
    plan: dict[str, Any],
    shiftable_mwh_per_day: float,
    days_per_year: int = 365,
) -> dict[str, Any]:
    """
    Scale per-MWh impact to an annual figure for a given flexible load.

    `shiftable_mwh_per_day` is the energy an operator can actually move — not
    their total consumption. Keeping the two separate is deliberate: the headline
    number is only as credible as the flexibility assumption behind it.
    """
    impact = plan.get("impact_per_mwh_shifted", {})
    saving = impact.get("cost_saving_eur")
    co2 = impact.get("co2_avoided_t")
    if saving is None:
        return {}

    annual_mwh = shiftable_mwh_per_day * days_per_year
    return {
        "shiftable_mwh_per_day": shiftable_mwh_per_day,
        "annual_shiftable_mwh": round(annual_mwh),
        "annual_cost_saving_eur": round(annual_mwh * saving),
        "annual_co2_avoided_t": round(annual_mwh * co2) if co2 is not None else None,
    }
