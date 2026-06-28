"""Feature engineering for the power price forecasting model."""
from __future__ import annotations

import numpy as np
import pandas as pd


# German public holidays (fixed dates only; approximate Easter-based omitted)
_DE_FIXED_HOLIDAYS = {
    (1, 1),   # New Year
    (5, 1),   # Labour Day
    (10, 3),  # German Unity Day
    (12, 25), # Christmas 1
    (12, 26), # Christmas 2
}


def _is_de_holiday(dt: pd.DatetimeIndex) -> np.ndarray:
    md = {(m, d) for m, d in zip(dt.month, dt.day)}
    return np.array([(m, d) in _DE_FIXED_HOLIDAYS for m, d in zip(dt.month, dt.day)])


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    # CET/CEST local time for calendar features
    local = idx.tz_convert("Europe/Berlin")
    df = df.copy()
    df["hour"] = local.hour
    df["dow"] = local.dayofweek          # 0=Mon … 6=Sun
    df["month"] = local.month
    df["week_of_year"] = local.isocalendar().week.astype(int)
    df["is_weekend"] = (local.dayofweek >= 5).astype(int)
    df["is_holiday"] = _is_de_holiday(local).astype(int)
    # Sine/cosine encoding for circular features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_features(df: pd.DataFrame, price_col: str = "price_da_eur_mwh") -> pd.DataFrame:
    df = df.copy()
    lags_hours = [24, 48, 72, 168, 336]   # 1d, 2d, 3d, 1w, 2w
    for lag in lags_hours:
        df[f"price_lag_{lag}h"] = df[price_col].shift(lag)
    # Rolling statistics over the most recent known window (lag 24+ to avoid leakage)
    df["price_roll_24h_mean"] = df[price_col].shift(24).rolling(24).mean()
    df["price_roll_168h_mean"] = df[price_col].shift(24).rolling(168).mean()
    df["price_roll_168h_std"] = df[price_col].shift(24).rolling(168).std()
    df["price_roll_24h_max"] = df[price_col].shift(24).rolling(24).max()
    df["price_roll_24h_min"] = df[price_col].shift(24).rolling(24).min()
    return df


def add_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add wind/solar/load features with 24 h lag (only yesterday's actuals are known DA)."""
    df = df.copy()
    fund_cols = [c for c in ["wind_onshore_mwh", "wind_offshore_mwh", "solar_mwh", "load_mwh"]
                 if c in df.columns]
    for col in fund_cols:
        df[f"{col}_lag24"] = df[col].shift(24)
        df[f"{col}_lag168"] = df[col].shift(168)
        df[f"{col}_roll24_mean"] = df[col].shift(24).rolling(24).mean()

    # Derived: total renewables
    wind_cols = [c for c in ["wind_onshore_mwh", "wind_offshore_mwh"] if c in df.columns]
    if wind_cols:
        df["wind_total_mwh"] = df[wind_cols].sum(axis=1)
        df["wind_total_lag24"] = df["wind_total_mwh"].shift(24)
        df["wind_total_lag168"] = df["wind_total_mwh"].shift(168)

    if "solar_mwh" in df.columns and "wind_total_mwh" in df.columns:
        df["renewables_total_lag24"] = (df["solar_mwh"] + df["wind_total_mwh"]).shift(24)

    return df


def build_feature_matrix(df: pd.DataFrame, target_col: str = "price_da_eur_mwh") -> tuple[pd.DataFrame, pd.Series]:
    """
    Build (X, y) for modelling.

    y  = price_da_eur_mwh (current hour — the forecast target)
    X  = all engineered features with strict leakage control:
         only information available before gate closure (~day-ahead auction)
         is included. Raw fundamentals are lagged ≥24 h.

    Returns X and y aligned, with NaN rows dropped.
    """
    df = df.copy()
    df = add_calendar_features(df)
    df = add_lag_features(df, target_col)
    df = add_fundamental_features(df)

    # Remove raw fundamental columns (only their lagged versions are used)
    raw_fund_cols = [c for c in ["wind_onshore_mwh", "wind_offshore_mwh", "solar_mwh", "load_mwh",
                                  "wind_total_mwh"] if c in df.columns]
    feature_cols = [c for c in df.columns if c != target_col and c not in raw_fund_cols]

    X = df[feature_cols]
    y = df[target_col]

    # Drop rows where target or any feature is NaN
    valid = y.notna() & X.notna().all(axis=1)
    return X[valid], y[valid]


FEATURE_DESCRIPTIONS: dict[str, str] = {
    "hour": "Hour of day (0–23, CET/CEST)",
    "dow": "Day of week (0=Mon, 6=Sun)",
    "month": "Calendar month",
    "is_weekend": "Weekend flag",
    "is_holiday": "German public holiday flag",
    "price_lag_24h": "Day-ahead price 24 h prior",
    "price_lag_168h": "Day-ahead price 1 week prior (same hour)",
    "price_roll_168h_mean": "7-day rolling average price",
    "wind_total_lag24": "Total wind generation 24 h prior",
    "solar_mwh_lag24": "Solar generation 24 h prior",
    "load_mwh_lag24": "Grid load 24 h prior",
    "renewables_total_lag24": "Total renewables 24 h prior",
}
