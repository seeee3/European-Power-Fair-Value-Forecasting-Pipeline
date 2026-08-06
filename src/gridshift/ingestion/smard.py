"""
SMARD (Bundesnetzagentur) data fetcher — no API key required.

Endpoint documentation:
  Base:  https://www.smard.de/app/chart_data/{filter}/{region}/
  Index: .../index_quarterhour.json  → list of week-start timestamps (ms UTC)
  Data:  .../{filter}_{region}_quarterhour_{ts}.json  → {"series": [[ts_ms, value], ...]}

Postman collection: https://www.smard.de/home/downloadcenter/download-marktdaten
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)

SMARD_BASE = "https://www.smard.de/app/chart_data"
REGION = "DE-LU"
RESOLUTION = "quarterhour"

# Filter IDs for DE-LU (Germany-Luxembourg bidding zone)
# Source: SMARD Downloadcenter filter reference
FILTERS: dict[str, int] = {
    "price_da_eur_mwh": 4169,   # EPEX Spot Day-Ahead auction
    "wind_onshore_mwh": 4066,   # Realised wind onshore generation
    "wind_offshore_mwh": 4065,  # Realised wind offshore generation
    "solar_mwh": 4067,          # Realised solar PV generation
    "load_mwh": 4381,           # Realised grid load
}

_SESSION = requests.Session()
_SESSION.headers.update({"Accept": "application/json"})


def _get(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            logger.warning("SMARD request failed (attempt %d/%d): %s", attempt + 1, retries, exc)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def _fetch_filter(
    filter_id: int,
    col_name: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.Series:
    """Download one SMARD series for the given date range."""
    index_url = f"{SMARD_BASE}/{filter_id}/{REGION}/index_{RESOLUTION}.json"
    index_data = _get(index_url)
    timestamps: list[int] = index_data.get("timestamps", [])

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    # Each index entry is the week-start; keep chunks overlapping our window
    relevant = [ts for ts in timestamps if ts <= end_ms]
    # Also include the chunk just before start to cover partial first week
    relevant = [ts for ts in relevant if ts >= start_ms - 7 * 24 * 3600 * 1000]

    rows: list[tuple[datetime, float]] = []
    for ts in relevant:
        data_url = f"{SMARD_BASE}/{filter_id}/{REGION}/{filter_id}_{REGION}_{RESOLUTION}_{ts}.json"
        try:
            payload = _get(data_url)
        except RuntimeError:
            logger.error("Skipping chunk ts=%d for filter %d", ts, filter_id)
            continue
        for point_ts, value in payload.get("series", []):
            if start_ms <= point_ts <= end_ms and value is not None:
                dt = datetime.fromtimestamp(point_ts / 1000, tz=timezone.utc)
                rows.append((dt, float(value)))

    if not rows:
        logger.warning("No data returned for filter %d (%s)", filter_id, col_name)
        return pd.Series(dtype=float, name=col_name)

    s = pd.Series(
        data=[v for _, v in rows],
        index=pd.DatetimeIndex([dt for dt, _ in rows], tz="UTC"),
        name=col_name,
    )
    # SMARD is 15-min; resample to hourly mean for prices, sum for energy
    agg = "mean" if "eur" in col_name else "sum"
    return s.resample("h").agg(agg)


def fetch_dataset(start: str, end: str) -> pd.DataFrame:
    """
    Fetch all SMARD series for [start, end] and return a merged hourly DataFrame.

    Args:
        start: ISO date string e.g. '2022-01-01'
        end:   ISO date string e.g. '2025-12-31'

    Returns:
        DataFrame indexed by UTC datetime with columns matching FILTERS keys.
    """
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(hour=23, minute=59, tzinfo=timezone.utc)

    series: dict[str, pd.Series] = {}
    for col_name, filter_id in FILTERS.items():
        logger.info("Fetching SMARD filter %d → %s", filter_id, col_name)
        try:
            s = _fetch_filter(filter_id, col_name, start_dt, end_dt)
            series[col_name] = s
        except Exception as exc:
            logger.error("Failed to fetch %s: %s", col_name, exc)

    if not series:
        raise RuntimeError("No data fetched from SMARD — check network connectivity")

    df = pd.DataFrame(series)
    df.index.name = "utc_timestamp"
    df = df.sort_index()

    # Remove rows entirely outside the requested window
    df = df.loc[start_dt:end_dt]

    logger.info(
        "SMARD dataset: %d rows from %s to %s",
        len(df),
        df.index.min(),
        df.index.max(),
    )
    return df
