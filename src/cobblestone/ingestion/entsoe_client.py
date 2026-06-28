"""
ENTSO-E Transparency Platform fetcher (optional, requires API key).

Register at: https://transparency.entsoe.eu/usrm/user/createPublicUser
API documentation: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html

Postman collection documenting all endpoints used is in /docs/entsoe_postman.json

Areas / EIC codes used:
  DE-LU bidding zone: 10Y1001A1001A82H
  DE control area:    10YDE-VE-------2
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ENTSOE_BASE = "https://web-api.tp.entsoe.eu/api"

# EIC codes — see https://www.entsoe.eu/data/energy-identification-codes-eic/
EIC = {
    "DE_LU": "10Y1001A1001A82H",   # Germany-Luxembourg bidding zone
    "DE":    "10YDE-VE-------2",    # Germany (DE control area)
}

# Document types for the ENTSO-E REST API
DOC_TYPES = {
    "day_ahead_prices":         "A44",
    "actual_load":              "A65",
    "actual_generation_wind":   "A75",
    "actual_generation_solar":  "A75",
}

# Process types
PROCESS = {
    "realised": "A16",
    "day_ahead": "A01",
}


class EntsoEClient:
    """Thin wrapper around the ENTSO-E Transparency REST API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._session = requests.Session()
        self._session.params = {"securityToken": api_key}  # type: ignore[assignment]

    def _get(self, params: dict) -> bytes:
        r = self._session.get(ENTSOE_BASE, params=params, timeout=60)
        r.raise_for_status()
        return r.content

    def _parse_xml_timeseries(self, xml_bytes: bytes, col: str) -> pd.Series:
        """Parse ENTSO-E XML envelope into a pandas Series."""
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            raise RuntimeError("xml.etree.ElementTree is required")

        ns = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}
        root = ET.fromstring(xml_bytes)
        rows = []
        for ts_elem in root.findall(".//ns:TimeSeries", ns):
            period = ts_elem.find("ns:Period", ns)
            if period is None:
                continue
            start_str = period.findtext("ns:timeInterval/ns:start", namespaces=ns)
            resolution_str = period.findtext("ns:resolution", namespaces=ns)
            if not start_str or not resolution_str:
                continue

            start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            # Map PT60M / PT15M to pandas offsets
            offset_map = {"PT60M": "h", "PT30M": "30min", "PT15M": "15min"}
            freq = offset_map.get(resolution_str, "h")

            for pt in period.findall("ns:Point", ns):
                pos = int(pt.findtext("ns:position", namespaces=ns) or 1)
                val = pt.findtext("ns:price.amount", namespaces=ns) or \
                      pt.findtext("ns:quantity", namespaces=ns)
                if val is None:
                    continue
                idx = start_dt + pd.tseries.frequencies.to_offset(freq) * (pos - 1)
                rows.append((idx, float(val)))

        if not rows:
            return pd.Series(dtype=float, name=col)

        s = pd.Series(
            data=[v for _, v in rows],
            index=pd.DatetimeIndex([d for d, _ in rows], tz="UTC"),
            name=col,
        )
        return s.resample("h").mean()

    def day_ahead_prices(self, start: datetime, end: datetime) -> pd.Series:
        params = {
            "documentType": "A44",
            "in_Domain": EIC["DE_LU"],
            "out_Domain": EIC["DE_LU"],
            "periodStart": start.strftime("%Y%m%d%H%M"),
            "periodEnd": end.strftime("%Y%m%d%H%M"),
        }
        xml = self._get(params)
        return self._parse_xml_timeseries(xml, "price_da_eur_mwh")

    def actual_load(self, start: datetime, end: datetime) -> pd.Series:
        params = {
            "documentType": "A65",
            "processType": "A16",
            "outBiddingZone_Domain": EIC["DE_LU"],
            "periodStart": start.strftime("%Y%m%d%H%M"),
            "periodEnd": end.strftime("%Y%m%d%H%M"),
        }
        xml = self._get(params)
        return self._parse_xml_timeseries(xml, "load_mwh")

    def actual_generation_by_type(
        self, start: datetime, end: datetime, psr_type: str, col: str
    ) -> pd.Series:
        """
        psr_type values (selected):
          B16 = Solar, B19 = Wind Onshore, B18 = Wind Offshore
        """
        params = {
            "documentType": "A75",
            "processType": "A16",
            "in_Domain": EIC["DE_LU"],
            "psrType": psr_type,
            "periodStart": start.strftime("%Y%m%d%H%M"),
            "periodEnd": end.strftime("%Y%m%d%H%M"),
        }
        xml = self._get(params)
        return self._parse_xml_timeseries(xml, col)


def fetch_dataset(api_key: str, start: str, end: str) -> pd.DataFrame:
    """Fetch all required ENTSO-E series and return a merged hourly DataFrame."""
    client = EntsoEClient(api_key)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(hour=23, tzinfo=timezone.utc)

    series: dict[str, pd.Series] = {}

    logger.info("Fetching ENTSO-E day-ahead prices (DE-LU)")
    series["price_da_eur_mwh"] = client.day_ahead_prices(start_dt, end_dt)

    logger.info("Fetching ENTSO-E actual load")
    series["load_mwh"] = client.actual_load(start_dt, end_dt)

    logger.info("Fetching ENTSO-E wind onshore generation")
    series["wind_onshore_mwh"] = client.actual_generation_by_type(
        start_dt, end_dt, "B19", "wind_onshore_mwh"
    )

    logger.info("Fetching ENTSO-E wind offshore generation")
    series["wind_offshore_mwh"] = client.actual_generation_by_type(
        start_dt, end_dt, "B18", "wind_offshore_mwh"
    )

    logger.info("Fetching ENTSO-E solar generation")
    series["solar_mwh"] = client.actual_generation_by_type(
        start_dt, end_dt, "B16", "solar_mwh"
    )

    df = pd.DataFrame(series)
    df.index.name = "utc_timestamp"
    return df.sort_index().loc[start_dt:end_dt]
