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

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ENTSOE_BASE = "https://web-api.tp.entsoe.eu/api"

# EIC codes — see https://www.entsoe.eu/data/energy-identification-codes-eic/
EIC = {
    "DE_LU": "10Y1001A1001A82H",   # Germany-Luxembourg bidding zone
    "DE":    "10YDE-VE-------2",    # Germany (DE control area)
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
        """
        Parse ENTSO-E XML envelope into a pandas Series.

        The namespace is extracted dynamically from the root element so this
        works for all ENTSO-E document types (A44 uses
        publicationdocument:7:3, A65/A75 use generationdocument:5:0, etc.).
        """
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml_bytes)

        # Detect namespace from root tag: {urn:...}ElementName
        tag = root.tag
        if tag.startswith("{"):
            ns_uri = tag[1:tag.index("}")]
            ns = {"ns": ns_uri}
        else:
            ns = {}

        # Log an API-level error message if the response is a rejection
        reason = root.findtext(".//ns:Reason/ns:text", namespaces=ns) or \
                 root.findtext(".//Reason/text")
        if reason:
            logger.warning("ENTSO-E API returned a reason message for %s: %s", col, reason)

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
            logger.warning("No data rows parsed from ENTSO-E response for %s", col)
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
            "contract_MarketAgreement.type": "A01",
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


def _month_chunks(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """
    Split [start, end] into calendar-month chunks.

    ENTSO-E caps A65 (load) and A75 (generation per type) at one month per
    request and answers anything longer with HTTP 400:
    "Provided time interval is larger than maximum allowed period 'P1M'".
    An earlier version chunked by year, so every load and generation request
    failed, the failure was caught and turned into an empty Series, and the
    dataset came back with price only. Prices (A44) do allow a year, but there
    is no reason to keep two chunking schemes.

    `periodEnd` is exclusive, so each chunk ends *at* the next month boundary
    rather than an hour before it. Ending an hour early instead drops the last
    hour of every month: 48 hours across four years, small enough to pass a
    missingness check and still leave a gap in the series.
    """
    last = end + pd.Timedelta(hours=1)
    bounds = pd.date_range(start, last, freq="MS", tz=start.tzinfo)
    edges = [start, *[b.to_pydatetime() for b in bounds if start < b.to_pydatetime() < last], last]
    return [(a, b) for a, b in zip(edges, edges[1:]) if a < b]


def _fetch_chunked(
    fn,
    start: datetime,
    end: datetime,
    col: str,
) -> pd.Series:
    """Call `fn(chunk_start, chunk_end)` for each monthly chunk and concatenate."""
    parts: list[pd.Series] = []
    for chunk_start, chunk_end in _month_chunks(start, end):
        logger.info("  chunk %s → %s", chunk_start.date(), chunk_end.date())
        try:
            part = fn(chunk_start, chunk_end)
            if len(part):
                parts.append(part)
        except Exception as exc:
            logger.error("  chunk failed (%s → %s): %s", chunk_start.date(), chunk_end.date(), exc)
    if not parts:
        return pd.Series(dtype=float, name=col)
    combined = pd.concat(parts).sort_index()
    return combined[~combined.index.duplicated(keep="first")]


def fetch_dataset(api_key: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch all required ENTSO-E series and return a merged hourly DataFrame.
    Requests are split into calendar-month chunks — the API rejects ranges > 1 month
    for load and generation documents.
    """
    client = EntsoEClient(api_key)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(hour=23, tzinfo=timezone.utc)

    series: dict[str, pd.Series] = {}

    logger.info("Fetching ENTSO-E day-ahead prices (DE-LU) in monthly chunks")
    series["price_da_eur_mwh"] = _fetch_chunked(
        client.day_ahead_prices, start_dt, end_dt, "price_da_eur_mwh"
    )

    logger.info("Fetching ENTSO-E actual load in monthly chunks")
    series["load_mwh"] = _fetch_chunked(
        client.actual_load, start_dt, end_dt, "load_mwh"
    )

    logger.info("Fetching ENTSO-E wind onshore generation in monthly chunks")
    series["wind_onshore_mwh"] = _fetch_chunked(
        lambda s, e: client.actual_generation_by_type(s, e, "B19", "wind_onshore_mwh"),
        start_dt, end_dt, "wind_onshore_mwh",
    )

    logger.info("Fetching ENTSO-E wind offshore generation in monthly chunks")
    series["wind_offshore_mwh"] = _fetch_chunked(
        lambda s, e: client.actual_generation_by_type(s, e, "B18", "wind_offshore_mwh"),
        start_dt, end_dt, "wind_offshore_mwh",
    )

    logger.info("Fetching ENTSO-E solar generation in monthly chunks")
    series["solar_mwh"] = _fetch_chunked(
        lambda s, e: client.actual_generation_by_type(s, e, "B16", "solar_mwh"),
        start_dt, end_dt, "solar_mwh",
    )

    df = pd.DataFrame(series)
    df.index.name = "utc_timestamp"
    return df.sort_index().loc[start_dt:end_dt]
