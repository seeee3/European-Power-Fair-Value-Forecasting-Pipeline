"""Standard data quality checks for the power market dataset."""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Domain-specific bounds for DE power market
DOMAIN_BOUNDS: dict[str, tuple[float, float]] = {
    "price_da_eur_mwh":   (-500.0, 3000.0),   # EPEX Spot hard limits
    "wind_onshore_mwh":   (0.0, 65_000.0),     # DE onshore capacity ~60 GW
    "wind_offshore_mwh":  (0.0, 10_000.0),     # DE offshore capacity ~9 GW
    "solar_mwh":          (0.0, 80_000.0),     # DE solar capacity ~75 GW
    "load_mwh":           (10_000.0, 90_000.0),# DE hourly load range
}


def _coverage_by_month(df: pd.DataFrame, col: str) -> dict[str, float]:
    """Fraction of non-null values per calendar month."""
    monthly = df[col].resample("ME").agg(lambda s: s.notna().mean())
    return {str(k.date()): round(float(v), 4) for k, v in monthly.items()}


def run_qa(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run standard QA checks and return a structured report dict.

    Checks:
      - Row count and time coverage
      - Missing values (count + fraction per column)
      - Duplicate timestamps
      - Hourly gaps (expected vs actual rows)
      - Domain-bound violations (outliers)
      - Coverage completeness by month
    """
    report: dict[str, Any] = {}

    # ── Overview ────────────────────────────────────────────────────────────
    report["overview"] = {
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "columns": list(df.columns),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "expected_hourly_rows": int(
            (df.index.max() - df.index.min()).total_seconds() / 3600 + 1
        ),
    }

    # ── Missing values ───────────────────────────────────────────────────────
    report["missing"] = {
        col: {
            "count": int(df[col].isna().sum()),
            "fraction": round(float(df[col].isna().mean()), 4),
        }
        for col in df.columns
    }

    # ── Duplicates ───────────────────────────────────────────────────────────
    dup_mask = df.index.duplicated()
    report["duplicates"] = {
        "count": int(dup_mask.sum()),
        "example_timestamps": [str(t) for t in df.index[dup_mask][:5].tolist()],
    }

    # ── Hourly continuity (gaps) ─────────────────────────────────────────────
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    missing_hours = full_range.difference(df.index)
    report["hourly_gaps"] = {
        "count": int(len(missing_hours)),
        "example_timestamps": [str(t) for t in missing_hours[:5].tolist()],
    }

    # ── Outlier / domain-bound checks ────────────────────────────────────────
    outliers: dict[str, Any] = {}
    for col in df.columns:
        if col not in DOMAIN_BOUNDS:
            continue
        lo, hi = DOMAIN_BOUNDS[col]
        series = df[col].dropna()
        below = series[series < lo]
        above = series[series > hi]
        outliers[col] = {
            "bounds": [lo, hi],
            "below_count": int(len(below)),
            "above_count": int(len(above)),
            "min": round(float(series.min()), 2) if len(series) else None,
            "max": round(float(series.max()), 2) if len(series) else None,
            "p01": round(float(series.quantile(0.01)), 2) if len(series) else None,
            "p99": round(float(series.quantile(0.99)), 2) if len(series) else None,
        }
    report["outliers"] = outliers

    # ── Monthly coverage ─────────────────────────────────────────────────────
    report["monthly_coverage"] = {
        col: _coverage_by_month(df, col) for col in df.columns
    }

    # ── Summary flag ─────────────────────────────────────────────────────────
    issues = []
    for col, m in report["missing"].items():
        if m["fraction"] > 0.05:
            issues.append(f"{col}: {m['fraction']:.1%} missing")
    if report["duplicates"]["count"]:
        issues.append(f"{report['duplicates']['count']} duplicate timestamps")
    if report["hourly_gaps"]["count"] > 24:
        issues.append(f"{report['hourly_gaps']['count']} hourly gaps")
    for col, o in outliers.items():
        total_out = o["below_count"] + o["above_count"]
        if total_out:
            issues.append(f"{col}: {total_out} domain-bound violations")
    report["issues"] = issues
    report["passed"] = len(issues) == 0

    logger.info("QA complete — %d issue(s) found", len(issues))
    for issue in issues:
        logger.warning("  QA issue: %s", issue)

    return report


def print_qa_summary(report: dict[str, Any]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    ov = report["overview"]
    console.print(f"\n[bold]Data QA Report[/bold]  {ov['start']} → {ov['end']}")
    console.print(f"  Rows: {ov['n_rows']}  |  Expected: {ov['expected_hourly_rows']}  |  Gaps: {report['hourly_gaps']['count']}")

    t = Table(title="Missing + Outliers by Column", show_lines=True)
    t.add_column("Column")
    t.add_column("Missing %", justify="right")
    t.add_column("Below bound", justify="right")
    t.add_column("Above bound", justify="right")
    t.add_column("Min / Max")

    for col in ov["columns"]:
        m = report["missing"][col]
        o = report["outliers"].get(col, {})
        t.add_row(
            col,
            f"{m['fraction']:.2%}",
            str(o.get("below_count", "—")),
            str(o.get("above_count", "—")),
            f"{o.get('min', '—')} / {o.get('max', '—')}",
        )
    console.print(t)

    status = "[green]PASSED[/green]" if report["passed"] else "[red]ISSUES[/red]"
    console.print(f"\nOverall status: {status}")
    for issue in report["issues"]:
        console.print(f"  ⚠  {issue}")
