# European Power Fair Value: Forecasting Day-Ahead and Translating to Prompt Curve Views

**Sneha Sunil · snehasunil385@gmail.com**

---

## Executive Summary

- **Market:** Germany (DE-LU bidding zone, EPEX Spot) — largest and most liquid European power market, with 4 years of fully public hourly data via SMARD (no API key required)
- **Data:** 35,064 hourly observations (2022–2025), zero missing values across all retained series; dual ingestion source — SMARD primary, ENTSO-E as keyed alternative with automatic annual chunking
- **Forecast (Option A):** Next-day hourly DA prices forecast by LightGBM; aggregated to weekly/monthly base/peak/offpeak delivery views
- **Model result:** LightGBM beats the seasonal naive baseline by **+42.2% MAE** (28.6 vs 49.4 EUR/MWh, Oct–Dec 2025 hold-out); 12-fold walk-forward CV — zero temporal leakage
- **Trading signal:** Oct 2025 directional = **NEUTRAL** (forecast 83.7 vs reference 83.5 EUR/MWh, within ±MAE); shape trade = **buy peak vs base** (spread +20.6 EUR/MWh, above €5 threshold)
- **AI component:** Claude (`claude-haiku-4-5`) called programmatically to propose 8–12 domain-aware QA validation rules; full prompt, raw response, and per-rule execution results logged to `outputs/llm_qa_log.json`

---

## 1. Market Selection & Data Sources

**Market:** Germany (DE-LU) — chosen for liquidity, public data availability, and well-studied renewable merit-order dynamics.

| Series | Source | Filter / Endpoint | Granularity |
|--------|--------|------------------|-------------|
| Day-Ahead prices | SMARD (Bundesnetzagentur) | Filter 4169, DE-LU | 15-min → hourly |
| Wind onshore generation | SMARD | Filter 4066, DE-LU | 15-min → hourly |
| Solar (PV) generation | SMARD | Filter 4067, DE-LU | 15-min → hourly |

ENTSO-E Transparency Platform (`src/cobblestone/ingestion/entsoe_client.py`) is implemented as an optional keyed alternative (DA prices: A44/A01; load: A65; wind/solar: A75 with psrType B19/B18/B16). Multi-year requests are automatically split into annual chunks to respect the API's per-call limit.

**Timezone / DST:** All timestamps stored as UTC. Calendar features and peak/offpeak flags derived via `pandas.DatetimeIndex.tz_convert("Europe/Berlin")`, correctly handling the CET↔CEST transition.

---

## 2. Data Quality

| Check | Result |
|-------|--------|
| Row count | 35,064 (zero hourly gaps) |
| Duplicate timestamps | 0 |
| Price missing | 0% |
| Wind onshore missing | 0% |
| Solar missing | 0% |
| Price range | −500 to +936 EUR/MWh (within EPEX hard limits of ±3,000) |

**Load data (SMARD filter 4381):** 62.6% nulls — dropped from feature engineering; rationale recorded in `outputs/qa_report.json`. ENTSO-E load (A65) is reliable (0.01% missing) and used automatically when `ENTSOE_API_KEY` is set.

**AI-driven QA (`src/cobblestone/quality/llm_qa.py`):** Claude is called via API with a structured system prompt to propose validation rules as JSON (8–12 rules covering physical feasibility, market realism, temporal consistency). The pipeline executes each rule against the dataset and logs: system prompt, raw response, parsed rules, per-rule violation count and fraction, and any parse/API errors. Controls: `temperature=0.2`; JSON-only output with markdown fence stripping; expressions evaluated on an isolated `pd.Series`; API key via environment variable only.

---

## 3. Forecasting Methodology

**Target (Option A):** Next-day hourly DA prices, aggregated to weekly/monthly delivery averages. Hourly granularity preserves the peak/offpeak shape that is directly tradeable as EEX products.

**Leakage control:** All fundamentals lagged ≥24h. Rolling price statistics are computed over a window shifted 24h before applying the rolling function.

**Features (31 total):** Calendar (hour, dow, month, week-of-year, is\_weekend, is\_holiday, sine/cosine encodings); price lags at 24h/48h/72h/168h/336h; 24h and 168h rolling mean/std/max/min (shifted); wind and solar lags at 24h/168h; total renewables lag.

**Baseline:** Seasonal Naive — `price[t] = price[t − 168h]` (same hour, prior week). Standard industry benchmark for hourly power prices.

**Model:** LightGBM regressor (800 estimators, lr=0.05, 63 leaves, subsample=0.8). Walk-forward CV with 12 monthly folds (expanding training window, Oct 2024 – Sep 2025). Hold-out test set: Oct–Dec 2025 (2,208 hours, never seen during model development).

| Metric | LightGBM CV (mean ± std) | LightGBM Test | Seasonal Naive |
|--------|--------------------------|---------------|----------------|
| MAE (EUR/MWh) | 24.7 ± 7.7 | **28.6** | 49.4 |
| RMSE (EUR/MWh) | 35.9 ± 13.5 | **39.2** | — |
| Pinball @ P10 | 12.4 | — | — |
| Pinball @ P90 | 12.9 | — | — |
| Improvement vs baseline | — | **+42.2%** | baseline |

**Prediction intervals:** Empirical 80% interval derived from CV fold residuals, calibrated per hour-of-day to capture intraday heteroskedasticity. Achieves 71.4% empirical coverage on the test set (nominal 80%); under-coverage reflects regime-change risk in a particularly volatile Q4 2025.

---

## 4. Prompt Curve Translation (`src/cobblestone/curve/translation.py`)

Hourly forecasts are aggregated into standard EEX delivery products for weekly and monthly horizons:

| Product | Definition (local CET/CEST) |
|---------|-----------------------------|
| Baseload | Arithmetic mean of all 24 hours |
| Peak | Hours 08:00–20:00, Monday–Friday |
| Off-peak | Remaining hours |

**Trading signal logic:**
```
fair_value > prompt + MAE  →  LONG prompt month base
fair_value < prompt − MAE  →  SHORT prompt month base
else                        →  NEUTRAL
```

The reference prompt price is derived from the 30-day trailing realized average immediately before the forecast window (proxy for live EEX screen price). The Oct 2025 forecast base is **83.7 EUR/MWh** vs reference **83.5 EUR/MWh** → **NEUTRAL** directional. Peak/base spread **+20.6 EUR/MWh** exceeds the €5 threshold → **buy peak vs base** via EEX shape products.

**Invalidation conditions:**
1. Unplanned nuclear outage >1 GW announced within 4h of gate closure
2. Cold snap: load >+15% vs 7-day average
3. Wind drought: generation <30% of installed capacity for ≥3 consecutive days
4. Cross-border congestion: DE/FR or DE/NL split >€20/MWh
5. Out-of-sample MAE >2× training MAE → reduce position to zero

---

## 5. Repository Structure & Setup

```
cobblestone/
├── run_pipeline.py              # Single entry point
├── app.py                       # Streamlit dashboard (6 tabs)
├── src/cobblestone/
│   ├── ingestion/               # SMARD + ENTSO-E fetchers
│   ├── quality/                 # Standard QA + LLM QA
│   ├── features/                # Feature engineering
│   ├── models/                  # Seasonal naive + LightGBM + walk-forward CV
│   └── curve/                   # Hourly → delivery-period translation + signal
└── outputs/
    ├── qa_report.json  llm_qa_log.json  trading_signal.json
    ├── delivery_views.csv  submission.csv
    └── figures/  (5 publication-quality figures)
```

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[notebook]"
cp .env.example .env          # add ANTHROPIC_API_KEY (required for LLM QA)
python run_pipeline.py        # generates all outputs; data cached after first fetch
streamlit run app.py          # interactive dashboard
```

`submission.csv` contains out-of-sample predictions for Oct–Dec 2025 with columns `id` (UTC timestamp), `y_pred`, `y_pred_p10`, `y_pred_p90`.
