# European Power Fair-Value Forecasting Pipeline

**Sneha Sunil · snehasunil385@gmail.com**

End-to-end prototype for the Cobblestone Energy Graduate ADE case study:
**European Power Fair Value — Forecasting Day-Ahead and Translating to Prompt Curve Views**

---

## What it does

| Step | Module | Description |
|------|--------|-------------|
| 1 | `ingestion/` | Hourly DE DA prices + wind/solar/load from SMARD (no key) or ENTSO-E |
| 2a | `quality/standard_qa.py` | Missingness, duplicates, hourly gaps, domain-bound outliers |
| 2b | `quality/llm_qa.py` | **AI component**: Claude proposes & executes domain-specific QA rules |
| 3 | `features/engineer.py` | Calendar, lag, rolling, and fundamental features (strict leakage control) |
| 4 | `models/` | Seasonal naive baseline + LightGBM with walk-forward CV |
| 5 | `curve/translation.py` | Hourly forecasts → Base/Peak blocks → prompt curve trading signal |

---

## Setup

```bash
# 1. Clone and create environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -e ".[notebook]"

# 3. Configure API keys
cp .env.example .env
# Edit .env:  add ANTHROPIC_API_KEY (required for LLM QA)
#             add ENTSOE_API_KEY (optional; falls back to SMARD)
#             set FORCE_REFETCH=true to bypass cached data and re-fetch from source
```

---

## Running the pipeline

```bash
# Full pipeline (ingest → QA → features → model → curve)
python run_pipeline.py

# First run fetches ~4 years of hourly data from SMARD (~30s–2min depending on network).
# Subsequent runs load from data/raw/de_power_market.parquet (instant).
```

### Interactive Dashboard

After running the pipeline, launch the Streamlit dashboard to explore all outputs interactively:

```bash
streamlit run app.py
```

The dashboard has six tabs:

| Tab | What it shows |
|-----|---------------|
| **Overview** | Pipeline summary, dataset metadata, data source table |
| **Data Quality** | Standard QA report (missing %, outliers, gaps) + LLM-generated rule results |
| **Model** | Walk-forward CV metrics table, hold-out test performance, CV and feature-importance figures |
| **Trading Signal** | Directional signal, confidence band, peak/base spread, desk actions and invalidation conditions |
| **Forecast** | Interactive daily chart of Oct–Dec 2025 predictions with 80% prediction interval; download button for `submission.csv` |
| **Figures** | All five publication figures rendered side-by-side |

The sidebar shows a live summary: dataset date range, LightGBM test MAE, and the current directional signal (long / short / neutral).

---

### Outputs

```
outputs/
├── qa_report.json          # Standard QA + model performance metrics
├── llm_qa_log.json         # Full LLM prompt, response, rule execution log
├── trading_signal.json     # DA→curve view with invalidation conditions
├── delivery_views.csv      # Weekly Base/Peak/Offpeak forecast bands
├── submission.csv          # Out-of-sample hourly predictions (id, y_pred)
└── figures/
    ├── fig1_price_renewables.png   # DA price + wind/solar time series
    ├── fig2_cv_performance.png     # Walk-forward CV actual vs forecast
    ├── fig3_feature_importance.png # LightGBM feature importances
    ├── fig4_price_heatmap.png      # Price by hour × month heatmap
    └── fig5_model_comparison.png   # Baseline vs LightGBM comparison
```

---

## Data Sources

### Primary (no key): SMARD – Bundesnetzagentur
- URL: `https://www.smard.de/app/chart_data/{filter}/{region}/`
- Granularity: 15-min, resampled to hourly
- Series fetched:

| Column | Filter ID | Description |
|--------|-----------|-------------|
| `price_da_eur_mwh` | 4169 | EPEX Spot Day-Ahead auction (€/MWh) |
| `wind_onshore_mwh` | 4066 | Realised wind onshore generation |
| `wind_offshore_mwh` | 4065 | Realised wind offshore generation |
| `solar_mwh` | 4067 | Realised solar PV generation |
| `load_mwh` | 4381 | Realised grid load (total consumption) |

### Alternative (with key): ENTSO-E Transparency Platform
Register at: https://transparency.entsoe.eu/usrm/user/createPublicUser

| Series | Document type | Domain |
|--------|---------------|--------|
| DA prices | A44 + `contract_MarketAgreement.type=A01` | DE-LU (10Y1001A1001A82H) |
| Actual load | A65 | DE-LU |
| Wind onshore | A75 / B19 | DE-LU |
| Wind offshore | A75 / B18 | DE-LU |
| Solar | A75 / B16 | DE-LU |

**Note:** The ENTSO-E API limits each request to **one calendar year**. The client (`entsoe_client.py`) automatically splits multi-year date ranges into annual chunks. To switch from SMARD to ENTSO-E on an existing installation, set `FORCE_REFETCH=true` in `.env` for the first run, then revert to `false`.

Full Postman documentation: see `docs/entsoe_postman.json`

---

## Timezone / DST Handling

All timestamps are stored in **UTC** throughout the pipeline. The SMARD API
returns millisecond epoch UTC timestamps. Calendar features (hour of day,
peak/off-peak flag) are derived from `Europe/Berlin` local time using
`pandas.DatetimeIndex.tz_convert`, which correctly handles the CET↔CEST
transition (last Sunday of March / October).

---

## Forecasting Methodology

### Target
Next-day hourly DA price (EUR/MWh) — Option A from the case study.

### Leakage control
Raw fundamental columns (wind, solar, load) are always lagged ≥ 24 h in the
feature matrix. The only information in X for hour H is what would have been
known **before gate closure** of the DA auction for day D+1.

### Validation
Walk-forward (blocked) CV — 12 monthly folds, training window expands
chronologically. No data from the validation month touches training.

### Metrics
- MAE (EUR/MWh) — primary level metric  
- RMSE — penalises large errors (relevant for extreme price events)  
- Pinball @ 10th + 90th percentile — tail coverage quality

### Models
| Model | Description |
|-------|-------------|
| Seasonal Naive | `price[t] = price[t - 168h]` — same hour last week |
| LightGBM | Gradient-boosted trees; calendar + lag + fundamental features |

---

## AI Component (LLM-Accelerated QA)

`src/cobblestone/quality/llm_qa.py`

Claude (`claude-haiku-4-5`) is called to propose validation rules
for the power market dataset:

1. The pipeline constructs a schema description (column stats + 5 sample rows)
2. A system prompt explains the DE power market context and asks for
   domain-specific JSON rules (physical limits, market realism, temporal jumps)
3. Claude returns 8–12 rules as structured JSON
4. The pipeline executes each rule against the full dataset using `eval()`
5. All prompts, raw responses, parsed rules, execution results, and
   any failures are written to `outputs/llm_qa_log.json`

API key is read from the `ANTHROPIC_API_KEY`
environment variable (`.env` file is in `.gitignore`).

---

## Prompt Curve Translation

`src/cobblestone/curve/translation.py`

Hourly DA forecasts are converted to:

| Product | Definition |
|---------|------------|
| Base | Arithmetic mean of all 24 h |
| Peak | Mean of hours 08–19 (local CET/CEST), Mon–Fri |
| Off-peak | Remaining hours |

The trading signal compares the base fair value against a hypothetical
prompt month price. Position sizing is inversely proportional to the
`p90–p10` forecast band width.

**Invalidation conditions** (any of these should override the signal):
- Unplanned nuclear outage > 1 GW within 4 h of gate closure
- Cold snap causing load > +15% vs forecast
- Wind drought (< 30% installed capacity for ≥ 3 consecutive days)
- Cross-border congestion causing zone split > €20/MWh from FR/NL
- Model MAE deteriorating > 2× the training-period MAE

---

## Project Structure

```
cobblestone/
├── run_pipeline.py                # Entry point
├── pyproject.toml
├── .env.example
├── data/
│   ├── raw/                       # Cached SMARD/ENTSO-E parquet
│   └── processed/                 # Feature matrix parquet
├── outputs/
│   ├── figures/                   # 5 publication-ready figures
│   ├── qa_report.json
│   ├── llm_qa_log.json
│   ├── trading_signal.json
│   ├── delivery_views.csv
│   └── submission.csv
└── src/cobblestone/
    ├── config.py
    ├── pipeline.py                # Orchestrator
    ├── plots.py
    ├── ingestion/
    │   ├── smard.py               # No-auth SMARD fetcher
    │   └── entsoe_client.py       # ENTSO-E fetcher (needs key)
    ├── quality/
    │   ├── standard_qa.py         # Rule-based QA checks
    │   └── llm_qa.py              # AI-powered QA (GPT-4o-mini)
    ├── features/
    │   └── engineer.py            # Feature engineering
    ├── models/
    │   ├── baseline.py            # Seasonal naive
    │   └── forecaster.py          # LightGBM + walk-forward CV
    └── curve/
        └── translation.py         # DA → prompt curve signal
```
