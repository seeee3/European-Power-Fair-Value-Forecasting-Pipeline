# European Power Fair Value: Forecasting Day-Ahead and Translating to Prompt Curve Views

**Sneha Sunil · snehasunil385@gmail.com**

---

## 1. Market Selection & Data Sources

**Market:** Germany (DE-LU bidding zone, EPEX Spot)

Germany was chosen as it is the largest and most liquid European power market, with well-documented public data via SMARD (Bundesnetzagentur) — accessible with no API key.

### Data Sources

| Series | Source | Filter / Endpoint | Granularity |
|--------|--------|------------------|-------------|
| Day-Ahead prices | SMARD — Bundesnetzagentur | Filter 4169, region `DE-LU` | 15-min → hourly |
| Wind onshore generation | SMARD | Filter 4066, region `DE-LU` | 15-min → hourly |
| Solar (PV) generation | SMARD | Filter 4067, region `DE-LU` | 15-min → hourly |

**API endpoint structure** (documented; no auth required):
```
Index: https://www.smard.de/app/chart_data/{filter}/DE-LU/index_quarterhour.json
Data:  https://www.smard.de/app/chart_data/{filter}/DE-LU/{filter}_DE-LU_quarterhour_{ts_ms}.json
```
Data is returned as `{"series": [[epoch_ms, value], ...]}` in UTC milliseconds.

Alternative with ENTSO-E API key (see `src/cobblestone/ingestion/entsoe_client.py`):
- DA prices: document type A44, domain `10Y1001A1001A82H`
- Load: A65 (processType A16)
- Wind/Solar: A75 with psrType B19/B18/B16

**Coverage:** 2022-01-01 to 2025-12-31 — 35,064 hourly observations

### Timezone / DST Handling

All timestamps are stored as **UTC** throughout the pipeline. SMARD returns millisecond epoch UTC natively. Calendar features (hour-of-day, peak/off-peak flag) are derived by converting to `Europe/Berlin` (CET/CEST) via `pandas.DatetimeIndex.tz_convert()`, which correctly handles the DST transition (last Sunday of March and October). The German market peak product definition (08:00–20:00 CET/CEST Mon–Fri) uses this local-time conversion.

---

## 2. Data Quality

Standard QA checks are implemented in `src/cobblestone/quality/standard_qa.py` and results are written to `outputs/qa_report.json`.

| Check | Result |
|-------|--------|
| Row count | 35,064 (matches expected hourly rows — **zero gaps**) |
| Duplicate timestamps | 0 |
| Price missing | 0 (0.00%) |
| Wind onshore missing | 0 (0.00%) |
| Solar missing | 0 (0.00%) |
| Price domain violations (< −500 or > 3,000 EUR/MWh) | 0 |
| Price range observed | −500 to +936 EUR/MWh |
| Solar range observed | 47 to 48,682 MWh |
| Wind onshore range observed | 3,096 to 52,848 MWh |

**Note on load data (SMARD filter 4381):** The load series returned 62.6% null values and the non-null values appear to be in GW rather than MWh scale (max 98 vs expected ~80,000 MWh). This filter is unreliable for the requested region/time window; the column is dropped from feature engineering and documented as a known data quality issue. Wind onshore and solar remain as the two fundamental drivers, which are the primary renewable merit-order drivers in the German market.

### AI-Accelerated QA (LLM Rule Generation)

**Implementation:** `src/cobblestone/quality/llm_qa.py`

GPT-4o-mini is called programmatically to propose domain-aware validation rules for the power market dataset. The pipeline:

1. Constructs a schema description (dtype, min, max, mean, std + 5 sample rows)
2. Sends a structured system prompt instructing the LLM to return 8–12 validation rules as JSON, covering physical feasibility, market realism, temporal consistency, and cross-series sanity
3. Parses the JSON response into executable Python expressions
4. Evaluates each rule against the full dataset using `eval()` on a `pd.Series`
5. Logs: timestamp, model, full system prompt, raw response, parsed rules, per-rule results (violations count + fraction), and any errors

All logs are written to `outputs/llm_qa_log.json`. The API key is read from `OPENAI_API_KEY` in `.env` — never committed to source control.

---

## 3. Forecasting Methodology

**Target choice: Option A** — Forecast next-day hourly Day-Ahead prices, then aggregate to weekly/monthly delivery-period averages.

*Justification:* Hourly granularity preserves the peak/off-peak shape information that is directly tradeable (EEX peak and off-peak products). Aggregating hourly forecasts to weekly/monthly base and peak averages is straightforward and gives the desk more levers than a single period-average target.

### Feature Engineering (`src/cobblestone/features/engineer.py`)

**Leakage control:** Only information available before DA gate closure (~12:00 CET, day D) is used to forecast prices for day D+1. Raw fundamental values are lagged ≥ 24 hours.

| Feature group | Features |
|---------------|----------|
| Calendar | Hour of day, day of week, month, week-of-year, is_weekend, is_holiday (DE fixed holidays), sine/cosine encodings for hour/dow/month |
| Price lags | lag_24h, lag_48h, lag_72h, lag_168h, lag_336h |
| Price rolling stats | 24h rolling mean/max/min, 168h rolling mean/std (all shifted 24h) |
| Fundamentals (lagged 24h/168h) | wind_onshore_lag24, wind_onshore_lag168, solar_lag24, solar_lag168, rolling 24h means |
| Derived | renewables_total_lag24 (wind + solar combined) |

**Total features:** 31

### Models

**Baseline: Seasonal Naive**
```
price[t] = price[t − 168h]    (same hour, same weekday, prior week)
```
This is the industry-standard benchmark for hourly power prices due to the strong weekly seasonality.

**Improved model: LightGBM**
- 800 estimators, learning rate 0.05, 63 leaves
- Trained on the full training period (2022-01 to 2025-09)
- No hyperparameter search (out of scope); defaults are well-calibrated for tabular time series

### Walk-Forward Cross-Validation

12 monthly validation folds from **October 2024 to September 2025** (last 12 months of training data). Training window expands chronologically — no data from the validation month is ever used in training.

| Metric | LightGBM CV (mean ± std) | LightGBM test set | Seasonal Naive (test set) |
|--------|--------------------------|------------------|--------------------------|
| MAE (EUR/MWh) | **24.7 ± 7.7** | **28.6** | 49.4 |
| RMSE (EUR/MWh) | **35.9 ± 13.5** | **39.2** | — |
| Pinball loss @ 10th pctile | 12.4 | — | — |
| Pinball loss @ 90th pctile | **12.9** | — | — |
| Improvement over baseline | — | **+42.2%** | baseline |

**Test set:** Oct–Dec 2025 (2,208 hourly observations, held out throughout all model development)

The pinball losses at P10 and P90 measure tail performance — the model captures both downside (negative/zero price events driven by excess renewables) and upside (cold snaps, low wind periods) reasonably well.

### Prediction Intervals

Empirical prediction intervals are derived from CV fold residuals (y_true − y_pred), calibrated per hour-of-day to capture intraday heteroskedasticity (prices are more volatile during peak hours). The 80% interval (P10–P90) achieves **71.4% empirical coverage** on the Oct–Dec 2025 test set (vs. 80% nominal). The under-coverage reflects regime-change risk: the interval is calibrated on 2024–2025 data but Oct–Dec 2025 had some unusually volatile weeks.

---

## 4. Prompt Curve Translation (`src/cobblestone/curve/translation.py`)

### Method

Hourly forecasts are aggregated into standard delivery products:

| Product | Definition (local CET/CEST) |
|---------|----------------------------|
| **Baseload** | Arithmetic mean of all 24 hours |
| **Peak** | Mean of hours 08:00–20:00, Monday–Friday only |
| **Off-peak** | Remaining hours |

Aggregations are computed for **weekly** and **monthly** horizons, producing a forward-looking view consistent with EEX-listed prompt products. Results are in `outputs/delivery_views.csv`.

### Prediction interval on delivery-period averages

The 80% prediction interval for each delivery period is computed as the mean of the per-hour P10/P90 bounds (derived from CV residuals). A narrower band → greater conviction → larger position size.

### Trading Signal Logic

```
if fair_value > prompt_price + model_MAE → LONG prompt month base
if fair_value < prompt_price − model_MAE → SHORT prompt month base
else → NEUTRAL (no directional edge within model error)
```

Peak/off-peak spread > €5/MWh → express via EEX peak vs base spread.

**What the desk does with it:**
1. Express directional bias via prompt month baseload futures on EEX
2. Express shape via peak/off-peak spread products
3. Size position inversely proportional to the P10–P90 band width (uncertainty)

### Invalidation Conditions

The signal should be overridden if any of the following occur:

1. **Unplanned nuclear outage > 1 GW** announced within 4 hours of gate closure → upside surprise to base
2. **Cold snap**: load > +15% vs 7-day average → demand shock not captured in fundamentals lag
3. **Wind drought**: onshore generation < 30% of installed capacity (≈ 18 GW) for ≥ 3 consecutive days
4. **Cross-border congestion**: price zone split between DE and FR/NL > €20/MWh
5. **Model degradation**: out-of-sample MAE > 2× the training-period CV MAE → reduce position to zero

---

## 5. AI-Accelerated Workflow

**Implementation:** `src/cobblestone/quality/llm_qa.py`

The LLM QA component is called **programmatically from code** — not as a manual chat step.

**Productivity gain:** A domain expert would need to manually specify validation rules for each dataset ingestion. The LLM proposes 8–12 rules in seconds, covering physical constraints (e.g., generation cannot be negative), market-specific bounds (EPEX Spot hard limits of −500 to +3,000 EUR/MWh), and cross-series logic (negative prices should co-occur with high renewable output). The pipeline then executes and audits all rules automatically.

**Auditability:**
- System prompt (domain context + JSON schema) is logged
- Raw model response is logged verbatim
- Each rule: name, description, expression, violation count, fraction
- Any API or parse errors are caught and logged with traceback
- See `outputs/llm_qa_log.json` for the full log

**Controls:**
- `temperature=0.2` for reproducible rule generation
- `response_format={"type": "json_object"}` enforces structured output
- Rules are executed via Python `eval()` on an isolated `pd.Series` (no write access to the DataFrame)
- API key via `OPENAI_API_KEY` environment variable, never in source

---

## 6. Repository Structure

```
cobblestone/
├── run_pipeline.py              # Single entry point
├── pyproject.toml / requirements.txt
├── .env.example                 # API key template
├── data/raw/                    # Cached SMARD parquet (auto-generated)
├── outputs/
│   ├── qa_report.json           # QA + model metrics
│   ├── llm_qa_log.json          # Full LLM prompt/response/execution log
│   ├── delivery_views.csv       # Weekly + monthly base/peak/offpeak views
│   ├── trading_signal.json      # Actionable desk signal
│   ├── submission.csv           # Out-of-sample predictions (id, y_pred, p10, p90)
│   └── figures/                 # 5 publication-quality figures
└── src/cobblestone/
    ├── ingestion/               # SMARD + ENTSO-E fetchers
    ├── quality/                 # Standard QA + LLM QA
    ├── features/                # Feature engineering
    ├── models/                  # Baseline + LightGBM
    └── curve/                   # DA→prompt curve translation
```

**Setup and run:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[notebook]"
cp .env.example .env  # add OPENAI_API_KEY
python run_pipeline.py
```

Outputs are regenerated on each run. Data is cached after the first fetch.
