# GridShift — Technical Report

**Scheduling flexible electricity demand from day-ahead price forecasts**
**Sneha Sunil · snehasunil385@gmail.com**

---

## Executive summary

- **What it does:** forecasts next-day hourly electricity prices and bands each day's 24 hours into green/amber/red, so operators of flexible load know which hours to run in.
- **Model:** LightGBM on 37 features. Hold-out **MAE 28.76 EUR/MWh** vs seasonal-naive **49.42** — **41.8% better**. 12-fold walk-forward CV (MAE 23.9 ± 7.3), no temporal leakage.
- **Actionable impact:** within-day shifting saves **~81 EUR/MWh** and avoids **~0.034 tCO₂/MWh** across 2022–2025.
- **The main finding is a negative one.** The original thesis — that price is a strong proxy for carbon intensity — does not hold on complete data. See §4.
- **AI component:** Claude (`claude-haiku-4-5`) generates executable data-quality rules; the pipeline runs and audits them behind a guard that quarantines malformed rules.

---

## 1. Data

**Source:** SMARD (Bundesnetzagentur), no API key. ENTSO-E is implemented as a keyed alternative (`ingestion/entsoe_client.py`) with automatic annual chunking.

| Series | Filter |
|---|---|
| Day-ahead price | 4169 |
| Load | 4381 |
| Wind onshore / offshore | 4066 / 4065 |
| Solar PV | 4067 |

**Timezone:** timestamps stored UTC; calendar features and reported clock hours derived via `tz_convert("Europe/Berlin")`, handling CET↔CEST correctly.

**Coverage:** 35,064 hourly rows, 2022-01-01 → 2025-12-31. After feature engineering (lags require warm-up), **34,728 usable rows**, 2022-01-15 → 2025-12-31.

### 1.1 The source is not reproducible run-to-run

This is the most operationally important finding in the project. Repeated SMARD fetches of the *same* date range returned materially different data:

| Fetch | price | load | wind onshore | wind offshore | solar |
|---|---|---|---|---|---|
| A | complete | complete | ends 2024-12-31 (25% missing) | complete | ends 2025-07-09 |
| B | complete | **62.6% missing** | complete | **HTTP 404, absent** | complete |

Neither fetch alone supports the analysis. The canonical dataset was assembled by taking the most complete version of each series across fetches, giving <0.05% missingness on all five. `FORCE_REFETCH` is now set to `false` so the pinned cache is used and results are stable.

**This mattered more than a data-plumbing footnote.** The incomplete fetch is what produced the erroneous r = 0.86 in §4. A production system needs pinned, versioned datasets and explicit coverage assertions *before* modelling, not quality checks after it.

---

## 2. Data quality

Standard checks (`quality/standard_qa.py`): missingness, duplicates, hourly gaps, domain-bound outliers, monthly coverage.

**LLM-driven QA (`quality/llm_qa.py`):** Claude receives a schema summary and five sample rows and returns 8–12 validation rules as JSON, each an executable Python expression over a single column. The pipeline runs every rule and logs the system prompt, raw response, parsed rules and per-rule violation counts to `outputs/llm_qa_log.json`. Recent runs yield 10–11 rules with all but one passing; the exact set varies between runs, which is itself an argument for executing and auditing rather than trusting any single generation.

### 2.1 The rules were wrong, and the pipeline believed them

An early run produced rules of the form:

```python
s.diff().abs() <= 8000 | s.diff().isna()
```

Python binds `|` more tightly than `<=`, so this parses as `s.diff().abs() <= (8000 | s.diff().isna())`. The right-hand side collapses to `True` (i.e. 1), so almost every row fails. The pipeline reported **720 violations out of 720 rows** when the true count was **6**.

Two fixes, both retained:

1. The system prompt names the precedence trap explicitly and requires every comparison to be fully parenthesised.
2. `_execute_rule` marks any rule flagging more than `SUSPECT_VIOLATION_FRACTION` (50%) of rows as **`suspect`** — a malformed rule — rather than reporting it as a data finding.

The second fix is the load-bearing one. The prompt fix depends on the model complying; the guard does not.

---

## 3. Forecasting

**Target:** next-day hourly day-ahead price.

**Leakage control:** fundamentals lagged ≥24h; rolling price statistics computed on an already-shifted series. Features for hour *H* contain only what was knowable before gate closure.

**Features (37):** calendar (hour, day-of-week, month, week-of-year, weekend, holiday, sine/cosine encodings); price lags at 24/48/72/168/336h; 24h and 168h rolling mean/std/max/min; wind, solar and load lags at 24h/168h plus 24h rolling means; total renewables lag.

**Baseline:** seasonal naive, `price[t] = price[t−168h]`.

**Model:** LightGBM (800 estimators, lr 0.05, 63 leaves, subsample 0.8); 12 monthly walk-forward folds, expanding training window.

| Metric | CV (mean ± std) | Hold-out | Seasonal naive |
|---|---|---|---|
| MAE (EUR/MWh) | 23.9 ± 7.3 | **28.76** | 49.42 |
| RMSE (EUR/MWh) | 35.3 ± 13.0 | **39.50** | — |
| Pinball @ P90 | 12.0 | — | — |
| Improvement | — | **+41.8%** | — |

**Hold-out:** 2025-10-01 → 2025-12-31, 2,208 hours, never seen during development.

**Prediction intervals:** empirical 80% interval from CV fold residuals, calibrated per hour-of-day. Empirical coverage **70.2%** against a nominal 80% — under-covered, reflecting a volatile Q4. Reported rather than tuned away; the confidence gate accounts for it.

---

## 4. The premise test, and why the headline changed

### 4.1 The original thesis

In a merit-order market the clearing price is set by the marginal generator, so abundant wind and solar should push both price and carbon intensity down together. If that holds strongly, a price forecast doubles as a forecast of when the grid is clean.

Carbon intensity is estimated per hour as:

```
renewable_share = (wind_onshore + wind_offshore + solar) / load   clipped to [0, 1]
intensity       = RESIDUAL_INTENSITY × (1 − renewable_share)
```

with `RESIDUAL_INTENSITY = 0.60 tCO₂/MWh` approximating Germany's non-renewable residual mix (roughly half coal at ~0.9–1.1, half gas CCGT at ~0.35–0.40). It is a parameter, so the module re-points at another grid.

### 4.2 What the complete data says

An early run reported **r = 0.859**. That run used the incomplete fetch described in §1.1, in which wind onshore was largely missing. With wind absent, `renewable_share` was effectively a solar proxy — and solar is strongly diurnal, so it is strongly anti-correlated with price by construction. Restoring the full wind series, which is Germany's largest renewable source and is *not* diurnal, gives:

| Window | Hours | Pearson r | Cheap tercile renewable | Pricey tercile renewable | CO₂ gap |
|---|---|---|---|---|---|
| Full 2022–2025 | 35,061 | **0.398** | 49.8% | 26.0% | 0.143 |
| Winter months (Oct–Mar) | 17,493 | 0.574 | 61.5% | 27.6% | 0.204 |
| Summer months (Apr–Sep) | 17,568 | 0.289 | 38.6% | 24.2% | 0.086 |
| Oct–Dec 2025 (hold-out) | 2,207 | 0.693 | 65.7% | 28.4% | 0.224 |

The relationship is real and positive but **moderate, and strongly seasonal** — not the near-deterministic link the first number implied.

### 4.3 The decomposition that actually matters

Even the surviving correlation is mostly not exploitable, because it lives *between* days rather than *within* them:

| | Renewable share, cheap vs expensive | CO₂ gap | Load shiftable here? |
|---|---|---|---|
| Between days (windy vs calm) | 50.3% vs 27.4% | 0.138 t/MWh | **No** |
| Within a day | 39.2% vs 33.5% | **0.034 t/MWh** | **Yes** |

An operator can move a chiller run from 19:00 to 03:00; they cannot move June demand into April. Only the within-day component is actionable, and its carbon content is small.

**Conclusion:** GridShift is primarily a **cost-optimisation and peak-shaving** tool — ~81 EUR/MWh shifted — with a **modest carbon co-benefit** of ~0.034 tCO₂/MWh. That is a defensible business case. Presenting it as a decarbonisation breakthrough would not be.

---

## 5. Recommendation output (`curve/load_shift.py`)

Hours are banded **within each day** — ranked against the other hours of their own day, not pooled across the window, since pooling would mostly rank days against each other and yield unactionable advice. A clock hour is reported in a band only if it lands there on ≥50% of days.

Hold-out window (2025-10-01 → 2025-12-31):

| Band | Local hours | Mean price | Renewable share |
|---|---|---|---|
| **Green** | 00–05, 12, 23 | 64.2 EUR/MWh | 48.7% |
| **Red** | 07–09, 15–20 | 127.8 EUR/MWh | 41.8% |

Per MWh shifted: **63.6 EUR saved, 0.041 tCO₂ avoided**. At 200 MWh/day of shiftable load: ~4.65M EUR/yr and ~2,990 tCO₂/yr — illustrative, since it assumes flexibility every day at the observed spread.

**Confidence** is gated on *both* forecast accuracy (spread ≥ 1.5× MAE) and the strength of the price/carbon link (r ≥ 0.4). A precise forecast of a signal that does not track carbon is worthless for the carbon claim.

**Override conditions** (any one suspends the recommendation):

1. Renewable share data stale or >6h delayed
2. Trailing-30-day MAE exceeds 2× hold-out MAE
3. Trailing-30-day price/carbon correlation below 0.25 — premise broken
4. A process constraint would be breached (storage limit, minimum output, departure deadline)
5. The grid operator issues a demand-response instruction

---

## 6. Limitations

- **The carbon case is modest** (§4.3) and seasonal. Lead with cost.
- **Carbon intensity is derived, not metered.** The correlation shows price tracks *renewable share*; production use needs the operator's published emissions factors.
- **Source data is unreliable** (§1.1) — pinned datasets and coverage assertions are prerequisites, not niceties.
- **Prediction intervals under-cover** (70.2% vs nominal 80%) in volatile quarters.
- **Germany, not the UAE.** Needs revalidation on local data and market rules.
- **Rebound.** At scale, coordinated shifting into the same hours erodes the spread it depends on.

---

## 7. Repository

```
gridshift/
├── run_pipeline.py                 # single entry point
├── app.py                          # Streamlit dashboard
├── PROPOSAL.md                     # one-page submission
├── src/gridshift/
│   ├── ingestion/                  # SMARD + ENTSO-E fetchers
│   ├── quality/                    # standard QA + LLM QA with suspect-rule guard
│   ├── features/engineer.py        # leakage-controlled feature engineering
│   ├── models/                     # seasonal naive + LightGBM + walk-forward CV
│   └── curve/
│       ├── translation.py          # hourly → delivery blocks (decision-free)
│       └── load_shift.py           # premise check + within-day banding + impact
└── outputs/
    ├── qa_report.json  llm_qa_log.json  load_shift_plan.json
    ├── delivery_views.csv  submission.csv
    └── figures/
```

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env          # add ANTHROPIC_API_KEY for the LLM QA step
python run_pipeline.py        # uses the pinned cache; FORCE_REFETCH=false
streamlit run app.py
```
