# GridShift — Technical Report

**Scheduling flexible electricity demand from day-ahead price forecasts**
**Sneha Sunil · snehasunil385@gmail.com**

> **Revision note.** The figures in this report are from a re-run after two
> ingestion bugs were found and fixed (§1.1). The version submitted on
> 2025-08-06 is preserved unchanged in `.submitted-2025-08-06/`. Every number
> that moved is listed in §1.2.

---

## Executive summary

- **What it does:** forecasts next-day hourly electricity prices and bands each day's 24 hours into green/amber/red, so operators of flexible load know which hours to run in.
- **Model:** LightGBM on 37 features. Hold-out **MAE 26.85 EUR/MWh** vs seasonal-naive **49.42** — **45.7% better**. 12-fold walk-forward CV (MAE 24.5 ± 8.0), no temporal leakage.
- **Actionable impact:** within-day shifting saves **~81 EUR/MWh** and avoids **~0.16 tCO₂/MWh** across 2022–2025.
- **The ingestion layer was wrong, twice, and silently** (§1.1). Both failures produced a dataset that looked complete and passed every quality check.
- **AI component:** Claude (`claude-haiku-4-5`) generates executable data-quality rules; the pipeline runs and audits them behind guards that quarantine malformed rules and refuse to score empty columns.

---

## 1. Data

**Source:** SMARD (Bundesnetzagentur), no API key. ENTSO-E is implemented as a keyed alternative (`ingestion/entsoe_client.py`) with automatic monthly chunking.

| Series | SMARD filter | Mean | Range |
|---|---|---|---|
| Day-ahead price | 4169 | 124.6 EUR/MWh | −500 → 936 |
| Wind onshore | 4067 | 12,579 MWh/h | 47 → 48,683 |
| Wind offshore | 1225 | 2,856 MWh/h | 0 → 8,278 |
| Solar PV | 4068 | 7,114 MWh/h | 1 → 52,466 |
| Load (Netzlast) | 410 | 53,942 MWh/h | 31,278 → 79,486 |

**Timezone:** timestamps stored UTC; calendar features and reported clock hours derived via `tz_convert("Europe/Berlin")`, handling CET↔CEST correctly.

**Coverage:** 35,064 hourly rows, 2022-01-01 → 2025-12-31, zero gaps and zero missing values on all five series. After feature engineering (lags require warm-up), **34,727 usable rows**, 2022-01-15 → 2025-12-31.

### 1.1 Two silent ingestion bugs

The original report described the source as "not reproducible run-to-run" and treated that as the project's most operationally important finding. That conclusion was right about the danger and wrong about the cause. Both were bugs in this repository, and both were silent: the pipeline received a dataset that looked complete, passed every quality check, and was wrong.

**Bug 1 — three of five SMARD filter IDs were wrong.**

| Column as labelled | Filter used | What that filter actually returns |
|---|---|---|
| `wind_onshore_mwh` | 4066 | **Biomass**, a near-constant 3.5–5.3 GW |
| `solar_mwh` | 4067 | **Onshore wind** |
| `wind_offshore_mwh` | 4065 | Nothing; not a valid DE-LU filter |
| `load_mwh` | 4381 | Almost nothing |
| — | 4068 | **Solar PV, which was never fetched** |

Nothing errored. Two columns arrived full of real data for the wrong technology, and two arrived empty. The tell was visible in the quality report all along: `wind_onshore_mwh` ranged from 3,096 to 5,285 MWh/h across four years, when German onshore wind swings between roughly 500 and 45,000, and `solar_mwh` had a minimum of 47 when solar is exactly zero every night. Domain bounds were checked; domain *shape* was not.

The cached parquet had in fact been built by an earlier, uncommitted version of the fetcher that used the correct 410 and 1225 for load and offshore wind, which is why those two columns held good data while the committed code could never reproduce them. Raw data was gitignored, so the discrepancy was invisible.

**Bug 2 — ENTSO-E was chunking by year against a one-month API limit.**

`ENTSOE_API_KEY` was set in `.env`, and `_step_ingest` prefers ENTSO-E whenever a key is present, so every re-fetch used ENTSO-E rather than the SMARD path the documentation described. ENTSO-E caps A65 (load) and A75 (generation per type) at one month per request:

```
HTTP 400 — Provided time interval (2022-01-01 - 2022-12-31) is larger than
maximum allowed period 'P1M' for 'ACTUAL_TOTAL_LOAD_R3:XML' export.
```

`_fetch_chunked` caught the exception, logged it, and returned an empty Series, which became an all-NaN column. A44 (day-ahead price) does allow a year, so price was the one series that ever arrived. A run of this path produced a dataset with price and nothing else.

**Together these explain the "unreliable source".** Different runs failed differently depending on which fetcher ran and which filter happened to be valid, which looked exactly like an upstream data provider behaving inconsistently.

**Fixes.** Correct filter IDs, verified by matching each series against the live endpoint before trusting it again. Monthly chunking for ENTSO-E. `min_count=1` on the hourly resample so an hour with no readings stays NaN instead of summing to a plausible 0 MWh. And one coverage assertion in `pipeline._assert_coverage`, placed where the SMARD path, the ENTSO-E path and the cache all converge, which refuses to model on any dataset where a required series is absent or more than 5% missing.

The lesson the original report drew from this — pin and version raw data, assert coverage *before* modelling — survives intact. It just has a different cause, and a sharper one: the failure was mine, it was silent, and the checks in place at the time were incapable of catching it.

### 1.2 What changed when the data was corrected

| | Submitted | Corrected |
|---|---|---|
| Hold-out MAE | 28.76 | **26.85** |
| Improvement over baseline | 41.8% | **45.7%** |
| CV MAE | 23.9 ± 7.3 | 24.5 ± 8.0 |
| 80% interval coverage | 70.2% | 73.4% |
| Price↔carbon r, full period | 0.398 | **0.534** |
| Price↔carbon r, hold-out | 0.693 | 0.739 |
| Within-day CO₂ gap | 0.034 t/MWh | **0.160 t/MWh** |
| Between-day CO₂ gap | 0.138 t/MWh | 0.170 t/MWh |
| Green hours | 00–05, 12, 23 | 00–05, 23 |
| Per MWh shifted (hold-out) | 63.6 EUR, 0.041 t | 58.6 EUR, 0.066 t |
| Within-day spread, full period | 81 EUR/MWh | 81 EUR/MWh |

Every cost figure is unchanged, because cost is derived from price alone and price was the one series that was always correct. Every carbon figure moved, because solar is the most diurnal renewable there is and it was missing.

---

## 2. Data quality

Standard checks (`quality/standard_qa.py`): missingness, duplicates, hourly gaps, domain-bound outliers, monthly coverage. On the corrected dataset the report is clean: 35,064 rows, 0 gaps, 0 duplicates, no domain-bound violations.

**LLM-driven QA (`quality/llm_qa.py`):** Claude receives a schema summary and five sample rows and returns 8–12 validation rules as JSON, each an executable Python expression over a single column. The pipeline runs every rule and logs the system prompt, raw response, parsed rules and per-rule violation counts to `outputs/llm_qa_log.json`. The latest run produced 12 rules: 9 pass, 3 fail (23 hours of large solar ramps, 22 of large load ramps, 1 extreme price spike — all genuine features of the data rather than errors). The exact set varies between runs, which is itself an argument for executing and auditing rather than trusting any single generation.

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

### 2.2 The QA layer could also pass a column that did not exist

`_execute_rule` calls `df[col].dropna()` before evaluating a rule. On an all-NaN column that leaves an empty Series, every rule is vacuously satisfied, and the run reports `pass, 0 violations`. When the ENTSO-E bug emptied four of five columns, the LLM QA step reported a clean bill of health on all of them.

This is the same class of failure as §2.1 in the opposite direction: §2.1 was a false alarm, this was false assurance, and false assurance is worse. An empty column is now reported as `skipped` with a reason.

---

## 3. Forecasting

**Target:** next-day hourly day-ahead price.

**Leakage control:** fundamentals lagged ≥24h; rolling price statistics computed on an already-shifted series. Features for hour *H* contain only what was knowable before gate closure.

**Features (37):** calendar (hour, day-of-week, month, week-of-year, weekend, holiday, sine/cosine encodings); price lags at 24/48/72/168/336h; 24h and 168h rolling mean/std/max/min; wind, solar and load lags at 24h/168h plus 24h rolling means; total renewables lag.

**Baseline:** seasonal naive, `price[t] = price[t−168h]`.

**Model:** LightGBM (800 estimators, lr 0.05, 63 leaves, subsample 0.8); 12 monthly walk-forward folds, expanding training window.

| Metric | CV (mean ± std) | Hold-out | Seasonal naive |
|---|---|---|---|
| MAE (EUR/MWh) | 24.5 ± 8.0 | **26.85** | 49.42 |
| RMSE (EUR/MWh) | 35.8 ± 13.4 | **36.10** | — |
| Pinball @ P90 | 12.4 | — | — |
| Improvement | — | **+45.7%** | — |

**Hold-out:** 2025-10-01 → 2025-12-31, 2,208 hours, never seen during development.

**Prediction intervals:** empirical 80% interval from CV fold residuals, calibrated per hour-of-day. Empirical coverage **73.4%** against a nominal 80% — still under-covered, reflecting a volatile Q4. Reported rather than tuned away; the confidence gate accounts for it.

---

## 4. The premise test

### 4.1 The thesis

In a merit-order market the clearing price is set by the marginal generator, so abundant wind and solar should push both price and carbon intensity down together. If that holds strongly, a price forecast doubles as a forecast of when the grid is clean.

Carbon intensity is estimated per hour as:

```
renewable_share = (wind_onshore + wind_offshore + solar) / load   clipped to [0, 1]
intensity       = RESIDUAL_INTENSITY × (1 − renewable_share)
```

with `RESIDUAL_INTENSITY = 0.60 tCO₂/MWh` approximating Germany's non-renewable residual mix (roughly half coal at ~0.9–1.1, half gas CCGT at ~0.35–0.40). It is a parameter, so the module re-points at another grid.

### 4.2 What the data says

| Window | Hours | Pearson r | Cheap tercile renewable | Pricey tercile renewable | CO₂ gap |
|---|---|---|---|---|---|
| Full 2022–2025 | 35,064 | **0.534** | 56.3% | 29.6% | 0.160 |
| Winter months (Oct–Mar) | 17,496 | 0.627 | — | — | — |
| Summer months (Apr–Sep) | 17,568 | 0.492 | — | — | — |
| Oct–Dec 2025 (hold-out) | 2,208 | 0.739 | 66.1% | 24.9% | 0.247 |

The relationship is real, positive and seasonal, stronger in winter when wind rather than solar sets the margin. It is not deterministic: r = 0.53 leaves most of the variance in price to fuel costs, outages, imports and demand.

### 4.3 The decomposition that matters

The correlation is worth having only to the extent that it lives *within* days, because that is the only axis load can move along:

| | Renewable share, cheap vs expensive | CO₂ gap | Load shiftable here? |
|---|---|---|---|
| Between days (windy vs calm) | 57.6% vs 29.3% | 0.170 t/MWh | **No** |
| Within a day | 56.3% vs 29.6% | **0.160 t/MWh** | **Yes** |

An operator can move a chiller run from 19:00 to 03:00; they cannot move June demand into April. The within-day component is the actionable one, and it is nearly as large as the between-day component — solar is what makes it so.

**On the earlier figures.** The submitted version reported a within-day gap of 0.034 t/MWh and concluded that carbon was a minor co-benefit. That conclusion was an artifact of the missing solar series, and the corrected figure is 4.7× larger. The submitted version also reported correcting an even earlier r = 0.859 down to 0.398, and attributed the drop to solar dominating the renewable share in an incomplete fetch. That explanation cannot have been right: there was no solar series in the dataset at any point. The 0.859 came from a dataset state that no longer exists, because raw data was gitignored and never versioned. That is the honest account, and it is also the strongest argument in this report for versioning the inputs: without them, a wrong number cannot be explained, only replaced.

**Conclusion.** GridShift is a cost-optimisation and peak-shaving tool — ~81 EUR/MWh shifted — with a **real carbon co-benefit of ~0.16 tCO₂/MWh**. Cost remains the claim to lead with, because it is larger relative to its uncertainty and because it is what the operator's finance team already measures.

---

## 5. Recommendation output (`curve/load_shift.py`)

Hours are banded **within each day** — ranked against the other hours of their own day, not pooled across the window, since pooling would mostly rank days against each other and yield unactionable advice. A clock hour is reported in a band only if it lands there on ≥50% of days.

Hold-out window (2025-10-01 → 2025-12-31):

| Band | Local hours | Mean price | Renewable share |
|---|---|---|---|
| **Green** | 00–05, 23 | 62.4 EUR/MWh | 47.5% |
| **Red** | 07–09, 15–20 | 121.0 EUR/MWh | 36.5% |

Per MWh shifted: **58.6 EUR saved, 0.066 tCO₂ avoided**. At 200 MWh/day of shiftable load: ~4.28M EUR/yr and ~4,820 tCO₂/yr — illustrative, since it assumes flexibility every day at the observed spread.

Midday (12:00) was green on the mislabelled data and is not on the corrected data. In an October–December window the solar dip is shallow and the midday hour no longer clears the 50%-of-days threshold, which is the banding rule working as intended rather than a regression.

**Confidence** is gated on *both* forecast accuracy (spread ≥ 1.5× MAE) and the strength of the price/carbon link (r ≥ 0.4). A precise forecast of a signal that does not track carbon is worthless for the carbon claim.

**Override conditions** (any one suspends the recommendation):

1. Renewable share data stale or >6h delayed
2. Trailing-30-day MAE exceeds 2× hold-out MAE
3. Trailing-30-day price/carbon correlation below 0.25 — premise broken
4. A process constraint would be breached (storage limit, minimum output, departure deadline)
5. The grid operator issues a demand-response instruction

---

## 6. Limitations

- **Carbon intensity is derived, not metered.** The correlation shows price tracks *renewable share*; production use needs the operator's published emissions factors. The 0.60 tCO₂/MWh residual figure is a blended approximation, and the CO₂ numbers scale linearly with it.
- **Prediction intervals under-cover** (73.4% vs nominal 80%) in volatile quarters.
- **The baseline is a low bar.** Seasonal naive at 49.42 EUR/MWh is the standard reference, but an hour-of-day climatology or a linear model on the same features would be a harder comparison, and neither was run.
- **2022 was an energy crisis.** Mean price was 235 EUR/MWh in 2022 against 78 in 2024. Training across that regime shift is defensible for a tree model, but the model has seen one structural break and would not anticipate another.
- **Germany, not the UAE.** Needs revalidation on local data and market rules; see the proposal on why the input signal itself has to change.
- **Rebound.** At scale, coordinated shifting into the same hours erodes the spread it depends on.
- **Ingestion is verified, not monitored.** The filter IDs are correct today and asserted for coverage on every run, but nothing detects a source silently re-pointing a filter ID at a different technology. A distribution check against the previous vintage would.

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
├── outputs/
│   ├── qa_report.json  llm_qa_log.json  load_shift_plan.json
│   ├── delivery_views.csv  submission.csv
│   └── figures/
└── .submitted-2025-08-06/          # the submitted data, outputs and docs, unchanged
```

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env          # add ANTHROPIC_API_KEY for the LLM QA step
python run_pipeline.py        # uses the pinned cache; FORCE_REFETCH=false
streamlit run app.py
```
