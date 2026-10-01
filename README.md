# GridShift — Scheduling Flexible Electricity Demand

**Sneha Sunil · snehasunil385@gmail.com**

Forecast tomorrow's hourly electricity prices and tell operators of flexible demand — district cooling, desalination, EV fleets, data centres — which hours to run in.

**On the carbon claim:** the thesis is that price is a proxy for carbon intensity via the merit-order effect. The pipeline tests this on every run rather than assuming it, and it holds moderately: **r = 0.534** over 2022–2025, split roughly evenly between a within-day component (0.160 tCO₂/MWh, which load can be shifted across) and a between-day one (0.170 tCO₂/MWh, which it cannot). GridShift leads on **cost and peak shaving** with a **real carbon co-benefit**.

**Revision note:** these numbers are from a re-run after two silent ingestion bugs were found and fixed. Three of five SMARD filter IDs were wrong — solar was never fetched at all, and biomass was labelled as onshore wind — and the ENTSO-E client chunked requests by year against a one-month API limit. The submitted version is preserved in `.submitted-2025-08-06/`; §1.1 and §1.2 of `REPORT.md` cover what happened and every number that moved.

See `PROPOSAL.md` for the one-page write-up and `REPORT.md` for the full technical report.

---

## What it does

| Step | Module | Description |
|------|--------|-------------|
| 1 | `ingestion/` | Hourly DE price, load and wind/solar from SMARD (no key) or ENTSO-E |
| 2a | `quality/standard_qa.py` | Missingness, duplicates, hourly gaps, domain-bound outliers |
| 2b | `quality/llm_qa.py` | **AI component**: Claude proposes executable QA rules; the pipeline runs and audits them |
| 3 | `features/engineer.py` | Calendar, lag and renewable features under strict leakage control |
| 4 | `models/` | Seasonal-naive baseline + LightGBM with 12-fold walk-forward CV |
| 5 | `curve/load_shift.py` | Premise check → green/amber/red hour banding → shift recommendation |

---

## Results

| | |
|---|---|
| Forecast error (hold-out, 2,208 h) | **26.85 EUR/MWh MAE** — **45.7% better** than seasonal naive (49.42) |
| Cross-validation | 12 walk-forward folds, MAE 24.5 ± 8.0 |
| Prediction interval | 80% nominal, 73.4% empirical coverage (under-covered; reported, not tuned away) |
| Price ↔ carbon correlation | **r = 0.534** full period · 0.739 in the hold-out window |
| Green hours (shift into) | **00–05, 23** · 62.4 EUR/MWh |
| Red hours (shift out of) | **07–09, 15–20** · 121.0 EUR/MWh |
| Impact per MWh shifted | **~81 EUR saved · ~0.16 tCO₂ avoided** (within-day, 2022–2025) |

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

cp .env.example .env
# ANTHROPIC_API_KEY  — required for the LLM QA step
# ENTSOE_API_KEY     — optional; falls back to SMARD
# FORCE_REFETCH      — leave false; see the data-source caveat below
```

```bash
python run_pipeline.py     # uses the pinned cache in data/raw/
streamlit run app.py       # interactive dashboard over the outputs
```

---

## AI component

`src/gridshift/quality/llm_qa.py` calls Claude (`claude-haiku-4-5`) with a schema summary and five sample rows, and asks for 8–12 validation rules as JSON. Each rule carries a Python expression evaluated against a single column. The pipeline executes every rule and writes the system prompt, raw response, parsed rules and per-rule violation counts to `outputs/llm_qa_log.json`.

**The pipeline does not trust the model.** An early run produced rules of the form `s.diff().abs() <= 8000 | s.diff().isna()`. Python binds `|` more tightly than `<=`, so this evaluates as `s.diff().abs() <= (8000 | ...)`, the right side collapses to `True`, and nearly every row "fails" — 720 reported violations against a true count of 6. Two fixes are in place: the system prompt names the precedence trap and requires full parenthesisation, and `_execute_rule` marks any rule flagging more than 50% of rows as **`suspect`** rather than reporting it as a data-quality finding. The guard is the load-bearing one, since it does not depend on the model complying.

A second guard covers the opposite failure. `_execute_rule` calls `df[col].dropna()` before evaluating, so on an all-NaN column every rule is vacuously satisfied and the run reports `pass`. When the ENTSO-E bug emptied four of five columns, LLM QA gave all of them a clean bill of health. An empty column is now `skipped` with a reason. A false alarm wastes an afternoon; false assurance ships.

API keys are read from environment variables only; `.env` is gitignored.

---

## Data sources

**Primary (no key): SMARD — Bundesnetzagentur**

| Column | Filter | Description |
|---|---|---|
| `price_da_eur_mwh` | 4169 | EPEX Spot day-ahead auction |
| `load_mwh` | 410 | Realised grid load (Netzlast) |
| `wind_onshore_mwh` | 4067 | Realised onshore wind |
| `wind_offshore_mwh` | 1225 | Realised offshore wind |
| `solar_mwh` | 4068 | Realised solar PV |

**These filter IDs were wrong in the submitted version, and the error was silent.** 4066 is biomass (a near-constant ~4 GW) and was labelled onshore wind; 4067 is onshore wind and was labelled solar; real solar (4068) was never fetched; 4065 and 4381 are not valid DE-LU filters and returned nothing. Separately, the ENTSO-E client chunked requests by year when the API caps load and generation at one month, so every re-fetch through that path returned price and nothing else. Different runs failing differently is what produced the "the source is unreliable" conclusion in the submitted report.

Three guards are now in place: `min_count=1` on the hourly resample so an empty hour stays NaN instead of summing to 0 MWh, a coverage assertion in `pipeline._assert_coverage` that refuses to model on any dataset where a required series is absent or >5% missing (it sits where SMARD, ENTSO-E and the cache all converge), and an LLM-QA guard that reports an empty column as `skipped` rather than passing it. `FORCE_REFETCH=false` still pins the cache.

**Alternative (with key): ENTSO-E Transparency Platform** — register at https://transparency.entsoe.eu/usrm/user/createPublicUser. DA prices A44/A01, load A65, wind/solar A75 with psrType B19/B18/B16, domain DE-LU (`10Y1001A1001A82H`). The client splits ranges into calendar-month chunks automatically, because the API caps A65 and A75 at one month per request and answers anything longer with HTTP 400. To switch an existing install, set `FORCE_REFETCH=true` for one run.

---

## Methodology notes

**Leakage control.** Raw fundamentals are always lagged ≥24h and rolling statistics are computed on an already-shifted series, so the features for hour *H* contain only what was knowable before day-ahead gate closure.

**Validation.** 12 monthly walk-forward folds with an expanding training window; no data from a validation month reaches training. Metrics: MAE (primary), RMSE, pinball loss at P10/P90.

**Within-day banding.** Hours are ranked against the other hours of their own day, not across the whole window — an operator can move a chiller run from 19:00 to 03:00 but cannot move June demand into April. A clock hour is only reported in a band if it lands there on at least half of days. Only the within-day component of the price/carbon relationship is actionable: 0.160 t/MWh of the total, against 0.170 t/MWh that lives between days and cannot be exploited by scheduling.

**Timezone.** All timestamps stored UTC; calendar features and reported clock hours derived via `tz_convert("Europe/Berlin")`, handling CET↔CEST correctly.

---

## Outputs

```
outputs/
├── qa_report.json          # standard QA + model performance
├── llm_qa_log.json         # full LLM prompt, response and rule execution log
├── load_shift_plan.json    # premise check, green/red windows, impact, overrides
├── delivery_views.csv      # weekly/monthly base/peak/offpeak blocks
├── submission.csv          # hourly out-of-sample predictions with P10/P90
└── figures/                # 5 figures
```
