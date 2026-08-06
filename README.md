# GridShift — Scheduling Flexible Electricity Demand

**Sneha Sunil · snehasunil385@gmail.com**

Forecast tomorrow's hourly electricity prices and tell operators of flexible demand — district cooling, desalination, EV fleets, data centres — which hours to run in.

**On the carbon claim:** the original thesis was that price is a strong proxy for carbon intensity via the merit-order effect. The pipeline tests this on every run rather than assuming it, and on complete data it only partly holds: **r = 0.398** over 2022–2025, and most of that lives *between* days rather than within them, where load can actually be shifted. GridShift is therefore primarily a **cost-optimisation and peak-shaving** tool with a **modest carbon co-benefit**. See §4 of `REPORT.md` for the full correction — an earlier r = 0.86 turned out to be an artifact of missing wind data.

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
| Forecast error (hold-out, 2,208 h) | **28.76 EUR/MWh MAE** — **41.8% better** than seasonal naive (49.42) |
| Cross-validation | 12 walk-forward folds, MAE 23.9 ± 7.3 |
| Prediction interval | 80% nominal, 70.2% empirical coverage (under-covered; reported, not tuned away) |
| Price ↔ carbon correlation | **r = 0.398** full period · 0.693 in the hold-out window |
| Green hours (shift into) | **00–05, 12, 23** · 64.2 EUR/MWh |
| Red hours (shift out of) | **07–09, 15–20** · 127.8 EUR/MWh |
| Impact per MWh shifted | **~81 EUR saved · ~0.034 tCO₂ avoided** (within-day, 2022–2025) |

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

API keys are read from environment variables only; `.env` is gitignored.

---

## Data sources

**Primary (no key): SMARD — Bundesnetzagentur**

| Column | Filter | Description |
|---|---|---|
| `price_da_eur_mwh` | 4169 | EPEX Spot day-ahead auction |
| `load_mwh` | 4381 | Realised grid load |
| `wind_onshore_mwh` | 4066 | Realised onshore wind |
| `wind_offshore_mwh` | 4065 | Realised offshore wind |
| `solar_mwh` | 4067 | Realised solar PV |

**Known limitation — the source is not reproducible run-to-run.** Repeated fetches of the same date range returned materially different coverage: one run lost `load` entirely (62.6% missing), another lost `wind_onshore` after 2024-12-31, another returned HTTP 404 for `wind_offshore`. The canonical dataset was assembled by taking the most complete version of each series across fetches, and `FORCE_REFETCH=false` pins that cache. This is not a footnote: the incomplete fetch is what produced the erroneous r = 0.86. Production needs pinned, versioned datasets and coverage assertions *before* modelling.

**Alternative (with key): ENTSO-E Transparency Platform** — register at https://transparency.entsoe.eu/usrm/user/createPublicUser. DA prices A44/A01, load A65, wind/solar A75 with psrType B19/B18/B16, domain DE-LU (`10Y1001A1001A82H`). The client splits multi-year ranges into annual chunks automatically. To switch an existing install, set `FORCE_REFETCH=true` for one run.

---

## Methodology notes

**Leakage control.** Raw fundamentals are always lagged ≥24h and rolling statistics are computed on an already-shifted series, so the features for hour *H* contain only what was knowable before day-ahead gate closure.

**Validation.** 12 monthly walk-forward folds with an expanding training window; no data from a validation month reaches training. Metrics: MAE (primary), RMSE, pinball loss at P10/P90.

**Within-day banding.** Hours are ranked against the other hours of their own day, not across the whole window — an operator can move a chiller run from 19:00 to 03:00 but cannot move June demand into April. A clock hour is only reported in a band if it lands there on at least half of days. This distinction is what reduces the headline carbon figure: the between-day CO₂ gap is 0.138 t/MWh, but only the within-day 0.034 t/MWh is actionable.

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
