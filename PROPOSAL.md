# GridShift: Scheduling Flexible Electricity Demand Around the Grid

**Sneha Sunil · snehasunil385@gmail.com · Presight Innovation Challenge**

### Problem

Large electricity consumers with genuinely flexible demand (district cooling plants, desalination trains, EV fleets, data centres) mostly run to fixed schedules set by habit. Meanwhile the cost and carbon content of a kilowatt-hour swing enormously across a single day: in the German market the cheapest third of hours in a day average **84 EUR/MWh** while the priciest third average **166 EUR/MWh**. Chilled water can be made early and stored, a desalination train can be throttled, an EV can charge at 2am instead of 7pm. The flexibility exists; the scheduling intelligence does not.

### Proposed solution

GridShift forecasts the next day's hourly electricity price and issues a concrete instruction to operators of flexible load: **run in these hours, not those.** It ranks each day's 24 hours into green, amber and red bands, attaches a confidence level, and states the conditions under which its own advice should be ignored.

### What I set out to prove, and what the data actually said

My starting thesis was stronger: that price is a good proxy for *carbon intensity*, because in a merit-order market abundant wind and solar push fossil plants out of the stack and the price down. If true, a price forecast would double as a forecast of when the grid is clean.

**I tested it, and it only partly holds.** Across four years of hourly German data the correlation between price and estimated carbon intensity is **r = 0.40**: real and positive, but moderate. More importantly, decomposing it shows the relationship lives mostly *between* days rather than *within* them:

| | Renewable share, cheap vs expensive | CO₂ gap | Can load actually be shifted here? |
|---|---|---|---|
| Between days (windy vs calm) | 50.3% vs 27.4% | 0.138 t/MWh | **No.** You cannot move June demand into April |
| Within a day | 39.2% vs 33.5% | **0.034 t/MWh** | **Yes** |

An earlier version of this analysis reported r = 0.86. That figure was an artifact of a gap in the source data: with wind onshore missing, the renewable share was dominated by solar, which is strongly diurnal and therefore strongly anti-correlated with price. Restoring the complete wind series cut the correlation by more than half. I am reporting the corrected figure because the first one would not have survived scrutiny.

**The honest conclusion:** GridShift is primarily a **cost-optimisation and peak-shaving** tool with a **modest carbon co-benefit**, roughly **81 EUR and 0.034 tCO₂ saved per MWh shifted**. That is still a strong business case, and peak shaving has real grid value beyond the operator's own bill. It is not a decarbonisation silver bullet, and pitching it as one would be wrong.

### Target users and stakeholders

Operators of large flexible loads: district cooling utilities, desalination and water authorities, EV charging networks, data centre operators. Grid operators benefit indirectly: shifting demand off the evening peak defers capacity investment.

**In a UAE deployment the primary customer is the national utility itself** (EWEC, TRANSCO or DEWA) rather than an individual site. Because those entities are the single buyer and the system operator, they capture the deferred-capacity benefit directly, at national scale, and they are the kind of government-linked customer Presight already serves. District cooling is the largest single component of Gulf summer peak demand and desalination is a genuinely flexible industrial load, so the addressable flexibility is unusually large.

### Technology, tools and data

Python; LightGBM for forecasting; hourly market data from public transparency platforms (SMARD, ENTSO-E); Claude (`claude-haiku-4-5`) via the Anthropic API for automated data-quality rule generation; Streamlit for the operator dashboard. All data used is public and free.

### How it works

Ingest hourly prices, demand and renewable generation → run automated quality checks → engineer calendar, lag and renewable features under strict leakage control → forecast next-day hourly prices with LightGBM, validated by 12-fold walk-forward cross-validation → **re-test the price/carbon premise on every run** → rank each day's hours green/amber/red → issue the recommendation with confidence and explicit override conditions.

### Evidence from the working prototype

Built on Germany, the most liquid public power market in the world, using four years of hourly data (34,728 usable hours, 2022–2025):

| | Result |
|---|---|
| Forecast error (Oct–Dec 2025 hold-out, 2,208 h unseen) | **28.8 EUR/MWh MAE**, **41.8% better** than the seasonal-naive industry baseline (49.4) |
| Validation | 12-fold walk-forward CV, MAE 23.9 ± 7.3, no temporal leakage |
| Prediction interval | 80% nominal, 70.2% empirical coverage. Under-covered in a volatile quarter, and reported rather than hidden |
| Recommendation (hold-out window) | Shift **into** 00:00–05:00, 12:00, 23:00; **out of** 07:00–09:00 and 15:00–20:00 |
| Impact per MWh shifted | **63.6 EUR saved, 0.041 tCO₂ avoided** |

### Business value

For an operator with 200 MWh/day of genuinely shiftable load, the hold-out window implies roughly **4.6M EUR and 3,000 tCO₂ per year**. That figure is illustrative: it assumes flexibility is available every day at the observed spread, which real operational constraints would not sustain. The credible claim is the per-MWh figure; the annual number scales with how much flexibility an operator actually has.

The strategic point is that the saving comes from *scheduling* rather than from building anything: no new generating plant and no grid-scale batteries, just existing equipment running at better times. And as solar penetration rises, intraday spreads widen, so the value grows rather than erodes.

**Deployment is not free, though.** The model has to be integrated with the site's control system and the operator needs interval metering. Two prerequisites matter more than either. The flexibility must physically exist, since a district cooling plant with no thermal store cannot shift its load at any price. And the operator must be exposed to time-varying prices, because on a flat regulated tariff the saving is zero however accurate the forecast. That second condition is the same one that makes the UAE market structure a live question rather than a footnote.

### Where AI is used

**In the product**, Claude automates data quality. Given a schema and sample rows it proposes 8–12 executable validation rules covering physical feasibility, market realism and temporal consistency, which the pipeline runs against the full dataset, logging every prompt, response and result. This compresses a day of analyst work into seconds and generalises: point it at another country's data and it writes that country's rules.

**Critically, the pipeline does not trust it.** The model produced rules containing a Python operator-precedence bug (`a <= b | c` parses as `a <= (b | c)`), which caused clean data to be reported as **720 quality failures when the true count was 6**. I found it, fixed the prompt, and added a guard that quarantines any rule flagging more than half the dataset as a malformed rule rather than a data finding. The guard is the load-bearing fix, because it does not depend on the model complying.

**In building this submission**, I used Claude Code to research the framing, refine the idea, restructure the analysis, review my code and generate the diagrams and slides. I set the direction and made the judgement calls; the AI accelerated execution and challenged my assumptions, including catching that my own headline correlation was a data artifact. Every number here comes from code that runs.

### Risks, limitations and key considerations

- **The carbon case is modest.** Only the within-day component of the price/carbon relationship is actionable, and it is small (0.034 tCO₂/MWh). Sell this on cost and peak shaving; treat carbon as a co-benefit.
- **Carbon intensity is estimated, not metered.** It is derived from renewable share of demand, so the correlation demonstrates that price tracks *renewable share*, not measured emissions. Production use needs the grid operator's published emissions factors.
- **The source data is unreliable.** Repeated fetches from SMARD returned different coverage each time. One run lost the load series entirely, another lost wind onshore, a third returned a 404. This is precisely how the r = 0.86 artifact arose. Any production version needs pinned, versioned datasets and coverage assertions before modelling, not after.
- **The UAE has no wholesale hourly price to forecast.** This is the most important limitation, and it is structural rather than statistical. The model consumes a wholesale clearing price set by a marginal generator; EWEC operates as a **single buyer** procuring through long-term PPAs, DEWA is vertically integrated, and tariffs are regulated. Lifting this system to Abu Dhabi unchanged would fail because the input signal does not exist. The fix is a substitution, not a rebuild: in a single-buyer market the utility already knows its own dispatch stack, so it forecasts **system marginal cost and grid carbon intensity directly** instead of inferring them from price. The forecasting, within-day banding, premise check and override logic are all unchanged; only the input series changes. It also improves the commercial fit, moving the customer from a private site operator to a national utility.
- **Physics bounds flexibility.** Storage limits, minimum output levels and delivery deadlines constrain what can move; the process constraint must always override the recommendation.
- **Rebound risk.** If everyone shifts into the same cheap hours, those hours stop being cheap. At scale this needs coordination with the grid operator.

### Success measures

1. **Forecast accuracy.** Beat the seasonal-naive baseline by >15%, monitored on a 30-day rolling basis.
2. **Premise integrity.** The price/carbon correlation is re-tested every run; below 0.25 the system suspends itself rather than issuing advice it cannot justify.
3. **Realised shift.** Share of flexible load actually moved into green hours versus a pre-deployment baseline.
4. **Cost per MWh consumed.** The primary commercial measure, and the one the operator's finance team already tracks.
5. **Avoided emissions.** tCO₂ reconciled against the grid operator's published factors, reported honestly against the modest expected effect.
6. **Trust.** Acceptance and override rates. A system that operators ignore has failed regardless of its accuracy.
