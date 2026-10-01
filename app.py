"""
GridShift dashboard.

Run with:  streamlit run app.py

Design note: this is used by an operations person deciding when to run a plant.
Each screen leads with a control and an answer; explanations live in tooltips
and closed expanders, one click away, rather than in front of the numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"


# ── Loading ───────────────────────────────────────────────────────────────────
def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


# ── Formatting helpers ────────────────────────────────────────────────────────
def hour_runs(hours: list[int]) -> list[tuple[int, int]]:
    """Collapse a list of clock hours into contiguous (start, end_exclusive) runs."""
    if not hours:
        return []
    runs, start, prev = [], hours[0], hours[0]
    for h in hours[1:]:
        if h == prev + 1:
            prev = h
            continue
        runs.append((start, prev + 1))
        start = prev = h
    runs.append((start, prev + 1))
    return runs


def clock(hours: list[int]) -> str:
    """[0, 1, 2, 23] -> "00:00–03:00, 23:00–24:00"."""
    return ", ".join(f"{a:02d}:00–{b:02d}:00" for a, b in hour_runs(hours))


def eur(v: float) -> str:
    """Currency with the minus sign outside the symbol, as people write it."""
    return f"−€{abs(v):,.0f}" if v < 0 else f"€{v:,.0f}"


def strip(labels: list[str], kinds: list[str]) -> str:
    """A row of hour boxes coloured run / avoid / normal."""
    cells = "".join(f'<div class="gs-cell {k}">{t}</div>' for t, k in zip(labels, kinds))
    return (f'<div class="gs-strip">{cells}</div>'
            '<div class="gs-legend"><b class="run">■ Run</b> &nbsp; <span>■ Normal</span> &nbsp; '
            '<b class="avoid">■ Avoid</b></div>')


# ── Replay: score the model's own daily schedule against real prices ─────────
@st.cache_data
def load_replay() -> pd.DataFrame | None:
    """Hold-out forecasts joined to what actually happened, in local time, full days only."""
    fc, raw = OUTPUTS / "submission.csv", ROOT / "data" / "raw" / "de_power_market.parquet"
    if not (fc.exists() and raw.exists()):
        return None
    f = pd.read_csv(fc)
    f["id"] = pd.to_datetime(f["id"], utc=True)
    d = f.set_index("id").join(pd.read_parquet(raw), how="inner").tz_convert("Europe/Berlin")
    d["share"] = ((d["wind_onshore_mwh"] + d["wind_offshore_mwh"] + d["solar_mwh"])
                  / d["load_mwh"]).clip(0, 1)
    d["date"] = d.index.date
    sizes = d.groupby("date").size()
    return d[d["date"].isin(sizes[sizes >= 23].index)]  # drop the partial day at the edge


def score_day(g: pd.DataFrame, n: int) -> dict:
    """
    Plan the day from the forecast alone (cheapest n hours = run, priciest n = avoid),
    then score that plan at the prices that actually cleared.
    """
    run, avoid = g["y_pred"].nsmallest(n).index, g["y_pred"].nlargest(n).index
    actual = g["price_da_eur_mwh"]
    return {
        "run": run, "avoid": avoid,
        "planned": g["y_pred"][avoid].mean() - g["y_pred"][run].mean(),
        "got": actual[avoid].mean() - actual[run].mean(),
        "best": actual.nlargest(n).mean() - actual.nsmallest(n).mean(),
        "share_gap": g["share"][run].mean() - g["share"][avoid].mean(),
    }


@st.cache_data
def replay_summary(n: int, gate: float) -> dict:
    d = load_replay()
    s = pd.DataFrame([score_day(g, n) for _, g in d.groupby("date")])
    loss = s["got"] < 0
    return {"days": len(s), "got": s["got"].mean(), "captured": s["got"].sum() / s["best"].sum(),
            "loss_days": int(loss.sum()), "loss_flagged": int((loss & (s["planned"] < gate)).sum())}


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="GridShift", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  .block-container { padding-top: 2.2rem; max-width: 1200px; }
  #MainMenu, footer { visibility: hidden; }
  .gs-strip { display: flex; gap: 3px; margin: .2rem 0 .35rem 0; }
  .gs-cell { flex: 1; min-width: 0; text-align: center; padding: 13px 0; border-radius: 3px;
             font-size: 12px; font-weight: 620; font-variant-numeric: tabular-nums; }
  .gs-cell.run   { background: #0F8C80; color: #fff; }
  .gs-cell.avoid { background: #C9531F; color: #fff; }
  .gs-cell.mid   { background: #EAE3CC; color: #5C6875; }
  .gs-legend { font-size: 13px; color: #8A8270; margin-bottom: .6rem; }
  .gs-legend b.run { color: #0F8C80; } .gs-legend b.avoid { color: #C9531F; }
  .gs-when { font-size: 22px; font-weight: 660; font-variant-numeric: tabular-nums; line-height: 1.25; }
  .gs-when.run { color: #0F8C80; } .gs-when.avoid { color: #C9531F; }
  .gs-do { font-size: 15px; color: #33404C; margin-top: .2rem; }
  div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
</style>
""", unsafe_allow_html=True)


qa = _load_json(OUTPUTS / "qa_report.json")
llm_log = _load_json(OUTPUTS / "llm_qa_log.json")
plan = _load_json(OUTPUTS / "load_shift_plan.json")
submission = _load_csv(OUTPUTS / "submission.csv")

perf = qa.get("model_performance", {})
overview = qa.get("overview", {})
green = plan.get("green_window", {})
red = plan.get("red_window", {})
impact = plan.get("impact_per_mwh_shifted", {})
link = plan.get("price_carbon_link", {})

mae = perf.get("lgbm_test_mae_eur_mwh")
improvement = perf.get("lgbm_improvement_over_baseline_pct")

sub: pd.DataFrame | None = None
if submission is not None and not submission.empty:
    sub = submission.copy()
    sub["id"] = pd.to_datetime(sub["id"], utc=True)
    sub = sub.set_index("id")


st.title("GridShift")
st.caption("Tells large electricity users which hours to run their equipment in, so they buy cheap, "
           "cleaner power instead of expensive evening power. Demo on German market data, "
           "October to December 2025.")

tab_plan, tab_replay, tab_more = st.tabs(["Plan my site", "Replay any day", "Behind the numbers"])


# ══════════════════════════════════════════════════════════════════════════════
# PLAN MY SITE
# ══════════════════════════════════════════════════════════════════════════════
with tab_plan:
    if not plan:
        st.warning("No recommendation yet. Run `python run_pipeline.py` first.")
    else:
        green_hours = green.get("local_hours", [])
        red_hours = red.get("local_hours", [])
        saving = impact.get("cost_saving_eur", 0)
        co2 = impact.get("co2_avoided_t", 0)
        sites = {
            "District cooling": ("Make chilled water and store it in the tank.",
                                 "Use the stored chilled water instead of the chillers.",
                                 "You can only move as much cooling as your tank holds.", 50),
            "Desalination": ("Run the trains at full output and fill the reservoir.",
                             "Turn the trains down and supply from the reservoir.",
                             "Never go below the plant's minimum safe output.", 40),
            "EV fleet": ("Charge the vehicles.",
                         "Pause charging. Vehicles stay plugged in.",
                         "Every vehicle must still be charged by its departure time.", 20),
            "Data centre": ("Run batch jobs, training and backups.",
                            "Hold deferrable jobs. Live services run as normal.",
                            "Only work that can wait should move.", 30),
        }

        c1, c2 = st.columns([1.4, 1], gap="large")
        with c1:
            site = st.radio("What do you run?", list(sites), horizontal=True)
        run_text, avoid_text, limit_text, default_mwh = sites[site]
        with c2:
            mwh = st.slider("Electricity use you can move each day (MWh)", 5, 300, default_mwh, 5,
                            key=f"mwh_{site}",
                            help="Count only what you can really move, given your storage, minimum "
                                 "output or deadlines. 1 MWh = 1,000 kWh.")

        st.markdown(strip([f"{h:02d}" for h in range(24)],
                          ["run" if h in green_hours else "avoid" if h in red_hours else "mid"
                           for h in range(24)]), unsafe_allow_html=True)

        a1, a2 = st.columns(2)
        a1.markdown(f'<div class="gs-when run">Run {clock(green_hours)}</div>'
                    f'<div class="gs-do">{run_text}</div>', unsafe_allow_html=True)
        a2.markdown(f'<div class="gs-when avoid">Avoid {clock(red_hours)}</div>'
                    f'<div class="gs-do">{avoid_text}</div>', unsafe_allow_html=True)
        st.write("")

        m1, m2, m3 = st.columns(3)
        m1.metric("Saved per day", eur(mwh * saving),
                  help=f"€{saving:.0f} saved for every MWh moved from an avoid hour to a run hour. "
                       f"Run hours averaged €{green.get('mean_price_eur_mwh', 0):.0f}/MWh, avoid hours "
                       f"€{red.get('mean_price_eur_mwh', 0):.0f}/MWh.")
        m2.metric("Saved per year", f"€{mwh * saving * 365 / 1e6:,.2f}m",
                  help="If you move this much every day. Real operations will not manage every day.")
        m3.metric("CO₂ avoided per year", f"{mwh * co2 * 365:,.0f} t",
                  help=f"{co2:.3f} t per MWh moved: run hours had "
                       f"{green.get('mean_renewable_share', 0):.0%} wind and solar, avoid hours "
                       f"{red.get('mean_renewable_share', 0):.0%}. Estimated from renewable output, "
                       f"not measured.")

        conf = plan.get("confidence", "unknown")
        st.caption(f"⚠ {limit_text} Your operating limits always come first. · "
                   f"Confidence in this advice: **{conf.capitalize()}**. · Other hours: run as normal.")

        with st.expander("When should I ignore this advice?"):
            for c in plan.get("override_conditions", []):
                st.markdown(f"- {c}")


# ══════════════════════════════════════════════════════════════════════════════
# REPLAY ANY DAY
# ══════════════════════════════════════════════════════════════════════════════
with tab_replay:
    st.caption("Pick a day the model was never trained on. GridShift plans it from its forecast alone, "
               "then the plan is scored at the prices that actually cleared.")

    rp = load_replay()
    if rp is None or not mae:
        st.warning("No forecast or data yet. Run `python run_pipeline.py` first.")
    else:
        gate = 1.5 * mae
        days = sorted(rp["date"].unique())
        demo = pd.Timestamp("2025-11-25").date()  # widest spread in the window: a clear first example
        k1, k2 = st.columns(2)
        day = k1.date_input("Day", value=demo if demo in days else days[len(days) // 2],
                            min_value=days[0], max_value=days[-1], format="DD/MM/YYYY")
        n = k2.slider("Hours of storage", 1, 8, 4,
                      help="How many hours you can shift. A 4-hour chilled-water tank moves 4 hours "
                           "of running out of the expensive hours.")

        g = rp[rp["date"] == day]
        res = score_day(g, n)

        if res["planned"] < gate:
            st.warning(f"**Low confidence.** The forecast sees only a €{res['planned']:.0f} gap between "
                       f"cheap and expensive hours, under its €{gate:.0f} threshold. Advice: keep your "
                       f"normal schedule. The plan is shown anyway.")
        else:
            st.success(f"**High confidence.** The forecast expects a €{res['planned']:.0f} gap between "
                       f"cheap and expensive hours.")

        st.markdown(strip([f"{t.hour:02d}" for t in g.index],
                          ["run" if t in res["run"] else "avoid" if t in res["avoid"] else "mid"
                           for t in g.index]), unsafe_allow_html=True)

        captured = res["got"] / res["best"] if res["best"] > 0 else float("nan")
        r1, r2, r3 = st.columns(3)
        r1.metric("Saved at real prices", f"{eur(res['got'])}/MWh",
                  help="Average real price in the avoid hours minus the run hours, per MWh moved.")
        r2.metric("Best possible", f"{eur(res['best'])}/MWh",
                  help="What you would have saved had you known the real prices in advance.")
        r3.metric("Share captured", f"{captured:.0%}",
                  help=f"Run hours had {res['share_gap'] * 100:+.0f} percentage points more wind "
                       f"and solar than avoid hours.")

        c = g.tz_localize(None).reset_index()  # local clock time, so the chart shows local hours
        long = c.melt(id_vars="id", value_vars=["y_pred", "price_da_eur_mwh"],
                      var_name="series", value_name="price")
        long["series"] = long["series"].map({"y_pred": "Forecast", "price_da_eur_mwh": "Actual price"})
        starts = list(res["run"].tz_localize(None)) + list(res["avoid"].tz_localize(None))
        band = pd.DataFrame({"start": starts, "band": ["Run"] * n + ["Avoid"] * n})
        band["end"] = band["start"] + pd.Timedelta(hours=1)
        rects = alt.Chart(band).mark_rect(opacity=0.18).encode(
            x="start:T", x2="end:T",
            color=alt.Color("band:N", scale=alt.Scale(domain=["Run", "Avoid"], range=["#0F8C80", "#C9531F"]),
                            legend=alt.Legend(title=None, orient="top")))
        lines = alt.Chart(long).mark_line(point=True).encode(
            x=alt.X("id:T", title=None, axis=alt.Axis(format="%H:%M")),
            y=alt.Y("price:Q", title="€ per MWh"),
            strokeDash=alt.StrokeDash("series:N", legend=alt.Legend(title=None, orient="top")),
            color=alt.value("#101821"),
            tooltip=[alt.Tooltip("id:T", title="Hour", format="%H:%M"), "series:N",
                     alt.Tooltip("price:Q", title="€/MWh", format=".0f")])
        st.altair_chart((rects + lines).properties(height=300), use_container_width=True)

        summ = replay_summary(n, gate)
        flagged = (f", {summ['loss_flagged']} of them flagged low confidence in advance"
                   if summ["loss_days"] else "")
        st.caption(f"**All {summ['days']} days, {n} h storage:** {summ['captured']:.0%} of the best possible "
                   f"saving captured · €{summ['got']:.0f}/MWh on average · lost money on "
                   f"{summ['loss_days']} day{'s' if summ['loss_days'] != 1 else ''}{flagged}.")


# ══════════════════════════════════════════════════════════════════════════════
# BEHIND THE NUMBERS
# ══════════════════════════════════════════════════════════════════════════════
with tab_more:
    if perf:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Typical forecast error", f"€{mae:.2f}/MWh",
                  help="Mean absolute error on October to December 2025, months the model was never "
                       "trained on.")
        b2.metric("Vs standard benchmark", f"{improvement:.1f}% better",
                  help=f"The benchmark assumes each hour costs what it cost a week earlier: off by "
                       f"€{perf.get('baseline_mae_eur_mwh', 0):.2f}/MWh.")
        b3.metric("80% range coverage", f"{perf.get('prediction_interval_coverage_80pct', 0):.1%}",
                  help="The forecast range should contain the real price 80% of the time. It managed "
                       "less, so the range is too narrow. Reported rather than widened.")
        b4.metric("Hours of history", f"{overview.get('n_rows', 0):,}",
                  help=f"{overview.get('start', '')[:10]} to {overview.get('end', '')[:10]}, hourly.")

    with st.expander("How accurate is the forecast?"):
        if perf:
            cv = perf.get("lgbm_cv", {})
            st.markdown(
                f"Retrained **{cv.get('n_folds', 0)} times**, each time on data from before the month it "
                f"predicted: average error €{cv.get('mae_mean', 0):.1f} ± {cv.get('mae_std', 0):.1f}/MWh. "
                f"The final test, {perf.get('test_window', {}).get('n_hours', 0):,} hours of October to "
                f"December 2025, was held back from all training.")
        if sub is not None:
            local = sub.tz_convert("Europe/Berlin")
            st.markdown("**The average forecast day**: cheap overnight, peaks in the morning and evening.")
            st.bar_chart(local.groupby(local.index.hour)["y_pred"].mean().rename("€ per MWh"),
                         color="#0F8C80", height=220)
        for fig, cap in [("fig5_model_comparison.png", "Forecast against reality"),
                         ("fig3_feature_importance.png", "What the model pays attention to")]:
            if (FIGURES / fig).exists():
                st.markdown(f"**{cap}**")
                st.image(str(FIGURES / fig), use_container_width=True)
        if sub is not None:
            st.download_button("Download the hourly forecast (CSV)",
                               data=sub.reset_index().to_csv(index=False).encode(),
                               file_name="gridshift_forecast.csv", mime="text/csv")

    with st.expander("Do cheap hours really mean cleaner hours?"):
        r = link.get("pearson_r")
        st.markdown(
            f"In the test period the cheapest third of hours had "
            f"**{link.get('mean_renewable_share_cheapest_tercile', 0):.0%}** wind and solar, the priciest "
            f"third **{link.get('mean_renewable_share_priciest_tercile', 0):.0%}** (correlation {r}). "
            f"The system re-checks this on every run and stops advising if it drops below 0.25.\n\n"
            f"Only part of that gap can be reached by scheduling: a plant can move a run from 7pm to 3am, "
            f"but not June's demand into April. Over 2022 to 2025 the within-day part is about "
            f"**0.16 t CO₂ per MWh**, nearly as large as the between-day part (0.17). Carbon is estimated "
            f"from renewable output, not measured, so cost is the headline and carbon a real co-benefit.")

    with st.expander("Is the data trustworthy?"):
        if qa:
            miss = max((v["fraction"] for v in qa.get("missing", {}).values()), default=0)
            gaps = qa.get("hourly_gaps", {}).get("count", 0) + qa.get("duplicates", {}).get("count", 0)
            st.markdown(f"Five hourly series (price, demand, onshore and offshore wind, solar) from the "
                        f"German regulator's public SMARD portal. Missing readings: **{miss:.2%}**. "
                        f"Gaps or duplicates: **{gaps}**.")
        st.markdown(
            "**Two silent bugs, found and fixed.** In an earlier version three of the five feeds were "
            "mislabelled in my own code: biomass was read as onshore wind, and solar was never downloaded. "
            "Nothing errored and every check passed. Each series is now checked against what that "
            "technology physically does, and the pipeline refuses to run if a series is missing or more "
            "than 5% empty.")
        if llm_log.get("results"):
            s = llm_log.get("summary", {})
            st.markdown(
                f"**AI-written checks.** Claude reads the data's shape and writes "
                f"{s.get('total_rules', 0)} validation rules; the pipeline runs every one: "
                f"{s.get('pass', 0)} passed, {s.get('fail', 0) + s.get('suspect', 0)} flagged real "
                f"exceptions. It does not trust them blindly: one generated rule once reported 720 "
                f"failures when the true count was 6, so any rule rejecting over half the data is now "
                f"quarantined.")
            wording = {"pass": "✓ passed", "fail": "flagged", "suspect": "quarantined",
                       "error": "failed to run", "skipped": "skipped"}
            st.dataframe(pd.DataFrame([{"Rule": x["description"], "Result": wording.get(x["status"], x["status"]),
                                        "Readings flagged": x["violations"] if x["status"] in ("fail", "suspect")
                                        else None} for x in llm_log["results"]]),
                         hide_index=True, use_container_width=True)
            if st.toggle("Show the exact prompt and the AI's reply"):
                st.code(llm_log.get("system_prompt", ""), language="text")
                st.code(llm_log.get("raw_response", ""), language="json")

    with st.expander("How it works"):
        st.markdown(
            "1. **Collect** four years of hourly prices, demand, wind and solar (public, free).\n"
            "2. **Check** the data with standard tests and the AI-written rules.\n"
            "3. **Prepare** 37 signals: time of day, day of week, recent prices, recent wind and solar. "
            "Every input is at least a day old.\n"
            "4. **Forecast** every hour of the next day with LightGBM.\n"
            "5. **Re-check the premise** that cheap hours are cleaner hours.\n"
            "6. **Recommend** run and avoid hours, with a confidence level and conditions for ignoring it.")
        st.markdown(
            "**Limitations.** Scheduling reaches only the within-day part of the carbon gap. Carbon is "
            "estimated, not measured. Built on Germany: the UAE has no hourly wholesale price, so a "
            "deployment there would forecast the utility's own dispatch cost instead. If everyone shifts "
            "into the same hours, those hours stop being cheap.")
        if (FIGURES / "fig1_price_renewables.png").exists():
            st.image(str(FIGURES / "fig1_price_renewables.png"), use_container_width=True)

    st.caption("Sneha Sunil · snehasunil385@gmail.com")
