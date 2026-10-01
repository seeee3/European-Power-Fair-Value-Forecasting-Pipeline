"""
GridShift dashboard.

Run with:  streamlit run app.py

Design note: this is read by an operations person deciding when to run a plant,
not by the analyst who built the model. So the plain-language answer comes
first and the statistics sit underneath it. Every number that has a unit or a
technical name is given a sentence explaining what it means in practice.
"""
from __future__ import annotations

import json
from pathlib import Path

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
    """
    Render clock hours as readable time ranges.

    A raw list like [0, 1, 2, 3, 4, 5, 12, 23] tells an operator nothing.
    "00:00-06:00, 12:00-13:00, 23:00-24:00" tells them when to run.
    """
    return ", ".join(f"{a:02d}:00–{b:02d}:00" for a, b in hour_runs(hours))


def eur(v: float) -> str:
    """Currency with the minus sign outside the symbol, as people write it."""
    return f"&minus;&euro;{abs(v):,.0f}" if v < 0 else f"&euro;{v:,.0f}"


def link_strength(r: float | None) -> tuple[str, str]:
    """Plain-language reading of a correlation, plus a tone for the status dot."""
    if r is None:
        return "Not measured", "grey"
    if r >= 0.7:
        return "Strong", "good"
    if r >= 0.4:
        return "Moderate", "warn"
    if r >= 0.25:
        return "Weak", "warn"
    return "Too weak to rely on", "bad"


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="GridShift", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .block-container { padding-top: 2.4rem; max-width: 1350px; }
  #MainMenu, footer { visibility: hidden; }

  html, body, [class*="css"] { font-feature-settings: "tnum" 0; }

  .gs-eyebrow {
    font-size: 11px; letter-spacing: .13em; text-transform: uppercase;
    color: #8794A1; font-weight: 600; margin-bottom: .35rem;
  }
  .gs-h {
    font-size: 27px; font-weight: 640; letter-spacing: -.01em;
    color: #101821; margin: 0 0 .25rem 0; line-height: 1.2;
  }
  .gs-sub { font-size: 15px; color: #5C6875; margin: 0 0 1.1rem 0; max-width: 78ch; line-height: 1.5; }

  .gs-card {
    border: 1px solid #DDE3E9; border-radius: 4px; padding: 18px 20px;
    background: #fff; height: 100%;
  }
  .gs-card.good { border-left: 3px solid #0F8C80; }
  .gs-card.bad  { border-left: 3px solid #C9531F; }
  .gs-card.flat { background: #F5F7F9; }

  .gs-label { font-size: 12px; letter-spacing: .06em; text-transform: uppercase;
              color: #8794A1; font-weight: 600; margin-bottom: .5rem; }
  .gs-big { font-size: 30px; font-weight: 640; letter-spacing: -.02em; color: #101821;
            font-variant-numeric: tabular-nums; line-height: 1.15; }
  .gs-big.teal { color: #0F8C80; }
  .gs-big.ember { color: #C9531F; }
  .gs-time { font-size: 23px; font-weight: 640; color: #101821;
             font-variant-numeric: tabular-nums; letter-spacing: -.01em; line-height: 1.3; }
  .gs-note { font-size: 13.5px; color: #5C6875; margin-top: .45rem; line-height: 1.5; }
  .gs-tiny { font-size: 12.5px; color: #8794A1; margin-top: .3rem; line-height: 1.45; }

  .gs-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
            margin-right: 7px; vertical-align: middle; }
  .gs-dot.good { background: #0F8C80; }
  .gs-dot.warn { background: #B8801A; }
  .gs-dot.bad  { background: #C9531F; }
  .gs-dot.grey { background: #A9B4BF; }

  .gs-row { display: flex; justify-content: space-between; gap: 1rem;
            padding: 9px 0; border-bottom: 1px solid #EDF1F4; font-size: 14.5px; }
  .gs-row:last-child { border-bottom: none; }
  .gs-row .k { color: #5C6875; }
  .gs-row .v { font-weight: 620; color: #101821; font-variant-numeric: tabular-nums; }

  .gs-rule { border-bottom: 1px solid #EDF1F4; padding: 10px 0; font-size: 14px; }
  .gs-rule:last-child { border-bottom: none; }
  .gs-rule .n { font-weight: 620; color: #101821; }
  .gs-rule .d { color: #5C6875; font-size: 13.5px; margin-top: .15rem; }

  .gs-strip { display: flex; gap: 3px; margin: .4rem 0 .5rem 0; }
  .gs-cell { flex: 1; min-width: 0; text-align: center; padding: 14px 0; border-radius: 3px;
             font-size: 12px; font-weight: 620; font-variant-numeric: tabular-nums; }
  .gs-cell.run   { background: #0F8C80; color: #fff; }
  .gs-cell.avoid { background: #C9531F; color: #fff; }
  .gs-cell.mid   { background: #EAE3CC; color: #5C6875; }
  .gs-legend { font-size: 13px; color: #5C6875; }
  .gs-legend b.run { color: #0F8C80; } .gs-legend b.avoid { color: #C9531F; }
  .gs-step { border: 1px solid #DDE3E9; border-radius: 4px; padding: 12px 16px; background: #F5F7F9;
             font-size: 14px; color: #33404C; line-height: 1.45; height: 100%; }
  .gs-step .num { font-weight: 700; color: #0F8C80; margin-right: 6px; }
  .gs-plan { font-size: 15.5px; line-height: 1.6; color: #101821; }
  .gs-plan .when { font-weight: 660; font-variant-numeric: tabular-nums; }

  section[data-testid="stSidebar"] { background: #F5F7F9; border-right: 1px solid #DDE3E9; }
  section[data-testid="stSidebar"] .gs-side-h { font-size: 19px; font-weight: 660; color: #101821; }
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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="gs-side-h">GridShift</div>', unsafe_allow_html=True)
    st.markdown('<div class="gs-tiny">Scheduling flexible electricity demand<br>'
                'German market (DE-LU)</div>', unsafe_allow_html=True)
    st.divider()

    if green.get("local_hours"):
        st.markdown('<div class="gs-label">Run during</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="gs-time" style="color:#0F8C80">{clock(green["local_hours"])}</div>',
                    unsafe_allow_html=True)
    if red.get("local_hours"):
        st.markdown('<div class="gs-label" style="margin-top:1rem">Avoid</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="gs-time" style="color:#C9531F">{clock(red["local_hours"])}</div>',
                    unsafe_allow_html=True)

    st.divider()
    conf = plan.get("confidence", "unknown")
    tone = {"high": "good", "medium": "warn", "low": "bad"}.get(conf, "grey")
    st.markdown('<div class="gs-label">Confidence in this advice</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:15px;font-weight:620">'
                f'<span class="gs-dot {tone}"></span>{conf.capitalize()}</div>', unsafe_allow_html=True)
    if mae:
        st.markdown(f'<div class="gs-tiny">Forecasts land within about &euro;{mae:.0f} per MWh '
                    f'of the real price.</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="gs-tiny">Sneha Sunil<br>snehasunil385@gmail.com</div>',
                unsafe_allow_html=True)


# ── Intro: what this is and how to use it, before anything else ───────────────
st.markdown('<div class="gs-eyebrow">GridShift · working prototype</div>', unsafe_allow_html=True)
st.markdown('<div class="gs-h">Know which hours to run your equipment in</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="gs-sub">Large electricity users such as cooling plants, desalination plants, EV fleets '
    'and data centres can often run their equipment at a different time of day. Electricity costs about '
    'twice as much in the evening as it does at night. GridShift forecasts tomorrow\'s hourly price and '
    'tells you which hours to use. Shown here on German market data for October to December 2025.</div>',
    unsafe_allow_html=True)
s1, s2, s3 = st.columns(3)
for col, n, text in [
    (s1, 1, "See the <b>green</b> hours to run in and the <b>red</b> hours to avoid."),
    (s2, 2, "Tell it <b>what you run</b> and how much electricity use you can move."),
    (s3, 3, "Get <b>your plan</b> and an estimate of what you would save."),
]:
    col.markdown(f'<div class="gs-step"><span class="num">{n}</span>{text}</div>', unsafe_allow_html=True)
st.write("")

tab_rec, tab_forecast, tab_acc, tab_data, tab_method = st.tabs(
    ["Your schedule", "The forecast", "Can I trust it?", "Data checks", "How it works"]
)


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_rec:
    if not plan:
        st.warning("No recommendation yet. Run `python run_pipeline.py` first.")
    else:
        green_hours = green.get("local_hours", [])
        red_hours = red.get("local_hours", [])

        st.markdown('<div class="gs-eyebrow">Step 1 · The hours</div>', unsafe_allow_html=True)
        st.markdown('<div class="gs-h" style="font-size:22px">Your day at a glance</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="gs-sub">Each box is one hour of the day, in local clock time. Use electricity '
            'in the green hours and cut back in the red ones. The pattern held on most days in the '
            'period analysed.</div>', unsafe_allow_html=True)
        cells = "".join(
            f'<div class="gs-cell {"run" if h in green_hours else "avoid" if h in red_hours else "mid"}">'
            f'{h:02d}</div>' for h in range(24))
        st.markdown(f'<div class="gs-strip">{cells}</div>'
                    '<div class="gs-legend"><b class="run">Green: run</b> &nbsp;·&nbsp; '
                    'Beige: normal &nbsp;·&nbsp; <b class="avoid">Red: avoid</b></div>',
                    unsafe_allow_html=True)
        st.write("")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="gs-card good">'
                f'<div class="gs-label">Run during these hours</div>'
                f'<div class="gs-time" style="color:#0F8C80">{clock(green.get("local_hours", []))}</div>'
                f'<div class="gs-note">Electricity here costs '
                f'<b>&euro;{green.get("mean_price_eur_mwh", 0):.0f} per MWh</b> on average, and '
                f'<b>{green.get("mean_renewable_share", 0):.0%}</b> of it comes from wind and solar.</div>'
                f'</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(
                f'<div class="gs-card bad">'
                f'<div class="gs-label">Avoid these hours</div>'
                f'<div class="gs-time" style="color:#C9531F">{clock(red.get("local_hours", []))}</div>'
                f'<div class="gs-note">Electricity here costs '
                f'<b>&euro;{red.get("mean_price_eur_mwh", 0):.0f} per MWh</b> on average, and only '
                f'<b>{red.get("mean_renewable_share", 0):.0%}</b> comes from wind and solar.</div>'
                f'</div>', unsafe_allow_html=True)

        # ── Steps 2 and 3: plan my site ─────────────────────────────────────────
        saving = impact.get("cost_saving_eur", 0)
        co2 = impact.get("co2_avoided_t", 0)
        sites = {
            "District cooling": (
                "Make chilled water and store it in the tank.",
                "Draw on the stored chilled water instead of running the chillers.",
                "You can only move as much cooling as your tank holds.", 50),
            "Desalination": (
                "Run the reverse-osmosis trains at full output and fill the reservoir.",
                "Turn the trains down towards minimum and supply from the reservoir.",
                "Never go below the plant's minimum safe output.", 40),
            "EV fleet": (
                "Charge the vehicles.",
                "Pause charging. Vehicles stay plugged in.",
                "Every vehicle must still be charged by its departure time.", 20),
            "Data centre": (
                "Run batch jobs, model training and backups.",
                "Hold back deferrable jobs. Live services run as normal.",
                "Only work that can wait should move. Customer-facing services never wait.", 30),
        }

        st.write("")
        st.markdown('<div class="gs-eyebrow">Steps 2 and 3 · Plan my site</div>', unsafe_allow_html=True)
        st.markdown('<div class="gs-h" style="font-size:22px">What should my site do?</div>',
                    unsafe_allow_html=True)
        p1, p2 = st.columns([1, 1.25], gap="large")
        with p1:
            site = st.radio("What do you run?", list(sites), horizontal=True)
            run_text, avoid_text, limit_text, default_mwh = sites[site]
            mwh = st.slider(
                "How much electricity use can you move each day? (MWh)", 5, 300, default_mwh, 5,
                key=f"mwh_{site}",
                help="A megawatt-hour (MWh) is 1,000 kWh. Count only what you can really move, "
                     "given your storage, minimum output or deadlines.")
            st.markdown(
                f'<div class="gs-plan" style="margin-top:.6rem">'
                f'<div><span class="when" style="color:#0F8C80">{clock(green_hours)}</span><br>{run_text}</div>'
                f'<div style="margin-top:.6rem"><span class="when" style="color:#C9531F">{clock(red_hours)}</span>'
                f'<br>{avoid_text}</div>'
                f'<div style="margin-top:.6rem"><span class="when">All other hours</span><br>Run as normal.</div>'
                f'</div><div class="gs-tiny" style="margin-top:.7rem">&#9888; {limit_text} '
                f'Your operating limits always come first.</div>', unsafe_allow_html=True)
        with p2:
            q1, q2 = st.columns(2)
            with q1:
                st.markdown(
                    f'<div class="gs-card good"><div class="gs-label">Saved per day</div>'
                    f'<div class="gs-big teal">&euro;{mwh * saving:,.0f}</div>'
                    f'<div class="gs-note">&euro;{saving:.0f} for every MWh moved from a red hour '
                    f'to a green one.</div></div>', unsafe_allow_html=True)
            with q2:
                st.markdown(
                    f'<div class="gs-card good"><div class="gs-label">Saved per year</div>'
                    f'<div class="gs-big teal">&euro;{mwh * saving * 365 / 1e6:,.2f}m</div>'
                    f'<div class="gs-note">if you move this much every day.</div></div>',
                    unsafe_allow_html=True)
            st.write("")
            st.markdown(
                f'<div class="gs-card"><div class="gs-label">Carbon avoided per year</div>'
                f'<div class="gs-big">{mwh * co2 * 365:,.0f} t CO&#8322;</div>'
                f'<div class="gs-note">{co2:.3f} t per MWh moved in this period, because the green hours '
                f'had more wind and solar. Estimated from renewable output, not measured.</div></div>',
                unsafe_allow_html=True)
            st.markdown('<div class="gs-tiny">Estimates use October to December 2025 prices. Real savings '
                        'depend on how much you can actually move each day.</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="gs-eyebrow">Is the reasoning sound?</div>', unsafe_allow_html=True)
        st.markdown('<div class="gs-h" style="font-size:20px">Do cheap hours really mean cleaner hours?</div>',
                    unsafe_allow_html=True)

        r = link.get("pearson_r")
        label, tone = link_strength(r)
        st.markdown(
            f'<div class="gs-sub">The whole idea rests on cheap electricity also being cleaner '
            f'electricity, because wind and solar push expensive gas and coal plants off the grid. '
            f'That is an assumption, so the system re-checks it every time it runs and lowers its own '
            f'confidence if it stops holding.</div>', unsafe_allow_html=True)

        e1, e2, e3 = st.columns([1.1, 1, 1])
        with e1:
            st.markdown(
                f'<div class="gs-card flat"><div class="gs-label">Strength of the link</div>'
                f'<div style="font-size:22px;font-weight:640;margin-bottom:.3rem">'
                f'<span class="gs-dot {tone}"></span>{label}</div>'
                f'<div class="gs-tiny">Statistically, a correlation of {r} across '
                f'{link.get("n_hours", 0):,} hours. Above 0.7 would be strong; below 0.25 the system '
                f'stops giving advice.</div></div>', unsafe_allow_html=True)
        with e2:
            st.markdown(
                f'<div class="gs-card flat"><div class="gs-label">In the cheapest hours</div>'
                f'<div class="gs-big teal">{link.get("mean_renewable_share_cheapest_tercile", 0):.0%}</div>'
                f'<div class="gs-tiny">of electricity came from wind and solar.</div></div>',
                unsafe_allow_html=True)
        with e3:
            st.markdown(
                f'<div class="gs-card flat"><div class="gs-label">In the priciest hours</div>'
                f'<div class="gs-big ember">{link.get("mean_renewable_share_priciest_tercile", 0):.0%}</div>'
                f'<div class="gs-tiny">of electricity came from wind and solar.</div></div>',
                unsafe_allow_html=True)

        st.write("")
        st.info(
            "**How much of the carbon gap can scheduling reach?** Cheap hours are cleaner for two "
            "reasons: some *days* are windier than others, and within each day some *hours* have more "
            "wind and sun. A plant can move a run from 7pm to 3am, but it cannot move June's demand into "
            "April, so only the within-day part counts. Over 2022 to 2025 that part is about 0.16 t CO₂ "
            "per MWh, nearly as large as the between-day part (0.17). Cost leads, because it comes "
            "straight from a market price; carbon is a real co-benefit, estimated rather than measured."
        )

        st.write("")
        st.markdown('<div class="gs-eyebrow">When to ignore this advice</div>', unsafe_allow_html=True)
        st.markdown('<div class="gs-sub">Any one of these means the schedule should not be followed '
                    'automatically.</div>', unsafe_allow_html=True)
        for c in plan.get("override_conditions", []):
            st.markdown(f'<div class="gs-rule"><span class="gs-dot bad"></span>{c}</div>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PRICE FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    st.markdown('<div class="gs-eyebrow">Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="gs-h">Predicted electricity prices, October to December 2025</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="gs-sub">These three months were held back completely. The model never saw '
                'them while it was being built or tuned, so this is a fair test of what it would have '
                'done in real life.</div>', unsafe_allow_html=True)

    if sub is None:
        st.warning("No forecast yet. Run `python run_pipeline.py` first.")
    else:
        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown(f'<div class="gs-card"><div class="gs-label">Hours forecast</div>'
                        f'<div class="gs-big">{len(sub):,}</div>'
                        f'<div class="gs-note">every hour of the three months.</div></div>',
                        unsafe_allow_html=True)
        with f2:
            st.markdown(f'<div class="gs-card"><div class="gs-label">Average price predicted</div>'
                        f'<div class="gs-big">&euro;{sub["y_pred"].mean():.0f}</div>'
                        f'<div class="gs-note">per MWh across the window.</div></div>',
                        unsafe_allow_html=True)
        with f3:
            st.markdown(f'<div class="gs-card"><div class="gs-label">Cheapest to priciest hour</div>'
                        f'<div class="gs-big">{eur(sub["y_pred"].min())} to {eur(sub["y_pred"].max())}</div>'
                        f'<div class="gs-note">the spread the schedule exploits.</div></div>',
                        unsafe_allow_html=True)

        st.write("")
        local = sub.tz_convert("Europe/Berlin")
        by_hour = local.groupby(local.index.hour)["y_pred"].mean()

        st.markdown('<div class="gs-h" style="font-size:19px">The average day</div>', unsafe_allow_html=True)
        st.markdown('<div class="gs-sub">Price by hour of day, averaged across the window. The dip in the '
                    'middle of the night and the spike in the early evening are what the schedule trades '
                    'between.</div>', unsafe_allow_html=True)
        st.bar_chart(by_hour.rename("Euro per MWh"), color="#0F8C80", height=260)

        st.markdown('<div class="gs-h" style="font-size:19px">Day by day</div>', unsafe_allow_html=True)
        st.markdown('<div class="gs-sub">Daily average of the predicted price. The shaded spread shows the '
                    'range the forecast expects to land inside most of the time.</div>',
                    unsafe_allow_html=True)
        daily = sub.resample("D").mean()
        cols = [c for c in ("y_pred_p10", "y_pred", "y_pred_p90") if c in daily.columns]
        renamed = daily[cols].rename(columns={
            "y_pred": "Predicted price", "y_pred_p10": "Lower end", "y_pred_p90": "Upper end"})
        st.line_chart(renamed, height=300,
                      color=["#C6D3DB", "#0F8C80", "#C6D3DB"][:len(cols)])

        st.download_button(
            "Download the hourly forecast (CSV)",
            data=sub.reset_index().to_csv(index=False).encode(),
            file_name="gridshift_forecast.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
with tab_acc:
    st.markdown('<div class="gs-eyebrow">Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="gs-h">How much should you trust the forecast?</div>', unsafe_allow_html=True)
    st.markdown('<div class="gs-sub">Everything here is measured on the three months the model never saw '
                'while it was being built.</div>', unsafe_allow_html=True)

    if not perf:
        st.warning("No results yet. Run `python run_pipeline.py` first.")
    else:
        base = perf.get("baseline_mae_eur_mwh")
        cov = perf.get("prediction_interval_coverage_80pct", 0)

        a1, a2, a3 = st.columns(3)
        with a1:
            st.markdown(
                f'<div class="gs-card good"><div class="gs-label">Typical error</div>'
                f'<div class="gs-big teal">&euro;{mae:.2f}</div>'
                f'<div class="gs-note">On an average hour the forecast is off by about this much per MWh. '
                f'Predicted prices in the window ran from {eur(sub["y_pred"].min())} to '
                f'{eur(sub["y_pred"].max())}, so the error is small next to the swing the schedule '
                f'is exploiting.</div></div>' if sub is not None else
                f'<div class="gs-card good"><div class="gs-label">Typical error</div>'
                f'<div class="gs-big teal">&euro;{mae:.2f}</div></div>', unsafe_allow_html=True)
        with a2:
            st.markdown(
                f'<div class="gs-card"><div class="gs-label">Against the industry benchmark</div>'
                f'<div class="gs-big teal">{improvement:.0f}% better</div>'
                f'<div class="gs-note">The standard benchmark is to assume this hour will cost what the '
                f'same hour cost last week. That is off by &euro;{base:.0f}; this model is off by '
                f'&euro;{mae:.0f}.</div></div>', unsafe_allow_html=True)
        with a3:
            st.markdown(
                f'<div class="gs-card"><div class="gs-label">Honesty of the range</div>'
                f'<div class="gs-big">{cov:.0%}</div>'
                f'<div class="gs-note">The forecast gives a range it expects the real price to fall inside '
                f'8 times out of 10. It actually managed {cov:.0%}, so the range is slightly too narrow. '
                f'Reported rather than quietly widened.</div></div>', unsafe_allow_html=True)

        st.write("")
        cv = perf.get("lgbm_cv", {})
        tw = perf.get("test_window", {})
        b1, b2 = st.columns(2)
        with b1:
            st.markdown('<div class="gs-h" style="font-size:19px">Tested the honest way</div>',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="gs-sub">The model was retrained {cv.get("n_folds", 0)} times, each time '
                f'using only data from before the month it was asked to predict. It can never see the '
                f'future, which is the easiest way to accidentally flatter a forecasting model.</div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div class="gs-card flat">'
                f'<div class="gs-row"><span class="k">Retrained and tested</span>'
                f'<span class="v">{cv.get("n_folds", 0)} times</span></div>'
                f'<div class="gs-row"><span class="k">Average error across those tests</span>'
                f'<span class="v">&euro;{cv.get("mae_mean", 0):.1f} per MWh</span></div>'
                f'<div class="gs-row"><span class="k">How much that varied</span>'
                f'<span class="v">&plusmn; &euro;{cv.get("mae_std", 0):.1f}</span></div>'
                f'<div class="gs-row"><span class="k">Final held-back test</span>'
                f'<span class="v">{tw.get("start", "")} to {tw.get("end", "")}</span></div>'
                f'<div class="gs-row"><span class="k">Hours in that test</span>'
                f'<span class="v">{tw.get("n_hours", 0):,}</span></div>'
                f'</div>', unsafe_allow_html=True)
        with b2:
            fig_cmp = FIGURES / "fig5_model_comparison.png"
            if fig_cmp.exists():
                st.markdown('<div class="gs-h" style="font-size:19px">Forecast against reality</div>',
                            unsafe_allow_html=True)
                st.markdown('<div class="gs-sub">The model tracks the real price more closely than the '
                            'benchmark does, particularly through the daily peaks.</div>',
                            unsafe_allow_html=True)
                st.image(str(fig_cmp), use_container_width=True)

        fig_fi = FIGURES / "fig3_feature_importance.png"
        if fig_fi.exists():
            st.markdown('<div class="gs-h" style="font-size:19px">What the model actually pays attention to</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="gs-sub">Recent prices matter most, then the time of day, then how much '
                        'wind and solar were on the grid. That ordering is what a trader would expect, which '
                        'is a good sign the model learned the real mechanism.</div>', unsafe_allow_html=True)
            st.image(str(fig_fi), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown('<div class="gs-eyebrow">Inputs</div>', unsafe_allow_html=True)
    st.markdown('<div class="gs-h">Is the underlying data trustworthy?</div>', unsafe_allow_html=True)

    if not qa:
        st.warning("No data report yet. Run `python run_pipeline.py` first.")
    else:
        miss = qa.get("missing", {})
        worst = max((v["fraction"] for v in miss.values()), default=0)
        g1, g2, g3 = st.columns(3)
        with g1:
            st.markdown(f'<div class="gs-card"><div class="gs-label">Hours of history</div>'
                        f'<div class="gs-big">{overview.get("n_rows", 0):,}</div>'
                        f'<div class="gs-note">four years, {overview.get("start", "")[:10]} '
                        f'to {overview.get("end", "")[:10]}.</div></div>', unsafe_allow_html=True)
        with g2:
            st.markdown(f'<div class="gs-card"><div class="gs-label">Missing readings</div>'
                        f'<div class="gs-big teal">{worst:.2%}</div>'
                        f'<div class="gs-note">at worst, across any single measurement.</div></div>',
                        unsafe_allow_html=True)
        with g3:
            gaps = qa.get("hourly_gaps", {}).get("count", 0)
            dupes = qa.get("duplicates", {}).get("count", 0)
            st.markdown(f'<div class="gs-card"><div class="gs-label">Gaps and duplicates</div>'
                        f'<div class="gs-big teal">{gaps + dupes}</div>'
                        f'<div class="gs-note">no missing hours and no repeated timestamps.</div></div>',
                        unsafe_allow_html=True)

        st.write("")
        st.info(
            "**Two silent bugs, found and fixed.** In an earlier version, three of the five data feeds "
            "were mislabelled in my own code: biomass was read in as onshore wind, and solar was never "
            "downloaded at all. Nothing errored and every check passed. Each series is now verified "
            "against what that technology physically does (solar is zero at night, wind has no daily "
            "cycle), and the pipeline refuses to run if any series is missing or more than 5% empty."
        )

    st.write("")
    st.markdown('<div class="gs-eyebrow">Automated checking</div>', unsafe_allow_html=True)
    st.markdown('<div class="gs-h" style="font-size:20px">An AI writes the quality checks, '
                'and the pipeline checks the AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gs-sub">Claude reads the shape of the data and writes rules for what a believable '
        'reading looks like, such as solar output never being negative or demand staying inside plausible '
        'bounds. The pipeline then runs every rule and records the result. Writing these by hand takes an '
        'analyst most of a day.</div>', unsafe_allow_html=True)

    if not llm_log or not llm_log.get("results"):
        st.warning("No checks recorded yet. Run the pipeline with an ANTHROPIC_API_KEY set.")
    else:
        s = llm_log.get("summary", {})
        h1, h2, h3 = st.columns(3)
        with h1:
            st.markdown(f'<div class="gs-card"><div class="gs-label">Rules written</div>'
                        f'<div class="gs-big">{s.get("total_rules", 0)}</div>'
                        f'<div class="gs-note">generated automatically, then run against every row.</div>'
                        f'</div>', unsafe_allow_html=True)
        with h2:
            st.markdown(f'<div class="gs-card"><div class="gs-label">Data passed</div>'
                        f'<div class="gs-big teal">{s.get("pass", 0)}</div>'
                        f'<div class="gs-note">rules the data satisfied completely.</div></div>',
                        unsafe_allow_html=True)
        with h3:
            st.markdown(f'<div class="gs-card"><div class="gs-label">Flagged for a look</div>'
                        f'<div class="gs-big">{s.get("fail", 0) + s.get("suspect", 0)}</div>'
                        f'<div class="gs-note">real exceptions worth checking, not errors in the data.</div>'
                        f'</div>', unsafe_allow_html=True)

        st.write("")
        st.error(
            "**Why the pipeline does not simply trust the AI.** Claude first wrote rules containing a "
            "subtle Python mistake, where `a <= b | c` is silently read as `a <= (b | c)`. Perfectly good "
            "data was reported as 720 failures when the real number was 6. The prompt now warns about that "
            "trap, and more importantly any rule that rejects over half the dataset is quarantined as a "
            "broken rule rather than believed as a finding. The second fix is the one that matters, because "
            "it does not depend on the AI behaving."
        )

        wording = {"pass": ("Data passed", "good"), "fail": ("Exceptions found", "warn"),
                   "suspect": ("Rule quarantined", "bad"), "error": ("Rule failed to run", "bad"),
                   "skipped": ("Not applicable", "grey")}
        st.markdown('<div class="gs-label" style="margin-top:.6rem">Every rule and what it found</div>',
                    unsafe_allow_html=True)
        for r in llm_log["results"]:
            text, tone = wording.get(r["status"], (r["status"], "grey"))
            found = (f' &middot; {r["violations"]:,} of the readings' if r["status"] in ("fail", "suspect")
                     else "")
            st.markdown(
                f'<div class="gs-rule"><span class="gs-dot {tone}"></span>'
                f'<span class="n">{r["description"]}</span>'
                f'<div class="d" style="margin-left:15px">{text}{found}</div></div>',
                unsafe_allow_html=True)

        with st.expander("See the exact instructions given to the AI"):
            st.code(llm_log.get("system_prompt", ""), language="text")
        with st.expander("See what the AI wrote back"):
            st.code(llm_log.get("raw_response", ""), language="json")


# ══════════════════════════════════════════════════════════════════════════════
# HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
with tab_method:
    st.markdown('<div class="gs-eyebrow">Method</div>', unsafe_allow_html=True)
    st.markdown('<div class="gs-h">From raw market data to a running schedule</div>', unsafe_allow_html=True)

    steps = [
        ("Collect", "Four years of hourly electricity prices, demand, wind and solar output from the "
                    "German grid operator's public portal. No licence or fee."),
        ("Check", "Standard checks for gaps and impossible values, plus the AI-written rules described "
                  "under Data quality."),
        ("Prepare", "Build the 37 signals the model learns from: time of day, day of week, recent prices, "
                    "and recent wind and solar. Everything is deliberately delayed by at least a day so "
                    "the model can never peek at information it would not have had."),
        ("Forecast", "Predict every hour of the next day, and check the result against both a simple "
                     "benchmark and repeated retraining on earlier periods only."),
        ("Check the premise", "Re-measure whether cheap hours really are cleaner hours. If that link "
                              "breaks down, say so instead of giving advice that cannot be justified."),
        ("Recommend", "Rank each day's hours from cheapest to priciest, and turn that into run and avoid "
                      "windows with a confidence level and clear conditions for ignoring it."),
    ]
    for i, (name, detail) in enumerate(steps, 1):
        st.markdown(
            f'<div class="gs-rule"><span class="n">{i}. {name}</span>'
            f'<div class="d">{detail}</div></div>', unsafe_allow_html=True)

    st.write("")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown('<div class="gs-h" style="font-size:19px">Where the data comes from</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="gs-card flat">'
            '<div class="gs-row"><span class="k">Electricity price, hour by hour</span>'
            '<span class="v">Complete</span></div>'
            '<div class="gs-row"><span class="k">Electricity demand</span><span class="v">Complete</span></div>'
            '<div class="gs-row"><span class="k">Onshore wind output</span><span class="v">Complete</span></div>'
            '<div class="gs-row"><span class="k">Offshore wind output</span><span class="v">Complete</span></div>'
            '<div class="gs-row"><span class="k">Solar output</span><span class="v">Complete</span></div>'
            '</div>', unsafe_allow_html=True)
        st.markdown('<div class="gs-tiny">Published by the German federal grid regulator. An alternative '
                    'European source is also supported.</div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="gs-h" style="font-size:19px">Honest limitations</div>',
                    unsafe_allow_html=True)
        for lim in [
            "Scheduling can only reach the within-day part of the clean-versus-dirty gap, not the "
            "difference between windy and calm days.",
            "Carbon intensity is estimated from how much wind and solar were running, not measured at "
            "the power station.",
            "Built and tested on Germany. The UAE has no hourly wholesale price to forecast, so a "
            "deployment there would forecast the utility's own dispatch cost instead.",
            "If everyone shifts into the same cheap hours, those hours stop being cheap.",
        ]:
            st.markdown(f'<div class="gs-rule"><span class="gs-dot warn"></span>'
                        f'<span style="font-size:13.5px;color:#5C6875">{lim}</span></div>',
                        unsafe_allow_html=True)

    fig1 = FIGURES / "fig1_price_renewables.png"
    if fig1.exists():
        st.write("")
        st.markdown('<div class="gs-h" style="font-size:19px">Four years of the market</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="gs-sub">Price alongside wind and solar output. The relationship the whole '
                    'idea rests on is visible here: when renewable output climbs, price tends to fall.</div>',
                    unsafe_allow_html=True)
        st.image(str(fig1), use_container_width=True)
