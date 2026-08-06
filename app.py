"""
Streamlit dashboard for GridShift — carbon-aware load shifting.
Run with:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GridShift Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data ─────────────────────────────────────────────────────────────────
qa = _load_json(OUTPUTS / "qa_report.json")
llm_log = _load_json(OUTPUTS / "llm_qa_log.json")
signal = _load_json(OUTPUTS / "load_shift_plan.json")
delivery = _load_csv(OUTPUTS / "delivery_views.csv")
submission = _load_csv(OUTPUTS / "submission.csv")

perf = qa.get("model_performance", {})
overview = qa.get("overview", {})

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ GridShift")
    st.caption("Scheduling flexible demand · DE-LU bidding zone")
    st.divider()

    if overview:
        st.markdown("**Dataset**")
        st.write(f"📅 {overview.get('start', '')[:10]} → {overview.get('end', '')[:10]}")
        st.write(f"📊 {overview.get('n_rows', 0):,} hourly rows")
        st.write(f"🗂 {len(overview.get('columns', []))} series loaded")

    st.divider()

    if perf:
        st.markdown("**Model (LightGBM)**")
        st.metric("Test MAE", f"{perf.get('lgbm_test_mae_eur_mwh', '—')} €/MWh")
        st.metric("vs Baseline", f"{perf.get('lgbm_improvement_over_baseline_pct', '—')}% better")

    st.divider()
    conf = signal.get("confidence", "")
    color = {"high": "🟢", "medium": "🟡", "low": "⚪"}.get(conf, "⚪")
    st.markdown(f"**Confidence: {color} {conf.upper()}**")
    _g = signal.get("green_window", {})
    if _g.get("local_hours"):
        st.write(f"Shift into **{_g['local_hours'][0]:02d}:00–{_g['local_hours'][-1]:02d}:00**")
    _lnk = signal.get("price_carbon_link", {})
    if _lnk.get("pearson_r") is not None:
        st.write(f"Price/carbon r: **{_lnk['pearson_r']}**")
    st.divider()
    st.caption("Sneha Sunil · snehasunil385@gmail.com")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_qa, tab_model, tab_signal, tab_forecast, tab_figures = st.tabs([
    "📋 Overview",
    "🔍 Data Quality",
    "📈 Model",
    "🟢 Load Shift",
    "🔮 Forecast",
    "🖼 Figures",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.header("GridShift — Scheduling Flexible Electricity Demand")
    st.caption("Forecasting tomorrow's hourly prices, so flexible demand runs in the cheap, cleaner hours")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Market", "DE-LU (EPEX Spot)")
    with col2:
        st.metric("Data rows", f"{overview.get('n_rows', 0):,}")
    with col3:
        st.metric("LightGBM CV MAE",
                  f"{perf.get('lgbm_cv', {}).get('mae_mean', '—')} €/MWh",
                  f"±{perf.get('lgbm_cv', {}).get('mae_std', '')} std")
    with col4:
        improvement = perf.get("lgbm_improvement_over_baseline_pct")
        st.metric("vs Seasonal Naive",
                  f"+{improvement}%" if improvement else "—",
                  delta_color="normal")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Pipeline steps")
        st.markdown("""
        1. **Data ingestion** — SMARD API (no auth) for DE-LU market
        2. **Standard QA** — missingness, gaps, domain-bound checks
        3. **LLM QA** — Claude proposes validation rules; pipeline executes and audits them
        4. **Feature engineering** — 37 features, strict 24h leakage control
        5. **Forecasting** — Seasonal Naive baseline + LightGBM (12-fold walk-forward CV)
        6. **Prediction intervals** — empirical P10/P90 from CV residuals, calibrated by hour
        7. **Load shift** — validate price/carbon link, then band each day green/amber/red
        """)
    with col_b:
        st.subheader("Data sources")
        src_df = pd.DataFrame([
            {"Series": "Day-Ahead price", "Source": "SMARD", "Filter": "4169", "Coverage": "100%"},
            {"Series": "Load (consumption)", "Source": "SMARD", "Filter": "4381", "Coverage": "100%"},
            {"Series": "Wind onshore", "Source": "SMARD", "Filter": "4066", "Coverage": "100%"},
            {"Series": "Wind offshore", "Source": "SMARD", "Filter": "4065", "Coverage": "100%"},
            {"Series": "Solar (PV)", "Source": "SMARD", "Filter": "4067", "Coverage": "100%"},
        ])
        st.dataframe(src_df, hide_index=True, use_container_width=True)
        st.caption("⚠️ SMARD is not reproducible run-to-run: repeated fetches of the same range returned "
                   "different coverage (one lost load, one lost wind onshore, one 404'd wind offshore). "
                   "This dataset is pinned from the most complete version of each series — see REPORT.md §1.1.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════
with tab_qa:
    st.header("Data Quality Report")

    if not qa:
        st.warning("No qa_report.json found. Run the pipeline first.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total rows", f"{overview.get('n_rows', 0):,}")
        col2.metric("Hourly gaps", qa.get("hourly_gaps", {}).get("count", 0))
        col3.metric("Duplicate timestamps", qa.get("duplicates", {}).get("count", 0))

        st.subheader("Completeness & outliers by column")
        missing = qa.get("missing", {})
        outliers = qa.get("outliers", {})
        cols_in_report = list(missing.keys())
        rows = []
        for col in cols_in_report:
            o = outliers.get(col, {})
            rows.append({
                "Column": col,
                "Missing %": f"{missing[col]['fraction']:.2%}",
                "Min": o.get("min", "—"),
                "Max": o.get("max", "—"),
                "P01": o.get("p01", "—"),
                "P99": o.get("p99", "—"),
                "Below bound": o.get("below_count", "—"),
                "Above bound": o.get("above_count", "—"),
            })
        qa_df = pd.DataFrame(rows)
        st.dataframe(qa_df, hide_index=True, use_container_width=True)

        if qa.get("issues"):
            st.warning("**Issues flagged:**")
            for issue in qa["issues"]:
                st.write(f"⚠  {issue}")
        else:
            st.success("All standard QA checks passed.")

        if qa.get("dropped_fundamentals"):
            st.subheader("Dropped fundamentals")
            for col, info in qa["dropped_fundamentals"].items():
                st.write(f"🗑 **{col}** — {info['missing_fraction']:.0%} missing — {info['reason']}")

    st.divider()
    st.subheader("AI-Accelerated QA (LLM Rule Generation)")
    st.caption("Claude proposed domain-aware validation rules; the pipeline executed and audited them. "
               "Rules flagging >50% of rows are quarantined as malformed rather than reported as data failures.")

    if not llm_log:
        st.warning("No llm_qa_log.json found. Run pipeline with a valid ANTHROPIC_API_KEY.")
    elif llm_log.get("parse_error") or not llm_log.get("results"):
        err = llm_log.get("parse_error", "Unknown error")
        st.error(f"LLM QA failed: {err}")
        st.caption("Set a valid ANTHROPIC_API_KEY in .env and re-run the pipeline.")

        with st.expander("View system prompt sent to model"):
            st.code(llm_log.get("system_prompt", ""), language="text")
    else:
        summ = llm_log.get("summary", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rules generated", summ.get("total_rules", 0))
        c2.metric("✅ Pass", summ.get("pass", 0))
        c3.metric("❌ Fail", summ.get("fail", 0))
        c4.metric("⚠ Error", summ.get("error", 0))

        rule_rows = []
        for r in llm_log.get("results", []):
            rule_rows.append({
                "Rule": r["name"],
                "Column": r["column"],
                "Status": r["status"].upper(),
                "Violations": r.get("violations", "—"),
                "Description": r["description"],
            })
        if rule_rows:
            rules_df = pd.DataFrame(rule_rows)
            st.dataframe(rules_df, hide_index=True, use_container_width=True)

        with st.expander("View system prompt"):
            st.code(llm_log.get("system_prompt", ""), language="text")
        with st.expander("View raw LLM response"):
            st.code(llm_log.get("raw_response", ""), language="json")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.header("Forecasting & Validation")

    if not perf:
        st.warning("No model_performance data found. Run the pipeline first.")
    else:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Baseline MAE", f"{perf['baseline_mae_eur_mwh']} €/MWh", help="Seasonal naive (same hour last week)")
        col2.metric("LightGBM MAE", f"{perf['lgbm_test_mae_eur_mwh']} €/MWh")
        col3.metric("LightGBM RMSE", f"{perf.get('lgbm_test_rmse_eur_mwh', '—')} €/MWh")
        col4.metric("Improvement", f"+{perf.get('lgbm_improvement_over_baseline_pct', '—')}%")
        col5.metric("80% PI coverage", f"{perf.get('prediction_interval_coverage_80pct', 0):.1%}",
                    help="Empirical coverage of P10–P90 prediction interval on test set")

        st.divider()
        cv = perf.get("lgbm_cv", {})
        tw = perf.get("test_window", {})
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Walk-forward CV summary")
            st.caption(f"{cv.get('n_folds', 0)} monthly folds (Oct 2024 – Sep 2025), expanding window")
            cv_df = pd.DataFrame([
                {"Metric": "MAE (EUR/MWh)", "Mean": cv.get("mae_mean"), "Std": cv.get("mae_std")},
                {"Metric": "RMSE (EUR/MWh)", "Mean": cv.get("rmse_mean"), "Std": cv.get("rmse_std")},
                {"Metric": "Pinball @ P10", "Mean": cv.get("pinball_10_mean"), "Std": "—"},
                {"Metric": "Pinball @ P90", "Mean": cv.get("pinball_90_mean"), "Std": "—"},
            ])
            st.dataframe(cv_df, hide_index=True, use_container_width=True)
        with col_b:
            st.subheader("Hold-out test set")
            st.caption(f"{tw.get('start')} → {tw.get('end')}  ({tw.get('n_hours', 0):,} hours)")
            st.markdown(f"""
            | | |
            |---|---|
            | Baseline MAE | **{perf['baseline_mae_eur_mwh']} €/MWh** |
            | LightGBM MAE | **{perf['lgbm_test_mae_eur_mwh']} €/MWh** |
            | LightGBM RMSE | **{perf.get('lgbm_test_rmse_eur_mwh', '—')} €/MWh** |
            | Improvement | **+{perf.get('lgbm_improvement_over_baseline_pct', '—')}%** |
            """)

    st.divider()
    col_left, col_right = st.columns(2)
    with col_left:
        fig_cv = FIGURES / "fig2_cv_performance.png"
        if fig_cv.exists():
            st.subheader("Walk-forward CV: Actual vs Forecast")
            st.image(str(fig_cv), use_container_width=True)
    with col_right:
        fig_fi = FIGURES / "fig3_feature_importance.png"
        if fig_fi.exists():
            st.subheader("Feature Importances (LightGBM)")
            st.image(str(fig_fi), use_container_width=True)

    fig_cmp = FIGURES / "fig5_model_comparison.png"
    if fig_cmp.exists():
        st.subheader("Baseline vs LightGBM — recent window")
        st.image(str(fig_cmp), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LOAD SHIFT
# ══════════════════════════════════════════════════════════════════════════════
with tab_signal:
    st.header("Load-Shift Recommendation")

    if not signal:
        st.warning("No load_shift_plan.json found. Run the pipeline first.")
    else:
        link = signal.get("price_carbon_link", {})
        green = signal.get("green_window", {})
        red = signal.get("red_window", {})
        impact = signal.get("impact_per_mwh_shifted", {})

        st.subheader("Premise check — does price actually track carbon?")
        st.caption(
            "Everything below is conditional on this holding. The pipeline re-tests it on every run "
            "and reports low confidence if the relationship weakens."
        )
        if link.get("status") == "ok":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Correlation (Pearson r)", f"{link.get('pearson_r', '—')}")
            c2.metric("Hours tested", f"{link.get('n_hours', 0):,}")
            c3.metric("Cheapest third", f"{link.get('mean_renewable_share_cheapest_tercile', 0):.0%} renewable")
            c4.metric("Priciest third", f"{link.get('mean_renewable_share_priciest_tercile', 0):.0%} renewable")
        else:
            st.warning(f"Premise could not be validated: {link.get('reason', 'unknown')}")

        st.divider()
        st.subheader("Today's recommendation")
        conf = signal.get("confidence", "low")
        badge = {"high": "🟢", "medium": "🟡", "low": "⚪"}.get(conf, "⚪")

        col_g, col_r = st.columns(2)
        with col_g:
            st.success(f"**🟢 Shift INTO** local hours {green.get('local_hours', [])}")
            st.write(f"Mean price: **{green.get('mean_price_eur_mwh', '—')} €/MWh**")
            st.write(f"Renewable share: **{green.get('mean_renewable_share', 0):.1%}**")
            st.write(f"Carbon intensity: **{green.get('mean_intensity_t_per_mwh', '—')} tCO₂/MWh**")
        with col_r:
            st.error(f"**🔴 Shift OUT OF** local hours {red.get('local_hours', [])}")
            st.write(f"Mean price: **{red.get('mean_price_eur_mwh', '—')} €/MWh**")
            st.write(f"Renewable share: **{red.get('mean_renewable_share', 0):.1%}**")
            st.write(f"Carbon intensity: **{red.get('mean_intensity_t_per_mwh', '—')} tCO₂/MWh**")

        m1, m2, m3 = st.columns(3)
        m1.metric("Saved per MWh shifted", f"{impact.get('cost_saving_eur', '—')} €")
        m2.metric("CO₂ avoided per MWh", f"{impact.get('co2_avoided_t', '—')} t")
        m3.metric("Confidence", f"{badge} {conf.upper()}",
                  help="Gated on both forecast accuracy and the strength of the price/carbon link")

        scaled = signal.get("illustrative_scale_up") or {}
        if scaled:
            st.info(
                f"**Illustrative scale-up** — at {scaled.get('shiftable_mwh_per_day')} MWh/day of genuinely "
                f"shiftable load: **{scaled.get('annual_cost_saving_eur', 0):,} €/yr** and "
                f"**{scaled.get('annual_co2_avoided_t', 0):,} tCO₂/yr**. Assumes that flexibility is available "
                "year-round at the spread observed in this window — winter would not sustain it."
            )

        st.subheader("Override conditions")
        st.caption("Any one of these suspends the recommendation.")
        for c in signal.get("override_conditions", []):
            st.write(f"✗ {c}")

    st.divider()
    st.subheader("Delivery Period Views")
    st.caption("Weekly + monthly Base / Peak / Off-peak blocks derived from hourly predictions")

    if delivery is not None and not delivery.empty:
        base_views = delivery[delivery["product_type"] == "base"].copy()
        base_views = base_views[["period_label", "forecast_mean_eur_mwh", "forecast_p10", "forecast_p90", "n_hours"]]
        base_views.columns = ["Period", "Forecast (€/MWh)", "P10 (€/MWh)", "P90 (€/MWh)", "Hours"]

        st.markdown("**Base (all hours)**")
        st.dataframe(base_views, hide_index=True, use_container_width=True)

        peak_views = delivery[delivery["product_type"] == "peak"].copy()
        offpeak_views = delivery[delivery["product_type"] == "offpeak"].copy()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Peak (08:00–20:00 Mon–Fri)**")
            pk = peak_views[["period_label", "forecast_mean_eur_mwh", "n_hours"]].copy()
            pk.columns = ["Period", "Forecast (€/MWh)", "Hours"]
            st.dataframe(pk, hide_index=True, use_container_width=True)
        with col2:
            st.markdown("**Off-peak**")
            op = offpeak_views[["period_label", "forecast_mean_eur_mwh", "n_hours"]].copy()
            op.columns = ["Period", "Forecast (€/MWh)", "Hours"]
            st.dataframe(op, hide_index=True, use_container_width=True)
    else:
        st.warning("No delivery_views.csv found.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    st.header("Out-of-Sample Forecast  ·  Oct–Dec 2025")
    st.caption("Hold-out test window — not used in any training or CV fold")

    if submission is None or submission.empty:
        st.warning("No submission.csv found. Run the pipeline first.")
    else:
        submission["id"] = pd.to_datetime(submission["id"])
        submission = submission.set_index("id")

        col1, col2, col3 = st.columns(3)
        col1.metric("Hours forecast", f"{len(submission):,}")
        col2.metric("Mean forecast", f"{submission['y_pred'].mean():.1f} €/MWh")
        col3.metric("Forecast range", f"{submission['y_pred'].min():.0f} – {submission['y_pred'].max():.0f} €/MWh")

        # Resample to daily for a cleaner chart
        daily = submission.resample("D").mean()

        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(daily.index, daily["y_pred"], color="#1a6faf", lw=1.5, label="Forecast (daily avg)")
        if "y_pred_p10" in daily.columns and "y_pred_p90" in daily.columns:
            ax.fill_between(
                daily.index, daily["y_pred_p10"], daily["y_pred_p90"],
                alpha=0.25, color="#1a6faf", label="80% prediction interval"
            )
        ax.set_ylabel("EUR/MWh")
        ax.set_title("LightGBM Day-Ahead Price Forecast — Oct–Dec 2025 (daily avg)")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.autofmt_xdate()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        with st.expander("View raw hourly forecast table (first 72 rows)"):
            st.dataframe(submission.head(72).reset_index(), hide_index=True, use_container_width=True)

        csv_bytes = submission.reset_index().to_csv(index=False).encode()
        st.download_button(
            label="⬇ Download submission.csv",
            data=csv_bytes,
            file_name="submission.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — FIGURES
# ══════════════════════════════════════════════════════════════════════════════
with tab_figures:
    st.header("Figures")

    figures = [
        ("fig1_price_renewables.png", "Fig 1 — DA Price & Wind/Solar Generation (2022–2025)"),
        ("fig2_cv_performance.png",   "Fig 2 — Walk-Forward CV: Actual vs Forecast (last 4 folds)"),
        ("fig3_feature_importance.png","Fig 3 — LightGBM Feature Importances"),
        ("fig4_price_heatmap.png",    "Fig 4 — Average Price by Hour × Month"),
        ("fig5_model_comparison.png", "Fig 5 — Baseline vs LightGBM (recent window)"),
    ]

    for fname, caption in figures:
        path = FIGURES / fname
        if path.exists():
            st.subheader(caption)
            st.image(str(path), use_container_width=True)
            st.divider()
        else:
            st.warning(f"{fname} not found — run the pipeline to generate it.")
