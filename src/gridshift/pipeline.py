"""Main pipeline orchestrator."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.rule import Rule

from gridshift import config as cfg
from gridshift.config import Config, ensure_dirs
from gridshift.curve import load_shift, translation as curve
from gridshift.features.engineer import build_feature_matrix
from gridshift.models.baseline import SeasonalNaive
from gridshift.models.forecaster import PowerPriceForecaster, cv_summary
from gridshift.quality import llm_qa, standard_qa
from gridshift import plots

logger = logging.getLogger(__name__)
console = Console()

RAW_PARQUET = cfg.DATA_RAW / "de_power_market.parquet"

# Every series the pipeline models on, and the share of hours each is allowed to
# be missing before the run is refused. Both fetchers have silently produced
# empty or wrong-technology columns — SMARD via wrong filter IDs, ENTSO-E via
# over-long request windows that 400 and were caught into an empty Series — and
# in both cases the pipeline modelled on the result without complaint. The
# check lives here rather than in either client so it also covers the cache.
REQUIRED_SERIES = (
    "price_da_eur_mwh",
    "load_mwh",
    "wind_onshore_mwh",
    "wind_offshore_mwh",
    "solar_mwh",
)
MAX_MISSING_FRACTION = 0.05
QA_REPORT = cfg.OUTPUTS / "qa_report.json"
LLM_QA_LOG = cfg.OUTPUTS / "llm_qa_log.json"
SUBMISSION_CSV = cfg.OUTPUTS / "submission.csv"
DELIVERY_VIEWS_CSV = cfg.OUTPUTS / "delivery_views.csv"


def _assert_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Refuse to model on a dataset with a missing or mostly-empty series."""
    incomplete = {
        col: (round(float(df[col].isna().mean()), 4) if col in df.columns else "absent")
        for col in REQUIRED_SERIES
        if col not in df.columns or df[col].isna().mean() > MAX_MISSING_FRACTION
    }
    if incomplete:
        raise RuntimeError(
            f"Ingested data is incomplete — missing fraction by series: {incomplete}. "
            "Check the source filter IDs / request windows before modelling on this."
        )
    return df


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        ensure_dirs()

    # ─────────────────────────────────────────────────────────────────────────
    def _step_ingest(self) -> pd.DataFrame:
        console.print(Rule("[bold blue]Step 1: Data Ingestion[/bold blue]"))

        if RAW_PARQUET.exists() and not self.config.force_refetch:
            console.print(f"  Loading cached data from {RAW_PARQUET}")
            return _assert_coverage(pd.read_parquet(RAW_PARQUET))

        if RAW_PARQUET.exists() and self.config.force_refetch:
            console.print("  [yellow]force_refetch=True — ignoring cache, re-fetching from source[/yellow]")

        if self.config.entsoe_api_key:
            console.print("  Source: ENTSO-E Transparency Platform")
            from gridshift.ingestion.entsoe_client import fetch_dataset as entsoe_fetch
            df = entsoe_fetch(
                self.config.entsoe_api_key,
                self.config.start_date,
                self.config.end_date,
            )
        else:
            console.print("  Source: SMARD (Bundesnetzagentur) — no API key required")
            console.print("  [dim]Set ENTSOE_API_KEY in .env to use ENTSO-E instead[/dim]")
            from gridshift.ingestion.smard import fetch_dataset as smard_fetch
            df = smard_fetch(self.config.start_date, self.config.end_date)

        _assert_coverage(df)
        df.to_parquet(RAW_PARQUET)
        console.print(f"  Saved {len(df):,} rows → {RAW_PARQUET}")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    def _step_qa(self, df: pd.DataFrame) -> None:
        console.print(Rule("[bold blue]Step 2: Data Quality[/bold blue]"))

        # Standard QA
        report = standard_qa.run_qa(df)
        with open(QA_REPORT, "w") as f:
            json.dump(report, f, indent=2, default=str)
        standard_qa.print_qa_summary(report)
        console.print(f"  QA report → {QA_REPORT}")

        # LLM-driven QA
        if not self.config.anthropic_api_key:
            console.print("  [yellow]Skipping LLM QA — ANTHROPIC_API_KEY not set[/yellow]")
            return

        console.print("  Running LLM QA rule generation...")
        # Use a 1-month sample for the LLM prompt (cost / context efficiency)
        sample_df = df.tail(24 * 30)
        llm_report = llm_qa.run_llm_qa(
            sample_df,
            api_key=self.config.anthropic_api_key,
            model=self.config.llm_model,
            log_path=LLM_QA_LOG,
        )
        llm_qa.print_llm_qa_summary(llm_report)
        console.print(f"  LLM QA log → {LLM_QA_LOG}")

    # ─────────────────────────────────────────────────────────────────────────
    def _step_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        console.print(Rule("[bold blue]Step 3: Feature Engineering[/bold blue]"))

        df = df.copy()
        # Remove rows with clearly bad price data
        df = df[df["price_da_eur_mwh"].between(-500, 3000)]

        # Drop fundamental columns with >50% missing — keeping them would NaN-drop
        # most feature-matrix rows and corrupt the train/test split.
        # Documented in qa_report.json under "dropped_fundamentals".
        fund_cols = [c for c in ["wind_onshore_mwh", "wind_offshore_mwh", "solar_mwh", "load_mwh"]
                     if c in df.columns]
        dropped_fundamentals = {}
        for col in fund_cols[:]:
            missing_frac = float(df[col].isna().mean())
            if missing_frac > 0.50:
                logger.warning("Dropping %s — %.0f%% missing (SMARD filter may be incorrect)", col, missing_frac * 100)
                console.print(f"  [yellow]Dropping {col} — {missing_frac:.0%} missing (SMARD filter unreliable)[/yellow]")
                dropped_fundamentals[col] = round(missing_frac, 4)
                df = df.drop(columns=[col])
                fund_cols.remove(col)

        # Forward-fill small gaps (≤3 h) in retained fundamentals; leave prices as-is
        if fund_cols:
            df[fund_cols] = df[fund_cols].ffill(limit=3)

        console.print(f"  Fundamentals used: {fund_cols}")
        X, y = build_feature_matrix(df)
        console.print(f"  Feature matrix: {X.shape}  —  {len(X.columns)} features")
        console.print(f"  Date range: {y.index.min().date()} → {y.index.max().date()}")

        # Record dropped columns in QA report for transparency
        if dropped_fundamentals and QA_REPORT.exists():
            with open(QA_REPORT) as f:
                qa = json.load(f)
            qa["dropped_fundamentals"] = {
                col: {"missing_fraction": frac, "reason": "exceeds 50% threshold — SMARD filter unreliable"}
                for col, frac in dropped_fundamentals.items()
            }
            qa["fundamentals_used"] = fund_cols
            with open(QA_REPORT, "w") as f:
                json.dump(qa, f, indent=2, default=str)

        return X, y

    # ─────────────────────────────────────────────────────────────────────────
    def _step_model(
        self, X: pd.DataFrame, y: pd.Series, df_raw: pd.DataFrame
    ) -> tuple[PowerPriceForecaster, pd.Series, pd.Series]:
        console.print(Rule("[bold blue]Step 4: Forecasting & Validation[/bold blue]"))

        # Reserve last test_months for final evaluation / submission.csv
        split_date = y.index.max() - pd.DateOffset(months=self.config.test_months)
        X_train, y_train = X[X.index <= split_date], y[y.index <= split_date]
        X_test, y_test = X[X.index > split_date], y[y.index > split_date]

        # ── Baseline ──────────────────────────────────────────────────────────
        console.print("  Baseline: Seasonal Naive (168h lag)")
        baseline = SeasonalNaive()
        baseline.fit(X_train, y_train)
        baseline_preds = pd.Series(
            baseline.predict(X_test), index=X_test.index, name="baseline_pred"
        )
        baseline_mae = float((y_test - baseline_preds).abs().mean())
        console.print(f"  Baseline MAE on test set: {baseline_mae:.2f} EUR/MWh")

        # ── LightGBM walk-forward CV ──────────────────────────────────────────
        console.print(f"  LightGBM walk-forward CV ({self.config.cv_folds} folds)...")
        forecaster = PowerPriceForecaster(random_state=self.config.random_seed)
        fold_results = forecaster.walk_forward_cv(X_train, y_train, n_folds=self.config.cv_folds)
        summary = cv_summary(fold_results)
        console.print(
            f"  CV results: MAE={summary['mae_mean']:.1f}±{summary['mae_std']:.1f}  "
            f"RMSE={summary['rmse_mean']:.1f}±{summary['rmse_std']:.1f}  "
            f"Pinball@90={summary['pinball_90_mean']:.1f}"
        )

        # ── Final model fit + prediction intervals from CV residuals ─────────────
        forecaster.fit_with_residuals(X_train, y_train, fold_results)
        lgbm_preds = forecaster.predict(X_test)
        lgbm_lower, lgbm_upper = forecaster.predict_interval(X_test)
        lgbm_mae = float((y_test - lgbm_preds).abs().mean())
        lgbm_rmse = float(np.sqrt(((y_test - lgbm_preds) ** 2).mean()))
        improvement_pct = (1 - lgbm_mae / baseline_mae) * 100
        console.print(
            f"  LightGBM test MAE: {lgbm_mae:.2f}  RMSE: {lgbm_rmse:.2f}  "
            f"  vs baseline MAE: {baseline_mae:.2f}  "
            f"  Improvement: {improvement_pct:.1f}%"
        )
        # Prediction interval coverage (should be ~80% for 10–90 interval)
        in_band = float(((y_test >= lgbm_lower) & (y_test <= lgbm_upper)).mean())
        console.print(f"  80% prediction interval empirical coverage: {in_band:.1%}")

        # ── Figures ───────────────────────────────────────────────────────────
        console.print("  Generating figures...")
        plots.fig_price_and_renewables(df_raw, cfg.FIGURES / "fig1_price_renewables.png")
        plots.fig_cv_performance(fold_results[-4:], cfg.FIGURES / "fig2_cv_performance.png")
        plots.fig_feature_importance(forecaster.feature_importances, cfg.FIGURES / "fig3_feature_importance.png")
        plots.fig_price_heatmap(df_raw, cfg.FIGURES / "fig4_price_heatmap.png")
        plots.fig_model_comparison(df_raw, baseline_preds, lgbm_preds, cfg.FIGURES / "fig5_model_comparison.png")

        # ── submission.csv ────────────────────────────────────────────────────
        submission = pd.DataFrame({
            "id": lgbm_preds.index.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "y_pred": lgbm_preds.values.round(2),
            "y_pred_p10": lgbm_lower.values.round(2),
            "y_pred_p90": lgbm_upper.values.round(2),
        })
        submission.to_csv(SUBMISSION_CSV, index=False)
        console.print(f"  submission.csv → {SUBMISSION_CSV}  ({len(submission)} rows, "
                      f"window: {y_test.index.min().date()} → {y_test.index.max().date()})")

        # Store full performance summary in QA report
        with open(QA_REPORT, "r") as f:
            qa = json.load(f)
        qa["model_performance"] = {
            "baseline_mae_eur_mwh": round(baseline_mae, 2),
            "lgbm_test_mae_eur_mwh": round(lgbm_mae, 2),
            "lgbm_test_rmse_eur_mwh": round(lgbm_rmse, 2),
            "lgbm_improvement_over_baseline_pct": round(improvement_pct, 1),
            "prediction_interval_coverage_80pct": round(in_band, 4),
            "lgbm_cv": summary,
            "test_window": {
                "start": str(y_test.index.min().date()),
                "end": str(y_test.index.max().date()),
                "n_hours": int(len(y_test)),
            },
        }
        with open(QA_REPORT, "w") as f:
            json.dump(qa, f, indent=2, default=str)

        return forecaster, lgbm_preds, baseline_preds

    # ─────────────────────────────────────────────────────────────────────────
    def _step_curve(self, forecaster: PowerPriceForecaster, df_raw: pd.DataFrame) -> None:
        console.print(Rule("[bold blue]Step 5: Carbon-Aware Load Shift[/bold blue]"))

        # Use the most recent month of test-set predictions as the "future" view
        preds_path = SUBMISSION_CSV
        if not preds_path.exists():
            console.print("  [yellow]No submission.csv found, skipping curve step[/yellow]")
            return

        preds = pd.read_csv(preds_path, parse_dates=["id"], index_col="id")
        preds.index = pd.DatetimeIndex(preds.index, tz="UTC")
        y_pred = preds["y_pred"].rename("y_pred")
        y_lower = preds["y_pred_p10"].rename("y_pred_p10") if "y_pred_p10" in preds.columns else None
        y_upper = preds["y_pred_p90"].rename("y_pred_p90") if "y_pred_p90" in preds.columns else None

        # Compute delivery block views — weekly and monthly aggregations
        weekly_views = curve.compute_delivery_views(y_pred, forecast_horizon="week",
                                                    y_lower=y_lower, y_upper=y_upper)
        monthly_views = curve.compute_delivery_views(y_pred, forecast_horizon="month",
                                                     y_lower=y_lower, y_upper=y_upper)
        all_views = weekly_views + monthly_views
        views_df = curve.views_to_dataframe(all_views)
        views_df.to_csv(DELIVERY_VIEWS_CSV, index=False)
        console.print(f"  Delivery views → {DELIVERY_VIEWS_CSV}")
        console.print(f"  Weekly base views: {len([v for v in weekly_views if v.product_type == 'base'])}")
        console.print(f"  Monthly base views: {len([v for v in monthly_views if v.product_type == 'base'])}")

        # Build the carbon-aware load-shift plan
        test_mae = 30.0
        try:
            with open(QA_REPORT) as f:
                qa = json.load(f)
            test_mae = qa.get("model_performance", {}).get("lgbm_test_mae_eur_mwh", 30.0)
        except Exception:
            pass

        plan = load_shift.build_shift_plan(y_pred, df_raw, model_mae=test_mae)

        link = plan.get("price_carbon_link", {})
        console.print("\n  [bold]Price/carbon link (premise check)[/bold]")
        if link.get("status") == "ok":
            console.print(f"    Pearson r:  {link['pearson_r']:.3f}   "
                          f"Spearman r: {link['spearman_r']:.3f}   "
                          f"({link['n_hours']:,} hours)")
            console.print(f"    Cheapest third of hours: {link['mean_renewable_share_cheapest_tercile']:.1%} renewable, "
                          f"{link['mean_intensity_cheapest_tercile_t_per_mwh']:.3f} tCO2/MWh")
            console.print(f"    Priciest third of hours: {link['mean_renewable_share_priciest_tercile']:.1%} renewable, "
                          f"{link['mean_intensity_priciest_tercile_t_per_mwh']:.3f} tCO2/MWh")
        else:
            console.print(f"    [yellow]Unavailable: {link.get('reason')}[/yellow]")

        green, red = plan["green_window"], plan["red_window"]
        impact = plan["impact_per_mwh_shifted"]
        console.print("\n  [bold]Load-Shift Recommendation[/bold]")
        console.print(f"    Period:     {plan['period']}")
        console.print(f"    [green]Green hours[/green] (local): {green['local_hours']}  "
                      f"@ {green['mean_price_eur_mwh']:.1f} EUR/MWh, "
                      f"{green['mean_renewable_share']:.1%} renewable")
        console.print(f"    [red]Red hours[/red]   (local): {red['local_hours']}  "
                      f"@ {red['mean_price_eur_mwh']:.1f} EUR/MWh, "
                      f"{red['mean_renewable_share']:.1%} renewable")
        co2 = impact["co2_avoided_t"]
        console.print(f"    Per MWh shifted: save {impact['cost_saving_eur']:.1f} EUR, "
                      f"avoid {'unavailable' if co2 is None else format(co2, '.3f')} tCO2")
        console.print(f"    Confidence: [bold]{plan['confidence'].upper()}[/bold] "
                      f"(forecast MAE {plan['forecast_mae_eur_mwh']:.1f} EUR/MWh)")

        # Illustrative scale-up: a mid-size district cooling plant with roughly
        # 200 MWh/day of genuinely shiftable chilled-water storage capacity.
        plan["illustrative_scale_up"] = load_shift.scale_impact(plan, shiftable_mwh_per_day=200)
        scaled = plan["illustrative_scale_up"]
        if scaled:
            console.print(f"    At 200 MWh/day shiftable: "
                          f"{scaled['annual_cost_saving_eur']:,} EUR/yr, "
                          f"{scaled['annual_co2_avoided_t']:,} tCO2/yr")

        console.print("\n  [dim]Override conditions:[/dim]")
        for cond in plan["override_conditions"]:
            console.print(f"    ✗ {cond}")

        plan_path = cfg.OUTPUTS / "load_shift_plan.json"
        with open(plan_path, "w") as f:
            json.dump(plan, f, indent=2, default=str)
        console.print(f"\n  Load-shift plan → {plan_path}")

    # ─────────────────────────────────────────────────────────────────────────
    def run(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
        )
        console.print(Rule("[bold green]GridShift — Carbon-Aware Load Shifting[/bold green]"))
        console.print(f"  Market: {self.config.market}  |  "
                      f"{self.config.start_date} → {self.config.end_date}")

        df_raw = self._step_ingest()
        self._step_qa(df_raw)
        X, y = self._step_features(df_raw)
        forecaster, lgbm_preds, baseline_preds = self._step_model(X, y, df_raw)
        self._step_curve(forecaster, df_raw)

        console.print(Rule("[bold green]Pipeline complete[/bold green]"))
        console.print(f"  Outputs: {cfg.OUTPUTS}")
