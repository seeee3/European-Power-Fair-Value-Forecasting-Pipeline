"""
LightGBM-based next-day electricity price forecaster with walk-forward CV.

Walk-forward scheme:
  - The dataset is split into monthly folds (oldest first).
  - For each fold k: train on all data up to fold k-1, validate on fold k.
  - This mimics the production use case (never look ahead) and avoids
    temporal leakage that would arise from standard k-fold shuffling.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    mae: float
    rmse: float
    pinball_10: float   # Expected shortfall proxy — how bad are the low-price errors?
    pinball_90: float   # Upside miss
    n_val: int
    y_pred: pd.Series = field(repr=False)
    y_true: pd.Series = field(repr=False)


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Quantile (pinball) loss at quantile q in [0, 1]."""
    e = y_true - y_pred
    return float(np.mean(np.where(e >= 0, q * e, (q - 1) * e)))


class PowerPriceForecaster:
    """LightGBM regressor with walk-forward cross-validation."""

    def __init__(
        self,
        n_estimators: int = 800,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self._model = None
        self._feature_names: list[str] = []

    def _make_model(self):
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(**self.params)
        except (ImportError, OSError):
            # Fallback: sklearn HistGradientBoosting (comparable performance, no system libs)
            logger.warning("LightGBM unavailable — falling back to HistGradientBoostingRegressor")
            from sklearn.ensemble import HistGradientBoostingRegressor
            return HistGradientBoostingRegressor(
                max_iter=self.params["n_estimators"],
                learning_rate=self.params["learning_rate"],
                max_leaf_nodes=self.params["num_leaves"],
                min_samples_leaf=self.params["min_child_samples"],
                random_state=self.params["random_state"],
            )

    def walk_forward_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 12,
    ) -> list[FoldResult]:
        """
        Walk-forward CV splitting by calendar month.

        Args:
            X: Feature DataFrame with DatetimeIndex
            y: Target Series with same DatetimeIndex
            n_folds: Number of monthly validation folds (from the end)

        Returns:
            List of FoldResult objects, one per validation fold.
        """
        # Strip timezone before converting to Period (avoids pandas UserWarning)
        naive_index = X.index.tz_localize(None) if X.index.tz is not None else X.index
        months = (
            pd.to_datetime(naive_index)
            .to_period("M")
            .unique()
            .sort_values()
        )

        if len(months) < n_folds + 2:
            raise ValueError(
                f"Not enough months ({len(months)}) for {n_folds} CV folds; "
                "reduce n_folds or extend the date range."
            )

        val_months = months[-(n_folds):]
        results: list[FoldResult] = []

        for fold_idx, val_month in enumerate(val_months):
            naive_ix = X.index.tz_localize(None) if X.index.tz is not None else X.index
            month_periods = pd.to_datetime(naive_ix).to_period("M")
            train_mask = np.array(month_periods < val_month)
            val_mask = np.array(month_periods == val_month)

            if train_mask.sum() < 24 * 7:
                logger.warning("Skipping fold %d — insufficient training data", fold_idx)
                continue

            X_tr, y_tr = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            model = self._make_model()
            model.fit(X_tr, y_tr)

            y_pred_arr = model.predict(X_val)
            y_pred = pd.Series(y_pred_arr, index=y_val.index)

            mae = float(mean_absolute_error(y_val, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
            p10 = _pinball_loss(y_val.values, y_pred_arr, q=0.1)
            p90 = _pinball_loss(y_val.values, y_pred_arr, q=0.9)

            results.append(
                FoldResult(
                    fold=fold_idx,
                    train_start=str(X_tr.index.min().date()),
                    train_end=str(X_tr.index.max().date()),
                    val_start=str(X_val.index.min().date()),
                    val_end=str(X_val.index.max().date()),
                    mae=mae,
                    rmse=rmse,
                    pinball_10=p10,
                    pinball_90=p90,
                    n_val=int(len(y_val)),
                    y_pred=y_pred,
                    y_true=y_val,
                )
            )
            logger.info(
                "Fold %2d [%s → %s]  MAE=%.1f  RMSE=%.1f",
                fold_idx, results[-1].val_start, results[-1].val_end, mae, rmse,
            )

        return results

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PowerPriceForecaster":
        """Train final model on the full dataset."""
        self._feature_names = list(X.columns)
        self._model = self._make_model()
        self._model.fit(X, y)
        logger.info("Final model trained on %d rows, %d features", len(X), len(X.columns))
        return self

    def fit_with_residuals(
        self, X: pd.DataFrame, y: pd.Series, fold_results: "list[FoldResult]"
    ) -> "PowerPriceForecaster":
        """
        Fit final model and store empirical residual quantiles from CV folds.
        These are used to build prediction intervals: for hour h, the interval is
        [point_forecast + q10_residual, point_forecast + q90_residual].
        Residuals are stored by hour-of-day to capture intraday heteroskedasticity
        (extreme prices are more common during peak hours).
        """
        self.fit(X, y)
        all_resid = pd.concat(
            [r.y_true - r.y_pred for r in fold_results]
        ).sort_index()
        # Hour-of-day in local CET/CEST for interval calibration
        local_hour = all_resid.index.tz_convert("Europe/Berlin").hour
        self._resid_q10 = float(np.percentile(all_resid.values, 10))
        self._resid_q90 = float(np.percentile(all_resid.values, 90))
        # Per-hour residual quantiles (richer intervals)
        self._hourly_q10 = {
            h: float(np.percentile(all_resid.values[local_hour == h], 10))
            for h in range(24)
            if (local_hour == h).sum() > 10
        }
        self._hourly_q90 = {
            h: float(np.percentile(all_resid.values[local_hour == h], 90))
            for h in range(24)
            if (local_hour == h).sum() > 10
        }
        logger.info(
            "Prediction interval calibrated from %d CV residuals: "
            "global [%.1f, %.1f] EUR/MWh",
            len(all_resid), self._resid_q10, self._resid_q90,
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Model not fitted; call .fit() first")
        preds = self._model.predict(X)
        return pd.Series(preds, index=X.index, name="y_pred")

    def predict_interval(
        self, X: pd.DataFrame, q_lo: float = 0.10, q_hi: float = 0.90
    ) -> tuple[pd.Series, pd.Series]:
        """
        Return (lower, upper) empirical prediction intervals.
        Uses hour-of-day calibrated residual quantiles from CV folds if available,
        otherwise falls back to the global quantiles.
        """
        point = self.predict(X)
        if not hasattr(self, "_resid_q10"):
            raise RuntimeError("Call fit_with_residuals() before predict_interval()")
        local_hour = X.index.tz_convert("Europe/Berlin").hour
        lo_offsets = np.array([self._hourly_q10.get(h, self._resid_q10) for h in local_hour])
        hi_offsets = np.array([self._hourly_q90.get(h, self._resid_q90) for h in local_hour])
        lower = pd.Series(point.values + lo_offsets, index=X.index, name="y_pred_p10")
        upper = pd.Series(point.values + hi_offsets, index=X.index, name="y_pred_p90")
        return lower, upper

    @property
    def feature_importances(self) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        if hasattr(self._model, "feature_importances_"):
            importances = self._model.feature_importances_
        else:
            # sklearn HGBR does not expose importances; use uniform placeholder
            importances = np.ones(len(self._feature_names))
        return pd.Series(
            importances,
            index=self._feature_names,
            name="importance",
        ).sort_values(ascending=False)


def cv_summary(results: list[FoldResult]) -> dict[str, float]:
    """Aggregate metrics across all CV folds."""
    return {
        "mae_mean":        round(float(np.mean([r.mae for r in results])), 2),
        "mae_std":         round(float(np.std([r.mae for r in results])), 2),
        "rmse_mean":       round(float(np.mean([r.rmse for r in results])), 2),
        "rmse_std":        round(float(np.std([r.rmse for r in results])), 2),
        "pinball_10_mean": round(float(np.mean([r.pinball_10 for r in results])), 2),
        "pinball_90_mean": round(float(np.mean([r.pinball_90 for r in results])), 2),
        "n_folds":         len(results),
    }
