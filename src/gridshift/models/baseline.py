"""Baseline forecaster: seasonal naive (same hour, same day last week)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class SeasonalNaive(BaseEstimator, RegressorMixin):
    """
    Predict next-day hourly price = realised price at the same hour 7 days ago.

    This is the standard industry benchmark for intra-week seasonality and
    serves as the lower bar our LightGBM model must beat.
    """

    LAG_HOURS = 168   # 7 × 24

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalNaive":
        # Store the training tail so we can look back when predicting
        self._y_train = y.copy()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        For each row in X, return the target value 168 hours earlier.
        X must have a DatetimeIndex so we can do the timestamp subtraction.
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex")

        combined = pd.concat([self._y_train, pd.Series(np.nan, index=X.index)])
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()

        preds = []
        freq = pd.tseries.frequencies.to_offset("h")
        for ts in X.index:
            lag_ts = ts - pd.Timedelta(hours=self.LAG_HOURS)
            val = combined.get(lag_ts, np.nan)
            preds.append(float(val) if val is not None else np.nan)
        return np.array(preds)
