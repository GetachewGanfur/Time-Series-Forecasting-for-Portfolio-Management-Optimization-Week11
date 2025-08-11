"""
Forecasting analysis utilities for Task 3: Forecast Future Market Trends.

This module provides an object-oriented, modular interface to:
- Fit or accept an already-fitted forecaster
- Generate future forecasts for a specified horizon
- Provide confidence intervals when available (ARIMA/SARIMA via statsmodels)
- Approximate confidence intervals for models without native intervals (LSTM)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    """Container for forecast outputs."""

    model_name: str
    mean: pd.Series
    lower: Optional[pd.Series]
    upper: Optional[pd.Series]
    meta: Dict[str, Any]


class ForecastAnalyzer:
    """High-level interface to run and analyze future forecasts.

    Usage:
        analyzer = ForecastAnalyzer(history=tsla_close)
        analyzer.attach_model('ARIMA', arima_forecaster.fit(history))
        result_6m = analyzer.forecast(horizon_days=126, include_intervals=True)
        result_12m = analyzer.forecast(horizon_days=252, include_intervals=True)
    """

    def __init__(self, history: pd.Series):
        if not isinstance(history, pd.Series):
            raise TypeError("history must be a pandas Series indexed by DatetimeIndex")
        if not isinstance(history.index, pd.DatetimeIndex):
            raise TypeError("history index must be a DatetimeIndex")
        self.history = history.sort_index()
        self.model_name: Optional[str] = None
        self.model = None

    def attach_model(self, model_name: str, model_obj: Any) -> None:
        """Attach a fitted or to-be-fitted model implementing the Task 2 forecaster API."""
        self.model_name = model_name
        self.model = model_obj

    def fit_if_needed(self) -> None:
        """Fit the attached model on the full history if not already fitted."""
        if self.model is None:
            raise ValueError("No model attached. Call attach_model first.")

        # BaseForecaster from Task 2 tracks is_fitted
        if getattr(self.model, "is_fitted", False):
            return
        # Fit on full history
        self.model.fit(self.history)

    def _future_index(self, steps: int) -> pd.DatetimeIndex:
        freq = pd.infer_freq(self.history.index)
        if freq is None:
            # Fallback to business day if not inferable
            freq = "B"
        last_date = self.history.index[-1]
        return pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]

    def forecast(self, horizon_days: int, include_intervals: bool = True, ci_alpha: float = 0.05) -> ForecastResult:
        """Generate a future forecast for the given horizon.

        For ARIMA/SARIMA (statsmodels), native confidence intervals are returned when requested.
        For LSTM (no native intervals), intervals are approximated using residual-based Monte Carlo simulation.
        """
        if self.model is None or self.model_name is None:
            raise ValueError("Model not attached. Use attach_model before forecasting.")

        self.fit_if_needed()
        steps = int(horizon_days)
        future_idx = self._future_index(steps)

        lower, upper = None, None

        # Statsmodels-based models (ARIMA/SARIMA) expose fitted_model with get_forecast
        fitted_model = getattr(self.model, "fitted_model", None)
        if include_intervals and fitted_model is not None and hasattr(fitted_model, "get_forecast"):
            # Use native forecast with confidence intervals
            fc = fitted_model.get_forecast(steps=steps)
            mean_forecast = pd.Series(fc.predicted_mean, index=future_idx, name="forecast")
            ci = fc.conf_int(alpha=ci_alpha)
            # conf_int returns a DataFrame with columns like 'lower y' 'upper y' or similar depending on model
            ci_cols = list(ci.columns)
            if len(ci_cols) >= 2:
                lower = pd.Series(ci.iloc[:, 0].values, index=future_idx, name="lower")
                upper = pd.Series(ci.iloc[:, 1].values, index=future_idx, name="upper")
            else:
                lower = None
                upper = None
        else:
            # Fallback: use model.predict and optionally approximate intervals
            preds = self.model.predict(steps=steps)
            mean_forecast = pd.Series(np.asarray(preds).flatten(), index=future_idx, name="forecast")

            if include_intervals:
                lower, upper = self._approximate_intervals(mean_forecast, ci_alpha=ci_alpha)

        return ForecastResult(
            model_name=self.model_name,
            mean=mean_forecast,
            lower=lower,
            upper=upper,
            meta={
                "horizon_days": steps,
                "alpha": ci_alpha,
                "last_history_date": self.history.index[-1],
            },
        )

    def _approximate_intervals(self, mean_forecast: pd.Series, ci_alpha: float = 0.05) -> Tuple[pd.Series, pd.Series]:
        """Approximate uncertainty bands when the model lacks native CIs.

        Approach: Estimate residual volatility from recent history via daily returns std and
        translate into price uncertainty bands that widen with sqrt(time).
        This is a simplified approximation for communication; not for production risk.
        """
        # Estimate daily return volatility from last 90 days
        recent = self.history.tail(120)
        returns = recent.pct_change().dropna()
        sigma_daily = float(returns.std()) if not returns.empty else 0.03  # fallback 3%

        # z for two-sided interval
        from scipy.stats import norm

        z = norm.ppf(1 - ci_alpha / 2)
        base = float(self.history.iloc[-1])

        horizon = np.arange(1, len(mean_forecast) + 1)
        # Price uncertainty grows roughly with sqrt(t)
        scale = z * sigma_daily * np.sqrt(horizon)

        lower = mean_forecast * (1 - scale)
        upper = mean_forecast * (1 + scale)
        lower.name = "lower"
        upper.name = "upper"
        return lower, upper


