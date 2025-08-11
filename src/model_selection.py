"""
Model selection helpers for time series forecasting
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def chronological_split(
    series: pd.Series,
    train_end_date: Optional[str] = None,
    train_ratio: Optional[float] = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Split a time series into train and test sets chronologically.

    One of train_end_date or train_ratio must be provided.
    """
    if train_end_date is None and train_ratio is None:
        raise ValueError("Provide either train_end_date or train_ratio")

    series = series.sort_index()

    if train_end_date is not None:
        train = series.loc[:train_end_date]
        test = series.loc[train_end_date:].iloc[1:]
        return train, test

    assert train_ratio is not None and 0 < train_ratio < 1
    cutoff = int(len(series) * train_ratio)
    train = series.iloc[:cutoff]
    test = series.iloc[cutoff:]
    return train, test


def arima_order_grid_search(
    series: pd.Series,
    p_values: Iterable[int] = (0, 1, 2, 3),
    d_values: Iterable[int] = (0, 1, 2),
    q_values: Iterable[int] = (0, 1, 2, 3),
    criterion: str = "aic",
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
) -> Tuple[Tuple[int, int, int], Dict[str, float]]:
    """
    Simple grid search for ARIMA(p,d,q) using AIC/BIC.
    Returns the best order and the corresponding metrics.
    """
    best_score = np.inf
    best_order: Optional[Tuple[int, int, int]] = None
    best_metrics: Dict[str, float] = {}

    y = series.astype(float)

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(
                        y,
                        order=(p, d, q),
                        enforce_stationarity=enforce_stationarity,
                        enforce_invertibility=enforce_invertibility,
                    )
                    fitted = model.fit()
                    aic = float(fitted.aic)
                    bic = float(fitted.bic)
                    score = aic if criterion.lower() == "aic" else bic

                    if score < best_score:
                        best_score = score
                        best_order = (p, d, q)
                        best_metrics = {"aic": aic, "bic": bic}
                except Exception:
                    # Skip non-invertible / non-stationary configs
                    continue

    if best_order is None:
        raise RuntimeError("ARIMA grid search failed to find a valid model.")

    return best_order, best_metrics


def sarima_order_grid_search(
    series: pd.Series,
    non_seasonal_grid: Iterable[Tuple[int, int, int]] = ((0, 1, 0), (1, 1, 0), (1, 1, 1)),
    seasonal_grid: Iterable[Tuple[int, int, int, int]] = ((0, 1, 1, 12), (1, 1, 1, 12)),
    criterion: str = "aic",
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int], Dict[str, float]]:
    """
    Simple grid search for SARIMA orders.
    Returns best (p,d,q), (P,D,Q,s), and metrics.
    """
    best_score = np.inf
    best_order: Optional[Tuple[int, int, int]] = None
    best_seasonal: Optional[Tuple[int, int, int, int]] = None
    best_metrics: Dict[str, float] = {}

    y = series.astype(float)

    for order in non_seasonal_grid:
        for seasonal_order in seasonal_grid:
            try:
                model = SARIMAX(y, order=order, seasonal_order=seasonal_order)
                fitted = model.fit(disp=False)
                aic = float(fitted.aic)
                bic = float(fitted.bic)
                score = aic if criterion.lower() == "aic" else bic

                if score < best_score:
                    best_score = score
                    best_order = order
                    best_seasonal = seasonal_order
                    best_metrics = {"aic": aic, "bic": bic}
            except Exception:
                continue

    if best_order is None or best_seasonal is None:
        raise RuntimeError("SARIMA grid search failed to find a valid model.")

    return best_order, best_seasonal, best_metrics


