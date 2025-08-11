"""
Task 4 data preparation utilities.

Build expected returns and covariance inputs for MPT given:
- Forecasted TSLA prices (converted to expected return)
- Historical average daily returns (annualized) for BND and SPY
- Covariance matrix from historical daily returns for all three assets
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class ExpectedInputs:
    """Container for expected returns and covariance."""

    mu: pd.Series  # expected annual return per asset
    cov: pd.DataFrame  # annualized covariance matrix
    daily_returns: pd.DataFrame  # historical daily returns used for covariance


class Task4DataBuilder:
    """Assemble expected returns and covariance for Task 4 portfolio optimization.

    Steps:
    - Compute historical daily returns for TSLA, BND, SPY
    - Convert TSLA forecasted prices into expected annual return
    - Use historical average daily returns (annualized) for BND and SPY
    - Build annualized covariance from historical daily returns for all assets
    """

    def __init__(self, prices: pd.DataFrame):
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError("prices index must be a DatetimeIndex")
        self.prices = prices.sort_index()

    @staticmethod
    def simple_returns(prices: pd.Series) -> pd.Series:
        return prices.pct_change().dropna()

    @staticmethod
    def annualize_mean_daily_return(mean_daily: float) -> float:
        return float(mean_daily * 252.0)

    @staticmethod
    def annualize_compounded(daily_returns: pd.Series) -> float:
        if len(daily_returns) == 0:
            return 0.0
        compounded = float(np.prod(1.0 + daily_returns) ** (252.0 / len(daily_returns)) - 1.0)
        return compounded

    @staticmethod
    def expected_return_from_forecast(forecast_prices: pd.Series, method: str = "compounded") -> float:
        """Convert forecasted prices into expected annual return.

        - Compute predicted daily simple returns from forecasted prices
        - Annualize using chosen method: 'compounded' (default) or 'mean'
        """
        forecast_prices = forecast_prices.dropna()
        pred_daily = forecast_prices.pct_change().dropna()
        if method == "compounded":
            return Task4DataBuilder.annualize_compounded(pred_daily)
        elif method == "mean":
            return Task4DataBuilder.annualize_mean_daily_return(float(pred_daily.mean()))
        else:
            raise ValueError("method must be 'compounded' or 'mean'")

    def build_expected_inputs(
        self,
        tsla_forecast_prices: pd.Series,
        method: str = "compounded",
        assets: Tuple[str, str, str] = ("TSLA", "BND", "SPY"),
    ) -> ExpectedInputs:
        tsla, bnd, spy = assets

        # Historical daily returns
        hist_returns = pd.DataFrame({
            tsla: self.simple_returns(self.prices[tsla]),
            bnd: self.simple_returns(self.prices[bnd]),
            spy: self.simple_returns(self.prices[spy]),
        }).dropna()

        # Expected return for TSLA from forecast
        mu_tsla = self.expected_return_from_forecast(tsla_forecast_prices, method=method)

        # Expected returns for BND, SPY from historical mean daily returns (annualized by mean*252)
        mu_bnd = self.annualize_mean_daily_return(float(hist_returns[bnd].mean()))
        mu_spy = self.annualize_mean_daily_return(float(hist_returns[spy].mean()))

        mu = pd.Series({tsla: mu_tsla, bnd: mu_bnd, spy: mu_spy})

        # Covariance matrix from historical daily returns (annualized)
        cov = hist_returns.cov() * 252.0

        return ExpectedInputs(mu=mu, cov=cov, daily_returns=hist_returns)


