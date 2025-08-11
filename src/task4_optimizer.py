"""
Task 4 portfolio optimization orchestrator.

This module integrates Task 4 expected inputs with optimization routines
defined in `src/portfolio_optimizer.py` and provides a clean OOP facade to
produce the Efficient Frontier and key portfolios (Max Sharpe, Min Volatility).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from src.portfolio_optimizer import PortfolioOptimizer


@dataclass
class FrontierResult:
    risks: List[float]
    returns: List[float]
    weights: List[np.ndarray]


@dataclass
class KeyPortfolios:
    max_sharpe: Dict[str, float]
    min_volatility: Dict[str, float]
    max_sharpe_weights: pd.Series
    min_volatility_weights: pd.Series


class Task4Optimizer:
    """High-level orchestrator for Task 4 optimization.

    It takes expected inputs (mu, covariance) and historical daily returns,
    and uses the existing `PortfolioOptimizer` to:
    - Generate an efficient frontier
    - Locate maximum Sharpe and minimum volatility portfolios
    - Compute performance metrics
    """

    def __init__(self, risk_free_rate: float = 0.02):
        self.po = PortfolioOptimizer(risk_free_rate=risk_free_rate)

    def efficient_frontier(self, daily_returns: pd.DataFrame, num_portfolios: int = 2000) -> FrontierResult:
        returns, risks, weights = self.po.generate_efficient_frontier(daily_returns, num_portfolios=num_portfolios)
        return FrontierResult(risks=risks, returns=returns, weights=weights)

    def key_portfolios(self, daily_returns: pd.DataFrame) -> KeyPortfolios:
        # Max Sharpe
        w_sharpe, stats_sharpe = self.po.optimize_portfolio_mpt(daily_returns, method="sharpe_max")
        # Min Volatility
        w_min, stats_min = self.po.optimize_portfolio_mpt(daily_returns, method="risk_min")

        tickers = list(daily_returns.columns)
        w_sharpe_s = pd.Series(w_sharpe, index=tickers, name="max_sharpe")
        w_min_s = pd.Series(w_min, index=tickers, name="min_volatility")

        return KeyPortfolios(
            max_sharpe=stats_sharpe,
            min_volatility=stats_min,
            max_sharpe_weights=w_sharpe_s,
            min_volatility_weights=w_min_s,
        )


