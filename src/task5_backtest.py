"""
Task 5 backtesting orchestrator.

Provides an object-oriented, modular interface to:
- Prepare the backtest window and daily returns
- Build strategy and benchmark weights DataFrames
- Run monthly-rebalanced backtests using `src.backtesting.PortfolioBacktester`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.backtesting import PortfolioBacktester


@dataclass
class BacktestRun:
    portfolio_values: pd.DataFrame
    performance_metrics: Dict[str, float]
    trade_history: list


class BacktestOrchestrator:
    """High-level backtest orchestrator for Task 5.

    Usage:
        orchestrator = BacktestOrchestrator(initial_capital=100000)
        windowed_returns = orchestrator.prepare_returns(prices, '2024-08-01', '2025-07-31')
        strat_weights_df = orchestrator.make_constant_weights(windowed_returns.index, {'TSLA': 0.3, 'BND': 0.4, 'SPY': 0.3})
        bench_weights_df = orchestrator.make_constant_weights(windowed_returns.index, {'TSLA': 0.0, 'BND': 0.4, 'SPY': 0.6})
        res_strat = orchestrator.run(windowed_returns, strat_weights_df, rebalance_frequency='monthly')
        res_bench = orchestrator.run(windowed_returns, bench_weights_df, rebalance_frequency='monthly')
    """

    def __init__(self, initial_capital: float = 100000.0, transaction_costs: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_costs = transaction_costs
        self.backtester = PortfolioBacktester(initial_capital=initial_capital, transaction_costs=transaction_costs)

    @staticmethod
    def prepare_returns(prices: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Compute daily simple returns and slice to a backtest window."""
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError("prices index must be a DatetimeIndex")
        prices = prices.sort_index()
        returns = prices.pct_change().dropna()
        windowed = returns.loc[start_date:end_date]
        # Ensure only assets with complete data in window
        windowed = windowed.dropna(axis=1, how='any')
        return windowed

    @staticmethod
    def make_constant_weights(dates: pd.DatetimeIndex, weights_map: Dict[str, float]) -> pd.DataFrame:
        """Create a daily weights DataFrame given a mapping of ticker->weight."""
        weights_series = {k: np.full(shape=(len(dates),), fill_value=float(v)) for k, v in weights_map.items()}
        weights_df = pd.DataFrame(weights_series, index=dates)
        # Normalize in case inputs do not sum exactly to 1.0
        row_sums = weights_df.sum(axis=1).replace(0, np.nan)
        weights_df = weights_df.div(row_sums, axis=0)
        return weights_df

    def run(self, windowed_returns: pd.DataFrame, weights_df: pd.DataFrame, rebalance_frequency: str = 'monthly') -> BacktestRun:
        # Align columns and index
        weights_df = weights_df.reindex(index=windowed_returns.index).fillna(method='ffill')
        weights_df = weights_df[windowed_returns.columns]
        res = self.backtester.run_backtest(windowed_returns, weights_df, rebalance_frequency=rebalance_frequency)
        return BacktestRun(
            portfolio_values=res['portfolio_values'],
            performance_metrics=res['performance_metrics'],
            trade_history=res['trade_history'],
        )


