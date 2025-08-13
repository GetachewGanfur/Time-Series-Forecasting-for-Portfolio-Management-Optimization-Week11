import os
import sys
import numpy as np
import pandas as pd

# Ensure project root is on sys.path for direct module imports
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.forecasting_models import ARIMAForecaster
from src.portfolio_optimizer import PortfolioOptimizer
from src.backtesting import PortfolioBacktester


def _make_sine_series(n=300, noise=0.01, seed=42):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 8 * np.pi, n)
    y = np.sin(x) + rng.normal(0, noise, size=n)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(y, index=idx, name="synthetic")


def test_end_to_end_minimal_pipeline():
    # Synthetic price series -> returns
    prices = _make_sine_series(n=260)
    returns = prices.pct_change().dropna()

    # Duplicate to create 3-asset DataFrame with correlated noise
    df = pd.DataFrame({
        "TSLA": returns,
        "BND": (returns * 0.3 + returns.rolling(3).mean().fillna(0) * 0.2).clip(-0.2, 0.2),
        "SPY": (returns * 0.6 + returns.shift(1).fillna(0) * 0.2).clip(-0.2, 0.2),
    }).dropna()

    # Forecasting (sanity: fit ARIMA and predict small horizon)
    forecaster = ARIMAForecaster(order=(1, 1, 1))
    forecaster.fit(prices)
    preds = forecaster.predict(steps=5)
    assert len(preds) == 5

    # Optimization via scipy path (works without PyPortfolioOpt)
    optimizer = PortfolioOptimizer(risk_free_rate=0.02, target_return=0.10)
    weights, stats = optimizer.optimize_portfolio_mpt(df, method="sharpe_max")
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    assert set(["expected_return", "volatility", "sharpe_ratio"]).issubset(stats.keys())

    # Backtesting
    weights_df = pd.DataFrame(index=df.index, columns=df.columns)
    weights_df[:] = weights
    backtester = PortfolioBacktester(initial_capital=10000, transaction_costs=0.0)
    results = backtester.run_backtest(df, weights_df, rebalance_frequency="monthly")
    assert "portfolio_values" in results and not results["portfolio_values"].empty

