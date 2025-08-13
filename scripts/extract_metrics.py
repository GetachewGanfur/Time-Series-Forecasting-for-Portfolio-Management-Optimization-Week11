import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_manager import DataManager
from src.portfolio_optimizer import PortfolioOptimizer


def annualize_return(daily_returns: pd.Series) -> float:
    return float(daily_returns.mean() * 252)


def annualize_volatility(daily_returns: pd.Series) -> float:
    return float(daily_returns.std() * np.sqrt(252))


def compute_portfolio_metrics(returns_df: pd.DataFrame, risk_free_rate: float = 0.02):
    optimizer = PortfolioOptimizer(risk_free_rate=risk_free_rate)
    rets, risks, weights = optimizer.generate_efficient_frontier(returns_df, num_portfolios=1500)

    # Max Sharpe and Min Vol
    sharpe = [(r - risk_free_rate) / v if v > 0 else -np.inf for r, v in zip(rets, risks)]
    max_sharpe_idx = int(np.argmax(sharpe))
    min_vol_idx = int(np.argmin(risks))

    max_sharpe_weights = weights[max_sharpe_idx]
    min_vol_weights = weights[min_vol_idx]

    max_sharpe_metrics = optimizer.calculate_portfolio_metrics(returns_df, max_sharpe_weights)
    min_vol_metrics = optimizer.calculate_portfolio_metrics(returns_df, min_vol_weights)

    # VaR/CVaR for max sharpe
    var_results = {}
    for method in ["historical", "parametric", "monte_carlo"]:
        var_results[method] = optimizer.calculate_var_cvar(
            returns_df, max_sharpe_weights, confidence_level=0.95, method=method
        )

    return {
        "frontier": {
            "returns": rets,
            "risks": risks,
        },
        "max_sharpe": {
            "weights": max_sharpe_weights,
            "metrics": max_sharpe_metrics,
        },
        "min_vol": {
            "weights": min_vol_weights,
            "metrics": min_vol_metrics,
        },
        "var_cvar": var_results,
    }


def compute_tsla_forecast_ci(series: pd.Series, steps: int = 126, alpha: float = 0.10):
    # Use a simple ARIMA with a small grid search for robustness if available
    try:
        from src.model_selection import arima_order_grid_search  # optional
        order, _ = arima_order_grid_search(series, p_values=range(0, 4), d_values=range(0, 2), q_values=range(0, 4))
    except Exception:
        order = (2, 1, 2)

    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(series, order=order)
    fit = model.fit()
    pred = fit.get_forecast(steps=steps)
    mean = pred.predicted_mean
    conf = pred.conf_int(alpha=alpha)  # two-sided, e.g., 90% CI

    last_price = float(series.iloc[-1])
    end_mean = float(mean.iloc[-1])
    end_low = float(conf.iloc[-1, 0])
    end_high = float(conf.iloc[-1, 1])

    # Convert to cumulative return over horizon, then to approximate annualized equivalent
    cum_mean_ret = (end_mean / last_price) - 1.0
    cum_low_ret = (end_low / last_price) - 1.0
    cum_high_ret = (end_high / last_price) - 1.0

    # Annualize cumulative horizon return approximately based on trading days
    annualization_factor = 252 / steps
    ann_mean = (1.0 + cum_mean_ret) ** annualization_factor - 1.0
    ann_low = (1.0 + cum_low_ret) ** annualization_factor - 1.0
    ann_high = (1.0 + cum_high_ret) ** annualization_factor - 1.0

    return {
        "order": order,
        "horizon_days": steps,
        "mean_price": end_mean,
        "ci_low_price": end_low,
        "ci_high_price": end_high,
        "annualized_return_mean": float(ann_mean),
        "annualized_return_ci_low": float(ann_low),
        "annualized_return_ci_high": float(ann_high),
    }


def main():
    tickers = ["TSLA", "BND", "SPY"]
    start_date = "2015-01-01"
    end_date = "2024-01-01"

    dm = DataManager(data_source="yfinance")
    raw = {}
    for t in tickers:
        raw[t] = dm.fetch_data([t], start_date, end_date)

    # Extract Close robustly from potential MultiIndex
    def extract_close(df: pd.DataFrame, tk: str) -> pd.Series:
        if isinstance(df.columns, pd.MultiIndex):
            if (tk, "Close") in df.columns:
                return df[(tk, "Close")].rename(tk)
            if ("Close", tk) in df.columns:
                return df[("Close", tk)].rename(tk)
            # Try xs on last level
            if "Close" in df.columns.get_level_values(-1):
                cs = df.xs("Close", level=-1, axis=1)
                if tk in cs.columns:
                    return cs[tk].rename(tk)
                if cs.shape[1] == 1:
                    return cs.iloc[:, 0].rename(tk)
            raise KeyError("Close not found for ticker " + tk)
        if "Close" in df.columns:
            s = df["Close"].rename(tk)
            return s
        raise KeyError("Close column missing")

    close_map = {t: extract_close(raw[t], t) for t in tickers}
    prices = pd.concat(close_map, axis=1).dropna()

    # Returns
    returns = prices.pct_change().apply(lambda c: np.log(1 + c)).dropna()

    # Portfolio metrics
    metrics = compute_portfolio_metrics(returns, risk_free_rate=0.02)

    # TSLA forecast CI
    tsla_ci = compute_tsla_forecast_ci(prices["TSLA"], steps=126, alpha=0.10)

    output = {
        "as_of": datetime.utcnow().isoformat() + "Z",
        "tickers": tickers,
        "period": {"start": start_date, "end": end_date},
        "portfolio": metrics,
        "tsla_forecast": tsla_ci,
    }

    os.makedirs("artifacts", exist_ok=True)
    out_path = os.path.join("artifacts", "latest_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote metrics to {out_path}")


if __name__ == "__main__":
    main()


