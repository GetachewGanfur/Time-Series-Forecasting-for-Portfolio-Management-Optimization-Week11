"""
Configuration file for Time Series Forecasting for Portfolio Management Optimization
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
UTILS_DIR = PROJECT_ROOT / "utils"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"

# Data configuration
DEFAULT_DATA_SOURCE = "yfinance"  # Alternative: "quandl", "pandas_datareader"
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2024-01-01"
DEFAULT_FREQUENCY = "1d"  # Daily frequency

# Portfolio configuration
DEFAULT_PORTFOLIO_SIZE = 10
DEFAULT_RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
DEFAULT_TARGET_RETURN = 0.10   # 10% annual target return
DEFAULT_RISK_TOLERANCE = 0.15  # 15% risk tolerance

# Forecasting configuration
FORECAST_HORIZON = 30  # Days to forecast
TRAINING_WINDOW = 252  # 1 year of trading days
VALIDATION_WINDOW = 63  # 3 months for validation

# Model parameters
ARIMA_ORDER = (1, 1, 1)
SARIMA_ORDER = (1, 1, 1, 12)
LSTM_UNITS = 50
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 32

# Risk management
VAR_CONFIDENCE_LEVEL = 0.95
CVAR_CONFIDENCE_LEVEL = 0.95
MAX_POSITION_SIZE = 0.25  # Maximum 25% in single asset
MIN_POSITION_SIZE = 0.01  # Minimum 1% in single asset

# Backtesting configuration
REBALANCE_FREQUENCY = "monthly"  # Alternative: "weekly", "quarterly"
TRANSACTION_COSTS = 0.001  # 0.1% transaction costs
SLIPPAGE = 0.0005  # 0.05% slippage

# Visualization settings
FIGURE_SIZE = (12, 8)
DPI = 300
COLOR_PALETTE = "viridis"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# API configuration (if using external data sources)
QUANDL_API_KEY = os.getenv("QUANDL_API_KEY", "")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

# Performance metrics
BENCHMARK_TICKER = "SPY"  # S&P 500 ETF as benchmark
PERFORMANCE_METRICS = [
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "information_ratio",
    "tracking_error"
]
