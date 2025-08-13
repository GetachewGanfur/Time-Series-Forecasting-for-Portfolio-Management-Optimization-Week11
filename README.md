# Time Series Forecasting for Portfolio Management Optimization

[GitHub Repository](https://github.com/GetachewGanfur/Time-Series-Forecasting-for-Portfolio-Management-Optimization-Week11.git)

## Overview
This project provides a comprehensive, step-by-step framework for optimizing portfolio management using advanced time series forecasting. By integrating both statistical and machine learning models with financial optimization strategies, the framework enables users to adaptively allocate assets and manage risk in dynamic market environments.

## Features
- **Diverse Forecasting Models**: Includes ARIMA, SARIMA, Prophet, LSTM, and ensemble approaches
- **Portfolio Optimization Techniques**: Implements Modern Portfolio Theory and risk-parity methods
- **Risk Management Tools**: Supports Historical, Parametric, and Monte Carlo VaR/CVaR calculations
- **Statistical Analysis Suite**: Provides ADF stationarity, Jarque-Bera normality, and Ljung-Box autocorrelation tests
- **Data Processing Enhancements**: Handles missing data and time-based interpolation; basic IQR outlier scan in pipeline
- **Rolling Analytics**: Calculates rolling volatility (30-day), rolling Sharpe ratios, and dynamic risk metrics
- **Backtesting Framework**: Evaluates performance with transaction cost considerations
- **Visualization Tools**: Generates efficient frontiers, correlation heatmaps, and drawdown analyses
- **Real-time Data Integration**: Integrates YFinance with robust error handling

1. **Data Acquisition**
   - Fetches historical and real-time financial data using YFinance with robust error handling.
   - Supports multiple asset classes and customizable timeframes.

2. **Data Preprocessing**
   - Cleans raw financial data by handling missing values, detecting and treating outliers, and performing time-based interpolation.
   - Conducts feature engineering to extract relevant financial indicators and technical features.

3. **Statistical Analysis**
   - Performs stationarity tests (ADF), normality checks (Jarque-Bera), and autocorrelation analysis (Ljung-Box) to assess data suitability for modeling.
   - Generates summary statistics and visualizations for exploratory data analysis.

4. **Forecasting Engine**
   - Implements a suite of time series forecasting models, including ARIMA, SARIMA, Prophet, LSTM, and ensemble methods.
   - Supports model selection, hyperparameter tuning, and cross-validation for optimal forecasting accuracy.

5. **Portfolio Optimization**
   - Applies Modern Portfolio Theory and risk-parity optimization techniques to construct efficient portfolios.
   - Incorporates historical/forecasted returns and risk metrics into the optimization process.

6. **Risk Management**
   - Calculates Value at Risk (VaR) and Conditional Value at Risk (CVaR) using Historical, Parametric, and Monte Carlo methods.
   - Provides dynamic risk allocation and real-time risk monitoring tools.

7. **Rolling Analytics**
   - Computes rolling volatility (e.g., 30-day), rolling Sharpe ratios, and other dynamic risk/return metrics to monitor portfolio performance over time.

8. **Backtesting Framework**
   - Simulates historical portfolio performance, accounting for transaction costs and slippage.
   - Compares different strategies and models using standardized performance metrics.

9. **Visualization and Reporting**
   - Generates efficient frontiers, correlation heatmaps, drawdown plots, and other insightful visualizations.
   - Produces comprehensive reports summarizing key findings and performance.

## Feature Summary
- **Forecasting**: ARIMA, SARIMA, Prophet, LSTM, ensemble
- **Optimization**: Modern Portfolio Theory, risk parity
- **Risk**: VaR/CVaR (historical, parametric, Monte Carlo)
- **Statistics**: ADF, Jarque-Bera, Ljung-Box
- **Processing**: Missing data handling; IQR outlier scan
- **Analytics**: Rolling volatility and Sharpe
- **Backtesting**: Transaction costs, monthly rebalancing
- **Visualization**: Efficient frontier, heatmaps, drawdown

## Project Structure

```
.
├─ config/
│  └─ config.py
├─ data/
│  ├─ processed_prices.csv
│  ├─ processed_returns.csv
│  └─ rolling_volatility.csv
├─ models/
│  ├─ __init__.py
│  └─ forecasting_models.py
├─ notebooks/
│  ├─ task1_data_preprocessing.ipynb
│  ├─ task2_tsla_forecasting.ipynb
│  ├─ task3_future_trends.ipynb
│  ├─ task4_portfolio_optimization.ipynb
│  └─ task5_strategy_backtest.ipynb
├─ src/
│  ├─ __init__.py
│  ├─ backtesting.py
│  ├─ data_manager.py
│  ├─ financial_analysis.py
│  ├─ forecasting_analysis.py
│  ├─ main.py
│  ├─ model_selection.py
│  ├─ portfolio_optimizer.py
│  ├─ task4_data.py
│  ├─ task4_optimizer.py
│  ├─ task5_backtest.py
│  └─ visualization.py
├─ tests/
│  └─ test_basic.py
├─ requirements.txt
└─ README.md
```

## Setup

1) Create environment and install dependencies

```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows
pip install -r requirements.txt
```

2) Optional: Set API keys (if using external sources)

Create a `.env` file with keys like `QUANDL_API_KEY` if needed.

## How to Run

- Run the end-to-end pipeline (data → analysis → optimization → backtest → visuals):

```bash
python -m src.main
```

Artifacts (saved figures) will be written to `artifacts/figures`, e.g.:
- `prices.png`
- `returns_distribution.png`
- `rolling_volatility.png`
- `correlation_heatmap.png`
- `efficient_frontier.png`
- `portfolio_weights.png`
- `performance.png`
- `drawdown.png`

- Work through the notebooks in order:
  1. `notebooks/task1_data_preprocessing.ipynb`
  2. `notebooks/task2_tsla_forecasting.ipynb`
  3. `notebooks/task3_future_trends.ipynb`
  4. `notebooks/task4_portfolio_optimization.ipynb`
  5. `notebooks/task5_strategy_backtest.ipynb`

## Testing

```bash
pytest -q
```

Tip: Run a single integration test quickly:
```bash
python -m pytest tests/test_integration.py -q
```

## Notes

- Requires internet access to download market data via YFinance.
- Visualizations are shown interactively; run in a notebook or an environment with a display backend.
