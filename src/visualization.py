"""
Visualization Module for Financial Data and Portfolio Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
from datetime import datetime

# Configure plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class FinancialVisualizer:
    """
    Comprehensive visualization tools for financial data analysis
    """

    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize FinancialVisualizer

        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    def plot_price_series(self, 
                         data: Dict[str, pd.DataFrame], 
                         column: str = 'Close',
                         title: str = 'Asset Prices Over Time') -> None:
        """
        Plot price series for multiple assets

        Args:
            data: Dictionary of asset data
            column: Column to plot (default: 'Close')
            title: Plot title
        """
        plt.figure(figsize=(15, 8))
        
        for i, (ticker, ticker_data) in enumerate(data.items()):
            plt.plot(ticker_data.index, ticker_data[column], 
                    label=ticker, linewidth=1.5, color=self.colors[i % len(self.colors)])
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel(f'{column} Price ($)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_returns_distribution(self, 
                                returns: pd.DataFrame,
                                title: str = 'Daily Returns Distribution') -> None:
        """
        Plot distribution of daily returns

        Args:
            returns: DataFrame of daily returns
            title: Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i, ticker in enumerate(returns.columns):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # Histogram
            ax.hist(returns[ticker].dropna(), bins=50, alpha=0.7, color=self.colors[i])
            ax.set_title(f'{ticker} Returns Distribution')
            ax.set_xlabel('Daily Return')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add normal distribution overlay
            mu, sigma = returns[ticker].mean(), returns[ticker].std()
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
            y = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            ax.plot(x, y * len(returns[ticker].dropna()) * (returns[ticker].max() - returns[ticker].min()) / 50, 
                   'r-', linewidth=2, label='Normal Distribution')
            ax.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_rolling_volatility(self, 
                               returns: pd.DataFrame,
                               window: int = 30,
                               title: str = 'Rolling Volatility') -> None:
        """
        Plot rolling volatility for multiple assets

        Args:
            returns: DataFrame of daily returns
            window: Rolling window size
            title: Plot title
        """
        plt.figure(figsize=(15, 8))
        
        for i, ticker in enumerate(returns.columns):
            rolling_vol = returns[ticker].rolling(window).std() * np.sqrt(252) * 100
            plt.plot(returns.index[window-1:], rolling_vol, 
                    label=f'{ticker} ({window}-day)', linewidth=1.5, 
                    color=self.colors[i % len(self.colors)])
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Annualized Volatility (%)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self, 
                                returns: pd.DataFrame,
                                title: str = 'Asset Returns Correlation Matrix') -> None:
        """
        Plot correlation heatmap for asset returns

        Args:
            returns: DataFrame of daily returns
            title: Plot title
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_efficient_frontier(self, 
                              returns_list: List[float],
                              risks_list: List[float],
                              weights_list: List[np.ndarray],
                              asset_names: List[str],
                              title: str = 'Efficient Frontier') -> None:
        """
        Plot efficient frontier with portfolio points

        Args:
            returns_list: List of portfolio returns
            risks_list: List of portfolio risks
            weights_list: List of portfolio weights
            asset_names: List of asset names
            title: Plot title
        """
        plt.figure(figsize=(12, 8))
        
        # Plot all portfolios
        plt.scatter(risks_list, returns_list, c=returns_list, cmap='viridis', 
                   alpha=0.6, s=30, label='Random Portfolios')
        
        # Find and highlight optimal portfolios
        max_sharpe_idx = np.argmax([r/r for r, risk in zip(returns_list, risks_list) if risk > 0])
        min_risk_idx = np.argmin(risks_list)
        
        plt.scatter(risks_list[max_sharpe_idx], returns_list[max_sharpe_idx], 
                   color='red', s=200, marker='*', label='Maximum Sharpe Ratio')
        plt.scatter(risks_list[min_risk_idx], returns_list[min_risk_idx], 
                   color='green', s=200, marker='*', label='Minimum Risk')
        
        plt.xlabel('Portfolio Risk (Volatility)', fontsize=12)
        plt.ylabel('Portfolio Return', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.colorbar(label='Return')
        plt.tight_layout()
        plt.show()

    def plot_portfolio_weights(self, 
                             weights: np.ndarray,
                             asset_names: List[str],
                             title: str = 'Portfolio Allocation') -> None:
        """
        Plot portfolio weights as a bar chart

        Args:
            weights: Portfolio weights array
            asset_names: List of asset names
            title: Plot title
        """
        plt.figure(figsize=(10, 6))
        
        bars = plt.bar(asset_names, weights, color=self.colors[:len(asset_names)], alpha=0.7)
        
        # Add value labels on bars
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{weight:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Assets', fontsize=12)
        plt.ylabel('Weight', fontsize=12)
        plt.ylim(0, max(weights) * 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()

    def plot_portfolio_performance(self, 
                                 portfolio_values: pd.DataFrame,
                                 benchmark_values: Optional[pd.Series] = None,
                                 title: str = 'Portfolio Performance') -> None:
        """
        Plot portfolio performance over time

        Args:
            portfolio_values: Portfolio values DataFrame
            benchmark_values: Benchmark values series (optional)
            title: Plot title
        """
        plt.figure(figsize=(15, 8))
        
        # Plot portfolio value
        plt.plot(portfolio_values.index, portfolio_values['portfolio_value'], 
                label='Portfolio', linewidth=2, color='blue')
        
        # Plot benchmark if provided
        if benchmark_values is not None:
            # Normalize benchmark to start at same value
            benchmark_normalized = benchmark_values * (portfolio_values['portfolio_value'].iloc[0] / benchmark_values.iloc[0])
            plt.plot(benchmark_values.index, benchmark_normalized, 
                    label='Benchmark (SPY)', linewidth=2, color='red', alpha=0.7)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_drawdown(self, 
                     portfolio_values: pd.DataFrame,
                     title: str = 'Portfolio Drawdown') -> None:
        """
        Plot portfolio drawdown over time

        Args:
            portfolio_values: Portfolio values DataFrame
            title: Plot title
        """
        plt.figure(figsize=(15, 8))
        
        # Calculate drawdown
        portfolio_value = portfolio_values['portfolio_value']
        running_max = np.maximum.accumulate(portfolio_value)
        drawdown = (portfolio_value - running_max) / running_max * 100
        
        plt.fill_between(portfolio_values.index, drawdown, 0, alpha=0.3, color='red')
        plt.plot(portfolio_values.index, drawdown, color='red', linewidth=1)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_rolling_metrics(self, 
                           rolling_metrics: pd.DataFrame,
                           title: str = 'Rolling Portfolio Metrics') -> None:
        """
        Plot rolling portfolio metrics

        Args:
            rolling_metrics: DataFrame with rolling metrics
            title: Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Rolling returns
        axes[0, 0].plot(rolling_metrics.index, rolling_metrics['rolling_return'] * 100)
        axes[0, 0].set_title('Rolling Returns (252-day)')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Rolling volatility
        axes[0, 1].plot(rolling_metrics.index, rolling_metrics['rolling_volatility'] * 100)
        axes[0, 1].set_title('Rolling Volatility (252-day)')
        axes[0, 1].set_ylabel('Volatility (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        axes[1, 0].plot(rolling_metrics.index, rolling_metrics['rolling_sharpe'])
        axes[1, 0].set_title('Rolling Sharpe Ratio (252-day)')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Rolling max drawdown
        axes[1, 1].plot(rolling_metrics.index, rolling_metrics['rolling_max_dd'] * 100)
        axes[1, 1].set_title('Rolling Max Drawdown (252-day)')
        axes[1, 1].set_ylabel('Drawdown (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_forecast_comparison(self, 
                               actual: pd.Series,
                               forecasts: Dict[str, np.ndarray],
                               title: str = 'Forecast Comparison') -> None:
        """
        Plot actual vs forecasted values

        Args:
            actual: Actual values series
            forecasts: Dictionary of forecast arrays
            title: Plot title
        """
        plt.figure(figsize=(15, 8))
        
        # Plot actual values
        plt.plot(actual.index, actual.values, label='Actual', linewidth=2, color='black')
        
        # Plot forecasts
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            forecast_index = pd.date_range(start=actual.index[-1], periods=len(forecast)+1, freq='D')[1:]
            plt.plot(forecast_index, forecast, label=f'{model_name} Forecast', 
                    linewidth=2, color=colors[i % len(colors)], linestyle='--')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def create_dashboard(self, 
                        data: Dict[str, pd.DataFrame],
                        returns: pd.DataFrame,
                        portfolio_values: Optional[pd.DataFrame] = None) -> None:
        """
        Create a comprehensive dashboard with multiple plots

        Args:
            data: Dictionary of asset data
            returns: DataFrame of daily returns
            portfolio_values: Portfolio values DataFrame (optional)
        """
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Financial Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # Price series
        ax1 = plt.subplot(3, 3, 1)
        for ticker, ticker_data in data.items():
            ax1.plot(ticker_data.index, ticker_data['Close'], label=ticker, linewidth=1)
        ax1.set_title('Asset Prices')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Returns distribution
        ax2 = plt.subplot(3, 3, 2)
        for ticker in returns.columns:
            ax2.hist(returns[ticker].dropna(), bins=30, alpha=0.5, label=ticker)
        ax2.set_title('Returns Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Correlation heatmap
        ax3 = plt.subplot(3, 3, 3)
        corr_matrix = returns.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3, square=True)
        ax3.set_title('Correlation Matrix')
        
        # Rolling volatility
        ax4 = plt.subplot(3, 3, 4)
        for ticker in returns.columns:
            rolling_vol = returns[ticker].rolling(30).std() * np.sqrt(252) * 100
            ax4.plot(returns.index[29:], rolling_vol, label=ticker, linewidth=1)
        ax4.set_title('30-Day Rolling Volatility')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Portfolio performance (if available)
        if portfolio_values is not None:
            ax5 = plt.subplot(3, 3, 5)
            ax5.plot(portfolio_values.index, portfolio_values['portfolio_value'])
            ax5.set_title('Portfolio Performance')
            ax5.grid(True, alpha=0.3)
            
            # Drawdown
            ax6 = plt.subplot(3, 3, 6)
            portfolio_value = portfolio_values['portfolio_value']
            running_max = np.maximum.accumulate(portfolio_value)
            drawdown = (portfolio_value - running_max) / running_max * 100
            ax6.fill_between(portfolio_values.index, drawdown, 0, alpha=0.3, color='red')
            ax6.set_title('Portfolio Drawdown')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
