"""
Backtesting Module for Portfolio Performance Evaluation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioBacktester:
    """
    Portfolio backtesting framework for evaluating trading strategies
    """

    def __init__(self, initial_capital: float = 100000, transaction_costs: float = 0.001):
        """
        Initialize PortfolioBacktester

        Args:
            initial_capital: Initial portfolio value
            transaction_costs: Transaction cost per trade
        """
        self.initial_capital = initial_capital
        self.transaction_costs = transaction_costs
        self.portfolio_history = []
        self.trade_history = []
        self.current_weights = None
        self.current_value = initial_capital

    def run_backtest(self, 
                    returns: pd.DataFrame, 
                    weights: pd.DataFrame,
                    rebalance_frequency: str = 'monthly') -> Dict[str, pd.DataFrame]:
        """
        Run portfolio backtest

        Args:
            returns: Asset returns DataFrame
            weights: Target weights DataFrame (same index as returns)
            rebalance_frequency: Rebalancing frequency ('daily', 'weekly', 'monthly')

        Returns:
            Dictionary with backtest results
        """
        try:
            # Initialize tracking variables
            portfolio_values = []
            current_weights = np.array([1/len(returns.columns)] * len(returns.columns))
            current_value = self.initial_capital
            
            # Determine rebalancing dates
            rebalance_dates = self._get_rebalance_dates(returns.index, rebalance_frequency)
            
            # Run backtest
            for i, date in enumerate(returns.index):
                # Get current returns
                daily_returns = returns.loc[date]
                
                # Update portfolio value
                portfolio_return = np.sum(current_weights * daily_returns)
                current_value *= (1 + portfolio_return)
                
                # Check if rebalancing is needed
                if date in rebalance_dates:
                    target_weights = weights.loc[date].values
                    current_weights, rebalance_cost = self._rebalance_portfolio(
                        current_weights, target_weights, current_value
                    )
                    current_value -= rebalance_cost
                
                # Record portfolio state
                portfolio_values.append({
                    'date': date,
                    'portfolio_value': current_value,
                    'weights': current_weights.copy(),
                    'daily_return': portfolio_return
                })
            
            # Convert to DataFrame
            portfolio_df = pd.DataFrame(portfolio_values)
            portfolio_df.set_index('date', inplace=True)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(portfolio_df)
            
            return {
                'portfolio_values': portfolio_df,
                'performance_metrics': performance_metrics,
                'trade_history': self.trade_history
            }
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            raise

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex, frequency: str) -> List[datetime]:
        """Get rebalancing dates based on frequency"""
        if frequency == 'daily':
            return list(dates)
        elif frequency == 'weekly':
            return [d for d in dates if d.weekday() == 0]  # Monday
        elif frequency == 'monthly':
            return [d for d in dates if d.day == 1]
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")

    def _rebalance_portfolio(self, 
                           current_weights: np.ndarray, 
                           target_weights: np.ndarray,
                           current_value: float) -> Tuple[np.ndarray, float]:
        """Rebalance portfolio and calculate costs"""
        # Calculate weight differences
        weight_diff = np.abs(target_weights - current_weights)
        
        # Calculate transaction costs
        total_cost = np.sum(weight_diff) * current_value * self.transaction_costs
        
        # Record trade
        self.trade_history.append({
            'date': datetime.now(),
            'current_weights': current_weights.copy(),
            'target_weights': target_weights.copy(),
            'transaction_cost': total_cost
        })
        
        return target_weights, total_cost

    def _calculate_performance_metrics(self, portfolio_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        try:
            # Calculate cumulative returns
            portfolio_df['cumulative_return'] = (portfolio_df['portfolio_value'] / self.initial_capital) - 1
            
            # Calculate daily returns
            portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
            
            # Annualized metrics
            total_days = len(portfolio_df)
            total_return = portfolio_df['cumulative_return'].iloc[-1]
            annualized_return = (1 + total_return) ** (252 / total_days) - 1
            
            # Volatility
            daily_returns = portfolio_df['daily_return'].dropna()
            annualized_volatility = daily_returns.std() * np.sqrt(252)
            
            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = portfolio_df['cumulative_return'] + 1
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # VaR and CVaR
            var_95 = np.percentile(daily_returns, 5)
            cvar_95 = np.mean(daily_returns[daily_returns <= var_95])
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'calmar_ratio': calmar_ratio
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            raise

    def compare_strategies(self, 
                          returns: pd.DataFrame,
                          strategies: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compare multiple trading strategies

        Args:
            returns: Asset returns DataFrame
            strategies: Dictionary of strategy weights

        Returns:
            DataFrame with strategy comparison
        """
        try:
            comparison_results = {}
            
            for strategy_name, weights in strategies.items():
                # Run backtest for this strategy
                results = self.run_backtest(returns, weights)
                comparison_results[strategy_name] = results['performance_metrics']
            
            # Convert to DataFrame
            comparison_df = pd.DataFrame(comparison_results).T
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {e}")
            raise

    def calculate_rolling_metrics(self, 
                                portfolio_df: pd.DataFrame, 
                                window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling performance metrics

        Args:
            portfolio_df: Portfolio values DataFrame
            window: Rolling window size (default: 252 trading days)

        Returns:
            DataFrame with rolling metrics
        """
        try:
            rolling_metrics = portfolio_df.copy()
            
            # Rolling returns
            rolling_metrics['rolling_return'] = portfolio_df['portfolio_value'].pct_change(window)
            
            # Rolling volatility
            rolling_metrics['rolling_volatility'] = (
                portfolio_df['daily_return'].rolling(window).std() * np.sqrt(252)
            )
            
            # Rolling Sharpe ratio
            rolling_metrics['rolling_sharpe'] = (
                rolling_metrics['rolling_return'] / rolling_metrics['rolling_volatility']
            )
            
            # Rolling maximum drawdown
            rolling_metrics['rolling_max_dd'] = portfolio_df['portfolio_value'].rolling(window).apply(
                lambda x: self._calculate_rolling_drawdown(x)
            )
            
            return rolling_metrics.dropna()
            
        except Exception as e:
            logger.error(f"Error calculating rolling metrics: {e}")
            raise

    def _calculate_rolling_drawdown(self, values: pd.Series) -> float:
        """Calculate rolling maximum drawdown"""
        cumulative = (values / values.iloc[0]) - 1
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

    def generate_report(self, backtest_results: Dict[str, pd.DataFrame]) -> str:
        """
        Generate comprehensive backtest report

        Args:
            backtest_results: Results from run_backtest

        Returns:
            Formatted report string
        """
        try:
            portfolio_df = backtest_results['portfolio_values']
            metrics = backtest_results['performance_metrics']
            
            report = f"""
            ========================================
            PORTFOLIO BACKTEST REPORT
            ========================================
            
            Initial Capital: ${self.initial_capital:,.2f}
            Final Value: ${portfolio_df['portfolio_value'].iloc[-1]:,.2f}
            Total Return: {metrics['total_return']:.2%}
            
            PERFORMANCE METRICS:
            - Annualized Return: {metrics['annualized_return']:.2%}
            - Annualized Volatility: {metrics['annualized_volatility']:.2%}
            - Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
            - Maximum Drawdown: {metrics['max_drawdown']:.2%}
            - VaR (95%): {metrics['var_95']:.2%}
            - Calmar Ratio: {metrics['calmar_ratio']:.3f}
            
            TRADING SUMMARY:
            - Total Trades: {len(backtest_results['trade_history'])}
            - Rebalancing Frequency: Monthly
            - Transaction Costs: {self.transaction_costs:.3%}
            
            ========================================
            """
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
