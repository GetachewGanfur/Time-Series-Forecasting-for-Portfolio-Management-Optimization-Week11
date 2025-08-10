"""
Portfolio Optimization Module for Time Series Forecasting and Portfolio Management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

# Portfolio optimization libraries
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.risk_models import CovarianceShrinkage
    from pypfopt.objective_functions import L2_reg
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False
    warnings.warn("PyPortfolioOpt not available. Some optimization features will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Portfolio optimization using Modern Portfolio Theory and advanced risk management
    """

    def __init__(self, risk_free_rate: float = 0.02, target_return: float = 0.10):
        """
        Initialize PortfolioOptimizer

        Args:
            risk_free_rate: Annual risk-free rate
            target_return: Target annual return
        """
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        self.optimal_weights = None
        self.portfolio_stats = None

    def calculate_portfolio_metrics(self, returns: pd.DataFrame, weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics

        Args:
            returns: Asset returns DataFrame
            weights: Asset weights array

        Returns:
            Dictionary of portfolio metrics
        """
        try:
            # Portfolio returns
            portfolio_returns = np.sum(returns * weights, axis=1)
            
            # Portfolio statistics
            expected_return = np.mean(portfolio_returns) * 252  # Annualized
            volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
            
            # Risk metrics
            sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # VaR and CVaR (95% confidence)
            var_95 = np.percentile(portfolio_returns, 5)
            cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
            
            return {
                'expected_return': expected_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'return_volatility_ratio': expected_return / volatility if volatility > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            raise

    def optimize_portfolio_mpt(self, returns: pd.DataFrame, method: str = "efficient_frontier") -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Optimize portfolio using Modern Portfolio Theory

        Args:
            returns: Asset returns DataFrame
            method: Optimization method ('efficient_frontier', 'sharpe_max', 'risk_min')

        Returns:
            Tuple of (optimal_weights, portfolio_stats)
        """
        try:
            if method == "efficient_frontier" and PYPFOPT_AVAILABLE:
                return self._optimize_with_pypfopt(returns)
            else:
                return self._optimize_with_scipy(returns, method)

        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            raise

    def _optimize_with_pypfopt(self, returns: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
        """Optimize using PyPortfolioOpt library"""
        try:
            # Calculate expected returns and covariance matrix
            mu = expected_returns.mean_historical_return(returns)
            S = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
            
            # Create efficient frontier
            ef = EfficientFrontier(mu, S)
            ef.add_objective(L2_reg, gamma=0.1)  # Add regularization
            
            # Optimize for maximum Sharpe ratio
            ef.max_sharpe()
            weights = ef.clean_weights()
            
            # Convert to numpy array
            optimal_weights = np.array(list(weights.values()))
            
            # Calculate portfolio statistics
            portfolio_stats = self.calculate_portfolio_metrics(returns, optimal_weights)
            
            return optimal_weights, portfolio_stats
            
        except Exception as e:
            logger.error(f"Error in PyPortfolioOpt optimization: {e}")
            raise

    def _optimize_with_scipy(self, returns: pd.DataFrame, method: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """Optimize using scipy.optimize"""
        try:
            n_assets = returns.shape[1]
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            
            # Initial weights (equal weight)
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            ]
            
            # Bounds (0 <= weight <= 1)
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            if method == "sharpe_max":
                # Maximize Sharpe ratio
                def objective(weights):
                    portfolio_return = np.sum(expected_returns * weights)
                    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
                    return -sharpe  # Minimize negative Sharpe ratio
                    
            elif method == "risk_min":
                # Minimize portfolio risk
                def objective(weights):
                    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    return portfolio_vol
                    
            else:
                # Target return with minimum risk
                constraints.append({'type': 'eq', 'fun': lambda x: np.sum(expected_returns * x) - self.target_return})
                
                def objective(weights):
                    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    return portfolio_vol
            
            # Optimize
            result = minimize(objective, initial_weights, method='SLSQP',
                           bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                portfolio_stats = self.calculate_portfolio_metrics(returns, optimal_weights)
                return optimal_weights, portfolio_stats
            else:
                raise ValueError(f"Optimization failed: {result.message}")
                
        except Exception as e:
            logger.error(f"Error in scipy optimization: {e}")
            raise

    def generate_efficient_frontier(self, returns: pd.DataFrame, num_portfolios: int = 1000) -> Tuple[List[float], List[float], List[np.ndarray]]:
        """
        Generate efficient frontier by simulating random portfolios

        Args:
            returns: Asset returns DataFrame
            num_portfolios: Number of portfolios to simulate

        Returns:
            Tuple of (returns_list, risks_list, weights_list)
        """
        try:
            n_assets = returns.shape[1]
            expected_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            
            returns_list = []
            risks_list = []
            weights_list = []
            
            for _ in range(num_portfolios):
                # Generate random weights
                weights = np.random.random(n_assets)
                weights = weights / np.sum(weights)
                
                # Calculate portfolio return and risk
                portfolio_return = np.sum(expected_returns * weights)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                returns_list.append(portfolio_return)
                risks_list.append(portfolio_risk)
                weights_list.append(weights)
            
            return returns_list, risks_list, weights_list
            
        except Exception as e:
            logger.error(f"Error generating efficient frontier: {e}")
            raise

    def calculate_var_cvar(self, returns: pd.DataFrame, weights: np.ndarray, confidence_level: float = 0.95, method: str = 'historical') -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR) using multiple methods

        Args:
            returns: Asset returns DataFrame
            weights: Asset weights array
            confidence_level: Confidence level for VaR calculation
            method: Calculation method ('historical', 'parametric', 'monte_carlo')

        Returns:
            Dictionary with VaR and CVaR values
        """
        try:
            portfolio_returns = np.sum(returns * weights, axis=1)
            
            if method == 'historical':
                # Historical simulation
                var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
                cvar = np.mean(portfolio_returns[portfolio_returns <= var])
                
            elif method == 'parametric':
                # Parametric (normal distribution assumption)
                mu = portfolio_returns.mean()
                sigma = portfolio_returns.std()
                z_score = norm.ppf(1 - confidence_level)
                var = mu + z_score * sigma
                # CVaR for normal distribution
                cvar = mu - sigma * norm.pdf(z_score) / (1 - confidence_level)
                
            elif method == 'monte_carlo':
                # Monte Carlo simulation
                mu = portfolio_returns.mean()
                sigma = portfolio_returns.std()
                np.random.seed(42)
                simulated_returns = np.random.normal(mu, sigma, 10000)
                var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
                cvar = np.mean(simulated_returns[simulated_returns <= var])
            
            # Annualize
            var_annual = var * np.sqrt(252)
            cvar_annual = cvar * np.sqrt(252)
            
            return {
                'var': var_annual,
                'cvar': cvar_annual,
                'var_daily': var,
                'cvar_daily': cvar,
                'method': method,
                'confidence_level': confidence_level
            }
            
        except Exception as e:
            logger.error(f"Error calculating VaR/CVaR: {e}")
            raise

    def rebalance_portfolio(self, current_weights: np.ndarray, target_weights: np.ndarray, 
                          transaction_costs: float = 0.001) -> Dict[str, float]:
        """
        Calculate rebalancing costs and new weights

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            transaction_costs: Transaction cost per trade

        Returns:
            Dictionary with rebalancing information
        """
        try:
            # Calculate weight differences
            weight_diff = np.abs(target_weights - current_weights)
            
            # Calculate transaction costs
            total_transaction_cost = np.sum(weight_diff) * transaction_costs
            
            # Calculate turnover
            turnover = np.sum(weight_diff) / 2
            
            # Calculate effective weights after costs
            effective_weights = target_weights.copy()
            
            return {
                'transaction_costs': total_transaction_cost,
                'turnover': turnover,
                'effective_weights': effective_weights,
                'weight_changes': weight_diff
            }
            
        except Exception as e:
            logger.error(f"Error calculating rebalancing costs: {e}")
            raise

    def calculate_risk_contribution(self, returns: pd.DataFrame, weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate risk contribution of each asset to portfolio risk

        Args:
            returns: Asset returns DataFrame
            weights: Asset weights array

        Returns:
            Dictionary with risk contributions
        """
        try:
            cov_matrix = returns.cov() * 252
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Calculate marginal risk contribution
            marginal_risk = np.dot(cov_matrix, weights) / portfolio_risk
            
            # Calculate risk contribution
            risk_contribution = weights * marginal_risk
            
            # Convert to dictionary
            asset_names = returns.columns
            risk_contrib_dict = {asset: contrib for asset, contrib in zip(asset_names, risk_contribution)}
            
            return risk_contrib_dict
            
        except Exception as e:
            logger.error(f"Error calculating risk contribution: {e}")
            raise

    def optimize_risk_parity(self, returns: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Optimize portfolio using risk parity approach

        Args:
            returns: Asset returns DataFrame

        Returns:
            Tuple of (optimal_weights, portfolio_stats)
        """
        try:
            n_assets = returns.shape[1]
            cov_matrix = returns.cov() * 252
            
            # Initial weights
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Objective function: minimize the variance of risk contributions
            def objective(weights):
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                marginal_risk = np.dot(cov_matrix, weights) / portfolio_risk
                risk_contrib = weights * marginal_risk
                
                # Variance of risk contributions
                return np.var(risk_contrib)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            ]
            
            # Bounds
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Optimize
            result = minimize(objective, initial_weights, method='SLSQP',
                           bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                portfolio_stats = self.calculate_portfolio_metrics(returns, optimal_weights)
                return optimal_weights, portfolio_stats
            else:
                raise ValueError(f"Risk parity optimization failed: {result.message}")
                
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            raise
