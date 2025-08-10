"""
Basic tests for the portfolio management system
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import modules
from data.data_manager import DataManager
from utils.portfolio_optimizer import PortfolioOptimizer
from utils.backtesting import PortfolioBacktester
from utils.visualization import FinancialVisualizer

class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality of the system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data_manager = DataManager()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.backtester = PortfolioBacktester(initial_capital=10000)
        self.visualizer = FinancialVisualizer()
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        self.sample_returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, len(dates)),
            'GOOGL': np.random.normal(0.0008, 0.025, len(dates)),
            'MSFT': np.random.normal(0.0012, 0.018, len(dates))
        }, index=dates)
    
    def test_data_manager_initialization(self):
        """Test DataManager initialization"""
        self.assertEqual(self.data_manager.data_source, "yfinance")
        self.assertIsInstance(self.data_manager.data_cache, dict)
    
    def test_portfolio_optimizer_initialization(self):
        """Test PortfolioOptimizer initialization"""
        self.assertEqual(self.portfolio_optimizer.risk_free_rate, 0.02)
        self.assertEqual(self.portfolio_optimizer.target_return, 0.10)
    
    def test_portfolio_metrics_calculation(self):
        """Test portfolio metrics calculation"""
        weights = np.array([0.4, 0.3, 0.3])
        metrics = self.portfolio_optimizer.calculate_portfolio_metrics(self.sample_returns, weights)
        
        self.assertIn('expected_return', metrics)
        self.assertIn('volatility', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIsInstance(metrics['expected_return'], float)
    
    def test_efficient_frontier_generation(self):
        """Test efficient frontier generation"""
        returns_list, risks_list, weights_list = self.portfolio_optimizer.generate_efficient_frontier(
            self.sample_returns, num_portfolios=100
        )
        
        self.assertEqual(len(returns_list), 100)
        self.assertEqual(len(risks_list), 100)
        self.assertEqual(len(weights_list), 100)
    
    def test_backtester_initialization(self):
        """Test PortfolioBacktester initialization"""
        self.assertEqual(self.backtester.initial_capital, 10000)
        self.assertEqual(self.backtester.transaction_costs, 0.001)

if __name__ == '__main__':
    unittest.main()