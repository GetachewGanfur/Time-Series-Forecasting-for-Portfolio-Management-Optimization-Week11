"""
Advanced Financial Analysis Module
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
import warnings

logger = logging.getLogger(__name__)

class AdvancedFinancialAnalysis:
    """Advanced financial analysis tools for comprehensive EDA and statistical testing"""
    
    def __init__(self):
        self.results = {}
    
    def adf_stationarity_test(self, series: pd.Series, name: str = "") -> Dict[str, float]:
        """
        Perform Augmented Dickey-Fuller test for stationarity
        
        Args:
            series: Time series data
            name: Series name for logging
            
        Returns:
            Dictionary with test results
        """
        try:
            result = adfuller(series.dropna(), autolag='AIC')
            
            adf_result = {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values_1%': result[4]['1%'],
                'critical_values_5%': result[4]['5%'],
                'critical_values_10%': result[4]['10%'],
                'is_stationary': result[1] < 0.05
            }
            
            logger.info(f"ADF Test for {name}: Stationary = {adf_result['is_stationary']}")
            return adf_result
            
        except Exception as e:
            logger.error(f"Error in ADF test: {e}")
            raise
    
    def kpss_stationarity_test(self, series: pd.Series, name: str = "") -> Dict[str, float]:
        """KPSS test for stationarity (null hypothesis: stationary)"""
        try:
            result = kpss(series.dropna(), regression='c')
            
            kpss_result = {
                'kpss_statistic': result[0],
                'p_value': result[1],
                'critical_values_1%': result[3]['1%'],
                'critical_values_5%': result[3]['5%'],
                'critical_values_10%': result[3]['10%'],
                'is_stationary': result[1] > 0.05
            }
            
            return kpss_result
            
        except Exception as e:
            logger.error(f"Error in KPSS test: {e}")
            raise
    
    def ljung_box_test(self, series: pd.Series, lags: int = 10) -> Dict[str, float]:
        """Ljung-Box test for autocorrelation"""
        try:
            result = acorr_ljungbox(series.dropna(), lags=lags, return_df=True)
            
            return {
                'lb_stat': result['lb_stat'].iloc[-1],
                'lb_pvalue': result['lb_pvalue'].iloc[-1],
                'has_autocorr': result['lb_pvalue'].iloc[-1] < 0.05
            }
            
        except Exception as e:
            logger.error(f"Error in Ljung-Box test: {e}")
            raise
    
    def jarque_bera_normality_test(self, series: pd.Series) -> Dict[str, float]:
        """Jarque-Bera test for normality"""
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(series.dropna())
            
            return {
                'jb_statistic': jb_stat,
                'jb_pvalue': jb_pvalue,
                'is_normal': jb_pvalue > 0.05
            }
            
        except Exception as e:
            logger.error(f"Error in Jarque-Bera test: {e}")
            raise
    
    def arch_effect_test(self, series: pd.Series) -> Dict[str, float]:
        """Test for ARCH effects (volatility clustering)"""
        try:
            # Fit ARCH model to test for heteroscedasticity
            model = arch_model(series.dropna(), vol='ARCH', p=1)
            fitted_model = model.fit(disp='off')
            
            return {
                'arch_lm_stat': fitted_model.arch_lm_test().stat,
                'arch_lm_pvalue': fitted_model.arch_lm_test().pvalue,
                'has_arch_effect': fitted_model.arch_lm_test().pvalue < 0.05
            }
            
        except Exception as e:
            logger.warning(f"ARCH test failed: {e}")
            return {'arch_lm_stat': np.nan, 'arch_lm_pvalue': np.nan, 'has_arch_effect': False}
    
    def comprehensive_statistical_analysis(self, returns_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Perform comprehensive statistical analysis on returns data
        
        Args:
            returns_df: DataFrame of asset returns
            
        Returns:
            Dictionary with comprehensive test results
        """
        results = {}
        
        for asset in returns_df.columns:
            series = returns_df[asset]
            
            asset_results = {
                'descriptive_stats': {
                    'mean': series.mean(),
                    'std': series.std(),
                    'skewness': stats.skew(series.dropna()),
                    'kurtosis': stats.kurtosis(series.dropna()),
                    'min': series.min(),
                    'max': series.max(),
                    'var': series.var()
                },
                'stationarity_tests': {
                    'adf': self.adf_stationarity_test(series, asset),
                    'kpss': self.kpss_stationarity_test(series, asset)
                },
                'autocorrelation_test': self.ljung_box_test(series),
                'normality_test': self.jarque_bera_normality_test(series),
                'arch_test': self.arch_effect_test(series)
            }
            
            results[asset] = asset_results
        
        self.results = results
        return results
    
    def monte_carlo_var(self, returns: pd.Series, confidence_level: float = 0.95, 
                       num_simulations: int = 10000, time_horizon: int = 1) -> Dict[str, float]:
        """
        Monte Carlo Value at Risk calculation
        
        Args:
            returns: Historical returns
            confidence_level: Confidence level (0.95 for 95%)
            num_simulations: Number of Monte Carlo simulations
            time_horizon: Time horizon in days
            
        Returns:
            Dictionary with VaR results
        """
        try:
            # Calculate parameters
            mu = returns.mean()
            sigma = returns.std()
            
            # Generate random returns
            np.random.seed(42)
            simulated_returns = np.random.normal(mu, sigma, num_simulations) * np.sqrt(time_horizon)
            
            # Calculate VaR
            var_level = (1 - confidence_level) * 100
            var = np.percentile(simulated_returns, var_level)
            
            # Calculate Expected Shortfall (CVaR)
            cvar = simulated_returns[simulated_returns <= var].mean()
            
            return {
                'monte_carlo_var': var,
                'monte_carlo_cvar': cvar,
                'confidence_level': confidence_level,
                'num_simulations': num_simulations,
                'time_horizon': time_horizon
            }
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo VaR: {e}")
            raise
    
    def parametric_var(self, returns: pd.Series, confidence_level: float = 0.95) -> Dict[str, float]:
        """Parametric VaR assuming normal distribution"""
        try:
            mu = returns.mean()
            sigma = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            
            var = mu + z_score * sigma
            
            return {
                'parametric_var': var,
                'confidence_level': confidence_level,
                'mean': mu,
                'std': sigma
            }
            
        except Exception as e:
            logger.error(f"Error in parametric VaR: {e}")
            raise
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive statistical analysis report"""
        if not self.results:
            return "No analysis results available. Run comprehensive_statistical_analysis first."
        
        report = "=" * 80 + "\n"
        report += "COMPREHENSIVE FINANCIAL STATISTICAL ANALYSIS REPORT\n"
        report += "=" * 80 + "\n\n"
        
        for asset, results in self.results.items():
            report += f"\n{asset} ANALYSIS:\n" + "-" * 40 + "\n"
            
            # Descriptive Statistics
            desc = results['descriptive_stats']
            report += f"Descriptive Statistics:\n"
            report += f"  Mean Return: {desc['mean']:.6f}\n"
            report += f"  Volatility: {desc['std']:.6f}\n"
            report += f"  Skewness: {desc['skewness']:.4f}\n"
            report += f"  Kurtosis: {desc['kurtosis']:.4f}\n"
            
            # Stationarity Tests
            adf = results['stationarity_tests']['adf']
            report += f"\nStationarity Tests:\n"
            report += f"  ADF Test: {'Stationary' if adf['is_stationary'] else 'Non-Stationary'} (p-value: {adf['p_value']:.4f})\n"
            
            # Normality Test
            norm = results['normality_test']
            report += f"  Normality: {'Normal' if norm['is_normal'] else 'Non-Normal'} (p-value: {norm['jb_pvalue']:.4f})\n"
            
            # Autocorrelation
            ljung = results['autocorrelation_test']
            report += f"  Autocorrelation: {'Present' if ljung['has_autocorr'] else 'Absent'} (p-value: {ljung['lb_pvalue']:.4f})\n"
            
            report += "\n"
        
        return report