"""
Data Management Module for Time Series Forecasting and Portfolio Optimization
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from typing import List, Dict, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    """
    Manages financial data operations including fetching, cleaning, and preprocessing
    """
    
    def __init__(self, data_source: str = "yfinance"):
        """
        Initialize DataManager
        
        Args:
            data_source: Data source to use ('yfinance', 'pandas_datareader')
        """
        self.data_source = data_source
        self.data_cache = {}
        
    def fetch_data(self, 
                   tickers: Union[List[str], str], 
                   start_date: str = "2020-01-01",
                   end_date: str = "2024-01-01",
                   frequency: str = "1d") -> pd.DataFrame:
        """
        Fetch financial data for given tickers
        
        Args:
            tickers: Single ticker symbol or list of stock tickers
            start_date: Start date for data
            end_date: End date for data
            frequency: Data frequency ('1d', '1wk', '1mo')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Normalize tickers to a list[str]
            if isinstance(tickers, str):
                tickers = [tickers]

            if self.data_source == "yfinance":
                return self._fetch_yfinance_data(tickers, start_date, end_date, frequency)
            elif self.data_source == "pandas_datareader":
                return self._fetch_pandas_datareader_data(tickers, start_date, end_date, frequency)
            else:
                raise ValueError(f"Unsupported data source: {self.data_source}")
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def _fetch_yfinance_data(self, tickers: List[str], start_date: str, end_date: str, frequency: str) -> pd.DataFrame:
        """Fetch data using yfinance"""
        try:
            # Use yfinance directly
            data = yf.download(tickers, start=start_date, end=end_date, interval=frequency)
            
            # Handle single ticker case: yfinance may return single-index columns; make them MultiIndex
            if len(tickers) == 1 and not isinstance(data.columns, pd.MultiIndex):
                data.columns = pd.MultiIndex.from_product([tickers, data.columns])
            
            return data
        except Exception as e:
            logger.error(f"Error fetching yfinance data: {e}")
            raise
    
    def _fetch_pandas_datareader_data(self, tickers: List[str], start_date: str, end_date: str, frequency: str) -> pd.DataFrame:
        """Fetch data using pandas_datareader"""
        try:
            data_dict = {}
            for ticker in tickers:
                ticker_data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
                data_dict[ticker] = ticker_data
            
            # Combine all tickers
            combined_data = pd.concat(data_dict, axis=1)
            return combined_data
        except Exception as e:
            logger.error(f"Error fetching pandas_datareader data: {e}")
            raise
    
    def calculate_returns(self, data: pd.DataFrame, method: str = "log") -> pd.DataFrame:
        """
        Calculate returns from price data
        
        Args:
            data: OHLCV data
            method: Return calculation method ('log' or 'simple')
            
        Returns:
            DataFrame with returns
        """
        try:
            if method == "log":
                returns = np.log(data / data.shift(1))
            elif method == "simple":
                returns = (data - data.shift(1)) / data.shift(1)
            else:
                raise ValueError(f"Unsupported return method: {method}")
            
            return returns.dropna()
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            raise
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the data
        
        Args:
            data: OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        try:
            indicators = data.copy()
            
            # Moving averages
            indicators['SMA_20'] = data['Close'].rolling(window=20).mean()
            indicators['SMA_50'] = data['Close'].rolling(window=50).mean()
            indicators['EMA_12'] = data['Close'].ewm(span=12).mean()
            indicators['EMA_26'] = data['Close'].ewm(span=26).mean()
            
            # MACD
            indicators['MACD'] = indicators['EMA_12'] - indicators['EMA_26']
            indicators['MACD_Signal'] = indicators['MACD'].ewm(span=9).mean()
            indicators['MACD_Histogram'] = indicators['MACD'] - indicators['MACD_Signal']
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            indicators['BB_Middle'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            indicators['BB_Upper'] = indicators['BB_Middle'] + (bb_std * 2)
            indicators['BB_Lower'] = indicators['BB_Middle'] - (bb_std * 2)
            
            # Volume indicators
            indicators['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            indicators['Volume_Ratio'] = data['Volume'] / indicators['Volume_SMA']
            
            return indicators.dropna()
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            raise
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess financial data
        
        Args:
            data: Raw financial data
            
        Returns:
            Cleaned DataFrame
        """
        try:
            cleaned_data = data.copy()
            
            # Remove rows with all NaN values
            cleaned_data = cleaned_data.dropna(how='all')
            
            # Forward fill missing values for OHLC data
            ohlc_columns = [col for col in cleaned_data.columns if any(x in col[1] for x in ['Open', 'High', 'Low', 'Close'])]
            cleaned_data[ohlc_columns] = cleaned_data[ohlc_columns].fillna(method='ffill')
            
            # Fill remaining NaN values with 0 for volume
            volume_columns = [col for col in cleaned_data.columns if 'Volume' in col[1]]
            cleaned_data[volume_columns] = cleaned_data[volume_columns].fillna(0)
            
            # Remove any remaining NaN values
            cleaned_data = cleaned_data.dropna()
            
            return cleaned_data
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            raise
    
    def split_data(self, data: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into training, validation, and testing sets
        
        Args:
            data: Input data
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        try:
            total_rows = len(data)
            train_end = int(total_rows * train_ratio)
            val_end = int(total_rows * (train_ratio + val_ratio))
            
            train_data = data.iloc[:train_end]
            val_data = data.iloc[train_end:val_end]
            test_data = data.iloc[val_end:]
            
            return train_data, val_data, test_data
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise
    
    def get_market_data(self, ticker: str = "SPY", start_date: str = "2020-01-01", end_date: str = "2024-01-01") -> pd.DataFrame:
        """
        Get market benchmark data
        
        Args:
            ticker: Market benchmark ticker
            start_date: Start date
            end_date: End date
            
        Returns:
            Market benchmark data
        """
        try:
            return self.fetch_data([ticker], start_date, end_date)
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            raise
