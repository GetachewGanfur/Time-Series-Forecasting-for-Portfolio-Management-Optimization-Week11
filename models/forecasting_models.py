"""
Time Series Forecasting Models for Portfolio Management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import warnings

# Time series models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# Deep learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not available. LSTM models will not work.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseForecaster:
    """Base class for all forecasting models"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.forecast_history = []
        
    def fit(self, data: pd.Series) -> 'BaseForecaster':
        """Fit the model to the data"""
        raise NotImplementedError
        
    def predict(self, steps: int) -> np.ndarray:
        """Make predictions"""
        raise NotImplementedError
        
    def evaluate(self, actual: pd.Series, predicted: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }

class ARIMAForecaster(BaseForecaster):
    """ARIMA model for time series forecasting"""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        super().__init__(f"ARIMA{order}")
        self.order = order
        
    def fit(self, data: pd.Series) -> 'ARIMAForecaster':
        """Fit ARIMA model"""
        try:
            self.model = ARIMA(data, order=self.order)
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            logger.info(f"ARIMA model fitted successfully with order {self.order}")
            return self
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            raise
            
    def predict(self, steps: int) -> np.ndarray:
        """Make ARIMA predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            self.forecast_history.append(forecast)
            return forecast
        except Exception as e:
            logger.error(f"Error making ARIMA predictions: {e}")
            raise

class SARIMAForecaster(BaseForecaster):
    """SARIMA model for seasonal time series forecasting"""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)):
        super().__init__(f"SARIMA{order}{seasonal_order}")
        self.order = order
        self.seasonal_order = seasonal_order
        
    def fit(self, data: pd.Series) -> 'SARIMAForecaster':
        """Fit SARIMA model"""
        try:
            self.model = SARIMAX(data, order=self.order, seasonal_order=self.seasonal_order)
            self.fitted_model = self.model.fit(disp=False)
            self.is_fitted = True
            logger.info(f"SARIMA model fitted successfully with order {self.order} and seasonal order {self.seasonal_order}")
            return self
        except Exception as e:
            logger.error(f"Error fitting SARIMA model: {e}")
            raise
            
    def predict(self, steps: int) -> np.ndarray:
        """Make SARIMA predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            self.forecast_history.append(forecast)
            return forecast
        except Exception as e:
            logger.error(f"Error making SARIMA predictions: {e}")
            raise

class ProphetForecaster(BaseForecaster):
    """Facebook Prophet model for time series forecasting"""
    
    def __init__(self, **kwargs):
        super().__init__("Prophet")
        self.kwargs = kwargs
        
    def fit(self, data: pd.Series) -> 'ProphetForecaster':
        """Fit Prophet model"""
        try:
            # Prepare data for Prophet (requires 'ds' and 'y' columns)
            df = pd.DataFrame({
                'ds': data.index,
                'y': data.values
            })
            
            self.model = Prophet(**self.kwargs)
            self.model.fit(df)
            self.is_fitted = True
            logger.info("Prophet model fitted successfully")
            return self
        except Exception as e:
            logger.error(f"Error fitting Prophet model: {e}")
            raise
            
    def predict(self, steps: int) -> np.ndarray:
        """Make Prophet predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Create future dates
            last_date = self.model.history['ds'].iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=steps + 1, freq='D')[1:]
            
            future_df = pd.DataFrame({'ds': future_dates})
            forecast = self.model.predict(future_df)
            
            self.forecast_history.append(forecast['yhat'].values)
            return forecast['yhat'].values
        except Exception as e:
            logger.error(f"Error making Prophet predictions: {e}")
            raise

class LSTMForecaster(BaseForecaster):
    """LSTM neural network for time series forecasting"""
    
    def __init__(self, units: int = 50, dropout: float = 0.2, epochs: int = 100, batch_size: int = 32):
        super().__init__("LSTM")
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
            
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        
    def _prepare_data(self, data: pd.Series, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM (create sequences)"""
        # Scale the data
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
        
    def fit(self, data: pd.Series) -> 'LSTMForecaster':
        """Fit LSTM model"""
        try:
            # Prepare data
            X, y = self._prepare_data(data)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Build LSTM model
            self.model = Sequential([
                LSTM(units=self.units, return_sequences=True, input_shape=(X.shape[1], 1)),
                Dropout(self.dropout),
                LSTM(units=self.units, return_sequences=False),
                Dropout(self.dropout),
                Dense(units=1)
            ])
            
            self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train the model
            self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            self.is_fitted = True
            logger.info("LSTM model fitted successfully")
            return self
        except Exception as e:
            logger.error(f"Error fitting LSTM model: {e}")
            raise
            
    def predict(self, steps: int) -> np.ndarray:
        """Make LSTM predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Get the last sequence from training data
            last_sequence = self.scaler.transform(self.last_data.values[-60:].reshape(-1, 1))
            last_sequence = last_sequence.reshape((1, 60, 1))
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(steps):
                # Make prediction
                pred = self.model.predict(current_sequence)
                predictions.append(pred[0, 0])
                
                # Update sequence for next prediction
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = pred[0, 0]
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions).flatten()
            
            self.forecast_history.append(predictions)
            return predictions
        except Exception as e:
            logger.error(f"Error making LSTM predictions: {e}")
            raise

class EnsembleForecaster(BaseForecaster):
    """Ensemble of multiple forecasting models"""
    
    def __init__(self, models: List[BaseForecaster], weights: Optional[List[float]] = None):
        super().__init__("Ensemble")
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
            
    def fit(self, data: pd.Series) -> 'EnsembleForecaster':
        """Fit all ensemble models"""
        try:
            for model in self.models:
                model.fit(data)
            self.is_fitted = True
            logger.info(f"Ensemble model fitted successfully with {len(self.models)} models")
            return self
        except Exception as e:
            logger.error(f"Error fitting ensemble model: {e}")
            raise
            
    def predict(self, steps: int) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            predictions = []
            for model in self.models:
                pred = model.predict(steps)
                predictions.append(pred)
            
            # Weighted average of predictions
            ensemble_pred = np.zeros(steps)
            for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
                ensemble_pred += weight * pred
            
            self.forecast_history.append(ensemble_pred)
            return ensemble_pred
        except Exception as e:
            logger.error(f"Error making ensemble predictions: {e}")
            raise

class ForecastingEngine:
    """Main engine for managing multiple forecasting models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def add_model(self, name: str, model: BaseForecaster):
        """Add a forecasting model to the engine"""
        self.models[name] = model
        
    def fit_all_models(self, data: pd.Series):
        """Fit all models in the engine"""
        for name, model in self.models.items():
            try:
                logger.info(f"Fitting {name} model...")
                model.fit(data)
                logger.info(f"{name} model fitted successfully")
            except Exception as e:
                logger.error(f"Error fitting {name} model: {e}")
                
    def predict_all_models(self, steps: int) -> Dict[str, np.ndarray]:
        """Get predictions from all models"""
        predictions = {}
        for name, model in self.models.items():
            if model.is_fitted:
                try:
                    predictions[name] = model.predict(steps)
                except Exception as e:
                    logger.error(f"Error getting predictions from {name}: {e}")
                    
        return predictions
        
    def evaluate_all_models(self, actual: pd.Series) -> Dict[str, Dict[str, float]]:
        """Evaluate all models"""
        evaluations = {}
        for name, model in self.models.items():
            if model.is_fitted and model.forecast_history:
                try:
                    predictions = model.forecast_history[-1]
                    evaluations[name] = model.evaluate(actual, predictions)
                except Exception as e:
                    logger.error(f"Error evaluating {name}: {e}")
                    
        return evaluations
