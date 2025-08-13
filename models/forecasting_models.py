"""
Time Series Forecasting Models for Portfolio Management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings

# Optional dependencies: scikit-learn (metrics + scaler)
try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error  # type: ignore
    from sklearn.preprocessing import MinMaxScaler  # type: ignore
    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - provide lightweight fallbacks
    SKLEARN_AVAILABLE = False

    def mean_squared_error(y_true, y_pred):  # type: ignore
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        return float(np.mean((y_true_arr - y_pred_arr) ** 2))

    def mean_absolute_error(y_true, y_pred):  # type: ignore
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        return float(np.mean(np.abs(y_true_arr - y_pred_arr)))

    class MinMaxScaler:  # minimal replacement for sklearn's scaler
        def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
            self.feature_range = feature_range
            self.data_min_: Optional[np.ndarray] = None
            self.data_max_: Optional[np.ndarray] = None
            self.scale_: Optional[np.ndarray] = None
            self.min_: Optional[np.ndarray] = None

        def fit(self, X: np.ndarray):
            X_arr = np.asarray(X, dtype=float)
            self.data_min_ = np.min(X_arr, axis=0)
            self.data_max_ = np.max(X_arr, axis=0)
            data_range = self.data_max_ - self.data_min_
            # prevent divide-by-zero when data is constant
            data_range = np.where(data_range == 0.0, 1.0, data_range)
            fr_min, fr_max = self.feature_range
            self.scale_ = (fr_max - fr_min) / data_range
            self.min_ = fr_min - self.data_min_ * self.scale_
            return self

        def transform(self, X: np.ndarray) -> np.ndarray:
            X_arr = np.asarray(X, dtype=float)
            return X_arr * self.scale_ + self.min_  # type: ignore[operator]

        def fit_transform(self, X: np.ndarray) -> np.ndarray:
            return self.fit(X).transform(X)

        def inverse_transform(self, X: np.ndarray) -> np.ndarray:
            X_arr = np.asarray(X, dtype=float)
            return (X_arr - self.min_) / self.scale_  # type: ignore[operator]

# Time series models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Optional dependency: prophet
try:
    from prophet import Prophet  # type: ignore
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False
    warnings.warn(
        "Prophet is not available. ProphetForecaster will not work until you install 'prophet'."
    )

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

        # Mean Absolute Percentage Error (exclude zeros to avoid division by zero)
        actual_array = np.asarray(actual).astype(float).flatten()
        predicted_array = np.asarray(predicted).astype(float).flatten()
        non_zero_mask = actual_array != 0
        if non_zero_mask.any():
            mape = np.mean(np.abs((actual_array[non_zero_mask] - predicted_array[non_zero_mask]) / actual_array[non_zero_mask])) * 100.0
        else:
            mape = float('nan')

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
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
            if not PROPHET_AVAILABLE:
                raise ImportError("Prophet is required for ProphetForecaster. Please install 'prophet'.")
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

    def __init__(
        self,
        units: int = 50,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        lookback: int = 60,
        learning_rate: float = 0.001,
        random_seed: int = 42,
    ):
        super().__init__("LSTM")
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")

        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lookback = lookback
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.scaler = MinMaxScaler()
        self.last_data: Optional[pd.Series] = None

    def _prepare_data(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM (create sequences)"""
        # Scale the data
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))

        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i - self.lookback : i, 0])
            y.append(scaled_data[i, 0])

        return np.array(X), np.array(y)

    def fit(self, data: pd.Series) -> 'LSTMForecaster':
        """Fit LSTM model"""
        try:
            # Store original data for prediction context
            self.last_data = data.copy()

            # Set seeds for reproducibility
            np.random.seed(self.random_seed)
            try:
                tf.random.set_seed(self.random_seed)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover
                pass

            # Prepare data
            X, y = self._prepare_data(data)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # Build LSTM model
            self.model = Sequential([
                LSTM(units=self.units, return_sequences=True, input_shape=(X.shape[1], 1)),
                Dropout(self.dropout),
                LSTM(units=self.units, return_sequences=False),
                Dropout(self.dropout),
                Dense(units=1),
            ])

            self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')

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
        if not self.is_fitted or self.last_data is None:
            raise ValueError("Model must be fitted before making predictions")

        try:
            # Get the last sequence from training data
            last_sequence_scaled = self.scaler.transform(
                self.last_data.values[-self.lookback :].reshape(-1, 1)
            )
            current_sequence = last_sequence_scaled.reshape((1, self.lookback, 1))

            predictions_scaled = []
            for _ in range(steps):
                pred_scaled = self.model.predict(current_sequence, verbose=0)
                predictions_scaled.append(pred_scaled[0, 0])

                # Update sequence for next prediction (append predicted value)
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = pred_scaled[0, 0]

            # Inverse transform predictions
            predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions_scaled).flatten()

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
