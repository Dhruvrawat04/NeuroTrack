# ==================== IMPORTS ====================
# Data processing
import pandas as pd
import numpy as np
from datetime import datetime, date

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Utilities
import warnings

# Local modules
from encoding_utils import CategoricalEncoder
from data_preprocessing import (
    ensure_numeric_columns,
    extract_hour_from_datetime,
    extract_date_components
)

warnings.filterwarnings('ignore')

class MLModelHandler:
    def __init__(self):
        self.completion_model = None
        self.time_model = None
        self.encoders = CategoricalEncoder()
        self.completion_accuracy = 0
        self.time_mae = 0
        
    def prepare_features(self, data: pd.DataFrame):
        """Prepare features for ML models."""
        if data.empty:
            return pd.DataFrame(), {}
        
        df = data.copy()
        
        # Use shared utility for numeric column preparation
        numeric_mapping = {
            'difficulty': (int, 3),
            'energy_level': (int, 5),
            'focus_level': (int, 5),
            'time_taken': (float, 30)
        }
        df = ensure_numeric_columns(df, numeric_mapping)
        
        # Use shared encoder for categorical columns
        categorical_cols = ['category', 'priority', 'mood', 'intent']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('unknown').astype(str)
                self.encoders.fit_categorical_column(df, col)
                df = self.encoders.encode_column(df, col)
            else:
                df[col] = 'unknown'
                self.encoders.fit_categorical_column(df, col)
                df = self.encoders.encode_column(df, col)
        
        # Extract date/time features using shared utility
        df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
        df['hour'] = extract_hour_from_datetime(df, 'start_time').fillna(12).astype(int)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['day_of_week'] = df['date'].dt.dayofweek.fillna(1).astype(int)
        
        return df

    def train_completion_model(self, data: pd.DataFrame):
        """Train task completion prediction model."""
        try:
            if data is None or data.empty:
                return "No data provided for training"
            
            df = self.prepare_features(data)
            
            if df.empty or len(df) < 20:
                return "Not enough data for training (need at least 20 samples)"
            
            feature_cols = [
                'difficulty', 'energy_level', 'focus_level', 'time_taken',
                'hour', 'day_of_week', 'category_encoded', 
                'priority_encoded', 'mood_encoded', 'intent_encoded'
            ]
            
            X = df[feature_cols].fillna(0)
            y = df['completed'].fillna(False).astype(bool)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
            
            self.completion_model = RandomForestClassifier(
                n_estimators=100,  # Increased from 50
                max_depth=12,      # Increased from 10
                random_state=42,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced'  # Handle class imbalance
            )
            
            self.completion_model.fit(X_train, y_train)
            y_pred = self.completion_model.predict(X_test)
            self.completion_accuracy = accuracy_score(y_test, y_pred)
            
            return f"Completion model trained with accuracy: {self.completion_accuracy:.1%}"
            
        except ValueError as ve:
            return f"Invalid data format: {str(ve)}"
        except MemoryError:
            return "Out of memory: Dataset too large for available resources"
        except Exception as e:
            import traceback
            return f"Error training completion model: {str(e)}"

    def train_time_estimation_model(self, data: pd.DataFrame):
        """Train task duration estimation model."""
        try:
            if data is None or data.empty:
                return "No data provided for training"
            
            df = self.prepare_features(data)
            
            if df.empty or len(df) < 20:
                return "Not enough data for training (need at least 20 samples)"
            
            feature_cols = [
                'difficulty', 'category_encoded', 
                'priority_encoded', 'intent_encoded'
            ]
            
            X = df[feature_cols].fillna(0)
            y = df['time_taken'].fillna(30)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )
            
            self.time_model = RandomForestRegressor(
                n_estimators=100,  # Increased from 50
                max_depth=12,     # Increased from 10
                random_state=42,
                min_samples_split=5,
                min_samples_leaf=2
            )
            
            self.time_model.fit(X_train, y_train)
            y_pred = self.time_model.predict(X_test)
            self.time_mae = mean_absolute_error(y_test, y_pred)
            
            return f"Time estimation model trained with MAE: {self.time_mae:.1f} minutes"
            
        except ValueError as ve:
            return f"Invalid data format: {str(ve)}"
        except MemoryError:
            return "Out of memory: Dataset too large for available resources"
        except Exception as e:
            import traceback
            return f"Error training time estimator: {str(e)}"

    def predict_completion_probability(self, task_data: dict):
        """Predict task completion probability."""
        try:
            if task_data is None or not isinstance(task_data, dict):
                return 50.0
            
            if self.completion_model is None:
                return 50.0  # Default if no model
            
            features = []
            features.append(max(1, min(5, task_data.get('difficulty', 3))))
            features.append(max(1, min(10, task_data.get('energy_level', 5))))
            features.append(max(1, min(10, task_data.get('focus_level', 5))))
            features.append(task_data.get('time_taken', 30))
            features.append(task_data.get('hour', 12))
            features.append(task_data.get('day_of_week', 1))
            
            # Use centralized encoder for categorical features
            for col in ['category', 'priority', 'mood', 'intent']:
                if col in self.encoders.encoders:
                    try:
                        value = task_data.get(col, 'unknown')
                        encoded = self.encoders.encoders[col].transform([str(value)])[0]
                        features.append(encoded)
                    except (ValueError, KeyError):
                        features.append(self.encoders.encoders[col].transform(['unknown'])[0])
                    except Exception:
                        features.append(0)
                else:
                    features.append(0)
            
            X = np.array(features).reshape(1, -1)
            prob = self.completion_model.predict_proba(X)[0][1]
            return round(prob * 100, 1)
            
        except (ValueError, TypeError) as e:
            print(f"Invalid task data format: {e}")
            return 50.0
        except Exception as e:
            import traceback
            print(f"Error predicting task completion: {e}")
            return 50.0

    def predict_task_duration(self, task_data: dict):
        """Predict task duration."""
        try:
            if task_data is None or not isinstance(task_data, dict):
                return 30
            
            if self.time_model is None:
                return 30
            
            features = []
            features.append(max(1, min(5, task_data.get('difficulty', 3))))
            
            for col in ['category', 'priority', 'intent']:
                if col in self.encoders.encoders:
                    try:
                        value = task_data.get(col, 'unknown')
                        encoded = self.encoders.encoders[col].transform([str(value)])[0]
                        features.append(encoded)
                    except (ValueError, KeyError):
                        features.append(self.encoders.encoders[col].transform(['unknown'])[0])
                    except Exception:
                        features.append(0)
                else:
                    features.append(0)
            
            X = np.array(features).reshape(1, -1)
            duration = self.time_model.predict(X)[0]
            duration = max(5, min(480, round(duration / 5) * 5))
            return int(duration)
            
        except (ValueError, TypeError) as e:
            print(f"Invalid task data format: {e}")
            return 30
        except Exception as e:
            import traceback
            print(f"Error predicting task duration: {e}")
            return 30