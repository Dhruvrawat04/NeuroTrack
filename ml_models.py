import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

class MLModelHandler:
    def __init__(self):
        self.completion_model = None
        self.time_model = None
        self.encoders = {}
        self.completion_accuracy = 0
        self.time_mae = 0
        
    def prepare_features(self, data: pd.DataFrame):
        """Prepare features for ML models."""
        if data.empty:
            return pd.DataFrame(), {}
        
        df = data.copy()
        df['difficulty'] = pd.to_numeric(df['difficulty'], errors='coerce').fillna(3).astype(int)
        df['energy_level'] = pd.to_numeric(df['energy_level'], errors='coerce').fillna(5).astype(int)
        df['focus_level'] = pd.to_numeric(df['focus_level'], errors='coerce').fillna(5).astype(int)
        df['time_taken'] = pd.to_numeric(df['time_taken'], errors='coerce').fillna(30).astype(float)
        
        categorical_cols = ['category', 'priority', 'mood', 'intent']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('unknown').astype(str)
                le = LabelEncoder()
                le.fit(list(df[col].unique()) + ['unknown'])
                df[col + '_encoded'] = le.transform(df[col])
                self.encoders[col] = le
            else:
                df[col] = 'unknown'
                le = LabelEncoder()
                le.fit(['unknown'])
                df[col + '_encoded'] = le.transform(df[col])
                self.encoders[col] = le
        
        df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
        df['hour'] = df['start_time'].dt.hour.fillna(12).astype(int)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['day_of_week'] = df['date'].dt.dayofweek.fillna(1).astype(int)
        
        return df

    def train_completion_model(self, data: pd.DataFrame):
        """Train task completion prediction model."""
        try:
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
            
        except Exception as e:
            return f"Error training completion model: {str(e)}"

    def train_time_estimation_model(self, data: pd.DataFrame):
        """Train task duration estimation model."""
        try:
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
            
        except Exception as e:
            return f"Error training time estimator: {str(e)}"

    def predict_completion_probability(self, task_data: dict):
        """Predict task completion probability."""
        try:
            if self.completion_model is None:
                return 50.0  # Default if no model
            
            features = []
            features.append(max(1, min(5, task_data.get('difficulty', 3))))
            features.append(max(1, min(10, task_data.get('energy_level', 5))))
            features.append(max(1, min(10, task_data.get('focus_level', 5))))
            features.append(task_data.get('time_taken', 30))
            features.append(task_data.get('hour', 12))
            features.append(task_data.get('day_of_week', 1))
            
            for col in ['category', 'priority', 'mood', 'intent']:
                if col in self.encoders:
                    try:
                        value = task_data.get(col, 'unknown')
                        encoded = self.encoders[col].transform([str(value)])[0]
                        features.append(encoded)
                    except ValueError:
                        encoded = self.encoders[col].transform(['unknown'])[0]
                        features.append(encoded)
                    except Exception:
                        features.append(0)
                else:
                    features.append(0)
            
            X = np.array(features).reshape(1, -1)
            prob = self.completion_model.predict_proba(X)[0][1]
            return round(prob * 100, 1)
            
        except Exception as e:
            print(f"Error predicting task completion: {e}")
            return 50.0

    def predict_task_duration(self, task_data: dict):
        """Predict task duration."""
        try:
            if self.time_model is None:
                return 30
            
            features = []
            features.append(max(1, min(5, task_data.get('difficulty', 3))))
            
            for col in ['category', 'priority', 'intent']:
                if col in self.encoders:
                    try:
                        value = task_data.get(col, 'unknown')
                        encoded = self.encoders[col].transform([str(value)])[0]
                        features.append(encoded)
                    except ValueError:
                        encoded = self.encoders[col].transform(['unknown'])[0]
                        features.append(encoded)
                    except Exception:
                        features.append(0)
                else:
                    features.append(0)
            
            X = np.array(features).reshape(1, -1)
            duration = self.time_model.predict(X)[0]
            duration = max(5, min(480, round(duration / 5) * 5))
            return int(duration)
            
        except Exception as e:
            print(f"Error predicting task duration: {e}")
            return 30