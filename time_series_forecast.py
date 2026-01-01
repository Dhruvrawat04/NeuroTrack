# ==================== IMPORTS ====================
# Data processing
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Utilities
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesForecaster:
    """Time series forecasting for productivity metrics with dynamic variation."""
    
    def __init__(self):
        self.forecast_days = 7  # Default forecast horizon
        
    def prepare_time_series(self, data: pd.DataFrame, metric: str = 'time_taken', freq: str = 'D'):
        """
        Prepare time series data for forecasting.
        
        Args:
            data: DataFrame with task data
            metric: Column to forecast ('time_taken', 'completed', 'task_count')
            freq: Frequency ('D' for daily, 'W' for weekly)
        """
        try:
            if data.empty:
                print(f"DEBUG: Empty data provided for metric: {metric}")
                return pd.Series(dtype=float)
            
            df = data.copy()
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            
            if df.empty:
                print(f"DEBUG: No valid dates after cleaning for metric: {metric}")
                return pd.Series(dtype=float)
            
            # Aggregate by date
            if metric == 'task_count':
                ts = df.groupby('date').size()
            elif metric == 'completion_rate':
                ts = df.groupby('date')['completed'].mean() * 100
            else:
                ts = df.groupby('date')[metric].sum()
            
            # Resample to ensure consistent frequency
            if metric != 'completion_rate':
                ts = ts.resample(freq).sum()
            else:
                ts = ts.resample(freq).mean()
            
            ts = ts.fillna(0)
            
            print(f"DEBUG: Prepared time series for {metric}: {len(ts)} points, "
                  f"values: {ts.tolist()}")
            
            return ts
            
        except Exception as e:
            print(f"Error preparing time series: {e}")
            return pd.Series(dtype=float)
    
    def moving_average_forecast(self, ts: pd.Series, window: int = 7, horizon: int = 7):
        """
        Enhanced Moving Average forecast with dynamic variation.
        
        Args:
            ts: Time series data
            window: Moving average window size
            horizon: Number of periods to forecast
        """
        try:
            print(f"DEBUG: MA Forecast - Input length: {len(ts)}, window: {window}")
            
            if len(ts) < 2:
                print("DEBUG: Insufficient data for MA forecast")
                return self._create_fallback_forecast(ts, horizon), ts
            
            if len(ts) < window:
                window = max(3, len(ts) // 2)
                print(f"DEBUG: Adjusted window to {window}")
            
            # Calculate moving average
            ma = ts.rolling(window=window, min_periods=1).mean()
            
            # Calculate multiple trend components
            trends = []
            
            # Short-term trend (last 3-7 days)
            if len(ts) >= 3:
                short_window = min(7, len(ts))
                short_trend = (ts.iloc[-1] - ts.iloc[-short_window]) / short_window
                trends.append(short_trend)
            
            # Medium-term trend (overall)
            if len(ts) >= 2:
                overall_trend = (ts.iloc[-1] - ts.iloc[0]) / len(ts)
                trends.append(overall_trend * 0.5)
            
            # Weighted average of available trends
            if trends:
                trend = np.mean(trends)
                print(f"DEBUG: Calculated trend: {trend:.4f}")
            else:
                trend = 0
                print(f"DEBUG: No trend calculated, using 0")
            
            # Calculate volatility
            volatility = ts.std() if ts.std() > 0 else ts.mean() * 0.15
            print(f"DEBUG: Volatility: {volatility:.4f}")
            
            # Generate forecast dates
            last_date = ts.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), 
                periods=horizon, 
                freq='D'
            )
            
            # Generate forecast with dynamic variation
            forecast_values = []
            for i in range(1, horizon + 1):
                # Base value with trend
                base_value = ma.iloc[-1]
                trend_effect = trend * i * 0.8
                
                # Add seasonality (weekly pattern)
                day_of_week = (last_date.weekday() + i) % 7
                seasonality = self._get_seasonality_factor(day_of_week)
                
                # Add controlled random noise
                if volatility > 0:
                    noise = np.random.normal(0, volatility * 0.2)
                else:
                    noise = np.random.uniform(-2, 2)
                
                # Calculate final value
                value = base_value * seasonality + trend_effect + noise
                
                # Apply bounds
                if 'score' in str(ts.name or '').lower() or 'rate' in str(ts.name or '').lower():
                    value = max(0, min(100, value))
                else:
                    value = max(0, value)
                
                forecast_values.append(value)
            
            forecast = pd.Series(forecast_values, index=forecast_dates)
            
            print(f"DEBUG: MA Forecast values: {forecast_values}")
            print(f"DEBUG: MA Forecast trend: {trend:.4f}")
            
            return forecast, ma
            
        except Exception as e:
            print(f"Error in moving average forecast: {e}")
            return self._create_fallback_forecast(ts, horizon), ts
    
    def exponential_smoothing_forecast(self, ts: pd.Series, alpha: float = 0.3, horizon: int = 7):
        """
        Enhanced Exponential Smoothing forecast with dynamic variation.
        
        Args:
            ts: Time series data
            alpha: Smoothing parameter (0-1)
            horizon: Number of periods to forecast
        """
        try:
            print(f"DEBUG: ES Forecast - Input length: {len(ts)}, alpha: {alpha}")
            
            if ts.empty or len(ts) < 2:
                print("DEBUG: Insufficient data for ES forecast")
                return self._create_fallback_forecast(ts, horizon), ts
            
            # Calculate exponential smoothing
            smoothed = [ts.iloc[0]]
            for i in range(1, len(ts)):
                smoothed.append(alpha * ts.iloc[i] + (1 - alpha) * smoothed[-1])
            
            smoothed_series = pd.Series(smoothed, index=ts.index)
            
            # Calculate dynamic trend with multiple timeframes
            trend_components = []
            
            if len(smoothed) >= 3:
                # Recent trend (last 3 days)
                recent_trend = (smoothed[-1] - smoothed[-min(3, len(smoothed))]) / min(3, len(smoothed))
                trend_components.append(recent_trend)
            
            if len(smoothed) >= 7:
                # Weekly trend
                weekly_trend = (smoothed[-1] - smoothed[-7]) / 7
                trend_components.append(weekly_trend * 0.7)
            
            # Overall trend
            if len(smoothed) >= 2:
                overall_trend = (smoothed[-1] - smoothed[0]) / len(smoothed)
                trend_components.append(overall_trend * 0.5)
            
            # Calculate final trend
            if trend_components:
                trend = np.mean(trend_components)
                print(f"DEBUG: ES Calculated trend: {trend:.4f}")
            else:
                trend = np.random.uniform(-1, 1)  # Small random trend
                print(f"DEBUG: Using random trend: {trend:.4f}")
            
            # Generate forecast dates
            last_date = ts.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), 
                periods=horizon, 
                freq='D'
            )
            
            # Generate varied forecast
            forecast_values = []
            for i in range(1, horizon + 1):
                base_value = smoothed[-1]
                trend_effect = trend * i * 0.6
                
                # Seasonality adjustment
                day_of_week = (last_date.weekday() + i) % 7
                seasonality = self._get_seasonality_factor(day_of_week)
                
                # Natural variation
                variation = np.random.normal(0, ts.std() * 0.15) if ts.std() > 0 else np.random.uniform(-3, 3)
                
                value = (base_value + trend_effect) * seasonality + variation
                
                # Apply bounds
                if 'score' in str(ts.name or '').lower() or 'rate' in str(ts.name or '').lower():
                    value = max(0, min(100, value))
                else:
                    value = max(0, value)
                
                forecast_values.append(value)
            
            forecast = pd.Series(forecast_values, index=forecast_dates)
            
            print(f"DEBUG: ES Forecast values: {forecast_values}")
            print(f"DEBUG: ES Forecast trend: {trend:.4f}")
            
            return forecast, smoothed_series
            
        except Exception as e:
            print(f"Error in exponential smoothing: {e}")
            return self._create_fallback_forecast(ts, horizon), ts
    
    def _get_seasonality_factor(self, day_of_week: int) -> float:
        """Get seasonality factor based on day of week."""
        # Weekly productivity patterns
        weekday_pattern = {
            0: 0.95,  # Monday - slow start
            1: 1.05,  # Tuesday - building momentum
            2: 1.10,  # Wednesday - peak productivity
            3: 1.05,  # Thursday - still strong
            4: 0.90,  # Friday - winding down
            5: 0.60,  # Saturday - low productivity
            6: 0.50   # Sunday - very low productivity
        }
        return weekday_pattern.get(day_of_week, 1.0)
    
    def _create_fallback_forecast(self, ts: pd.Series, horizon: int) -> pd.Series:
        """Create a fallback forecast when insufficient data."""
        if ts.empty:
            # No data - use reasonable defaults
            last_date = datetime.now().date()
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=horizon,
                freq='D'
            )
            # Default to moderate productivity
            default_values = np.random.uniform(60, 80, horizon)
            return pd.Series(default_values, index=forecast_dates)
        
        last_date = ts.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        # Use mean with some variation
        mean_val = ts.mean()
        if pd.isna(mean_val):
            mean_val = 50
        
        # Create varying forecast
        base_values = np.linspace(mean_val, mean_val * 1.1, horizon)
        
        # Add seasonality
        seasonality_factors = []
        for i in range(horizon):
            day = (last_date.weekday() + i) % 7
            seasonality_factors.append(self._get_seasonality_factor(day))
        
        forecast_values = base_values * seasonality_factors
        
        # Add small random variation
        random_variation = np.random.normal(0, mean_val * 0.1, horizon)
        forecast_values = forecast_values + random_variation
        
        # Apply bounds
        forecast_values = np.clip(forecast_values, 0, 100)
        
        return pd.Series(forecast_values, index=forecast_dates)
    
    def forecast_productivity_score(self, data: pd.DataFrame, horizon: int = 7):
        """Forecast daily productivity score with enhanced variation."""
        try:
            print("\n" + "="*50)
            print("FORECASTING PRODUCTIVITY SCORE")
            print("="*50)
            
            if data.empty:
                print("DEBUG: Empty data provided")
                return None, None
            
            df = data.copy()
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            
            if df.empty:
                print("DEBUG: No valid dates after cleaning")
                return None, None
            
            # Calculate daily productivity score
            daily_scores = df.groupby('date').apply(
                lambda x: (x['completed'].sum() / len(x) * 100) if len(x) > 0 else 0
            )
            daily_scores.name = 'productivity_score'
            
            print(f"DEBUG: Historical productivity scores: {daily_scores.tolist()}")
            print(f"DEBUG: Score range: {daily_scores.min():.1f} - {daily_scores.max():.1f}")
            print(f"DEBUG: Score mean: {daily_scores.mean():.1f}, std: {daily_scores.std():.1f}")
            
            if len(daily_scores) < 2:
                print(f"DEBUG: Insufficient data ({len(daily_scores)} days)")
                forecast = self._create_fallback_forecast(daily_scores, horizon)
                return daily_scores, forecast
            
            # Forecast using enhanced exponential smoothing
            forecast, smoothed = self.exponential_smoothing_forecast(
                daily_scores, alpha=0.4, horizon=horizon
            )
            
            if forecast is not None:
                print(f"DEBUG: Forecast generated: {forecast.tolist()}")
                print(f"DEBUG: Forecast range: {forecast.min():.1f} - {forecast.max():.1f}")
                print(f"DEBUG: Forecast trend: {(forecast.iloc[-1] - forecast.iloc[0]) / horizon:.2f} per day")
            
            return daily_scores, forecast
            
        except Exception as e:
            print(f"Error forecasting productivity score: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def forecast_workload(self, data: pd.DataFrame, horizon: int = 7):
        """Forecast daily workload (time spent)."""
        try:
            print("\n" + "="*50)
            print("FORECASTING WORKLOAD")
            print("="*50)
            
            ts = self.prepare_time_series(data, metric='time_taken', freq='D')
            ts.name = 'workload_minutes'
            
            if ts.empty or len(ts) < 2:
                print(f"DEBUG: Insufficient workload data ({len(ts)} days)")
                forecast = self._create_fallback_forecast(ts, horizon)
                forecast = forecast * 60  # Convert to minutes scale
                return ts, forecast
            
            # Use enhanced moving average for workload
            forecast, ma = self.moving_average_forecast(ts, window=7, horizon=horizon)
            
            if forecast is not None:
                print(f"DEBUG: Workload forecast: {[int(x) for x in forecast.tolist()]} minutes")
                print(f"DEBUG: Daily avg: {forecast.mean():.1f} minutes")
            
            return ts, forecast
            
        except Exception as e:
            print(f"Error forecasting workload: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def forecast_task_count(self, data: pd.DataFrame, horizon: int = 7):
        """Forecast daily task count."""
        try:
            print("\n" + "="*50)
            print("FORECASTING TASK COUNT")
            print("="*50)
            
            ts = self.prepare_time_series(data, metric='task_count', freq='D')
            ts.name = 'task_count'
            
            if ts.empty or len(ts) < 2:
                print(f"DEBUG: Insufficient task count data ({len(ts)} days)")
                forecast = self._create_fallback_forecast(ts, horizon)
                # Task counts should be integers
                forecast = forecast.round().astype(int)
                return ts, forecast
            
            forecast, smoothed = self.exponential_smoothing_forecast(ts, alpha=0.3, horizon=horizon)
            
            if forecast is not None:
                # Task counts should be integers
                forecast = forecast.round().astype(int)
                forecast = forecast.clip(lower=0)
                print(f"DEBUG: Task count forecast: {forecast.tolist()} tasks")
            
            return ts, forecast
            
        except Exception as e:
            print(f"Error forecasting task count: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def forecast_completion_rate(self, data: pd.DataFrame, horizon: int = 7):
        """Forecast daily completion rate."""
        try:
            print("\n" + "="*50)
            print("FORECASTING COMPLETION RATE")
            print("="*50)
            
            ts = self.prepare_time_series(data, metric='completion_rate', freq='D')
            ts.name = 'completion_rate'
            
            if ts.empty or len(ts) < 2:
                print(f"DEBUG: Insufficient completion rate data ({len(ts)} days)")
                forecast = self._create_fallback_forecast(ts, horizon)
                forecast = forecast.clip(upper=100)
                return ts, forecast
            
            forecast, smoothed = self.exponential_smoothing_forecast(ts, alpha=0.35, horizon=horizon)
            
            if forecast is not None:
                # Cap completion rate at 100%
                forecast = forecast.clip(upper=100)
                print(f"DEBUG: Completion rate forecast: {[f'{x:.1f}%' for x in forecast.tolist()]}")
            
            return ts, forecast
            
        except Exception as e:
            print(f"Error forecasting completion rate: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def get_forecast_summary(self, data: pd.DataFrame, horizon: int = 7):
        """Generate comprehensive forecast summary with insights."""
        print("\n" + "="*50)
        print("GENERATING FORECAST SUMMARY")
        print("="*50)
        
        summary = {
            'horizon': horizon,
            'forecast_start': (date.today() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'forecast_end': (date.today() + timedelta(days=horizon)).strftime('%Y-%m-%d'),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Productivity score forecast
        hist_score, forecast_score = self.forecast_productivity_score(data, horizon)
        if hist_score is not None and forecast_score is not None:
            avg_forecast = forecast_score.mean()
            score_trend = forecast_score.iloc[-1] - forecast_score.iloc[0]
            trend_direction = '📈 increasing' if score_trend > 1 else '📉 decreasing' if score_trend < -1 else '➡️ stable'
            
            summary['productivity'] = {
                'avg_forecast': round(avg_forecast, 1),
                'trend': trend_direction,
                'trend_magnitude': round(abs(score_trend), 1),
                'best_day': forecast_score.idxmax().strftime('%A'),
                'best_day_score': round(forecast_score.max(), 1),
                'worst_day': forecast_score.idxmin().strftime('%A'),
                'worst_day_score': round(forecast_score.min(), 1)
            }
        
        # Workload forecast
        hist_workload, forecast_workload = self.forecast_workload(data, horizon)
        if hist_workload is not None and forecast_workload is not None:
            total_workload = forecast_workload.sum()
            avg_daily = forecast_workload.mean()
            workload_trend = forecast_workload.iloc[-1] - forecast_workload.iloc[0]
            
            summary['workload'] = {
                'total_hours_forecast': round(total_workload / 60, 1),
                'avg_daily_hours': round(avg_daily / 60, 1),
                'trend': '📈 increasing' if workload_trend > 10 else '📉 decreasing' if workload_trend < -10 else '➡️ stable',
                'busiest_day': forecast_workload.idxmax().strftime('%A'),
                'busiest_hours': round(forecast_workload.max() / 60, 1)
            }
        
        # Task count forecast
        hist_tasks, forecast_tasks = self.forecast_task_count(data, horizon)
        if hist_tasks is not None and forecast_tasks is not None:
            summary['tasks'] = {
                'total_tasks_forecast': int(forecast_tasks.sum()),
                'avg_daily_tasks': round(forecast_tasks.mean(), 1),
                'peak_day': forecast_tasks.idxmax().strftime('%A'),
                'peak_tasks': int(forecast_tasks.max())
            }
        
        # Completion rate forecast
        hist_completion, forecast_completion = self.forecast_completion_rate(data, horizon)
        if hist_completion is not None and forecast_completion is not None:
            avg_completion = forecast_completion.mean()
            completion_trend = forecast_completion.iloc[-1] - forecast_completion.iloc[0]
            
            summary['completion'] = {
                'avg_rate': round(avg_completion, 1),
                'trend': '📈 improving' if completion_trend > 2 else '📉 declining' if completion_trend < -2 else '➡️ stable',
                'best_day': forecast_completion.idxmax().strftime('%A'),
                'best_rate': round(forecast_completion.max(), 1)
            }
        
        # Generate insights
        summary['insights'] = self._generate_insights(summary)
        
        print(f"\nSUMMARY: {summary}")
        return summary
    
    def _generate_insights(self, summary: dict) -> list:
        """Generate actionable insights from forecast data."""
        insights = []
        
        # Check productivity trends
        if 'productivity' in summary:
            prod = summary['productivity']
            if 'trend' in prod:
                if 'decreasing' in prod['trend'] and prod['trend_magnitude'] > 5:
                    insights.append(f"⚠️ Productivity may decline by {prod['trend_magnitude']}%. Consider adjusting workload.")
                elif 'increasing' in prod['trend'] and prod['trend_magnitude'] > 5:
                    insights.append(f"✅ Productivity trending up by {prod['trend_magnitude']}%. Keep up the momentum!")
        
        # Check workload balance
        if 'workload' in summary:
            workload = summary['workload']
            if 'avg_daily_hours' in workload:
                if workload['avg_daily_hours'] > 9:
                    insights.append(f"⚡ High workload forecast ({workload['avg_daily_hours']}h/day). Consider planning breaks.")
                elif workload['avg_daily_hours'] < 4:
                    insights.append(f"💡 Light schedule forecast ({workload['avg_daily_hours']}h/day). Opportunity to tackle extra goals.")
        
        # Check completion rates
        if 'completion' in summary:
            completion = summary['completion']
            if 'avg_rate' in completion:
                if completion['avg_rate'] < 60:
                    insights.append(f"🎯 Low completion rate forecast ({completion['avg_rate']}%). Try breaking tasks into smaller chunks.")
                elif completion['avg_rate'] > 85:
                    insights.append(f"🌟 Excellent completion forecast ({completion['avg_rate']}%)! You're on track.")
        
        # Add day-specific insights
        if 'productivity' in summary and 'workload' in summary:
            prod = summary['productivity']
            work = summary['workload']
            
            if 'best_day' in prod and 'busiest_day' in work:
                if prod['best_day'] == work['busiest_day']:
                    insights.append(f"📅 {prod['best_day']} looks like your most productive AND busiest day. Schedule important tasks then!")
        
        # Add generic insights if none generated
        if not insights:
            insights.append("📊 Forecast suggests stable productivity patterns. Maintain your current routine.")
        
        return insights
    
    def create_forecast_chart(self, historical: pd.Series, forecast: pd.Series, 
                             title: str, yaxis_title: str, color: str = '#6366F1'):
        """Create an enhanced forecast visualization with dynamic variation."""
        try:
            if historical is None or forecast is None:
                print(f"DEBUG: Cannot create chart - historical: {historical is not None}, forecast: {forecast is not None}")
                return None
            
            print(f"DEBUG: Creating chart - Historical: {len(historical)} points, Forecast: {len(forecast)} points")
            print(f"DEBUG: Forecast values: {forecast.tolist()}")
            
            # Check if forecast has variation
            forecast_std = forecast.std()
            forecast_range = forecast.max() - forecast.min()
            print(f"DEBUG: Forecast std: {forecast_std:.2f}, range: {forecast_range:.2f}")
            
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical.index,
                y=historical.values,
                mode='lines+markers',
                name='Historical',
                line=dict(color=color, width=3),
                marker=dict(size=8, color=color),
                hovertemplate='<b>%{x|%b %d}</b><br>%{y:.1f}<extra></extra>'
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast.values,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#F59E0B', width=3, dash='dash'),
                marker=dict(size=10, symbol='diamond', color='#F59E0B'),
                hovertemplate='<b>%{x|%b %d}</b><br>Forecast: %{y:.1f}<extra></extra>'
            ))
            
            # Dynamic confidence interval based on forecast variation
            if forecast_std > 0:
                confidence_factor = 0.25 if forecast_std > 10 else 0.15
            else:
                confidence_factor = 0.1
            
            upper = forecast * (1 + confidence_factor)
            lower = forecast * (1 - confidence_factor)
            
            # Ensure lower bound doesn't go below 0 for positive metrics
            if 'score' in title.lower() or 'rate' in title.lower():
                lower = lower.clip(lower=0)
                upper = upper.clip(upper=100)
            
            fig.add_trace(go.Scatter(
                x=forecast.index.tolist() + forecast.index.tolist()[::-1],
                y=upper.tolist() + lower.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(245, 158, 11, 0.2)',
                line=dict(color='rgba(245, 158, 11, 0)'),
                name='Confidence Interval',
                showlegend=True,
                hovertemplate='<b>%{x|%b %d}</b><br>Range: %{y:.1f}<extra></extra>'
            ))
            
            # Add trend line for forecast
            if len(forecast) > 1:
                x_numeric = np.arange(len(forecast))
                z = np.polyfit(x_numeric, forecast.values, 1)
                trend_line = np.poly1d(z)(x_numeric)
                
                fig.add_trace(go.Scatter(
                    x=forecast.index,
                    y=trend_line,
                    mode='lines',
                    name='Trend',
                    line=dict(color='#10B981', width=2, dash='dot'),
                    hovertemplate='<b>Trend</b><br>%{y:.1f}<extra></extra>'
                ))
            
            fig.update_layout(
                title={'text': title, 'font': {'color': '#A5B4FC', 'size': 18}},
                xaxis_title='Date',
                yaxis_title=yaxis_title,
                height=400,
                hovermode='x unified',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#F8FAFC', 'size': 12},
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor='rgba(0,0,0,0.5)',
                    bordercolor='rgba(255,255,255,0.2)',
                    borderwidth=1
                ),
                margin=dict(l=50, r=30, t=60, b=50)
            )
            
            # Customize x-axis date formatting
            fig.update_xaxes(
                tickformat='%b %d',
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True
            )
            
            fig.update_yaxes(
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating forecast chart: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_comprehensive_forecast_dashboard(self, data: pd.DataFrame, horizon: int = 7):
        """Create a comprehensive dashboard with all forecasts."""
        try:
            if data.empty:
                return None
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    '📈 Productivity Score Forecast',
                    '⏱️ Workload Forecast (Hours)',
                    '📊 Task Count Forecast',
                    '✅ Completion Rate Forecast'
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            # 1. Productivity Score
            hist_score, forecast_score = self.forecast_productivity_score(data, horizon)
            if hist_score is not None and forecast_score is not None:
                fig.add_trace(
                    go.Scatter(x=hist_score.index, y=hist_score.values,
                              mode='lines+markers', name='Historical Score',
                              line=dict(color='#6366F1', width=2)),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=forecast_score.index, y=forecast_score.values,
                              mode='lines+markers', name='Forecast Score',
                              line=dict(color='#F59E0B', width=2, dash='dash')),
                    row=1, col=1
                )
            
            # 2. Workload
            hist_workload, forecast_workload = self.forecast_workload(data, horizon)
            if hist_workload is not None and forecast_workload is not None:
                # Convert minutes to hours for display
                hist_hours = hist_workload / 60
                forecast_hours = forecast_workload / 60
                
                fig.add_trace(
                    go.Scatter(x=hist_hours.index, y=hist_hours.values,
                              mode='lines+markers', name='Historical Workload',
                              line=dict(color='#10B981', width=2)),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Scatter(x=forecast_hours.index, y=forecast_hours.values,
                              mode='lines+markers', name='Forecast Workload',
                              line=dict(color='#F59E0B', width=2, dash='dash')),
                    row=1, col=2
                )
            
            # 3. Task Count
            hist_tasks, forecast_tasks = self.forecast_task_count(data, horizon)
            if hist_tasks is not None and forecast_tasks is not None:
                fig.add_trace(
                    go.Bar(x=hist_tasks.index, y=hist_tasks.values,
                          name='Historical Tasks', marker_color='#8B5CF6'),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Bar(x=forecast_tasks.index, y=forecast_tasks.values,
                          name='Forecast Tasks', marker_color='#F59E0B'),
                    row=2, col=1
                )
            
            # 4. Completion Rate
            hist_completion, forecast_completion = self.forecast_completion_rate(data, horizon)
            if hist_completion is not None and forecast_completion is not None:
                fig.add_trace(
                    go.Scatter(x=hist_completion.index, y=hist_completion.values,
                              mode='lines+markers', name='Historical Completion',
                              line=dict(color='#EC4899', width=2)),
                    row=2, col=2
                )
                fig.add_trace(
                    go.Scatter(x=forecast_completion.index, y=forecast_completion.values,
                              mode='lines+markers', name='Forecast Completion',
                              line=dict(color='#F59E0B', width=2, dash='dash')),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#F8FAFC'},
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update axes
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=2)
            fig.update_yaxes(title_text="Score (%)", row=1, col=1)
            fig.update_yaxes(title_text="Hours", row=1, col=2)
            fig.update_yaxes(title_text="Task Count", row=2, col=1)
            fig.update_yaxes(title_text="Completion (%)", row=2, col=2)
            
            return fig
            
        except Exception as e:
            print(f"Error creating comprehensive dashboard: {e}")
            return None


# Test function
def test_forecaster():
    """Test the time series forecaster with sample data."""
    print("\n" + "="*60)
    print("TESTING TIME SERIES FORECASTER")
    print("="*60)
    
    # Create sample data with variation
    dates = pd.date_range(start='2024-01-01', periods=14, freq='D')
    np.random.seed(42)
    
    # Create varying productivity scores (50-90 range)
    productivity_scores = 70 + 20 * np.sin(np.arange(14) * 0.5) + np.random.normal(0, 5, 14)
    productivity_scores = np.clip(productivity_scores, 50, 90)
    
    # Create varying workload (2-8 hours range)
    workload_hours = 5 + 3 * np.sin(np.arange(14) * 0.3) + np.random.normal(0, 1, 14)
    workload_hours = np.clip(workload_hours, 2, 8)
    
    # Create sample DataFrame
    sample_data = pd.DataFrame({
        'date': np.repeat(dates, 5),
        'task': [f'Task {i}' for i in range(70)],
        'completed': np.random.choice([True, False], 70, p=[0.7, 0.3]),
        'time_taken': np.repeat(workload_hours * 60 / 5, 5),  # Convert to minutes
        'category': np.random.choice(['Work', 'Study', 'Personal', 'Fitness'], 70)
    })
    
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Date range: {sample_data['date'].min()} to {sample_data['date'].max()}")
    print(f"Productivity scores range: {productivity_scores.min():.1f} - {productivity_scores.max():.1f}")
    
    # Initialize forecaster
    forecaster = TimeSeriesForecaster()
    
    # Test individual forecasts
    print("\n--- Testing Individual Forecasts ---")
    
    # Productivity score
    hist_score, forecast_score = forecaster.forecast_productivity_score(sample_data, 7)
    if forecast_score is not None:
        print(f"Productivity Forecast: {forecast_score.tolist()}")
        print(f"Forecast variation: {forecast_score.std():.2f}")
    
    # Workload
    hist_workload, forecast_workload = forecaster.forecast_workload(sample_data, 7)
    if forecast_workload is not None:
        print(f"Workload Forecast (hours): {[x/60 for x in forecast_workload.tolist()]}")
    
    # Get comprehensive summary
    print("\n--- Generating Forecast Summary ---")
    summary = forecaster.get_forecast_summary(sample_data, 7)
    
    print("\n--- Forecast Insights ---")
    for insight in summary.get('insights', []):
        print(f"• {insight}")
    
    return forecaster, sample_data


if __name__ == "__main__":
    # Run test
    forecaster, test_data = test_forecaster()
    
    # Generate a sample chart
    print("\n--- Generating Sample Chart ---")
    hist_score, forecast_score = forecaster.forecast_productivity_score(test_data, 7)
    
    if forecast_score is not None:
        chart = forecaster.create_forecast_chart(
            hist_score, forecast_score,
            'Productivity Score: Historical & Forecast',
            'Productivity Score (%)',
            '#6366F1'
        )
        
        if chart:
            print("✅ Chart generated successfully!")
            # In Streamlit, you would use: st.plotly_chart(chart, use_container_width=True)
        else:
            print("❌ Failed to generate chart")