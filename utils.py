# ==================== IMPORTS ====================
# Data processing
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Optional

# Logging
import logging

# Local modules
from data_constants import DEFAULT_PRODUCTIVE_CATEGORIES, DEFAULT_WEIGHTS, NUMERIC_FIELDS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> Tuple[bool, str]:
    """Validate dataframe structure and content with improved checks."""
    if df.empty:
        return False, "Dataframe is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    # Enhanced numeric column checks
    numeric_cols = ['time_taken', 'difficulty', 'energy_level', 'focus_level']
    for col in numeric_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='raise')
                except Exception as e:
                    return False, f"Column {col} contains non-numeric values: {str(e)}"
    
    # Robust datetime checks
    datetime_cols = ['date', 'start_time', 'end_time']
    for col in datetime_cols:
        if col in df.columns:
            try:
                pd.to_datetime(df[col])
            except Exception as e:
                return False, f"Column {col} contains invalid date/time values: {str(e)}"
    
    return True, "Data validation passed"

def calculate_productivity_score(
    df: pd.DataFrame,
    productive_categories: Optional[List[str]] = None,
    weights: Optional[dict] = None
) -> Tuple[float, float, float, float]:
    """
    Calculate comprehensive productivity score with configurable parameters.
    
    Args:
        df: Input DataFrame containing task data
        productive_categories: List of categories considered productive
        weights: Dictionary containing 'time' and 'completion' weights
        
    Returns:
        Tuple of (score, productive_time, total_time, completion_rate)
    """
    if df.empty:
        logger.warning("Empty DataFrame passed to productivity calculator")
        return 0.0, 0.0, 0.0, 0.0
    
    # Use defaults if not specified
    productive_cats = productive_categories or DEFAULT_PRODUCTIVE_CATEGORIES
    weights = weights or DEFAULT_WEIGHTS
    
    # Validate weights
    if not (0 <= weights['time'] <= 1) or not (0 <= weights['completion'] <= 1):
        raise ValueError("Weights must be between 0 and 1")
    if not abs(weights['time'] + weights['completion'] - 1.0) < 0.0001:
        raise ValueError("Weights must sum to 1.0")
    
    try:
        df = df.copy()
        df["time_taken"] = pd.to_numeric(df["time_taken"], errors="coerce").fillna(0)
        df["completed"] = df["completed"].fillna(False).astype(bool)
        
        total_time = df["time_taken"].sum()
        productive_time = df[df["completed"] & df["category"].isin(productive_cats)]["time_taken"].sum()
        
        completed_tasks = df["completed"].sum()
        total_tasks = len(df)
        completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Calculate weighted score
        time_component = (productive_time / total_time) if total_time > 0 else 0
        completion_component = (completion_rate / 100)
        
        score = ((time_component * weights['time']) + 
                (completion_component * weights['completion'])) * 100
        
        return (
            round(score, 1),
            round(productive_time, 1),
            round(total_time, 1),
            round(completion_rate, 1)
        )
        
    except Exception as e:
        logger.error(f"Error calculating productivity score: {str(e)}")
        return 0.0, 0.0, 0.0, 0.0

def filter_recent_data(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """Filter data for the last N days with improved date handling."""
    if df.empty or 'date' not in df.columns:
        return pd.DataFrame()
    
    try:
        # Convert date column if needed
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Filter based on days
        cutoff_date = datetime.now() - timedelta(days=days-1)
        return df[df['date'] >= cutoff_date].copy()
    
    except Exception as e:
        logger.error(f"Error filtering recent data: {str(e)}")
        return pd.DataFrame()

# Additional utility functions can be added below