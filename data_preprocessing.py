# ==================== IMPORTS ====================
# Data processing
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import logging

# Local modules
from data_constants import NUMERIC_DEFAULTS, CATEGORICAL_DEFAULTS, STRING_DEFAULTS

logger = logging.getLogger(__name__)

# ==================== MODULE DESCRIPTION ====================
"""
Consolidated data preprocessing utilities to reduce code duplication.
Handles date/time conversions, data cleaning, and preparation across the application.
"""

# ==================== DATE/TIME UTILITIES ====================

def prepare_datetime_columns(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """
    Standardized date/time conversion for all DataFrame operations.
    Converts object dtype date/time columns to proper datetime types.
    
    Args:
        df: Input DataFrame
        copy: Whether to copy the DataFrame before modifying
        
    Returns:
        DataFrame with properly formatted date/time columns
    """
    if df.empty:
        return df
    
    result = df.copy() if copy else df
    
    # Convert date column
    if "date" in result.columns and pd.api.types.is_object_dtype(result["date"]):
        result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.date
    
    # Convert start_time column
    if "start_time" in result.columns and pd.api.types.is_object_dtype(result["start_time"]):
        result["start_time"] = pd.to_datetime(result["start_time"], errors="coerce")
    
    # Convert end_time column
    if "end_time" in result.columns and pd.api.types.is_object_dtype(result["end_time"]):
        result["end_time"] = pd.to_datetime(result["end_time"], errors="coerce")
    
    return result


def extract_hour_from_datetime(df: pd.DataFrame, time_column: str = "start_time") -> pd.Series:
    """
    Extract hour from datetime column.
    
    Args:
        df: Input DataFrame
        time_column: Name of the datetime column
        
    Returns:
        Series with hour values (0-23)
    """
    if time_column not in df.columns:
        return pd.Series(dtype=int)
    
    try:
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
        return df[time_column].dt.hour
    except Exception as e:
        logger.error(f"Error extracting hour from {time_column}: {e}")
        return pd.Series(dtype=int)


def extract_date_components(df: pd.DataFrame, date_column: str = "date") -> dict:
    """
    Extract date components (day of week, week, etc.) from date column.
    
    Args:
        df: Input DataFrame
        date_column: Name of the date column
        
    Returns:
        Dictionary with extracted date components as Series
    """
    if df.empty or date_column not in df.columns:
        return {}
    
    result = {}
    try:
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            temp = pd.to_datetime(df[date_column], errors="coerce")
        else:
            temp = df[date_column]
        
        result["hour"] = temp.dt.hour
        result["day_of_week"] = temp.dt.dayofweek
        result["day_name"] = temp.dt.day_name()
        result["week_start"] = temp - pd.to_timedelta(temp.dt.dayofweek, unit='d')
        
    except Exception as e:
        logger.error(f"Error extracting date components: {e}")
    
    return result


# ==================== DATA CLEANING UTILITIES ====================

def ensure_numeric_columns(df: pd.DataFrame, column_mapping: dict = None) -> pd.DataFrame:
    """
    Ensure numeric columns have correct data types with default fill values.
    
    Args:
        df: Input DataFrame
        column_mapping: Dict mapping column names to (dtype, default_value) tuples
                       If None, uses standard numeric columns
        
    Returns:
        DataFrame with properly typed numeric columns
    """
    if df.empty:
        return df
    
    result = df.copy()
    
    if column_mapping is None:
        column_mapping = {
            "time_taken": (float, NUMERIC_DEFAULTS["time_taken"]),
            "energy_level": (int, NUMERIC_DEFAULTS["energy_level"]),
            "focus_level": (int, NUMERIC_DEFAULTS["focus_level"]),
            "difficulty": (int, NUMERIC_DEFAULTS["difficulty"])
        }
    
    for col, (dtype, default_val) in column_mapping.items():
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(default_val).astype(dtype)
    
    return result


def ensure_categorical_columns(df: pd.DataFrame, column_defaults: dict = None) -> pd.DataFrame:
    """
    Ensure categorical columns are strings with proper default values.
    
    Args:
        df: Input DataFrame
        column_defaults: Dict mapping column names to default string values
                        If None, uses standard categorical columns
        
    Returns:
        DataFrame with properly typed categorical columns
    """
    if df.empty:
        return df
    
    result = df.copy()
    
    if column_defaults is None:
        column_defaults = {
            "priority": CATEGORICAL_DEFAULTS["priority"],
            "mood": CATEGORICAL_DEFAULTS["mood"],
            "intent": CATEGORICAL_DEFAULTS["intent"],
            "category": CATEGORICAL_DEFAULTS.get("category", "General"),
            "task_type": CATEGORICAL_DEFAULTS["task_type"]
        }
    
    for col, default_val in column_defaults.items():
        if col in result.columns:
            result[col] = result[col].fillna(default_val).astype(str)
    
    return result


def clean_numeric_range(df: pd.DataFrame, column: str, min_val: int, max_val: int) -> pd.DataFrame:
    """
    Clip numeric column values to specified range.
    
    Args:
        df: Input DataFrame
        column: Column name
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        DataFrame with clipped values
    """
    if df.empty or column not in df.columns:
        return df
    
    result = df.copy()
    result[column] = pd.to_numeric(result[column], errors="coerce").clip(min_val, max_val)
    return result


# ==================== FILTERING UTILITIES ====================

def filter_by_date_range(df: pd.DataFrame, start_date: date = None, end_date: date = None,
                         date_column: str = "date") -> pd.DataFrame:
    """
    Filter DataFrame by date range.
    
    Args:
        df: Input DataFrame
        start_date: Start date (inclusive), defaults to None for no lower bound
        end_date: End date (inclusive), defaults to today
        date_column: Name of the date column
        
    Returns:
        Filtered DataFrame
    """
    if df.empty or date_column not in df.columns:
        return df
    
    result = df.copy()
    
    # Convert date column if needed
    if not pd.api.types.is_datetime64_any_dtype(result[date_column]):
        result[date_column] = pd.to_datetime(result[date_column], errors="coerce").dt.date
    else:
        result[date_column] = result[date_column].dt.date
    
    if end_date is None:
        end_date = date.today()
    
    if start_date is not None:
        result = result[result[date_column] >= start_date]
    
    result = result[result[date_column] <= end_date]
    
    return result


def filter_by_days_back(df: pd.DataFrame, days: int = 7, date_column: str = "date") -> pd.DataFrame:
    """
    Filter DataFrame for last N days.
    
    Args:
        df: Input DataFrame
        days: Number of days to include (defaults to 7)
        date_column: Name of the date column
        
    Returns:
        Filtered DataFrame
    """
    cutoff_date = datetime.now().date() - timedelta(days=days-1)
    return filter_by_date_range(df, start_date=cutoff_date, date_column=date_column)


# ==================== TAGS UTILITIES ====================

def parse_tags_from_string(tags_str: str) -> list:
    """Parse comma-separated tags string into list."""
    if not tags_str or not isinstance(tags_str, str):
        return []
    return [t.strip() for t in tags_str.split(',') if t.strip()]


def convert_tags_to_string(tags: list) -> str:
    """Convert tags list to comma-separated string."""
    if not tags or not isinstance(tags, list):
        return ""
    return ','.join(str(t) for t in tags)


def normalize_task_names(df: pd.DataFrame, task_column: str = "task") -> pd.DataFrame:
    """
    Normalize task names: lowercase and strip whitespace.
    
    Args:
        df: Input DataFrame
        task_column: Name of task column
        
    Returns:
        DataFrame with normalized task names
    """
    if df.empty or task_column not in df.columns:
        return df
    
    result = df.copy()
    result[task_column] = result[task_column].astype(str).str.lower().str.strip()
    return result
