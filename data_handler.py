# ==================== IMPORTS ====================
# Data processing
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time
import os

# Local modules
from data_constants import (
    COLUMN_ORDER, NUMERIC_DEFAULTS, CATEGORICAL_DEFAULTS,
    STRING_DEFAULTS, BOOLEAN_DEFAULTS, DATETIME_FORMAT, DATE_FORMAT, TAGS_SEPARATOR
)

DATA_FILE = "data.csv"

def load_data() -> pd.DataFrame:
    """
    Loads data from data.csv, ensuring correct data types for AI/ML compatibility.
    Converts time-related columns to datetime objects.
    """
    if not os.path.exists(DATA_FILE):
        # Create an empty DataFrame with the defined columns if the file doesn't exist
        empty_df = pd.DataFrame(columns=COLUMN_ORDER)
        print(f"'{DATA_FILE}' not found. Creating an empty DataFrame.")
        return empty_df
    
    try:
        df = pd.read_csv(DATA_FILE)
        
        # Use clean_data for robust conversion after initial load
        df = clean_data(df) # Call clean_data immediately after loading
        
        print(f"Data loaded successfully from '{DATA_FILE}'.")
        return df
    except Exception as e:
        print(f"Error loading data from '{DATA_FILE}': {e}")
        # Return an empty DataFrame with correct columns on error
        return pd.DataFrame(columns=COLUMN_ORDER)

def save_data(df: pd.DataFrame):
    """
    Saves the DataFrame to data.csv, ensuring consistent column order and no index.
    Converts datetime objects back to ISO format strings for saving.
    """
    if df.empty:
        # If DataFrame is empty, create an empty CSV file with headers
        pd.DataFrame(columns=COLUMN_ORDER).to_csv(DATA_FILE, index=False)
        print(f"Empty DataFrame saved to '{DATA_FILE}'.")
        return
    
    # Ensure all required columns exist, add missing ones with default values if necessary
    for col in COLUMN_ORDER:
        if col not in df.columns:
            if col in BOOLEAN_DEFAULTS:
                df[col] = BOOLEAN_DEFAULTS[col]
            elif col in NUMERIC_DEFAULTS:
                df[col] = NUMERIC_DEFAULTS[col]
            elif col in CATEGORICAL_DEFAULTS:
                df[col] = CATEGORICAL_DEFAULTS[col]
            elif col in STRING_DEFAULTS:
                df[col] = STRING_DEFAULTS[col]
            else:
                df[col] = ""

    # Convert datetime objects to ISO format strings before saving
    if 'start_time' in df.columns:
        # Ensure it's datetime first, then format to full timestamp
        df["start_time"] = pd.to_datetime(df["start_time"], errors='coerce').dt.strftime(DATETIME_FORMAT)
    if 'end_time' in df.columns:
        # Ensure it's datetime first, then format to full timestamp
        df["end_time"] = pd.to_datetime(df["end_time"], errors='coerce').dt.strftime(DATETIME_FORMAT)
    if 'date' in df.columns:
        # For date objects, convert to datetime first then format
        df["date"] = pd.to_datetime(df["date"], errors='coerce').dt.strftime(DATE_FORMAT)

    # Convert tags list to comma-separated string for CSV saving
    if 'tags' in df.columns:
        df['tags'] = df['tags'].apply(lambda x: TAGS_SEPARATOR.join(x) if isinstance(x, list) else x)
        df['tags'] = df['tags'].fillna('') # Fill NaN tags (which would be float nan) with empty string

    # Select and reorder columns
    df = df[COLUMN_ORDER]
    
    try:
        df.to_csv(DATA_FILE, index=False)
        print(f"Data saved successfully to '{DATA_FILE}'.")
    except Exception as e:
        print(f"Error saving data to '{DATA_FILE}': {e}")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DataFrame by converting data types, normalizing task names,
    dropping invalid rows, and removing duplicates.
    Handles new AI/ML readiness fields.
    """
    if df.empty:
        return df

    cleaned_df = df.copy()

    # Ensure 'date' column is properly converted and invalid entries are handled
    if "date" in cleaned_df.columns:
        # Convert to datetime64[ns], coercing errors to NaT
        temp_dates = pd.to_datetime(cleaned_df["date"], errors="coerce")
        # Convert to Python date objects, NaT becomes pd.NaT which is then NaN for dropna
        cleaned_df["date"] = temp_dates.dt.date
        # Drop rows where date conversion resulted in NaT/NaN
        cleaned_df.dropna(subset=["date"], inplace=True)

    # Convert to datetime objects, coercing errors will turn invalid dates into NaT
    if "start_time" in cleaned_df.columns:
        cleaned_df["start_time"] = pd.to_datetime(cleaned_df["start_time"], errors="coerce")
    if "end_time" in cleaned_df.columns:
        cleaned_df["end_time"] = pd.to_datetime(cleaned_df["end_time"], errors="coerce")
    
    if "time_taken" in cleaned_df.columns:
        cleaned_df["time_taken"] = pd.to_numeric(cleaned_df["time_taken"], errors="coerce").fillna(0.0) # Ensure float and fill NaN
    
    # Normalize task names: lowercase and strip spaces
    if "task" in cleaned_df.columns:
        cleaned_df["task"] = cleaned_df["task"].astype(str).str.lower().str.strip()
    
    # Fill 'completed' NaNs with False and convert to boolean
    if "completed" in cleaned_df.columns:
        cleaned_df["completed"] = cleaned_df["completed"].fillna(False).astype(bool)

    # Handle new columns: fill NaNs and convert types
    # These defaults are applied if columns are missing or contain NaNs after loading
    if "priority" in cleaned_df.columns:
        cleaned_df["priority"] = cleaned_df["priority"].fillna(CATEGORICAL_DEFAULTS["priority"]).astype(str)
    if "mood" in cleaned_df.columns:
        cleaned_df["mood"] = cleaned_df["mood"].fillna(CATEGORICAL_DEFAULTS["mood"]).astype(str)
    if "energy_level" in cleaned_df.columns:
        cleaned_df["energy_level"] = pd.to_numeric(cleaned_df["energy_level"], errors="coerce").fillna(NUMERIC_DEFAULTS["energy_level"]).astype(int)
    if "focus_level" in cleaned_df.columns:
        cleaned_df["focus_level"] = pd.to_numeric(cleaned_df["focus_level"], errors="coerce").fillna(NUMERIC_DEFAULTS["focus_level"]).astype(int)
    if "intent" in cleaned_df.columns:
        cleaned_df["intent"] = cleaned_df["intent"].fillna(CATEGORICAL_DEFAULTS["intent"]).astype(str)
    if "difficulty" in cleaned_df.columns:
        cleaned_df["difficulty"] = pd.to_numeric(cleaned_df["difficulty"], errors="coerce").fillna(NUMERIC_DEFAULTS["difficulty"]).astype(int)
    if "tags" in cleaned_df.columns:
        # Convert comma-separated string from CSV to list, handle NaNs
        cleaned_df["tags"] = cleaned_df["tags"].fillna("").astype(str).apply(lambda x: [tag.strip() for tag in x.split(TAGS_SEPARATOR) if tag.strip()])
    else: # If 'tags' column is completely missing, add it as empty lists
        cleaned_df["tags"] = [[]] * len(cleaned_df)

    if "notes" in cleaned_df.columns:
        cleaned_df["notes"] = cleaned_df["notes"].fillna(STRING_DEFAULTS["notes"]).astype(str)
    if "task_type" in cleaned_df.columns:
        cleaned_df["task_type"] = cleaned_df["task_type"].fillna(CATEGORICAL_DEFAULTS["task_type"]).astype(str)

    if "notes" in cleaned_df.columns:
        cleaned_df["notes"] = cleaned_df["notes"].fillna("").astype(str)
    if "task_type" in cleaned_df.columns:
        cleaned_df["task_type"] = cleaned_df["task_type"].fillna("manual").astype(str)
    
    # Drop rows with missing essential data (re-check after new column handling)
    initial_rows = len(cleaned_df)
    cleaned_df.dropna(subset=["task", "date", "start_time", "time_taken"], inplace=True)
    cleaned_df = cleaned_df[cleaned_df["time_taken"] > 0]
    
    if len(cleaned_df) < initial_rows:
        print(f"Dropped {initial_rows - len(cleaned_df)} invalid rows during cleaning.")

    # Remove duplicates based on key columns (add new columns to subset if they contribute to uniqueness)
    initial_rows = len(cleaned_df)
    # Assuming date, task, start_time, time_taken are sufficient for uniqueness
    cleaned_df.drop_duplicates(subset=["date", "task", "start_time", "time_taken"], inplace=True)
    if len(cleaned_df) < initial_rows:
        print(f"Dropped {initial_rows - len(cleaned_df)} duplicate rows during cleaning.")
    
    # Recalculate end_time for any rows where it might be missing or incorrect
    if "end_time" in cleaned_df.columns and "start_time" in cleaned_df.columns and "time_taken" in cleaned_df.columns:
        missing_end_time_mask = cleaned_df["end_time"].isna() | (cleaned_df["end_time"] < cleaned_df["start_time"])
        cleaned_df.loc[missing_end_time_mask, "end_time"] = \
            cleaned_df.loc[missing_end_time_mask, "start_time"] + \
            pd.to_timedelta(cleaned_df.loc[missing_end_time_mask, "time_taken"], unit="m")

    # Ensure all datetime columns are indeed datetime objects for further processing
    if "start_time" in cleaned_df.columns:
        cleaned_df["start_time"] = pd.to_datetime(cleaned_df["start_time"])
    if "end_time" in cleaned_df.columns:
        cleaned_df["end_time"] = pd.to_datetime(cleaned_df["end_time"])
    
    return cleaned_df

def is_overlapping(new_start_datetime: datetime, new_end_datetime: datetime, task_date: date, df: pd.DataFrame) -> bool:
    """
    Checks if a new task's timeframe overlaps with any existing tasks on the same day.
    
    Args:
        new_start_datetime (datetime): The full start datetime of the new task.
        new_end_datetime (datetime): The full end datetime of the new task.
        task_date (date): The date of the new task.
        df (pd.DataFrame): The DataFrame containing existing tasks.
        
    Returns:
        bool: True if an overlap exists, False otherwise.
    """
    if df.empty:
        return False

    # Filter tasks for the specific date
    # Ensure 'date' column in df is datetime.date objects for comparison
    # This conversion is also done in clean_data, but a defensive check here is fine.
    df_filtered = df[pd.to_datetime(df['date'], errors='coerce').dt.date == task_date].copy()

    if df_filtered.empty:
        return False

    # Ensure 'start_time' and 'end_time' in daily_tasks are datetime objects for comparison
    df_filtered["start_time"] = pd.to_datetime(df_filtered["start_time"], errors="coerce")
    df_filtered["end_time"] = pd.to_datetime(df_filtered["end_time"], errors="coerce")

    # Drop rows where start_time or end_time could not be converted
    df_filtered.dropna(subset=["start_time", "end_time"], inplace=True)

    for _, row in df_filtered.iterrows():
        existing_start = row["start_time"]
        existing_end = row["end_time"]

        # Check for overlap: (start1 < end2) and (end1 > start2)
        # This condition correctly identifies any overlap, including touch points
        if (new_start_datetime < existing_end) and (new_end_datetime > existing_start):
            return True
            
    return False

def add_manual_task(
    df: pd.DataFrame,
    task_name: str,
    time_taken: int,
    task_date: date,
    start_time_obj: time,
    end_time_obj: time = None, # Now optional, will be computed if None
    category: str = "Other",
    priority: str = "Medium",
    mood: str = "😐 Neutral",
    energy_level: int = 5,
    focus_level: int = 5,
    intent: str = "Complete",
    difficulty: int = 3,
    tags: list = None, # Expects a list from app.py
    notes: str = "",
    task_type: str = "manual"
) -> pd.DataFrame:
    """
    Adds a new manual task to the DataFrame after checking for time overlaps.
    Accepts and stores new AI/ML readiness fields.
    
    Args:
        df (pd.DataFrame): The current DataFrame of tasks.
        task_name (str): The name of the task.
        time_taken (int): Time taken for the task in minutes.
        task_date (date): The date of the task.
        start_time_obj (time): The start time of the task (e.g., datetime.time(9, 0)).
        end_time_obj (time, optional): The end time of the task. If None, it's computed.
        category (str): Category of the task.
        priority (str): Priority of the task ("Low", "Medium", "High").
        mood (str): User's mood during the task.
        energy_level (int): User's energy level (1-10).
        focus_level (int): User's focus level (1-10).
        intent (str): User's intent for the task.
        difficulty (int): Perceived difficulty (1-5).
        tags (list): List of tags for the task.
        notes (str): Additional notes for the task.
        task_type (str): Type of task (e.g., "manual").
        
    Returns:
        pd.DataFrame: The updated DataFrame.
        
    Raises:
        Exception: If the new task overlaps with an existing one.
    """
    # Combine date and time to create full datetime objects for internal logic
    full_start_datetime = datetime.combine(task_date, start_time_obj)
    
    if end_time_obj is None:
        full_end_datetime = full_start_datetime + timedelta(minutes=time_taken)
    else:
        full_end_datetime = datetime.combine(task_date, end_time_obj)

    # Check for overlap before adding
    if is_overlapping(full_start_datetime, full_end_datetime, task_date, df):
        raise Exception("Task time overlaps with an existing task on this date. Please adjust the time.")

    # Ensure tags is a list, default to empty list if None
    if tags is None:
        tags = []

    new_task = {
        "date": task_date,
        "task": task_name.strip(),
        "start_time": full_start_datetime, # Store as full datetime object internally
        "end_time": full_end_datetime,     # Store as full datetime object internally
        "time_taken": float(time_taken), # Ensure float
        "category": category,
        "priority": priority,
        "mood": mood,
        "energy_level": int(energy_level), # Ensure int
        "focus_level": int(focus_level),   # Ensure int
        "intent": intent,
        "difficulty": int(difficulty),     # Ensure int
        "tags": tags, # Store as list internally
        "notes": notes.strip(),
        "task_type": task_type,
        "completed": False # Default for new tasks
    }
    
    new_df = pd.DataFrame([new_task])
    data = pd.concat([df, new_df], ignore_index=True)
    
    data = clean_data(data) # Clean data after concatenation to ensure all types are correct
    save_data(data) # save_data will handle string conversions for CSV
    
    return data

# Example usage (for testing purposes, not part of the main app flow)
if __name__ == "__main__":
    # Ensure data.csv exists for testing
    if not os.path.exists(DATA_FILE):
        initial_data = pd.DataFrame(columns=COLUMN_ORDER)
        save_data(initial_data)

    print("--- Loading Data ---")
    current_data = load_data()
    print("Current Data Head:\n", current_data.head())
    print("Current Data Dtypes:\n", current_data.dtypes)

    print("\n--- Cleaning Data ---")
    current_data = clean_data(current_data)
    print("Cleaned Data Head:\n", current_data.head())
    print("Cleaned Data Dtypes:\n", current_data.dtypes)
    save_data(current_data) # Save after cleaning


    print("\n--- Adding a new task (no overlap) ---")
    try:
        updated_data = add_manual_task(
            current_data,
            "Daily Standup",
            30,
            date.today(),
            time(9, 0, 0),
            category="Meeting",
            priority="High",
            mood="😄 Happy",
            energy_level=8,
            focus_level=7,
            intent="Complete",
            difficulty=2,
            tags=["team", "daily"],
            notes="Quick sync with the team."
        )
        print("Task added successfully. Updated Data Head:\n", updated_data.tail())
        current_data = updated_data
    except Exception as e:
        print(f"Error adding task: {e}")

    print("\n--- Adding another task (no overlap) ---")
    try:
        updated_data = add_manual_task(
            current_data,
            "Work on Project X",
            120,
            date.today(),
            time(10, 0, 0),
            category="Development",
            priority="High",
            mood="💡 Inspired",
            energy_level=9,
            focus_level=9,
            intent="Complete",
            difficulty=4,
            tags=["feature", "sprint"],
            notes="Implementing new user authentication."
        )
        print("Task added successfully. Updated Data Head:\n", updated_data.tail())
        current_data = updated_data
    except Exception as e:
        print(f"Error adding task: {e}")

    print("\n--- Attempting to add an overlapping task ---")
    try:
        # This should cause an overlap with "Daily Standup" (9:00-9:30)
        updated_data = add_manual_task(
            current_data,
            "Quick Meeting",
            15,
            date.today(),
            time(9, 15, 0),
            category="Meeting",
            priority="Medium",
            mood="😐 Neutral",
            energy_level=6,
            focus_level=5,
            intent="Complete",
            difficulty=1,
            tags=["quick", "discussion"],
            notes="Brief discussion on a blocker."
        )
        print("Task added successfully (THIS SHOULD NOT HAPPEN IF OVERLAP WORKS). Updated Data Head:\n", updated_data.tail())
    except Exception as e:
        print(f"Caught expected error: {e}")

    print("\n--- Attempting to add a task that starts exactly when another ends (no overlap) ---")
    try:
        # This should NOT cause an overlap if the logic is (start1 < end2) and (end1 > start2)
        updated_data = add_manual_task(
            current_data,
            "Review Docs",
            60,
            date.today(),
            time(9, 30, 0), # Starts exactly when "Daily Standup" ends
            category="Learning",
            priority="Low",
            mood="😐 Neutral",
            energy_level=7,
            focus_level=6,
            intent="Review",
            difficulty=2,
            tags=["docs", "read"],
            notes="Reviewing project documentation."
        )
        print("Task added successfully. Updated Data Head:\n", updated_data.tail())
        current_data = updated_data
    except Exception as e:
        print(f"Error adding task: {e}")

    print("\n--- Final Data State ---")
    print(load_data())
