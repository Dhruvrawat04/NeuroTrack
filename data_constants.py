# constants.py
"""
Unified constants for the NeuroTrack application.
Includes data schema, defaults, types, validation rules, and UI settings.
"""

# ==================== COLUMN DEFINITIONS ====================
COLUMN_ORDER = [
    "date",
    "task",
    "start_time",
    "end_time",
    "time_taken",
    "category",
    "priority",
    "mood",
    "energy_level",
    "focus_level",
    "intent",
    "difficulty",
    "tags",
    "notes",
    "task_type",
    "completed"
]

# ==================== DATA DEFAULTS ====================
# Used consistently across all modules when filling NaN values
NUMERIC_DEFAULTS = {
    "time_taken": 0.0,        # float: minutes
    "energy_level": 5,        # int: 1-10 scale
    "focus_level": 5,         # int: 1-10 scale
    "difficulty": 3           # int: 1-5 scale
}

CATEGORICAL_DEFAULTS = {
    "priority": "Medium",
    "mood": "😐 Neutral",
    "intent": "Complete",
    "category": "General",
    "task_type": "manual"
}

STRING_DEFAULTS = {
    "notes": "",
    "tags": ""
}

BOOLEAN_DEFAULTS = {
    "completed": False
}

# Flat dictionary for backward compatibility
DEFAULT_VALUES = {
    "completed": False,
    "time_taken": 0.0,
    "energy_level": 5,
    "focus_level": 5,
    "difficulty": 3,
    "priority": "Medium",
    "mood": "😐 Neutral",
    "intent": "Complete",
    "tags": "",
    "notes": "",
    "task_type": "manual"
}

# ==================== DATA TYPE DEFINITIONS ====================
# Strictly define expected types for each column
COLUMN_TYPES = {
    "date": "date",              # Python date object
    "task": "str",               # lowercase, stripped
    "start_time": "datetime64",  # pandas datetime
    "end_time": "datetime64",    # pandas datetime
    "time_taken": "float",       # minutes
    "category": "str",
    "priority": "str",           # "Low", "Medium", "High"
    "mood": "str",               # emoji mood
    "energy_level": "int",       # 1-10
    "focus_level": "int",        # 1-10
    "intent": "str",             # "Complete", "Learn", etc.
    "difficulty": "int",         # 1-5
    "tags": "list",              # list of strings
    "notes": "str",
    "task_type": "str",
    "completed": "bool"
}

# ==================== DATA VALIDATION RULES ====================
NUMERIC_RANGES = {
    "energy_level": (1, 10),
    "focus_level": (1, 10),
    "difficulty": (1, 5),
    "time_taken": (0, float('inf'))
}

NUMERIC_FIELDS = {
    "energy_level": {"min": 1, "max": 10, "default": 5},
    "focus_level": {"min": 1, "max": 10, "default": 5},
    "difficulty": {"min": 1, "max": 5, "default": 3},
    "time_taken": {"min": 0, "max": 1440, "default": 30}
}

VALID_PRIORITY_VALUES = ["Low", "Medium", "High"]
VALID_INTENT_VALUES = ["Complete", "Learn", "Review", "Plan", "Practice", "Explore"]

CATEGORICAL_FIELDS = {
    "priority": ["Low", "Medium", "High"],
    "intent": ["Complete", "Learn", "Review", "Plan", "Practice", "Explore"],
    "mood": ["😊 Happy", "😐 Neutral", "😞 Tired", "😤 Frustrated", "💪 Energized"]
}

# ==================== DATA SERIALIZATION RULES ====================
# How to convert to/from CSV format
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"
TAGS_SEPARATOR = ","

# ==================== BUSINESS LOGIC CONSTANTS ====================
# Productivity categories
DEFAULT_PRODUCTIVE_CATEGORIES = ["Coding", "Academics", "Development", "Project"]

# Productivity scoring weights
DEFAULT_WEIGHTS = {"time": 0.7, "completion": 0.3}

# ==================== VISUALIZATION COLORS ====================
# Heatmap colors for insights charts
HEATMAP_COLORS = [
    [0.0, '#0B0B3B'],     # Deep navy
    [0.1, '#1A1A7A'],     # Royal blue
    [0.3, '#4B0082'],     # Indigo
    [0.5, '#8B0000'],     # Dark red
    [0.7, '#ef4444'],     # Red hot
    [0.9, '#FF4500'],     # Orange-red
    [1.0, '#FFD700']      # Gold/yellow (hottest)
]

# Priority-based color mapping
PRIORITY_COLORS = {
    'High': '#ef4444',    # Red (hottest)
    'Medium': '#f59e0b',  # Amber (warm)
    'Low': '#1d4ed8'      # Blue (cool)
}
