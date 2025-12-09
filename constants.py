# constants.py
"""
Backward compatibility wrapper for data_constants.py
All constants have been consolidated in data_constants.py
Import from there for all project needs.
"""

# Import everything from the unified constants file
from data_constants import (
    # Column definitions
    COLUMN_ORDER,
    
    # Data defaults
    NUMERIC_DEFAULTS,
    CATEGORICAL_DEFAULTS,
    STRING_DEFAULTS,
    BOOLEAN_DEFAULTS,
    DEFAULT_VALUES,
    
    # Data types
    COLUMN_TYPES,
    
    # Validation rules
    NUMERIC_RANGES,
    NUMERIC_FIELDS,
    VALID_PRIORITY_VALUES,
    VALID_INTENT_VALUES,
    CATEGORICAL_FIELDS,
    
    # Serialization
    DATETIME_FORMAT,
    DATE_FORMAT,
    TAGS_SEPARATOR,
    
    # Business logic
    DEFAULT_PRODUCTIVE_CATEGORIES,
    DEFAULT_WEIGHTS,
    
    # Visualization colors
    HEATMAP_COLORS,
    PRIORITY_COLORS
)

__all__ = [
    'COLUMN_ORDER',
    'NUMERIC_DEFAULTS',
    'CATEGORICAL_DEFAULTS',
    'STRING_DEFAULTS',
    'BOOLEAN_DEFAULTS',
    'DEFAULT_VALUES',
    'COLUMN_TYPES',
    'NUMERIC_RANGES',
    'NUMERIC_FIELDS',
    'VALID_PRIORITY_VALUES',
    'VALID_INTENT_VALUES',
    'CATEGORICAL_FIELDS',
    'DATETIME_FORMAT',
    'DATE_FORMAT',
    'TAGS_SEPARATOR',
    'DEFAULT_PRODUCTIVE_CATEGORIES',
    'DEFAULT_WEIGHTS',
    'HEATMAP_COLORS',
    'PRIORITY_COLORS'
]
