# ==================== IMPORTS ====================
# Data processing
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Logging
import logging

logger = logging.getLogger(__name__)

# ==================== MODULE DESCRIPTION ====================
"""
Consolidated encoding utilities for categorical and text feature engineering.
Reduces duplication between ml_models.py and recommendations.py
"""


class CategoricalEncoder:
    """Handles encoding of categorical columns across the application."""
    
    def __init__(self):
        self.encoders = {}
        self.categories_known = {}
    
    def fit_categorical_column(self, df: pd.DataFrame, column: str, handle_unknown: str = 'unknown') -> None:
        """
        Fit encoder for a categorical column.
        
        Args:
            df: Input DataFrame
            column: Column name to encode
            handle_unknown: Default value for unknown categories
        """
        if column not in df.columns:
            logger.warning(f"Column {column} not found in DataFrame")
            return
        
        try:
            values = df[column].fillna(handle_unknown).astype(str).unique().tolist()
            if handle_unknown not in values:
                values.append(handle_unknown)
            
            le = LabelEncoder()
            le.fit(values)
            
            self.encoders[column] = le
            self.categories_known[column] = set(values)
            logger.info(f"Fitted encoder for {column} with {len(values)} categories")
            
        except Exception as e:
            logger.error(f"Error fitting encoder for {column}: {e}")
    
    def encode_column(self, df: pd.DataFrame, column: str, target_column: str = None) -> pd.DataFrame:
        """
        Encode a categorical column to numeric values.
        
        Args:
            df: Input DataFrame
            column: Column name to encode
            target_column: Name of output column (defaults to {column}_encoded)
            
        Returns:
            DataFrame with encoded column added
        """
        if column not in self.encoders:
            logger.warning(f"No encoder found for {column}. Fit the encoder first.")
            return df
        
        if column not in df.columns:
            logger.warning(f"Column {column} not found in DataFrame")
            return df
        
        target_col = target_column or f"{column}_encoded"
        result = df.copy()
        
        try:
            le = self.encoders[column]
            
            # Handle unknown values
            values = result[column].fillna('unknown').astype(str)
            values = values.apply(
                lambda x: x if x in self.categories_known.get(column, set()) else 'unknown'
            )
            
            result[target_col] = le.transform(values)
            logger.debug(f"Encoded {column} to {target_col}")
            
        except Exception as e:
            logger.error(f"Error encoding {column}: {e}")
            result[target_col] = 0
        
        return result
    
    def fit_and_encode(self, df: pd.DataFrame, column: str, target_column: str = None) -> pd.DataFrame:
        """Fit encoder and encode column in one step."""
        self.fit_categorical_column(df, column)
        return self.encode_column(df, column, target_column)
    
    def encode_multiple_columns(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Encode multiple categorical columns at once."""
        result = df.copy()
        for col in columns:
            result = self.encode_column(result, col)
        return result


class TextFeatureExtractor:
    """Handles TF-IDF and text feature extraction."""
    
    def __init__(self, stop_words: str = 'english', min_df: int = 2, ngram_range: tuple = (1, 2)):
        self.vectorizer = None
        self.stop_words = stop_words
        self.min_df = min_df
        self.ngram_range = ngram_range
    
    def create_combined_text_features(self, df: pd.DataFrame,
                                     text_columns: list = None) -> pd.DataFrame:
        """
        Create a combined text feature column from multiple text columns.
        
        Args:
            df: Input DataFrame
            text_columns: List of column names to combine (defaults to ['task', 'tags', 'notes'])
            
        Returns:
            DataFrame with 'combined_text' column added
        """
        if df.empty:
            return df
        
        if text_columns is None:
            text_columns = ['task', 'tags', 'notes']
        
        result = df.copy()
        
        # Initialize combined_text
        result['combined_text'] = ''
        
        for col in text_columns:
            if col not in result.columns:
                continue
            
            # Handle lists (like tags) and strings
            col_text = result[col].apply(
                lambda x: ' '.join(x) if isinstance(x, list) else str(x)
            ).fillna('')
            
            result['combined_text'] += ' ' + col_text
        
        result['combined_text'] = result['combined_text'].str.strip()
        return result
    
    def fit_tfidf(self, df: pd.DataFrame, text_column: str = 'combined_text') -> 'TextFeatureExtractor':
        """
        Fit TF-IDF vectorizer on text column.
        
        Args:
            df: Input DataFrame
            text_column: Column containing text to vectorize
            
        Returns:
            Self for method chaining
        """
        if text_column not in df.columns:
            logger.error(f"Text column {text_column} not found")
            return self
        
        try:
            self.vectorizer = TfidfVectorizer(
                stop_words=self.stop_words,
                min_df=self.min_df,
                ngram_range=self.ngram_range
            )
            self.vectorizer.fit(df[text_column].fillna(''))
            logger.info(f"Fitted TF-IDF vectorizer with {len(self.vectorizer.vocabulary_)} features")
            
        except Exception as e:
            logger.error(f"Error fitting TF-IDF: {e}")
        
        return self
    
    def transform_text(self, texts) -> np.ndarray:
        """
        Transform texts to TF-IDF features.
        
        Args:
            texts: Text or list of texts to transform
            
        Returns:
            Sparse matrix or dense array of TF-IDF features
        """
        if self.vectorizer is None:
            logger.error("Vectorizer not fitted. Call fit_tfidf first.")
            return np.array([])
        
        try:
            result = self.vectorizer.transform(
                [texts] if isinstance(texts, str) else texts
            )
            # Convert to dense array
            if hasattr(result, 'toarray'):
                return result.toarray()
            return result
            
        except Exception as e:
            logger.error(f"Error transforming text: {e}")
            return np.array([])


def prepare_ml_features(df: pd.DataFrame, 
                       categorical_cols: list = None,
                       numeric_cols: list = None,
                       text_cols: list = None) -> tuple:
    """
    Prepare all types of features for ML models in one step.
    
    Args:
        df: Input DataFrame
        categorical_cols: List of categorical column names
        numeric_cols: List of numeric column names
        text_cols: List of text column names for TF-IDF
        
    Returns:
        Tuple of (feature_matrix, encoders, vectorizer)
    """
    if df.empty:
        return np.array([]), {}, None
    
    if categorical_cols is None:
        categorical_cols = ['category', 'priority', 'mood', 'intent']
    if numeric_cols is None:
        numeric_cols = ['energy_level', 'focus_level', 'difficulty']
    if text_cols is None:
        text_cols = ['task', 'tags', 'notes']
    
    result_df = df.copy()
    encoders = {}
    features_list = []
    
    # Encode categorical features
    cat_encoder = CategoricalEncoder()
    for col in categorical_cols:
        if col in result_df.columns:
            result_df = cat_encoder.fit_and_encode(result_df, col)
            encoders[col] = cat_encoder.encoders.get(col)
            features_list.append(result_df[[f"{col}_encoded"]])
    
    # Add numeric features
    for col in numeric_cols:
        if col in result_df.columns:
            features_list.append(result_df[[col]])
    
    # Extract text features
    text_extractor = TextFeatureExtractor()
    result_df = text_extractor.create_combined_text_features(result_df, text_cols)
    result_df = text_extractor.fit_tfidf(result_df)
    
    text_features = text_extractor.transform_text(result_df['combined_text'].fillna('').values)
    if len(text_features) > 0:
        features_list.append(text_features)
    
    # Combine all features
    if features_list:
        combined_features = np.hstack([
            f.values if isinstance(f, pd.DataFrame) else f 
            for f in features_list
        ])
    else:
        combined_features = np.array([])
    
    return combined_features, encoders, text_extractor
