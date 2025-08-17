import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

class TaskRecommender:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.scaler = None
        self.encoder = None
        
    def preprocess_data(self, df: pd.DataFrame):
        """Preprocess data for recommendations."""
        processed_df = df.copy()
        
        # Text features
        text_cols = ['task', 'tags', 'notes']
        for col in text_cols:
            if col not in processed_df.columns:
                processed_df[col] = ''
            processed_df[col] = processed_df[col].fillna('').astype(str)
        
        # Numeric features
        numeric_cols = ['energy_level', 'focus_level', 'difficulty']
        for col in numeric_cols:
            if col not in processed_df.columns:
                processed_df[col] = 0
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
            processed_df[col] = processed_df[col].replace(-1, np.nan)
            processed_df[col] = processed_df[col].fillna(processed_df[col].median()).astype(float)
            if col in ['energy_level', 'focus_level']:
                processed_df[col] = processed_df[col].clip(1, 10)
            elif col == 'difficulty':
                processed_df[col] = processed_df[col].clip(1, 5)

        # Categorical features
        categorical_cols = ['category', 'priority', 'mood', 'intent']
        for col in categorical_cols:
            if col not in processed_df.columns:
                processed_df[col] = 'unknown'
            processed_df[col] = processed_df[col].fillna('unknown').astype(str)

        # Prepare combined text for TF-IDF
        processed_df['tags_str'] = processed_df['tags'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else x
        )
        processed_df['combined_text'] = (
            processed_df['task'] + ' ' + 
            processed_df['tags_str'] + ' ' + 
            processed_df['notes']
        )
        
        return processed_df

    def fit_models(self, processed_df: pd.DataFrame):
        """Fit TF-IDF, scaler and encoder on processed data."""
        # TF-IDF for text
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english', 
            min_df=2,
            ngram_range=(1, 2)  # Add bigrams
        )
        text_features = self.tfidf_vectorizer.fit_transform(processed_df['combined_text'])
        
        # Scale numeric features
        numeric_cols = ['energy_level', 'focus_level', 'difficulty']
        self.scaler = StandardScaler()
        numeric_features = self.scaler.fit_transform(processed_df[numeric_cols])
        
        # Encode categorical features
        categorical_cols = ['category', 'priority', 'mood', 'intent']
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        categorical_features = self.encoder.fit_transform(processed_df[categorical_cols])
        
        # Combine all features
        if hasattr(text_features, 'toarray'):
            text_features = text_features.toarray()
        return np.hstack((text_features, numeric_features, categorical_features))

    def recommend_tasks(self, input_task: dict, df: pd.DataFrame, top_n: int = 3):
        """Recommend similar tasks based on input task."""
        if df.empty or len(df) < 5:
            return pd.DataFrame()
            
        # Define the column lists here so they're available throughout the method
        numeric_cols = ['energy_level', 'focus_level', 'difficulty']
        categorical_cols = ['category', 'priority', 'mood', 'intent']
        
        processed_df = self.preprocess_data(df)
        all_features = self.fit_models(processed_df)
        
        # Prepare input task
        input_task_df = pd.DataFrame([input_task])
        input_task_df = self.preprocess_data(input_task_df)
        
        # Transform input features
        input_text_features = self.tfidf_vectorizer.transform(input_task_df['combined_text'])
        input_numeric_features = self.scaler.transform(input_task_df[numeric_cols])
        input_categorical_features = self.encoder.transform(input_task_df[categorical_cols])
        
        if hasattr(input_text_features, 'toarray'):
            input_text_features = input_text_features.toarray()
        input_features = np.hstack((input_text_features, input_numeric_features, input_categorical_features))
        # Calculate similarity
        similarity_scores = cosine_similarity(input_features, all_features).flatten()
        similar_tasks_series = pd.Series(similarity_scores, index=df.index)
        sorted_similar_tasks = similar_tasks_series.sort_values(ascending=False)
        
        # Remove perfect matches (likely the input task itself)
        perfect_matches = sorted_similar_tasks[sorted_similar_tasks == 1.0].index
        sorted_similar_tasks = sorted_similar_tasks.drop(perfect_matches, errors='ignore')
        
        # Get top N recommendations
        top_indices = sorted_similar_tasks.head(top_n).index
        recommendations = df.loc[top_indices].copy()
        recommendations['similarity_score'] = sorted_similar_tasks.head(top_n).values
        
        return recommendations.sort_values('similarity_score', ascending=False)