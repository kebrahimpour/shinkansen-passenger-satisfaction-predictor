"""Minimal SatisfactionPredictor implementation for Shinkansen passenger satisfaction prediction."""

import pickle
import pandas as pd
from typing import Union, List, Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator


class SatisfactionPredictor:
    """Minimal satisfaction predictor with fit, save, load, and predict functionality.
    
    This class provides a simple interface for training machine learning models
    to predict passenger satisfaction scores, with support for model persistence.
    """
    
    def __init__(self, model: BaseEstimator = None):
        """Initialize the predictor.
        
        Args:
            model: Scikit-learn estimator to use for prediction.
                  Defaults to LinearRegression if None.
        """
        self.model = model if model is not None else LinearRegression()
        self._is_fitted = False
    
    def fit(self, X: Union[pd.DataFrame, List[Dict]], y: Union[pd.Series, List]) -> 'SatisfactionPredictor':
        """Fit the model using training data.
        
        Args:
            X: Training features as DataFrame or list of dictionaries
            y: Training targets as Series or list
            
        Returns:
            self: Returns the fitted predictor instance
            
        Examples:
            >>> predictor = SatisfactionPredictor()
            >>> X = [{'duration': 120, 'class': 'Green'}, {'duration': 90, 'class': 'Ordinary'}]
            >>> y = [4.5, 3.8]
            >>> predictor.fit(X, y)  # doctest: +ELLIPSIS
            <...SatisfactionPredictor object at 0x...>
        """
        # Convert input to DataFrame if it's a list of dicts
        if isinstance(X, list) and all(isinstance(item, dict) for item in X):
            X = pd.DataFrame(X)
        
        # Handle categorical variables with simple encoding
        X_processed = self._preprocess_features(X)
        
        # Fit the model
        self.model.fit(X_processed, y)
        self._is_fitted = True
        
        return self
    
    def predict(self, journey_data: Union[Dict, List[Dict], pd.DataFrame]) -> Union[float, List[float]]:
        """Predict satisfaction score(s) for journey data.
        
        Args:
            journey_data: Journey data as dict, list of dicts, or DataFrame
            
        Returns:
            Predicted satisfaction score(s)
            
        Examples:
            >>> predictor = SatisfactionPredictor()
            >>> # Mock a fitted model for doctest
            >>> predictor._is_fitted = True
            >>> import numpy as np
            >>> predictor.model.coef_ = np.array([0.1])
            >>> predictor.model.intercept_ = 4.0
            >>> journey = {'duration': 120, 'class': 'Green'}
            >>> score = predictor.predict(journey)
            >>> isinstance(score, (int, float))
            True
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Handle single dictionary input
        if isinstance(journey_data, dict):
            journey_data = [journey_data]
            single_prediction = True
        else:
            single_prediction = False
        
        # Convert to DataFrame if needed
        if isinstance(journey_data, list):
            journey_data = pd.DataFrame(journey_data)
        
        # Preprocess features
        X_processed = self._preprocess_features(journey_data)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        # Return single value or list based on input
        if single_prediction:
            return float(predictions[0])
        else:
            return predictions.tolist()
    
    def save_model(self, model_path: str) -> None:
        """Save the trained model to disk using pickle.
        
        Args:
            model_path: Path where to save the model
            
        Examples:
            >>> import tempfile
            >>> import os
            >>> predictor = SatisfactionPredictor()
            >>> with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            ...     predictor.save_model(f.name)
            ...     os.path.exists(f.name)
            True
        """
        model_data = {
            'model': self.model,
            'is_fitted': self._is_fitted
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, model_path: str) -> None:
        """Load a trained model from disk using pickle.
        
        Args:
            model_path: Path to the saved model file
            
        Examples:
            >>> import tempfile
            >>> import os
            >>> # Create and save a model first
            >>> predictor1 = SatisfactionPredictor()
            >>> with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            ...     predictor1.save_model(f.name)
            ...     # Load in a new instance
            ...     predictor2 = SatisfactionPredictor()
            ...     predictor2.load_model(f.name)
            ...     isinstance(predictor2.model, LinearRegression)
            True
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self._is_fitted = model_data['is_fitted']
    
    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Simple preprocessing of features.
        
        This is a minimal implementation that handles basic categorical encoding.
        In a production system, this would be more sophisticated.
        
        Args:
            X: Input features as DataFrame
            
        Returns:
            Preprocessed features
        """
        X_processed = X.copy()
        
        # Simple categorical encoding for common columns
        if 'class' in X_processed.columns:
            # Map service classes to numeric values
            class_mapping = {'Ordinary': 1, 'Green': 2, 'GranClass': 3}
            X_processed['class'] = X_processed['class'].map(class_mapping).fillna(1)
        
        # Fill missing numeric values with median
        numeric_columns = X_processed.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())
        
        # For any remaining categorical columns, use simple label encoding
        categorical_columns = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X_processed[col] = pd.Categorical(X_processed[col]).codes
        
        return X_processed


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
    # Additional simple tests
    print("Running basic functionality tests...")
    
    # Test basic workflow
    predictor = SatisfactionPredictor()
    
    # Sample training data
    X_train = [
        {'duration': 120, 'class': 'Green', 'on_time': 0.95},
        {'duration': 90, 'class': 'Ordinary', 'on_time': 0.88},
        {'duration': 180, 'class': 'GranClass', 'on_time': 0.99},
        {'duration': 60, 'class': 'Ordinary', 'on_time': 0.92}
    ]
    y_train = [4.5, 3.8, 4.9, 4.1]
    
    # Fit the model
    predictor.fit(X_train, y_train)
    print("✓ Model fitted successfully")
    
    # Test prediction
    test_journey = {'duration': 100, 'class': 'Green', 'on_time': 0.93}
    score = predictor.predict(test_journey)
    print(f"✓ Prediction successful: {score:.2f}")
    
    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        model_path = f.name
    
    predictor.save_model(model_path)
    print("✓ Model saved successfully")
    
    # Load in new instance
    new_predictor = SatisfactionPredictor()
    new_predictor.load_model(model_path)
    new_score = new_predictor.predict(test_journey)
    print(f"✓ Model loaded successfully: {new_score:.2f}")
    
    # Clean up
    import os
    os.unlink(model_path)
    
    print("All tests passed! 🚅")
