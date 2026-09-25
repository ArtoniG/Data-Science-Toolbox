"""
src/credit_toolbox/transformers/woe_encoder.py

Scikit-Learn compatible Weight of Evidence (WOE) transformer.
Designed to integrate seamlessly into `sklearn.pipeline.Pipeline` to prevent 
data leakage during cross-validation and hyperparameter tuning.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from credit_toolbox.core.exceptions import TransformerError
from credit_toolbox.core.types import ArrayLike
from credit_toolbox.logging.decorators import log_execution_time


class WOEEncoder(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence (WOE) Encoder for categorical features.
    
    WOE evaluates the predictive power of a categorical bin in relation to a binary target.
    Formula: $WOE = \\ln( \\% \\text{Non-Events} / \\% \\text{Events} )$
    
    If the target y=1 represents a "Bad" loan (Event) and y=0 represents a "Good" loan 
    (Non-Event), a negative WOE implies higher risk, while a positive WOE implies lower risk.
    """

    def __init__(
        self, 
        cols: Optional[List[str]] = None, 
        epsilon: float = 1e-6, 
        unknown_strategy: Union[float, str] = 0.0
    ):
        """
        Args:
            cols: List of column names to encode. If None, encodes all `object` and `category` dtype columns.
            epsilon: Small constant added to the numerator and denominator to prevent log(0) or division by zero.
            unknown_strategy: What to do with categories in `transform` that were not seen in `fit`. 
                              Defaults to 0.0 (neutral WOE).
        """
        self.cols = cols
        self.epsilon = epsilon
        self.unknown_strategy = unknown_strategy

    @log_execution_time
    def fit(self, X: pd.DataFrame, y: ArrayLike) -> "WOEEncoder":
        """
        Calculates and stores the WOE mappings for each specified categorical column based on the training data.
        """
        if not isinstance(X, pd.DataFrame):
            raise TransformerError("WOEEncoder currently only supports pandas DataFrames as input.")

        y_s = pd.Series(y)
        if y_s.nunique() != 2:
            raise TransformerError("WOEEncoder requires a binary target variable (exactly 2 unique classes).")

        # Automatically select categorical columns if none are provided
        if self.cols is None:
            self.cols_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            self.cols_ = self.cols

        if not self.cols_:
            raise TransformerError("No categorical columns found or specified to encode.")

        self.woe_maps_: Dict[str, Dict[str, float]] = {}
        
        # Total counts of Events (1) and Non-Events (0) in the training target
        total_events = y_s.sum()
        total_non_events = len(y_s) - total_events

        # Ensure totals are safe for division
        safe_total_events = max(total_events, self.epsilon)
        safe_total_non_events = max(total_non_events, self.epsilon)

        for col in self.cols_:
            if col not in X.columns:
                raise TransformerError(f"Column '{col}' not found in the input DataFrame.")

            # Combine feature and target to easily group by category
            df_temp = pd.DataFrame({'feature': X[col], 'target': y_s})
            
            # Aggregate the number of events (1s) and non-events (0s) per category
            grouped = df_temp.groupby('feature')['target'].agg(['sum', 'count'])
            grouped.columns = ['events', 'total']
            grouped['non_events'] = grouped['total'] - grouped['events']

            # Calculate relative percentages
            pct_events = grouped['events'] / safe_total_events
            pct_non_events = grouped['non_events'] / safe_total_non_events

            # Apply epsilon smoothing to prevent log(0)
            pct_events_safe = np.maximum(pct_events, self.epsilon)
            pct_non_events_safe = np.maximum(pct_non_events, self.epsilon)

            # Calculate WOE: ln( % Non-Events / % Events )
            woe_values = np.log(pct_non_events_safe / pct_events_safe)
            
            # Store the mapping for this specific column
            self.woe_maps_[col] = woe_values.to_dict()

        return self

    @log_execution_time
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the stored WOE mappings to the data. Replaces categories with their computed WOE continuous values.
        """
        check_is_fitted(self, 'woe_maps_')

        if not isinstance(X, pd.DataFrame):
            raise TransformerError("WOEEncoder requires a pandas DataFrame for transformation.")

        X_transformed = X.copy()

        for col in self.cols_:
            if col not in X_transformed.columns:
                raise TransformerError(f"Column '{col}' expected by WOEEncoder but not found in input.")

            # Map categories to WOE values
            mapping = self.woe_maps_[col]
            
            # Use pandas map. Unseen categories will become NaN.
            X_transformed[col] = X_transformed[col].map(mapping)

            # Handle unknown categories (NaNs) created by the mapping
            if X_transformed[col].isnull().any():
                X_transformed[col] = X_transformed[col].fillna(self.unknown_strategy)
                
            # Coerce to float64 to ensure ML model compatibility
            X_transformed[col] = X_transformed[col].astype(np.float64)

        return X_transformed