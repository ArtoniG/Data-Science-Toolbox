"""
src/credit_toolbox/transformers/outlier_capper.py

Stateful Winsorization transformer for Scikit-Learn pipelines.
Clips extreme values in numerical features to specified percentiles (e.g., 1st and 99th) 
calculated strictly on the training set to prevent data leakage during cross-validation.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from credit_toolbox.core.exceptions import TransformerError
from credit_toolbox.core.types import ArrayLike
from credit_toolbox.logging.decorators import log_execution_time


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Stateful percentile capper (Winsorizer) for numerical features.
    
    In credit risk, features like Income, Revolving Balance, or Delinquency Days 
    are highly right-skewed. Extreme outliers distort linear models (Logistic Regression) 
    and can cause instability in Weight of Evidence mapping if not binned correctly. 
    This transformer clips values outside the configured lower and upper percentiles.
    """

    def __init__(
        self, 
        cols: Optional[List[str]] = None, 
        lower_percentile: float = 0.01, 
        upper_percentile: float = 0.99
    ):
        """
        Args:
            cols: List of numerical column names to cap. If None, applies to all numeric columns.
            lower_percentile: The lower bound quantile (0.0 to 1.0). Default is 0.01 (1st percentile).
                              Set to 0.0 to disable lower-bound capping.
            upper_percentile: The upper bound quantile (0.0 to 1.0). Default is 0.99 (99th percentile).
                              Set to 1.0 to disable upper-bound capping.
        """
        if not (0.0 <= lower_percentile < upper_percentile <= 1.0):
            raise ValueError(
                "Invalid percentiles: lower_percentile must be >= 0, upper_percentile <= 1, "
                "and lower_percentile < upper_percentile."
            )
            
        self.cols = cols
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    @log_execution_time
    def fit(self, X: pd.DataFrame, y: Optional[ArrayLike] = None) -> "OutlierCapper":
        """
        Calculates and stores the absolute cut-off values for the requested percentiles 
        based strictly on the training data.
        """
        if not isinstance(X, pd.DataFrame):
            raise TransformerError("OutlierCapper requires a pandas DataFrame.")

        # Default to all numeric columns if none specified
        if self.cols is None:
            self.cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.cols_ = self.cols

        if not self.cols_:
            raise TransformerError("No numerical columns found or specified for outlier capping.")

        # Dictionary to store the fitted boundaries: {column_name: (lower_val, upper_val)}
        self.caps_: Dict[str, Tuple[float, float]] = {}

        for col in self.cols_:
            if col not in X.columns:
                raise TransformerError(f"Column '{col}' not found in the DataFrame.")

            # Calculate quantiles. Pandas automatically ignores NaNs during this calculation.
            lower_val = X[col].quantile(self.lower_percentile) if self.lower_percentile > 0.0 else -np.inf
            upper_val = X[col].quantile(self.upper_percentile) if self.upper_percentile < 1.0 else np.inf
            
            # Fallback for columns with zero variance or entirely NaNs
            if pd.isna(lower_val):
                lower_val = -np.inf
            if pd.isna(upper_val):
                upper_val = np.inf

            self.caps_[col] = (float(lower_val), float(upper_val))

        return self

    @log_execution_time
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the stored absolute cut-off values to clip the data.
        """
        check_is_fitted(self, 'caps_')

        if not isinstance(X, pd.DataFrame):
            raise TransformerError("OutlierCapper requires a pandas DataFrame.")

        X_transformed = X.copy()

        for col in self.cols_:
            if col not in X_transformed.columns:
                raise TransformerError(f"Column '{col}' expected by OutlierCapper but not found in input.")

            lower_val, upper_val = self.caps_[col]
            
            # Apply stateful clipping. 
            # Pandas .clip() safely ignores NaNs, leaving missing data untouched for the next pipeline step.
            X_transformed[col] = X_transformed[col].clip(lower=lower_val, upper=upper_val)

        return X_transformed