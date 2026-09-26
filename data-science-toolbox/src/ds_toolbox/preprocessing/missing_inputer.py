"""
src/credit_toolbox/transformers/missing_imputer.py

Credit-specific missing value imputation for Scikit-Learn pipelines.
Handles "Thin File" or "No Hit" scenarios by mapping missing numerical values 
to specific out-of-range integers (e.g., -9999) and missing categorical values 
to distinct string labels (e.g., "Missing"), preserving the predictive signal of missingness.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from credit_toolbox.core.exceptions import TransformerError
from credit_toolbox.core.types import ArrayLike
from credit_toolbox.logging.decorators import log_execution_time


class CreditMissingImputer(BaseEstimator, TransformerMixin):
    """
    Stateful imputation transformer designed for credit bureau data.
    
    Standard ML often uses mean/median imputation, which destroys the risk signal 
    associated with missing data. In credit risk, a missing 'Months Since Oldest Trade' 
    usually means the applicant has no credit history (Thin File). By filling with a 
    constant like -9999, tree algorithms and binning functions can cleanly isolate 
    the "missing" population into its own risk bucket.
    """

    def __init__(
        self, 
        cols: Optional[List[str]] = None, 
        numerical_strategy: str = 'constant', 
        numerical_constant: float = -9999.0,
        categorical_constant: str = 'Missing'
    ):
        """
        Args:
            cols: Columns to impute. If None, applies to all columns containing NaNs.
            numerical_strategy: 'constant', 'median', or 'mean'. 'constant' is highly 
                                recommended for credit risk tree-based models.
            numerical_constant: The out-of-range value used when strategy is 'constant'.
            categorical_constant: The string label used to replace NaNs in object/category columns.
        """
        if numerical_strategy not in ['constant', 'median', 'mean']:
            raise ValueError("numerical_strategy must be one of: 'constant', 'median', 'mean'.")

        self.cols = cols
        self.numerical_strategy = numerical_strategy
        self.numerical_constant = numerical_constant
        self.categorical_constant = categorical_constant

    @log_execution_time
    def fit(self, X: pd.DataFrame, y: Optional[ArrayLike] = None) -> "CreditMissingImputer":
        """
        Calculates and stores the imputation values for each column based on the training data.
        """
        if not isinstance(X, pd.DataFrame):
            raise TransformerError("CreditMissingImputer requires a pandas DataFrame.")

        if self.cols is None:
            self.cols_ = X.columns.tolist()
        else:
            self.cols_ = self.cols

        # Dictionary to store the exact scalar fill value for each column
        self.fill_values_: Dict[str, Any] = {}

        for col in self.cols_:
            if col not in X.columns:
                raise TransformerError(f"Column '{col}' not found in the DataFrame.")

            # Identify if the column is numerical or categorical
            is_numeric = pd.api.types.is_numeric_dtype(X[col])

            if is_numeric:
                if self.numerical_strategy == 'constant':
                    self.fill_values_[col] = self.numerical_constant
                elif self.numerical_strategy == 'median':
                    # Calculate median strictly on training data
                    med_val = X[col].median()
                    # Fallback to constant if the entire training column is NaN
                    self.fill_values_[col] = med_val if not pd.isna(med_val) else self.numerical_constant
                elif self.numerical_strategy == 'mean':
                    mean_val = X[col].mean()
                    self.fill_values_[col] = mean_val if not pd.isna(mean_val) else self.numerical_constant
            else:
                # Categorical / Object column
                self.fill_values_[col] = self.categorical_constant

        return self

    @log_execution_time
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the stored imputation dictionary to fill missing values in the data.
        """
        check_is_fitted(self, 'fill_values_')

        if not isinstance(X, pd.DataFrame):
            raise TransformerError("CreditMissingImputer requires a pandas DataFrame.")

        X_transformed = X.copy()
        
        # Verify all expected columns exist before bulk imputation
        missing_cols = [c for c in self.cols_ if c not in X_transformed.columns]
        if missing_cols:
            raise TransformerError(f"Columns expected but not found in input: {missing_cols}")

        # Pandas .fillna() with a dictionary is highly optimized in C and avoids 
        # the performance penalty and fragmentation of iterating column-by-column.
        X_transformed = X_transformed.fillna(value=self.fill_values_)

        return X_transformed