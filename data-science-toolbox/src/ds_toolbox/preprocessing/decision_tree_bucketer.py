"""
src/credit_toolbox/transformers/dt_bucketer.py

Continuous feature discretization using a Decision Tree approach.
Transforms continuous numerical features into optimal categorical bins 
that maximize the separation of the binary target variable (Good vs. Bad).
Designed to be chained immediately before the WOEEncoder.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from credit_toolbox.core.exceptions import TransformerError
from credit_toolbox.core.types import ArrayLike
from credit_toolbox.logging.decorators import log_execution_time


class DecisionTreeBucketer(BaseEstimator, TransformerMixin):
    """
    Supervised binning for numerical features using a Decision Tree Classifier.
    
    Instead of arbitrary percentiles, this algorithm finds exact continuous cut-offs 
    (e.g., Income < $45,230) that yield the highest information gain against the target.
    """

    def __init__(
        self, 
        cols: Optional[List[str]] = None, 
        max_depth: int = 3, 
        min_samples_leaf: Union[int, float] = 0.05,
        random_state: int = 42
    ):
        """
        Args:
            cols: Numerical columns to bucket. If None, applies to all numeric columns.
            max_depth: Maximum depth of the tree. A depth of 3 yields up to 8 bins (2^3).
            min_samples_leaf: Minimum volume of data in a bin. If a float, it represents a fraction 
                              of the total population (e.g., 0.05 = 5%). Prevents overfitting.
            random_state: Seed for reproducibility in the tree algorithm.
        """
        self.cols = cols
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    @log_execution_time
    def fit(self, X: pd.DataFrame, y: ArrayLike) -> "DecisionTreeBucketer":
        """
        Fits a shallow decision tree to each continuous feature to extract optimal split points.
        """
        if not isinstance(X, pd.DataFrame):
            raise TransformerError("DecisionTreeBucketer requires a pandas DataFrame.")
            
        y_s = pd.Series(y)
        if y_s.nunique() != 2:
            raise TransformerError("DecisionTreeBucketer requires a binary target variable.")

        # Default to all numeric columns if none specified
        if self.cols is None:
            self.cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.cols_ = self.cols

        if not self.cols_:
            raise TransformerError("No numerical columns found or specified for bucketing.")

        self.bin_edges_: Dict[str, List[float]] = {}

        for col in self.cols_:
            if col not in X.columns:
                raise TransformerError(f"Column '{col}' not found in the DataFrame.")

            # Filter out NaNs for training the tree, as sklearn's default DecisionTree
            # cannot natively route missing values without complex imputation.
            valid_idx = X[col].notna()
            if not valid_idx.any():
                # If the entire column is NaN, set boundaries that encompass everything
                self.bin_edges_[col] = [-np.inf, np.inf]
                continue

            X_subset = X.loc[valid_idx, [col]]
            y_subset = y_s.loc[valid_idx]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
            tree.fit(X_subset, y_subset)

            # Extract thresholds from the underlying C structure of the tree.
            # Scikit-learn uses -2 (Tree.UNDEFINED) to represent leaf nodes.
            thresholds = tree.tree_.threshold
            splits = thresholds[thresholds != -2]

            # Construct the definitive list of bin edges, bounded by infinity
            boundaries = [-np.inf] + sorted(list(splits)) + [np.inf]
            
            # Remove any duplicate edges to prevent pd.cut from crashing
            boundaries = sorted(list(set(boundaries)))
            
            self.bin_edges_[col] = boundaries

        return self

    @log_execution_time
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Discretizes the continuous features into string interval categories based on fitted edges.
        """
        check_is_fitted(self, 'bin_edges_')
        
        if not isinstance(X, pd.DataFrame):
            raise TransformerError("DecisionTreeBucketer requires a pandas DataFrame.")

        X_transformed = X.copy()

        for col in self.cols_:
            if col not in X_transformed.columns:
                raise TransformerError(f"Column '{col}' expected but not found in input.")

            edges = self.bin_edges_[col]

            # pd.cut creates categorical intervals (e.g., "(-inf, 25000.0]").
            # We cast to string so the WOEEncoder downstream can treat them as standard categorical variables.
            X_transformed[col] = pd.cut(
                X_transformed[col], 
                bins=edges, 
                include_lowest=True, 
                duplicates='drop'
            ).astype(str)

            # pd.cut converts np.nan into the string "nan". 
            # We explicitly replace it with "Missing" so the WOEEncoder can calculate 
            # a specific Weight of Evidence penalty/bonus for missing data.
            X_transformed[col] = X_transformed[col].replace('nan', 'Missing')

        return X_transformed