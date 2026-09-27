"""
src/credit_toolbox/transformers/decision_tree_bucketer.py

Supervised decision-tree discretization for continuous numerical features.
Automatically discovers non-linear split boundaries that maximize default separation 
(e.g., target = default / non-default) while maintaining strict state-serialization 
capabilities for Phase 6 enterprise auditability.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from credit_toolbox.core.base import StatefulCreditTransformer


class DecisionTreeBucketer(StatefulCreditTransformer):
    """
    Discretizes continuous numerical features into discrete ordinal bins using 1D decision trees.
    
    Guarantees:
    1. Supervised optimal binning based on default risk.
    2. Preservation of feature names and column metadata.
    3. Full export/import state serialization without needing the training dataset during audit.
    """

    def __init__(
        self,
        max_leaf_nodes: Optional[int] = 5,
        min_samples_leaf: Union[int, float] = 0.05,
        criterion: str = "gini",
        features: Optional[List[str]] = None,
    ):
        """
        Args:
            max_leaf_nodes (Optional[int]): Maximum number of bins (leaf nodes) per feature. Default is 5.
            min_samples_leaf (Union[int, float]): Minimum proportion or absolute count of samples required in a bin.
                Default is 0.05 (5% minimum population per bin to avoid micro-segmentation).
            criterion (str): Function to measure split quality ('gini' or 'entropy').
            features (Optional[List[str]]): Specific features to bucket. If None, process all numerical columns.
        """
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.features = features

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DecisionTreeBucketer":
        """
        Fits univariate decision trees on each target numerical column to extract split thresholds.
        
        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Binary default target (0 = Non-Default, 1 = Default).
            
        Returns:
            DecisionTreeBucketer: Fitted transformer instance.
        """
        X = self._validate_dataframe(X)
        if y is None:
            raise ValueError(
                f"[{self.__class__.__name__}] is a supervised transformer and requires a target array 'y'."
            )

        # Determine target columns
        if self.features is None:
            cols_to_fit = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols_to_fit = [c for c in self.features if c in X.columns]

        self.splits_: Dict[str, List[float]] = {}
        
        for col in cols_to_fit:
            # Handle missing values during fitting by isolating non-null rows
            valid_mask = X[col].notna() & y.notna()
            if valid_mask.sum() == 0:
                continue
                
            X_col = X.loc[valid_mask, [col]]
            y_col = y.loc[valid_mask]

            # Fit 1D Decision Tree
            dt = DecisionTreeClassifier(
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                random_state=42,
            )
            dt.fit(X_col, y_col)

            # Extract interior split thresholds from tree internal nodes
            tree_thresholds = dt.tree_.threshold[dt.tree_.feature >= 0]
            
            # Construct sorted bin boundaries with -infinity and +infinity
            sorted_bounds = sorted(list(set(tree_thresholds)))
            bins = [-np.inf] + sorted_bounds + [np.inf]
            
            self.splits_[col] = bins

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms continuous features into ordinal bin indices (0, 1, 2, ...).
        
        Args:
            X (pd.DataFrame): Data to transform.
            
        Returns:
            pd.DataFrame: DataFrame with discretized ordinal features.
        """
        X = self._validate_dataframe(X)
        if not getattr(self, "is_fitted_", False):
            raise ValueError(f"[{self.__class__.__name__}] must be fitted before calling transform().")

        X_out = X.copy()
        
        for col, bins in self.splits_.items():
            if col in X_out.columns:
                # Discretize continuous values into integer bin indices
                # Preserves NaN values as NaN for downstream imputers
                X_out[col] = pd.cut(
                    X_out[col],
                    bins=bins,
                    labels=False,
                    include_lowest=True
                )

        return X_out

    def export_state(self) -> Dict[str, Any]:
        """
        Extracts learned split boundaries into a JSON-serializable dictionary.
        
        Returns:
            Dict[str, Any]: Serialized state dictionary.
        """
        if not getattr(self, "is_fitted_", False):
            raise ValueError(f"[{self.__class__.__name__}] must be fitted before exporting state.")

        return {
            "splits_": self.splits_,
            "hyperparams": {
                "max_leaf_nodes": self.max_leaf_nodes,
                "min_samples_leaf": self.min_samples_leaf,
                "criterion": self.criterion,
                "features": self.features,
            },
        }

    @classmethod
    def load_state(cls, state: Dict[str, Any]) -> "DecisionTreeBucketer":
        """
        Instantiates a pre-fitted DecisionTreeBucketer directly from serialized state.
        
        Args:
            state (Dict[str, Any]): Serialized state dictionary.
            
        Returns:
            DecisionTreeBucketer: Restored, ready-to-transform object.
        """
        hyperparams = state.get("hyperparams", {})
        instance = cls(
            max_leaf_nodes=hyperparams.get("max_leaf_nodes", 5),
            min_samples_leaf=hyperparams.get("min_samples_leaf", 0.05),
            criterion=hyperparams.get("criterion", "gini"),
            features=hyperparams.get("features", None),
        )
        
        # Reconstruct float representation (convert JSON lists back to float lists)
        raw_splits = state.get("splits_", {})
        restored_splits = {}
        for col, bounds in raw_splits.items():
            restored_splits[col] = [float(b) for b in bounds]
            
        instance.splits_ = restored_splits
        instance.is_fitted_ = True
        return instance