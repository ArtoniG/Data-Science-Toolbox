"""
src/credit_toolbox/transformers/outlier_capper.py

Domain-specific outlier treatment for credit risk modeling.
Caps extreme values (e.g., at the 1st and 99th percentiles) to prevent 
heavy-tailed financial distributions (like revolving balances or incomes) 
from destabilizing linear models, while ensuring strict state serialization.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from credit_toolbox.core.base import StatefulCreditTransformer


class OutlierCapper(StatefulCreditTransformer):
    """
    Caps continuous numeric features at specified lower and upper quantiles.
    
    Guarantees:
    1. Robustness against heavy-tailed financial variables.
    2. Vectorized DataFrame clipping for performance.
    3. Full export/import state serialization without needing the training dataset during audit.
    """

    def __init__(
        self,
        quantiles: Tuple[float, float] = (0.01, 0.99),
        features: Optional[List[str]] = None,
    ):
        """
        Args:
            quantiles (Tuple[float, float]): The lower and upper quantiles to compute for capping. 
                Default is (0.01, 0.99). Pass None for a boundary to disable capping on that side 
                e.g., (None, 0.99).
            features (Optional[List[str]]): Specific features to cap. If None, processes all numeric columns.
        """
        self.quantiles = quantiles
        self.features = features

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "OutlierCapper":
        """
        Calculates the exact numeric values for the specified quantiles on the training data.
        
        Args:
            X (pd.DataFrame): Training features.
            y (Optional[pd.Series]): Ignored, present for Scikit-Learn pipeline compatibility.
            
        Returns:
            OutlierCapper: Fitted transformer instance.
        """
        X = self._validate_dataframe(X)
        
        if self.features is None:
            cols_to_fit = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols_to_fit = [c for c in self.features if c in X.columns]

        self.lower_caps_: Dict[str, float] = {}
        self.upper_caps_: Dict[str, float] = {}

        # Calculate exact cutoffs, ignoring NaNs naturally via pandas
        for col in cols_to_fit:
            if self.quantiles[0] is not None:
                # Cast to standard python float for JSON serialization safety
                self.lower_caps_[col] = float(X[col].quantile(self.quantiles[0]))
            
            if self.quantiles[1] is not None:
                self.upper_caps_[col] = float(X[col].quantile(self.quantiles[1]))

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the learned upper and lower limits to the features.
        
        Args:
            X (pd.DataFrame): Data to transform.
            
        Returns:
            pd.DataFrame: DataFrame with capped values.
        """
        X = self._validate_dataframe(X)
        if not getattr(self, "is_fitted_", False):
            raise ValueError(f"[{self.__class__.__name__}] must be fitted before calling transform().")

        X_out = X.copy()
        
        # Determine the intersection of columns we have caps for and columns in X
        cols_with_lower = [c for c in self.lower_caps_.keys() if c in X_out.columns]
        cols_with_upper = [c for c in self.upper_caps_.keys() if c in X_out.columns]

        # Apply clipping using vectorized pandas operations
        if cols_with_lower:
            lower_series = pd.Series(self.lower_caps_)
            X_out[cols_with_lower] = X_out[cols_with_lower].clip(lower=lower_series, axis=1)

        if cols_with_upper:
            upper_series = pd.Series(self.upper_caps_)
            X_out[cols_with_upper] = X_out[cols_with_upper].clip(upper=upper_series, axis=1)

        return X_out

    def export_state(self) -> Dict[str, Any]:
        """
        Extracts the learned cutoff values into a JSON-serializable dictionary.
        
        Returns:
            Dict[str, Any]: Serialized state dictionary.
        """
        if not getattr(self, "is_fitted_", False):
            raise ValueError(f"[{self.__class__.__name__}] must be fitted before exporting state.")

        return {
            "lower_caps_": self.lower_caps_,
            "upper_caps_": self.upper_caps_,
            "hyperparams": {
                "quantiles": self.quantiles,
                "features": self.features,
            },
        }

    @classmethod
    def load_state(cls, state: Dict[str, Any]) -> "OutlierCapper":
        """
        Instantiates a pre-fitted OutlierCapper directly from serialized state.
        
        Args:
            state (Dict[str, Any]): Serialized state dictionary.
            
        Returns:
            OutlierCapper: Restored, ready-to-transform object.
        """
        hyperparams = state.get("hyperparams", {})
        
        # Tuple conversion for safety if it was serialized as a list in JSON
        raw_quantiles = hyperparams.get("quantiles", (0.01, 0.99))
        quantiles = tuple(raw_quantiles) if isinstance(raw_quantiles, list) else raw_quantiles

        instance = cls(
            quantiles=quantiles,
            features=hyperparams.get("features", None),
        )
        
        instance.lower_caps_ = state.get("lower_caps_", {})
        instance.upper_caps_ = state.get("upper_caps_", {})
        instance.is_fitted_ = True
        
        return instance