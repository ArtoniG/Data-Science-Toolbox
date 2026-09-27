"""
src/credit_toolbox/transformers/missing_imputer.py

Domain-specific missing value imputation for credit risk modeling.
Replaces statistical smoothing (mean/median) with distinct risk-isolation values 
(e.g., -9999 for numerics, 'Missing' for categoricals) to preserve 'Thin File' signals,
while guaranteeing state serialization for auditability.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from credit_toolbox.core.base import StatefulCreditTransformer


class CreditMissingImputer(StatefulCreditTransformer):
    """
    Imputes missing values using fixed out-of-distribution constants rather than 
    sample statistics. This preserves the predictive power of missingness 
    (e.g., lack of credit history) for downstream tree models or WOE binning.
    
    Guarantees:
    1. Vectorized fill operations for OOM prevention on large bureau files.
    2. Preservation of feature names and DataFrame structures.
    3. Full export/import state serialization for Phase 6 auditability.
    """

    def __init__(
        self,
        numeric_fill_value: Union[int, float] = -9999,
        categorical_fill_value: str = "Missing",
        features: Optional[List[str]] = None,
    ):
        """
        Args:
            numeric_fill_value (Union[int, float]): Constant to replace NaNs in numeric columns. Default is -9999.
            categorical_fill_value (str): Constant to replace NaNs in object/category columns. Default is 'Missing'.
            features (Optional[List[str]]): Specific features to impute. If None, imputes all columns.
        """
        self.numeric_fill_value = numeric_fill_value
        self.categorical_fill_value = categorical_fill_value
        self.features = features

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CreditMissingImputer":
        """
        Learns the datatypes of the target columns and maps them to their respective fill values.
        
        Args:
            X (pd.DataFrame): Training features.
            y (Optional[pd.Series]): Ignored, present for Scikit-Learn pipeline compatibility.
            
        Returns:
            CreditMissingImputer: Fitted transformer instance.
        """
        X = self._validate_dataframe(X)
        
        if self.features is None:
            cols_to_fit = X.columns.tolist()
        else:
            cols_to_fit = [c for c in self.features if c in X.columns]

        self.fill_values_: Dict[str, Any] = {}
        
        for col in cols_to_fit:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.fill_values_[col] = self.numeric_fill_value
            else:
                self.fill_values_[col] = self.categorical_fill_value

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the learned fill values vectorially.
        
        Args:
            X (pd.DataFrame): Data to transform.
            
        Returns:
            pd.DataFrame: Imputed DataFrame.
        """
        X = self._validate_dataframe(X)
        if not getattr(self, "is_fitted_", False):
            raise ValueError(f"[{self.__class__.__name__}] must be fitted before calling transform().")

        # Pandas fillna with a dictionary is highly optimized and only applies to columns 
        # that exist in both the DataFrame and the dictionary, silently ignoring missing keys.
        X_out = X.fillna(value=self.fill_values_)
        
        return X_out

    def export_state(self) -> Dict[str, Any]:
        """
        Extracts the explicit column-to-fill-value mapping.
        
        Returns:
            Dict[str, Any]: Serialized state dictionary.
        """
        if not getattr(self, "is_fitted_", False):
            raise ValueError(f"[{self.__class__.__name__}] must be fitted before exporting state.")

        return {
            "fill_values_": self.fill_values_,
            "hyperparams": {
                "numeric_fill_value": self.numeric_fill_value,
                "categorical_fill_value": self.categorical_fill_value,
                "features": self.features,
            },
        }

    @classmethod
    def load_state(cls, state: Dict[str, Any]) -> "CreditMissingImputer":
        """
        Instantiates a pre-fitted CreditMissingImputer directly from serialized state.
        
        Args:
            state (Dict[str, Any]): Serialized state dictionary.
            
        Returns:
            CreditMissingImputer: Restored, ready-to-transform object.
        """
        hyperparams = state.get("hyperparams", {})
        instance = cls(
            numeric_fill_value=hyperparams.get("numeric_fill_value", -9999),
            categorical_fill_value=hyperparams.get("categorical_fill_value", "Missing"),
            features=hyperparams.get("features", None),
        )
        
        instance.fill_values_ = state.get("fill_values_", {})
        instance.is_fitted_ = True
        
        return instance