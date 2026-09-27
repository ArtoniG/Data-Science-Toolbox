"""
src/credit_toolbox/transformers/woe_encoder.py

Weight of Evidence (WOE) and Information Value (IV) encoder for credit risk modeling.
Transforms discrete bins or categorical variables into continuous risk weights based on 
log-odds calculations, establishing a linear relationship with the logit of default probability.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from credit_toolbox.core.base import StatefulCreditTransformer


class WoeEncoder(StatefulCreditTransformer):
    """
    Computes Weight of Evidence mapping for categorical features or binned numerics.
    
    Guarantees:
    1. Laplace smoothing (regularization) to prevent divide-by-zero or log(0) on sparse bins.
    2. Automatic Information Value (IV) calculation stored in model state for feature selection.
    3. JSON-safe serialization by enforcing string-type mapping keys.
    4. Safe handling of unseen categories during inference (mapping to neutral WOE = 0.0).
    """

    def __init__(
        self,
        regularization: float = 0.001,
        features: Optional[List[str]] = None,
    ):
        """
        Args:
            regularization (float): Small constant added to bin counts to prevent zero-frequency problems.
                Default is 0.001.
            features (Optional[List[str]]): Specific features to encode. If None, encodes all categorical/object columns.
        """
        self.regularization = regularization
        self.features = features

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoeEncoder":
        """
        Calculates the WOE for each category/bin and the overall Information Value (IV).
        
        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Binary default target (0 = Good/Non-Default, 1 = Bad/Default).
            
        Returns:
            WoeEncoder: Fitted transformer instance.
        """
        X = self._validate_dataframe(X)
        if y is None:
            raise ValueError(f"[{self.__class__.__name__}] is a supervised transformer and requires a target array 'y'.")

        if self.features is None:
            # Default to encoding all columns if no specific features are passed,
            # assuming continuous variables were binned upstream
            cols_to_fit = X.columns.tolist()
        else:
            cols_to_fit = [c for c in self.features if c in X.columns]

        self.woe_mapping_: Dict[str, Dict[str, float]] = {}
        self.iv_: Dict[str, float] = {}

        # Global event counts
        total_bads = y.sum()
        total_goods = y.count() - total_bads

        for col in cols_to_fit:
            # Cast to string to ensure JSON serialization compatibility later
            x_series = X[col].astype(str)
            
            # Combine into a temporary dataframe for fast grouped aggregation
            temp_df = pd.DataFrame({"x": x_series, "y": y})
            grouped = temp_df.groupby("x")["y"].agg(["count", "sum"])
            
            bads = grouped["sum"]
            goods = grouped["count"] - bads

            # Calculate distributions with Laplace smoothing
            dist_bads = (bads + self.regularization) / (total_bads + 2 * self.regularization)
            dist_goods = (goods + self.regularization) / (total_goods + 2 * self.regularization)

            # Weight of Evidence: ln(Distribution of Goods / Distribution of Bads)
            woe = np.log(dist_goods / dist_bads)
            
            # Information Value (IV) for the column
            iv = (dist_goods - dist_bads) * woe
            
            # Store mapping (using native python floats for JSON)
            self.woe_mapping_[col] = {k: float(v) for k, v in woe.items()}
            self.iv_[col] = float(iv.sum())

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces categorical/binned values with their calculated WOE mappings.
        
        Args:
            X (pd.DataFrame): Data to transform.
            
        Returns:
            pd.DataFrame: DataFrame with WOE-encoded features.
        """
        X = self._validate_dataframe(X)
        if not getattr(self, "is_fitted_", False):
            raise ValueError(f"[{self.__class__.__name__}] must be fitted before calling transform().")

        X_out = X.copy()
        
        for col, mapping in self.woe_mapping_.items():
            if col in X_out.columns:
                # Cast to string to match fit-time keys, map, and fill unseen with 0.0 (neutral risk)
                X_out[col] = X_out[col].astype(str).map(mapping).fillna(0.0)

        return X_out

    def export_state(self) -> Dict[str, Any]:
        """
        Extracts the WOE mappings and IV statistics into a JSON-serializable dictionary.
        
        Returns:
            Dict[str, Any]: Serialized state dictionary.
        """
        if not getattr(self, "is_fitted_", False):
            raise ValueError(f"[{self.__class__.__name__}] must be fitted before exporting state.")

        return {
            "woe_mapping_": self.woe_mapping_,
            "iv_": self.iv_,
            "hyperparams": {
                "regularization": self.regularization,
                "features": self.features,
            },
        }

    @classmethod
    def load_state(cls, state: Dict[str, Any]) -> "WoeEncoder":
        """
        Instantiates a pre-fitted WoeEncoder directly from serialized state.
        
        Args:
            state (Dict[str, Any]): Serialized state dictionary.
            
        Returns:
            WoeEncoder: Restored, ready-to-transform object.
        """
        hyperparams = state.get("hyperparams", {})
        instance = cls(
            regularization=hyperparams.get("regularization", 0.001),
            features=hyperparams.get("features", None),
        )
        
        instance.woe_mapping_ = state.get("woe_mapping_", {})
        instance.iv_ = state.get("iv_", {})
        instance.is_fitted_ = True
        
        return instance