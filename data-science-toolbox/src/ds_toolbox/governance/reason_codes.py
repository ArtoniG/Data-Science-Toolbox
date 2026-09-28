"""
src/credit_toolbox/governance/reason_codes.py

Translates quantitative model feature contributions (like SHAP values or linear WOE impacts)
into compliant, human-readable adverse action codes. 

Optimized for high-throughput batch processing using vectorized NumPy operations,
ensuring strict regulatory compliance for credit decisions.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Phase 1: Core Exceptions
from credit_toolbox.core.exceptions import CreditToolboxError

logger = logging.getLogger(__name__)


class ReasonCodeError(CreditToolboxError):
    """Raised when there is an issue generating adverse action reason codes."""
    pass


class ReasonCodeExtractor:
    """
    Deterministically extracts the top N penalizing features for each credit application 
    and maps them to compliant regulatory reason codes.
    """

    def __init__(
        self, 
        feature_to_code_map: Dict[str, str], 
        code_to_description_map: Dict[str, str],
        strict_mapping: bool = True
    ):
        """
        Args:
            feature_to_code_map: Maps internal feature names to regulatory code IDs (e.g., {'utilization_rate': 'RC001'}).
            code_to_description_map: Maps code IDs to human-readable text (e.g., {'RC001': 'Proportion of balances to credit limits is too high.'}).
            strict_mapping: If True, raises an error if a feature is missing from the mapping. If False, uses the raw feature name as a fallback.
        """
        self.feature_to_code_map = feature_to_code_map
        self.code_to_description_map = code_to_description_map
        self.strict_mapping = strict_mapping

    def get_top_reasons_from_shap(
        self, 
        shap_values: pd.DataFrame, 
        top_n: int = 4, 
        risk_direction: str = 'positive'
    ) -> pd.DataFrame:
        """
        Extracts reason codes from a batch of SHAP values using vectorized NumPy operations.
        
        Args:
            shap_values (pd.DataFrame): The SHAP value matrix (Applicants x Features).
            top_n (int): Maximum number of reason codes to extract per applicant.
            risk_direction (str): 'positive' if higher SHAP means higher default risk (standard), 
                                  'negative' if lower SHAP means higher default risk.
                                  
        Returns:
            pd.DataFrame: A DataFrame with columns [Reason_1_Code, Reason_1_Desc, Reason_2_Code, ...]
        """
        if shap_values.empty:
            raise ReasonCodeError("Input SHAP values DataFrame is empty.")

        logger.info(f"Extracting top {top_n} reason codes for {len(shap_values)} applicants...")

        # 1. Align direction (we want to sort descending, so the biggest risk drivers are first)
        matrix = shap_values.values
        if risk_direction == 'negative':
            matrix = -matrix

        # 2. Vectorized sorting of feature contributions
        # argsort sorts ascending, so we negate the matrix to get descending order efficiently
        sorted_indices = np.argsort(-matrix, axis=1)[:, :top_n]
        
        # 3. Extract the actual feature names based on the sorted indices
        feature_names = shap_values.columns.values
        top_features = feature_names[sorted_indices]

        # 4. Map features to codes and descriptions
        return self._build_results_dataframe(top_features, shap_values.index, top_n)

    def _build_results_dataframe(
        self, 
        top_features_matrix: np.ndarray, 
        index: pd.Index, 
        top_n: int
    ) -> pd.DataFrame:
        """
        Constructs the final formatted DataFrame containing mapped codes and descriptions.
        """
        results_dict = {}
        
        for i in range(top_n):
            # Extract the i-th column from the top features matrix
            feature_col = top_features_matrix[:, i]
            
            # Map feature names to Reason Code IDs
            codes = [self._safe_map_feature(f) for f in feature_col]
            
            # Map Reason Code IDs to Descriptions
            descriptions = [self._safe_map_description(c) for c in codes]
            
            results_dict[f'Reason_{i+1}_Code'] = codes
            results_dict[f'Reason_{i+1}_Desc'] = descriptions
            
        return pd.DataFrame(results_dict, index=index)

    def _safe_map_feature(self, feature: str) -> str:
        """Maps a feature to its code, respecting the strict_mapping flag."""
        if feature in self.feature_to_code_map:
            return self.feature_to_code_map[feature]
        
        if self.strict_mapping:
            raise ReasonCodeError(
                f"Feature '{feature}' is missing from the feature_to_code_map. "
                "Update the governance dictionary or disable strict_mapping."
            )
        return feature # Fallback to raw feature name

    def _safe_map_description(self, code: str) -> str:
        """Maps a code to its description, respecting the strict_mapping flag."""
        if code in self.code_to_description_map:
            return self.code_to_description_map[code]
            
        if self.strict_mapping:
            raise ReasonCodeError(
                f"Code '{code}' is missing from the code_to_description_map."
            )
        return "Description not configured."