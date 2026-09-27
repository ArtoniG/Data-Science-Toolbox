"""
src/credit_toolbox/core/base.py

Base classes and interfaces for the credit_toolbox package.
Enforces strict contracts for state serialization and DataFrame preservation,
which are mandatory for the Phase 6 Governance and Auditability requirements.
"""

import abc
from typing import Any, Dict

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class StatefulCreditTransformer(BaseEstimator, TransformerMixin, abc.ABC):
    """
    Abstract base class for all proprietary credit risk transformers.
    
    Extends Scikit-Learn's native API to enforce two enterprise requirements:
    1. Feature Name Preservation: Strict Pandas DataFrame validation.
    2. Auditability: Mandatory export/load state methods for declarative pipeline reconstruction.
    """

    def _validate_dataframe(self, X: Any) -> pd.DataFrame:
        """
        Ensures the input is a Pandas DataFrame.
        
        Why this matters: If a Scikit-Learn pipeline accidentally converts data into a 
        nameless NumPy array, we lose the feature names. Without feature names, 
        Phase 6 cannot generate human-readable Adverse Action / Reason Codes.
        
        Args:
            X (Any): The input data (expected pd.DataFrame).
            
        Returns:
            pd.DataFrame: The validated DataFrame.
            
        Raises:
            TypeError: If X is not a Pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"[{self.__class__.__name__}] requires a Pandas DataFrame. "
                f"Raw NumPy arrays lose feature names, which violates credit risk "
                f"governance and reason-code generation requirements. Got {type(X)}."
            )
        return X

    @abc.abstractmethod
    def export_state(self) -> Dict[str, Any]:
        """
        Extracts the fitted parameters (learned during .fit()) into a standard, 
        JSON-serializable Python dictionary.
        
        This is extracted by the PipelineAuditor in Phase 6 to generate the 
        `pipeline_state.json` artifact.
        
        Returns:
            Dict[str, Any]: A dictionary containing all fitted states.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def load_state(cls, state: Dict[str, Any], **hyperparams) -> 'StatefulCreditTransformer':
        """
        Factory method to instantiate a "pre-fitted" transformer directly from a state dictionary.
        
        This completely bypasses `.fit()`. It is used by `reproduce_model.py` to rebuild 
        the exact inference pipeline without needing the original training dataset.
        
        Args:
            state (Dict[str, Any]): The fitted state dictionary (e.g., from pipeline_state.json).
            **hyperparams: The initialization parameters (e.g., quantiles, bounds).
            
        Returns:
            StatefulCreditTransformer: An instantiated, ready-to-transform object.
        """
        pass