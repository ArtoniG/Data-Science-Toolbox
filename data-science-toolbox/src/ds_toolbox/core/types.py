"""
src/credit_toolbox/core/types.py

Core type aliases and deterministic Pydantic schemas.
By strictly defining inputs and outputs here, we ensure that every metric and 
transformer in the toolbox is fully type-hinted and returns predictable, 
JSON-serializable objects for APIs and audit logs.
"""

from enum import Enum
from typing import Any, Dict, List, TypeAlias, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

# ------------------------------------------------------------------------------
# TYPE ALIASES (For clean type hinting in function signatures)
# ------------------------------------------------------------------------------

# Represents a 1-dimensional array (target vector, single feature, or probabilities)
ArrayLike: TypeAlias = Union[np.ndarray, pd.Series, List[float], List[int]]

# Represents a 2-dimensional dataset (feature matrix)
MatrixLike: TypeAlias = Union[np.ndarray, pd.DataFrame]


# ------------------------------------------------------------------------------
# ENUMS (For strict parameter control)
# ------------------------------------------------------------------------------

class FeatureType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"


class DriftThresholds(float, Enum):
    """
    Standard industry thresholds for Population Stability Index (PSI).
    """
    SAFE = 0.10          # Below 0.10: No significant drift
    WARNING = 0.25       # 0.10 to 0.25: Moderate drift, monitor closely
    CRITICAL = 0.25      # Above 0.25: Severe drift, model requires retraining


# ------------------------------------------------------------------------------
# PYDANTIC SCHEMAS (Deterministic Outputs for Metrics)
# ------------------------------------------------------------------------------
# Returning structured Pydantic models instead of raw floats or tuples ensures 
# that FastAPIs can instantly serialize the results, and audit logs capture 
# named key-value pairs without ambiguity.

class BaseToolboxModel(BaseModel):
    """Base Pydantic model with strict configuration."""
    model_config = ConfigDict(
        strict=True,
        validate_assignment=True,
        arbitrary_types_allowed=True # Allows Pandas/Numpy objects if absolutely necessary
    )


class KSResult(BaseToolboxModel):
    """Output schema for Kolmogorov-Smirnov calculations."""
    ks_stat: float = Field(..., ge=0.0, le=1.0, description="The maximum divergence between CDFs")
    p_value: float = Field(..., ge=0.0, le=1.0, description="Statistical significance of the KS stat")
    score_at_ks: float | None = Field(default=None, description="The specific score/probability where max divergence occurs")


class IVResult(BaseToolboxModel):
    """Output schema for Information Value and Weight of Evidence calculations."""
    total_iv: float = Field(..., ge=0.0, description="Total predictive power of the feature")
    predictive_strength: str = Field(..., description="Categorical label (e.g., 'Weak', 'Strong', 'Suspicious')")
    woe_mapping: Dict[str, float] = Field(..., description="Mapping of bins to their computed WOE values")


class PSIResult(BaseToolboxModel):
    """Output schema for Population Stability Index calculations."""
    total_psi: float = Field(..., ge=0.0, description="Total distribution drift score")
    is_drifting: bool = Field(..., description="True if total_psi exceeds the configured threshold")
    bucket_details: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Detailed expected vs actual percentages per bin"
    )


class ModelCardMetadata(BaseToolboxModel):
    """Standardized metadata schema for automated regulatory documentation."""
    model_name: str = Field(..., description="Formal name of the risk model")
    version: str = Field(..., description="Semantic versioning (e.g., 1.0.0)")
    developer: str = Field(..., description="Name or team responsible for the model")
    contains_pii: bool = Field(default=False, description="Flag indicating if training data used PII")
    algorithm: str = Field(..., description="e.g., 'LightGBM', 'Logistic Regression'")