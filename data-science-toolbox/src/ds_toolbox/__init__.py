"""
src/credit_toolbox/__init__.py

Public API definition for the custom-credit-toolbox.
This file enforces a clean Developer Experience (DX) by exposing only the fully 
validated, production-ready modules at the root package level. 

Internal utility functions and base classes remain hidden from the end-user namespace.
"""

__version__ = "0.1.0"

# Phase 1: Core Exceptions
from credit_toolbox.core.exceptions import (
    CreditToolboxError,
    DataValidationError,
    ModelDriftError,
    QualityGateError,
)

# Phase 3: Mathematical Metrics
from credit_toolbox.metrics.supervised import calculate_gini, calculate_ks, calculate_iv
from credit_toolbox.metrics.stability import calculate_psi

# Phase 4: Stateful Transformers
from credit_toolbox.transformers.missing_imputer import MissingImputer
from credit_toolbox.transformers.outlier_capper import OutlierCapper
from credit_toolbox.transformers.woe_encoder import WOEEncoder

# Phase 6: Governance & Auditability
from credit_toolbox.governance.model_card import ArtifactExporter
from credit_toolbox.governance.quality_gates import PipelineAuditor
from credit_toolbox.governance.reason_codes import ReasonCodeExtractor

# Restrict the public import wildcard (*) to these explicitly vetted tools
__all__ = [
    # Metadata
    "__version__",
    
    # Exceptions
    "CreditToolboxError",
    "DataValidationError",
    "ModelDriftError",
    "QualityGateError",
    
    # Metrics
    "calculate_gini",
    "calculate_ks",
    "calculate_iv",
    "calculate_psi",
    
    # Transformers
    "MissingImputer",
    "OutlierCapper",
    "WOEEncoder",
    
    # Governance
    "ArtifactExporter",
    "PipelineAuditor",
    "ReasonCodeExtractor",
]