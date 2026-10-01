"""
src/ds_toolbox/__init__.py

Public API definition for the ds-toolbox.
This file enforces a clean Developer Experience (DX) by exposing only the fully 
validated, production-ready modules at the root package level. 

Internal utility functions and base classes remain hidden from the end-user namespace.
"""

__version__ = "0.1.0"

# Phase 1: Core Exceptions
from ds_toolbox.core.exceptions import (
    DSToolboxError,
    DataValidationError,
    ModelDriftError,
    QualityGateError,
)

# Phase 3: Mathematical Metrics
from ds_toolbox.metrics.supervised import calculate_gini, calculate_ks, calculate_iv
from ds_toolbox.metrics.stability import calculate_psi

# Phase 4: Stateful Transformers
from ds_toolbox.preprocessing.missing_imputer import MissingImputer
from ds_toolbox.preprocessing.outlier_capper import OutlierCapper
from ds_toolbox.preprocessing.woe_encoder import WOEEncoder

# Phase 6: Governance & Auditability
from ds_toolbox.governance.model_card import ArtifactExporter
from ds_toolbox.governance.quality_gates import PipelineAuditor
from ds_toolbox.governance.reason_codes import ReasonCodeExtractor

# Restrict the public import wildcard (*) to these explicitly vetted tools
__all__ = [
    # Metadata
    "__version__",
    
    # Exceptions
    "DSToolboxError",
    "DataValidationError",
    "ModelDriftError",
    "QualityGateError",
    
    # Metrics
    "calculate_gini",
    "calculate_ks",
    "calculate_iv",
    "calculate_psi",
    
    # Preprocessing
    "MissingImputer",
    "OutlierCapper",
    "WOEEncoder",
    
    # Governance
    "ArtifactExporter",
    "PipelineAuditor",
    "ReasonCodeExtractor",
]