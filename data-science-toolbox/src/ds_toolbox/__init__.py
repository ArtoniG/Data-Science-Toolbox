"""
src/ds_toolbox/__init__.py

Public package API for ds_toolbox.
This module intentionally exposes a minimal, reliable root namespace that avoids
legacy import mismatches and keeps the package usable in production code.
"""

__version__ = "0.1.0"

from ds_toolbox.core.exceptions import (
    BinningError,
    DataValidationError,
    DSToolboxError,
    GenAIGuardrailError,
    MetricCalculationError,
    ModelDriftError,
    SchemaMismatchError,
)
from ds_toolbox.models.logistic_regression import (
    ClassificationMetrics,
    LogisticRegressionConfig,
    LogisticRegressionTrainer,
)

__all__ = [
    "__version__",
    "DSToolboxError",
    "DataValidationError",
    "SchemaMismatchError",
    "BinningError",
    "MetricCalculationError",
    "ModelDriftError",
    "GenAIGuardrailError",
    "LogisticRegressionConfig",
    "LogisticRegressionTrainer",
    "ClassificationMetrics",
]
