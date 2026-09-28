"""
src/credit_toolbox/governance/quality_gates.py

Implements deterministic Quality Gates for Scikit-Learn Pipelines.
Acts as an automated risk committee that intercepts the pipeline before serialization 
or documentation, ensuring enterprise standards (imputation, capping, monotonicity) 
are strictly met.
"""

from typing import Any, Dict, List, Optional
import logging

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

# Phase 1: Core Exceptions
from credit_toolbox.core.exceptions import QualityGateError

# Phase 3: Metrics
from credit_toolbox.metrics.supervised import evaluate_monotonicity

# Phase 4: Transformers (Used for structural validation)
# Assuming these were built in Phase 4 per the architecture blueprint
from credit_toolbox.transformers.missing_imputer import MissingImputer
from credit_toolbox.transformers.outlier_capper import OutlierCapper
from credit_toolbox.transformers.woe_encoder import WOEEncoder

logger = logging.getLogger(__name__)


class PipelineAuditor:
    """
    Automated pipeline validator that enforces structural and mathematical 
    quality gates on credit risk models prior to production deployment.
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Args:
            strict_mode (bool): If True, fails the pipeline immediately upon the first 
                                breached gate. If False, aggregates warnings.
        """
        self.strict_mode = strict_mode
        self.audit_results: Dict[str, Any] = {}

    def audit(self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> bool:
        """
        Executes all quality gates against the provided pipeline.
        
        Args:
            pipeline (Pipeline): The fitted Scikit-Learn pipeline.
            X_train (pd.DataFrame): The training features used to fit the pipeline.
            y_train (pd.Series): The binary ground truth for the training set.
            
        Returns:
            bool: True if all gates pass. Raises QualityGateError if strict_mode is True and a gate fails.
        """
        logger.info("Initiating Pipeline Quality Gates Audit...")
        
        try:
            self._check_is_fitted(pipeline)
            self._check_structural_components(pipeline)
            self._check_monotonicity(pipeline, X_train, y_train)
            
            logger.info("Pipeline successfully passed all Quality Gates.")
            return True
            
        except QualityGateError as e:
            logger.error(f"Quality Gate Failed: {str(e)}")
            if self.strict_mode:
                raise
            return False

    def _check_is_fitted(self, pipeline: Pipeline) -> None:
        """
        Information Leakage / State Check: Ensures the pipeline has been fitted.
        Unfitted pipelines cannot be serialized as production artifacts.
        """
        try:
            check_is_fitted(pipeline)
            self.audit_results["is_fitted"] = "PASS"
        except NotFittedError:
            self.audit_results["is_fitted"] = "FAIL"
            raise QualityGateError(
                "Pipeline is not fitted. State cannot be extracted for reproducibility. "
                "Ensure the pipeline is fitted on the training fold without data leakage."
            )

    def _check_structural_components(self, pipeline: Pipeline) -> None:
        """
        Structural Checks: Verifies the presence of mandatory enterprise transformations.
        """
        step_instances = [type(step).__name__ for _, step in pipeline.steps]
        
        # 1. Imputation Check
        has_imputer = any(isinstance(step, MissingImputer) for _, step in pipeline.steps)
        if not has_imputer:
            raise QualityGateError(
                "Imputation Check Failed: Pipeline lacks a Phase 4 MissingImputer. "
                "Model will crash on missing bureau data."
            )
            
        # 2. Outlier Check
        has_capper = any(isinstance(step, OutlierCapper) for _, step in pipeline.steps)
        if not has_capper:
            raise QualityGateError(
                "Outlier Check Failed: Pipeline lacks a Phase 4 OutlierCapper. "
                "Numerical features are exposed to extreme value instability."
            )
            
        self.audit_results["structural_checks"] = "PASS"

    def _check_monotonicity(self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Monotonicity Check: Ensures WOE encoded features maintain a strictly monotonic 
        relationship with the default rate (y_train).
        """
        # Isolate the WOE Encoder step
        woe_step_name, woe_encoder = None, None
        for name, step in pipeline.steps:
            if isinstance(step, WOEEncoder):
                woe_step_name, woe_encoder = name, step
                break
                
        if not woe_encoder:
            logger.warning("No WOEEncoder found in pipeline. Skipping monotonicity check.")
            self.audit_results["monotonicity"] = "SKIPPED"
            return

        # Transform data up to the WOE step to evaluate the bins
        # We slice the pipeline up to and including the WOE encoder
        woe_index = list(pipeline.named_steps.keys()).index(woe_step_name)
        pipeline_up_to_woe = Pipeline(pipeline.steps[:woe_index + 1])
        
        X_encoded = pipeline_up_to_woe.transform(X_train)
        
        # If the output is a numpy array (standard sklearn behavior), convert back to DataFrame
        if isinstance(X_encoded, np.ndarray):
            # Try to get feature names if the encoder supports it
            try:
                cols = pipeline_up_to_woe.get_feature_names_out()
            except AttributeError:
                cols = X_train.columns
            X_encoded = pd.DataFrame(X_encoded, columns=cols, index=X_train.index)

        # Evaluate monotonicity for every encoded feature
        failed_features = []
        for feature in X_encoded.columns:
            metrics = evaluate_monotonicity(X_encoded[feature], y_train)
            if not metrics.get("is_strictly_monotonic", False):
                failed_features.append(feature)

        if failed_features:
            self.audit_results["monotonicity"] = "FAIL"
            raise QualityGateError(
                f"Monotonicity Check Failed: The following features lack strictly monotonic "
                f"WOE binning with the target variable: {', '.join(failed_features)}. "
                f"Model rejected due to regulatory compliance risk."
            )
            
        self.audit_results["monotonicity"] = "PASS"