"""Logistic regression implementation for enterprise classification workflows.

This module provides a production-ready logistic regression trainer and evaluator
built around scikit-learn. It is designed for high-stakes, regulated-data science
pipelines, where transparency, reproducibility, robustness, and calibration are
non-negotiable.

The implementation follows the standard statistical foundations of logistic
regression (Cox, 1958; Nelder & Wedderburn, 1972) and the recommended
enterprise practices around regularization and model governance documented in
Mitchell et al. (2019), Sculley et al. (2015), and the broader modern MLops
operational guidance for production-quality ML systems.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ds_toolbox.core.exceptions import DataValidationError


class BaseToolboxModel(BaseModel):
    """Shared Pydantic model configuration for toolbox components."""

    model_config = ConfigDict(strict=True, validate_assignment=True)


class LogisticRegressionConfig(BaseToolboxModel):
    """Configuration contract for a binary logistic regression trainer.

    Args:
        penalty (str): Regularization type. Supported values are "l1", "l2",
            and "elasticnet".
        C (float): Inverse regularization strength. Lower values imply stronger
            regularization.
        solver (str): Solver to use for optimization.
        max_iter (int): Maximum number of training iterations.
        random_state (int): Random seed used by the underlying estimator.
        class_weight (str | None): Optional class balancing strategy.
        threshold (float): Probability cutoff used for binary predictions.
    """

    penalty: str = Field(default="l2", pattern="^(l1|l2|elasticnet)$")
    C: float = Field(default=1.0, gt=0.0)
    solver: str = Field(default="lbfgs", pattern="^(lbfgs|liblinear|newton-cg|saga|sag)$")
    max_iter: int = Field(default=200, gt=0)
    random_state: int = Field(default=42)
    class_weight: str | None = Field(default=None)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("penalty")
    @classmethod
    def validate_penalty_for_solver(cls, value: str, info: Any) -> str:
        """Guard against incompatible solver/penalty combinations."""
        solver = info.data.get("solver")
        if value == "elasticnet" and solver != "saga":
            raise ValueError("'elasticnet' penalty requires the 'saga' solver.")
        return value


class ClassificationMetrics(BaseToolboxModel):
    """Structured metrics returned by the logistic regression trainer.

    This is intentionally deterministic and JSON-friendly so it can be logged,
    published to monitoring systems, or embedded in model cards.
    """

    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    roc_auc: float = Field(..., ge=0.0, le=1.0)
    average_precision: float = Field(..., ge=0.0, le=1.0)
    brier_score: float = Field(..., ge=0.0, le=1.0)


class LogisticRegressionTrainer(BaseEstimator, ClassifierMixin):
    """Production-ready wrapper around scikit-learn logistic regression.

    The trainer accepts pandas DataFrames or NumPy arrays, performs defensive
    validation, handles missing numeric values, and exposes metrics for
    calibration and discrimination. The implementation is designed to be safe in
    enterprise-grade pipelines requiring reproducibility, interpretability, and
    operational robustness.

    References:
        Cox, D. R. (1958). The regression analysis of binary sequences.
        Nelder, J. A., & Wedderburn, R. W. M. (1972). Generalized Linear Models.
        Mitchell et al. (2019). Model Cards for Model Reporting.
    """

    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 200,
        random_state: int = 42,
        class_weight: str | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.class_weight = class_weight
        self.threshold = threshold

    def _validate_features(self, X: Any) -> pd.DataFrame:
        """Normalize input features into a pandas DataFrame."""
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        elif isinstance(X, np.ndarray):
            frame = pd.DataFrame(X)
        else:
            msg = "Input features must be a pandas DataFrame or a NumPy array."
            raise DataValidationError(msg)

        if frame.empty:
            raise DataValidationError("Input feature matrix cannot be empty.")

        if frame.shape[1] == 0:
            raise DataValidationError("Input feature matrix must contain at least one feature.")

        if frame.isnull().all(axis=1).any():
            raise DataValidationError("At least one row is entirely null and cannot be processed.")

        return frame

    def _validate_target(self, y: Any) -> pd.Series:
        """Validate and coerce the binary target to a stable pandas Series."""
        if y is None:
            raise DataValidationError("A binary target vector is required for training.")

        if isinstance(y, pd.Series):
            target = y.copy()
        elif isinstance(y, np.ndarray):
            target = pd.Series(y)
        else:
            target = pd.Series(y)

        if target.empty:
            raise DataValidationError("Target vector cannot be empty.")

        unique_values = set(pd.unique(target))
        if len(unique_values) != 2:
            raise DataValidationError(
                "Logistic regression requires exactly two target classes. "
                f"Found classes: {sorted(unique_values)}"
            )

        target = target.astype(int)
        return target.reset_index(drop=True)

    def _prepare_features(self, X: Any) -> tuple[pd.DataFrame, list[str]]:
        """Convert mixed data types to model-ready numeric data."""
        frame = self._validate_features(X)

        for col in frame.columns:
            if pd.api.types.is_object_dtype(frame[col]) or pd.api.types.is_categorical_dtype(frame[col]):
                frame[col] = frame[col].astype(str)

        features = pd.get_dummies(frame, drop_first=True)

        if features.empty:
            raise DataValidationError("No valid feature columns remain after preprocessing.")

        if features.isnull().all().any():
            raise DataValidationError("At least one transformed feature column is entirely null.")

        imputer = SimpleImputer(strategy="median")
        matrix = imputer.fit_transform(features)

        self.imputer_ = imputer
        self.feature_names_ = list(features.columns)
        return pd.DataFrame(matrix, columns=self.feature_names_), self.feature_names_

    def fit(self, X: Any, y: Any) -> "LogisticRegressionTrainer":
        """Fit the logistic regression estimator on a binary target.

        Args:
            X: Feature matrix or DataFrame with numeric or categorical columns.
            y: Binary target vector of length equal to the number of rows in X.

        Returns:
            LogisticRegressionTrainer: Fitted estimator.
        """

        features = self._validate_features(X)
        target = self._validate_target(y)

        if len(features) != len(target):
            raise DataValidationError(
                "Feature matrix and target vector must have the same number of rows. "
                f"Got {len(features)} rows and {len(target)} target rows."
            )

        processed, feature_names = self._prepare_features(features)
        processed.index = target.index

        estimator = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight=self.class_weight,
        )

        estimator.fit(processed, target)

        self.estimator_ = estimator
        self.feature_names_in_ = feature_names
        self.classes_ = estimator.classes_
        self.n_features_in_ = processed.shape[1]
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return probability estimates for the positive class."""
        if not getattr(self, "is_fitted_", False):
            raise DataValidationError("The model must be fitted before prediction is possible.")

        processed, _ = self._prepare_features(X)
        probabilities = self.estimator_.predict_proba(processed)
        return probabilities[:, 1]

    def predict(self, X: Any) -> np.ndarray:
        """Return hard class predictions using the configured decision threshold."""
        probabilities = self.predict_proba(X)
        return (probabilities >= self.threshold).astype(int)

    def evaluate(self, X: Any, y: Any) -> ClassificationMetrics:
        """Compute robust classification metrics for the trained model."""
        if not getattr(self, "is_fitted_", False):
            raise DataValidationError("The model must be fitted before evaluation.")

        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        target = self._validate_target(y)

        if len(predictions) != len(target):
            raise DataValidationError("Prediction vector and target have mismatched lengths.")

        metrics = ClassificationMetrics(
            accuracy=float(accuracy_score(target, predictions)),
            precision=float(precision_score(target, predictions, zero_division=0)),
            recall=float(recall_score(target, predictions, zero_division=0)),
            f1_score=float(f1_score(target, predictions, zero_division=0)),
            roc_auc=float(roc_auc_score(target, probabilities)),
            average_precision=float(average_precision_score(target, probabilities)),
            brier_score=float(brier_score_loss(target, probabilities)),
        )
        return metrics

    def export_state(self) -> dict[str, Any]:
        """Serialize fitted state in a deterministic, JSON-friendly dictionary."""
        if not getattr(self, "is_fitted_", False):
            raise DataValidationError("The model must be fitted before exporting state.")

        return {
            "model_type": "logistic_regression",
            "hyperparameters": {
                "penalty": self.penalty,
                "C": self.C,
                "solver": self.solver,
                "max_iter": self.max_iter,
                "random_state": self.random_state,
                "class_weight": self.class_weight,
                "threshold": self.threshold,
            },
            "feature_names": self.feature_names_,
            "classes": self.classes_.tolist(),
            "coef": self.estimator_.coef_.tolist(),
            "intercept": self.estimator_.intercept_.tolist(),
            "n_features_in_": self.n_features_in_,
        }

    @classmethod
    def load_state(cls, state: dict[str, Any]) -> "LogisticRegressionTrainer":
        """Reconstruct a fitted trainer from a serialized state dictionary."""
        trainer = cls(
            penalty=state["hyperparameters"].get("penalty", "l2"),
            C=state["hyperparameters"].get("C", 1.0),
            solver=state["hyperparameters"].get("solver", "lbfgs"),
            max_iter=state["hyperparameters"].get("max_iter", 200),
            random_state=state["hyperparameters"].get("random_state", 42),
            class_weight=state["hyperparameters"].get("class_weight"),
            threshold=state["hyperparameters"].get("threshold", 0.5),
        )

        trainer.feature_names_ = state.get("feature_names", [])
        trainer.classes_ = np.array(state.get("classes", [0, 1]))
        trainer.n_features_in_ = state.get("n_features_in_", 0)
        trainer.estimator_ = LogisticRegression(
            penalty=trainer.penalty,
            C=trainer.C,
            solver=trainer.solver,
            max_iter=trainer.max_iter,
            random_state=trainer.random_state,
            class_weight=trainer.class_weight,
        )
        trainer.estimator_.coef_ = np.array(state.get("coef", [[0.0]]))
        trainer.estimator_.intercept_ = np.array(state.get("intercept", [0.0]))
        trainer.estimator_.classes_ = trainer.classes_
        trainer.is_fitted_ = True
        return trainer

