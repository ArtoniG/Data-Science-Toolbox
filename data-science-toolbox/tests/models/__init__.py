"""Unit tests for the enterprise logistic regression trainer.

These tests verify that the first production-ready modelling component behaves
reliably on realistic binary classification scenarios, handles missing values,
and fails gracefully under invalid inputs.

References used by this test suite:
- Cox, D. R. (1958). The regression analysis of binary sequences.
- Niculescu-Mizil & Caruana (2005). Predicting Good Probabilities with
  Supervised Learning.
- Sculley et al. (2015). Hidden Technical Debt in Machine Learning Systems.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ds_toolbox.core.exceptions import DataValidationError
from ds_toolbox.models.logistic_regression import LogisticRegressionTrainer


@pytest.fixture
def binary_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Create a deterministic, well-separated binary classification dataset."""
    rng = np.random.default_rng(42)
    n = 600
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    latent = 1.2 * x1 - 1.0 * x2 + 0.5
    prob = 1 / (1 + np.exp(-latent))
    target = (rng.random(n) < prob).astype(int)

    features = pd.DataFrame({
        "feature_a": x1,
        "feature_b": x2,
        "feature_c": rng.normal(0, 1, n),
    })
    features.loc[::37, "feature_b"] = np.nan
    return features, pd.Series(target, name="target")


def test_logistic_regression_trainer_fits_and_scores(binary_dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    """The trainer should fit, predict, and produce discriminative metrics."""
    X, y = binary_dataset
    trainer = LogisticRegressionTrainer(C=1.0, max_iter=500, random_state=42)

    trainer.fit(X, y)
    probabilities = trainer.predict_proba(X)
    predictions = trainer.predict(X)

    assert probabilities.shape[0] == len(y)
    assert predictions.shape[0] == len(y)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert trainer.evaluate(X, y).roc_auc > 0.80


def test_logistic_regression_handles_missing_values() -> None:
    """Missing values should be handled safely without failing or producing NaNs."""
    X = pd.DataFrame({
        "age": [20, 30, np.nan, 50, 60, 45],
        "income": [1000, 1200, 1350, np.nan, 2200, 1800],
        "segment": ["A", "B", "A", "C", "B", np.nan],
    })
    y = pd.Series([0, 0, 1, 1, 1, 0])

    trainer = LogisticRegressionTrainer(random_state=42, max_iter=500)
    trainer.fit(X, y)

    probabilities = trainer.predict_proba(X)
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_logistic_regression_raises_for_single_class() -> None:
    """Training must reject targets without both classes."""
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [0.5, 1.5, 2.0, 2.5]})
    y = pd.Series([1, 1, 1, 1])

    trainer = LogisticRegressionTrainer()
    with pytest.raises(DataValidationError, match="exactly two target classes"):
        trainer.fit(X, y)


def test_logistic_regression_reproducible_state_export() -> None:
    """The fitted model should export a deterministic state dictionary."""
    X = pd.DataFrame({
        "x1": [0.1, 0.2, 0.4, 0.8, 1.3, 1.9],
        "x2": [1.2, 0.9, 2.1, 1.6, 2.4, 2.9],
    })
    y = pd.Series([0, 0, 1, 0, 1, 1])

    trainer = LogisticRegressionTrainer(random_state=42)
    trainer.fit(X, y)

    state = trainer.export_state()
    restored = LogisticRegressionTrainer.load_state(state)
    assert restored.is_fitted_
    assert restored.feature_names_ == trainer.feature_names_
    assert restored.threshold == trainer.threshold

