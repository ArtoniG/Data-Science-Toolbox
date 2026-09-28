"""
tests/test_metrics/test_supervised.py

Unit tests for supervised credit metrics: Kolmogorov-Smirnov (KS), 
Gini Coefficient, and Information Value (IV).
"""

import numpy as np
import pandas as pd
import pytest

from credit_toolbox.core.exceptions import DataValidationError
from credit_toolbox.metrics.supervised import (
    calculate_gini,
    calculate_iv,
    calculate_ks,
)


# ==============================================================================
# Kolmogorov-Smirnov (KS) Tests
# ==============================================================================

def test_ks_perfect_separation():
    """KS must equal 1.0 (or 100%) when scores perfectly separate defaults from non-defaults."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.2, 0.3, 0.7, 0.8, 0.85, 0.9])
    
    ks_stat, _ = calculate_ks(y_true, y_prob)
    assert pytest.approx(ks_stat, abs=1e-4) == 1.0


def test_ks_random_predictions():
    """KS must approach 0.0 when predictions have zero discriminative power."""
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.5, size=1000)
    y_prob = np.random.uniform(0, 1, size=1000)
    
    ks_stat, _ = calculate_ks(y_true, y_prob)
    assert ks_stat < 0.10


def test_ks_mismatched_length_raises_error():
    """Validates input validation when y_true and y_prob lengths disagree."""
    y_true = np.array([0, 1, 0])
    y_prob = np.array([0.1, 0.8])
    
    with pytest.raises(DataValidationError, match="Length mismatch"):
        calculate_ks(y_true, y_prob)


# ==============================================================================
# Gini Coefficient Tests
# ==============================================================================

def test_gini_theoretical_bounds(sample_bureau_dataset):
    """Gini must fall strictly within the [-1.0, 1.0] interval."""
    X, y = sample_bureau_dataset
    # Synthetic probability proportional to revolving utilization
    y_prob = X["revolving_utilization"].values
    
    gini = calculate_gini(y, y_prob)
    assert -1.0 <= gini <= 1.0


def test_gini_perfect_vs_inverted():
    """Perfect model yields Gini = 1.0; completely inverted probabilities yield Gini = -1.0."""
    y_true = np.array([0, 0, 1, 1])
    y_prob_perfect = np.array([0.1, 0.2, 0.8, 0.9])
    y_prob_inverted = np.array([0.9, 0.8, 0.2, 0.1])
    
    assert pytest.approx(calculate_gini(y_true, y_prob_perfect), abs=1e-4) == 1.0
    assert pytest.approx(calculate_gini(y_true, y_prob_inverted), abs=1e-4) == -1.0


# ==============================================================================
# Information Value (IV) Tests
# ==============================================================================

def test_iv_monotonic_predictive_power():
    """Higher predictive feature must yield higher Information Value."""
    np.random.seed(42)
    y = pd.Series([0]*500 + [1]*500)
    
    # Feature 1: Strong predictive signal
    x_strong = pd.Series(np.concatenate([np.random.normal(0, 1, 500), np.random.normal(2, 1, 500)]))
    # Feature 2: Pure noise
    x_noise = pd.Series(np.random.normal(0, 1, 1000))
    
    iv_strong = calculate_iv(x_strong, y, bins=10)
    iv_noise = calculate_iv(x_noise, y, bins=10)
    
    assert iv_strong > iv_noise
    assert iv_noise < 0.05  # Uninformative (< 0.02 to 0.05 rule of thumb)


def test_iv_zero_division_guard():
    """Ensures Laplace smoothing or epsilon adjustment prevents division by zero in empty bins."""
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    
    # Should execute without throwing ZeroDivisionError or returning Inf/NaN
    iv_val = calculate_iv(x, y, bins=5)
    assert not np.isnan(iv_val)
    assert not np.isinf(iv_val)