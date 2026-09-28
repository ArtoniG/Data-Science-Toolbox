"""
tests/test_metrics/test_stability.py

Unit tests for Population Stability Index (PSI) monitoring.
"""

import numpy as np
import pandas as pd
import pytest

from credit_toolbox.metrics.stability import calculate_psi


def test_psi_identical_distributions_is_zero():
    """Identical baseline and target distributions must produce PSI close to 0.0."""
    np.random.seed(42)
    base = np.random.normal(0, 1, size=5000)
    target = np.random.normal(0, 1, size=5000)
    
    psi_value = calculate_psi(base, target, bins=10)
    
    # Industry consensus: PSI < 0.1 indicates no significant distribution shift
    assert psi_value < 0.02


def test_psi_significant_shift():
    """Distribution shift (e.g., macroeconomic downturn or policy change) must trigger PSI > 0.25."""
    np.random.seed(42)
    base = np.random.normal(0, 1, size=2000)
    target = np.random.normal(1.5, 1.2, size=2000)  # Mean shift and variance increase
    
    psi_value = calculate_psi(base, target, bins=10)
    
    # Industry consensus: PSI > 0.25 indicates significant action required
    assert psi_value > 0.25


def test_psi_categorical_support():
    """Calculates PSI correctly for discrete categorical feature distributions."""
    base = pd.Series(["A"] * 500 + ["B"] * 300 + ["C"] * 200)
    target = pd.Series(["A"] * 200 + ["B"] * 300 + ["C"] * 500)  # Shift from A to C
    
    psi_value = calculate_psi(base, target, categorical=True)
    assert psi_value > 0.10


def test_psi_handles_unseen_categories_gracefully():
    """Target population containing new categories not in base must not crash execution."""
    base = pd.Series(["A", "A", "B", "B"])
    target = pd.Series(["A", "B", "C", "C"])  # 'C' is unseen in baseline
    
    psi_value = calculate_psi(base, target, categorical=True)
    assert not np.isnan(psi_value)
    assert psi_value > 0.0