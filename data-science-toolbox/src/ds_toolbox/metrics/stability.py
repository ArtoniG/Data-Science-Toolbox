"""
src/credit_toolbox/metrics/stability.py

Calculates distribution drift metrics for models and features in production.
Implements the Population Stability Index (PSI) for continuous model scores and the 
Characteristic Stability Index (CSI) for categorical features. 
Built with strict boundaries (-inf to +inf) to prevent unhandled out-of-bounds 
data from crashing production monitoring jobs.
"""

import numpy as np
import pandas as pd

from credit_toolbox.core.exceptions import MetricCalculationError
from credit_toolbox.core.types import ArrayLike, DriftThresholds, PSIResult
from credit_toolbox.logging.decorators import log_execution_time


def _compute_psi_core(
    expected_counts: pd.Series, 
    actual_counts: pd.Series, 
    threshold: float
) -> PSIResult:
    """
    Internal helper to compute the PSI math and format the Pydantic output.
    Formula: $PSI = \\sum ((\\% \\text{Actual} - \\% \\text{Expected}) \\times \\ln(\\% \\text{Actual} / \\% \\text{Expected}))$
    """
    # 1. Convert absolute counts to relative percentages
    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()

    # 2. Protect against zero division (Infinite PSI) using epsilon substitution
    epsilon = 1e-6
    expected_pct_safe = np.maximum(expected_pct, epsilon)
    actual_pct_safe = np.maximum(actual_pct, epsilon)

    # 3. Calculate PSI per bucket and total
    psi_array = (actual_pct_safe - expected_pct_safe) * np.log(actual_pct_safe / expected_pct_safe)
    total_psi = float(psi_array.sum())

    # 4. Construct detailed bucket payload for front-end heatmaps and audit logs
    bucket_details = []
    for bucket_name in expected_counts.index:
        bucket_details.append({
            "bucket": str(bucket_name),
            "expected_pct": float(expected_pct[bucket_name]),
            "actual_pct": float(actual_pct[bucket_name]),
            "psi": float(psi_array[bucket_name])
        })

    return PSIResult(
        total_psi=total_psi,
        is_drifting=(total_psi >= threshold),
        bucket_details=bucket_details
    )


@log_execution_time
def calculate_psi(
    expected: ArrayLike, 
    actual: ArrayLike, 
    num_buckets: int = 10,
    threshold: float = DriftThresholds.WARNING
) -> PSIResult:
    """
    Calculates Population Stability Index (PSI) for continuous data (e.g., Credit Scores).
    Automatically bins the `expected` distribution into quantiles and maps the `actual`
    distribution into those exact same bin boundaries.
    """
    expected_s = pd.Series(expected).dropna()
    actual_s = pd.Series(actual).dropna()

    if len(expected_s) == 0 or len(actual_s) == 0:
        raise MetricCalculationError("Cannot calculate PSI: One or both input arrays are empty after dropping nulls.")

    # 1. Calculate quantile bins based EXCLUSIVELY on the expected (training) data
    try:
        _, bins = pd.qcut(expected_s, q=num_buckets, retbins=True, duplicates="drop")
    except ValueError as e:
        raise MetricCalculationError(f"Failed to generate quantile bins for PSI: {str(e)}")

    if len(bins) < 2:
        raise MetricCalculationError("Expected data lacks sufficient variance to create distinct buckets.")

    # 2. Modify outer bounds to prevent crashes from unseen extremes in production data
    # (e.g., an applicant scoring higher than anyone in the training dataset)
    bins[0] = -np.inf
    bins[-1] = np.inf

    # 3. Map both datasets to the fixed expected bins
    expected_counts = pd.cut(expected_s, bins=bins).value_counts().sort_index()
    actual_counts = pd.cut(actual_s, bins=bins).value_counts().sort_index()

    return _compute_psi_core(expected_counts, actual_counts, threshold)


@log_execution_time
def calculate_csi(
    expected: ArrayLike, 
    actual: ArrayLike, 
    threshold: float = DriftThresholds.WARNING
) -> PSIResult:
    """
    Calculates Characteristic Stability Index (CSI) for discrete/categorical data 
    (e.g., 'Employment Status', 'State'). 
    It compares the shift in unique category distributions rather than numeric ranges.
    """
    expected_s = pd.Series(expected).astype(str)
    actual_s = pd.Series(actual).astype(str)

    # 1. Identify all unique categories across both the expected and actual timelines
    all_categories = list(set(expected_s.unique()).union(set(actual_s.unique())))

    # 2. Count occurrences and reindex to ensure both series have identical shapes
    expected_counts = expected_s.value_counts().reindex(all_categories, fill_value=0).sort_index()
    actual_counts = actual_s.value_counts().reindex(all_categories, fill_value=0).sort_index()

    return _compute_psi_core(expected_counts, actual_counts, threshold)