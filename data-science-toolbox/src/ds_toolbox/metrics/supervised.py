"""
src/credit_toolbox/metrics/supervised.py

Supervised evaluation metrics tailored for credit risk modeling.
Includes standard discriminatory power metrics (Gini, KS) and structural 
quality gate checks (Monotonicity) required for Phase 6 regulatory compliance 
and automated pipeline auditing.
"""

from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def gini_score(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculates the Gini Coefficient, the standard discriminatory metric in credit scoring.
    
    Args:
        y_true (Union[pd.Series, np.ndarray]): Binary ground truth (0 = Good, 1 = Bad).
        y_pred (Union[pd.Series, np.ndarray]): Predicted probabilities or scores.
        
    Returns:
        float: Gini coefficient ranging from -1.0 to 1.0.
    """
    auc = roc_auc_score(y_true, y_pred)
    return (2 * auc) - 1


def ks_statistic(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculates the Kolmogorov-Smirnov (KS) statistic, measuring the maximum separation 
    between the cumulative distribution of Goods and Bads.
    
    Args:
        y_true (Union[pd.Series, np.ndarray]): Binary ground truth (0 = Good, 1 = Bad).
        y_pred (Union[pd.Series, np.ndarray]): Predicted probabilities or scores.
        
    Returns:
        float: KS statistic ranging from 0.0 to 1.0.
    """
    df = pd.DataFrame({"y": y_true, "p": y_pred})
    df = df.sort_values(by="p", ascending=False)
    
    total_bads = df["y"].sum()
    total_goods = len(df) - total_bads
    
    # Cumulative distributions
    cum_bads = df["y"].cumsum() / total_bads
    cum_goods = (1 - df["y"]).cumsum() / total_goods
    
    ks = np.abs(cum_bads - cum_goods).max()
    return float(ks)


def evaluate_monotonicity(x_binned: pd.Series, y_true: pd.Series) -> Dict[str, Any]:
    """
    Evaluates the monotonicity of default rates across discrete bins or WOE values.
    Critical for regulatory compliance to ensure risk increases/decreases consistently.
    
    Args:
        x_binned (pd.Series): Discretized feature, ordinal bins, or encoded WOE values.
        y_true (pd.Series): Binary ground truth (0 = Good, 1 = Bad).
        
    Returns:
        Dict[str, Any]: Dictionary containing Spearman correlation, strict monotonicity flag, 
                        and the calculated event rates per bin.
    """
    df = pd.DataFrame({"x": x_binned, "y": y_true})
    
    # Drop NaNs to isolate the mathematical relationship of populated bins
    df = df.dropna()
    
    # Calculate default rate (mean of y) per bin, sorted by bin value
    grouped = df.groupby("x")["y"].mean().sort_index()
    
    bin_values = grouped.index.values
    event_rates = grouped.values
    
    if len(bin_values) < 2:
        return {
            "is_strictly_monotonic": True,
            "spearman_correlation": 1.0,
            "trend": "insufficient_bins",
            "event_rates": grouped.to_dict()
        }
    
    # Calculate Spearman Rank Correlation between bin index and event rate
    # Using nan_policy='omit' to be safe, though dropna() was called above
    correlation, p_value = spearmanr(bin_values, event_rates, nan_policy="omit")
    
    # Calculate strict monotonic differences
    differences = np.diff(event_rates)
    is_increasing = np.all(differences >= 0)
    is_decreasing = np.all(differences <= 0)
    
    if is_increasing:
        trend = "increasing"
    elif is_decreasing:
        trend = "decreasing"
    else:
        trend = "non_monotonic"
        
    return {
        "is_strictly_monotonic": is_increasing or is_decreasing,
        "spearman_correlation": float(correlation) if not np.isnan(correlation) else 0.0,
        "spearman_p_value": float(p_value) if not np.isnan(p_value) else 1.0,
        "trend": trend,
        "event_rates": grouped.to_dict()
    }