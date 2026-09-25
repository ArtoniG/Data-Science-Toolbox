"""
src/credit_toolbox/metrics/supervised.py

Core statistical metrics for Supervised Learning in Credit Risk.
Calculates traditional banking metrics ($KS$, $Gini$, $IV$) and modern ML metrics 
($AUC$, Brier Score) with strict edge-case handling (e.g., zero-division protection)
and deterministic output validation via Pydantic.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import brier_score_loss, roc_auc_score

from credit_toolbox.core.exceptions import MetricCalculationError
from credit_toolbox.core.types import ArrayLike, IVResult, KSResult
from credit_toolbox.logging.decorators import log_execution_time


@log_execution_time
def calculate_auc(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve ($AUC$).
    Expects y_true as binary labels (0 or 1) and y_pred as probabilities.
    """
    try:
        return float(roc_auc_score(y_true, y_pred))
    except ValueError as e:
        # Catches cases where y_true only contains one class (e.g., all 0s)
        raise MetricCalculationError(f"Failed to calculate AUC: {str(e)}")


@log_execution_time
def calculate_gini(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Calculates the Gini Coefficient.
    In credit scoring, Gini is mathematically defined as: $Gini = 2 * AUC - 1$.
    """
    auc = calculate_auc(y_true, y_pred)
    return 2.0 * auc - 1.0


@log_execution_time
def calculate_brier_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Calculates the Brier Score, which measures the calibration of the probabilities.
    Lower is better (0.0 represents perfect accuracy).
    """
    try:
        return float(brier_score_loss(y_true, y_pred))
    except ValueError as e:
        raise MetricCalculationError(f"Failed to calculate Brier Score: {str(e)}")


@log_execution_time
def calculate_ks(y_true: ArrayLike, y_pred: ArrayLike) -> KSResult:
    """
    Calculates the Kolmogorov-Smirnov ($KS$) statistic.
    Measures the maximum divergence between the Cumulative Distribution Function (CDF) 
    of the Good accounts vs the Bad accounts.
    
    Returns:
        KSResult: Pydantic model containing ks_stat, p_value, and the score_at_ks.
    """
    df = pd.DataFrame({"target": np.array(y_true), "prob": np.array(y_pred)})
    
    # 1. Validation
    if df["target"].nunique() < 2:
        raise MetricCalculationError("KS requires both positive and negative classes in y_true.")

    # 2. Extract arrays for Scipy KS test (gets the exact p-value and stat)
    goods = df[df["target"] == 0]["prob"]
    bads = df[df["target"] == 1]["prob"]
    stat, p_val = ks_2samp(goods, bads)

    # 3. Manual calculation to find the exact threshold (score) where maximum separation occurs
    df_sorted = df.sort_values(by="prob", ascending=False).reset_index(drop=True)
    total_bads = len(bads)
    total_goods = len(goods)

    df_sorted["cum_bads"] = df_sorted["target"].cumsum() / total_bads
    df_sorted["cum_goods"] = (1 - df_sorted["target"]).cumsum() / total_goods
    df_sorted["ks_diff"] = np.abs(df_sorted["cum_bads"] - df_sorted["cum_goods"])
    
    # Find the probability/score at the point of maximum divergence
    max_idx = df_sorted["ks_diff"].idxmax()
    score_at_ks = float(df_sorted.loc[max_idx, "prob"])

    return KSResult(
        ks_stat=float(stat),
        p_value=float(p_val),
        score_at_ks=score_at_ks
    )


@log_execution_time
def calculate_iv(y_true: ArrayLike, feature_bins: ArrayLike) -> IVResult:
    """
    Calculates Information Value ($IV$) and Weight of Evidence ($WOE$) for a binned feature.
    Automatically handles zero-division via epsilon substitution.
    
    Formula:
        $WOE = \\ln(\\% \\text{Goods} / \\% \\text{Bads})$
        $IV = (\\% \\text{Goods} - \\% \\text{Bads}) * WOE$
    """
    df = pd.DataFrame({"target": np.array(y_true), "bin": np.array(feature_bins)})
    
    total_bads = df["target"].sum()
    total_goods = len(df) - total_bads

    if total_bads == 0 or total_goods == 0:
        raise MetricCalculationError("y_true must contain both classes (0 and 1) to calculate IV.")

    # Group by bins to count Goods and Bads
    grouped = df.groupby("bin")["target"].agg(total="count", bads="sum").reset_index()
    grouped["goods"] = grouped["total"] - grouped["bads"]

    # Protect against zero division (infinite WOE) by adding a small epsilon
    epsilon = 1e-6
    grouped["bads_safe"] = np.where(grouped["bads"] == 0, epsilon, grouped["bads"])
    grouped["goods_safe"] = np.where(grouped["goods"] == 0, epsilon, grouped["goods"])

    # Calculate distributions
    grouped["pct_bads"] = grouped["bads_safe"] / total_bads
    grouped["pct_goods"] = grouped["goods_safe"] / total_goods

    # Calculate WOE and IV
    grouped["woe"] = np.log(grouped["pct_goods"] / grouped["pct_bads"])
    grouped["iv"] = (grouped["pct_goods"] - grouped["pct_bads"]) * grouped["woe"]

    total_iv = float(grouped["iv"].sum())
    woe_mapping = dict(zip(grouped["bin"].astype(str), grouped["woe"]))

    # Rule of thumb heuristic for credit risk predictive strength
    if total_iv < 0.02:
        strength = "Useless"
    elif total_iv < 0.10:
        strength = "Weak"
    elif total_iv < 0.30:
        strength = "Medium"
    elif total_iv < 0.50:
        strength = "Strong"
    else:
        strength = "Suspicious (Too Good - Check for Data Leakage)"

    return IVResult(
        total_iv=total_iv,
        predictive_strength=strength,
        woe_mapping=woe_mapping
    )