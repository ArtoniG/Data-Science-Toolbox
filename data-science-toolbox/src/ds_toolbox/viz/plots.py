"""
src/credit_toolbox/viz/plots.py

Enterprise-grade plotting functions for credit risk models.
Generates standard validation charts (ROC, KS, Score Distributions, and PSI) 
tightly integrated with the corporate visual identity defined in theme.py.
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

from credit_toolbox.core.types import ArrayLike
from credit_toolbox.viz.theme import CreditTheme


def plot_roc_curve(
    y_true: ArrayLike, 
    y_prob: ArrayLike, 
    title: str = "Receiver Operating Characteristic (ROC)", 
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plots the ROC curve and calculates the AUC score.
    """
    CreditTheme.apply_theme()
    
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    # Plot actual ROC curve using the Primary Corporate Blue
    ax.plot(
        fpr, tpr, 
        color=CreditTheme.COLORS["primary"], 
        label=f"Model (AUC = {auc_score:.3f})"
    )
    
    # Plot random baseline
    ax.plot(
        [0, 1], [0, 1], 
        color=CreditTheme.COLORS["tertiary"], 
        linestyle="--", 
        label="Random (AUC = 0.500)"
    )

    ax.set_title(title)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.legend(loc="lower right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    return ax


def plot_ks_statistic(
    y_true: ArrayLike, 
    y_prob: ArrayLike, 
    title: str = "Kolmogorov-Smirnov (KS) Statistic", 
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plots the cumulative distribution of Good vs. Bad populations and highlights 
    the point of maximum separation (the KS Statistic).
    """
    CreditTheme.apply_theme()
    
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    df = pd.DataFrame({'target': y_true, 'prob': y_prob}).sort_values('prob', ascending=False)
    df.reset_index(drop=True, inplace=True)
    
    # Calculate cumulative proportions
    total_bads = df['target'].sum()
    total_goods = len(df) - total_bads
    
    df['cum_bads'] = df['target'].cumsum() / total_bads
    df['cum_goods'] = (1 - df['target']).cumsum() / total_goods
    df['ks_diff'] = np.abs(df['cum_bads'] - df['cum_goods'])
    
    ks_stat = df['ks_diff'].max()
    ks_idx = df['ks_diff'].idxmax()
    ks_prob = df.loc[ks_idx, 'prob']

    # Plot CDFs
    ax.plot(df.index / len(df), df['cum_bads'], color=CreditTheme.COLORS["bad"], label="Cumulative Bads (Defaults)")
    ax.plot(df.index / len(df), df['cum_goods'], color=CreditTheme.COLORS["good"], label="Cumulative Goods (Pays)")
    
    # Highlight KS point
    ax.vlines(
        ks_idx / len(df), 
        ymin=df.loc[ks_idx, 'cum_goods'], 
        ymax=df.loc[ks_idx, 'cum_bads'], 
        color=CreditTheme.COLORS["warning"], 
        linestyle=":", 
        linewidth=2.5,
        label=f"KS = {ks_stat:.3f} (at Top {ks_idx/len(df)*100:.1f}%)"
    )

    ax.set_title(title)
    ax.set_xlabel("Population % (Sorted by Risk Score Descending)")
    ax.set_ylabel("Cumulative Proportion")
    ax.legend(loc="lower right")

    return ax


def plot_score_distribution(
    y_true: ArrayLike, 
    y_prob: ArrayLike, 
    bins: int = 50,
    title: str = "Risk Score Distribution by Target", 
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plots normalized histograms/KDEs of model probabilities separated by the target class.
    Crucial for visualizing the overlap (grey area) between approved and rejected populations.
    """
    CreditTheme.apply_theme()
    
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    df = pd.DataFrame({'target': y_true, 'prob': y_prob})
    
    # Use seaborn histplot for clean, normalized density fills
    sns.histplot(
        data=df, x='prob', hue='target', 
        stat='density', common_norm=False, 
        bins=bins, kde=True, 
        palette={0: CreditTheme.COLORS["good"], 1: CreditTheme.COLORS["bad"]},
        ax=ax, alpha=0.5, edgecolor=None
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted Probability of Default (PD)")
    ax.set_ylabel("Density")
    
    # Custom legend
    handles = [
        plt.Rectangle((0,0),1,1, color=CreditTheme.COLORS["good"], alpha=0.5),
        plt.Rectangle((0,0),1,1, color=CreditTheme.COLORS["bad"], alpha=0.5)
    ]
    ax.legend(handles, ["Good (Target=0)", "Bad (Target=1)"], loc="upper right")

    return ax


def plot_psi_summary(
    psi_df: pd.DataFrame, 
    title: str = "Population Stability Index (PSI) Summary", 
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plots a horizontal bar chart of feature PSI values, color-coded by risk severity.
    Expects the DataFrame output from DataProfiler.calculate_population_drift().
    """
    CreditTheme.apply_theme()
    
    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(6, len(psi_df) * 0.4)))

    # Sort ascending for horizontal bar chart (largest at the top)
    df_sorted = psi_df.sort_values('PSI', ascending=True).copy()

    # Dynamic color mapping based on standard credit thresholds
    colors = df_sorted['PSI'].apply(
        lambda x: CreditTheme.COLORS["bad"] if x >= 0.25 
        else (CreditTheme.COLORS["warning"] if x >= 0.1 else CreditTheme.COLORS["good"])
    ).tolist()

    bars = ax.barh(df_sorted['Feature'], df_sorted['PSI'], color=colors)

    # Threshold markers
    ax.axvline(0.10, color=CreditTheme.COLORS["warning"], linestyle="--", alpha=0.7, label="Slight Drift (0.10)")
    ax.axvline(0.25, color=CreditTheme.COLORS["bad"], linestyle="--", alpha=0.7, label="Significant Drift (0.25)")

    ax.set_title(title)
    ax.set_xlabel("PSI Value")
    ax.legend(loc="lower right")

    # Annotate bars with exact PSI values
    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{width:.3f}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),  # 3 points horizontal offset
            textcoords="offset points",
            ha="left", va="center",
            fontsize=9, color=CreditTheme.COLORS["text"]
        )

    # Remove Y-axis grid for cleaner look on horizontal bars
    ax.yaxis.grid(False)
    
    return ax