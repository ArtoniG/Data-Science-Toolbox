"""
src/credit_toolbox/viz/theme.py

Corporate visual identity configuration for matplotlib and seaborn.
Standardizes colors, fonts, and chart layouts across all model reporting 
to ensure executive-ready deliverables without repetitive boilerplate.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

class CreditTheme:
    """
    Centralized configuration class for enterprise visualization.
    Manages global rcParams and semantic color palettes specific to credit risk.
    """
    
    # Define enterprise color palette (Clean blues, neutral grays, and semantic risk colors)
    COLORS = {
        "primary": "#003366",       # Deep Corporate Blue
        "secondary": "#4A90E2",     # Highlight Blue
        "tertiary": "#87939F",      # Neutral Gray
        "background": "#FFFFFF",    # Pure white background for PDF/PPT export
        "text": "#212529",          # Dark slate for readable text
        "good": "#28A745",          # Green (Non-Default / Stable PSI)
        "bad": "#DC3545",           # Red (Default / Significant Drift)
        "warning": "#FFC107"        # Yellow (Review / Slight Drift)
    }

    # Matplotlib rcParams dictionary for global styling
    RC_PARAMS = {
        # Figure layout
        "figure.figsize": (10, 6),
        "figure.facecolor": COLORS["background"],
        "figure.dpi": 150,           # High DPI for crisp presentation renders
        "axes.facecolor": COLORS["background"],
        "axes.edgecolor": "#DEE2E6",
        "axes.linewidth": 1.0,
        
        # Gridlines (Subtle, for cleaner data presentation)
        "axes.grid": True,
        "grid.color": "#E9ECEF",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,      # Puts grid behind data points/bars
        
        # Typography
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlepad": 15,
        "axes.labelsize": 12,
        "axes.labelweight": "medium",
        "axes.labelpad": 10,
        
        # Ticks (Clean axes without harsh tick marks)
        "xtick.color": "#495057",
        "ytick.color": "#495057",
        "xtick.bottom": False,
        "ytick.left": False,
        
        # Legend (Unobtrusive)
        "legend.frameon": False,
        "legend.fontsize": 10,
        
        # Lines and markers
        "lines.linewidth": 2.5,
        "lines.markersize": 6
    }

    @classmethod
    def apply_theme(cls) -> None:
        """
        Applies the corporate theme to the active matplotlib/seaborn environment.
        Should be called once at the start of any plotting script or Jupyter Notebook.
        """
        # Update matplotlib global parameters
        plt.rcParams.update(cls.RC_PARAMS)
        
        # Set Seaborn theme with our specific color cycle
        sns.set_theme(
            style="whitegrid", 
            palette=[cls.COLORS["primary"], cls.COLORS["secondary"], cls.COLORS["tertiary"]]
        )
        
        # Override seaborn's default grid lines with our custom hex codes
        sns.set_style("whitegrid", {
            "axes.edgecolor": cls.RC_PARAMS["axes.edgecolor"],
            "grid.color": cls.RC_PARAMS["grid.color"]
        })

    @classmethod
    def get_risk_cmap(cls) -> LinearSegmentedColormap:
        """
        Generates a custom continuous colormap from 'Good' (Green) to 'Bad' (Red).
        Critical for Population Stability Index (PSI) heatmaps or Risk Tier visualizations.
        """
        colors = [cls.COLORS["good"], cls.COLORS["warning"], cls.COLORS["bad"]]
        return LinearSegmentedColormap.from_list("RiskMap", colors)
        
    @classmethod
    def get_corporate_cmap(cls) -> LinearSegmentedColormap:
        """
        Generates a branded continuous colormap (Light Blue to Dark Blue).
        Used for score distributions and correlation matrices.
        """
        colors = ["#E3F2FD", cls.COLORS["secondary"], cls.COLORS["primary"]]
        return LinearSegmentedColormap.from_list("CorporateBlue", colors)