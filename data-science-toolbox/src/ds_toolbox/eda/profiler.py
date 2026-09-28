"""
src/credit_toolbox/eda/profiler.py

Automated exploratory data analysis (EDA) for credit risk modeling.
Generates comprehensive summary statistics, null ratios, cardinality checks, 
and automated distribution drift reports using Population Stability Index (PSI).
Designed to operate efficiently on multi-gigabyte bureau datasets.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from credit_toolbox.core.exceptions import ProfilerError
from credit_toolbox.logging.decorators import log_execution_time
from credit_toolbox.metrics.stability import calculate_psi


class DataProfiler:
    """
    High-performance data profiling utility for credit modeling.
    
    Replaces resource-heavy external libraries (like ydata-profiling) that often 
    crash on large financial datasets. Focuses strictly on the metadata required 
    for credit scoring: missingness, cardinality, extreme values, and population drift.
    """

    @staticmethod
    @log_execution_time
    def generate_summary(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates a consolidated statistical profile of all columns in the DataFrame.
        
        Args:
            df: The raw or processed pandas DataFrame.
            
        Returns:
            A transposed pandas DataFrame where rows are features and columns are metrics.
        """
        if df.empty:
            raise ProfilerError("Cannot profile an empty DataFrame.")

        total_rows = len(df)
        
        # Calculate base metadata
        metadata = pd.DataFrame({
            'Data_Type': df.dtypes.astype(str),
            'Total_Count': df.count(),
            'Missing_Count': df.isnull().sum(),
            'Unique_Count': df.nunique()
        })
        
        metadata['Missing_Ratio'] = metadata['Missing_Count'] / total_rows

        # Calculate numerical statistics only for numeric columns to avoid warnings/errors
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            # describe() computes count, mean, std, min, 25%, 50%, 75%, max
            stats = numeric_df.describe().T
            stats.drop('count', axis=1, inplace=True)  # Redundant with metadata['Total_Count']
        else:
            # Create empty stats columns if no numeric features exist
            stats = pd.DataFrame(
                columns=['mean', 'std', 'min', '25%', '50%', '75%', 'max'], 
                index=df.columns
            )

        # Merge metadata with numerical statistics
        profile = metadata.join(stats, how='left')

        # Reorder columns for optimal readability
        col_order = [
            'Data_Type', 'Total_Count', 'Missing_Count', 'Missing_Ratio', 'Unique_Count',
            'mean', 'std', 'min', '25%', '50%', '75%', 'max'
        ]
        
        return profile[col_order].round(4)

    @staticmethod
    @log_execution_time
    def calculate_population_drift(
        df_expected: pd.DataFrame, 
        df_actual: pd.DataFrame, 
        features: List[str], 
        bins: int = 10
    ) -> pd.DataFrame:
        """
        Calculates the Population Stability Index (PSI) across a list of features 
        to detect systemic distribution shifts between two datasets.
        
        Args:
            df_expected: The baseline DataFrame (e.g., Training set or Old Originations).
            df_actual: The comparative DataFrame (e.g., Out-of-Time test set or Recent Applications).
            features: List of numerical features to evaluate.
            bins: Number of quantiles to use for continuous feature binning.
            
        Returns:
            A pandas DataFrame mapping each feature to its PSI score and risk status.
        """
        missing_expected = [f for f in features if f not in df_expected.columns]
        missing_actual = [f for f in features if f not in df_actual.columns]
        
        if missing_expected or missing_actual:
            raise ProfilerError(
                f"Features missing from expected: {missing_expected}. "
                f"Features missing from actual: {missing_actual}."
            )

        psi_results = []

        for feature in features:
            expected_series = df_expected[feature].dropna()
            actual_series = df_actual[feature].dropna()
            
            # Delegate mathematical calculation to Phase 3 metrics module
            psi_value = calculate_psi(expected_series, actual_series, bins=bins)
            
            # Standard industry thresholds for PSI interpretation
            if psi_value < 0.1:
                status = "Stable"
            elif psi_value < 0.25:
                status = "Slight Drift"
            else:
                status = "Significant Drift"
                
            psi_results.append({
                'Feature': feature,
                'PSI': psi_value,
                'Status': status
            })

        return pd.DataFrame(psi_results).sort_values(by='PSI', ascending=False).reset_index(drop=True)