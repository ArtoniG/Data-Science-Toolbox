"""
tests/conftest.py

Shared Pytest fixtures for the credit_toolbox test suite.
Provides standardized, deterministic credit risk datasets, mappings, and pre-fitted 
pipelines to ensure unit tests run quickly and consistently across all environments.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@pytest.fixture(scope="session")
def sample_bureau_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Generates a deterministic credit bureau dataset simulating realistic
    financial features (e.g., BRL income, utilization rates, PIX activity).
    
    Scope is set to 'session' so the matrix is generated only once per test run,
    drastically reducing overhead when running the full CI/CD suite.
    """
    # Fix the seed for deterministic test outcomes
    np.random.seed(42)
    n_samples = 1500
    
    # Simulate realistic distributions for credit features
    X = pd.DataFrame({
        "age": np.random.randint(18, 75, n_samples),
        "monthly_income_brl": np.random.lognormal(mean=8.5, sigma=0.8, size=n_samples).round(2),
        "revolving_utilization": np.random.beta(a=2, b=5, size=n_samples).round(4),
        "num_pix_transactions_30d": np.random.poisson(lam=15, size=n_samples),
        "months_since_delinquency": np.random.randint(0, 120, n_samples)
    })
    
    # Inject missing values to test Phase 4 Imputers
    X.loc[0:50, "months_since_delinquency"] = np.nan
    
    # Create a mock target variable (default flag) mathematically tied to utilization and income
    risk_score = (X["revolving_utilization"] * 5) - (np.log(X["monthly_income_brl"]) * 0.5)
    prob_default = 1 / (1 + np.exp(-risk_score))
    y = pd.Series(np.random.binomial(1, prob_default), name="default_flag")
    
    return X, y


@pytest.fixture
def pre_fitted_credit_pipeline(sample_bureau_dataset) -> Pipeline:
    """
    Provides a standard Scikit-Learn pipeline pre-fitted on the mock bureau data.
    Essential for testing Phase 6 (ArtifactExporter and PipelineAuditor) 
    without having to retrain models inside every individual test.
    """
    X, y = sample_bureau_dataset
    
    # Fill missing values safely for the sake of the baseline Scikit-Learn pipeline
    X_clean = X.fillna(-1)
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42, max_iter=100))
    ])
    
    pipeline.fit(X_clean, y)
    return pipeline


@pytest.fixture
def governance_mapping_dicts() -> tuple[dict, dict]:
    """
    Provides standardized dictionaries specifically for validating the 
    ReasonCodeExtractor in Phase 6.
    """
    feature_to_code = {
        "revolving_utilization": "RC001",
        "months_since_delinquency": "RC002",
        "num_pix_transactions_30d": "RC003",
        "monthly_income_brl": "RC004",
        "age": "RC005"
    }
    
    code_to_desc = {
        "RC001": "Proportion of balances to credit limits is too high.",
        "RC002": "Recent delinquency or derogatory public record.",
        "RC003": "Insufficient recent transactional velocity.",
        "RC004": "Income-to-debt capability does not meet thresholds.",
        "RC005": "Length of credit history is too short."
    }
    
    return feature_to_code, code_to_desc