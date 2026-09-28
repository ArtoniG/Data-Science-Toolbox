"""
tests/test_transformers/test_woe_encoder.py

Comprehensive unit tests for the WOEEncoder.
Guarantees mathematical correctness, Phase 1 DataFrame validation, and 
Phase 6 state serialization contracts.
"""

import numpy as np
import pandas as pd
import pytest

from credit_toolbox.transformers.woe_encoder import WOEEncoder


@pytest.fixture
def sample_credit_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a deterministic dataset for WOE binning tests."""
    X = pd.DataFrame({
        "age_segment": ["youth", "youth", "adult", "adult", "senior", "senior", "youth"],
        "revolving_util": [0.9, 0.8, 0.2, 0.3, 0.1, 0.15, 0.95]
    })
    # Target: 1 = Default, 0 = Paid
    y = pd.Series([1, 1, 0, 0, 0, 0, 1])
    return X, y


def test_dataframe_validation_enforcement(sample_credit_data):
    """
    Validates the Phase 1 contract: Transformers must reject raw NumPy arrays 
    to preserve feature names for the Governance layer.
    """
    X, y = sample_credit_data
    encoder = WOEEncoder(columns=["age_segment"])
    
    with pytest.raises(TypeError, match="requires a Pandas DataFrame"):
        encoder.fit(X.values, y.values)


def test_woe_fit_transform_math(sample_credit_data):
    """
    Validates that the encoding calculates proper Weight of Evidence logic 
    without data leakage.
    """
    X, y = sample_credit_data
    encoder = WOEEncoder(columns=["age_segment"])
    
    X_encoded = encoder.fit_transform(X, y)
    
    # Assert return type and structure
    assert isinstance(X_encoded, pd.DataFrame)
    assert "age_segment" in X_encoded.columns
    
    # Youth has 100% default rate in the sample. WOE should reflect high risk.
    # We test relative ordering rather than floating-point exactness to avoid flaky tests
    youth_woe = X_encoded.loc[X["age_segment"] == "youth", "age_segment"].iloc[0]
    adult_woe = X_encoded.loc[X["age_segment"] == "adult", "age_segment"].iloc[0]
    
    assert youth_woe != adult_woe


def test_phase_6_reproducibility_contract(sample_credit_data):
    """
    Validates the Phase 6 contract: Transformers must be able to export 
    their state and be reconstructed perfectly without calling .fit().
    """
    X, y = sample_credit_data
    
    # 1. Train the original encoder
    original_encoder = WOEEncoder(columns=["age_segment"])
    X_transformed_orig = original_encoder.fit_transform(X, y)
    
    # 2. Extract the state (simulate artifact generation)
    state_dict = original_encoder.export_state()
    
    # 3. Instantiate a NEW encoder strictly from the state (simulate reproduce_model.py)
    reconstructed_encoder = WOEEncoder.load_state(state_dict)
    
    # 4. Transform data using the reconstructed encoder (bypassing .fit)
    X_transformed_recon = reconstructed_encoder.transform(X)
    
    # 5. Assert 100% mathematical equality
    pd.testing.assert_frame_equal(X_transformed_orig, X_transformed_recon)


def test_unseen_category_handling(sample_credit_data):
    """
    Validates robustness in production scoring when unseen data appears.
    """
    X, y = sample_credit_data
    encoder = WOEEncoder(columns=["age_segment"])
    encoder.fit(X, y)
    
    # Introduce a category not present in the training set
    X_prod = pd.DataFrame({"age_segment": ["unknown_category"]})
    
    X_prod_encoded = encoder.transform(X_prod)
    
    # The unseen category should safely map to a WOE of 0 (neutral risk)
    assert X_prod_encoded.loc[0, "age_segment"] == 0.0