"""
src/credit_toolbox/core/exceptions.py

Core exception hierarchy for the custom-credit-toolbox.
By subclassing a single root exception, we allow downstream client applications
to catch and log all toolbox-specific errors systematically.
"""

class CreditToolboxError(Exception):
    """
    Base exception for all custom errors within the custom-credit-toolbox.
    Clients can use `except CreditToolboxError:` to safely catch any SDK-generated failure
    without catching standard Python runtime errors.
    """
    pass


class DataValidationError(CreditToolboxError):
    """
    Raised when input data violates structural or statistical assumptions.
    Examples: Unexpected infinite values, all-null columns, or negative values in age fields.
    """
    pass


class SchemaMismatchError(CreditToolboxError):
    """
    Raised when a DataFrame schema does not match the expected feature manifest.
    Examples: Missing required columns, or a feature passing as an object/string 
    when the model expects a float64.
    """
    def __init__(self, message: str, missing_columns: list[str] | None = None):
        super().__init__(message)
        self.missing_columns = missing_columns or []


class BinningError(CreditToolboxError):
    """
    Raised when Weight of Evidence (WOE) or quantile binning fails.
    Examples: A feature has zero variance (all identical values), or identical 
    percentile edges prevent the creation of discrete bins.
    """
    pass


class MetricCalculationError(CreditToolboxError):
    """
    Raised when a core statistical calculation fails.
    Examples: Zero division when calculating Information Value (IV), or 
    attempting to calculate Gini with a single target class.
    """
    pass


class ModelDriftError(CreditToolboxError):
    """
    Raised when distribution drift metrics (like PSI or CSI) exceed critical thresholds,
    indicating the model is operating outside its training bounds.
    """
    def __init__(self, message: str, feature_name: str, drift_score: float, threshold: float):
        super().__init__(message)
        self.feature_name = feature_name
        self.drift_score = drift_score
        self.threshold = threshold


class GenAIGuardrailError(CreditToolboxError):
    """
    Raised when a Generative AI or LLM output fails deterministic validation.
    Examples: The LLM hallucinated a feature name, or returned a narrative 
    when a strict JSON schema was required.
    """
    pass