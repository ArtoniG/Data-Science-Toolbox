# Credit Toolbox

[![CI/CD](https://github.com/your-org/credit-toolbox/actions/workflows/lint_and_test.yml/badge.svg)](https://github.com/your-org/credit-toolbox/actions)
[![PyPI version](https://badge.fury.io/py/credit-toolbox.svg)](https://badge.fury.io/py/credit-toolbox)

An enterprise-grade Python SDK for credit risk modeling, enforcing strict governance, mathematically rigorous metrics, and declarative pipeline reproducibility.

## Core Capabilities
* Supervised & Stability Metrics (Gini, KS, IV, PSI).
* Stateful Transformers (WOE, Imputers, Cappers) with DataFrame preservation.
* Governance & Auditability (Reason Codes, Artifact Export).

## Quickstart
```python
from credit_toolbox import calculate_gini, WOEEncoder
import pandas as pd

# 1. Initialize strict transformer
encoder = WOEEncoder(columns=["bureau_score"])

# 2. Fit and enforce Pandas feature names
X_encoded = encoder.fit_transform(X_train, y_train)

# 3. Export state for Phase 6 Auditability
state = encoder.export_state()