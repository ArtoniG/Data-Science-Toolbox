The construction of a standalone SDK follows an Inside-Out Dependency Strategy. Every Python module in the toolbox must only depend on modules built in earlier phases. If a downstream module (like a model card generator) is built before its upstream dependencies (like metrics or visual themes), you create circular imports, missing imports, and untestable code.
​The exact implementation sequence guarantees a 100% stable, fully testable, and zero-circular-dependency build of custom-credit-toolbox.
​Phase 0: Packaging & Development Harness
​Before writing a single line of business logic, establish the package boundaries, dependency resolution rules, and linting standards.
​pyproject.toml: Define core metadata, minimum Python version (>=3.10), entry points, and optional feature flags ([genai], [viz], [api]).
​ruff.toml & .pre-commit-config.yaml: Set strict linting, import sorting (isort), and type-checking rules (mypy).
​src/credit_toolbox/__init__.py: Expose global package metadata (__version__ = "0.1.0").
​Phase 1: Zero-Dependency Primitives & Core Types
​These files have zero internal dependencies. Every subsequent module imports from this layer.
​src/credit_toolbox/core/exceptions.py:
​Define base exception CreditToolboxError(Exception).
​Define specific subclasses: DataValidationError, ModelDriftError, BinningError, SchemaMismatchError.
​src/credit_toolbox/core/types.py:
​Define Pydantic base models, typing aliases (ArrayLike = Union[np.ndarray, pd.Series]), and dataclasses for metric outputs.
​Phase 2: Observability & Logging Layer
​Every operational function, transformer, and metric calculation must log its execution context using a unified schema.
​src/credit_toolbox/logging/logger.py:
​Implements a structured JSON logger (using standard library or structlog).
​Ensures output contains timestamp, log level, module name, and session/audit IDs.
​src/credit_toolbox/logging/decorators.py:
​Implements @audit_trail and @log_execution_time.
​Imports src/credit_toolbox/core/exceptions.py to catch and log unhandled operational failures.
​Phase 3: Mathematical & Statistical Core
​Pure algorithmic functions operating on NumPy arrays and Pandas DataFrames. Zero internal dependencies except core/ and logging/.
​src/credit_toolbox/metrics/supervised.py:
​Core scoring math: KS (Kolmogorov-Smirnov), Gini, IV (Information Value), AUC, Brier Score.
​src/credit_toolbox/metrics/unsupervised.py:
​Clustering and segmentation evaluation: Silhouette score, Dunn index.
​src/credit_toolbox/metrics/stability.py:
​Distribution drift math: PSI (Population Stability Index), CSI (Characteristic Stability Index).
​src/credit_toolbox/metrics/genai.py (Optional Extra):
​Evaluation metrics for credit LLM applications: Faithfulness, Hallucination Rate, Toxicity.
Phase 4: Scikit-Learn Pipelines & Data Profiling
​Data manipulation and feature engineering logic that utilizes the mathematical formulas from Phase 3.
​src/credit_toolbox/transformers/woe_encoder.py:
​Custom Scikit-learn transformer implementing Weight of Evidence binning. Imports IV math from metrics/supervised.py.
​src/credit_toolbox/transformers/outlier_capper.py:
​Winsorization and percentile capping logic.
​src/credit_toolbox/transformers/missing_imputer.py:
​Credit-specific imputation routines (e.g., special codes for missing bureau data).
​src/credit_toolbox/eda/profiler.py:
​Automated exploratory data analysis generating null ratios, cardinality checks, and feature distributions using metrics/stability.py.
​Phase 5: Corporate Visual Identity & Plotting
​Visualization tools that take processed data and metric outputs to generate clean, enterprise-ready charts.
​src/credit_toolbox/viz/theme.py:
​Matplotlib/Seaborn global configuration (rcParams), corporate color palettes (hex codes), and font defaults.
​src/credit_toolbox/viz/plots.py:
​Standard plotting routines: ROC curves, KS charts, score distribution histograms, and PSI heatmaps. Depends on viz/theme.py and outputs from metrics/.
​Phase 6: Compliance, Governance & Automated Reporting
​The highest-level orchestration layer that aggregates data from transformers, metrics, and visualization modules to generate business deliverables.
​src/credit_toolbox/governance/reason_codes.py:
​Converts raw SHAP values into compliant human-readable adverse action codes (business text).
​src/credit_toolbox/governance/model_card.py:
​Aggregates model metadata, performance metrics from metrics/, and charts from viz/ into an automated PDF/Markdown documentation artifact.
​Phase 7: Parallel Unit Testing & Public API Clean-up
​To ensure long-term reliability and clean developer experience (DX), finalize tests and public namespace exposure.
1.tests/ Mirror Directory:
​Implement 1:1 unit test files (tests/test_metrics/test_supervised.py, tests/test_transformers/test_woe_encoder.py) targeting 100% logic coverage on edge cases (e.g., zero-division handling in IV).
2.Top-Level Exports (src/credit_toolbox/__init__.py):
​Expose key classes and functions at the root package level so users can write concise imports:
# Expose clean public interface
from credit_toolbox.metrics.supervised import calculate_ks, calculate_gini
from credit_toolbox.logging.logger import get_logger

__all__ = ["calculate_ks", "calculate_gini", "get_logger"]