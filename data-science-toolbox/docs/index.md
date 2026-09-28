### 2. `docs/index.md` & `docs/api/` (The Developer Experience)
With the README pointing to the docs, we establish the site structure (assuming MkDocs with Material theme and `mkdocstrings` for automatic docstring pulling).

**`docs/index.md`** (The Documentation Homepage):
```markdown
# Welcome to Credit Toolbox

The `credit_toolbox` is designed for Data Scientists working in advanced analytics for credit bureaus and financial institutions. It bridges the gap between Scikit-Learn's flexibility and strict financial regulatory requirements.

## Architecture Map
- [Metrics Layer](api/metrics.md): Gini, KS, IV, and PSI functions.
- [Transformers Layer](api/transformers.md): WOE and imputation modules.
- [Governance Layer](api/governance.md): Pipeline auditors and reason codes.