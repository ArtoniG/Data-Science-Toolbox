"""Fix broken f-string in governance/model_card.py (PR A, one-shot)."""
from pathlib import Path

p = Path("src/ds_toolbox/governance/model_card.py")
lines = p.read_text().splitlines(keepends=True)

# Locate the broken f-string start
start = next(
    i for i, l in enumerate(lines) if 'md_content = f"""# Model Card' in l
)

# Locate the trailing f.write that got dedented to column 0
end = next(
    i for i, l in enumerate(lines[start:], start)
    if l.strip() == 'f.write(md_content)'
)

# Replacement block: closes the string, dedents the with-block back inside the method
fixed = '''        md_content = f"""# Model Card

**Model Name:** {self.model_name}
**Author:** Guilherme Artoni
**Organization:** Advanced Analytics
**Compilation Date:** 2026-09-27

## System Architecture

This pipeline enforces deterministic mathematical constraints prior to production deployment. All features have passed Phase 6 Quality Gates for imputation, capping, and strict WOE monotonicity.

## Reproducibility

The internal state of this model (boundaries, thresholds, weights) has been serialized. To recreate this Scikit-Learn Pipeline identically for auditing or future quantitative research:
python reproduce_model.py
"""
        md_path.write_text(md_content)
'''

lines[start:end + 1] = [fixed]
p.write_text("".join(lines))
print(f"Fixed lines {start + 1}-{end + 1} of {p}")

# Sanity check
import ast
ast.parse(p.read_text())
print("Syntax OK")