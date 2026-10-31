---
mode: 'agent'
description: 'Add or change a public symbol following the API contract'
tools: ['codebase', 'editFiles', 'runCommands', 'search']
---

You are adding or changing a public symbol in `ds-toolbox`. Strict contract.
Follow AGENTS.md §2, §6, §7, §8.

**Inputs (ask if missing):**
- `module` — e.g. `transformers/outlier_capper.py`
- `symbol` — function or class name
- `kind` — `new` | `signature-change` | `behavior-change` | `deprecation`

**Contract, in order:**

1. Full type annotations (including return).
2. Google docstring: `Args`, `Returns`, `Raises`, `Example`.
3. Errors are subclasses of `ds_toolbox.core.exceptions.DSToolboxError`.
4. Logging via `observability.logger.get_logger(__name__)`. No `print`.
5. Determinism: any randomness accepts `random_state`.
6. Update `__all__` in the module package's `__init__.py`.
7. Tests in `tests/<module_dir>/test_<file>.py`:
   - happy path
   - invalid input raises the documented exception
   - one edge case that matters (empty / NaN / single-row)
8. `CHANGELOG.md` under `## Unreleased`:
   - `### Added` for `new`
   - `### Changed` for `signature-change` / `behavior-change`
   - `### Deprecated` for `deprecation`
   Include a one-line summary and the issue reference.
9. Public docs: confirm the symbol appears in `docs/api/<module>.md`
   (mkdocstrings picks it up from the docstring — just verify it renders).

**Breaking changes (AGENTS.md §2 rule 1):**
- `signature-change` and `behavior-change` → add a `## Breaking changes`
  section to the PR body.
- `deprecation` → raise `DeprecationWarning` + `MIGRATION.md` entry.

**Verify:**
```bash
ruff check src/ tests/
mypy src/ds_toolbox/<module_dir>
pytest tests/<module_dir> -q
lint-imports
```

Report back: symbol, new test names, CHANGELOG line, breaking yes/no.