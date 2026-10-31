---
mode: 'agent'
description: 'Scaffold a new ds_toolbox module with tests, exports, and stub symbols'
tools: ['codebase', 'editFiles', 'runCommands', 'search']
---

You are scaffolding a new module for `ds-toolbox`. Follow AGENTS.md §3, §4,
§6, §7 exactly.

**Inputs (ask if missing):**
- `module_name` — snake_case, matches directory under `src/ds_toolbox/`
- `purpose` — one sentence: what this module owns
- `public_symbols` — list of functions/classes to stub (may be empty)

**Dependency check (AGENTS.md §4):**
- List imports the new module needs.
- Confirm each import is legal per the dependency graph.
- If illegal, stop and propose an ADR in `docs/decisions/` first.

**Create:**

1. `src/ds_toolbox/<module_name>/__init__.py`
   - one-line module docstring
   - `__all__` listing public symbols
   - re-exports only, no logic

2. `src/ds_toolbox/<module_name>/<module_name>.py` (split if >3 symbols)
   - full type annotations + Google-style docstrings
   - bodies are `raise NotImplementedError("<reason>")`
   - private helpers prefixed `_`

3. `tests/<module_name>/__init__.py` (empty)
4. `tests/<module_name>/test_<module_name>.py`
   - `test_<symbol>_happy_path` and `test_<symbol>_raises_on_invalid`
     per public symbol
   - mark `@pytest.mark.xfail(strict=True, reason="not implemented")`
   - no network

**Verify:**
```bash
pip install -e ".[dev]"
ruff check src/ tests/
mypy src/ds_toolbox/<module_name>
pytest tests/<module_name>
lint-imports
```

Report back:

· Files created (paths only)
· Imports used and why they're legal
· Any dep needing an ADR (stop and ask)
· Symbols still unimplemented

Do not implement real logic. Do not touch files outside
src/ds_toolbox/<module_name>/, tests/<module_name>/, and (only if
needed) pyproject.toml under a new extra.