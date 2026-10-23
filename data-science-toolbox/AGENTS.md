# AGENTS.md — ds-toolbox

> Read this file first. It is the single source of truth for how agents
> (Copilot, Cursor, Claude Code, Aider) must work in this repository.
> If this file conflicts with any other doc, this file wins.
> If a rule here is wrong, open an issue titled `AGENTS.md: <rule>`.

---

## 1. What this repository is

`ds-toolbox` is an installable Python library of reusable, production-grade
utilities for data science and ML engineering, with deep specialization in
**credit risk / financial modeling** and growing coverage in **MLOps** and
**agentic systems**.

- **Owner:** Artoni
- **License:** proprietary
- **Current version:** 0.1.0
- **Python support:** >=3.11
- **Package name (PyPI):** `ds-toolbox`
- **Import name:** `ds_toolbox`
- **Repository purpose:** reusable IP for a solo data-science consultancy.
  Every module must be defensible as "something a DS Company would pay to reuse."

Non-goals (do not add these to core):
- Application code, dashboards, or notebooks at the repo root.
- Anything client-specific or containing PII.
- Vendored third-party libraries.

---

## 2. Golden rules (non-negotiable)

1. **Never break the public API silently.** Any change to a public symbol
   (`__all__`, public function signature, class attribute) requires a
   CHANGELOG entry and, if pre-1.0, a note in the PR description.
2. **`src/` layout is mandatory.** No top-level modules, no `import ds_toolbox`
   from the repo root without `pip install -e .`.
3. **Every public module has a test.** PRs that add code without tests are
   rejected by CI.
4. **Every public symbol has a docstring** (Google style). Private helpers
   (`_name`) do not require docstrings but should be self-explanatory.
5. **No new runtime dependency** without an explicit ADR in
   `docs/decisions/`. Optional dependencies go in `pyproject.toml` extras.
6. **No secrets, tokens, or credentials** in any file. Use `config/settings.py`
   (Pydantic Settings) or environment variables.
7. **No agent may modify CI workflows, `pyproject.toml`, `AGENTS.md`,
   or `docs/decisions/` without explicit human approval in the PR.**
8. **Follow the dependency rules in §4.** Violations are blocked by
   `import-linter` in CI.
9. **Tests must not hit the network.** Use fixtures, `responses`, or
   `respx`. Mark integration tests with `@pytest.mark.integration`.
10. **Prefer small PRs.** One module or one concern per PR. If a task spans
    more than two modules, split it into multiple issues first.

---

## 3. Repository map

Each directory has ONE responsibility. Do not mix.

```
ds-toolbox/
├── src/ds_toolbox/
│   ├── init.py            # public re-exports only; no logic
│   ├── py.typed               # PEP 561 marker; keep present
│   ├── core/                  # base classes, exceptions, shared types
│   ├── config/                # Pydantic Settings; env + YAML loading
│   ├── io/                    # readers/writers (parquet, csv, cloud)
│   ├── validation/            # pandera schemas, data contracts
│   ├── eda/                   # profiling, summaries
│   ├── governance/            # model cards, quality gates, reason codes
│   ├── observability/         # logging, decorators, (future) tracing
│   ├── metrics/               # supervised, unsupervised, stability, genai
│   ├── transformers/          # sklearn-compatible feature transformers
│   ├── mlops/                 # tracking, registry, serving, pipelines
│   ├── agentic/               # [FILL: scope if in v0; else "future"]
│   └── viz/                   # plots, shared theme
├── tests/                     # mirrors src/ one-to-one
├── examples/                  # runnable notebooks/scripts per use case
├── docs/                      # mkdocs site + ADRs + architecture
├── .github/workflows/         # CI (ci.yml, publish.yml)
├── pyproject.toml
├── ruff.toml
├── .pre-commit-config.yaml
├── Makefile
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
├── LICENSE
└── README.md
```

---

## 4. Module dependency rules (enforced by import-linter)

Imports flow **downward only**. If an arrow is missing, it is forbidden.

```
agentic   → mlops → metrics → transformers → validation → io → config → core
viz       → core
governance → metrics → core
observability → core
eda       → io → validation → core
```

Additional rules:

- `core/` imports **nothing** from `ds_toolbox`.
- `config/` imports only `core/`.
- No module imports `agentic/` except `agentic/` itself and `__init__.py`.
- Circular imports are always a bug — refactor, do not suppress.

If you believe a new edge is needed, open an ADR first.

---

## 5. Commands (run these, in this order)

Replace placeholders only if you adopt a different toolchain; then update
this section AND `.github/workflows/ci.yml` in the same PR.

```bash
# Setup
uv sync --all-extras --dev

# Lint + format
ruff check . && ruff format --check .

# Type check
mypy src

# Import rules
lint-imports

# Tests
pytest -q --cov=ds_toolbox --cov-fail-under=80

# Everything (what CI runs)
make check
```

If you add a command, add it to both Makefile and CI. Never one alone.

---

## 6. Code conventions

· Style: ruff (config in ruff.toml). Do not reformat unrelated lines.
· Types: full annotations on public functions. mypy strict mode is planned.
· Docstrings: Google style. Public functions need Args, Returns,
  Raises (if applicable), and an Example for anything non-obvious.
· Errors: raise subclasses of ds_toolbox.core.exceptions.DSToolboxError.
  Never raise bare Exception. Never swallow exceptions silently.
· Logging: use ds_toolbox.observability.logger.get_logger(__name__).
  Never print(). Never logging.getLogger directly.
· Config: read config via ds_toolbox.config.settings.Settings, never
  os.environ directly outside config/.
· DataFrames: prefer polars where performance matters; pandas is
  acceptable at API boundaries. Document the choice per module.
· Determinism: any function with randomness accepts random_state.
· No globals that hold state across calls. Pass dependencies explicitly.

---

## 7. Testing conventions

· Location: tests/<module_path>/test_<file>.py, mirroring src/.
· Framework: pytest. Fixtures in tests/conftest.py (root) and
  per-module conftest.py when scoped.
· Structure: Arrange / Act / Assert, separated by blank lines.
· Naming: test_<behavior>_<condition>_<expected>.
· Required per public symbol: at least one happy path and one
  error/edge test.
· Coverage target: 100% lines. CI fails below.
· Markers: @pytest.mark.integration for tests hitting real services
  (skipped by default locally, run in CI nightly).
· No network in unit tests. No real credentials anywhere.

---

## 8. Documentation

· Every public module has a matching page under docs/api/ (generated by
  mkdocstrings from docstrings — keep docstrings accurate).
· Architectural changes require an ADR in docs/decisions/ADR-NNNN-<slug>.md.
· User-facing changes require an entry in CHANGELOG.md under Unreleased.
· Examples in examples/<use_case>/ must be runnable end-to-end.

---

## 9. Pull request workflow

1. Issue first. Every PR references an issue (Closes #N).
2. Branch name: feat/…, fix/…, docs/…, refactor/…, chore/….
3. Commit style: Conventional Commits
   (feat(transformers): add …). Squash on merge.
4. PR description must include:
   · What changed and why
   · Which module(s) touched
   · How it was tested
   · Any dependency/API changes
   · Checklist from the PR template
5. CI must be green before review request.
6. Reviewers: at least one human. Agent self-approval is not accepted.
7. Size: target <400 changed lines. Split larger PRs.

---

## 10. What agents MUST NOT do

· Rewrite files outside the scope of the current issue.
· Introduce new dependencies, new modules, or new top-level packages
  without an ADR and human approval.
· Delete or weaken tests to make CI pass.
· Change pyproject.toml, AGENTS.md, .github/workflows/*, or
  docs/decisions/* without human approval.
· Generate placeholder code with TODO, pass, or NotImplementedError
  and leave it unmarked in the PR description.
· Fabricate API behavior — if unsure, open an issue and stop.
· Commit secrets, API keys, or environment dumps.
· Refactor while fixing a bug. Separate PRs.
· Make claims about performance without a benchmark in examples/benchmarks/.

---

## 11. Deeper context

Agents should consult these before non-trivial work:

· docs/architecture.md — target architecture and rationale.
· docs/roadmap.md — current phase and priorities.
· docs/decisions/ — all ADRs (read the most recent 3 before proposing
  structural changes).
· .github/instructions/*.md — path-scoped rules (loaded when editing
  matching paths).
· CONTRIBUTING.md — human-facing summary of §6–§9.

---

## 12. When in doubt

1. Re-read §2 (Golden rules).
2. Check docs/decisions/ for a prior decision.
3. Open an issue — do not guess.
4. Prefer the smaller change.
5. If you are an agent: stop and ask a human rather than proceed with
   assumptions that touch §10.

---

<!-- This file is versioned. Bump the "Last reviewed" line on any change. -->

Last reviewed: 2026-10-22 by Artoni

```

---
