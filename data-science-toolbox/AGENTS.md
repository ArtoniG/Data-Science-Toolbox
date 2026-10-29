# AGENTS.md ds-toolbox
> Read this file first. It is the single source of truth for how agents
> (Copilot, Cursor, Claude, Aider) must work in this repository.
> If this file conflicts with any other doc, this file wins.
> If a rule here is wrong, open an issue titled `AGENTS.md: <rule>`.

---

## 1. What this repository is

`ds-toolbox` is an installable Python library of reusable, production-grade
utilities for data science and ML engineering, with deep specialization in
credit risk / financial modeling and growing coverage in MLOps and agentic
systems.

- **Owner:** Guilherme Artoni
- **License:** see `LICENSE`
- **Current version:** 0.1.0
- **Python support:** >=3.10 (tests run on 3.10 and 3.11)
- **PyPI package:** `ds-toolbox`
- **Import name:** `ds_toolbox`
- **Purpose:** reusable IP for a solo data-science consultancy. Every module
  must be defensible as "something a 6-year DS would pay to reuse."

Non-goals (do not add these to core):
- Application code, dashboards, or notebooks at the repo root.
- Client-specific or PII-bearing content.
- Vendored third-party libraries.

---

## 2. Golden rules (non-negotiable)

1. **Never break the public API silently.** Any change to a public symbol
   (`__all__`, public function signature, class attribute) requires a
   CHANGELOG entry and, if pre-1.0, a note in the PR description.
2. **`src/` layout is mandatory.** No top-level modules.
3. **Every public module has a test.** PRs that add code without tests are
   rejected by CI.
4. **Every public symbol has a docstring** (Google style). Private helpers
   (`_name`) do not require docstrings but should be self-explanatory.
5. **No new runtime dependency** without an explicit ADR in
   `docs/decisions/`. Optional dependencies go in `pyproject.toml` extras.
6. **No secrets, tokens, or credentials** in any file. Use `config/` or
   environment variables.
7. **No agent may modify CI workflows, `pyproject.toml`, `AGENTS.md`, or
   `docs/decisions/` without explicit human approval in the PR.**
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
│   ├── init.py            public re-exports only; no logic
│   ├── py.typed               PEP 561 marker; keep present
│   ├── core/                  base classes, exceptions, shared types
│   ├── config/                Pydantic Settings; env + YAML loading
│   ├── io/                    readers/writers (parquet, csv, cloud)
│   ├── validation/            pandera schemas, data contracts
│   ├── eda/                   profiling, summaries
│   ├── governance/            model cards, quality gates, reason codes
│   ├── observability/         logging, decorators, (future) tracing
│   ├── metrics/               supervised, unsupervised, stability, genai
│   ├── transformers/          sklearn-compatible feature transformers
│   ├── mlops/                 tracking, registry, serving, pipelines
│   ├── agentic/               tools, memory, tracing, evals
│   └── viz/                   plots, shared theme
├── tests/                     mirrors src/ one-to-one
├── examples/                  runnable notebooks/scripts per use case
├── docs/                      mkdocs site + ADRs + architecture
├── .github/
│   ├── workflows/             ci.yml, publish.yml
│   ├── prompts/               PIVS workflows (see §9)
│   ├── instructions/          path-scoped rules
│   └── ISSUE_TEMPLATE/
├── pyproject.toml, ruff.toml, .pre-commit-config.yaml, Makefile
├── AGENTS.md, CHANGELOG.md, CONTRIBUTING.md, SECURITY.md, LICENSE
└── README.md

```

---

## 4. Module dependency rules (enforced by import-linter)

Imports flow downward only. If an arrow is missing, it is forbidden.

```

agentic    → mlops → metrics → transformers → validation → io → config → core
viz        → core
governance → metrics → core
observability → core
eda        → io → validation → core

```

Additional rules:
- `core/` imports nothing from `ds_toolbox`.
- `config/` imports only `core/`.
- No module imports `agentic/` except itself and `__init__.py`.
- Circular imports are always a bug — refactor, never suppress.

If you believe a new edge is needed, open an ADR first.

---

## 5. Commands (run in this order)

```bash
# Setup
pip install -e ".[dev]"

# Lint + format
ruff check src/ tests/
ruff format --check src/ tests/

# Type check
mypy src

# Import rules
lint-imports

# Tests
pytest

# Security
bandit -q -r src/

# Everything at once (what CI runs)
make check
```

If you add a command, add it to both Makefile and CI. Never one alone.

---

6. Code conventions

· Style: ruff (config in pyproject.toml). Do not reformat unrelated lines.
· Types: full annotations on public functions. mypy strict mode enabled.
· Docstrings: Google style. Public functions need Args, Returns,
  Raises (if applicable), and an Example for anything non-obvious.
· Errors: raise subclasses of ds_toolbox.core.exceptions.DSToolboxError.
  Never raise bare Exception. Never swallow exceptions silently.
· Logging: use ds_toolbox.observability.logger.get_logger(__name__).
  Never print(). Never logging.getLogger directly.
· Config: read via ds_toolbox.config, never os.environ directly
  outside config/.
· DataFrames: prefer polars where performance matters; pandas is
  acceptable at API boundaries. Document the choice per module.
· Determinism: any function with randomness accepts random_state.
· No global mutable state. Pass dependencies explicitly.

---

7. Testing conventions

· Location: tests/<module_path>/test_<file>.py, mirroring src/.
· Framework: pytest. Fixtures in tests/conftest.py and per-module.
· Structure: Arrange / Act / Assert, separated by blank lines.
· Naming: test_<behavior>_<condition>_<expected>.
· Required per public symbol: one happy path and one error/edge test.
· Coverage target: 80% lines (see [tool.coverage.report]). Enforced by
  make coverage-gate once the foundation stabilizes; until then, make
    check runs coverage as report-only.
· Markers: @pytest.mark.integration for real-service tests (skipped
  locally, run in CI nightly). @pytest.mark.slow for >1s tests.
· No network in unit tests. No real credentials anywhere.

---

8. Documentation

· Every public module has a matching page under docs/api/ (generated by
  mkdocstrings from docstrings — keep docstrings accurate).
· Architectural changes require an ADR in docs/decisions/ADR-NNNN-<slug>.md.
· User-facing changes require an entry in CHANGELOG.md under Unreleased.
· Examples in examples/<use_case>/ must be runnable end-to-end.

---

9. Ship cycle — Plan → Implement → Verify → Ship (PIVS)

Every non-trivial change follows PIVS. Do not collapse steps.

1. Plan. Open an issue using the Feature template. Fill the ## Plan
   section fully: approach, files to touch, tests to add, risks. The plan
   is reviewed and approved before any code is written. Trivial changes
   (typos, single-line docs) may skip the issue, but the commit must
   reference the eventual PR.
2. Implement. One module per PR. Every commit references the issue.
   Stay under 400 changed lines; if the plan exceeds that, split the issue
   in the Plan step, not later.
3. Verify. CI must be green on all gates: ruff, mypy, pytest with
   coverage, import-linter, bandit. Local make check output must match CI.
4. Ship. Merge to main, add CHANGELOG under ## Unreleased, then run
   /release for version bump, tag, and publish.

Prompt files under .github/prompts/ encode this cycle. Agents must
prefer them over improvising:

· /scaffold-module — steps 1–2 for a new module
· /add-public-api — step 2 for any public symbol change
· /debug-ci — step 3 recovery loop when CI is red
· /release — step 4

GitHub workflow for PIVS:

· Issue created on phone (dictate it — see §13)
· Plan reviewed in the issue thread
· PR opened from the issue; CI runs; human approves
· Squash-merge on green
· /release runs on main only

---

10. What agents MUST NOT do

· Rewrite files outside the scope of the current issue.
· Introduce new dependencies, modules, or top-level packages without an
  ADR and human approval.
· Delete or weaken tests to make CI pass.
· Change pyproject.toml, AGENTS.md, .github/workflows/*, or
  docs/decisions/* without human approval.
· Generate placeholder code with TODO, pass, or NotImplementedError
  and leave it unmarked in the PR description.
· Fabricate API behavior — if unsure, open an issue and stop.
· Commit secrets, API keys, or environment dumps.
· Refactor while fixing a bug. Separate PRs.
· Make performance claims without a benchmark in examples/benchmarks/.
· Run git push, git tag, pip install, or any dependency-changing
  command without explicit human approval in the current turn.
· Approve or merge their own PR. Agents may summarize; a human reviews.
· Modify .vscode/settings.json or .pre-commit-config.yaml without
  approval.

---

11. Deeper context

Consult these before non-trivial work:

· docs/architecture.md — target architecture and rationale.
· docs/roadmap.md — current phase and priorities.
· docs/decisions/ — all ADRs (read the most recent 3 before proposing
  structural changes).
· .github/instructions/*.md — path-scoped rules.
· CONTRIBUTING.md — human-facing summary of §6–§9.

---

12. When in doubt

1. Re-read §2 (Golden rules).
2. Check docs/decisions/ for a prior decision.
3. Open an issue — do not guess.
4. Prefer the smaller change.
5. If you are an agent: stop and ask a human rather than proceed with
   assumptions that touch §10.

---

13. Permission modes and mobile workflow

Permission defaults (VS Code / Copilot)

Action Mode
Edit files in src/ and tests/ Auto-approved
Format on save (ruff via VS Code) Auto
Run make lint, make type, make test, make imports Auto-approved
git commit Auto-approved
git push, git tag Ask
pip install, dependency changes Ask
Modify pyproject.toml, CI, AGENTS.md, docs/decisions/ Ask + PR review
Delete files Ask
Anything under ~/.ssh, ~/.aws, .env* Deny

Mirror this in .vscode/settings.json where VS Code supports it. Where a
key is absent, VS Code defaults to "ask" — safe.

Mobile workflow

· Author issues by voice. Use OS keyboard dictation (iOS/Android)
  inside GitHub Mobile. The Feature template's four required fields map
  to a 3–5 minute dictation.
· Review PRs on GitHub Mobile. Diff view + CI status is sufficient for
  the review gate. Do not edit code on mobile.
· Run agent sessions in Codespaces (120 core-hours/month free). Treat
  Codespaces as the desktop; the phone as the review surface.
· Never git push from the phone. Push from Codespaces after
  make check is green, so CI is confirmation, not discovery.

---
