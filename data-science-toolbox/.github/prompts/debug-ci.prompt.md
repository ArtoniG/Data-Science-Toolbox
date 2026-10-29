---
mode: 'agent'
description: 'Systematically reproduce and fix a CI failure with a proving test'
tools: ['codebase', 'editFiles', 'runCommands', 'search', 'githubRepo']
---

You are debugging a CI failure. Do not skip steps. Reference AGENTS.md §2
rule 10 (separate bugfix from refactor).

**Inputs (ask if missing):**
- `run_url` — GitHub Actions run URL or job name
- `module` — the module the failure belongs to

**Step 1 — Classify**
Fetch logs (`gh run view <id> --log-failed`). Classify the failure:
`lint` / `type` / `test` / `import-linter` / `security` / `flaky`.
Write a one-paragraph root-cause hypothesis.

**Step 2 — Reproduce locally**
Run the exact failing command from AGENTS.md §5. If it does not fail
locally, it is flaky or env-dependent: report and stop. Do not fix blind.

**Step 3 — Proving test (test/type/behavior failures only)**
Add `test_<behavior>_regression_<short-id>` in `tests/<module_dir>/`.
It must FAIL on current code and PASS after the fix.
For lint/type/import-linter/security failures, skip — the failing gate is
already the proof.

**Step 4 — Fix forward, minimally**
Only the lines needed. No reformat of unrelated code. No refactor while
fixing. If a refactor is warranted, open a separate issue and link it.

**Step 5 — Verify**
```bash
make check
```

Then re-run the exact command that failed in CI.

Step 6 — CHANGELOG
If released behavior was affected, add ### Fixed under ## Unreleased
with the issue reference.

Report back: classification, root cause (1–2 sentences), files changed,
proving test name, recommended follow-up issue title, whether a public API
changed (if yes, escalate per §2 rule 1).

Do not: delete/skip tests, add # type: ignore without justification,
edit import-linter contracts, or push if make check is red.