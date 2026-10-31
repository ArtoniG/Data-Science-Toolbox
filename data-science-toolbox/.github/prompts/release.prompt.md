---
mode: 'agent'
description: 'Cut a release: bump version, finalize CHANGELOG, tag, publish'
tools: ['codebase', 'editFiles', 'runCommands', 'search']
---

You are cutting a release of `ds-toolbox`. Stop if any pre-flight fails.

**Inputs (ask if missing):**
- `version_bump` — `patch` | `minor` | `major` (default: `minor` while pre-1.0)
- `reason` — one sentence

**Pre-flight:**
- `git status` clean, branch is `main`, up to date with `origin/main`.
- Latest CI run on `main` is green.
- `CHANGELOG.md` `## Unreleased` is non-empty.

**Steps:**

1. Compute next version per semver. Pre-1.0 rule: breaking → minor bump.

2. Finalize `CHANGELOG.md`:
   - `## Unreleased` → `## [<version>] - <YYYY-MM-DD>`
   - insert a fresh empty `## Unreleased` above
   - update link refs at the bottom

3. Bump `[project].version` in `pyproject.toml`. Nothing else.

4. Commit + tag:
```bash
git add CHANGELOG.md pyproject.toml
git commit -m "chore(release): v<version>"
git tag -a v<version> -m "v<version>"
git push origin main --follow-tags
```

5. Verify publish: watch publish.yml (triggered by tag). Confirm the
   GitHub Release, wheel and sdist on PyPI. Report URLs.

Report back: old→new version, CHANGELOG header, commit SHA, tag,
publish conclusion, PyPI URL.

Do not: tag a non-main commit, skip CHANGELOG, force-push, or bump
version from a feature branch.