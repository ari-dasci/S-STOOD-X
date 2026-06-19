# Tasks: Linter and CI

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | 500-700 (format churn ~300-400 whitespace-only, config+CI+fixes ~200-300 semantic) |
| 800-line budget risk | Medium |
| Chained PRs recommended | No — last PR in chain, single PR appropriate |
| Suggested split | Single PR — semantic changes are small; format churn is whitespace-only |
| Delivery strategy | force-chained stacked-to-main |
| Chain strategy | stacked-to-main (terminal PR, base = main) |

Decision needed before apply: No
Chained PRs recommended: No
Chain strategy: stacked-to-main
400-line budget risk: Medium

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | Ruff config + dev dep + lockfile | PR 5 | pyproject.toml + uv.lock |
| 2 | Safe src/ autofixes | PR 5 | W605/F541/F841/E711/src-F401 |
| 3 | Test F401 suppression | PR 5 | Per-file noqa in test files |
| 4 | One-time format | PR 5 | Broad whitespace churn (expected ~300-400 lines) |
| 5 | CI workflow | PR 5 | .github/workflows/ci.yml (new file) |
| 6 | README + verification | PR 5 | Docs + exit-0 gates |

## Phase 1: Ruff Configuration

- [ ] 1.1 Add `ruff` to `[dependency-groups] dev` in `pyproject.toml`
- [ ] 1.2 Add `[tool.ruff]` section: `line-length=120`, `target-version="py310"`, select `E,F,W,I`
- [ ] 1.3 Add `[tool.ruff.lint.per-file-ignores]` for test files with registration imports (`F401`)
- [ ] 1.4 Run `uv lock` to refresh lockfile; verify `uv run ruff --version` exits 0

## Phase 2: Safe Source Fixes

- [ ] 2.1 Run `uv run ruff check src/ --fix` (targets W605, F541, F841, E711, F401)
- [ ] 2.2 Review each autofix diff — verify W605 → raw strings preserve LaTeX verbatim
- [ ] 2.3 Confirm `uv run ruff check src/` exits 0

## Phase 3: Test F401 Surgical Suppression

- [ ] 3.1 Run `uv run ruff check tests/` to identify F401 violations in test files
- [ ] 3.2 Add targeted `# noqa: F401` on lines with suspected registration imports only
- [ ] 3.3 Confirm `uv run ruff check tests/` exits 0 — no imports deleted

## Phase 4: One-Time Format

- [ ] 4.1 Run `uv run ruff format src/ tests/`; review diff (expect whitespace churn across ~14 files)
- [ ] 4.2 Confirm `uv run ruff format --check src/ tests/` exits 0

## Phase 5: CI Workflow

- [ ] 5.1 Create `.github/workflows/ci.yml` with 3 jobs: `lint`, `format-check`, `test`
- [ ] 5.2 Set trigger: `push` + `pull_request` on `main`
- [ ] 5.3 Use `astral-sh/setup-uv@v3` with `enable-cache: true`, Python 3.10, `uv sync --dev`
- [ ] 5.4 `test` job runs only `test_public_api.py` + `test_postprocessor_contract.py`
- [ ] 5.5 Validate YAML syntax

## Phase 6: README and Verification

- [ ] 6.1 Add "Linting" section to README: `uv run ruff check`, `uv run ruff format`, CI description
- [ ] 6.2 Verify pytest collection clean (318 tests)
- [ ] 6.3 Smoke + contract tests green (`pytest tests/test_public_api.py tests/test_postprocessor_contract.py`)
- [ ] 6.4 Final gate: `ruff check` exit 0 + `ruff format --check` exit 0

## Phase 7: PR #5 Creation

- [ ] 7.1 Commit grouping: 1 commit config+lock, 1 commit src-fixes, 1 commit test-suppress, 1 commit format, 1 commit ci+readme (OR single commit — both acceptable for terminal PR)
- [ ] 7.2 Create PR #5 stacked-to-main, base = main (C4 already merged)
