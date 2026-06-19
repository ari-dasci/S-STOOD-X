# Proposal: c5-linter-and-ci

## Intent

S-STOOD-X ships zero linting and zero CI today: no `.github/` directory, no formatter, no
linter config (`quality.linter/formatter/type_checker = "not configured"`). Regressions and
style drift land unguarded. As the **last PR in the C1–C5 stacked-to-main chain**, C5 adds
`ruff` (lint + format) and a pragmatic, fast GitHub Actions CI that runs the two hermetic
tests — catching regressions, enforcing a baseline style, and protecting the public-API smoke
+ postprocessor contract without requiring GPU/datasets. `mypy` is deferred.

## Scope

### In Scope
- Add `ruff` to `[dependency-groups] dev` in `pyproject.toml` (lockfile reproducibility — currently runs only via transitive/global resolution).
- Add `[tool.ruff]` config: `line-length = 120`, `target-version = "py310"`, select `E,F,W,I` (isort), ignore noisy `E501` (absorbed by line-length), per-file `F401` ignore for test files whose openood network imports are suspected registration fixtures.
- Apply **safe `ruff check --fix` to `src/` ONLY** on REAL+MEDIUM categories: `W605` (raw-string LaTeX docstrings in `stoodx.py`), `F541`, `F841`, `E711`, src-only `F401`.
- One-time **`ruff format` across `src/` + `tests/`** (rewrites 13/14 files — broad but semantically safe churn; explicitly accepted, called out).
- **Do NOT blind-fix test `F401`**: targeted `# noqa: F401` or per-file-ignore (registration imports may be intentional).
- Add `.github/workflows/ci.yml`: 3 jobs — `lint` (`ruff check`), `format-check` (`ruff format --check`), `test` (`uv sync --dev`; run ONLY `tests/test_public_api.py` + `tests/test_postprocessor_contract.py`, both GPU/data-free). Single Python 3.10, `astral-sh/setup-uv@v3` + `enable-cache: true`. Trigger on `push` + `pull_request` to `main`.
- Update README: document `uv run ruff check`, `uv run ruff format`, `uv run pytest tests/test_public_api.py`.

### Out of Scope
- mypy strict (deferred to C6+; no advisory job in C5 by default).
- Full `eval_ood` in push CI (needs GPU + gitignored checkpoints/datasets — stays manual).
- Fixing pre-existing `FeatureStractor` typo / README method-body drift (separate docs pass).
- Multi-Python matrix (3.11/3.12 unverified on the openood/libmr/torch pin — defer).
- E501-at-88 / separate W29x enforcement (relaxed via `line-length=120` + format).
- Coverage gate (`pytest-cov` not declared).
- Resolving the remaining ~104 noisy violations beyond what format absorbs.

## Capabilities

### New Capabilities
- `linter-and-ci`: style/format enforcement (`ruff`) and a GitHub Actions CI with hermetic
  test gating. Becomes `openspec/specs/linter-and-ci/spec.md`.

### Modified Capabilities
- `environment`: extends the dev-tooling domain — `ruff` added to `[dependency-groups] dev`
  (must stay excluded from the published wheel); CI codifies `uv sync --dev` + `import STOODX`
  into an automated gate. Delta spec in the change folder.

## Approach

Incremental, dependency-ordered: (1) ruff config in `pyproject.toml`; (2) safe `src/`-only
`ruff check --fix` (W605 before format to avoid docstring rework); (3) one-time
`ruff format` on `src/`+`tests/`; (4) `.github/workflows/ci.yml`; (5) README. F401 handled
surgically, never blanket.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `pyproject.toml` | Modified | +ruff dev dep, +`[tool.ruff]` |
| `src/STOODX/*.py` | Modified | safe fixes (W605/F541/F841/E711/src-F401) + format |
| `tests/**/*.py` | Modified | format only; `# noqa: F401` per-file where needed |
| `.github/workflows/ci.yml` | New | lint + format-check + hermetic test jobs |
| `README.md` | Modified | lint/CI usage docs |
| `uv.lock` | Modified | ruff locked into dev group |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| `ruff format` churn broad (13/14 files) | High | Last PR in chain; within 800-line budget; semantically safe |
| CI `test` job needs heavy `uv sync --dev` (torch+openood+libmr build) under CI | Med | `setup-uv@v3` + `enable-cache: true`; hermetic tests are sub-second |
| `F401` per-file-ignore masks a real unused import | Low | Acceptable tradeoff vs breaking intentional registration imports |
| openood/libmr build fails on CI runner | Med | Falls back to `setup-uv` cache + existing `extra-build-dependencies` |

## Rollback Plan

Revert the PR. No database/data/numerical state is touched. CI simply stops running; code
returns to unlinted state. `ruff` removal from dev group + lock revert restores prior
resolution. No downstream PRs depend on C5 (it is the chain terminus → `main`).

## Dependencies

- C1–C4 merged (uv migration, src/ layout, OpenOOD extension, public API).
- GitHub Actions runners + `astral-sh/setup-uv@v3`.

## Success Criteria

- [ ] `uv run ruff check src/ tests/` exits 0 (safe categories fixed; F401 suppressed with rationale).
- [ ] `uv run ruff format --check src/ tests/` exits 0.
- [ ] CI runs green on PR + push: `lint`, `format-check`, `test` all pass on Python 3.10.
- [ ] `test` job runs only the 2 hermetic tests; both pass without GPU/datasets.
- [ ] `ruff` is locked in `uv.lock` dev group and excluded from the published wheel.
- [ ] No numerical/statistical behavior change (W605 = docstring-only; format = whitespace).
