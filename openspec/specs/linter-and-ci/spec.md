# Linter and CI Specification (introduced by c5-linter-and-ci)

> **Capability**: `linter-and-ci` — **non-behavioral** tooling/automation domain. `ruff`
> lint/format + a GitHub Actions CI gating push/PR to `main` with hermetic tests. New domain —
> FULL spec, not a delta. SHALL NOT alter STOOD-X numerical behavior.

## Purpose

Guard S-STOOD-X against style drift and regressions with one fast, hermetic CI gate: `ruff`
enforces lint/format and a 3-job workflow runs the two GPU/data-free tests on every push/PR to
`main`.

## Requirements

### Requirement: Ruff is a dev-only dependency

`ruff` MUST be declared under `[dependency-groups] dev` in `pyproject.toml` and SHALL NOT
appear among runtime dependencies. `uv sync --dev` MUST install it from the locked resolution.

#### Scenario: ruff installs via the dev group

- GIVEN a clean clone with uv installed
- WHEN `uv sync --dev` runs from the repository root
- THEN `ruff` is installed and `uv run ruff --version` exits 0
- AND `ruff` is locked in `uv.lock` within the `dev` group

#### Scenario: ruff is excluded from the published wheel

- GIVEN `ruff` is declared only under `[dependency-groups] dev`
- WHEN the project is built into a wheel
- THEN `ruff` is not among the wheel's runtime dependencies

### Requirement: Ruff configuration is declared

`[tool.ruff]` MUST set `line-length = 120`, `target-version = "py310"`, and select `E, F, W,
I`. `E501` MAY be ignored (absorbed by `line-length` + `ruff format`). Test files importing
networks as registration fixtures MUST receive a per-file `F401` ignore.

#### Scenario: Config drives the declared rule set

- GIVEN the `[tool.ruff]` section is present in `pyproject.toml`
- WHEN the section is read
- THEN `line-length = 120` and `target-version = "py310"` are set
- AND the selected rule set includes `E`, `F`, `W`, and `I`

#### Scenario: Test registration imports are exempted, not deleted

- GIVEN a test file whose unused imports are suspected registration fixtures
- WHEN ruff analyzes it
- THEN a per-file `F401` ignore (or targeted `# noqa: F401`) suppresses the warning
- AND the import statements themselves remain in the file

### Requirement: Ruff passes on source and tests after safe fixes

After safe autofixes applied to `src/` ONLY, `uv run ruff check src/` MUST exit 0. A one-time
`ruff format` MUST then be applied across `src/` and `tests/` so `uv run ruff format --check
src/ tests/` exits 0. Fixes SHALL be limited to `W605` (raw-string LaTeX docstrings), `F541`,
`F841`, `E711`, and src-only `F401`; formatting SHALL be whitespace-only. Test `F401`s SHALL
NOT be blanket-removed.

#### Scenario: src/ passes lint after safe autofix

- GIVEN the safe `ruff check --fix` has been applied to `src/`
- WHEN `uv run ruff check src/` runs
- THEN the command exits 0

#### Scenario: Format check passes after one-time format

- GIVEN the one-time `ruff format` has been applied to `src/` and `tests/`
- WHEN `uv run ruff format --check src/ tests/` runs
- THEN the command exits 0

#### Scenario: W605 is fixed via raw strings, not removal

- GIVEN `W605` invalid-escape violations in LaTeX-bearing docstrings of `src/STOODX/*.py`
- WHEN the safe fix is applied
- THEN each affected docstring becomes a raw string (`r"""..."""`)
- AND the LaTeX content is preserved verbatim

### Requirement: Test F401 policy is surgical suppression

Unused-import warnings in `tests/` MUST be resolved by targeted `# noqa: F401` or per-file
ignore, NEVER by blind deletion — OpenOOD network imports may be intentional registration
fixtures.

#### Scenario: Registration imports are preserved

- GIVEN a test module with F401-flagged network imports
- WHEN the F401 is resolved
- THEN the suppression is targeted (per-file ignore or `# noqa: F401`)
- AND no registration import is deleted

### Requirement: CI workflow gates pushes and PRs to main

`.github/workflows/ci.yml` MUST trigger on `push` and `pull_request` against `main`. It MUST
define three jobs — `lint`, `format-check`, `test` — each using `astral-sh/setup-uv@v3` with
`enable-cache: true` on a single Python 3.10.

#### Scenario: Workflow triggers and runs all three jobs

- GIVEN `.github/workflows/ci.yml` is committed
- WHEN a push or pull_request targets `main`
- THEN the `lint`, `format-check`, and `test` jobs all run
- AND all three use `astral-sh/setup-uv@v3` with `enable-cache: true` on Python 3.10

### Requirement: CI test job is hermetic and fast

The CI `test` job MUST run ONLY `tests/test_public_api.py` and
`tests/test_postprocessor_contract.py` via `uv run pytest`, both GPU/data-free. The full
`eval_ood` pipeline SHALL NOT run in CI.

#### Scenario: Only hermetic tests run in CI

- GIVEN the `test` job definition
- WHEN the job executes
- THEN it runs exactly the two hermetic test modules via `uv run pytest`
- AND no GPU, dataset, or `eval_ood` pipeline is required

### Requirement: README documents lint and CI commands

`README.md` MUST document how to run `uv run ruff check` and `uv run ruff format` locally, and
MUST state which tests CI runs.

#### Scenario: Lint/format usage is documented

- GIVEN `README.md` after the change
- WHEN the README is inspected
- THEN it documents `uv run ruff check` and `uv run ruff format`
- AND it states that CI runs the two hermetic tests on Python 3.10

### Requirement: Non-goal guards

This change SHALL NOT introduce `mypy`, a multi-Python matrix, or any numerical/statistical
behavior change. `W605` fixes SHALL be docstring-only; formatting SHALL be whitespace-only.

#### Scenario: Deferred and excluded scope is honored

- GIVEN the change diff
- WHEN the diff is reviewed
- THEN no mypy job, mypy dependency, or multi-Python matrix is introduced
- AND no statistical/numerical result of STOODX is altered
