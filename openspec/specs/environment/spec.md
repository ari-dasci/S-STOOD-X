# Environment Specification (introduced by c1-uv-migration)

> **Capability**: `environment` — **non-behavioral**. Captures reproducibility and invariant
> gates of the STOOD-X install + dev environment under uv. It does NOT describe product/library
> behavior (that stays in C2/C3). New domain — no prior spec in `openspec/specs/`, so this is a
> FULL spec, not a delta.

## Purpose

Guarantee that the STOOD-X environment is reproducible, locked, and importable from a clean
clone using uv — the contract that unblocks C2 (structural refactor) and C3 (OpenOOD extension
verification). It does NOT guarantee the package is functionally correct, only that the
environment resolves, locks, and the package imports at the env level.

## Requirements

### Requirement: uv-managed resolution and install

The system MUST resolve and install the editable package together with the `dev` dependency
group from a clean state via `uv sync`, with no resolver errors. The `openood` dependency
MUST resolve at the pinned commit `3c35632ee91b54b09d1f085d04f94744cece7d0b`, never at a
floating HEAD.

#### Scenario: Clean clone resolves editable package and dev group

- GIVEN a clean clone with no `.venv` and uv installed
- WHEN `uv sync` is run from the repository root
- THEN the editable package and the `dev` dependency group are installed
- AND the command exits 0 with no resolver conflict

#### Scenario: OpenOOD resolves at the pinned commit

- GIVEN `[tool.uv.sources]` pins `openood` to rev `3c35632ee91b54b09d1f085d04f94744cece7d0b`
- WHEN `uv sync` resolves dependencies
- THEN `openood` is fetched at exactly that commit, not at a moving HEAD

#### Scenario: Dev tooling excluded from the published wheel

- GIVEN `pytest`, `sphinx*`, `twine`, `tensorboard`, `torchsummary`, `IPython`, `ipywidgets`,
  `wget`, `pandoc`, `nbsphinx` are declared only under `[dependency-groups] dev`
- WHEN the project is built or published
- THEN the runtime wheel excludes dev-only tooling

### Requirement: Environment-level importability

The system MUST allow importing the top-level package in the uv-managed environment without an
`ImportError` attributable to the environment (missing or mismatched dependencies).

#### Scenario: Top-level package imports in the uv env

- GIVEN `uv sync` has completed successfully
- WHEN `uv run python -c "import STOODX"` is executed
- THEN the import completes without raising `ImportError` or `ModuleNotFoundError` caused by
  the environment

### Requirement: Test suite collects without import errors

The system MUST allow pytest to collect the full test suite without environment-induced import
errors. Test **execution** (running collected tests) is out of scope here — only collection.

> **DEFERRED — accepted as a known limitation in C1** (author decision, 2026-06-18).
> `uv run pytest --collect-only` currently raises collection errors on 5 modules
> (`test_featureVisualization`, `test_postProcessor_cifar10`, `test_postProcessor_cifar100`,
> `test_postProcessor_imagenet`, `test_postProcessor_imagenet200`). Root cause: `imgaug`
> (transitive via the pinned `openood` dependency's `draem_preprocessor`) uses `numpy.sctypes`,
> removed in NumPy 2.0; `imgaug` is unmaintained and STOOD-X pins `numpy>=2.1.2`. This latent
> conflict predates the uv migration and is surfaced explicitly by the lock. The requirement is
> retained verbatim below; it is NOT satisfied by C1 and MUST be re-evaluated in a follow-up
> change (C2/C3). See the "Known limitations" section of `README.md` for the user-facing note.
> C1's accepted scope is environment reproducibility + top-level importability only.

#### Scenario: pytest collection succeeds

- GIVEN `uv sync` has completed successfully
- WHEN `uv run pytest --collect-only` is run from the repository root
- THEN pytest collects the suite with no collection/import errors
- AND the outcome reflects collection only, not test pass/fail

### Requirement: Reproducibility artifacts committed

The system MUST commit `uv.lock` and `.python-version` so any clone reproduces the same
resolution. The committed lock MUST be CPU-only.

#### Scenario: Lockfile and python version are tracked

- GIVEN the change is applied
- THEN `uv.lock` exists at repo root and is tracked by git
- AND `.python-version` exists with content `3.10` and is tracked by git

#### Scenario: Lock is CPU-only; CUDA is opt-in only

- GIVEN the committed `uv.lock`
- THEN it resolves `torch` from PyPI (CPU build)
- AND the CUDA build is documented in README as an opt-in recipe
  (`UV_PYTHON_INDEX_URL=https://download.pytorch.org/whl/cu124`), NOT reflected in the lock

### Requirement: Scope boundary (non-goals)

This change SHALL NOT alter library/application/test logic, the build backend, the module
layout, or numerical/statistical behavior. Environment reproducibility is NOT a guarantee of
functional correctness.

#### Scenario: No structural or behavioral changes

- GIVEN the change diff
- THEN the setuptools build backend is unchanged (no hatchling)
- AND there is no src/ layout move, no `__init__.py` export fix, no OpenOOD-adapter extraction
- AND there is no backend swap, dependency removal, or public API change

#### Scenario: Functional correctness is not asserted by this change

- GIVEN a passing environment per the requirements above
- THEN functional and numerical correctness of STOODX, and OpenOOD extension behavior, are NOT
  asserted
- AND that guarantee is deferred to C3
