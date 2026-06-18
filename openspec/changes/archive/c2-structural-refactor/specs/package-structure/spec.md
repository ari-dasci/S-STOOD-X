# Package Structure Specification (introduced by c2-structural-refactor)

> **Capability**: `package-structure` — **non-behavioral**. Structural invariants (layout,
> naming, import surface) verifiable by static checks. Does NOT describe runtime/numerical/XAI
> behavior (deferred to C3). New domain — FULL spec. The C1 `environment` capability is
> **preserved, not modified**: env R2 (`import STOODX`) and the DEFERRED imgaug/numpy2 item
> (env R3) are referenced as preserved invariants.

## Purpose

Guarantee S-STOOD-X ships as a clean installable library: `snake_case` modules, `src/` layout,
a minimal public API that does not transitively import OpenOOD, and a private adapter
subpackage — all statically verifiable without compute-bound test execution.

## Requirements

### Requirement: src/ layout and package discovery

Package source MUST live under `src/STOODX/`. `pyproject.toml` MUST declare
`[tool.setuptools.packages.find] where = ["src"]` and MUST keep setuptools as the build backend.

#### Scenario: src layout resolves and installs

- GIVEN the change is applied and the old top-level `STOODX/` directory is removed
- WHEN `uv sync` runs from the repo root
- THEN the editable package is discovered under `src/STOODX/` and installed with exit 0
- AND `src/STOODX/_openood_adapter/` is discovered as a subpackage

### Requirement: snake_case module naming

Core modules MUST use PEP 8 `snake_case` filenames. No mixedCase module file MUST remain under
`src/STOODX/`.

#### Scenario: core modules are snake_case

- GIVEN the change is applied
- THEN `stoodx.py`, `feature_extractor.py`, and `feature_visualization.py` exist under `src/STOODX/`
- AND no mixedCase module file remains under `src/STOODX/`

### Requirement: Public class rename (FeatureExtractor)

The feature-extraction class MUST be named `FeatureExtractor`. The old `FeatureStractor` and
the typo `feature_estractor` MUST NOT appear in `src/`, `tests/`, or `README.md`.

#### Scenario: class is FeatureExtractor with no residual typo

- GIVEN the change diff
- WHEN grepping `src/`, `tests/`, `README.md` for `feature_estractor` or `FeatureStractor`
- THEN zero matches are returned
- AND `class FeatureExtractor` is defined in `src/STOODX/feature_extractor.py`

### Requirement: Public API surface and no openood pull-in

`__init__.py` MUST export exactly `STOODX` and `FeatureExtractor` via `__all__`; it MUST NOT
export `FeatureExplanation` or `STOODXPostprocessor` at top level. Importing `STOODX` MUST NOT
transitively import `openood` — this PRESERVES env R2 and MUST NOT regress it.

#### Scenario: two-symbol public API

- GIVEN the change is applied
- WHEN `uv run python -c "from STOODX import STOODX, FeatureExtractor"` runs
- THEN it succeeds without `ImportError`
- AND `STOODX.__all__ == ["STOODX", "FeatureExtractor"]`

#### Scenario: top-level import does not pull openood

- GIVEN the change is applied
- WHEN `uv run python -c "import STOODX; import sys; assert 'openood' not in sys.modules"`
- THEN the assertion passes
- AND `FeatureExplanation` and `STOODXPostprocessor` are absent from `STOODX.__all__`

### Requirement: OpenOOD adapter privatization

The adapter MUST live at `src/STOODX/_openood_adapter/postprocessor.py`. The subpackage MUST
be underscore-prefixed and MUST carry a module docstring marking it internal / not public API.
The class name `STOODXPostprocessor` MUST be unchanged.

#### Scenario: adapter is private and self-documented

- GIVEN the change is applied
- THEN `src/STOODX/_openood_adapter/postprocessor.py` exists
- AND its module docstring states it is internal / not public API
- AND `class STOODXPostprocessor` remains defined there

### Requirement: Exhaustive rename (no residual old names)

No residual old module path, old class name, or typo MUST remain in source, tests, or README.

#### Scenario: grep returns zero hits

- GIVEN the change diff
- WHEN running
  `grep -rn "featureStractor\|featureVisualization\.py\|STOODXPostprocessor\.py\|FeatureStractor\|feature_estractor" src/ tests/ README.md`
- THEN the command returns zero matches

### Requirement: Import correctness preserved

pytest collection MUST collect `test_feature_extractor.py` with no import errors from this
change. The 5 OpenOOD-pulling modules MAY still fail collection on the DEFERRED imgaug/numpy2
issue (env R3, same root cause as C1), but NOT on new import typos.

#### Scenario: rename-correct collection

- GIVEN the change is applied and `uv sync` succeeded
- WHEN `uv run pytest --collect-only` runs
- THEN `test_feature_extractor.py` is collected without `ImportError`
- AND any errors on the 5 OpenOOD-pulling modules cite imgaug/numpy2, not
  `FeatureStractor`/`feature_estractor`/missing-module typos

### Requirement: README accuracy

The README quick-start MUST reference `FeatureExtractor` (not `FeatureStractor`), and the
project-structure tree MUST reflect `src/STOODX/` paths.

#### Scenario: README matches new API and layout

- GIVEN the change is applied
- THEN the quick-start shows `from STOODX import STOODX, FeatureExtractor`
- AND the structure tree lists modules under `src/STOODX/`, including `_openood_adapter/`

### Requirement: Scope boundary (non-goals)

This change SHALL NOT alter statistics, numerics, or XAI algorithm logic. It SHALL NOT decouple
`feature_visualization` from the OpenOOD adapter, fix the imgaug/numpy2 collection issue (C3),
switch the build backend, or add linter/type-checker/CI configuration.

#### Scenario: only structural changes

- GIVEN the change diff
- THEN no statistics/numerics/XAI function body is modified
- AND the `feature_visualization -> _openood_adapter.postprocessor` import edge is preserved
- AND setuptools remains the build backend with no new linter/CI config
