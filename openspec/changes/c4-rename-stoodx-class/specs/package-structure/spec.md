# Delta for package-structure

> Capability: `package-structure` — **non-behavioral** mechanical class rename. Package
> `STOODX` stays; the detector class `STOODX` becomes `STOODXDetector`, removing the
> package/class name shadowing. No numerics, no API shape change beyond the single rename.

## ADDED Requirements

### Requirement: No residual bare `STOODX` class references

Post-rename, no in-source reference to a class named exactly `STOODX` MUST remain in `src/`,
`tests/`, or `README.md`. The package directory `src/STOODX/`, the `STOODXPostprocessor`
sibling class, and the `stoodx.py` module filename MUST remain unchanged and MUST be excluded
from this check. The class name `STOODXDetector` MAY be referenced anywhere.

#### Scenario: grep for the bare class returns zero hits

- GIVEN the change diff is applied
- WHEN running `grep -rnE '\bSTOODX\b' src/ tests/ README.md`
- THEN every match is one of: the package dir path, `STOODXPostprocessor`, `stoodx.py`, or `STOODXDetector`
- AND no isolated `class STOODX` or `STOODX(` instantiation remains

### Requirement: Public API import smoke test

A smoke test at `tests/test_public_api.py` MUST assert that the renamed class imports cleanly
from the package and resolves to its defining module. The test MUST NOT require GPU, datasets,
or pretrained checkpoints (closing the coverage gap from explore §3c).

#### Scenario: smoke test passes without compute

- GIVEN the change is applied and `uv sync` succeeded
- WHEN `uv run pytest tests/test_public_api.py` runs
- THEN `from STOODX import STOODXDetector` succeeds
- AND `STOODXDetector.__module__ == "STOODX.stoodx"`
- AND the test completes without GPU/dataset access

## MODIFIED Requirements

### Requirement: Public API surface and no openood pull-in

`__init__.py` MUST export exactly `STOODXDetector`, `FeatureExtractor`, and `FeatureExplanation`
via `__all__`; it MUST NOT export `STOODXPostprocessor` at top level. The package directory
remains `STOODX` and the detector class becomes `STOODXDetector`, so `import STOODX` binds the
package without shadowing the class on its own surface. Importing `STOODX` MUST NOT transitively
import `openood` — this PRESERVES env R2 and MUST NOT regress it WITH all three exported symbols
present.

(Previously: `__all__` exported `STOODX` (the class), which was shadowed by the package name on
its own surface; the class is renamed to `STOODXDetector`.)

#### Scenario: three-symbol public API

- GIVEN the change is applied
- WHEN `uv run python -c "from STOODX import STOODXDetector, FeatureExtractor, FeatureExplanation"` runs
- THEN it succeeds without `ImportError`
- AND `STOODX.__all__ == ["STOODXDetector", "FeatureExtractor", "FeatureExplanation"]`
- AND `STOODXPostprocessor` is absent from `STOODX.__all__`
- AND `STOODXDetector` is not shadowed by the package on its own surface

#### Scenario: top-level import does not pull openood (3 symbols)

- GIVEN the change is applied
- WHEN `uv run python -c "import STOODX; from STOODX import STOODXDetector; import sys; assert 'openood' not in sys.modules"`
- THEN the assertion passes
- AND `openood` is absent from `sys.modules` WITH all three exported symbols imported

### Requirement: README accuracy

The README quick-start MUST reference `STOODXDetector` (the renamed class, not the shadowed
bare `STOODX`) and `FeatureExtractor` (not `FeatureStractor`). The project-structure tree MUST
reflect `src/STOODX/` paths.

(Previously: quick-start imported the shadowed bare `STOODX` class from `STOODX`; it now imports
`STOODXDetector`.)

#### Scenario: README matches new API and layout

- GIVEN the change is applied
- THEN the quick-start shows `from STOODX import STOODXDetector` (and `FeatureExtractor` where relevant)
- AND no README line imports or instantiates the bare class `STOODX`
- AND the structure tree lists modules under `src/STOODX/`, including `_openood_adapter/`
