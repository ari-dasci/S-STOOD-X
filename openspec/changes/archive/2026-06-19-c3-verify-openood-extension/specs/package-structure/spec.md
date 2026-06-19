# Delta for package-structure

> **Capability**: `package-structure` — MODIFIED by `c3-verify-openood-extension`.
> R4 (public API surface) moves from 2 symbols to 3 (`FeatureExplanation` added). The
> `feature_visualization -> _openood_adapter.postprocessor` import edge (preserved in C2's scope
> boundary as a non-goal) is now DECOUPLED via `TYPE_CHECKING`. The no-openood-pull-in invariant
> (env R2) MUST still hold WITH all three exported symbols present. Other requirements untouched.

## MODIFIED Requirements

### Requirement: Public API surface and no openood pull-in

`__init__.py` MUST export exactly `STOODX`, `FeatureExtractor`, and `FeatureExplanation` via
`__all__`; it MUST NOT export `STOODXPostprocessor` at top level. Importing `STOODX` MUST NOT
transitively import `openood` — this PRESERVES env R2 and MUST NOT regress it WITH all three
exported symbols present.
(Previously: C2 exported exactly `STOODX` and `FeatureExtractor` (2 symbols); `FeatureExplanation`
was withheld to keep the no-openood invariant while `feature_visualization` eagerly imported the
adapter. C3 lifts that withhold now that the edge is decoupled.)

#### Scenario: three-symbol public API

- GIVEN the change is applied
- WHEN `uv run python -c "from STOODX import STOODX, FeatureExtractor, FeatureExplanation"` runs
- THEN it succeeds without `ImportError`
- AND `STOODX.__all__ == ["STOODX", "FeatureExtractor", "FeatureExplanation"]`
- AND `STOODXPostprocessor` is absent from `STOODX.__all__`

#### Scenario: top-level import does not pull openood (3 symbols)

- GIVEN the change is applied
- WHEN `uv run python -c "import STOODX; from STOODX import FeatureExplanation; import sys; assert 'openood' not in sys.modules"`
- THEN the assertion passes
- AND `openood` is absent from `sys.modules` WITH all three exported symbols imported

## ADDED Requirements

### Requirement: feature_visualization is decoupled from the adapter

`src/STOODX/feature_visualization.py` MUST import `STOODXPostprocessor` ONLY under
`typing.TYPE_CHECKING` (paired with `from __future__ import annotations`), so that importing
`FeatureExplanation` MUST NOT eagerly import the adapter and MUST NOT pull `openood` into
`sys.modules`. This REMOVES the import edge that C2's scope boundary explicitly preserved as a
non-goal.

#### Scenario: importing FeatureExplanation does not pull openood

- GIVEN the change is applied
- WHEN `uv run python -c "from STOODX.feature_visualization import FeatureExplanation; import sys; assert 'openood' not in sys.modules"`
- THEN the assertion passes
- AND `STOODX._openood_adapter.postprocessor` is NOT loaded as a side effect

#### Scenario: adapter type annotation stays resolvable statically

- GIVEN `feature_visualization.py` guards the adapter import with `if TYPE_CHECKING:`
- WHEN a static type-checker resolves the adapter annotation
- THEN the annotation resolves to `STOODXPostprocessor` at analysis time
- AND runtime import of `feature_visualization` does not trigger the adapter's module load
