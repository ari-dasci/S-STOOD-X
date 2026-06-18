# Tasks: c2-structural-refactor

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated handwritten lines | ~50-60 |
| 400-line budget risk | Low |
| Chained PRs recommended | No |
| Suggested split | Single PR #2 |
| Delivery strategy | force-chained (but C2 itself fits in one PR) |
| Chain strategy | stacked-to-main |

Decision needed before apply: No
Chained PRs recommended: No
Chain strategy: stacked-to-main
400-line budget risk: Low

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | Full structural refactor (Steps A–I) | PR #2 | Single cohesive unit; base = main |

## Phase 1: src/ Layout + Module Moves (Steps A–B)

- [x] 1.1 `mkdir -p src && git mv STOODX src/STOODX` — verify `src/STOODX/` contains all 4 modules + `__init__.py`
- [x] 1.2 `git mv src/STOODX/STOODX.py src/STOODX/stoodx.py`
- [x] 1.3 `git mv src/STOODX/featureStractor.py src/STOODX/feature_extractor.py`
- [x] 1.4 `git mv src/STOODX/featureVisualization.py src/STOODX/feature_visualization.py`
- [x] 1.5 `mkdir -p src/STOODX/_openood_adapter && git mv src/STOODX/STOODXPostprocessor.py src/STOODX/_openood_adapter/postprocessor.py`
- [x] 1.6 Verify: old `STOODX/` dir no longer exists at repo root; `git status` shows rename history

## Phase 2: Package Init + Adapter Setup (Step C)

- [x] 2.1 Write `src/STOODX/__init__.py` with docstring, `from .stoodx import STOODX`, `from .feature_extractor import FeatureExtractor`, `__all__ = ["STOODX", "FeatureExtractor"]`
- [x] 2.2 Create `src/STOODX/_openood_adapter/__init__.py` with internal-only docstring (no re-exports)
- [x] 2.3 Add module docstring (internal/not public API) and `__all__ = ["STOODXPostprocessor"]` to head of `postprocessor.py`

## Phase 3: Import Rewrites + Class Rename (Step D)

- [x] 3.1 `src/STOODX/stoodx.py:3` — rewrite import → `from .feature_extractor import FeatureExtractor`
- [x] 3.2 `src/STOODX/_openood_adapter/postprocessor.py:9-10` — rewrite imports → `from ..feature_extractor import FeatureExtractor` and `from ..stoodx import STOODX`
- [x] 3.3 `src/STOODX/feature_visualization.py:7` — rewrite import → `from ._openood_adapter.postprocessor import STOODXPostprocessor` (PRESERVE this edge — C3 territory)
- [x] 3.4 `src/STOODX/feature_extractor.py:5` — class def `FeatureStractor` → `FeatureExtractor`
- [x] 3.5 `src/STOODX/feature_extractor.py:29` — `super(FeatureStractor, self)` → `super(FeatureExtractor, self)`
- [x] 3.6 `src/STOODX/stoodx.py:21` — type annotation `FeatureStractor` → `FeatureExtractor`
- [x] 3.7 `src/STOODX/_openood_adapter/postprocessor.py:53` — instantiation `FeatureStractor` → `FeatureExtractor`

## Phase 4: Test Moves + Import Rewrites (Step E)

- [x] 4.1 `git mv tests/test_featureStractor.py tests/test_feature_extractor.py`
- [x] 4.2 `git mv tests/test_featureVisualization.py tests/test_feature_visualization.py`
- [x] 4.3 `mkdir -p tests/_openood_adapter && git mv tests/test_postProcessor_{cifar10,cifar100,imagenet,imagenet200}.py tests/_openood_adapter/` (NO `__init__.py` — pytest rootdir convention)
- [x] 4.4 Rewrite `tests/test_feature_extractor.py:4` import → `from STOODX.feature_extractor import FeatureExtractor`
- [x] 4.5 Fix typo `feature_estractor` → `feature_extractor` at L26-28 in `tests/test_feature_extractor.py`
- [x] 4.6 Rewrite `tests/test_feature_visualization.py:11-12` → `from STOODX._openood_adapter.postprocessor import STOODXPostprocessor` and `from STOODX.feature_visualization import FeatureExplanation`
- [x] 4.7 Rewrite import in each `tests/_openood_adapter/test_postProcessor_{...}.py` → `from STOODX._openood_adapter.postprocessor import STOODXPostprocessor`

## Phase 5: Build Config + Install (Steps F–G)

- [x] 5.1 Edit `pyproject.toml`: replace `[tool.setuptools]` + `packages = ["STOODX"]` with `[tool.setuptools.packages.find] where = ["src"]`
- [x] 5.2 `rm -rf *.egg-info && uv sync` — verify exit 0, editable install discovers `src/STOODX/`

## Phase 6: README (Step H)

- [x] 6.1 README L126: `FeatureStractor` → `FeatureExtractor` in quick-start
- [x] 6.2 README L129: update instantiation symbol to match new class name
- [x] 6.3 README L168-183: update structure tree → `src/STOODX/` paths with `_openood_adapter/` subpackage
- [x] 6.4 README L104-110 (Known limitations): updated 5 deferred-test paths to new layout

## Phase 7: Verification Gates (Step I)

- [x] 7.1 Public API gate: `uv run python -c "from STOODX import STOODX, FeatureExtractor; import STOODX; assert STOODX.__all__==['STOODX','FeatureExtractor']; print('OK')"` (via `import STOODX as pkg` alias due to class/package name shadowing)
- [x] 7.2 No-openood gate: `uv run python -c "import STOODX; from STOODX import STOODX, FeatureExtractor; import sys; assert 'openood' not in sys.modules; print('clean')"`
- [x] 7.3 Grep-zero gate: `grep -rn "featureStractor\|featureVisualization\.py\|STOODXPostprocessor\.py\|FeatureStractor\|feature_estractor" src/ tests/ README.md` → ZERO hits
- [x] 7.4 Collection gate: `uv run pytest --collect-only tests/test_feature_extractor.py` → 12 tests collected, no `ImportError`
- [x] 7.5 No-new-errors gate: `uv run pytest --collect-only` → 5 openood modules fail only on imgaug/numpy2 (same C1 root cause), NOT import typos
- [x] 7.6 Install gate: `uv sync` exit 0
