# Tasks: c3-verify-openood-extension

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | 70–80 handwritten (~15 conftest + ~3 FV + ~3 init + ~35 contract test + ~20 README) |
| 400-line budget risk | Low |
| Chained PRs recommended | No |
| Suggested split | Single PR #3 final |
| Delivery strategy | force-chained (chain slot C3, stacked-to-main) |
| Chain strategy | stacked-to-main |

Decision needed before apply: No
Chained PRs recommended: No
Chain strategy: stacked-to-main
400-line budget risk: Low

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | All 5 edits + gates in one commit | PR #3 | Single work-unit; ~70–80 lines; reversible via git revert |

## Phase 1: Infrastructure (conftest shim)

- [x] 1.1 Create root `conftest.py` with exact `np.sctypes` shim (hasattr guard, docstring cites imgaug issue #595) per design §A
  - DoD: file exists; `hasattr(np, "sctypes")` is True after import under numpy≥2

- [x] 1.2 Verify F-3: `uv run pytest --collect-only` → ALL 7 modules collect, ZERO import errors (env R3 RESOLVED)
  - DoD: exit 0; collection count includes `test_featureVisualization` + 4 postProcessor modules
  - RESOLVED: `timm>=0.9` + `foolbox>=3.3` added to pyproject.toml (undeclared OpenOOD deps). F-3 = 314 tests collected, exit 0.

## Phase 2: Decoupling & Export

- [x] 2.1 Edit `src/STOODX/feature_visualization.py`: add `from __future__ import annotations` as FIRST line; wrap adapter import under `if TYPE_CHECKING:` per design §B
  - DoD: no eager adapter import at runtime

- [x] 2.2 Verify F-2: `uv run python -c "from STOODX.feature_visualization import FeatureExplanation; import sys; assert 'openood' not in sys.modules"` → OK (decoupling proven)
  - DoD: assertion passes; openood absent from sys.modules

- [x] 2.3 Edit `src/STOODX/__init__.py`: `__all__` 2→3 symbols + `from .feature_visualization import FeatureExplanation` per design §C
  - DoD: `STOODX.__all__ == ["STOODX", "FeatureExtractor", "FeatureExplanation"]`

- [x] 2.4 Verify F-1: `uv run python -c "import STOODX; from STOODX import STOODX, FeatureExtractor, FeatureExplanation; import sys; assert 'openood' not in sys.modules"` → OK
  - DoD: assertion passes; 3-symbol import keeps no-openood invariant

## Phase 3: Contract Test

- [x] 3.1 Create `tests/_openood_adapter/test_postprocessor_contract.py`: construct `STOODXPostprocessor(cfg)` with minimal dict (K=5, distance=cosine, feature_name=layer4, + upstream typo keys); assert isinstance(BasePostprocessor); assert callable for __init__/setup/postprocess/inference; assert attrs (K, distance, feature_name, device, oodTest); oodTest is None. NO method calls with data per design §D
  - DoD: test file exists; imports resolve

- [x] 3.2 Verify F-4: `uv run pytest tests/_openood_adapter/test_postprocessor_contract.py` → PASSES (green, no data/GPU)
  - DoD: exit 0; all assertions pass (3/3 green)

## Phase 4: Documentation & Commit

- [x] 4.1 Edit `README.md`: rewrite known-limitations L102–118 (drop "5 modules fail collection"); document shim + imgaug-rot note; document eval_ood as manual/prereq (pretrained+datasets+GPU); update L171 comment to 3 symbols per design §E
  - DoD: README no longer claims collection failure; eval_ood prereq documented

- [x] 4.2 Commit as single work-unit: `feat(openood): verify extension contract, decouple core, resolve imgaug collection` per design Apply Sequence
  - DoD: single commit on chain branch; no AI attribution
  - RESOLVED: commit `53cf69c` exists on branch `c3-verify-openood-extension`. No Co-Authored-By. Includes 7 files (shim, decouple, __all__, contract test, README, pyproject.toml, uv.lock).
