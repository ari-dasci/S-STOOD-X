# Proposal: c3-verify-openood-extension

## Intent

Prove **model Z**: STOOD-X core is standalone (no `openood` pull-in) and the privatized OpenOOD adapter is a valid OpenOOD extension. Resolves BOTH deferred blockers carried from C1/C2 — the imgaug/numpy2 collection failure (env R3) and the `feature_visualization`↔adapter coupling (C2 deferral #2). First time the full test suite collects cleanly.

## Classification

**Structural (safe)** — every task. Test infrastructure (shim), type-annotation refactor, export declaration, structural-conformance test, docs. No statistics/numerics/XAI logic, no `eval_ood` runs.

## Scope

### In Scope
- **(1) `np.sctypes` shim** — new root `conftest.py` re-adds `numpy.sctypes` before any openood import. Implements decision #1. Reversible. Comment cites imgaug version + upstream issue. **Resolves env R3 DEFERRED.**
- **(2) Decouple `feature_visualization`** — `from __future__ import annotations` + `if TYPE_CHECKING:` guard around the adapter import (`feature_visualization.py:7`). Implements decision #2. Zero runtime behavior change (import was annotation-only). **Resolves C2 deferral #2.**
- **(3) Promote `FeatureExplanation`** — `__all__ = ["STOODX", "FeatureExtractor", "FeatureExplanation"]` + `from .feature_visualization import FeatureExplanation`. Implements decision #3. **Modifies package-structure R4 (2→3 symbols).**
- **(4) Contract unit test** — `tests/_openood_adapter/test_postprocessor_contract.py` asserting isinstance + 4 callable methods + config attrs. No data/GPU. Implements decision #4.
- **(5) README rewrite** — remove the now-false "5 modules fail collection" paragraph (L102-118); document the shim, decoupling, and contract test. Implements decision #5.

### Out of Scope
- Statistics/numerics/XAI algorithm logic.
- Heavy `eval_ood` tests (need pretrained + datasets + GPU → manual/prereq, OUT of automated gate).
- OpenOOD fork creation (separate future effort).
- numpy downgrade / OpenOOD pin change (stays rev `3c35632…`).
- Build backend swap, linter/CI.
- `STOODXPostprocessor` implementation changes (only test its contract).

## Capabilities

### New Capabilities
- `openood-extension`: STOODX adapter as a valid OpenOOD extension — `BasePostprocessor` contract (4 methods) + `Evaluator` isinstance acceptance. Structural conformance only, no numerics.

### Modified Capabilities
- `environment`: R3 (test collection) transitions DEFERRED → RESOLVED via the `np.sctypes` shim (test infrastructure).
- `package-structure`: R4 (public API surface) moves 2→3 symbols (`FeatureExplanation` added); the `feature_visualization`→adapter import edge is now decoupled. Preserves the no-openood-pull-in invariant (re-asserted with 3 symbols).

## Approach

Shim sits in root `conftest.py` (runs before any openood import). Decoupling uses `TYPE_CHECKING` so the eager adapter import vanishes from the core path. Contract test builds a `STOODXPostprocessor` instance and asserts `isinstance(BasePostprocessor)` + presence/callability of `__init__`/`setup`/`postprocess`/`inference` + expected config attributes — mirroring how OpenOOD's `Evaluator` accepts a pre-built instance (isinstance gate). No numerical execution.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `conftest.py` (new, root) | New | `np.sctypes` shim; test infra only |
| `src/STOODX/feature_visualization.py` | Modified | 3-line decouple (annotations + TYPE_CHECKING) |
| `src/STOODX/__init__.py` | Modified | `__all__` 2→3 symbols |
| `tests/_openood_adapter/test_postprocessor_contract.py` | New | structural contract test |
| `README.md` | Modified | rewrite L102-118 known-limitations |
| `openspec/specs/environment/spec.md` | Modified | R3 DEFERRED → RESOLVED (delta) |
| `openspec/specs/package-structure/spec.md` | Modified | R4 2→3 symbols (delta) |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Shim masks upstream rot (imgaug unmaintained) | Med | Comment cites imgaug version + upstream issue; fork is long-term fix |
| Contract test = structural, not numerical confidence | Med | Boundary stated explicitly; full `eval_ood` documented as manual/prereq |
| R4 promotion changes no-openood invariant | Low | Verify re-asserts no `openood` in `sys.modules` WITH 3 symbols |
| "Standalone core" misread as "no deps" | Low | Scope to "no openood"; zennit/crp remain as runtime deps (acceptable) |

## Rollback Plan

Fully reversible: delete `conftest.py`, revert `feature_visualization.py` (3 lines), `__init__.py` (`__all__`), delete the new test, revert README. No lockfile/dependency changes to unwind.

## Dependencies

- Existing: uv env (C1), src/ layout + privatized adapter (C2). No new deps.

## Success Criteria

- [ ] `uv run python -c "import STOODX; from STOODX import STOODX, FeatureExtractor, FeatureExplanation; import sys; assert 'openood' not in sys.modules"` → OK (3-symbol, no openood)
- [ ] `uv run pytest --collect-only` → ALL 7 test modules collect, ZERO import errors (env R3 RESOLVED)
- [ ] `uv run pytest tests/_openood_adapter/test_postprocessor_contract.py` → PASSES (green, no data/GPU)
- [ ] `uv run python -c "from STOODX.feature_visualization import FeatureExplanation; import sys; assert 'openood' not in sys.modules"` → OK (decoupling proven)
- [ ] README no longer claims "5 modules fail collection"

## Chain Context

C3 = PR #3 (final), stacked-to-main chain (target: `main`). After C3: uv-managed, src/ snake_case layout, standalone core + privatized adapter + working OpenOOD extension + clean test collection. Closes the chain.
