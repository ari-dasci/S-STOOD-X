# Proposal: Rename the public `STOODX` class to `STOODXDetector` (C4)

## Intent

The package `STOODX` and the class `STOODX` share a name, so `import STOODX` binds the
**package** and shadows the class on its own surface. Inspecting `STOODX.__all__` /
`dir(STOODX)` requires an alias (`import STOODX as pkg`; C2 tasks.md:74). Behaviorally
benign but a recurring introspection paper-cut for REPL, tooling, and docs. Pre-publication
the fix is cheap; post-publication it becomes a breaking API churn. Locked rename target
from explore: **`STOODXDetector`**.

## Classification (per `rules.proposal`)

- **Type**: structural (safe) — mechanical rename, no behavior/numerics change. No author
  sign-off required on methodology.
- **Compute prerequisites**: none. No model, dataset, or GPU path touched.

## Scope

### In Scope
- Rename `class STOODX` → `class STOODXDetector` at `src/STOODX/stoodx.py:9`.
- Update re-export + `__all__` string literal in `src/STOODX/__init__.py` (lines 2, 6).
- Update import + sole instantiation in `src/STOODX/_openood_adapter/postprocessor.py` (lines 18, 62).
- Update README quick-start + comment (`README.md:168, 179, 213`).
- Delta spec updating the `package-structure` public-API + README-accuracy requirements.
- Add `tests/test_public_api.py` smoke test closing the coverage gap (explore §3c).

### Out of Scope (non-goals)
- `STOODXPostprocessor` class (locked by C2 D7 — different class).
- `feature_visualization.py`, all `tests/_openood_adapter/*`, `tests/test_feature_*`.
- `pyproject.toml`, `configs/`, `openspec/changes/archive/**`.
- Pre-existing README method-body drift and `stoodx.py:15` `FeatureEstractor` typo (explore
  §6 — same README region; NOT bundled; C4 stays surgical).
- No numerics/logic/algorithm change; no API surface change beyond the single rename.

## Capabilities

> Contract for sdd-spec.

### New Capabilities
None.

### Modified Capabilities
- `package-structure`: "Public API surface and no openood pull-in" and "README accuracy"
  requirements change — `__all__` and canonical import become `STOODXDetector`. Preserved
  behavior: three-symbol surface, no `openood` pull-in, `STOODXPostprocessor` absent.

## Approach

Mechanical rename at the 4 source sites from explore §3b. No dynamic dispatch to update
(verified — no `importlib`/registry/`__name__` reads point at `"STOODX"`). README + delta
spec, then a new smoke test asserting `from STOODX import STOODXDetector` and
`STOODXDetector.__module__ == "STOODX.stoodx"`. `__module__` is unchanged because the module
**file** is not renamed.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/STOODX/stoodx.py` | Modified | class def line 9 |
| `src/STOODX/__init__.py` | Modified | import + `__all__` string literal |
| `src/STOODX/_openood_adapter/postprocessor.py` | Modified | import + instantiation |
| `README.md` | Modified | quick-start lines 168/179, comment 213 |
| `openspec/specs/package-structure/spec.md` | Modified (via delta) | public-API + README-accuracy requirements |
| `tests/test_public_api.py` | New | import/identity smoke test |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Weak automated coverage of bare class (explore §3c) | Med | Add `tests/test_public_api.py` in this change |
| Stray stale reference to old class name | Low | Grep `src/ tests/ README.md` post-apply for `\bSTOODX\b` (excludes `STOODXPostprocessor`, package dir, `stoodx.py`) |
| Pickle qualifier break for external pipelines | Low | Pre-publication; adapter never pickles the class (explore §6 — only tensors at `postprocessor.py:103-110`) |
| Scope creep into README doc drift | Low | Explicitly excluded; C4 surgical |

## Rollback Plan

`git revert` the C4 commit. No persisted-state migration: no class instance is pickled by
the adapter.

## Dependencies

None external. Builds on merged C1→C3.

## Success Criteria

- [ ] `uv run python -c "from STOODX import STOODXDetector"` succeeds.
- [ ] `STOODX.__all__ == ["STOODXDetector", "FeatureExtractor", "FeatureExplanation"]`.
- [ ] `tests/test_public_api.py` passes (no GPU/data needed).
- [ ] `uv run pytest --collect-only` still collects all 7 test modules, zero import errors.
- [ ] Grep for residual bare `STOODX` class refs in `src/` returns zero (excluding
      `STOODXPostprocessor`, the package dir, and `stoodx.py`).
- [ ] Delta spec covers the updated public-API + README-accuracy requirements.

## PR Strategy

800-line force-chained budget. C4 is ~30–50 net lines → single PR slice, no chaining.
C5 (linter/CI) remains the next slice in the stacked-to-main chain.
