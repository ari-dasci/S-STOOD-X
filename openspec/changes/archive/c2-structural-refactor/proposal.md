# Proposal: c2-structural-refactor

## Intent

Pure structural refactor — turn S-STOOD-X into a clean, installable library: PEP 8 `snake_case` modules, `src/` layout, working public API, and a privatized OpenOOD adapter. Fixes the broken README quick-start (`from STOODX import STOODX, FeatureExtractor`). No logic, numerics, statistics, or XAI algorithm changes. Second of three chained changes (C1 uv ✓ → **C2 refactor** → C3 verify OpenOOD extension). All design decisions LOCKED by the author.

## Scope

### In Scope

**1. src/ layout** — `STOODX/` → `src/STOODX/`. `pyproject.toml`: `[tool.setuptools] packages=["STOODX"]` → `[tool.setuptools.packages.find] where = ["src"]` (auto-discovers `STOODX` + `STOODX._openood_adapter`). Setuptools backend unchanged.

**2. Module renames (snake_case) + class rename (author chose option B):**

| From | To |
|------|----|
| `STOODX/STOODX.py` | `src/STOODX/stoodx.py` |
| `STOODX/featureStractor.py` (`class FeatureStractor`) | `src/STOODX/feature_extractor.py` (`class FeatureExtractor`) |
| `STOODX/featureVisualization.py` | `src/STOODX/feature_visualization.py` |
| `STOODX/STOODXPostprocessor.py` | `src/STOODX/_openood_adapter/postprocessor.py` (CLASS name `STOODXPostprocessor` unchanged; module docstring marks it private) |

**3. `__init__.py` exports** — `__all__ = ["STOODX", "FeatureExtractor"]`. Export ONLY these two (neither pulls openood). Do NOT export `FeatureExplanation` (transitively imports openood via featureVisualization→adapter → would regress C1 R2 gate). Do NOT export the adapter.

**4. Intra-package relative imports** (switch 3 absolute → relative):

| File | Rewrite |
|------|---------|
| `stoodx.py:3` | `from .feature_extractor import FeatureExtractor` |
| `_openood_adapter/postprocessor.py:9` | `from ..feature_extractor import FeatureExtractor` |
| `_openood_adapter/postprocessor.py:10` | `from ..stoodx import STOODX` |
| `feature_visualization.py:7` | `from ._openood_adapter.postprocessor import STOODXPostprocessor` (PRESERVE this edge — decoupling is C3) |

**5. Test rewrites + relocations** (7 import lines + renames, from exploration obs #1184):

| From | To | Import becomes |
|------|----|----------------|
| `tests/test_featureStractor.py` | `tests/test_feature_extractor.py` | `from STOODX.feature_extractor import FeatureExtractor`; fix typo `feature_estractor`→`feature_extractor` (L26-28) |
| `tests/test_featureVisualization.py` | `tests/test_feature_visualization.py` | `:11` `from STOODX._openood_adapter.postprocessor import STOODXPostprocessor`; `:12` `from STOODX.feature_visualization import FeatureExplanation` |
| `tests/test_postProcessor_{cifar10,cifar100,imagenet,imagenet200}.py` | `tests/_openood_adapter/test_postProcessor_{...}.py` | `from STOODX._openood_adapter.postprocessor import STOODXPostprocessor` |

**6. README updates** — Quick-start L126 `FeatureStractor`→`FeatureExtractor`; L129 instantiation symbol; structure tree L169-173 → new `src/STOODX/` paths. (Method-body drift at L137-161 — `addFeatures`/`finalizeFeatures`/`test` vs actual API — is pre-existing doc drift, NOT fixed here.)

### Out of Scope (non-goals — deferred / separate)

- C3 territory: decouple `feature_visualization` from adapter; fix imgaug/numpy2 collection issue (5 modules still fail collection on the SAME root cause).
- Separate future change: linter/type-checker/CI; build-backend swap; dep changes (libmr/baycomp vestigial check).
- No body-logic changes. No rename of `STOODX`, `FeatureExplanation`, or `STOODXPostprocessor` classes. No `feature_visualization` → `feature_explanation` module rename (collision risk with `crp.visualization.FeatureVisualization`).

## Capabilities

> Contract for sdd-spec. Researched `openspec/specs/` (only `environment` exists, synced from C1).

### New Capabilities
- `package-structure`: src/ layout, snake_case module naming convention, public exports `{STOODX, FeatureExtractor}`, private `_openood_adapter` subpackage boundary.

### Modified Capabilities
- None. The `environment` capability's invariants (R1 uv-resolves, R2 `import STOODX`, R4 lock committed) are preserved, not changed. The DEFERRED imgaug/numpy2 item (R3) stays deferred — C2 does not resolve it.

## Approach

Sequential mechanical moves following the exhaustive import graph from exploration #1184. Order: (a) create `src/STOODX/` + `src/STOODX/_openood_adapter/`; (b) move+rename 4 modules; (c) rewrite 3 intra-package imports to relative; (d) populate `__init__.py`; (e) pyproject discovery change; (f) move+rename+rewrite 6 test files; (g) README sync. Verify gates run after each structural step. OpenOOD pin (`3c35632…`) unchanged.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/STOODX/` | New | Moved + renamed core modules |
| `src/STOODX/_openood_adapter/` | New | Privatized adapter subpackage |
| `src/STOODX/__init__.py` | Modified | Populated with `__all__` + 2 exports |
| `pyproject.toml` | Modified | `[tool.setuptools.packages.find]` |
| `tests/` | Modified | 2 renames, 4 relocations, 7 import rewrites |
| `README.md` | Modified | Quick-start + structure tree |
| `STOODX/` (old) | Removed | Entire old dir deleted |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Silent import breakage in 5 un-collectable tests (no live check until C3) | Medium | Grep-verify ZERO residual `featureStractor\|STOODXPostprocessor.py\|featureVisualization.py\|FeatureStractor` in `src/ tests/ README.md`; bake into tasks DoD |
| `import STOODX` regresses if `__init__` over-exports | Medium | Verify gate: `uv run python -c "import STOODX; from STOODX import STOODX, FeatureExtractor"` must NOT pull openood |
| Public API rename `FeatureStractor`→`FeatureExtractor` breaks external callers | Low | Pre-publication (v0.0.1); documented here; README updated |
| Stale `*.egg-info/` refs | Low | Non-issue: gitignored, auto-regenerates on rebuild |

## Rollback Plan

`git revert` the C2 commit(s). Old `STOODX/` dir, mixedCase names, and empty `__init__.py` are restored by VCS. `uv sync` re-resolves to the prior (still-valid) layout. No data migration, no irreversible state.

## Dependencies

- Builds on **C1** (`c1-uv-migration`, archived PASS_WITH_DEFERRALS) — uv environment must be functional.
- Unblocks **C3** (verify OpenOOD extension against clean structure).
- OpenOOD pin unchanged from C1: rev `3c35632ee91b54b09d1f085d04f94744cece7d0b`.

## Success Criteria

- [ ] `uv run python -c "import STOODX"` works (no openood pull-in).
- [ ] `uv run python -c "from STOODX import STOODX, FeatureExtractor"` works (the previously-broken README quick-start).
- [ ] `uv run pytest --collect-only` collects `test_feature_extractor.py` with NO NEW import errors beyond the deferred imgaug/numpy2 issue (the 5 openood-pulling modules may still fail collection, but for the SAME root cause, not import typos).
- [ ] `grep -rn "featureStractor\|STOODXPostprocessor.py\|featureVisualization.py\|FeatureStractor" src/ tests/ README.md` returns ZERO hits.
- [ ] `uv sync` still works (package discovery updated to `src/`).
- [ ] README quick-start code is symbol-accurate (`FeatureExtractor`).

## Chain Context

**PR #2** in the stacked-to-main chain. C1 ✓ → **C2 (this)** → C3. Stays within the 800-line review budget (moves+renames are diff-heavy but mechanical). C2 unblocks C3 by giving it a clean, privatized adapter boundary to verify the OpenOOD extension against.
