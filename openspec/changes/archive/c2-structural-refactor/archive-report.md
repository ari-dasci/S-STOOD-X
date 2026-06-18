# Archive Report — c2-structural-refactor

- **Change**: `c2-structural-refactor`
- **Status**: PASS_WITH_DEFERRALS (no CRITICAL, no WARNING)
- **Branch**: `c2-structural-refactor` (PR #2, stacked-to-main chain)
- **Implementation commit**: `ebae7bf` (19 files, +571/-32)
- **Archive date**: 2026-06-18

## What Was Done

Pure structural refactor turning S-STOOD-X into a clean installable library:

- `src/` layout (`STOODX/` moved to `src/STOODX/`)
- PEP 8 snake_case module renames (`featureStractor.py` -> `feature_extractor.py`, etc.)
- Class rename `FeatureStractor` -> `FeatureExtractor`
- 2-symbol public API via `__init__.py` (`STOODX`, `FeatureExtractor`)
- Privatized OpenOOD adapter at `src/STOODX/_openood_adapter/`
- Relative intra-package imports (4 lines, 3 files)
- Test relocations + import rewrites (6 files, 7 import lines)
- `pyproject.toml` discovery stanza update (`packages.find where=["src"]`)
- README quick-start + structure tree sync

All changes purely structural. No logic, numerics, statistics, or XAI algorithm changes.

## Specs Synced

| Domain | Action | Details |
|--------|--------|---------|
| `package-structure` | Created (seed) | New capability spec with 9 requirements (R1-R9), all PASS. Full spec copied from delta. |

The `environment` capability (`openspec/specs/environment/spec.md`) was NOT touched — env R3 DEFERRED annotation preserved as-is.

## Archive Contents

- proposal.md
- specs/package-structure/spec.md
- design.md
- tasks.md (32/32 tasks complete)
- verify-report.md

## Deferred Items Carried Forward

1. **imgaug/numpy2 collection issue** (env R3) — 5 OpenOOD-pulling test modules fail collection on `np.sctypes` removed in NumPy 2.0. Pre-existing from C1, same root cause. C2 preserves env invariants; this is C3-bound.
2. **feature_visualization <-> adapter decoupling** — the `feature_visualization -> _openood_adapter.postprocessor` import edge is preserved by design (D6). Decoupling is C3 scope.

## Next Steps

- Begin C3: verify OpenOOD extension against the clean structure (`/sdd-new` or `/sdd-explore` for `c3-verify-openood-extension`).

## SDD Cycle Complete

The change has been fully planned, implemented, verified, and archived.
Ready for the next change.
