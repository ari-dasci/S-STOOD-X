# Archive Report: c3-verify-openood-extension

**Status**: PASS (no CRITICAL; 3 WARNINGs resolved during archive)
**Date**: 2026-06-19
**Branch**: `c3-verify-openood-extension`
**Commit**: `53cf69c` (7 files, +396/-19)

## Change Summary

C3 = PR #3 final in the stacked-to-main chain (C1 -> C2 -> C3). Proved **model Z**: the privatized
OpenOOD adapter is a valid extension (contract test), STOOD-X core is decoupled from openood
(3-symbol public API), and the previously-DEFERRED imgaug/numpy2 collection failure (env R3) is
resolved. The 3-change chain is now COMPLETE.

## What Was Done

1. **np.sctypes shim** — new root `conftest.py` re-adds `numpy.sctypes` before any openood import.
   Hasattr guard; cites imgaug issue #595.
2. **Decouple feature_visualization** — `from __future__ import annotations` +
   `if TYPE_CHECKING:` guard on adapter import. Zero runtime behavior change.
3. **Promote FeatureExplanation** — `__all__` 2->3 symbols + top-level import.
4. **Contract unit test** — `test_postprocessor_contract.py` asserts isinstance(BasePostprocessor) +
   4 callable methods + config attrs. No data/GPU.
5. **README rewrite** — removed "5 modules fail collection" claim; documented shim, decoupling,
   contract test, and eval_ood as manual/prereq.
6. **Dependency additions** — `timm>=0.9` + `foolbox>=3.3` added to pyproject.toml + uv.lock.
   Undeclared OpenOOD deps (upstream packaging bug); required for full collection.
7. **Exhaustive dep audit methodology** —
   - `timm`: eagerly imported by `openood.attacks.misc` -> NOT in openood metadata -> ADDED
   - `foolbox`: eagerly imported by `openood.evaluation_api.attackdataset` -> NOT in openood metadata -> ADDED
   - `statsmodels`: imported conditionally by `openood.datasets`; NOT needed at collection time -> already satisfied
   - `libmr`: imported by `openood.postprocessors.base`; already in STOOD-X deps -> already satisfied
   - `clip` (OpenAI CLIP): imported by `openood.datasets`; NOT imported at collection time -> guarded by runtime path, not added
   - `mmcls`/`mmcv`: imported by `openood.models`; NOT imported at collection time; dead code paths in the pinned rev -> not added

## Specs Synced

| Domain | Action | Details |
|--------|--------|---------|
| openood-extension | Created | Full spec seeded from C3 delta. R3 mechanism amended per W3 to cite BOTH conftest shim AND timm/foolbox additions. |
| environment | Updated | R3 transitioned DEFERRED -> RESOLVED. Mechanism now correctly attributes the collection fix to BOTH the conftest np.sctypes shim AND the timm/foolbox dep additions. |
| package-structure | Updated | R4 updated from 2-symbol to 3-symbol API. Added `feature_visualization decoupled from adapter` requirement (TYPE_CHECKING). Updated scope boundary and import-correctness requirements to reflect C3 completions. |

## Resolved Deferrals

- **env R3 imgaug/numpy2 collection** (C1 DEFERRED) -> CLOSED in C3 via conftest shim + timm/foolbox deps
- **C2 deferral #2 feature_visualization coupling** (C2 non-goal) -> CLOSED in C3 via TYPE_CHECKING decoupling

## W2 Design-Deviation Note

The design's File Changes table listed 5 files and stated "no lockfile/dep changes". The commit
added 2 extra files (`pyproject.toml`, `uv.lock`) and 2 deps (`timm>=0.9`, `foolbox>=3.3`). This
deviation is spec-required, not unsanctioned: the openood-extension R3 and environment R3 specs
require ALL 7 modules to collect with ZERO import errors, and the conftest shim alone was
insufficient (reverting timm while keeping the shim breaks F-3). The version-sensitive pins
explicitly protected in the design (numpy >=2.1.2, openood rev 3c35632) are unchanged.

## Warning Resolution

- **W1** (stale task checkboxes 1.2, 4.2): Flipped to [x] with resolution notes. DoDs met per
  verify-report (F-3 = 314 tests, commit 53cf69c exists).
- **W2** (design dep-deviation): Recorded above. Benign, spec-required.
- **W3** (spec mechanism narrative): Amended all 3 spec artifacts during sync to attribute the
  collection fix to BOTH the conftest shim AND timm/foolbox additions.

## Chain Status

**C1 -> C2 -> C3 COMPLETE**. The full 3-change stacked-to-main chain is archived:
- C1 (`c1-uv-migration`): uv-managed env, reproducible lock, src/ layout
- C2 (`c2-structural-refactor`): snake_case rename, privatized adapter, 2-symbol API
- C3 (`c3-verify-openood-extension`): shim, decouple, 3-symbol API, contract test, dep audit

## Archive Contents

- proposal.md
- specs/openood-extension/spec.md
- specs/environment/spec.md (delta)
- specs/package-structure/spec.md (delta)
- design.md
- tasks.md (10/10 tasks complete; stale checkboxes reconciled)
- verify-report.md (PASS WITH WARNINGS; all 13 scenarios PASS)
- archive-report.md (this file)

## Next Steps

The 3-change chain is COMPLETE. Future work (outside SDD scope):
- OpenOOD fork creation (proper fix for imgaug/numpy2 without shim)
- Linter/type-checker/CI configuration
- Full `eval_ood` validation with pretrained checkpoints + datasets + GPU
