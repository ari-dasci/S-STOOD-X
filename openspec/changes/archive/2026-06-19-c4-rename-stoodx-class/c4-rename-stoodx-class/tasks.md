# Tasks: Rename `STOODX` class to `STOODXDetector` (C4)

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | 18-25 |
| 400-line budget risk | Low |
| Chained PRs recommended | No |
| Suggested split | Single PR #4 |
| Delivery strategy | auto-chain (below threshold — no chain needed) |
| Chain strategy | stacked-to-main (session config) |

Decision needed before apply: No
Chained PRs recommended: No
Chain strategy: stacked-to-main
400-line budget risk: Low

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | Rename class + update all refs + smoke test | PR #4 | base = main; single commit sufficient |

## Phase 1: Class Rename (Foundation)

- [x] 1.1 Rename `class STOODX` → `class STOODXDetector` in `src/STOODX/stoodx.py:9`
- [x] 1.2 Update import `from .stoodx import STOODX` → `from .stoodx import STOODXDetector` in `src/STOODX/__init__.py:2`
- [x] 1.3 Update `__all__` string `"STOODX"` → `"STOODXDetector"` in `src/STOODX/__init__.py:6`

## Phase 2: Internal Consumer Update

- [x] 2.1 Update import `from ..stoodx import STOODX` → `from ..stoodx import STOODXDetector` in `src/STOODX/_openood_adapter/postprocessor.py:18`
- [x] 2.2 Update instantiation `STOODX(` → `STOODXDetector(` in `src/STOODX/_openood_adapter/postprocessor.py:62`

## Phase 3: Documentation

- [x] 3.1 Update `from STOODX import STOODX` → `from STOODX import STOODXDetector` in `README.md:168`
- [x] 3.2 Update `detector = STOODX(` → `detector = STOODXDetector(` in `README.md:179`
- [x] 3.3 Update `# Public API: STOODX,` → `# Public API: STOODXDetector,` in `README.md:213`

## Phase 4: Smoke Test

- [x] 4.1 Create `tests/test_public_api.py`: assert `from STOODX import STOODXDetector` succeeds, `__module__ == "STOODX.stoodx"`, `__all__` contains `"STOODXDetector"` not `"STOODX"`

## Phase 5: Verification Gates

- [x] 5.1 Grep `src/ tests/ README.md` for bare `\bSTOODX\b` — every match must be `STOODXPostprocessor`, package dir, `stoodx.py`, or `STOODXDetector`
- [x] 5.2 Run `uv run pytest --collect-only` — all 7+ test modules collect clean, zero import errors
- [x] 5.3 Run `uv run pytest tests/test_public_api.py` — passes without GPU/dataset

## Phase 6: Commit & PR

- [x] 6.1 Single commit: `refactor: rename STOODX class to STOODXDetector` (includes smoke test + README)
- [ ] 6.2 Create PR #4 stacked-to-main, base = main
