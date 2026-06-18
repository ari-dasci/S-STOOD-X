# Archive Report — c1-uv-migration

> **Archived**: 2026-06-18
> **Status**: PASS_WITH_DEFERRALS
> **Artifact store**: openspec

## Change

**c1-uv-migration** — migrated environment and dependency management to uv (pyproject.toml, .python-version, uv.lock, README.md).

## Commits

| SHA | Description |
|-----|-------------|
| `4888ba8` | build: migrate environment management to uv (pyproject.toml, .python-version, uv.lock, README.md, spec.md) |
| `cd3e4c9` | docs(sdd): correct torch index wording, record c1 verify trail |

## Verification Status

PASS_WITH_DEFERRALS — all in-scope requirements (R1, R2, R4, R5) independently verified.
No CRITICAL findings.

### Warnings Resolved

- **W1** (torch "CPU build" wording imprecise): Fixed in commit `cd3e4c9` — README and spec reworded to "default PyPI torch (reproducible)".
- **W2** (tasks.md checkboxes not reconciled): Fixed in commit `cd3e4c9` — all completed tasks marked `[x]`, task 3.3 marked `[-]` (DEFERRED).

### Suggestions Carried Forward

- **S1** (design.md does not document `[tool.uv.extra-build-dependencies] libmr`): Not addressed; non-blocking. Can be recorded in a future change if desired.

## Deferred Items Carried Forward

| Item | Root Cause | Target Change |
|------|-----------|---------------|
| R3 — pytest collection (5 modules: test_featureVisualization, test_postProcessor_{cifar10,cifar100,imagenet,imagenet200}) | `imgaug` (transitive via openood's `draem_preprocessor`) uses `np.sctypes`, removed in NumPy 2.0; `imgaug` is unmaintained; STOOD-X pins `numpy>=2.1.2` | C2 or C3 |

## Specs Synced

| Domain | Action | Details |
|--------|--------|---------|
| environment | Created (seed) | Full spec — 5 requirements (R1–R5), 8 scenarios. R3 annotated DEFERRED with preserved annotation. |

## Archive Contents

- proposal.md
- specs/environment/spec.md (delta)
- design.md
- tasks.md (all implementation tasks marked complete; 3.3 DEFERRED)
- verify-report.md
- archive-report.md (this file)

## Next Steps

- Begin **C2** (structural refactor: src/ layout, `__init__.py`, adapter extraction) — unblocked by C1's reproducible environment.
- Resolve the imgaug/numpy-2 pytest collection issue (either via dependency override, imgaug fork, or OpenOOD pin update).
