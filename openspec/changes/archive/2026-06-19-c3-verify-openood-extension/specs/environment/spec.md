# Delta for environment

> **Capability**: `environment` — MODIFIED by `c3-verify-openood-extension`.
> Transitions R3 (test collection) from DEFERRED (C1) to RESOLVED via a root `conftest.py` that
> re-adds `numpy.sctypes` (see the `openood-extension` shim requirement for the mechanism).
> R1 (uv resolution), R2 (top-level importability), R4 (reproducibility artifacts), and R5/R6
> (scope boundary) are untouched.

## MODIFIED Requirements

### Requirement: Test suite collects without import errors

The system MUST allow pytest to collect the full test suite — all 7 modules — without
environment-induced import errors, achieved via a root `conftest.py` that re-adds
`numpy.sctypes` (removed in NumPy 2.0) before any openood import, restoring `imgaug`
compatibility. Test **execution** (running collected tests) remains out of scope — only
collection.
(Previously: C1 DEFERRED this — `uv run pytest --collect-only` raised collection errors on 5
modules because the transitive `imgaug` (via the pinned `openood` dependency's `draem_preprocessor`)
uses `numpy.sctypes`, removed in NumPy 2.0; STOOD-X pins `numpy>=2.1.2` and `imgaug` is
unmaintained. The requirement is now satisfied by C3's test-infrastructure shim.)

#### Scenario: pytest collection succeeds across all 7 modules

- GIVEN `uv sync` has completed successfully AND the root `conftest.py` shim is present
- WHEN `uv run pytest --collect-only` is run from the repository root
- THEN pytest collects ALL 7 test modules with no collection/import errors
- AND the previously-failing 5 OpenOOD-pulling modules now collect
- AND the outcome reflects collection only, not test pass/fail
