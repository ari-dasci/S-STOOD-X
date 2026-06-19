# Design: c3-verify-openood-extension

> Store: `openspec` · Delivery: `force-chained` (C3 = PR #3, stacked-to-main, final) · **Structural (safe)**: test infra + type-annotation refactor + export declaration + structural test + docs. No statistics/numerics/XAI logic, no `eval_ood`.

## Technical Approach

Five surgical edits prove **model Z**. A root `conftest.py` re-adds `numpy.sctypes` (removed in NumPy 2.0) before any openood import, unblocking collection. `from __future__ import annotations` + `if TYPE_CHECKING:` removes the eager `feature_visualization → adapter` edge, so `FeatureExplanation` no longer pulls openood. `__all__` grows 2→3 symbols. A data/GPU-free contract test asserts `isinstance(BasePostprocessor)` + the 4 contract methods + config attrs — the isinstance gate OpenOOD's `Evaluator` uses for pre-built instances. README drops the now-false collection-failure note. Verified against installed source: `BasePostprocessor.__init__` is literally `self.config = config`; venv numpy 2.2.6 lacks `sctypes`; imgaug touches only `sctypes["float"/"int"/"uint"]`.

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| imgaug/numpy2 fix | root `conftest` `np.sctypes` shim | Exploration proved imgaug uses ONLY `np.sctypes` (3 keys) and OpenOOD's eager registry forces imgaug with no switch — skip-draem/numpy-downgrade/fork all infeasible or out of scope. Reversible, test-infra only. |
| Decoupling | `TYPE_CHECKING` + `__future__` | Import is annotation-only (line 10 param); lazy string, zero runtime change, statically resolvable. |
| Contract test boundary | structural isinstance only | `eval_ood` needs pretrained+datasets+GPU (prereq); isinstance+callable+attrs is the minimum verifiable "valid extension" claim. |
| Test config object | plain `dict` | `BasePostprocessor.__init__` does only `self.config = config`; adapter reads `self.config[k]`/`.get()` — dict satisfies both. |
| New test filename | `test_postprocessor_contract.py` | Orchestrator directive (snake_case); deviates from mixedCase `test_postProcessor_*` siblings — flagged in Open Questions. |

## Data Flow

```
pytest collect → root conftest.py → np.sctypes shim → openood import OK
import STOODX  → feature_visualization (TYPE_CHECKING: no eager import) → adapter NOT loaded
tests/_openood_adapter/test_postprocessor_contract.py → STOODXPostprocessor(cfg) → isinstance(BasePostprocessor)
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `conftest.py` (root) | Create | `np.sctypes` shim; runs before any test module. Resolves env R3. |
| `src/STOODX/feature_visualization.py` | Modify | `from __future__ import annotations` (L1); guard L7 import under `if TYPE_CHECKING:`. Resolves C2 deferral #2. |
| `src/STOODX/__init__.py` | Modify | `__all__` 2→3 symbols + import `FeatureExplanation`. |
| `tests/_openood_adapter/test_postprocessor_contract.py` | Create | structural contract test (no data/GPU). |
| `README.md` | Modify | rewrite known-limitations L102-118; L171 comment → 3 symbols. |

## Interfaces / Contracts

### A. Root `conftest.py` (exact)
```python
"""Pytest configuration: NumPy 2.0 compatibility shim for imgaug.

imgaug (transitive via OpenOOD's draem_preprocessor) uses ``numpy.sctypes``,
removed in NumPy 2.0. imgaug is unmaintained (last release 2021, see
imgaug issue #595). This shim re-adds ``np.sctypes`` so collection succeeds
under numpy>=2.1.2. Long-term fix: the OpenOOD fork (out of scope here).
"""
import numpy as np

if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": {np.int8, np.int16, np.int32, np.int64},
        "uint": {np.uint8, np.uint16, np.uint32, np.uint64},
        "float": {np.float16, np.float32, np.float64},
        "complex": {np.complex64, np.complex128},
        "others": {bool, object, bytes, str},
    }
```
> numpy 1.x used lists incl. `longdouble`/`void`; the only consumer (imgaug) reads `float/int/uint` and `set()`-wraps, so provided sets are equivalent. `hasattr` guard = no-op on numpy<2.

### B. `feature_visualization.py` decoupling (exact edit)
- L1: `from __future__ import annotations`
- add `from typing import TYPE_CHECKING`; wrap the existing adapter import:
```python
if TYPE_CHECKING:
    from ._openood_adapter.postprocessor import STOODXPostprocessor
```
The annotation becomes a lazy string → never evaluated → no openood pull-in.

### C. `__init__.py` (exact)
```python
"""STOOD-X: nonparametric statistical OOD detection + XAI."""
from .stoodx import STOODX
from .feature_extractor import FeatureExtractor
from .feature_visualization import FeatureExplanation

__all__ = ["STOODX", "FeatureExtractor", "FeatureExplanation"]
```

### D. Contract test (structure)
Minimal config (required keys only, exact spelling incl. upstream typos `quantil`/`atribut`):
```python
{"K": 5, "distance": "cosine", "feature_name": "layer4",
 "intraclass": False, "quantil": 0.75, "atribut": False}
```
Then `adapter = STOODXPostprocessor(cfg)` and assert: `isinstance(adapter, BasePostprocessor)`; `callable(getattr(adapter, m))` for `m in ("__init__","setup","postprocess","inference")`; `hasattr(adapter, a)` for `a in ("K","distance","feature_name","device","oodTest")`; `adapter.oodTest is None`. Methods NOT called with data (need net/loaders); `torch.device(...)` falls back to CPU — no GPU.

## Apply Sequence (ordered)

- **A** — create root `conftest.py`.
- **B** — edit `feature_visualization.py` (`__future__` + `TYPE_CHECKING`).
- **C** — edit `__init__.py` (3-symbol `__all__` + import).
- **D** — create `tests/_openood_adapter/test_postprocessor_contract.py`.
- **E** — edit `README.md`: rewrite known-limitations (drop "5 modules fail collection"; add shim + imgaug-rot note + "full `eval_ood` runs are manual/prereq: pretrained + datasets + GPU"); L171 comment → 3 symbols.
- **F** — gates (all pass): **F-1** 3-symbol import keeps `'openood' out of sys.modules` (`from STOODX import STOODX, FeatureExtractor, FeatureExplanation`); **F-2** `from STOODX.feature_visualization import FeatureExplanation` keeps openood out (decoupling); **F-3** `uv run pytest --collect-only` → ALL 7 modules, zero import errors; **F-4** `uv run pytest tests/_openood_adapter/test_postprocessor_contract.py` → PASS.
- **Commit** — single work-unit: `feat(openood): verify extension contract, decouple core, resolve imgaug collection`. No AI attribution.

## Verification Mapping (specs → steps)

| Requirement | Step |
|-------------|------|
| `openood-extension` contract conformance (isinstance + 4 methods + attrs, no data/GPU) | D, F-4 |
| `openood-extension` Evaluator acceptance / import path resolves | D, F-4 |
| `openood-extension` np.sctypes shim, 7 modules collect | A, F-3 |
| `openood-extension` scope boundary (eval_ood manual in README) | D, E |
| `environment` R3 DEFERRED→RESOLVED (collection) | A, F-3 |
| `package-structure` R4 3-symbol API, no openood pull-in | C, F-1 |
| `package-structure` ADDED feature_visualization decoupled | B, F-2 |

## Testing Strategy

| Layer | What | Approach |
|-------|------|----------|
| Structural unit | adapter contract | `test_postprocessor_contract.py` (isinstance+callable+attrs, no data) |
| Collection | import health | `pytest --collect-only` → 7 modules |
| Invariant | standalone core | F-1/F-2 assert `'openood' not in sys.modules` |
| E2E (manual, OUT of gate) | `eval_ood` AUROC | README prereq (pretrained+datasets+GPU) |

## Migration / Rollback

No data migration; no lockfile/dep changes (numpy stays `>=2.1.2`, openood stays rev `3c35632`). Single `git revert` (one commit) unwinds the shim file, 3-line decouple, `__all__`/import, new test, and README edit.

## Risk Handling

- **Contract construction fails**: verified `BasePostprocessor.__init__` is `self.config = config` reading only dict ops → plain dict works. If a future openood pin needs a yacs `CfgNode`, use `CfgNode(MINIMAL_CONFIG)` inline. Last resort: assert method presence on a stub instance.
- **Shim masks upstream rot**: comment cites imgaug version + issue #595; fork is the long-term fix (out of scope).
- **R4 invariant regression**: F-1 re-asserts no-openood WITH 3 symbols.

## Open Questions

- [ ] Accept snake_case `test_postprocessor_contract.py` vs rename to `test_postProcessor_contract.py` for sibling consistency — non-blocking.
