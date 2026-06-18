# Design: c2-structural-refactor

> PR #2 in the stacked-to-main chain (target `main`, after C1). Pure mechanical refactor —
> NO logic/numerics/XAI changes. All 10 decisions LOCKED in proposal. This design is
> **edit-level and ordered**: the apply phase executes Steps A–I verbatim.

## Technical Approach

Move `STOODX/` → `src/STOODX/`, rename 4 modules to PEP 8 `snake_case`, privatize the OpenOOD
adapter into `src/STOODX/_openood_adapter/`, populate a 2-symbol public API (`STOODX`,
`FeatureExtractor`), and re-point all **11 import lines** + **6 in-src class-name sites** via
`git mv` (history-preserving) + relative imports. `pyproject.toml` switches to
`packages.find`. The `feature_visualization → adapter` edge is **preserved** (decoupling = C3).

## Architecture Decisions (locked — rationale recorded)

| # | Decision | Choice | Why (alternatives rejected) |
|---|----------|--------|------------------------------|
| D1 | Layout | `src/` flat (single `src/STOODX/`) | PEP + setuptools `packages.find` auto-discovers `STOODX` + `STOODX._openood_adapter`. Rejected: namespace/monorepo (overkill for 1 pkg). |
| D2 | Adapter | underscore subpackage `_openood_adapter/` | Private signal + isolates OpenOOD seam. Rejected: top-level module (no privacy cue), sibling package (splits the lib). |
| D3 | Class rename | `FeatureStractor` → `FeatureExtractor` (option B) | Fixes the "Estractor" misspelling in public API. Rejected: keep name (perpetuates typo). |
| D4 | `__init__` exports | exactly `{STOODX, FeatureExtractor}` | Neither transitively imports `openood` → preserves env R2. `FeatureExplanation`/adapter EXCLUDED (would regress R2 via featureVisualization→adapter→openood). |
| D5 | Intra-pkg imports | relative (`from .`, `from ..`) | Conventional, decouples from top-level name. Rejected: keep absolute (works but non-idiomatic). |
| D6 | `feature_visualization↔adapter` edge | PRESERVED | Decoupling is C3 scope. Rejected: break now (scope creep, risks R9). |
| D7 | `STOODX`, `FeatureExplanation`, `STOODXPostprocessor` class names | UNCHANGED | Minimize API churn / collision risk (`crp.visualization.FeatureVisualization`). |
| D8 | `tests/_openood_adapter/` `__init__.py` | NOT created | pytest rootdir convention — adding one risks test-name collisions. |
| D9 | Build backend | setuptools (UNCHANGED) | Only discovery stanza changes. Rejected: hatchling (R9 non-goal). |
| D10 | `super(FeatureStractor,self)` & type hints | renamed too | Class rename is global within src; grep gate `FeatureStractor` enforces zero residue. |

## File Operations Map

| Op | From | To | Notes |
|----|------|----|-------|
| MOVE dir | `STOODX/` | `src/STOODX/` | `git mv` (history). `mkdir -p src` first. |
| RENAME | `src/STOODX/STOODX.py` | `src/STOODX/stoodx.py` | `git mv` |
| RENAME | `src/STOODX/featureStractor.py` | `src/STOODX/feature_extractor.py` | `git mv` + class rename inside |
| RENAME | `src/STOODX/featureVisualization.py` | `src/STOODX/feature_visualization.py` | `git mv` + 1 import rewrite |
| MOVE+RENAME | `src/STOODX/STOODXPostprocessor.py` | `src/STOODX/_openood_adapter/postprocessor.py` | `mkdir _openood_adapter`; `git mv` |
| CREATE | — | `src/STOODX/_openood_adapter/__init__.py` | docstring-only (no re-export) |
| EDIT | `src/STOODX/__init__.py` | (was empty) | 2 exports + `__all__` |
| EDIT | `pyproject.toml` | `[tool.setuptools]` → `[tool.setuptools.packages.find]` | discovery stanza |
| MOVE dir | `tests/test_postProcessor_{cifar10,cifar100,imagenet,imagenet200}.py` | `tests/_openood_adapter/test_postProcessor_{...}.py` | `mkdir`; `git mv` ×4. NO `__init__.py` (D8) |
| RENAME | `tests/test_featureStractor.py` | `tests/test_feature_extractor.py` | `git mv` + import + typo fix |
| RENAME | `tests/test_featureVisualization.py` | `tests/test_feature_visualization.py` | `git mv` + 2 imports |
| EDIT | `README.md` | Quick-start L126/129 + tree L168-183 | symbol + layout |
| DELETE | `STOODX/` (old) | — | implicit after MOVE |

## Import Rewrite Map (exhaustive — 11 import lines + class-rename sites)

### Intra-src → relative imports (4 lines, 3 files)

| File:line | Before | After |
|-----------|--------|-------|
| `src/STOODX/stoodx.py:3` | `from STOODX.featureStractor import FeatureStractor` | `from .feature_extractor import FeatureExtractor` |
| `src/STOODX/_openood_adapter/postprocessor.py:9` | `from STOODX.featureStractor import FeatureStractor` | `from ..feature_extractor import FeatureExtractor` |
| `src/STOODX/_openood_adapter/postprocessor.py:10` | `from STOODX.STOODX import STOODX` | `from ..stoodx import STOODX` |
| `src/STOODX/feature_visualization.py:7` | `from STOODX.STOODXPostprocessor import STOODXPostprocessor` | `from ._openood_adapter.postprocessor import STOODXPostprocessor` (**PRESERVE edge**) |

### In-src class-name rename sites (`FeatureStractor` → `FeatureExtractor`) — NOT imports

| File:line | Before | After |
|-----------|--------|-------|
| `src/STOODX/feature_extractor.py:5` | `class FeatureStractor(torch.nn.Module):` | `class FeatureExtractor(torch.nn.Module):` |
| `src/STOODX/feature_extractor.py:29` | `super(FeatureStractor, self).__init__()` | `super(FeatureExtractor, self).__init__()` |
| `src/STOODX/stoodx.py:21` | `def __init__(self, model: FeatureStractor,` | `def __init__(self, model: FeatureExtractor,` |
| `src/STOODX/_openood_adapter/postprocessor.py:53` | `feature_extractor = FeatureStractor(model=net, ...)` | `feature_extractor = FeatureExtractor(model=net, ...)` |

> Note: `stoodx.py:15` docstring typo `FeatureEstractor` is **out of scope** (NOT in the grep
> gate; fixing is optional doc cleanup). Leave it to respect the scope boundary, or fix as a
> pure-comment change — apply phase decides; gate is unaffected.

### Tests → new absolute imports (7 lines)

| File:line | Before | After |
|-----------|--------|-------|
| `tests/test_feature_extractor.py:4` | `from STOODX.featureStractor import FeatureStractor` | `from STOODX.feature_extractor import FeatureExtractor` |
| `tests/test_feature_extractor.py:26` | `feature_estractor = FeatureStractor(...)` | `feature_extractor = FeatureExtractor(...)` |
| `tests/test_feature_extractor.py:27` | `feature_estractor.feature_activations(...)` | `feature_extractor.feature_activations(...)` |
| `tests/test_feature_extractor.py:28` | `feature_estractor.atribute(...)` | `feature_extractor.atribute(...)` |
| `tests/test_feature_visualization.py:11` | `from STOODX.STOODXPostprocessor import STOODXPostprocessor` | `from STOODX._openood_adapter.postprocessor import STOODXPostprocessor` |
| `tests/test_feature_visualization.py:12` | `from STOODX.featureVisualization import FeatureExplanation` | `from STOODX.feature_visualization import FeatureExplanation` |
| `tests/_openood_adapter/test_postProcessor_{cifar10,cifar100,imagenet,imagenet200}.py:16` | `from STOODX.STOODXPostprocessor import STOODXPostprocessor` | `from STOODX._openood_adapter.postprocessor import STOODXPostprocessor` (×4 files) |

## Package `__init__.py` Contents (exact)

`src/STOODX/__init__.py`:
```python
"""STOOD-X: nonparametric statistical OOD detection + XAI."""
from .stoodx import STOODX
from .feature_extractor import FeatureExtractor

__all__ = ["STOODX", "FeatureExtractor"]
```

`src/STOODX/_openood_adapter/__init__.py` (docstring-only; NO re-export to keep private surface minimal):
```python
"""Internal OpenOOD adapter — NOT public API.

This subpackage bridges STOOD-X to the OpenOOD evaluation framework. Importing it
pulls `openood` and its (numpy-2-incompatible) transitive deps. Do not import from
outside the package; treat as private.
"""
```

`src/STOODX/_openood_adapter/postprocessor.py` — **add `__all__`** (recommended, explicit private API):
```python
__all__ = ["STOODXPostprocessor"]
```
(prepend a module docstring marking it internal/not public API — satisfies R5.)

## pyproject.toml Edit

Replace (L66-67):
```toml
[tool.setuptools]
packages = ["STOODX"]
```
with:
```toml
[tool.setuptools.packages.find]
where = ["src"]
```
Backend (`[build-system]` setuptools) and `[project] name = "stood-x"` UNCHANGED. `find`
auto-discovers `STOODX` + `STOODX._openood_adapter` under `src/`.

## Apply Sequence (ordered — sdd-apply executes verbatim)

- **Step A** — Move package to src layout:
  `mkdir -p src && git mv STOODX src/STOODX`
- **Step B** — Rename core modules + privatize adapter (all `git mv`):
  `git mv src/STOODX/STOODX.py src/STOODX/stoodx.py` ·
  `git mv src/STOODX/featureStractor.py src/STOODX/feature_extractor.py` ·
  `git mv src/STOODX/featureVisualization.py src/STOODX/feature_visualization.py` ·
  `mkdir -p src/STOODX/_openood_adapter && git mv src/STOODX/STOODXPostprocessor.py src/STOODX/_openood_adapter/postprocessor.py`
- **Step C** — Populate `src/STOODX/__init__.py` (2 exports + `__all__`) and create
  `src/STOODX/_openood_adapter/__init__.py` (docstring-only) + add docstring/`__all__` head of
  `postprocessor.py`.
- **Step D** — Rewrite the 4 intra-src imports → relative (table above) AND the 4 in-src
  class-rename sites (`FeatureStractor`→`FeatureExtractor`).
- **Step E** — Tests: `git mv` 2 renames + `mkdir tests/_openood_adapter` + `git mv` 4 postProcessor
  tests into it; rewrite 7 test imports; fix `feature_estractor` typo (L26-28). NO `__init__.py`.
- **Step F** — Edit `pyproject.toml` `[tool.setuptools]` → `[tool.setuptools.packages.find]`.
- **Step G** — `uv sync` (editable reinstall from `src/`). If stale `*.egg-info`/resolver error:
  `rm -rf stood_x.egg-info *.egg-info && uv sync` (or `uv sync --reinstall-package stood-x`).
- **Step H** — Edit `README.md`: Quick-start L126 `FeatureStractor`→`FeatureExtractor`, L129
  instantiation symbol; structure tree L168-183 → `src/STOODX/` + `_openood_adapter/` paths.
- **Step I** — Verification gates (see below). Pass → single work-unit commit
  (`refactor(structure): src/ layout, snake_case modules, private OpenOOD adapter`). ~15 files,
  all small → ONE commit (split only if >800 lines, unlikely).

## Verification Gates (Step I, run in order)

1. **Public API**: `uv run python -c "from STOODX import STOODX, FeatureExtractor; import STOODX; assert STOODX.__all__==['STOODX','FeatureExtractor']; print('OK')"`
2. **No openood pull-in** (preserves env R2): `uv run python -c "import STOODX; import sys; assert 'openood' not in sys.modules; print('clean')"`
3. **Zero residual old names** (DoD gate): `grep -rn "featureStractor\|featureVisualization\.py\|STOODXPostprocessor\.py\|FeatureStractor\|feature_estractor" src/ tests/ README.md` → **MUST return nothing**.
4. **Clean collection**: `uv run pytest --collect-only tests/test_feature_extractor.py` → collects, no `ImportError`.
5. **No NEW import errors**: `uv run pytest --collect-only` → the 5 openood-pulling modules may fail ONLY on imgaug/numpy2 (same root cause as C1), no `FeatureStractor`/missing-module typos.
6. **Install**: `uv sync` exit 0.

## Verification Mapping (spec R1–R9 → steps)

| Req | Requirement | Satisfied by |
|-----|-------------|--------------|
| R1 | src/ layout + discovery | A, B, F, G |
| R2 | snake_case module naming | B |
| R3 | Public class `FeatureExtractor`, no residual typo | D (class sites), E (test+typo), I-3 |
| R4 | 2-symbol API, no openood pull-in | C, I-1, I-2 |
| R5 | Adapter privatization (path + docstring) | B, C, D (preserve edge) |
| R6 | Exhaustive rename (grep zero) | D, E, H, I-3 |
| R7 | Import correctness (collection) | D, E, G, I-4, I-5 |
| R8 | README accuracy | H |
| R9 | Scope boundary (non-goals) | whole diff (no body-logic; edge preserved; backend unchanged) |

## Testing Strategy

| Layer | What | Approach |
|-------|------|----------|
| Static | grep zero hits, `__all__` literal | Gate I-3, I-1 |
| Env | `import STOODX` no openood | Gate I-2 (preserves env R2) |
| Collection | `test_feature_extractor.py` collects | Gate I-4 (only live-collectable test) |
| Live (deferred) | 5 openood modules | CANNOT run (imgaug/numpy2) — grep is sole safeguard |

## Rollback

Single-commit refactor → `git revert <c2-commit>` cleanly restores old `STOODX/`, mixedCase
names, empty `__init__.py`, and `[tool.setuptools] packages=["STOODX"]`. `uv sync` re-resolves to
the prior layout. No data migration; no irreversible state.

## Risks & Handling

- **`uv sync` fails after move** (stale `stood_x.egg-info`/SOURCES.txt at old path) → `rm -rf *.egg-info` then `uv sync` (egg-info is gitignored, auto-regenerates).
- **grep returns residual hits** → DoD FAILURE: fix before commit (re-run Gate I-3).
- **`import STOODX` pulls openood** → `__init__` over-exports; remove offending import (keep only `STOODX` + `FeatureExtractor`).
- **Invisible breakage in 5 un-collectable tests** → exhaustively enumerated import table (above) is the only safeguard; Gate I-3 enforces it statically.

## Open Questions

- [ ] `stoodx.py:15` docstring typo `FeatureEstractor` — fix (pure comment, zero behavior) or leave (out of grep gate)? Recommend: leave, to honor scope boundary; flag for a doc-cleanup change. Non-blocking.
