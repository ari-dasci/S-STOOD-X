# Exploration: Rename the public `STOODX` class (C4)

> Read-only investigation. No source code modified. Scope: distinguish the **class**
> `STOODX` from the **package** `STOODX`, map every reference, assess rename targets.

## 1. The shadowing problem (why C4 exists)

- **Package** = directory `src/STOODX/` (distribution name `stood-x` in `pyproject.toml`).
- **Class** = `class STOODX` defined in `src/STOODX/stoodx.py:9`, re-exported from `__init__.py`.

Because package and class share the name `STOODX`:
- `import STOODX` binds the **package**; `STOODX.STOODX` resolves to the **class**
  (`__module__ == "STOODX.stoodx"`, per C2 verify-report:148).
- Inspecting `__all__` requires an alias: `import STOODX as pkg; pkg.__all__`
  (C2 tasks.md:74 already documents this workaround).
- It is behaviorally benign (no bug) but a recurring paper-cut for tooling, REPL use,
  and `dir()`/introspection. C2 explicitly deferred the class rename (decision **D7**);
  C4 is that deferred work.

## 2. Class responsibility (read from source)

`src/STOODX/stoodx.py` — `class STOODX` (docstring: *"class for OOD Test detector"*):

> The **core OOD detector**. It wraps a `FeatureExtractor`, accumulates a validation
> feature store (`addFeatures` / `finalizeFeatures`), and runs a nonparametric OOD test
> (`test()`): a Wilcoxon signed-rank comparison between the query's k-NN distance
> distribution and each neighbor's own k-NN distance distribution, returning p-values.

Evidence it IS a detector (not an explainer/orchestrator):
- README quick-start binds it to a variable literally named `detector`
  (`README.md:179`: `detector = STOODX(...)`).
- The OpenOOD adapter instantiates it as `self.oodTest` (`postprocessor.py:62`),
  i.e. the "OOD test" the adapter delegates to.

## 3. Import graph & usage map (every reference to the **class** `STOODX`)

Legend — `[CLASS]` = the bare `STOODX` class (in scope); `[PKG]` = the package (out of scope);
`[ADAPTER]` = `STOODXPostprocessor` (different class, NOT renamed — C2 D7 locked).

### 3a. Class definition (1 site)
| File:line | Reference | Kind |
|---|---|---|
| `src/STOODX/stoodx.py:9` | `class STOODX:` | **definition** |

### 3b. Internal references inside `src/STOODX/` (4 sites)
| File:line | Reference | Kind |
|---|---|---|
| `src/STOODX/__init__.py:2` | `from .stoodx import STOODX` | re-export `[CLASS]` |
| `src/STOODX/__init__.py:6` | `__all__ = ["STOODX", "FeatureExtractor", "FeatureExplanation"]` | string literal `[CLASS]` |
| `src/STOODX/_openood_adapter/postprocessor.py:18` | `from ..stoodx import STOODX` | import `[CLASS]` |
| `src/STOODX/_openood_adapter/postprocessor.py:62` | `self.oodTest = STOODX(model=feature_extractor, ...)` | **the only instantiation site** `[CLASS]` |

### 3c. Tests (`tests/`) — **ZERO direct references**
No test imports or instantiates the bare `STOODX` class. The detector is exercised only
indirectly through the adapter (`STOODXPostprocessor`) in the gated OpenOOD integration
tests, which are skipped without GPU/data:
- `tests/_openood_adapter/test_postProcessor_{cifar10,cifar100,imagenet,imagenet200}.py` → all use `[ADAPTER]` only.
- `tests/_openood_adapter/test_postprocessor_contract.py` → `[ADAPTER]` only (its `getattr` at :46 iterates *method names*, not the class name — not affected).
- `tests/test_feature_extractor.py` → `FeatureExtractor` only.
- `tests/test_feature_visualization.py` → `[ADAPTER]` + `FeatureExplanation`.

> ⚠️ **Coverage gap:** there is no unit test that does `from STOODX import STOODX` or
> asserts anything about the class itself. The rename therefore has **weak automated
> coverage**; recommend the proposal/tasks phase add a direct import smoke test.

### 3d. The OpenOOD adapter — does it reference the class?
**Yes, it instantiates it** (`postprocessor.py:62`, see 3b). That is the sole internal
construction. `feature_visualization.py` does NOT touch the bare class — every
`self.STOODXPostprocessor` there is an *attribute* on `FeatureExplanation` pointing at the
**adapter**, not the detector.

### 3e. Docs / README (4 sites, mixed `[CLASS]`/`[PKG]`)
| File:line | Reference | Kind |
|---|---|---|
| `README.md:168` | `from STOODX import STOODX, FeatureExtractor` | 1st=`[PKG]`, 2nd=`[CLASS]` |
| `README.md:179` | `detector = STOODX(` | `[CLASS]` instantiation |
| `README.md:213` | `# Public API: STOODX, FeatureExtractor, FeatureExplanation` | comment `[CLASS]` |
| `README.md:214` | `# Main STOOD-X implementation` | comment (module) — optional |

### 3f. Spec / config artifacts (mutable in this change)
| File:line | Reference | Kind |
|---|---|---|
| `openspec/specs/package-structure/spec.md:54,62,64,65` | public-API requirement + `from STOODX import STOODX,...` / `__all__==["STOODX",...]` scenarios | `[CLASS]` — becomes the C4 delta spec |
| `openspec/config.yaml:13-14` | "STOODX/ = core (STOODX detector, …)" | prose `[CLASS]` — optional doc tweak |

### 3g. Explicitly NOT in scope / NOT affected
- `openspec/changes/archive/**` — immutable history (C1/C2/C3 reports mention `STOODX`
  heavily). Never edited.
- `pyproject.toml` — distribution name `stood-x`, package discovery by directory
  (`[tool.setuptools.packages.find] where=["src"]`). Untouched by a *class* rename.
- `configs/postprocessors/mds.yml` — references OpenOOD registry name `mds`, not the class.
- All `STOODXPostprocessor` usages (different class, C2 D7 locked).

## 4. Candidate new names

| # | Name | `from STOODX import …` reads as | Pros | Cons | Effort |
|---|------|---------------------------------|------|------|--------|
| 1 | **`STOODXDetector`** | `STOODXDetector` | Preserves brand prefix; self-documenting (it IS a detector); matches README's `detector` var; mirrors sibling `STOODXPostprocessor` → clean `STOODX*` brand pair; role-suffix convention (`*Detector`/`*Postprocessor`/`*Extractor`) | `STOODX.STOODXDetector` mildly repetitive (but no longer *shadowing*) | Low |
| 2 | `OODDetector` | `OODDetector` | Generic, clean, short | Loses brand identity; collision risk on `import *` | Low |
| 3 | `Detector` | `Detector` | Minimal | Too generic; loses all brand context | Low |
| 4 | `STOODXTest` | `STOODXTest` | Matches docstring "OOD Test" | "Test" collides with unit-test semantics (`STOODX.STOODXTest` in a `tests/` file is confusing) | Low |
| 5 | `BayesOODDetector` / `WilcoxonOODDetector` | algorithm name | Descriptive of method | Over-specifies; algorithm may evolve; long | Med |

### Recommendation: **`STOODXDetector`**

Reasoning:
1. **It is a detector** — the README's own variable name and the adapter's `self.oodTest`
   both confirm the role.
2. **Eliminates the shadowing** — package `STOODX` and class `STOODXDetector` no longer
   collide; `dir(STOODX)` and `STOODX.__all__` become inspectable without an alias.
3. **Brand + symmetry** — pairs naturally with `STOODXPostprocessor` (the OpenOOD adapter
   wrapper). The public surface then reads as two coherent pairs:
   `STOODX*` (`STOODXDetector`, `STOODXPostprocessor`) and `Feature*`
   (`FeatureExtractor`, `FeatureExplanation`).
4. **Minimal churn** — the recognizable `STOODX` token survives as a prefix, so existing
   mental models / snippets stay recognizably related.
5. **Ecosystem fit** — OpenOOD itself uses role-suffix naming (`KNNPostprocessor`,
   `MDSPostprocessor`, `OdinPostprocessor`); `STOODXDetector` + `STOODXPostprocessor`
   follow the same idiom.

## 5. Breaking-change surface

- **Pre-publication** → breaking is explicitly acceptable (user pref: *"ahora barato /
  después caro"*). No external consumers known.
- **Public API change**: `from STOODX import STOODX` → `from STOODX import STOODXDetector`;
  `STOODX.__all__` updated.
- **Internal**: `postprocessor.py` import (line 18) + instantiation (line 62).
- **Docs**: README quick-start + structure comment.
- **Specs**: `package-structure` public-API requirement/scenario updated via the C4 delta
  spec (then synced to `openspec/specs/` at archive time).
- **No serialization migration needed** — `STOODX` instances are never pickled by the
  adapter (only their `.feats`/`.classes` *tensors* are `torch.save`d at
  `postprocessor.py:103-110`).

## 6. Gotchas / string references / introspection risks

- **String literal in `__all__`** (`__init__.py:6`) — must be updated in lockstep with the
  class name. This is the only string reference to the class by name.
- **No dynamic dispatch** — grep found no `importlib` / `__import__` / class-name registry
  pointing at `"STOODX"`. The one `getattr` (`test_postprocessor_contract.py:46`) iterates
  method names, not the class. **Clean.**
- **`configs/` is unrelated** — `mds.yml` keys OpenOOD's own postprocessor registry by
  `name: mds`, decoupled from the Python class name.
- **`__name__` introspection** — no code reads `STOODX.__name__` or `__qualname__`. Safe.
  (`__module__` stays `"STOODX.stoodx"` regardless, since the module file isn't renamed.)
- **Pickle caveat (documentation only)** — if any *external* pipeline ever pickled a
  `STOODX` instance, the stored `STOODX.stoodx.STOODX` qualifier would break on unpickle
  post-rename. The adapter never does this; flagged only for completeness. Acceptable
  pre-publication.
- **Pre-existing doc drift (NOT C4's job)** — `stoodx.py:15` docstring still says
  `model : FeatureEstractor` (typo + stale type); README method-body drift
  (`addFeatures`/`finalizeFeatures`/`test`) flagged out-of-scope in C2. The rename touches
  the same README region — keep the change surgical and do NOT bundle these fixes.
- **Coverage gap** — see §3c; add a direct-import smoke test in the tasks phase.

## 7. Recommendation summary

- **Rename target:** `STOODX` → **`STOODXDetector`**.
- **Scope (files to edit):**
  1. `src/STOODX/stoodx.py` (class def, line 9; docstring optional refinement).
  2. `src/STOODX/__init__.py` (import line 2 + `__all__` line 6).
  3. `src/STOODX/_openood_adapter/postprocessor.py` (import line 18 + instantiation line 62).
  4. `README.md` (quick-start lines 168, 179; comment line 213).
  5. `openspec/specs/package-structure/spec.md` (public-API requirement + scenarios) — via delta spec.
  6. *(optional)* `openspec/config.yaml:13-14` prose.
- **Out of scope:** `STOODXPostprocessor` class (locked), `feature_visualization.py`,
  all `tests/_openood_adapter/*` and `tests/test_feature_*`, `pyproject.toml`, `configs/`,
  archived `openspec/changes/archive/**`.
- **Add in tasks phase:** a direct import/identity smoke test
  (`from STOODX import STOODXDetector; assert STOODXDetector.__module__ == "STOODX.stoodx"`)
  to close the coverage gap, since no test currently exercises the class directly.

## Ready for Proposal
**Yes.** The change is small, mechanical, well-bounded, and the rename target is decided.
The `sdd-propose` phase should formalize: intent (kill package/class shadowing), scope
(the 5–6 files above), approach (single rename + delta-spec update + add a smoke test),
and the no-behavior-change non-goal (no numerics/logic/algorithm touched).
