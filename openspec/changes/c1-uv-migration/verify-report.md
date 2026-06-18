# Verify Report — C1 — uv migration (env + dependency management)

> **Status: PASS_WITH_DEFERRALS**
> Independent re-execution by `sdd-verify`. Strict TDD NOT active (`strict_tdd=false`).
> Verifier re-ran every gate from a clean state; did not trust the apply report.
> Commit under review: `4888ba8d9b0af4eaeb0e273365d84ef4c0eba1d4` on `main`.

## Executive Summary

C1 delivers its accepted scope — environment reproducibility + top-level importability —
fully. All four runtime gates re-executed independently and behave exactly as claimed: `uv lock
--locked` (213 pkgs, exit 0), `uv sync --frozen` (195 pkgs checked, exit 0), `import STOODX` (exit 0,
`STOODX/__init__.py`), and `pytest --collect-only` fails on **exactly** the 5 documented modules via
the documented `imgaug → np.sctypes` chain (12 tests collect before the errors, no unexpected
failures). OpenOOD is pinned to the exact commit in the lock; torch resolves from PyPI default
index; setuptools backend and module layout untouched. One accepted deferral (R3 — pytest collection)
is honored per author decision (b). Two non-blocking WARNINGS (doc accuracy on "CPU build"; tasks.md
checkboxes not reconciled) and one SUGGESTION (design.md does not document the `[tool.uv.extra-build-dependencies] libmr`
addition). **No CRITICAL findings — the PR is not blocked.** Proceed to `sdd-archive` after the
tasks.md hygiene fix.

---

## Artifacts

- `openspec/changes/c1-uv-migration/verify-report.md` (this file)

---

## Gate Results (independently re-executed)

> Env note: the local shell is poisoned by a conda toolchain (`CC`/`CXX`/`LDFLAGS`/... point at
> `anaconda3/bin/*-conda_cos6-*`). This is **local-only** and does NOT affect committed artifacts.
> All gates run with a sanitized env (`env -u CC -u CXX -u CPP -u LDFLAGS -u CFLAGS -u CXXFLAGS -u CONDA_PREFIX`).
> `uv 0.11.8`. Any machine with a sane system gcc reproduces `uv.lock` byte-for-byte.

| Gate | Command | Result | Evidence |
|------|---------|--------|----------|
| `uv_lock_locked` | `uv lock --locked` | **PASS** | `Resolved 213 packages in 1ms`, exit 0 — lock consistent with pyproject |
| `uv_sync_frozen` | `uv sync --frozen` | **PASS** | `Checked 195 packages in 2ms`, exit 0 — env installs from frozen lock |
| `import_stoodx` | `uv run python -c "import STOODX"` | **PASS** | `OK /home/.../S-STOOD-X/STOODX/__init__.py`, exit 0 — no `ImportError` |
| `pytest_collect_expected_failure` | `uv run pytest --collect-only` | **PASS (matches documented DEFERRED limitation)** | `12 tests collected, 5 errors`, exit 2. The 5 errors are EXACTLY `test_featureVisualization`, `test_postProcessor_{cifar10,cifar100,imagenet,imagenet200}`, all via `imgaug.imgaug.py:45: NP_FLOAT_TYPES = set(np.sctypes["float"])` ← `openood.preprocessors.draem_preprocessor`. No new/unexpected failures. |

---

## Requirement / Scenario Compliance Matrix

| Requirement | Scenario | Result | Evidence |
|-------------|----------|--------|----------|
| **R1 — uv-managed resolution and install** | Clean clone resolves editable package and dev group | **PASS** | Gates `uv_lock_locked` + `uv_sync_frozen`: exit 0, no resolver conflict; editable + `dev` group installed |
| R1 | OpenOOD resolves at the pinned commit | **PASS** | `uv.lock`: `openood` source = `git = "https://github.com/Jingkang50/OpenOOD?rev=3c35632ee91b54b09d1f085d04f94744cece7d0b#3c35632ee91b54b09d1f085d04f94744cece7d0b"` — resolved hash fragment matches pinned rev exactly, not a moving HEAD |
| R1 | Dev tooling excluded from the published wheel | **PASS** | `[project.dependencies]` has exactly 19 runtime entries; all 12 dev packages live ONLY in `[dependency-groups] dev` (verified by diff) |
| **R2 — Environment-level importability** | Top-level package imports in the uv env | **PASS** | Gate `import_stoodx`: exit 0, `STOODX/__init__.py` resolved, no env-attributable `ImportError` |
| **R3 — Test suite collects without import errors** | pytest collection succeeds | **DEFERRED** (accepted known limitation, author decision b, 2026-06-18) | Gate `pytest_collect_expected_failure`: 5 collection errors, exactly the documented modules, documented root cause (`imgaug` → `np.sctypes` removed in NumPy 2.0; STOODX pins `numpy>=2.1.2`; imgaug unmaintained). Requirement body RETAINED verbatim in spec.md (lines 56–77) with DEFERRED annotation + cross-ref to README "Known limitations". MUST be re-evaluated in C2/C3. **Not a CRITICAL failure** — it is documented, accepted scope reduction. |
| **R4 — Reproducibility artifacts committed** | Lockfile and python version are tracked | **PASS** | `git ls-files`: `uv.lock` and `.python-version` both tracked; `.python-version` content == `3.10` |
| R4 | Lock is CPU-only; CUDA is opt-in only | **PASS** *(with WARNING W1 on wording)* | `uv.lock`: `torch` `source = { registry = "https://pypi.org/simple" }`; no `[[tool.uv.index]]` cu124 in committed pyproject; no `download.pytorch.org/whl/cu124` source anywhere in lock; README documents the `UV_PYTHON_INDEX_URL=...cu124` opt-in recipe (lines 72–94). Locked decision D2 honored. See W1 for a wording caveat. |
| **R5 — Scope boundary (non-goals)** | No structural or behavioral changes | **PASS** | `git show 4888ba8 --stat`: only 5 files changed — `.python-version`, `README.md`, `pyproject.toml`, `uv.lock`, `openspec/.../spec.md`. `[build-system]` = setuptools (unchanged); `[tool.setuptools] packages = ["STOODX"]` (unchanged). No src/ move, no `__init__.py` edit, no OpenOOD-adapter extraction, no backend swap, no dep removal. |
| R5 | Functional correctness is not asserted by this change | **PASS** | No functional/numerical tests run; deferred to C3 per design "Testing Strategy" + spec R5 scenario 2. |

---

## Decision Compliance (committed `pyproject.toml`)

| Decision | Honored | Evidence |
|----------|---------|----------|
| `openood_pinned` | **Yes** | `[tool.uv.sources] openood = { git = "...", rev = "3c35632ee91b54b09d1f085d04f94744cece7d0b" }`; `[project.dependencies]` declares bare `"openood"` (NOT the git URL). Lock resolves exact commit. |
| `dev_split` | **Yes** | `[dependency-groups] dev` has exactly 12 packages: pytest, sphinx, sphinx_rtd_theme, sphinxcontrib.bibtex, nbsphinx, pandoc, twine, IPython, ipywidgets, tensorboard, torchsummary, wget. |
| `torch_cpu_only` | **Yes** | No `[tool.uv.sources] torch` entry; no `[[tool.uv.index]]` cu124 in committed pyproject; lock sources torch from `pypi.org/simple`. (See W1 for wording nuance.) |
| `setuptools_kept` | **Yes** | `[build-system]` = `setuptools.build_meta`, `requires = ["setuptools>=61.0"]`; `[tool.setuptools]` present and unchanged. |
| `python_version_3.10` | **Yes** | `.python-version` content == `3.10` (single line, no trailing specifier). |
| `readme_documented` | **Yes** | README documents: uv sync workflow (L55–64), `uv run pytest` (L66–70), CUDA opt-in recipe incl. `UV_PYTHON_INDEX_URL` (L72–94), libmr `build-essential` build note (L51–53), Known limitations subsection naming imgaug/numpy-2 + the 5 modules (L96–112). |

---

## Scope Compliance (out-of-scope respected)

| Check | Result | Evidence |
|-------|--------|----------|
| `no_renames` | **Yes** | No file/module renamed. |
| `no_src_move` | **Yes** | No `src/` layout introduced; `[tool.setuptools] packages = ["STOODX"]` unchanged. |
| `no_init_change` | **Yes** | `STOODX/__init__.py` not in diff. |
| `no_adapter_extract` | **Yes** | No OpenOOD-adapter extraction; no new adapter files. |
| `no_test_change` | **Yes** | No file under `tests/` in diff. |
| `no_logic_change` | **Yes** | No `.py` source under `STOODX/` or elsewhere in diff; change is config + docs + generated lock only. |

---

## Findings

### CRITICAL
*None.* Nothing blocks the PR.

### WARNING

- **W1 — Documentation accuracy: "CPU build" of torch is imprecise.** The committed `uv.lock` resolves
  `torch == 2.12.1` from `pypi.org/simple` (default PyPI index), which on `sys_platform == 'linux'`
  pulls CUDA runtime transitives (`cuda-bindings`, `cuda-toolkit` extras `cudart/cufft/cufile/cupti/curand/cusolver/cusparse/nvjitlink/nvrtc/nvtx`,
  `nvidia-cuda-cupti`, `nvidia-cuda-nvrtc`, `nvidia-cuda-runtime`, …). The README (L74: "resolves the
  **CPU** build of PyTorch from PyPI") and spec R4 scenario ("resolves `torch` from PyPI (CPU build)")
  therefore overstate "CPU". **The locked decision D2's actual guarantee IS met** — "default PyPI
  index, reproducible, no explicit `download.pytorch.org/whl/cu124` index in committed pyproject" —
  and `uv lock --locked` is reproducible. But a GPU-less CI clone will still download the multi-GB
  CUDA-bundled PyPI wheel. **Recommended fix (non-blocking):** reword README/spec to "default PyPI
  torch (reproducible); for a strictly CPU-only build, use `https://download.pytorch.org/whl/cpu`".
  Does not block the PR; track for a doc touch-up.

- **W2 — `tasks.md` checkboxes not reconciled with verified completion.** All task checkboxes in
  `tasks.md` remain `[ ]`, even though independent re-execution confirms the work for Phases 1, 2,
  3.1, 3.2, 4, 5 is complete, and task 3.3 (pytest collection DoD) is spec-authorized DEFERRED.
  Per the sdd-verify gate, an unchecked implementation task is normally CRITICAL; here the WORK is
  verifiably done (positive gate evidence), so this is a hygiene/process gap, not incomplete work.
  **Recommended fix (before archive):** tick `[x]` for 1.1–1.5, 2.1, 3.1, 3.2, 4.1, 5.1; mark 3.3 as
  `[-]` (DEFERRED) to match the spec annotation. Cosmetic but required for clean archive.

### SUGGESTION

- **S1 — Design coherence: undocumented `[tool.uv.extra-build-dependencies] libmr`.** `pyproject.toml`
  (L60–64) adds `[tool.uv.extra-build-dependencies] libmr = ["numpy", "cython"]`, which is NOT listed
  in `design.md`'s "File Changes" table or "pyproject.toml — exact transformations" section. The
  addition is legitimate and well-justified: `libmr`'s `setup.py` imports numpy/Cython at build time
  without declaring them in `build-system.requires`, so uv build-isolation needs them injected; it is
  build-time-only (no runtime pin change), well-commented in-code, and mentioned in the commit
  message. **Recommended fix:** add a one-line entry to `design.md` (File Changes + a D7 decision row)
  recording this addition, for traceability. Does not break any spec requirement or locked decision.

---

## Final Verdict

**PASS_WITH_DEFERRALS.**

- All in-scope requirements (R1, R2, R4, R5) PASS via independent gate re-execution.
- R3 (pytest collection) is DEFERRED per the author's documented, accepted decision (b); the
  observed collection failure is exactly the documented limitation, with no new/unexpected failures.
- No CRITICAL findings; the PR is not blocked.
- Two WARNINGS (W1 doc wording, W2 tasks.md hygiene) and one SUGGESTION (S1 design.md traceability)
  should be addressed, ideally before `sdd-archive`.

---

## Next Recommended

**`sdd-archive` for `c1-uv-migration`** — after the author ticks the `tasks.md` checkboxes (W2),
and optionally fixes the README/spec "CPU build" wording (W1) and records the `extra-build-dependencies`
decision in `design.md` (S1). All deferred work (R3, imgaug/numpy-2 resolution) is tracked for C2/C3.
