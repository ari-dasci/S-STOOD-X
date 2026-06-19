# Exploration: c5-linter-and-ci

> Read-only investigation. No code/config changed. Defines scope for the
> proposal phase. PR #5 (last in the C1–C5 stacked-to-main chain).

## 1. Codebase shape

| Area | Files | LOC |
|------|-------|-----|
| `src/STOODX/` (incl. `_openood_adapter/`) | 6 `.py` (4 substantive + 2 `__init__`) | 722 |
| `tests/` (incl. `_openood_adapter/`) | 8 `.py` | 1306 |
| **Total Python** | **14** | **~2028** |

Substantive modules: `stoodx.py` (272), `feature_visualization.py` (163),
`feature_extractor.py` (135), `_openood_adapter/postprocessor.py` (140).

**Non-Python files that matter for CI:**
- `pyproject.toml` — build/dep definition; `[dependency-groups] dev` exists (C1 uv migration). **No `[tool.ruff]`/`[tool.mypy]`/`[tool.black]` config exists.**
- `conftest.py` (root) — NumPy 2.0 `np.sctypes` shim for imgaug (transitive via OpenOOD). Must remain; ruff reports it CLEAN.
- `configs/postprocessors/mds.yml` — postprocessor config.
- `openspec/config.yaml` — SDD project config.
- `README.md`, `README.pdf`, `LICENSE`.

**No notebooks, no shell scripts** in the project tree (only inside the vendored `.conda/`).

**No `.github/` directory** — CI absent, confirmed.

**Runtime data (all gitignored — lines 164–171):** `data/`, `results*/`, `utils/*`, `pretrained_models/*`. None tracked by git → **full `eval_ood` tests CANNOT run in CI** (they need pretrained checkpoints + datasets + GPU). This is the central constraint shaping the CI test job.

## 2. Current quality surface

**ruff 0.15.16 is ALREADY runnable** via `uv run ruff` and `uvx ruff` — BUT it is **NOT declared in `[dependency-groups] dev`**, so it must be added explicitly for lockfile reproducibility (currently resolves via a transitive/global path).

### 2a. Violation statistics

`uv run ruff check src/ tests/ --select=E,F,W --statistics` → **136 errors** (48 auto-fixable):

| Rule | Count | Class | Notes |
|------|-------|-------|-------|
| E501 line-too-long | 71 | **NOISY** | Long ML/PyTorch idioms + LaTeX math in docstrings. Line-length distribution: 93 lines >88, **17 >120**. |
| F401 unused-import | 24 | **MEDIUM (needs review)** | 20 in test files — openood network imports (`ResNet50`, `ViT_B_16`, `Swin_T`, `RegNet_Y_16GF`), `torchvision`, `json`, `pandas`, `crp.*`. May be intentional registration/fixture imports — **do NOT blind-autofix**. |
| W293 blank-line-ws | 24 | NOISY | Cosmetic, auto-fixable. |
| W291 trailing-ws | 8 | NOISY | Cosmetic, auto-fixable. |
| **W605 invalid-escape** | **5** | **REAL (bug-class)** | `stoodx.py:156,160,162` — `\sum`, `\{`, `\}` in LaTeX-math docstrings. Produces `DeprecationWarning`; fix = raw-string docstrings (`r"""..."""`). Semantically safe (docstring only). |
| E402 import-not-at-top | 1 | review | Likely conditional import. |
| E711 `== None` | 1 | REAL | Use `is None`. |
| F541 f-string-no-placeholder | 1 | REAL | Auto-fixable. |
| F841 unused-variable | 1 | REAL | Auto-fixable. |

Default ruff profile (E4,E7,E9,F only) → **28 errors**, 25 auto-fixable (essentially the 24 F401 + the 4 real ones).

### 2b. Real vs noisy triage

- **REAL — enforce & fix in C5:** W605 (5), F541 (1), F841 (1), E711 (1) = **8 violations**, all low-risk. The W605 are genuine latent `DeprecationWarning`s.
- **MEDIUM — enforce but require author review before fix:** F401 (24). Confirm test-file network imports aren't registration side-effects. If confirmed unused, autofix; else add `# noqa: F401` with rationale.
- **NOISY — exclude or relax:** E501 (71), W293 (24), W291 (8). Recommend `line-length = 120` (drops E501 to 17) and letting `ruff format` absorb W29x on the one-time format pass rather than enforcing as separate checks.

### 2c. `ruff format` impact (budget-relevant)

`ruff format --check` → **13 of 14 files would be reformatted** (only `__init__.py` is clean). A one-time `ruff format` is **semantically safe** but produces a broad whitespace/wrapping diff (review-heavy, not logic-heavy). Acceptable within the 800-line budget; must be a conscious decision in the proposal.

## 3. mypy feasibility

**Partial, inconsistent typing.** ~10 of 29 `def`s in `src/` carry return annotations; many args annotated (`torch.Tensor`, `int`), many not (`dataset`, `x_`). `from typing import ...` appears in 3 files (`Callable`, `Any`, `TYPE_CHECKING`). The codebase returns `pd.DataFrame`, uses lambdas as defaults, and depends on `openood`/`torch` which lack complete stubs.

**Recommendation: DEFER strict mypy.** Enforcing mypy now would block on OpenOOD/torch stub noise and become a maintenance burden in the final PR of the chain. Two realistic options for the proposal:
- **(A) Defer entirely** to a future C6+ change (cleanest; recommended for MVP).
- **(B) Advisory-only** mypy in CI: `mypy --ignore-missing-imports src/` as **non-blocking** (`continue-on-error: true`), baseline of current state, no gate. Adds visibility without friction.

Either way, **strict mypy is out of scope for C5.**

## 4. CI design options (GitHub Actions)

### 4a. Jobs

| Job | Runs | Cost | Verdict |
|-----|------|------|---------|
| **lint** (`ruff check`) | always | seconds, no deps | ✅ IN — run via `uvx ruff` (no env sync needed) |
| **format-check** (`ruff format --check`) | always | seconds, no deps | ✅ IN — **only after** a one-time `ruff format` lands in C5 |
| **test (fast)** | always | minutes — needs `uv sync` (torch+openood) | ✅ IN — run ONLY the 2 hermetic tests (`test_public_api.py`, `test_postprocessor_contract.py`) |
| **test (full eval_ood)** | never in push CI | GPU + checkpoints + datasets (gitignored) | ⛔ DEFER — manual `workflow_dispatch` or skip |
| **mypy** | optional | seconds, no deps | 🔶 OPTIONAL advisory (see §3) |

**Test-job reality:** both hermetic tests import `STOODX` (needs `pip install -e .`) and the contract test imports `openood` (git dep). So the test job MUST `uv sync --dev` (pulls torch/openood, builds libmr via the existing `extra-build-dependencies` mechanism). The sync — not the tests — is the slow part; uv caching is important. The hermetic tests themselves are sub-second.

### 4b. Python matrix

`requires-python = ">=3.10"`, `.python-version = 3.10`. The OpenOOD git pin + `libmr` build + torch wheels are **unverified on 3.11/3.12**. Multi-matrix adds failure risk in the final PR.

**Recommendation: single Python 3.10** for the MVP. Note 3.11/3.12 as a future task once deps are verified.

### 4c. uv in CI & caching

Use **`astral-sh/setup-uv@v3`** — reads `.python-version` + `uv.lock` for reproducibility, and exposes `enable-cache: true` (caches `~/.cache/uv`, the torch/openood wheels). This is the standard path and matches the local `uv run` workflow.

### 4d. Approaches compared

| Approach | Pros | Cons | Effort |
|----------|------|------|--------|
| **Strict** (ruff + mypy strict + 3.10/3.11/3.12 + full tests) | max rigor | blocks on stubs/unverified matrix/GPU-data; high maintenance; wrong fit for research lib | High |
| **Pragmatic (MVP)** (ruff check + format-check + fast tests, py 3.10, setup-uv) | fast green CI, hermetic, no GPU dep, low maintenance | no type gate, single Python | **Low–Med** |

## 5. Scope recommendation

**MVP — IN (C5):**
1. Add `ruff` to `[dependency-groups] dev`; add `[tool.ruff]` config in `pyproject.toml` (`line-length = 120`; target `py310`; select a pragmatic rule set excluding noisy E501/W29x or letting format absorb them).
2. One-time **safe fixes for REAL violations only**: W605 (raw-string docstrings in `stoodx.py:156-162`), F541, F841, E711. F401 (24) → author review, autofix-if-confirmed-unused or `# noqa: F401` with rationale.
3. One-time `ruff format` across `src/` + `tests/` (13 files; semantically safe; gets format-check to green).
4. `.github/workflows/ci.yml` with three jobs: `lint`, `format-check`, `test` (fast subset only), on Python 3.10 via `astral-sh/setup-uv@v3` + `enable-cache: true`.

**DEFER (out of C5):**
- mypy strict (future C6+, or advisory-only if option B chosen).
- Multi-Python matrix (3.11/3.12) until deps verified.
- Full `eval_ood` tests in push CI (GPU + gitignored data/checkpoints) — keep `workflow_dispatch`-only or documented-skip.
- Coverage gate (`pytest-cov` not declared).
- E501 at 88 / separate W29x enforcement (relaxed via `line-length=120` + format).

## 6. Gotchas

- **`*.txt` globally gitignored** (known) — affects any text fixtures.
- **`__pycache__/` at repo root** (untracked) — ruff ignores it; ensure CI doesn't pick it up.
- **`conftest.py` np.sctypes shim** — ruff reports it CLEAN; must not be touched by format/lint (runtime monkeypatch). Preserve verbatim.
- **F401 in tests** — openood network imports may be intentional; investigate before removing. **Do not blind-autofix.**
- **ruff not in dev deps** despite being runnable — MUST add explicitly for lockfile reproducibility.
- **`ruff format` rewrites 13/14 files** — broad but safe diff; the W605 raw-string fix and the format pass touch overlapping docstrings — coordinate ordering (fix W605, then format) to avoid rework.
- **Test job sync is heavy** (torch+openood+libmr build) — relies on `extra-build-dependencies` (numpy+cython for libmr) working under uv in CI exactly as locally.
- **No GPU in CI** → any test that constructs a net/loader must stay out of the fast subset; the two chosen tests are explicitly marked GPU/data-free (verified).
- C5 is the **last PR in the chain** → all formatting churn is contained here; downstream is `main`.

## Ready for Proposal
**Yes.** The orchestrator should tell the user: scope is a pragmatic ruff + minimal GitHub Actions CI (lint + format-check + fast hermetic tests on Python 3.10 via setup-uv); mypy deferred; full eval tests stay manual. Key open decisions for the user: (1) one-time `ruff format` of 13 files — accept the churn? (2) F401 test imports — autofix vs `# noqa`? (3) mypy: fully defer, or advisory-only in CI?
