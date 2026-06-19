# STOOD-X

[![arXiv](https://img.shields.io/badge/arXiv-2504.02685-b31b1b.svg)](https://arxiv.org/abs/2504.02685)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **STOOD-X Methodology: Using Statistical Nonparametric Test for OOD Detection in Large-Scale Datasets Enhanced with Explainability**

Official implementation of the STOOD-X (Statistical Test for Out-Of-Distribution with Explainability) methodology for detecting out-of-distribution samples in deep learning models with built-in explainability features.

## Overview

STOOD-X is a **two-stage post-hoc OOD detection methodology** that combines:

1. **Statistical OOD Detection**: Uses the Wilcoxon-Mann-Whitney nonparametric test on feature-space distances to identify OOD samples without restrictive distributional assumptions.

2. **Explainability Enhancement**: Generates concept-based visual explanations aligned with the BLUE XAI paradigm (responsi**B**le, **L**ega**l**, tr**U**st, **E**thics) to provide human-interpretable insights.

### Key Features

- **No parametric assumptions** - Uses nonparametric statistical testing instead of assuming Gaussian distributions
- **Scalable** - Efficient for large-scale datasets and high-dimensional features
- **Explainable** - Provides visual explanations showing nearest neighbors and feature importance
- **Architecture-agnostic** - Works with CNNs (ResNet) and Transformers (ViT)
- **Competitive performance** - State-of-the-art results on CIFAR-10, CIFAR-100, ImageNet-200, and ImageNet-1K

## Paper

This repository implements the methodology described in:

> **STOOD-X: Explainable out-of-distribution detection via nonparametric statistical testing on large-scale datasets**  
> Iván Sevillano-García, Julián Luengo, Francisco Herrera  
> *University of Granada, Spain*  
> Pattern Recognition, Vol. 177, 2026, Article 113254

**Links:**
- [📄 Pattern Recognition (Official Publication)](https://doi.org/10.1016/j.patcog.2026.113254)
- [📄 arXiv Abstract](https://arxiv.org/abs/2504.02685)
- [📄 arXiv HTML Version](https://arxiv.org/html/2504.02685v1)

## Installation

STOOD-X uses [uv](https://docs.astral.sh/uv/) for reproducible environment management.
All dependencies are pinned through the committed `uv.lock`, so any clone of the
repository resolves to the exact same environment.

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (any recent version)
- Python 3.10 (uv will fetch it automatically based on `.python-version`)
- A C/C++ compiler toolchain, required to build the `libmr` C extension
  (`build-essential` on Debian/Ubuntu, `gcc`/`gcc-c++` on Fedora/RHEL,
  "Desktop development with C++" on Windows).

### Install from source

```bash
git clone https://github.com/ari-dasci/S-STOOD-X.git
cd S-STOOD-X
uv sync
```

`uv sync` installs the editable package plus the `dev` dependency group (pytest,
sphinx, twine, tensorboard, …) into a local `.venv`, using the pinned `uv.lock`.

### Run the tests

```bash
uv run pytest
```

### PyTorch: CPU and GPU

The committed `uv.lock` resolves `torch` from the default PyPI index. On Linux
this is the **CUDA-compatible wheel**: it bundles the CUDA runtime libraries
(pulled in as `nvidia-cuda-*` transitive dependencies) and runs on **both** CPU
and GPU. On a CPU-only machine PyTorch automatically falls back to CPU; on a
machine with a CUDA device it uses the GPU. No extra configuration is needed for
inference/training on GPU servers — `uv sync` already gives you a GPU-capable
environment.

The trade-off of the default lock is disk size: CPU-only machines still
download the unused CUDA libraries. If you want a **true CPU-only** (small,
CUDA-free) environment, point uv at the PyTorch CPU index:

```bash
UV_PYTHON_INDEX_URL=https://download.pytorch.org/whl/cpu uv sync
```

Or, to pin it persistently in a local (uncommitted) `pyproject.toml`:

```toml
# === LOCAL ONLY — uncomment for a CPU-only env. Do not commit. ===
# [tool.uv.index]
# pytorch-cpu = "https://download.pytorch.org/whl/cpu"

# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchvision = { index = "pytorch-cpu" }
```

### Known limitations

- **`numpy.sctypes` shim for imgaug.** `imgaug` (pulled in transitively by the
  pinned `openood` dependency through its `draem_preprocessor`) uses
  `numpy.sctypes`, which was removed in NumPy 2.0. STOOD-X pins
  `numpy>=2.1.2`, and `imgaug` is unmaintained (last release in 2021,
  [imgaug issue #595](https://github.com/aleju/imgaug/issues/595)), so no
  NumPy-2-compatible release exists. To keep the full test suite collecting
  cleanly under NumPy 2, the root `conftest.py` re-adds `np.sctypes` with a
  `hasattr` guard (no-op on NumPy <2). This is test infrastructure only; the
  long-term fix is an OpenOOD fork (out of scope here). No numpy downgrade and
  no OpenOOD pin change were applied.

- **Undeclared OpenOOD runtime dependencies.** The pinned `openood` (rev
  `3c35632`) does not declare several third-party packages it imports
  unconditionally at top level (upstream packaging bug). This caused a chain
  of masked `ModuleNotFoundError` failures during `pytest --collect-only`.
  STOOD-X declares the affected packages directly so the full test suite
  collects cleanly:

  - **`timm`** — imported by `openood.attacks.misc`; not in OpenOOD metadata.
    Added as a direct STOOD-X dependency.
  - **`foolbox`** — imported eagerly by
    `openood.evaluation_api.attackdataset` (reached via `Evaluator`); not in
    OpenOOD metadata. Added as a direct STOOD-X dependency.
  - **`statsmodels`**, **`libmr`** — imported by OpenOOD postprocessors
    (`odin`, `openmax`, `adaptive scaling`) but undeclared in its metadata;
    both are already declared by STOOD-X directly for its own statistics, so
    they are satisfied transitively.

  An exhaustive audit of every top-level import across the OpenOOD package
  confirmed two further undeclared-but-harmless imports that are **not** added
  as dependencies because they cannot break collection:
  - `clip` (OpenAI CLIP) — imported by `openood.networks.clip`, but the import
    in `openood.networks.__init__` is wrapped in
    `try/except ModuleNotFoundError: pass`, so it is silently skipped when
    absent.
  - `mmcls` / `mmcv` (OpenMMLab) — imported only by `openood.networks.net_utils_`,
    a legacy module referenced solely by a commented-out line in
    `openood.networks.__init__` and therefore never on the import path.

  The correct long-term fix is for OpenOOD to declare these in its own
  metadata; until then STOOD-X carries `timm` and `foolbox` as direct deps.
  OpenOOD pin was not changed.

- **Full `eval_ood` runs are a manual prerequisite.** Reproducing the paper's
  AUROC numbers through OpenOOD's `eval_ood` requires pretrained models, the
  benchmark datasets, and (for non-trivial scale) a CUDA GPU. Those are not
  provisioned by `uv sync`, so `eval_ood` is intentionally **out of the
  automated test gate**. The automated test suite verifies the structural
  OpenOOD-extension contract (`STOODXPostprocessor` is a `BasePostprocessor`
  exposing the four contract methods and config attributes, with no data/GPU)
  plus the standalone-core invariant (`import STOODX` pulls no `openood`).

- **STOOD-X core is standalone.** `import STOODX` (the public 3-symbol API
  `STOODXDetector`, `FeatureExtractor`, `FeatureExplanation`) does not import
  `openood`; the OpenOOD adapter lives under the private
  `_openood_adapter` package and is only loaded on demand. This is asserted by
  the test suite (`'openood' not in sys.modules`).

## Quick Start

### Basic Usage

```python
import torch
from STOODX import STOODXDetector, FeatureExtractor

# Initialize your model and feature extractor
model = FeatureExtractor(
    model=your_torch_model,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    feature_name="layer4",
    atribut=True
)

# Create STOOD-X detector
detector = STOODXDetector(
    model=model,
    k_neighbors=500,    # Number of neighbors for comparison
    k_NNs=50,           # Number of nearest neighbors
    quantile=0.99,      # Feature selection quantile
    whole_test=True
)

# Add validation features (in-distribution data)
for batch in validation_loader:
    detector.addFeatures(batch)
detector.finalizeFeatures()

# Test new samples
def is_ood(sample_tensor, threshold=0.05):
    """
    Returns True if sample is OOD, False if in-distribution.
    Uses p-value from Wilcoxon test - lower values indicate OOD.
    """
    results = detector.test(sample_tensor, intraclass=True)
    mean_p_value = results["p_value"].mean()
    return mean_p_value < threshold

# Test a sample
result = detector.test(test_sample)
print(f"Mean p-value: {result['p_value'].mean():.4f}")
```

## Project Structure

```
S-STOOD-X/
├── src/
│   └── STOODX/                   # Core library
│       ├── __init__.py           # Public API: STOODXDetector, FeatureExtractor, FeatureExplanation
│       ├── stoodx.py             # Main STOOD-X implementation
│       ├── feature_extractor.py  # Feature extraction (FeatureExtractor)
│       ├── feature_visualization.py  # Visualization and explainability
│       └── _openood_adapter/     # Internal OpenOOD adapter (private)
│           └── postprocessor.py  # STOODXPostprocessor for OpenOOD integration
├── configs/                     # Configuration files
│   └── postprocessors/         # Post-processor configs
├── data/                        # Data and benchmark lists
├── tests/                       # Unit tests
│   ├── test_feature_extractor.py
│   ├── test_feature_visualization.py
│   └── _openood_adapter/        # OpenOOD-adapter integration tests
├── results/                     # Experimental results
├── pretrained_models/           # Pretrained model storage
├── README.md                    # This file
├── pyproject.toml              # Package configuration
└── LICENSE                      # GPL v3 License
```

## Development

STOOD-X uses [`uv`](https://docs.astral.sh/uv/) for environment and dependency management and
[`ruff`](https://docs.astral.sh/ruff/) for linting and formatting. All commands use the `uv run`
prefix so they execute against the project's locked environment.

### Linting and formatting

```bash
# Check lint (E, F, W, I rule families; line-length 120, target py310)
uv run ruff check src/ tests/

# Auto-apply safe fixes
uv run ruff check src/ tests/ --fix

# Check formatting
uv run ruff format --check src/ tests/

# Apply formatting
uv run ruff format src/ tests/
```

Configuration lives under `[tool.ruff]` in `pyproject.toml`.

### Tests

```bash
# Hermetic (GPU/dataset-free) tests — the ones CI runs:
uv run pytest tests/test_public_api.py tests/_openood_adapter/test_postprocessor_contract.py -v

# Full test suite (includes eval_ood tests that need GPU + gitignored checkpoints/datasets)
uv run pytest
```

### Continuous Integration

The [`.github/workflows/ci.yml`](.github/workflows/ci.yml) workflow runs on every push and pull
request to `main` with three jobs on Python 3.10 (`astral-sh/setup-uv@v3`, cache enabled):

- **lint** — `uv run ruff check src/ tests/`
- **format-check** — `uv run ruff format --check src/ tests/`
- **test** — `uv run pytest` on the two hermetic test modules above.

The full `eval_ood` pipeline is **not** run in CI because it requires GPU hardware and
gitignored pretrained checkpoints and datasets.

## Performance

STOOD-X achieves competitive performance on standard OOD detection benchmarks:

| Dataset | Architecture | Near-OOD AUROC | Far-OOD AUROC |
|---------|-------------|----------------|---------------|
| CIFAR-10 | ResNet18 | 89.53% | 92.01% |
| CIFAR-100 | ResNet18 | ~85% | ~90% |
| ImageNet-200 | ViT-B/16 | ~78% | ~88% |
| ImageNet-1K | ViT-B/16 | 81.95% | 92.20% |

*Note: See paper for complete experimental results.*

## Key Parameters

- **k_neighbors** (default: 500): Number of nearest neighbors from validation set to compare against
- **k_NNs** (default: 50): Number of nearest neighbors to use for Wilcoxon test
- **quantile** (default: 0.99): Quantile for feature selection (0.99 keeps top 1% of features)
- **intraclass** (default: True): Whether to compare only against same-class samples
- **whole_test** (default: True): Use full test vs paired comparison

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{sevillano2026stoodx,
  title={STOOD-X: Explainable out-of-distribution detection via nonparametric statistical testing on large-scale datasets},
  author={Sevillano-Garc{\'i}a, Iv{\'a}n and Luengo, Juli{\'a}n and Herrera, Francisco},
  journal={Pattern Recognition},
  volume={177},
  pages={113254},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.patcog.2026.113254}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Authors

- **Iván Sevillano-García** - [isevillano@ugr.es](mailto:isevillano@ugr.es)
- **Julián Luengo** 
- **Francisco Herrera**

Research Group: [Andalusian Research Institute in Data Science and Computational Intelligence (DaSCI)](https://dasci.es)  
University of Granada, Spain

## Related Work

- [OpenOOD](https://github.com/Jingkang50/OpenOOD) - Benchmark for OOD detection
- [Zennit](https://github.com/chr5tphr/zennit) - Attribution framework used for explainability
- [ReVel](https://github.com/pmichel31415/revel) - Relevance visualization library

## Contact

For questions or issues, please:
- Open an issue on GitHub: [https://github.com/ari-dasci/S-STOOD-X/issues](https://github.com/ari-dasci/S-STOOD-X/issues)
- Contact the authors: [isevillano@ugr.es](mailto:isevillano@ugr.es)

---

**Note**: This is a research implementation. For production use, additional error handling and optimizations may be required.
