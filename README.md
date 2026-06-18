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

- **`pytest` collection currently fails on 5 modules.** Running
  `uv run pytest --collect-only` raises collection errors for
  `tests/test_featureVisualization.py`,
  `tests/test_postProcessor_cifar10.py`,
  `tests/test_postProcessor_cifar100.py`,
  `tests/test_postProcessor_imagenet.py`, and
  `tests/test_postProcessor_imagenet200.py`.
  Root cause: `imgaug` (pulled in transitively by the pinned `openood`
  dependency through its `draem_preprocessor`) uses `numpy.sctypes`, which was
  removed in NumPy 2.0. STOOD-X pins `numpy>=2.1.2`, and `imgaug` is
  unmaintained (last release in 2021), so no NumPy-2-compatible release of
  `imgaug` exists. This is a latent conflict that predates the uv migration and
  is now made explicit by the locked environment. Resolution is tracked for a
  follow-up change. Environment-level imports (`uv run python -c "import STOODX"`)
  and the rest of the collection succeed.

## Quick Start

### Basic Usage

```python
import torch
from STOODX import STOODX, FeatureStractor

# Initialize your model and feature extractor
model = FeatureStractor(
    model=your_torch_model,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    feature_name="layer4",
    atribut=True
)

# Create STOOD-X detector
detector = STOODX(
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
├── STOODX/                      # Core library
│   ├── STOODX.py               # Main STOOD-X implementation
│   ├── STOODXPostprocessor.py  # Post-processor for OpenOOD integration
│   ├── featureStractor.py      # Feature extraction utilities
│   └── featureVisualization.py # Visualization and explainability
├── configs/                     # Configuration files
│   └── postprocessors/         # Post-processor configs
├── data/                        # Data and benchmark lists
├── tests/                       # Unit tests
├── results/                     # Experimental results
├── pretrained_models/           # Pretrained model storage
├── README.md                    # This file
├── pyproject.toml              # Package configuration
└── LICENSE                      # GPL v3 License
```

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
