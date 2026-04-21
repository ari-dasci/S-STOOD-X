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

> **STOOD-X Methodology: Using Statistical Nonparametric Test for OOD Detection in Large-Scale Datasets Enhanced with Explainability**  
> Iván Sevillano-García, Julián Luengo, Francisco Herrera  
> *University of Granada, Spain*  
> arXiv:2504.02685, April 2025

**Links:**
- [📄 arXiv Abstract](https://arxiv.org/abs/2504.02685)
- [📄 arXiv HTML Version](https://arxiv.org/html/2504.02685v1)

## Installation

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.4.1
- CUDA (optional, for GPU acceleration)

### Install from source

```bash
git clone https://github.com/ari-dasci/S-STOOD-X.git
cd S-STOOD-X
pip install -e .
```

### Install dependencies manually

```bash
pip install torch>=2.4.1 torchvision>=0.19.1
pip install numpy>=2.1.2 pandas>=2.2.3 scipy>=1.14.1
pip install scikit-learn>=1.5.2 scikit-image>=0.24.0
pip install matplotlib>=3.9.2 seaborn>=0.13.2 plotly>=5.24.1
pip install zennit-crp[fast_img]>=0.6.0 revel-xai>=1.0.3
pip install baycomp>=1.0.3 opencv-python>=4.10.0.84
pip install tqdm>=4.66.5 tensorboard>=2.18.0
```

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
@article{sevillano2025stoodx,
  title={STOOD-X Methodology: Using Statistical Nonparametric Test for OOD Detection in Large-Scale Datasets Enhanced with Explainability},
  author={Sevillano-Garc{\'i}a, Iv{\'a}n and Luengo, Juli{\'a}n and Herrera, Francisco},
  journal={arXiv preprint arXiv:2504.02685},
  year={2025}
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
