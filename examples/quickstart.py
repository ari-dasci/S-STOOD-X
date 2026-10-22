"""Runnable end-to-end example of STOOD-X.

No dataset and no download: a torchvision ResNet-18 with randomly initialised weights
stands in for the model (`weights=None`), and the validation set is random noise. It adds
validation features, finalizes them and runs the nonparametric test on one sample.

The p-values of an untrained model mean nothing; what the script shows is the call sequence
and the shape of what comes back.

Run from the repository root:

    python examples/quickstart.py
"""

import torch
import torchvision

from STOODX import FeatureExtractor, STOODXDetector

torch.manual_seed(0)

IMG_SIZE = 64
N_VALIDATION = 32
device = torch.device("cpu")

# `feature_name` must name a module of the model, and its output must already be globally
# pooled: the detector squeezes the batch dimension and expects one vector per sample. For a
# ResNet that is `avgpool` (layer4 would leave a (C, H, W) map and break the distance).
model = torchvision.models.resnet18(weights=None).eval()

extractor = FeatureExtractor(
    model=model,
    device=device,
    feature_name="avgpool",
    atribut=False,
)

detector = STOODXDetector(
    model=extractor,
    k_neighbors=10,
    k_NNs=5,
    quantile=0.99,
)

# 1. Feed the in-distribution (validation) set, then close it. One sample at a time: the
# attribution step resolves a single predicted class, so addFeatures handles batch size 1.
for _ in range(N_VALIDATION):
    detector.addFeatures(torch.rand(1, 3, IMG_SIZE, IMG_SIZE))
detector.finalizeFeatures()

# 2. Test a single sample. Note the shape change: addFeatures takes a batch of one,
# (1, C, H, W), while test() unsqueezes internally and therefore expects (C, H, W).
# intraclass=False compares against the whole validation set, so the example does not depend
# on which class the untrained model happens to predict.
sample = torch.rand(3, IMG_SIZE, IMG_SIZE)
result = detector.test(sample, intraclass=False)

print("Per-neighbour test result:")
print(result.round(4).to_string(index=False))
print("\nMean p-value:", round(float(result["p_value"].mean()), 4))
print(
    "\nNaN p-values can appear when the Bayes test cannot separate the distance\n"
    "distributions, which is expected with an untrained model on random noise.\n"
    "pandas' mean skips them."
)
