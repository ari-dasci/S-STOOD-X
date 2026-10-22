"""Shapes and batch handling of the detector.

Hermetic: no downloads, no dataset, and a ResNet-18 built with `weights=None`.
"""

import pytest
import torch
import torchvision

from STOODX import FeatureExtractor, STOODXDetector

IMG_SIZE = 64
N_FEATURES = 512  # resnet18 avgpool


def build_detector() -> STOODXDetector:
    model = torchvision.models.resnet18(weights=None).eval()
    extractor = FeatureExtractor(
        model=model,
        device=torch.device("cpu"),
        feature_name="avgpool",
        atribut=False,
    )
    return STOODXDetector(model=extractor, k_neighbors=2, k_NNs=2, quantile=0.9)


def test_add_features_keeps_one_row_per_sample():
    """A batch must produce one feature row per sample, not one flat vector."""
    detector = build_detector()
    detector.addFeatures(torch.rand(3, 3, IMG_SIZE, IMG_SIZE))
    detector.finalizeFeatures()

    assert detector.feats.shape == (3, N_FEATURES)
    assert detector.classes.shape == (3,)


def test_add_features_accumulates_batches():
    """Batches and single samples must stack in the same order."""
    detector = build_detector()
    detector.addFeatures(torch.rand(1, 3, IMG_SIZE, IMG_SIZE))
    detector.addFeatures(torch.rand(4, 3, IMG_SIZE, IMG_SIZE))
    detector.finalizeFeatures()

    assert detector.feats.shape == (5, N_FEATURES)
    assert detector.classes.shape == (5,)


@pytest.mark.parametrize("shape", [(3, IMG_SIZE, IMG_SIZE), (1, 3, IMG_SIZE, IMG_SIZE)])
def test_test_accepts_one_sample_either_shape(shape):
    """test() accepts (C, H, W) and (1, C, H, W); both are one sample."""
    detector = build_detector()
    detector.addFeatures(torch.rand(5, 3, IMG_SIZE, IMG_SIZE))
    detector.finalizeFeatures()

    result = detector.test(torch.rand(*shape), intraclass=False)

    assert "p_value" in result.columns
    assert len(result) > 0


def test_test_rejects_a_real_batch():
    """A batch larger than one is rejected with a clear message, not a shape error."""
    detector = build_detector()
    detector.addFeatures(torch.rand(5, 3, IMG_SIZE, IMG_SIZE))
    detector.finalizeFeatures()

    with pytest.raises(ValueError, match="one sample at a time"):
        detector.test(torch.rand(2, 3, IMG_SIZE, IMG_SIZE), intraclass=False)
