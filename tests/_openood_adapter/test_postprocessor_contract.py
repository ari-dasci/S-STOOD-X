"""Structural contract test for the internal OpenOOD adapter postprocessor.

Verifies that ``STOODXPostprocessor`` is a valid OpenOOD extension by checking
the contract OpenOOD's ``Evaluator`` relies on when accepting a pre-built
instance: subclassing ``BasePostprocessor`` and exposing the four contract
methods (``__init__``/``setup``/``postprocess``/``inference``) plus the
expected config attributes.

This is a STRUCTURAL test only. No data, no net, no loaders, no GPU work.
``eval_ood`` (full AUROC runs) is a manual prerequisite (pretrained models +
datasets + GPU) and is intentionally out of the automated gate.
"""

from openood.postprocessors import BasePostprocessor

from STOODX._openood_adapter.postprocessor import STOODXPostprocessor

# Minimal config dict: required keys only, including the upstream typos
# (`quantil`, `atribut`) that the adapter reads verbatim. `BasePostprocessor`
# just stores `config = config`, so a plain dict satisfies both the base
# class and the adapter's `self.config[k]` / `.get()` access pattern.
MINIMAL_CONFIG = {
    "K": 5,
    "distance": "cosine",
    "feature_name": "layer4",
    "intraclass": False,
    "quantil": 0.75,
    "atribut": False,
}


def test_postprocessor_is_basepostprocessor():
    """STOODXPostprocessor must subclass BasePostprocessor.

    OpenOOD's Evaluator gates pre-built instances on this isinstance check.
    """
    adapter = STOODXPostprocessor(MINIMAL_CONFIG)
    assert isinstance(adapter, BasePostprocessor)


def test_postprocessor_contract_methods_callable():
    """The four contract methods (__init__/setup/postprocess/inference)
    must exist and be callable. They are NOT invoked with data here —
    setup/postprocess/inference need a real net + loaders (manual prereq)."""
    adapter = STOODXPostprocessor(MINIMAL_CONFIG)
    for method_name in ("__init__", "setup", "postprocess", "inference"):
        method = getattr(adapter, method_name, None)
        assert method is not None, f"missing contract method: {method_name}"
        assert callable(method), f"contract method not callable: {method_name}"


def test_postprocessor_config_attrs_set():
    """Construction must populate the config-driven attributes and leave
    oodTest as None until setup() is run against a real net/loaders."""
    adapter = STOODXPostprocessor(MINIMAL_CONFIG)
    for attr in ("K", "distance", "feature_name", "device", "oodTest"):
        assert hasattr(adapter, attr), f"missing config attribute: {attr}"
    assert adapter.K == 5
    assert adapter.distance == "cosine"
    assert adapter.feature_name == "layer4"
    assert adapter.oodTest is None
