"""Public API smoke test for the STOODX package.

Verifies the renamed detector class imports cleanly and resolves to its defining
module, without requiring GPU, datasets, or pretrained checkpoints. Closes the
coverage gap surfaced during explore (§3c) for the `STOODX` -> `STOODXDetector`
rename (C4).
"""

import STOODX


def test_renamed_detector_imports_from_package():
    """`from STOODX import STOODXDetector` must succeed post-rename."""
    from STOODX import STOODXDetector

    assert STOODXDetector is not None


def test_detector_module_path():
    """The class must resolve to the module file `STOODX.stoodx`."""
    from STOODX import STOODXDetector

    assert STOODXDetector.__module__ == "STOODX.stoodx"


def test_all_exports_renamed_detector_not_bare_class():
    """`__all__` must list `STOODXDetector` and must NOT list the bare `STOODX`."""
    assert "STOODXDetector" in STOODX.__all__
    assert "STOODX" not in STOODX.__all__


def test_all_exports_three_symbol_surface():
    """Public surface stays three symbols; STOODXPostprocessor remains absent."""
    assert STOODX.__all__ == ["STOODXDetector", "FeatureExtractor", "FeatureExplanation"]
    assert "STOODXPostprocessor" not in STOODX.__all__
