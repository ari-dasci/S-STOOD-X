# openood-extension Specification

> **Capability**: `openood-extension` — **structural conformance**. Captures that the privatized
> STOODX adapter (`STOODXPostprocessor`) is a VALID OpenOOD extension: it subclasses the right
> base, exposes the right contract methods, and is accepted by OpenOOD's `Evaluator` along the
> isinstance-gated pre-built-instance path. NEW domain — FULL spec, not a delta. Asserts
> STRUCTURE only; numerical `eval_ood` behavior is out of scope (R4).

## Purpose

Prove **model Z** — the deferred OpenOOD extension is wired correctly without needing pretrained
weights, datasets, or GPU. The contract test exercises the adapter the way OpenOOD's
`Evaluator` accepts a pre-built postprocessor instance (isinstance gate). This is the minimum
verifiable claim that the privatized adapter is a real OpenOOD extension.

## Requirements

### Requirement: Adapter contract conformance

`STOODXPostprocessor` MUST subclass `openood.postprocessors.BasePostprocessor` and MUST implement
`__init__`, `setup`, `postprocess`, and `inference`. The class MUST be constructible from a
config dict and MUST expose the config attributes its `__init__` stores.

#### Scenario: Contract test asserts subclass + callable contract

- GIVEN `src/STOODX/_openood_adapter/postprocessor.py` defines `STOODXPostprocessor`
- WHEN `tests/_openood_adapter/test_postprocessor_contract.py` constructs an instance
- THEN `isinstance(instance, openood.postprocessors.BasePostprocessor)` is `True`
- AND `__init__`, `setup`, `postprocess`, `inference` are all present and callable
- AND the test runs WITHOUT pretrained weights, datasets, or GPU

#### Scenario: Config attributes are present at construction

- GIVEN an instance built from a minimal config dict
- THEN the instance exposes the config attributes the adapter's `__init__` stores
- AND no attribute access raises `AttributeError` at construction time

### Requirement: OpenOOD Evaluator acceptance

The `STOODXPostprocessor` instance MUST be accepted by OpenOOD's `Evaluator` along the
isinstance-gated pre-built-instance path. The adapter MUST import cleanly from
`STOODX._openood_adapter.postprocessor`.

#### Scenario: isinstance gate holds for the pre-built instance

- GIVEN OpenOOD's `Evaluator` accepts a pre-built postprocessor guarded by `isinstance(BasePostprocessor)`
- WHEN a `STOODXPostprocessor` instance is supplied as that pre-built postprocessor
- THEN the isinstance check passes and the instance is accepted (not rejected as wrong type)

#### Scenario: Import path resolves in the uv env

- GIVEN the uv env is synced
- WHEN `from STOODX._openood_adapter.postprocessor import STOODXPostprocessor` runs
- THEN the import resolves without `ImportError` or `ModuleNotFoundError`

### Requirement: numpy.sctypes test-infrastructure shim

A root `conftest.py` MUST re-add `numpy.sctypes` (removed in NumPy 2.0) when it is absent,
BEFORE any openood import, restoring `imgaug` compatibility. The shim MUST be reversible and
MUST cite the `imgaug` version and upstream issue. This RESOLVES env R3 (previously DEFERRED).

#### Scenario: pytest collects all 7 modules with zero import errors

- GIVEN `uv sync` has completed AND the root `conftest.py` shim is present
- WHEN `uv run pytest --collect-only` is run
- THEN ALL 7 test modules are collected with ZERO collection/import errors
- AND the 5 previously-failing modules (`test_featureVisualization`, `test_postProcessor_cifar10`,
  `test_postProcessor_cifar100`, `test_postProcessor_imagenet`, `test_postProcessor_imagenet200`) collect
- AND the outcome reflects collection only, not test pass/fail

#### Scenario: Shim is a no-op when sctypes is already present

- GIVEN a numpy build where `numpy.sctypes` already exists
- WHEN the shim executes
- THEN it MUST NOT overwrite the existing attribute
- AND pytest collection behavior is unchanged

### Requirement: Scope boundary — structural conformance only

This capability MUST NOT assert numerical AUROC, p-values, or `eval_ood` outputs. Full
`eval_ood` runs (requiring pretrained checkpoints + datasets + GPU) are OUT of the automated
gate and MUST be documented as a manual/prereq step in `README.md`.

#### Scenario: Contract test is structural, not numerical

- GIVEN the contract test in `tests/_openood_adapter/`
- THEN it asserts isinstance + method presence/callability + config attrs ONLY
- AND it makes NO assertions on AUROC, scores, or `eval_ood` outputs

#### Scenario: Heavy eval_ood is documented as manual

- GIVEN README's testing section
- THEN `eval_ood` end-to-end runs are documented as requiring pretrained + datasets + GPU
- AND they are NOT in the automated gate that runs on `--collect-only` or the contract test
