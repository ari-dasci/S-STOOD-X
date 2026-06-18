"""Internal OpenOOD adapter — NOT public API.

This subpackage bridges STOOD-X to the OpenOOD evaluation framework. Importing it
pulls `openood` and its (numpy-2-incompatible) transitive deps. Do not import from
outside the package; treat as private.
"""
