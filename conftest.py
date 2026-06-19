"""Pytest configuration: NumPy 2.0 compatibility shim for imgaug.

imgaug (transitive via OpenOOD's draem_preprocessor) uses ``numpy.sctypes``,
removed in NumPy 2.0. imgaug is unmaintained (last release 2021, see
imgaug issue #595). This shim re-adds ``np.sctypes`` so collection succeeds
under numpy>=2.1.2. Long-term fix: the OpenOOD fork (out of scope here).
"""
import numpy as np

if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": {np.int8, np.int16, np.int32, np.int64},
        "uint": {np.uint8, np.uint16, np.uint32, np.uint64},
        "float": {np.float16, np.float32, np.float64},
        "complex": {np.complex64, np.complex128},
        "others": {bool, object, bytes, str},
    }
