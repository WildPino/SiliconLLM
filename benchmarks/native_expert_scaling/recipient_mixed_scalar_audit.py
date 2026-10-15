#!/usr/bin/env python3
"""Run the frozen mixed-recipient auditor with scalar F32 exponential values.

Only the auditor module's NumPy `exp` binding changes. Every input, numerical
gate, output schema, and failure receipt remains owned by the original auditor.
"""

import importlib.util
import math
from pathlib import Path


ORIGINAL = Path(__file__).with_name("recipient_mixed_audit.py")
spec = importlib.util.spec_from_file_location("recipient_mixed_audit_frozen", ORIGINAL)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load frozen auditor: {ORIGINAL}")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
_numpy = A.np


class ScalarExpNumPy:
    def __getattr__(self, name):
        return getattr(_numpy, name)

    @staticmethod
    def exp(values):
        array = _numpy.asarray(values)
        return _numpy.fromiter(
            (math.exp(float(value)) for value in array.flat),
            dtype=_numpy.float32,
            count=array.size,
        ).reshape(array.shape)


A.np = ScalarExpNumPy()

if __name__ == "__main__":
    A.main()
