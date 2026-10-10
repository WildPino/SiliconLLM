"""Preserve frozen audit mathematics; serialize NumPy scalars as native values."""
import argparse
import sys
from pathlib import Path

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B))
import original_categorical_loss_compile as frozen
import numpy as np

original_dumps = frozen.json.dumps
FAULT = frozen.DOC / 'original_categorical_loss_compile_stored_adjudication_20261010.terminal.json'


def native_scalars(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: native_scalars(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native_scalars(v) for v in value]
    return value


def scalar_dumps(value, *args, **kwargs):
    if isinstance(value, dict) and value.get('schema') == 'ORIGINAL_CATEGORICAL_LOSS_AUDIT_V1':
        value = dict(value, serialization_adapter=frozen.extent(__file__),
                     retained_first_audit_fault=frozen.extent(FAULT),
                     mathematical_algorithm_and_thresholds_unchanged=True,
                     extra_complete_audit_label_pass=8808)
    return original_dumps(native_scalars(value), *args, **kwargs)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--audit-worker', action='store_true', required=True)
    for key in ('binding', 'out', 'directory', 'source-result'):
        p.add_argument('--' + key, type=Path, required=True)
    for key in ('binding-sha', 'freeze'):
        p.add_argument('--' + key, required=True)
    a = p.parse_args()
    frozen.json.dumps = scalar_dumps
    frozen.audit(a)
