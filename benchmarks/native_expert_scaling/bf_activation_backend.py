"""Load the qualified row backend with lifetime-safe shared activation tables."""
import ctypes
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'benchmarks/native_expert_scaling'),
             str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent
from row_state import interface

# C uses process-global table pointers. Retain owners for the process lifetime.
# Rebinding a DLL to identically qualified tables leaves every existing context safe.
_LIVE_TABLES={}


def load_qualified_backend(closure_path):
    """Return the established row_state ABI tuple; parameter owners belong to caller.

    Reads deployment DLL/tables and qualification metadata, never teacher traces.
    Source FFN/attention/head and same-artifact CPU model quality remain separate.
    """
    import numpy as np
    c=json.loads(Path(closure_path).read_bytes())
    assert c['decision']=='EXACT_STORED_ACCELERATION_ELIGIBLE'
    assert c['all_declared_hashes_and_numeric_qualification_pass']
    for key in ('report','audit','binding','producer_activation','audit_activation'):
        assert extent(c[key]['path'])==c[key]
    b=json.loads(Path(c['binding']['path']).read_bytes())
    assert b['adapter']['meaning']=='finite-BF activation lookup, ORIGINAL readout'
    assert extent(b['candidate_compile']['dll']['path'])==b['candidate_compile']['dll']
    owned=[]
    for item in b['activation_tables']:
        assert extent(item['file']['path'])==item['file']
        owned.append(np.memmap(item['file']['path'],dtype=item['dtype'],mode='r',shape=(65536,)))
    assert [item['role'] for item in b['activation_tables']]==['conv','gate','delta']
    assert sum(a.nbytes for a in owned)==786432
    api=interface(dict(compile=b['candidate_compile']))
    bind=api[0].act_bind;bind.argtypes=[ctypes.c_void_p]*3+[ctypes.c_uint];bind.restype=ctypes.c_int
    rc=bind(*(a.ctypes.data for a in owned),b['FP_control'])
    if rc!=0:raise RuntimeError('Activation table binding rejected: require qualified FE_TONEAREST/MXCSR control and masked exceptions.')
    _LIVE_TABLES[str(Path(b['candidate_compile']['dll']['path']).resolve())]=tuple(owned)
    return api
