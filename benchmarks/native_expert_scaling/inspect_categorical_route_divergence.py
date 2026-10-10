"""Bounded stored-route localization AFTER full campaign custody adjudication.

Standard library only. No model, Torch, NumPy, native child or GPU call.
Preparation freezes small sealed inputs; execution requires that binding hash.
No change to the consumed campaign or its numeric gates.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

L, K, E = 6, 8, 1152
LIMITS = dict(seconds=60, OS_bytes=256 << 20, input_bytes=128 << 20,
              output_bytes=2 << 20, max_file_bytes=8 << 20)


class Guard:
    def __init__(self):
        self.start = time.monotonic()
        self.bytes = 0
        self.peak = 0
        if os.name != 'nt':
            raise RuntimeError('Actual observer requires Windows memory accounting')
        class Counters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_uint32), ('PageFaultCount', ctypes.c_uint32),
                        *[(name, ctypes.c_size_t) for name in (
                            'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                            'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]]
        self.counter = Counters()
        self.counter.cb = ctypes.sizeof(self.counter)
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.GetCurrentProcess.restype = ctypes.c_void_p
        self.psapi = ctypes.WinDLL('psapi', use_last_error=True)
        self.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p,
                                                   ctypes.POINTER(Counters), ctypes.c_uint32]
        self.psapi.GetProcessMemoryInfo.restype = ctypes.c_int

    def check(self):
        if not self.psapi.GetProcessMemoryInfo(self.kernel.GetCurrentProcess(),
                                             ctypes.byref(self.counter), self.counter.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        self.peak = max(self.peak, self.counter.PeakWorkingSetSize)
        if time.monotonic() - self.start > LIMITS['seconds'] or self.peak > LIMITS['OS_bytes']:
            raise RuntimeError('Observer resource cap exceeded')

    def read(self, path):
        path = Path(path).resolve()
        size = path.stat().st_size
        if size > LIMITS['max_file_bytes'] or self.bytes + size > LIMITS['input_bytes']:
            raise RuntimeError('Observer input cap exceeded')
        self.check()
        data = path.read_bytes()
        self.bytes += len(data)
        if len(data) != size:
            raise RuntimeError('Input extent changed during read')
        self.check()
        return data, dict(path=str(path), bytes=size, sha256=hashlib.sha256(data).hexdigest())


def core(item):
    return {key: item[key] for key in ('path', 'bytes', 'sha256')}


def write_new(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if len(data) > LIMITS['output_bytes']:
        raise RuntimeError('Observer output cap exceeded')
    with Path(path).open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def receipt(value):
    if value['exit_code'] != 0 or value['error'] is not None or not value['inputs_before_after_exact']:
        raise RuntimeError('Parent custody did not close successfully')


def prepare(a, guard):
    inputs = []
    def load(path, expected=None):
        data, item = guard.read(path)
        if expected is not None and item != core(expected):
            raise RuntimeError('Sealed input extent differs')
        inputs.append(item)
        return data, item
    _, script = load(__file__)
    raw, ai = load(a.audit)
    if ai['sha256'] != a.audit_sha:
        raise RuntimeError('Wrong adjudication hash')
    audit = json.loads(raw)
    raw, ti = load(a.audit_terminal)
    if ti['sha256'] != a.audit_terminal_sha:
        raise RuntimeError('Wrong audit-terminal hash')
    at = json.loads(raw)
    receipt(at)
    if at['result'] != ai:
        raise RuntimeError('Audit-terminal/result mismatch')
    raw, ti = load(a.capture_terminal)
    if ti['sha256'] != a.capture_terminal_sha:
        raise RuntimeError('Wrong capture-terminal hash')
    ct = json.loads(raw)
    receipt(ct)
    if audit['result'] != ct['result']:
        raise RuntimeError('Capture custody/adjudication mismatch')
    for key in ('complete_campaign', 'complete_input_output_hashes',
                'independent_integer_byte_transitions', 'independent_fsum_all_proposals',
                'full_original_packing_all_completed_trials', 'all_completed_gradients_and_bridges'):
        if audit.get(key) is not True:
            raise RuntimeError('Full stored adjudication missing: ' + key)
    raw, bi = load(audit['binding']['path'], audit['binding'])
    original = json.loads(raw)
    if ct['binding_sha256'] != bi['sha256']:
        raise RuntimeError('Original binding mismatch')
    # Direction1 reuses previously qualified traces, sealed as original INPUTS.
    sealed = {item['path']: core(item) for item in original['inputs'] + ct['outputs']}
    cases = []
    for ordinal, audited in enumerate(audit['audited'], 1):
        if not audited['completed_direction'] or audited['trials'] != 2:
            raise RuntimeError('Incomplete direction')
        candidates = [item for path, item in sealed.items()
                      if Path(path).name == 'step.json' and Path(path).parent.name == f'step_{ordinal:02d}']
        if len(candidates) != 1:
            raise RuntimeError('Step metadata is not uniquely sealed')
        raw, si = load(candidates[0]['path'], candidates[0])
        step = json.loads(raw)
        if step['ordinal'] != ordinal or step['id'] != audited['id']:
            raise RuntimeError('Step identity mismatch')
        art = step['history']['artifacts']
        ids, mass = art['forward_route_ids'], art['forward_route_mass']
        t = ids['shape'][0]
        if ids['shape'] != [t, L, K] or mass['shape'] != [t, L, K] or \
                ids['dtype'] != '<i4' or mass['dtype'] != '<f4' or not 0 < t <= 1507:
            raise RuntimeError('Unexpected route schema')
        route_inputs = dict(GPU_ids=core(ids), GPU_mass=core(mass), C_routes=core(step['baseline_native']['routes']))
        for item in route_inputs.values():
            if item != sealed.get(item['path']):
                raise RuntimeError('Route input not sealed by producer')
            load(item['path'], item)
        cases.append(dict(ordinal=ordinal, id=step['id'], history=t, routes=route_inputs,
                          bridge=step['bridge']['numeric'], bridge_flags=audited['bridge_flags']))
    if len(cases) != 24:
        raise RuntimeError('Expected full24-case campaign')
    guard.check()
    # Recheck every small frozen extent; these are not the full model inputs.
    for item in inputs:
        _, now = guard.read(item['path'])
        if now != item:
            raise RuntimeError('Small input changed before binding')
    write_new(a.out, dict(schema='CATEGORICAL_ROUTE_DIVERGENCE_BINDING_V1', inputs=inputs,
                         cases=cases, numeric=original['numeric'], limits=LIMITS,
                         preparation_seconds=time.monotonic()-guard.start,
                         preparation_OS_peak=guard.peak, script=script,
                         scope='Post-audit stored routes only; no score margins/common operands available'))


def localize(gpu_ids, gpu_mass, native, t, threshold):
    if len(gpu_ids) != t*L*K*4 or len(gpu_mass) != t*L*K*4 or len(native) != t*L*K*8:
        raise RuntimeError('Route byte count mismatch')
    out = dict(ranked_slot_disagreements=0, membership_rows=0, order_only_rows=0,
               mass_threshold_rows_same_IDs=0, common_ID_mass_threshold_rows=0, max_rank_mass_error=0.,
               max_common_ID_mass_error=0., first_ranked_ID=None, first_membership=None,
               first_order_only=None, first_mass_same_IDs=None, first_common_ID_mass=None, max_rank_mass_at=None,
               max_common_ID_mass_at=None)
    for row in range(t*L):
        gi = struct.unpack_from('<8i', gpu_ids, row*32)
        gm = struct.unpack_from('<8f', gpu_mass, row*32)
        ci = struct.unpack_from('<8i', native, row*64)
        cm = struct.unpack_from('<8f', native, row*64+32)
        point = dict(position=row//L, layer=row%L)
        if len(set(gi)) != K or len(set(ci)) != K or any(x < 0 or x >= E for x in gi+ci):
            raise RuntimeError('Invalid IDs')
        if any(not math.isfinite(x) or x < 0 for x in gm+cm):
            raise RuntimeError('Invalid mass')
        slots = [rank for rank in range(K) if gi[rank] != ci[rank]]
        out['ranked_slot_disagreements'] += len(slots)
        if slots and out['first_ranked_ID'] is None:
            out['first_ranked_ID'] = dict(**point, ranks=slots, GPU=list(gi), C=list(ci))
        if set(gi) != set(ci):
            out['membership_rows'] += 1
            if out['first_membership'] is None:
                out['first_membership'] = dict(**point, GPU_only=sorted(set(gi)-set(ci)),
                                                C_only=sorted(set(ci)-set(gi)))
        elif slots:
            out['order_only_rows'] += 1
            if out['first_order_only'] is None:
                out['first_order_only'] = dict(**point, GPU=list(gi), C=list(ci))
        maximum = max(abs(a-b) for a, b in zip(gm, cm))
        if maximum > out['max_rank_mass_error']:
            out['max_rank_mass_error'] = maximum
            out['max_rank_mass_at'] = point
        if gi == ci and maximum > threshold:
            out['mass_threshold_rows_same_IDs'] += 1
            if out['first_mass_same_IDs'] is None:
                out['first_mass_same_IDs'] = dict(**point, error=maximum, IDs=list(gi), GPU=list(gm), C=list(cm))
        cg = dict(zip(ci, cm))
        common_exceeded = False
        for rank, expert in enumerate(gi):
            if expert not in cg:
                continue
            error = abs(gm[rank]-cg[expert])
            if error > threshold:
                common_exceeded = True
                if out['first_common_ID_mass'] is None:
                    out['first_common_ID_mass'] = dict(**point, expert=expert, error=error,
                                                       GPU_mass=gm[rank], C_mass=cg[expert])
            if error > out['max_common_ID_mass_error']:
                out['max_common_ID_mass_error'] = error
                out['max_common_ID_mass_at'] = dict(**point, expert=expert)
        out['common_ID_mass_threshold_rows'] += int(common_exceeded)
    return out


def execute(a, guard):
    raw, binding_item = guard.read(a.binding)
    if binding_item['sha256'] != a.binding_sha:
        raise RuntimeError('Observer binding hash mismatch')
    binding = json.loads(raw)
    if binding['schema'] != 'CATEGORICAL_ROUTE_DIVERGENCE_BINDING_V1' or binding['limits'] != LIMITS:
        raise RuntimeError('Observer binding schema/caps mismatch')
    _, script = guard.read(__file__)
    if script != binding['script']:
        raise RuntimeError('Observer code changed after preparation')
    for item in binding['inputs']:
        _, got = guard.read(item['path'])
        if got != item:
            raise RuntimeError('Frozen small input changed')
    rows = []
    for case in binding['cases']:
        data = {}
        for name, item in case['routes'].items():
            data[name], got = guard.read(item['path'])
            if got != item:
                raise RuntimeError('Route extent changed')
        diag = localize(data['GPU_ids'], data['GPU_mass'], data['C_routes'], case['history'],
                        binding['numeric']['routing_mass_absolute'])
        if diag['ranked_slot_disagreements'] != case['bridge']['route_ID_disagreements'] or \
                diag['max_rank_mass_error'] != case['bridge']['route_mass']:
            raise RuntimeError('Localization disagrees with audited aggregate')
        rows.append(dict(id=case['id'], ordinal=case['ordinal'], history=case['history'], **diag))
        guard.check()
    for item in binding['inputs']:
        _, got = guard.read(item['path'])
        if got != item:
            raise RuntimeError('Frozen small input changed during observation')
    _, got = guard.read(a.binding)
    if got != binding_item:
        raise RuntimeError('Observer binding changed during observation')
    guard.check()
    write_new(a.out, dict(schema='CATEGORICAL_ROUTE_DIVERGENCE_RESULT_V1', binding=binding_item,
                         rows=rows, seconds=time.monotonic()-guard.start, OS_peak=guard.peak,
                         bytes_read=guard.bytes, all_small_input_extents_before_after_exact=True,
                         model_loads=0, history_forwards=0, backwards=0, native_calls=0, GPU_calls=0,
                         scope='Positions/layers are zero-based; first observed difference is not first causal arithmetic error. '
                               'Common-ID mass comparison removes slot-order mismatch, not different causal inputs. '
                               'No missing raw scores/margins reconstructed and no campaign gate changed.'))


def main():
    p = argparse.ArgumentParser()
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--execute', action='store_true')
    for name in ('audit', 'audit-terminal', 'capture-terminal', 'binding', 'out'):
        p.add_argument('--'+name, type=Path, required=name == 'out')
    for name in ('audit-sha', 'audit-terminal-sha', 'capture-terminal-sha', 'binding-sha'):
        p.add_argument('--'+name)
    a = p.parse_args()
    required = ('audit', 'audit_terminal', 'capture_terminal', 'audit_sha', 'audit_terminal_sha',
                'capture_terminal_sha') if a.prepare else ('binding', 'binding_sha')
    if any(getattr(a, name) is None for name in required):
        p.error('Missing mode-specific input/hash')
    if a.out.exists() or a.out.with_suffix('.fault.json').exists():
        p.error('Exclusive output namespace already consumed')
    try:
        guard = Guard()
        (prepare if a.prepare else execute)(a, guard)
    except Exception as exc:
        write_new(a.out.with_suffix('.fault.json'), dict(schema='CATEGORICAL_ROUTE_DIVERGENCE_FAULT_V1',
                  mode='prepare' if a.prepare else 'execute', error_type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == '__main__':
    main()
