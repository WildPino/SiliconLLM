"""Small reproducible format fixtures, not an actual campaign observation."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def main(out):
    script = Path(__file__).with_name('inspect_categorical_route_divergence.py')
    raw = script.read_bytes()
    ast.parse(raw, filename=str(script))
    spec = importlib.util.spec_from_file_location('route_fixture', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    s = module.struct
    ids = tuple(range(8))
    mass = (1/16, 2/16, 3/16, 4/16, 1/16, 2/16, 1/16, 2/16)
    gi, gm, ci, cm = [ids]*6, [mass]*6, [ids]*6, [mass]*6
    ci[1], cm[1] = (1, 0, 2, 3, 4, 5, 6, 7), (mass[1], mass[0], *mass[2:])
    ci[2] = (0, 1, 2, 3, 4, 5, 6, 8)
    cm[3] = (mass[0]+1/64, mass[1]-1/64, *mass[2:])
    ib = b''.join(s.pack('<8i', *v) for v in gi)
    mb = b''.join(s.pack('<8f', *v) for v in gm)
    def native():
        return b''.join(s.pack('<8i8f', *a, *b) for a, b in zip(ci, cm))
    r = module.localize(ib, mb, native(), 1, 1e-5)
    assert (r['ranked_slot_disagreements'], r['membership_rows'], r['order_only_rows'],
            r['mass_threshold_rows_same_IDs'], r['common_ID_mass_threshold_rows']) == (3, 1, 1, 1, 1)
    assert r['first_ranked_ID']['layer'] == r['first_order_only']['layer'] == 1
    assert r['first_membership']['layer'] == 2
    assert r['first_mass_same_IDs']['layer'] == r['first_common_ID_mass']['layer'] == 3
    assert r['max_rank_mass_error'] == 1/16 and r['max_common_ID_mass_error'] == 1/64
    # A mass change hidden by a rank permutation must be localized by expert ID.
    cm[1] = (mass[1]+1/64, mass[0]-1/64, *mass[2:])
    r = module.localize(ib, mb, native(), 1, 1e-5)
    assert r['first_common_ID_mass']['layer'] == 1 and r['first_mass_same_IDs']['layer'] == 3
    exact = b''.join(s.pack('<8i8f', *a, *b) for a, b in zip(gi, gm))
    r = module.localize(ib, mb, exact, 1, 1e-5)
    assert r['ranked_slot_disagreements'] == r['max_rank_mass_error'] == 0
    assert r['first_mass_same_IDs'] is r['first_common_ID_mass'] is None
    bad_id = bytearray(exact)
    s.pack_into('<i', bad_id, 0, -1)
    bad_mass = bytearray(exact)
    s.pack_into('<f', bad_mass, 32, float('nan'))
    for broken in (exact[:-1], bytes(bad_id), bytes(bad_mass)):
        try:
            module.localize(ib, mb, broken, 1, 1e-5)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Invalid fixture accepted')
    module.receipt(dict(exit_code=0, error=None, inputs_before_after_exact=True))
    for receipt in (dict(exit_code=1, error=None, inputs_before_after_exact=True),
                    dict(exit_code=0, error='fault', inputs_before_after_exact=True),
                    dict(exit_code=0, error=None, inputs_before_after_exact=False)):
        try:
            module.receipt(receipt)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Bad parent custody accepted')
    module.write_new(out, dict(schema='CATEGORICAL_ROUTE_DIVERGENCE_PREFLIGHT_V1',
                observer=dict(path=str(script.resolve()), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
                fixture_source=dict(path=str(Path(__file__).resolve()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                fixture_flags=dict(AST=True, ordered_vs_membership=True, common_ID_mass=True,
                                   exact_routes=True, invalid_extents_IDs_mass=True, parent_receipt_rejection=True),
                scope='Synthetic byte-format fixtures only; no actual campaign routes, Guard, binding preparation, '
                      'model, neural/native/GPU/optimizer or benchmark execution. Actual post-audit observer remains unexecuted.'))
    print('Stored-route format fixtures PASS; actual observer UNEXECUTED')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
