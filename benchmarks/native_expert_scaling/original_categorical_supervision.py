"""FIT-only consumer for completely adjudicated fixed-head supervision."""
import json
import sys
from pathlib import Path

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B))
from original_packed_capacity import SITE, extent
sys.path.insert(0, str(SITE))


def load_fit_supervision(result_path, adjudication_path):
    """Load immutable FIT arrays and per-case onset/continuation weights.

    Weights sum to one per case. Equal-case training averages these case losses;
    sampling cases uniformly gives the same expected objective. No DEV selector
    exists in this training API. The fixed head cannot be changed without
    recompiling supervision.
    """
    import numpy as np
    result_path, adjudication_path = Path(result_path), Path(adjudication_path)
    r, a = [json.loads(p.read_bytes()) for p in (result_path, adjudication_path)]
    t = json.loads(adjudication_path.with_suffix('.terminal.json').read_bytes())
    assert r['schema'] == 'ORIGINAL_CATEGORICAL_LOSS_RESULT_V1'
    assert a['schema'] == 'ORIGINAL_CATEGORICAL_LOSS_AUDIT_V1'
    assert a['decision'] == 'LOSS_SUPERVISION_QUALIFIED' and all(a['numeric_flags'].values()) and all(r['numeric_flags'].values())
    assert a['complete_input_output_hashes'] and a['split_boundary_verified'] and a['all8808_moments_entropy_argmax_checked']
    assert a['result'] == extent(result_path) and a['freeze'] == r['freeze']
    assert t['exit_code'] == 0 and t['error'] is None and t['inputs_before_after_exact'] and t['result'] == extent(adjudication_path)
    assert a['binding']['sha256'] == r['binding_sha256'] == t['binding_sha256']
    for key in ('head', 'final_norm'):
        assert extent(r[key]['path']) == {k: r[key][k] for k in ('path', 'bytes', 'sha256')}
    assert r['head']['shape'] == [65537, 256] and r['final_norm']['shape'] == [256]
    fit = [row for row in r['records'] if row['split'] == 'FIT']
    assert [row['id'] for row in fit] == r['FIT_ids'] and len(fit) == 24
    assert set(r['FIT_ids']).isdisjoint(r['DEV_ids']) and sum(row['labels'] for row in fit) == 4422
    records = []
    for row in fit:
        arrays = {}
        for key in ('moments', 'negative_entropy', 'source_argmax'):
            item = row[key]
            assert extent(item['path']) == {k: item[k] for k in ('path', 'bytes', 'sha256')}
            x = np.fromfile(item['path'], dtype=item['dtype']).reshape(item['shape'])
            assert np.isfinite(x).all(); x.setflags(write=False); arrays[key] = x
        n = row['labels']
        assert n > 1 and arrays['moments'].shape == (n, 256)
        assert arrays['negative_entropy'].shape == arrays['source_argmax'].shape == (n,)
        assert arrays['moments'].dtype == np.dtype('<f8') and arrays['negative_entropy'].dtype == np.dtype('<f8')
        assert arrays['source_argmax'].dtype == np.dtype('<i4')
        weights = np.full(n, .5 / (n - 1), dtype='<f8'); weights[0] = .5
        assert abs(float(weights.sum()) - 1.) <= 1e-12; weights.setflags(write=False)
        positions = np.asarray(row['positions'], dtype='<i8'); assert positions.shape == (n,)
        assert np.all(np.diff(positions) == 1); positions.setflags(write=False)
        records.append(dict(id=row['id'], split='FIT', domain=row['domain'], labels=n,
                            positions=positions, loss_weights=weights, **arrays))
    return dict(head=r['head'], final_norm=r['final_norm'], records=records,
                FIT_ids=tuple(r['FIT_ids']), equal_case_factor=1./24,
                compiler_result=extent(result_path), adjudication=extent(adjudication_path),
                scope='Qualified fixed-head FIT training arrays. No model/head update, DEV access, GPU/native equivalence or quality admission.')
