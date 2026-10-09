"""CPU stored-only checks of rank96 identities and all144 saved aggregates.

No source tensors are recaptured, no predictions or bases fitted again.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
SITE = ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def extent(path):
    p = Path(path).resolve()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p))


def save(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode('utf8')
    with Path(path).open('xb') as f:
        f.write(data)


def bind(a):
    result = DOC/'original_falcon_recurrent_projection_finish_result_20261009.json'
    r = json.loads(result.read_bytes())
    terminal = result.with_suffix('.terminal.json')
    t = json.loads(terminal.read_bytes())
    assert t['exit_code'] == 0 and t['resource_gates'] and digest(result) == t['result_sha256']
    old = ROOT/'results/native_expert_scaling/original_falcon_recurrent_projection_20261009'
    new = ROOT/'results/native_expert_scaling/original_falcon_recurrent_projection_finish_20261009'
    files = [Path(__file__), Path(sys.executable), result, terminal,
        old/'first_fault.json', DOC/'original_falcon_recurrent_projection_result_20261009.launcher_failure.json',
        DOC/'ORIGINAL_FALCON_RECURRENT_PROJECTION_AUDIT_PROTOCOL_20261009.md',
        SITE/'numpy/__init__.py', SITE/'psutil/__init__.py']
    for p in r['bases']:
        assert extent(p['basis']['path']) == p['basis']
        files.append(Path(p['basis']['path']))
    for row in r['records']:
        name = f"{row['id']}.site{row['site']:02d}.metrics.json"
        matches = [p/name for p in (old, new) if (p/name).exists()]
        assert len(matches) == 1 and json.loads(matches[0].read_bytes()) == row
        files.extend(matches)
    save(a.out, dict(schema='ORIGINAL_FALCON_RECURRENT_PROJECTION_AUDIT_BINDING_V1',
        result=str(result.resolve()), terminal=str(terminal.resolve()), parent_fault=str((old/'first_fault.json').resolve()),
        parent_failure=str((DOC/'original_falcon_recurrent_projection_result_20261009.launcher_failure.json').resolve()),
        inputs=[extent(p) for p in dict.fromkeys(files)],
        limits=dict(seconds=180, reserve_seconds=20, OS_bytes=512 << 20, output_bytes=1 << 20),
        criteria=dict(identity=1e-8, pair_kernel_relative_residual_delta=1e-10,
            truncated_kernel_relative_error=1e-8, aggregate_absolute_delta=1e-12)))
    print(json.dumps(dict(binding=str(a.out), sha256=digest(a.out))))


def run(a):
    start = time.monotonic()
    assert digest(a.binding) == a.binding_sha
    b = json.loads(a.binding.read_bytes())
    assert not a.out.exists() and not a.out.with_suffix('.fault.json').exists()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    sys.path.insert(0, str(SITE))
    import numpy as np
    import psutil
    assert (np.__version__, psutil.__version__) == ('2.4.6', '7.2.2')
    proc = psutil.Process(); proc.cpu_affinity([0])
    stage = 'precheck'

    def guard():
        assert time.monotonic()-start <= b['limits']['seconds']-b['limits']['reserve_seconds']
        assert proc.memory_info().peak_wset <= b['limits']['OS_bytes']
        assert not proc.children(recursive=True)

    def check():
        for item in b['inputs']:
            assert extent(item['path']) == item, item['path']
        guard()

    try:
        check(); r = json.loads(Path(b['result']).read_bytes())
        fault = json.loads(Path(b['parent_fault']).read_bytes())
        assert len(r['bases']) == 24 and len(r['records']) == 144
        assert r['bases'] == fault['completed_bases']
        rows = {(i['id'], i['site']):i for i in r['records']}; assert len(rows) == 144
        assert len(fault['completed_measurements']) == r['inherited_case_sites'] == 18
        for row in fault['completed_measurements']:
            assert rows[(row['id'],row['site'])] == row
        assert r['new_case_sites'] == 126 and r['basis_fits'] == 0
        identity = []
        for packet in r['bases']:
            stage = f"basis{packet['site']}"; guard()
            with np.load(packet['basis']['path']) as z:
                gb, gc, v, w, rr, sv = [z[k] for k in ('G_B','G_C','V','W','R','singular_values')]
            roots = []
            for gram in (gb, gc):
                eig, vectors = np.linalg.eigh(gram)
                assert eig.min() >= -1e-12
                roots.append((vectors*np.sqrt(np.maximum(eig,0)))@vectors.T)
            rb, rc = roots; hh = rc@rb
            uu, ss, qq = np.linalg.svd(hh, full_matrices=False)
            truncated = (uu[:,:96]*ss[:96])@qq[:96]
            projected = rc@v@w.T@rb
            kernel_error = float(np.linalg.norm(projected-truncated)/np.linalg.norm(hh))
            residual = float(np.linalg.norm(hh-projected)/np.linalg.norm(hh))
            optimal = float(np.linalg.norm(ss[96:])/np.linalg.norm(ss))
            tail_delta = abs(residual-packet['optimal_pair_relative_error'])
            bio = float(np.max(np.abs(w.T@v-np.eye(96))))
            orth = float(np.max(np.abs(rr.T@rr-np.eye(96))))
            assert max(bio,orth) <= b['criteria']['identity']
            assert kernel_error <= b['criteria']['truncated_kernel_relative_error']
            assert max(tail_delta,abs(optimal-residual)) <= b['criteria']['pair_kernel_relative_residual_delta']
            identity.append(dict(site=packet['site'], biorthogonality_error=bio, orthogonality_error=orth,
                truncated_kernel_relative_error=kernel_error, measured_pair_residual=residual,
                independent_optimal_tail=optimal, reported_tail_delta=tail_delta,
                singular_value_max_delta=float(np.max(np.abs(ss-sv)))))
        stage = 'aggregates'; aggregate_delta = 0.; domains = {}; case_comparison = {}
        for site in (0,12,23):
            domains[str(site)] = {}; case_comparison[str(site)] = {}
            for split in ('FIT','DEV'):
                selected = [i for i in r['records'] if i['site']==site and i['split']==split]
                assert len(selected)==24 and len({i['domain'] for i in selected})==12
                by_domain = {}
                for domain in sorted({i['domain'] for i in selected}):
                    dr = [i for i in selected if i['domain']==domain]; assert len(dr)==2
                    by_domain[domain] = {arm:{key:dict(mean=sum(i['arms'][arm][key] for i in dr)/2,
                        worst=max(i['arms'][arm][key] for i in dr)) for key in dr[0]['arms'][arm]} for arm in dr[0]['arms']}
                domains[str(site)][split] = by_domain
                for arm in selected[0]['arms']:
                    for key in selected[0]['arms'][arm]:
                        actual = dict(mean=sum(i['arms'][arm][key] for i in selected)/24,
                            worst=max(i['arms'][arm][key] for i in selected))
                        for stat in actual:
                            aggregate_delta = max(aggregate_delta,abs(actual[stat]-r['aggregates'][str(site)][split][arm][key][stat]))
                case_comparison[str(site)][split] = {key:dict(dual_better_than_dense=sum(i['arms']['dual96'][key]<i['arms']['dense96'][key] for i in selected),
                    dual_better_than_coordinate=sum(i['arms']['dual96'][key]<i['arms']['coordinate96'][key] for i in selected),cases=24)
                    for key in selected[0]['arms']['dual96']}
        assert aggregate_delta <= b['criteria']['aggregate_absolute_delta']
        assert all(max(i['reconstruction']['scan_relative_RMS'],i['reconstruction']['output_relative_RMS'])==0 for i in r['records'])
        first = json.loads(Path(b['parent_failure']).read_bytes())
        terminal = json.loads(Path(b['terminal']).read_bytes())
        stage = 'postcheck'; check()
        result = dict(schema='ORIGINAL_FALCON_RECURRENT_PROJECTION_AUDIT_RESULT_V1',
            decision='STORED_IDENTITIES_AND_AGGREGATES_VERIFIED', freeze=a.freeze, binding_sha256=a.binding_sha,
            identities=identity, maximum_aggregate_absolute_delta=aggregate_delta, domains=domains,
            case_comparison=case_comparison, adopted_bases=24, adopted_case_sites=18,new_case_sites=126,
            combined_held_projection_families_seconds=first['elapsed_seconds']+terminal['elapsed_seconds'],
            process_pid=proc.pid, process_creation_time=proc.create_time(), OS_peak_snapshot=proc.memory_info().peak_wset,
            elapsed_seconds=time.monotonic()-start, input_pre_post_equal=True,
            source_forwards=0, optimizer_updates=0, prediction_replays=0, basis_refits=0, GPU_calls=0,
            scope='Algebra of saved FIT Grams/bases and stored metrics only; no whole-chatbot or speed admission. '
                'CPU process self measurements; no separately held launcher-family resource record.')
        assert len((json.dumps(result)+'\n').encode())<=b['limits']['output_bytes']
        save(a.out,result); print(json.dumps(dict(decision=result['decision'],sha256=digest(a.out),
            seconds=result['elapsed_seconds'],OS_peak=result['OS_peak_snapshot'],aggregate_delta=aggregate_delta)))
    except BaseException as error:
        save(a.out.with_suffix('.fault.json'),dict(stage=stage,error=repr(error),seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--bind',action='store_true'); mode.add_argument('--run',action='store_true')
    p.add_argument('--binding',type=Path); p.add_argument('--binding-sha'); p.add_argument('--freeze')
    p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    bind(a) if a.bind else run(a)
