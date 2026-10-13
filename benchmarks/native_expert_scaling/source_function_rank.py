"""FIT-only equal-history finite linear response geometry, not a model converter."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
SITE = ROOT / 'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0] = [str(B), str(SITE)]
from falcon_auxiliary_s6_transport import extent, write, tensor_reader

START = time.monotonic()
SECTIONS = [('gate', 0, 3072), ('x', 3072, 6144), ('B', 6144, 6400),
            ('C', 6400, 6656), ('delta', 6656, 6704)]
RANKS = [128, 256, 512, 1024, 2048]
OWN = 'reconstructed_denominator__raw_scores'


def prepare(a):
    import numpy as np
    import psutil
    cp = DOC / 'true_generator_input_capture_result_v3_20261013.json'
    ca = DOC / 'true_generator_input_capture_adjudication_20261013.json'
    tp = DOC / 'true_input_temporal_result_20261013.json'
    ta = DOC / 'true_input_temporal_adjudication_20261013.json'
    c = json.loads(cp.read_bytes())
    t = json.loads(tp.read_bytes())
    cad = json.loads(ca.read_bytes())
    tad = json.loads(ta.read_bytes())
    assert cad['decision'] == 'INDEPENDENT_DATA_AND_SOURCE_CONTINUITY_PASS'
    assert tad['decision'] == 'INDEPENDENT_STORED_PASS'
    assert cad['report'] == extent(cp) and tad['report'] == extent(tp)
    cb = DOC / 'true_generator_input_capture_binding_v4_20261013.json'
    assert extent(cb)['sha256'] == c['binding_sha256']
    assert t['temporal_all_pass'] and all(all(t['gates'][str(s)]['reconstructed_denominator'].values())
                                         for s in [0,12,23])
    previous = DOC / 'true_input_function_replay_binding_20261013.json'
    pb = json.loads(previous.read_bytes())
    storage = DOC / 'source_function_rank_storage_20261013.json'
    st = json.loads(storage.read_text(encoding='utf-8-sig'))
    assert not a.out.exists() and not a.directory.exists()
    assert a.directory.resolve() == Path(st['new_output_directory']).resolve()
    by_id = {r['id']: r for r in c['records'] if r['split'] == 'FIT'}
    assert len(by_id) == 24 and sum(r['history'] for r in by_id.values()) == 11148
    records = []
    for site in [0, 12, 23]:
        for r in t['records']:
            if r['site'] != site or r['split'] != 'FIT':
                continue
            q = by_id[r['id']]
            assert r['history'] == q['history'] and r['domain'] == q['domain']
            h = q['sites'][site]['fields']['H']
            assert h['dtype'] == 'BF16' and h['shape'] == [r['history'], 2048]
            records.append(dict(site=site, id=r['id'], split='FIT', domain=r['domain'],
                                history=r['history'], H=h, native=r['file']))
    assert len(records) == 72
    paths = [Path(__file__), B / 'source_function_rank_audit.py',
             B / 'falcon_auxiliary_s6_transport.py',
             DOC / 'SOURCE_FUNCTION_RANK_PROTOCOL_20261013.md', storage,
             cp, ca, tp, ta, previous, cb,
             Path(t['binding']['path']), Path(pb['source_weights']), Path(pb['source_config']),
             Path(sys.executable), Path(sys.executable).parent / 'python312.dll',
             Path(np.__file__), Path(np._core._multiarray_umath.__file__),
             Path(np.linalg._umath_linalg.__file__), Path(psutil.__file__)]
    nr = Path(np.__file__).resolve().parent
    pr = Path(psutil.__file__).resolve().parent
    paths += list((nr.parent / 'numpy.libs').glob('*.dll')) + list(pr.glob('*.pyd'))
    paths += [ROOT / p for p in ['benchmarks/phase60/engine.c',
              'benchmarks/donor_adaptation/configs/_manifest.json',
              'benchmarks/donor_adaptation/density/build_document_holdout.py',
              'docs/research/RESEARCH_INDEX.md']]
    paths += [Path(r[k]['path']) for r in records for k in ['H', 'native']]
    inputs = [extent(p) for p in dict.fromkeys(paths)]
    assert time.monotonic() - START < 900 and psutil.Process().memory_info().peak_wset < 6 << 30
    b = dict(schema='SOURCE_FUNCTION_RANK_BINDING_V1', freeze=a.freeze,
             python=str(Path(sys.executable).resolve()),
             versions=dict(numpy=np.__version__, psutil=psutil.__version__),
             source_weights=pb['source_weights'], source_config=pb['source_config'],
             capture=extent(cp), capture_adjudication=extent(ca),
             capture_binding=extent(cb),
             temporal=extent(tp), temporal_adjudication=extent(ta),
             sites=[0, 12, 23], sections=SECTIONS, ranks=RANKS, records=records,
             output_key=OWN, inputs=inputs, output_directory=str(a.directory.resolve()),
             numeric=dict(matrix_relative=1e-12, symmetry_relative=1e-12,
                          eigen_residual_relative=1e-10, orthogonal_max=1e-10,
                          negative_eigenvalue_relative=1e-10,
                          root_square_relative=1e-10, direct_squared_spectrum_relative=1e-8,
                          metadata_absolute_or_relative=1e-10),
             limits=dict(seconds=900, audit_seconds=1200, OS_bytes=6 << 30,
                         output_bytes=1 << 30),
             scope='Uncentered equal-history FIT-only globally-linear generator and output-space '
                   'response-rank envelopes at fixed ranks. No DEV array reads/basis choice, '
                   'source/GPU/native rerun, factor kernels, nonlinear field/state/causal/chatbot '
                   'quality or speed/ternary expert/n admission. Stored F32 outputs define '
                   'the output proxy; they are not exact real O*Z contractions.')
    write(a.out, b)
    print(json.dumps(dict(binding=extent(a.out), seconds=time.monotonic()-START)), flush=True)


def decode(bits, np):
    return (bits.astype('<u4') << 16).view('<f4').astype('<f8')


def spectral(matrix, np, numeric):
    asym = float(np.linalg.norm(matrix-matrix.T)/np.linalg.norm(matrix))
    matrix = (matrix+matrix.T)*.5
    values, vectors = np.linalg.eigh(matrix)
    positive = np.maximum(values, 0.)
    total = float(positive.sum())
    orth = float(np.max(np.abs(vectors.T@vectors-np.eye(len(values)))))
    residual = float(np.linalg.norm(matrix@vectors-vectors*values)/np.linalg.norm(matrix))
    negative = float(max(0., -values[0])/values[-1])
    metrics = dict(asymmetry_relative=asym, eigen_residual_relative=residual,
                   orthogonal_max=orth, raw_minimum_eigenvalue=float(values[0]),
                   raw_maximum_eigenvalue=float(values[-1]),
                   negative_eigenvalue_relative=negative,
                   negative_modes=int((values<0).sum()), exact_zero_modes=int((values==0).sum()),
                   negative_mass_removed=float(-values[values<0].sum()),
                   negative_mass_relative=float(-values[values<0].sum()/total),
                   trace=float(np.trace(matrix)), nonnegative_spectrum_sum=total)
    gates = dict(symmetry=asym<=numeric['symmetry_relative'],
                 residual=residual<=numeric['eigen_residual_relative'],
                 orthogonality=orth<=numeric['orthogonal_max'],
                 PSD=negative<=numeric['negative_eigenvalue_relative'])
    return matrix, values, vectors, metrics, gates


def tails(values, np):
    v = np.maximum(values, 0.)
    total = float(v.sum())
    assert total > 0
    return {str(r): dict(discarded_squared_energy=float(v[:-r].sum()),
                        relative_squared_energy=float(v[:-r].sum()/total),
                        relative_RMS=float(np.sqrt(v[:-r].sum()/total))) for r in RANKS}


def run(a):
    import numpy as np
    import psutil
    assert extent(a.binding)['sha256'] == a.binding_sha
    b = json.loads(a.binding.read_bytes())
    assert Path(sys.executable).resolve() == Path(b['python']).resolve()
    assert np.__version__ == b['versions']['numpy'] and psutil.__version__ == b['versions']['psutil']
    assert not a.directory.exists() and not a.out.exists()
    assert a.directory.resolve() == Path(b['output_directory']).resolve()
    proc = psutil.Process()
    proc.cpu_affinity(list(range(6)))
    receipt = dict(schema='SOURCE_FUNCTION_RANK_TERMINAL_V1', command=sys.orig_argv,
                   pid=proc.pid, creation_time=proc.create_time(), freeze=a.freeze,
                   timing_admitted=False)
    stream = None
    stage = 'startup'
    rows = []

    def guard():
        assert time.monotonic()-START <= b['limits']['seconds']
        assert proc.memory_info().peak_wset <= b['limits']['OS_bytes']
        assert not proc.children(recursive=True)
        if a.directory.exists():
            assert sum(p.stat().st_size for p in a.directory.iterdir()) <= b['limits']['output_bytes']

    def check():
        for e in b['inputs']:
            guard()
            assert extent(e['path']) == {k:e[k] for k in ['path','bytes','sha256']}, e['path']

    try:
        check()
        a.directory.mkdir()
        get, stream = tensor_reader(b['source_weights'], np)
        for site in b['sites']:
            stage = f'{site}/moments'
            guard()
            rr = [r for r in b['records'] if r['site']==site]
            assert len(rr) == 24 and all(r['split']=='FIT' for r in rr)
            CH = np.zeros((2048,2048))
            CY = np.zeros_like(CH)
            for r in rr:
                guard()
                H = decode(np.fromfile(r['H']['path'], '<u2').reshape(r['H']['shape']), np)
                with np.load(r['native']['path'], allow_pickle=False) as z:
                    Y = z[b['output_key']].astype('<f8')
                assert H.shape == Y.shape == (r['history'],2048)
                assert np.isfinite(H).all() and np.isfinite(Y).all()
                CH += (H.T@H)/(24*r['history'])
                CY += (Y.T@Y)/(24*r['history'])
            stage = f'{site}/spectra'
            guard()
            CH, eh, uh, hm, hg = spectral(CH, np, b['numeric'])
            L = (uh*np.sqrt(np.maximum(eh,0.)))@uh.T
            rooterr = float(np.linalg.norm(L@L-CH)/np.linalg.norm(CH))
            hg['root_square'] = rooterr <= b['numeric']['root_square_relative']
            hm['root_square_relative'] = rooterr
            W = get(f'model.layers.{site}.mamba.in_proj.weight')
            assert W.shape == (6704,2048)
            energies = np.array([np.sum((W[lo:hi]@CH)*W[lo:hi]) for _,lo,hi in SECTIONS])
            assert np.isfinite(energies).all() and (energies>0).all()
            scales = np.concatenate([np.full(hi-lo, 1/np.sqrt(en))
                                     for (_,lo,hi),en in zip(SECTIONS, energies)])
            SW = scales[:,None]*W
            K = L@(SW.T@SW)@L
            K, ek, uk, km, kg = spectral(K, np, b['numeric'])
            km['balanced_trace_relative_to_five'] = float(abs(np.trace(K)-5)/5)
            kg['balanced_trace'] = km['balanced_trace_relative_to_five']<=b['numeric']['root_square_relative']
            CY, ey, uy, ym, yg = spectral(CY, np, b['numeric'])
            path = a.directory/f'site{site:02d}.npz'
            np.savez(path, C_H=CH, H_eigenvalues=eh, H_eigenvectors=uh, root=L,
                     section_energies=energies, row_scales=scales, K=K,
                     input_eigenvalues=ek, input_eigenvectors=uk, C_Y=CY,
                     output_eigenvalues=ey, output_eigenvectors=uy)
            rows.append(dict(site=site, cases=len(rr), positions=sum(r['history'] for r in rr),
                             file=extent(path), input_covariance_metrics=hm,
                             input_metrics=km, output_metrics=ym,
                             numeric_gates=dict(covariance=hg,input=kg,output=yg),
                             section_energies={s:float(e) for (s,_,_),e in zip(SECTIONS,energies)},
                             input_tails=tails(ek,np), output_tails=tails(ey,np)))
            print(json.dumps(dict(site=site, complete=len(rows), seconds=time.monotonic()-START,
                                  input_tails=rows[-1]['input_tails'], output_tails=rows[-1]['output_tails'])), flush=True)
            del W, SW, CH, CY, eh, uh, L, K, ek, uk, ey, uy, H, Y
        stream.close()
        stream = None
        check()
        guard()
        ok = all(all(g.values()) for r in rows for g in r['numeric_gates'].values())
        result = dict(schema='SOURCE_FUNCTION_RANK_RESULT_V1', binding=extent(a.binding),
                      freeze=a.freeze, records=rows, numeric_all_pass=ok,
                      before_after_hashes_exact=True, DEV_arrays_read=0, source_forwards=0,
                      native_calls=0, optimizer_updates=0, factor_kernels_built=0,
                      decision='COMPLETE_PENDING_INDEPENDENT_AUDIT',
                      output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()),
                      seconds=time.monotonic()-START, OS_peak=proc.memory_info().peak_wset,
                      scope=b['scope'])
        write(a.out,result)
        receipt.update(exit_code=0,error=None,result=extent(a.out))
        print(json.dumps(dict(result=extent(a.out), numeric_all_pass=ok)), flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),stage=stage,traceback=traceback.format_exc())
        raise
    finally:
        if stream is not None:
            stream.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--prepare',action='store_true')
    m.add_argument('--run',action='store_true')
    for k in ['binding','directory','out']:
        p.add_argument('--'+k,type=Path)
    for k in ['binding-sha','freeze']:
        p.add_argument('--'+k)
    a=p.parse_args()
    if a.run:
        run(a)
    else:
        import psutil
        proc=psutil.Process()
        receipt=dict(schema='SOURCE_FUNCTION_RANK_PREPARATION_TERMINAL_V1',
                     command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),
                     freeze=a.freeze,timing_admitted=False)
        try:
            prepare(a)
            receipt.update(exit_code=0,error=None,result=extent(a.out))
        except BaseException as e:
            import traceback
            receipt.update(exit_code=1,error=repr(e),traceback=traceback.format_exc())
            raise
        finally:
            receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset)
            write(a.out.with_suffix('.preparation_terminal.json'),receipt)
