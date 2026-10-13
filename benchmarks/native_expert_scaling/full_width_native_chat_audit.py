"""Stored bridge audit: all traces, independent F64 math, exact integer outputs.

No source model or CUDA call. Native block replay is additionally compared with
the worker's single-token execution. Source attention/FFN/head remain witnessed
by source hashes and the worker; their GPU arithmetic is not replayed here.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
B = ROOT/'benchmarks/native_expert_scaling'
sys.path[:0] = [str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader


def main(a):
    import numpy as np
    import psutil
    from row_state import interface
    from token_state import arrays,arguments,ptrs
    from original_engine_chat import ChatTokenizer
    from chatbot_falcon_usability import normalized
    from full_width_native_chat import quality
    begin = time.monotonic(); proc = psutil.Process(); proc.cpu_affinity(list(range(6)))
    report, terminal = extent(a.report), extent(a.terminal)
    assert report['sha256'] == a.report_sha and terminal['sha256'] == a.terminal_sha
    r = json.loads(a.report.read_bytes()); t = json.loads(a.terminal.read_bytes())
    assert t['exit_code'] == 0 and t['result'] == report
    assert extent(r['binding']['path']) == r['binding'] and not a.out.exists()
    b = json.loads(Path(r['binding']['path']).read_bytes())
    assert Path(sys.executable).resolve() == Path(b['python']).resolve()
    receipt = dict(schema='FULL_WIDTH_NATIVE_CHAT_AUDIT_TERMINAL_V1',command=sys.orig_argv,
                   pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze)
    stage = 'hashes'; context = None; stream = None
    maxima = {}; gates = {}; counts = dict(sequence_sites=0, contextual_rows=0,
        native_positions=0, output_words=0, output_mismatches=0,
        integer_words=0, integer_mismatches=0, raw_mismatches=0,
        BF16_mismatches=0, final_state_mismatches=0, final_ring_mismatches=0)
    def guard():
        assert time.monotonic()-begin <= b['limits']['audit_seconds'], 'audit deadline'
        assert proc.memory_info().peak_wset <= b['limits']['OS_bytes'], 'audit OS cap'
        assert not proc.children(recursive=True)
    def check():
        for e in b['inputs']+r['outputs']+[report,terminal,r['binding']]:
            guard(); assert extent(e['path']) == e, e['path']
    def decode(x):
        return (x.astype('u4') << 16).view('f4').astype('f8')
    def bf(x):
        u = x.astype('f4').view('u4'); hi = u >> 16; lo = u & 65535
        return (hi+((lo>32768)|((lo==32768)&((hi&1)!=0))).astype('u4')).astype('u2')
    def test(key,x,y,tol):
        error = float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
        maxima[key] = max(maxima.get(key,0),error)
        gates[key] = bool(gates.get(key,True) and np.isfinite(error) and error <= tol)
    try:
        check()
        assert json.loads(Path(b['parent_audit']['path']).read_bytes())['decision'] == 'INDEPENDENT_STORED_PASS'
        assert len(r['tasks']) == 16 and len(r['records']) == 48 and len(r['prefix_controls']) == 16
        sequences = [x['sequence'] for x in r['tasks']+[r['followup']]+r['records']]
        sequences += [c[k] for c in r['prefix_controls'] for k in ('wrapper','continuation')]
        assert len(sequences) == 97 and len({s['label'] for s in sequences}) == 97
        assert sum(s['seen'] for s in sequences) == r['processed_tokens'] <= b['processed_tokens_bound']
        for s in sequences:
            assert s['seen'] == len(s['consumed_ids']) > 0
            assert len(s['sites']) == 24 and len(s['attention']) == 48
            assert s['selected_functions_per_step'] == 24
            assert all(0 <= token < 65537 for token in s['consumed_ids'])
        dll,create,destroy,tokens,rows,snapshot = interface(b)
        table = np.memmap(b['table']['path'],'u2','r',shape=(65537,6704))
        source = Path(b['source'])/'model.safetensors'
        get,stream = tensor_reader(source,np)
        heads = np.repeat(np.arange(48),64)
        for export in b['exports']:
            site = export['site']; stage = f'site{site}/parameters'; guard()
            with np.load(export['file']['path'],allow_pickle=False) as z:
                p = {k:z[k].copy() for k in z.files}
            w = ptrs([p[k] for k in ['Ws','Vs','Alog','D','gamma']])
            convp = ptrs([p[k] for k in ['cw','cb','db']])
            G = get(f'model.layers.{site}.mamba.in_proj.weight') if site else None
            q64 = p['q'].astype('f8')
            for seq in sequences:
                stage = f'site{site}/{seq["label"]}'; guard()
                item = seq['sites'][site]; assert item['site'] == site
                n = seq['seen']; width = 2048 if site == 0 else 10800
                assert item['trace']['shape'] == [n,width] and item['trace']['bytes'] == n*width*2
                trace = np.memmap(item['trace']['path'],'u2','r',shape=(n,width))
                tags = np.array(seq['consumed_ids'],'u4')
                if site == 0:
                    input_rows = np.ascontiguousarray(table[tags])
                    recorded = trace
                else:
                    H = decode(trace[:,:2048])
                    expected = bf(decode(bf(H@G.T))*decode(np.array(b['MuP_bits'],'u2')))
                    input_rows = np.ascontiguousarray(trace[:,2048:8752])
                    test('contextual_generator',decode(input_rows),decode(expected),b['contextual_row_relative'])
                    recorded = trace[:,8752:]
                    counts['contextual_rows'] += n
                    del H,expected
                context = create(); assert context
                out = arrays(n,np)
                for lo in range(0,n,128):
                    hi = min(n,lo+128)
                    assert rows(context,hi-lo,tags[lo:hi].ctypes.data,input_rows[lo:hi].ctypes.data,w,convp,
                        p['q'].ctypes.data,p['scale'].ctypes.data,*arguments(out,lo,hi,np)) == 0
                state = np.empty((3072,96),'f4'); ring = np.empty((3,3584),'u2'); seen = np.zeros(1,'u8')
                assert snapshot(context,state.ctypes.data,ring.ctypes.data,seen.ctypes.data) == 0
                destroy(context); context = None
                assert int(seen[0]) == n
                counts['output_words'] += recorded.size
                counts['output_mismatches'] += int(np.count_nonzero(out['output'] != recorded))
                with np.load(item['final']['path'],allow_pickle=False) as z:
                    assert z['seen'][0] == n and z['last'][0] == tags[-1]
                    counts['final_state_mismatches'] += int(np.count_nonzero(z['state'].view('u4') != state.view('u4')))
                    counts['final_ring_mismatches'] += int(np.count_nonzero(z['ring'] != ring))
                # Independent convolution -> activation -> delta rounding equations.
                raw = decode(input_rows); bc = raw[:,3072:6656]
                pad = np.pad(bc,((3,0),(0,0)))
                summed = np.broadcast_to(p['cb'],(n,3584)).copy()
                for tap in range(4): summed += pad[tap:tap+n]*p['cw'][:,tap]
                cv = decode(bf(summed))
                activated = decode(bf(cv*np.exp(-np.logaddexp(0,-cv))))
                delta = decode(bf(raw[:,6656:]+p['db']))
                soft = np.logaddexp(0,delta); soft[delta>20] = delta[delta>20]
                fields = np.concatenate([bf(raw[:,:3072]),bf(activated),bf(soft)],axis=1)
                test('fields',decode(out['fields']),decode(fields),b['numeric']['field_relative'])
                fd = decode(out['fields']); x = fd[:,3072:6144]
                bb,cc,dt,gate = fd[:,6144:6400],fd[:,6400:6656],fd[:,6656:],fd[:,:3072]
                test('B_projection',out['Bp'].astype('f8'),bb@p['Ws'].astype('f8').T,b['numeric']['projected_relative'])
                test('C_projection',out['Cp'].astype('f8'),cc@p['Vs'].astype('f8').T,b['numeric']['projected_relative'])
                fs = np.zeros((3072,96)); latent = np.empty((n,3072)); A = -np.exp(p['Alog'].astype('f8'))
                for pos in range(n):
                    if pos % 128 == 0: guard()
                    dd = dt[pos,heads]; fs *= np.exp(dd*A[heads])[:,None]
                    fs += (dd*x[pos])[:,None]*out['Bp'][pos].astype('f8')[None,:]
                    latent[pos] = np.sum(fs*out['Cp'][pos].astype('f8')[None,:],axis=1)+p['D'][heads]*x[pos]
                test('scan',out['latent'].astype('f8'),latent,b['numeric']['scan_relative'])
                test('state',state.astype('f8'),fs,b['numeric']['state_relative'])
                ug = out['latent'].astype('f8')*gate*np.exp(-np.logaddexp(0,-gate))
                norm = bf(ug/np.sqrt(np.mean(ug**2,axis=1,keepdims=True)+1e-5)*p['gamma'])
                test('norm',decode(out['norm']),decode(norm),b['numeric']['controlled_relative'])
                Z = decode(out['norm']); scales = np.max(np.abs(Z),axis=1)/32767
                ints = np.clip(np.rint(np.divide(Z,scales[:,None],out=np.zeros_like(Z),
                              where=scales[:,None]!=0)),-32767,32767).astype('i2')
                gates['activation_quantizer'] = bool(gates.get('activation_quantizer',True) and
                    np.array_equal(ints,out['r']) and np.array_equal(scales.view('u8'),out['a'].view('u8')))
                # Integer products/sums are exact in F64 under the frozen 36.64B bound.
                dots = ints.astype('f8')@q64.T
                assert np.array_equal(dots,np.rint(dots)) and np.max(np.abs(dots)) <= 36640321536
                counts['integer_words'] += dots.size
                counts['integer_mismatches'] += int(np.count_nonzero(dots.astype('i8') != out['dots']))
                gold = dots*scales[:,None]*p['scale']
                counts['raw_mismatches'] += int(np.count_nonzero(gold.view('u8') != out['raw'].view('u8')))
                counts['BF16_mismatches'] += int(np.count_nonzero(bf(gold) != recorded))
                counts['native_positions'] += n; counts['sequence_sites'] += 1
                print(json.dumps(dict(site=site,sequence=seq['label'],seconds=time.monotonic()-begin,
                                      output_mismatches=counts['output_mismatches'])),flush=True)
                del out,trace,input_rows,recorded,raw,bc,pad,summed,cv,activated,delta,soft,fields,fd
                del x,bb,cc,dt,gate,fs,latent,ug,norm,Z,scales,ints,dots,gold
        stream.close(); stream = None
        stage = 'generation_and_metrics'; tokenizer = ChatTokenizer(b['source'])
        for task in r['tasks']+[r['followup']]:
            ids = tokenizer.encode(task['messages']); new = task['output_ids']
            assert ids == task['input_ids'] and 1 <= len(new) <= 64
            assert task['sequence']['consumed_ids'] == ids+new[:-1]
            scores = np.memmap(task['scores']['path'],'f4','r',shape=(len(new),65537))
            assert np.isfinite(scores).all() and scores.argmax(axis=1).tolist() == new
            text = tokenizer.decode(new)
            assert text == task['output_text'] and normalized(text) == task['normalized']
            assert task['correct'] == (normalized(text) == task['expected'])
            assert task['special_leak'] == any(i in set(b['special_ids'])-set(b['eos_ids']) for i in new)
            assert task['stop_policy'] == (not any(i in b['eos_ids'] for i in new[:-1]) and
                                          (new[-1] in b['eos_ids'] or len(new) == 64))
        original = next(c for c in b['cases'] if c['id'] == b['followup']['case_id'])
        candidate = next(c for c in r['tasks'] if c['id'] == b['followup']['case_id'])
        assert r['followup']['messages'] == original['messages']+[
            dict(role='assistant',content=candidate['output_text']),dict(role='user',content=b['followup']['prompt'])]
        metric_rows = []
        for rec in b['records']:
            guard(); item = next(x for x in r['records'] if x['id'] == rec['id'])
            assert item['sequence']['consumed_ids'] == rec['student_input_ids']
            n = len(rec['positions']); teacher = np.memmap(rec['logits']['path'],'u2','r',shape=(n,65537))
            scores = np.memmap(item['scores']['path'],'f4','r',shape=(n,65537))
            losses = []; wrong = 0
            for i in range(n):
                q,pv = decode(teacher[i]),scores[i].astype('f8')
                q -= np.logaddexp.reduce(q); pv -= np.logaddexp.reduce(pv)
                losses.append(float(np.sum(np.exp(q)*(q-pv))))
                wrong += int(q.argmax() != pv.argmax())
            assert np.allclose(losses,item['KL_per_label'],rtol=1e-10,atol=1e-10)
            assert abs(float(np.mean(losses))-item['KL']) <= 1e-10 and wrong == item['disagreement']
            metric_rows.append(dict(id=rec['id'],split=rec['split'],domain=rec['domain'],labels=n,
                KL=float(np.mean(losses)),disagreement=wrong,disagreement_rate=wrong/n))
        for control in r['prefix_controls']:
            first,second = control['wrapper'],control['continuation']
            assert first['consumed_ids'] == second['consumed_ids']
            with np.load(control['scores']['path'],allow_pickle=False) as z:
                same = bool(np.array_equal(z['wrapper'].view('u4'),z['continuation'].view('u4')))
            same &= first['attention'] == second['attention']
            for x,y in zip(first['sites'],second['sites'],strict=True):
                same &= x['trace']['sha256'] == y['trace']['sha256']
                with np.load(x['final']['path']) as zx,np.load(y['final']['path']) as zy:
                    same &= all(np.array_equal(zx[k].view('u1'),zy[k].view('u1')) for k in zx.files)
            assert bool(same) == control['exact']
        q = quality(r['tasks'],r['followup'],metric_rows,r['prefix_controls'],b['gates'])
        assert q['gates'] == r['quality']['gates']
        gates.update(native_output_exact=counts['output_mismatches'] == 0,
            final_state_exact=counts['final_state_mismatches'] == 0,
            final_ring_exact=counts['final_ring_mismatches'] == 0,
            integer_witness_bit_exact=counts['integer_mismatches'] == 0 and
                counts['raw_mismatches'] == 0 and counts['BF16_mismatches'] == 0)
        assert counts['sequence_sites'] == 97*24
        assert counts['native_positions'] == r['processed_tokens']*24
        check(); guard()
        audit = dict(schema='FULL_WIDTH_NATIVE_CHAT_AUDIT_V1',source=extent(Path(__file__)),
            report=report,terminal=terminal,binding=r['binding'],freeze=a.freeze,
            counts=counts,maxima=maxima,numeric_gates=gates,quality=q,
            all_numeric_pass=all(gates.values()), quality_pass=q['all_pass'],
            decision=('INDEPENDENT_STORED_QUALITY_PASS' if all(gates.values()) and q['all_pass'] else
                      'INDEPENDENT_STORED_QUALITY_FAILURE' if all(gates.values()) else
                      'INDEPENDENT_STORED_NUMERIC_FAILURE'),
            before_after_hashes_exact=True,seconds=time.monotonic()-begin,
            OS_peak=proc.memory_info().peak_wset,GPU_calls=0,source_model_forwards=0,
            scope='All97 sequence traces/all24 own native states/fields/scan/norm/integer/readout '
                  'and known categorical/generation decisions audited. GPU source attention/FFN/head '
                  'are producer/source-identity witnesses,not independently replayed GPU arithmetic. '
                  'No final CPU/useful-n/same useful50 admission.')
        write(a.out,audit); receipt.update(exit_code=0,error=None,result=extent(a.out))
        print(json.dumps(dict(adjudication=extent(a.out),decision=audit['decision'])),flush=True)
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),stage=stage,traceback=traceback.format_exc()); raise
    finally:
        if context is not None: destroy(context)
        if stream is not None: stream.close()
        receipt.update(seconds=time.monotonic()-begin,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('report','terminal','out'): parser.add_argument('--'+key,type=Path,required=True)
    for key in ('report-sha','terminal-sha','freeze'): parser.add_argument('--'+key,required=True)
    main(parser.parse_args())
