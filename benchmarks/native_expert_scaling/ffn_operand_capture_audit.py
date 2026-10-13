"""All source FFN nodes and parent-continuity custody, stored CPU audit."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
os.environ['OPENBLAS_NUM_THREADS'] = '6'
os.environ['OMP_NUM_THREADS'] = '6'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'benchmarks/native_expert_scaling'),
               str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader


def main(a):
    import numpy as np
    import psutil
    import torch
    begin = time.monotonic(); proc = psutil.Process(); proc.cpu_affinity(list(range(6)))
    torch.set_num_threads(6); torch.set_num_interop_threads(1)
    re,te = extent(a.report),extent(a.terminal)
    assert re['sha256'] == a.report_sha and te['sha256'] == a.terminal_sha
    r,t = [json.loads(p.read_bytes()) for p in (a.report,a.terminal)]
    assert t['exit_code'] == 0 and t['result'] == re and not a.out.exists()
    assert extent(r['binding']['path']) == r['binding']
    b = json.loads(Path(r['binding']['path']).read_bytes())
    assert Path(sys.executable).resolve() == Path(b['python']).resolve() and torch.__version__ == b['versions']['torch']
    assert np.__version__ == b['versions']['numpy'] and psutil.__version__ == b['versions']['psutil']
    assert not torch.cuda.is_initialized()
    get,stream = tensor_reader(Path(b['source'])/'model.safetensors',np)
    stage = 'hashes'; gates = {}; maximum = {}; counts = dict(case_sites=0,FFN_positions=0,
        field_words=0,gate_scalar_mismatches=0,label_words=0,label_mismatches=0,
        final_state_mismatches=0,final_ring_mismatches=0,parent_trace_bytes=0)
    receipt = dict(schema='FFN_OPERAND_CAPTURE_AUDIT_TERMINAL_V1',command=sys.orig_argv,
                   pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze)
    def guard():
        assert time.monotonic()-begin <= b['limits']['audit_seconds'],'audit deadline'
        assert proc.memory_info().peak_wset <= b['limits']['OS_bytes'],'audit OS cap'
        assert not proc.children(recursive=True) and not torch.cuda.is_initialized()
    def check():
        for e in b['inputs']+r['outputs']+[re,te,r['binding']]:
            guard(); assert extent(e['path']) == e,e['path']
    def dec(x): return (x.astype('u4') << 16).view('f4').astype('f8')
    def bf(x):
        u = x.astype('f4').view('u4'); hi,lo = u>>16,u&65535
        return (hi+((lo>32768)|((lo==32768)&((hi&1)!=0))).astype('u4')).astype('u2')
    def tensor(bits): return torch.from_numpy(np.ascontiguousarray(bits)).view(torch.bfloat16)
    def bits(value): return value.contiguous().view(torch.uint16).numpy().copy()
    def test(key,x,y,tol):
        error = float(np.linalg.norm(dec(x)-dec(y))/max(np.linalg.norm(dec(y)),1e-30))
        maximum[key] = max(maximum.get(key,0),error)
        gates[key] = bool(gates.get(key,True) and np.isfinite(error) and error <= tol)
    try:
        check()
        pa = json.loads(Path(b['parent_audit']['path']).read_bytes())
        assert pa['decision'] == 'INDEPENDENT_STORED_QUALITY_PASS' and pa['report'] == b['parent_report']
        assert len(r['records']) == 48 and r['field_files'] == 6912
        parents = {x['id']:x for x in b['records']}
        for site in range(24):
            pre = f'model.layers.{site}.feed_forward.'
            U,G,D = [get(pre+key+'.weight') for key in ('up_proj','gate_proj','down_proj')]
            assert U.shape == G.shape == (4608,2048) and D.shape == (2048,4608)
            for rec in r['records']:
                stage = f'{site}/{rec["id"]}'; guard(); parent = parents[rec['id']]
                assert all(rec[k] == parent[k] for k in ('id','split','domain','input_ids','positions'))
                n = rec['history']; assert n == len(rec['input_ids'])
                fields = rec['sites'][site]; assert set(fields) == set(b['fields'])
                v = {}
                for name,width in b['fields'].items():
                    e = fields[name]; assert e['shape'] == [n,width] and e['bytes'] == n*width*2 and e['dtype'] == 'BF16'
                    v[name] = np.memmap(e['path'],'u2','r',shape=(n,width))
                    assert not np.any((v[name]&0x7f80) == 0x7f80)
                    counts['field_words'] += v[name].size
                # Matrices independently evaluated in F64,then original BF nodes.
                X = dec(v['x']); up = bf(X@U.T); gate = bf(X@G.T)
                test('up_matrix',up,v['up'],b['criteria']['source_matrix_each_case_field_relative_RMS'])
                test('gate_matrix',gate,v['gate_raw'],b['criteria']['source_matrix_each_case_field_relative_RMS'])
                # Literal Python scalar and BF16 SiLU/product primitives on CPU;
                # no premature BF16 rounding of alpha or beta.
                with torch.inference_mode():
                    scaled = bits(tensor(v['gate_raw'])*b['gate_multiplier'])
                    feature = bits(tensor(v['up'])*torch.nn.functional.silu(tensor(v['gate_scaled'])))
                counts['gate_scalar_mismatches'] += int(np.count_nonzero(scaled != v['gate_scaled']))
                test('feature_node',feature,v['features'],b['criteria']['actual_feature_node_relative_RMS'])
                raw_down = bf(dec(v['features'])@D.T)
                with torch.inference_mode(): output = bits(tensor(raw_down)*b['down_multiplier'])
                test('down_matrix_and_scalar',output,v['output'],b['criteria']['source_matrix_each_case_field_relative_RMS'])
                control = rec['controls']; old = parent['parent']['sequence']['sites'][site]
                witness = control['actual_native_trace_digests'][site]
                assert witness == dict(bytes=old['trace']['bytes'],sha256=old['trace']['sha256'])
                counts['parent_trace_bytes'] += witness['bytes']
                with np.load(control['final_snapshots'][site]['path'],allow_pickle=False) as cur,np.load(old['final']['path'],allow_pickle=False) as prior:
                    assert set(cur.files) == set(prior.files) == {'state','ring','seen','last'}
                    counts['final_state_mismatches'] += int(np.count_nonzero(cur['state'].view('u4') != prior['state'].view('u4')))
                    counts['final_ring_mismatches'] += int(np.count_nonzero(cur['ring'] != prior['ring']))
                    assert np.array_equal(cur['seen'],prior['seen']) and np.array_equal(cur['last'],prior['last'])
                assert control['attention'] == parent['parent']['sequence']['attention']
                assert control['all_parent_words_and_states_exact'] and rec['parent_score_bits_exact']
                counts['case_sites'] += 1; counts['FFN_positions'] += n
                print(json.dumps(dict(site=site,id=rec['id'],seconds=time.monotonic()-begin)),flush=True)
        for rec in r['records']:
            parent = parents[rec['id']]; shape = (len(rec['positions']),65537)
            current = np.memmap(rec['scores']['path'],'u4','r',shape=shape)
            old = np.memmap(parent['parent']['scores']['path'],'u4','r',shape=shape)
            assert rec['scores']['bytes'] == len(rec['positions'])*65537*4
            assert not np.any(current&65535), 'source BF16 logits must decode losslessly to F32'
            counts['label_words'] += current.size
            counts['label_mismatches'] += int(np.count_nonzero(current != old))
        assert counts['case_sites'] == 1152 and counts['FFN_positions'] == 541128
        assert counts['field_words']*2 == r['fields_payload_bytes'] == b['fields_payload_bytes']
        gates.update(gate_scalar_exact=counts['gate_scalar_mismatches'] == 0,
            label_bits_exact=counts['label_mismatches'] == 0,
            final_state_exact=counts['final_state_mismatches'] == 0,
            final_ring_exact=counts['final_ring_mismatches'] == 0)
        assert all(r[k] == 0 for k in ('new_generations','optimizer_updates','RESERVED_queries'))
        assert r['source_base_forwards'] == 22547 and r['LM_head_calls'] == 8808
        assert r['all411_source_parameters_exact_before_after'] and r['before_after_hashes_exact']
        assert r['all_parent_native_traces_states_cache_and_label_bits_exact']
        check(); guard()
        result = dict(schema='FFN_OPERAND_CAPTURE_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,
            report=re,terminal=te,binding=r['binding'],counts=counts,maximum=maximum,numeric_gates=gates,
            all_numeric_pass=all(gates.values()),before_after_hashes_exact=True,
            decision='INDEPENDENT_STORED_PASS' if all(gates.values()) else 'INDEPENDENT_STORED_NUMERIC_FAILURE',
            seconds=time.monotonic()-begin,OS_peak=proc.memory_info().peak_wset,
            GPU_initialized=False,source_model_forwards=0,
            scope='All source FFN field values/F64 matrix controls/literal CPU scalar and '
                  'SiLU/product nodes plus actual final states/rings/label logits audited. '
                  'Native input/output continuous stream and attention cache hashes are '
                  'producer exact-comparison receipts to qualified parent;source parameter '
                  'identity inherits before/after worker witnesses. No independent GPU '
                  'attention arithmetic/new expert fit/model quality/speed claim.')
        write(a.out,result); receipt.update(exit_code=0,error=None,result=extent(a.out))
        print(json.dumps(dict(adjudication=extent(a.out),decision=result['decision'],maximum=maximum)),flush=True)
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),stage=stage,traceback=traceback.format_exc()); raise
    finally:
        stream.close(); receipt.update(seconds=time.monotonic()-begin,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('report','terminal','out'): p.add_argument('--'+k,type=Path,required=True)
    for k in ('report-sha','terminal-sha','freeze'): p.add_argument('--'+k,required=True)
    main(p.parse_args())
