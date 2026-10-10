"""Actual 92-master transition encoding and independent integer-byte audit."""
import argparse,hashlib,json,os,sys,time,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from original_packed_capacity import SITE,extent,sha
sys.path.insert(0,str(SITE))
from original_categorical_causal_preflight import emit,read
from original_master_transition import encode,bits,digest,file_digest


def bind(a):
    import numpy as np,psutil,torch
    assert not a.binding.exists()
    names=['original_categorical_trust_step_binding_20261010.json','original_categorical_trust_step_result_20261010.json','original_categorical_trust_fsum_adjudication_20261010.json']
    paths=[DOC/n for n in names];b,r,ar=[json.loads(p.read_bytes()) for p in paths]
    assert r['decision']==ar['decision']=='GROUPED_DISCRETE_DESCENT_QUALIFIED' and all(ar['selection']['flags'].values())
    assert ar['result']==extent(paths[1]) and ar['binding']==extent(paths[0]) and ar['all3_92_master_states_110_fields_verified']
    for p in paths[1:]:
        t=json.loads(p.with_suffix('.terminal.json').read_bytes());assert t['exit_code']==0 and t['error'] is None and t['inputs_before_after_exact'] and t['result']==extent(p)
    row=next(x for x in r['records'] if x['alpha']==r['selection']['alpha']);assert row['alpha']==.001
    files=[Path(__file__),a.protocol,*paths,*[p.with_suffix('.terminal.json') for p in paths[1:]],
        Path(b['source_state']['path']),Path(row['artifacts']['masters']['path']),Path(b['head']['path']),Path(b['final_norm']['path'])]
    for m in list(sys.modules.values()):
        for k in ('__file__','__cached__'):
            p=getattr(m,k,None)
            if isinstance(p,(str,os.PathLike)) and Path(p).is_file():files.append(Path(p))
    files += [Path(sys.executable),Path(sys.executable).parent/'python312.dll']+sorted((SITE/'torch/lib').glob('*.dll'))+sorted((SITE/'numpy.libs').glob('*.dll'))+sorted((SITE/'psutil').glob('*.pyd'))
    inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in files)]
    for item in (b['source_state'],row['artifacts']['masters'],b['head'],b['final_norm']):assert extent(item['path'])=={k:item[k] for k in ('path','bytes','sha256')}
    emit(a.binding,dict(schema='ORIGINAL_MASTER_TRANSITION_PROBE_BINDING_V1',inputs=inputs,source_state=b['source_state'],target_state=row['artifacts']['masters'],source_parameters=b['source_parameters'],target_parameters=row['parameters'],head=b['head'],final_norm=b['final_norm'],parent_binding=extent(paths[0]),alpha=.001,chunk_bytes=4<<20,
        capture_limits=dict(seconds=900,reserve_seconds=120,OS_bytes=20<<30,output_bytes=4<<30,log_bytes=8<<20),audit_limits=dict(seconds=900,OS_bytes=20<<30,output_bytes=2<<20,log_bytes=8<<20),
        runtime=dict(Torch=torch.__version__,NumPy=np.__version__,psutil=psutil.__version__,zlib=zlib.ZLIB_RUNTIME_VERSION),
        scope='Stored actual baseline with canonical head/norm -> audited alpha.001 masters. One lossless bit encoding and complete independent integer-byte decoding audit; no proposal/optimizer/model/history/native/GPU/source/DEV/T4. Compression on this transition is not a future campaign bound.'))
    print(json.dumps(dict(binding_sha256=sha(a.binding),inputs=len(inputs),bytes=sum(x['bytes'] for x in inputs))),flush=True)


def states(b):
    import torch
    s=torch.load(b['source_state']['path'],weights_only=True,map_location='cpu',mmap=True)
    t=torch.load(b['target_state']['path'],weights_only=True,map_location='cpu',mmap=True)
    assert s['schema']=='ORIGINAL_DELTA_SEEDED_STATE_V1' and s['updates']==27 and not s['optimizer_partial_possible']
    assert t['schema']=='ORIGINAL_CATEGORICAL_GROUPED_MASTERS_V1' and t['alpha']==b['alpha']
    assert t['binding_sha256']==b['parent_binding']['sha256'] and t['source_state']==b['source_state']
    assert t['optimizer_updates']==0 and t['baseline_updates']==27 and t['saved_gradient_displacements']==1
    old=dict(s['model']);new=t['model'];assert len(old)==len(new)==92 and set(old)==set(new)==set(b['source_parameters'])==set(b['target_parameters'])
    for key in ('head','final_norm'):old[key]=torch.from_numpy(read(b[key]).copy())
    return old,new


def capture(a):
    import psutil,torch
    start=time.monotonic();b=json.loads(a.binding.read_bytes());assert sha(a.binding)==a.binding_sha
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';last=0.
    def guard():assert time.monotonic()-start<b['capture_limits']['seconds']-b['capture_limits']['reserve_seconds'] and proc.memory_info().peak_wset<=b['capture_limits']['OS_bytes'] and not proc.children(recursive=True)
    def progress(name,n,total):
        nonlocal last
        guard()
        if time.monotonic()-last>15:print(json.dumps(dict(parameter=name,n=n,total=total,seconds=time.monotonic()-start)),flush=True);last=time.monotonic()
    old,new=states(b);index=encode(old,new,a.directory,b['chunk_bytes'],progress)
    for row in index['parameters']:
        assert row['source_sha256']==b['source_parameters'][row['name']]['sha256'] and row['target_sha256']==b['target_parameters'][row['name']]['sha256']
        assert row['shape']==b['target_parameters'][row['name']]['shape']
    assert index['raw_bytes']==721008128*4 and index['blob_bytes']<=index['raw_bytes'];guard()
    outputs={k:extent(a.directory/name) for k,name in [('index','transition.json'),('blob','transition.bin')]}
    emit(a.out,dict(schema='ORIGINAL_MASTER_TRANSITION_PROBE_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,decision='ENCODED_PENDING_INDEPENDENT_DECODE',artifacts=outputs,raw_bytes=index['raw_bytes'],blob_bytes=index['blob_bytes'],changed_words=index['changed_words'],codec_counts=index['codec_counts'],parameters=92,numeric_updates=0,model_forwards=0,backward_calls=0,native_calls=0,GPU_calls=0,optimizer_updates=0,source_calls=0,worker_OS_peak=proc.memory_info().peak_wset,seconds=time.monotonic()-start,scope=b['scope']));print(json.dumps(dict(blob_bytes=index['blob_bytes'],raw_bytes=index['raw_bytes'],seconds=time.monotonic()-start)),flush=True)


def audit(a):
    import numpy as np,psutil,torch
    start=time.monotonic();b=json.loads(a.binding.read_bytes());r=json.loads(a.source_result.read_bytes());t=json.loads(a.source_result.with_suffix('.terminal.json').read_bytes());proc=psutil.Process();proc.cpu_affinity([1]);torch.set_num_threads(1)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and sha(a.binding)==a.binding_sha==r['binding_sha256']==t['binding_sha256']
    assert r['freeze']==a.freeze==t['freeze'] and t['exit_code']==0 and t['error'] is None and t['inputs_before_after_exact'] and t['result']==extent(a.source_result)
    def guard():assert time.monotonic()-start<b['audit_limits']['seconds'] and proc.memory_info().peak_wset<=b['audit_limits']['OS_bytes'] and not proc.children(recursive=True)
    for item in b['inputs']+t['outputs']:assert extent(item['path'])==item;guard()
    by={x['path']:x for x in t['outputs']}
    for item in r['artifacts'].values():assert item==by[item['path']]
    old,new=states(b);index=json.loads(Path(r['artifacts']['index']['path']).read_bytes());blob=Path(r['artifacts']['blob']['path'])
    assert index['schema']=='ORIGINAL_MASTER_TRANSITION_V1' and index['chunk_bytes']==b['chunk_bytes'] and index['blob_bytes']==blob.stat().st_size and index['blob_sha256']==file_digest(blob)
    assert [x['name'] for x in index['parameters']]==sorted(old)
    counts=dict(copy=0,xor_zlib=0,raw=0);offset=0;total=0;moved=0
    with blob.open('rb') as f:
        for row in index['parameters']:
            # Independent raw-byte XOR via Python integers, not NumPy XOR decoder.
            name=row['name'];o,shape=bits(old[name]);v,target_shape=bits(new[name]);assert shape==target_shape==row['shape'] and row['dtype']=='<f4'
            assert row['raw_bytes']==o.size*4;oh=hashlib.sha256();nh=hashlib.sha256();pos=0
            for c in row['chunks']:
                n=c['raw_bytes'];assert 0<n<=b['chunk_bytes'] and n%4==0 and pos+n<=row['raw_bytes']
                assert c['raw_offset']==pos and c['encoded_offset']==offset and 0<=c['encoded_bytes']<=n
                source=memoryview(o).cast('B')[pos:pos+n].tobytes();target=memoryview(v).cast('B')[pos:pos+n]
                assert digest(source)==c['source_sha256'];data=f.read(c['encoded_bytes']);assert len(data)==c['encoded_bytes'] and digest(data)==c['encoded_sha256']
                codec=c['codec']
                if codec=='copy':assert not data;decoded=source
                elif codec=='raw':assert len(data)==n;decoded=data
                else:
                    assert codec=='xor_zlib' and 0<len(data)<n;z=zlib.decompressobj();delta=z.decompress(data,n+1)
                    assert z.eof and not z.unused_data and not z.unconsumed_tail and len(delta)==n
                    decoded=(int.from_bytes(source,'little')^int.from_bytes(delta,'little')).to_bytes(n,'little')
                assert decoded==target and digest(decoded)==c['target_sha256'];oh.update(source);nh.update(decoded)
                changed=int(np.count_nonzero(np.frombuffer(source,'<u4')!=np.frombuffer(decoded,'<u4')))
                assert changed==c['changed_words'];moved+=changed;counts[codec]+=1;pos+=n;total+=n;offset+=len(data);guard()
            assert pos==row['raw_bytes'] and oh.hexdigest()==row['source_sha256']==b['source_parameters'][name]['sha256']
            assert nh.hexdigest()==row['target_sha256']==b['target_parameters'][name]['sha256']
        assert not f.read(1) and offset==index['blob_bytes']
    assert total==index['raw_bytes']==r['raw_bytes']==721008128*4 and moved==index['changed_words']==r['changed_words'] and counts==index['codec_counts']==r['codec_counts']
    assert r['parameters']==92 and r['decision']=='ENCODED_PENDING_INDEPENDENT_DECODE' and r['blob_bytes']==index['blob_bytes']<=total
    assert all(r[k]==0 for k in ('numeric_updates','model_forwards','backward_calls','native_calls','GPU_calls','optimizer_updates','source_calls'))
    assert r['worker_OS_peak']<=b['capture_limits']['OS_bytes'] and r['seconds']<b['capture_limits']['seconds']-b['capture_limits']['reserve_seconds'];guard()
    emit(a.out,dict(schema='ORIGINAL_MASTER_TRANSITION_PROBE_AUDIT_V1',result=extent(a.source_result),binding=extent(a.binding),decision='LOSSLESS_MASTER_TRANSITION_QUALIFIED',all92_parameters_721008128_words_exact=True,complete_input_output_hashes=True,independent_Python_integer_byte_XOR=True,raw_bytes=total,blob_bytes=offset,changed_words=moved,codec_counts=counts,numeric_updates=0,model_forwards=0,backward_calls=0,native_calls=0,GPU_calls=0,optimizer_updates=0,source_calls=0,seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,scope='Full stored decode/custody, no numerical proposal or inference replay. Compression evidence for one real transition only.'))
    print(json.dumps(dict(decision='LOSSLESS_MASTER_TRANSITION_QUALIFIED',seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('bind','capture-worker','audit-worker'):p.add_argument('--'+k,action='store_true')
    for k in ('binding','directory','out','source-result','protocol'):p.add_argument('--'+k,type=Path)
    for k in ('binding-sha','freeze'):p.add_argument('--'+k)
    a=p.parse_args()
    if a.bind:bind(a)
    elif a.capture_worker:capture(a)
    elif a.audit_worker:audit(a)
    else:raise RuntimeError('Use held CPU launcher')
