"""Independent full context->norm/residual/A16 head/logit retention audit."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import time
START = time.monotonic()
import argparse
from datetime import datetime
import hashlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import traceback
import numpy as np
import psutil
import threadpoolctl

ROOT = Path.cwd(); DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
OUT = ROOT / 'results/native_expert_scaling/meth477_r1_head_observable_recovery'
SOURCE_OUT = ROOT / 'results/native_expert_scaling/meth477_switch_head_observable'
FAIL = DOC / 'meth477_switch_head_observable_result.failure.json'
RAW = DOC / 'meth477_r1_head_observable_result.json'
BIND = DOC / 'meth477_r1_prospective_bindings.json'
BIND_SHA = 'e304e512c361f03af06db46bbe7b0497f841e2516709aae02af2ad126e379bb9'
PROTO = DOC / 'METH_477_R1_HEAD_OBSERVABLE_RECOVERY_PROTOCOL_20261005.md'
parser = argparse.ArgumentParser(); parser.add_argument('--out', required=True, type=Path); args = parser.parse_args()
assert args.out.resolve() == RAW.resolve()
assert not OUT.exists() and not RAW.exists()
OUT.mkdir()
progress=(OUT/'progress.jsonl').open('x',encoding='utf8')
fatal=(OUT/'fatal_native.log').open('xb')
import faulthandler
faulthandler.enable(file=fatal,all_threads=True)
phase='binding_admission'
def checkpoint(**values):
    progress.write(json.dumps({'phase':phase,'seconds':time.monotonic()-START,'pid':os.getpid(),**values})+'\n');progress.flush()
checkpoint()
assert not RAW.with_suffix('.failure.json').exists()
peak = hashed = 0
parent = psutil.Process(); parent.cpu_affinity([0])
limits = threadpoolctl.threadpool_limits(limits=1)
np.seterr(over='raise', invalid='raise', divide='raise', under='ignore')
def guard():
    global peak
    info = parent.memory_info(); peak = max(peak, info.rss, getattr(info, 'peak_wset', 0))
    assert time.monotonic() - START <= 600 and peak <= 2 << 30
    assert sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())+(RAW.stat().st_size if RAW.exists() else 0)<=8<<20
def digest(path):
    global hashed
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(4 << 20): h.update(data); hashed += len(data); guard()
    return h.hexdigest()
def head(path):
    rel = Path(path).resolve().relative_to(ROOT).as_posix()
    assert Path(path).read_bytes() == subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'cat-file', '--filters', '--path=' + rel, 'HEAD:' + rel])
def exact(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), (a.shape, b.shape)
def fail_hook(kind, value, tb):
    path = OUT / 'retention_audit_first_failure.json'
    record = {'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'failure_source_sha256': hashlib.sha256(FAIL.read_bytes()).hexdigest(),
              'traceback': ''.join(traceback.format_exception(kind, value, tb)), 'seconds': time.monotonic() - START, 'OS_peak_bytes': peak}
    if not path.exists():
        with path.open('xb') as stream: stream.write((json.dumps(record, indent=2) + '\n').replace('\n', '\r\n').encode())
    sys.__excepthook__(kind, value, tb)
sys.excepthook = fail_hook
assert digest(BIND)==BIND_SHA;head(BIND)
b=json.loads(BIND.read_bytes())
assert digest(FAIL)=='f28571f5ee24d07ef6e786fcef3852516069305c1bdead5731b0686d4274c339'
j=json.loads(FAIL.read_bytes());assert len(j['apparatus_gates'])==3 and all(j['apparatus_gates'].values())
assert j['native_or_model_commands']==2 and j['updates']==0
head(PROTO)
assert sys.version == b['runtime']['python'] and Path(sys.executable).resolve() == Path(b['runtime']['executable']).resolve()
for path, h in j['scientific_sources'].items(): head(path); assert digest(path) == h
head(__file__)
assert (ROOT / '.gitattributes').read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:.gitattributes'])
for rel, r in b['helpers'].items(): head(ROOT / rel); assert digest(ROOT / rel) == r['sha256']
for name, r in b['records'].items(): head(DOC / name); assert (DOC / name).stat().st_size == r['bytes'] and digest(DOC / name) == r['sha256']
for path, r in b['runtime']['files'].items(): assert Path(path).stat().st_size == r['bytes'] and digest(path) == r['sha256']
for package, item in b['runtime']['packages'].items():
    assert metadata.version(package) == item['version']
    for path, r in item['files'].items(): assert Path(path).stat().st_size == r['bytes'] and digest(path) == r['sha256']
for module in (np, psutil, threadpoolctl):
    assert module.__version__ == b['runtime']['packages'][module.__name__]['version']
    assert str(Path(module.__file__).resolve()) in b['runtime']['packages'][module.__name__]['files']
for pool in threadpoolctl.threadpool_info(): assert pool['num_threads'] == 1 and str(Path(pool['filepath']).resolve()) in b['runtime']['packages']['numpy']['files']
for key in ('preparation_helper', 'controller_text_generation_helper'): assert digest(b[key]['path']) == b[key]['sha256']
for r in b['predecessor_preparation_helpers']: assert digest(r['path']) == r['sha256']
for path, r in b['native_files'].items(): assert Path(path).stat().st_size == r['bytes'] and digest(path) == r['sha256']
assert len(b['retained_inventory']) == 6591
for r in b['retained_inventory']:
    p = Path(r['path']); assert p.stat().st_size == r['bytes'] and p.stat().st_mtime_ns == r['mtime_ns'] and digest(p) == r['sha256']
for r in [*b['additional_source_files'], *b['native_FFN_controls']]: assert Path(r['path']).stat().st_size == r['bytes'] and digest(r['path']) == r['sha256']
for path, r in b['integer_fixtures'].items(): assert digest(path) == r['sha256']
payload = Path(j['artifact']['payload']); assert j['artifact'] == b['original_artifact'] and digest(payload) == j['artifact']['sha256']
assert [payload.stat().st_size, payload.stat().st_mtime_ns] == j['payload_stat_before'] == [b['payload_stat_before']['bytes'], b['payload_stat_before']['mtime_ns']]
assert digest(j['artifact']['manifest']) == j['artifact']['manifest_sha256']
for r in b['first_failure_files']: assert Path(r['path']).stat().st_size==r['bytes'] and digest(r['path'])==r['sha256']
assert (SOURCE_OUT / 'fatal_native.log').stat().st_size == 0
assert [c['returncode'] for c in j['commands']]==[0,0,2]
own = {os.getpid(), *(p.pid for p in parent.parents())}; daemons = []
for p in psutil.process_iter(['name', 'cmdline']):
    if p.pid in own: continue
    try:
        name, argv = (p.info['name'] or '').lower(), p.info['cmdline'] or []
        if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve(): daemons.append(p.pid); continue
        assert not (name.startswith('python') or name.startswith('clang') or (name.startswith('meth') and name.endswith('.exe'))), (p.pid, name)
    except (psutil.NoSuchProcess, psutil.AccessDenied): pass
events_path = ROOT / 'results/native_expert_scaling/meth477_first_failure_windows_terminal.json'
events = json.loads(events_path.read_bytes()); assert events['query_available'] and not events['matching_scientific_events']
expected_instances = {(j['main_process_instance']['pid'], j['main_process_instance']['create_time_unix'])}
for cmd in j['commands']:
    expected_instances.add((cmd['process_instance']['pid'], cmd['process_instance']['create_time_unix']))
    expected_instances.update((r['pid'], r['create_time_unix']) for r in cmd['descendant_process_peaks'])
assert {(r['pid'], r['create_time_unix']) for r in events['instances']} == expected_instances
source_progress=[json.loads(line) for line in (SOURCE_OUT/'progress.jsonl').read_text().splitlines()]
assert source_progress[-1]['terminal_failure'] and source_progress[-1]['seconds']<=180
assert j['resource_on_failure']['OS_peak_bytes']<=2<<30
assert (SOURCE_OUT/'native.stderr.log').read_bytes()==b'switch_reference_error:ALL_terminal_counts_EOF\r\n'
gates={'immutable_first_failed477_sources_runtime_prior6591_extra_inputs_actual_terminal_instances_bound':True}
head(__file__)
checkpoint(file_bytes_hashed=hashed)
assert time.monotonic()-START<=150
phase='independent_complete_head_and_observable_recovery'
def quant(x):
    maximum = np.max(np.abs(x), axis=1)
    alpha = np.where(maximum == 0, np.float32(1), maximum / np.float32(32767)).astype('<f4')
    codes = np.clip(np.rint(x / alpha[:, None]), -32767, 32767).astype('<i2')
    return codes, alpha
def norm(x, w, epsilon):
    total = np.zeros(len(x), '<f8')
    for coordinate in range(768): total += (x[:, coordinate] * x[:, coordinate]).astype('<f8')
    mean = (total / np.float64(768)).astype('<f4')
    scale = np.float32(1) / np.sqrt(mean + epsilon)
    return ((x * scale[:, None]) * w).astype('<f4')
def scaled(codes, w, row_scales, alpha):
    dots = codes.astype('<f8') @ w.T
    # Integer terms and every absolute partial sum <=768*128*32767<2^53.
    assert np.isfinite(dots).all() and np.equal(dots, np.rint(dots)).all()
    return ((dots * row_scales.astype('<f8')[None, :]) * alpha.astype('<f8')[:, None]).astype('<f4')
fixture = ROOT / 'results/native_expert_scaling/meth356_switch_all_a16_contract/integer_cases.bin'
answers = ROOT / 'results/native_expert_scaling/meth374_switch_physical_workers_contract/head-int-dot.bin'
fd, ad = fixture.read_bytes(), answers.read_bytes(); assert fd[:12] == struct.pack('<8sI', b'SWI8D001', 13) and ad[:12] == struct.pack('<8sI', b'SW16R001', 13)
assert (SOURCE_OUT / 'head-int-dot.bin').read_bytes() == ad
fc = ac = 12
for i in range(13):
    nr, nc = struct.unpack_from('<2I', fd, fc); fc += 8
    w = np.frombuffer(fd, '<i1', nr * nc, fc).reshape(nr, nc).astype('<f8'); fc += nr * nc
    si = np.frombuffer(fd, '<f4', nr, fc); fc += nr * 4
    x = np.frombuffer(fd, '<f4', nc, fc).reshape(1, nc); fc += nc * 4
    assert struct.unpack_from('<2I', ad, ac) == (nr, nc); ac += 8
    y = np.frombuffer(ad, '<f4', nr, ac); ac += nr * 4
    q = np.frombuffer(ad, '<i2', nc, ac); ac += nc * 2
    alpha = np.frombuffer(ad, '<f4', 1, ac); ac += 4
    dots = np.frombuffer(ad, '<i8', nr, ac); ac += nr * 8
    q1, a1 = quant(x); exact(q1[0], q); exact(a1, alpha)
    exact((q1.astype('<f8') @ w.T).astype('<i8')[0], dots); exact(scaled(q1, w, si, a1)[0], y)
assert fc == len(fd) and ac == len(ad)
gates['independent_exact_I8_A16_codes_dot_scales13_native_fixtures'] = True
export = json.loads((DOC / 'meth380_switch_base128_export_result.json').read_bytes())
def tensor(name):
    item = export['tensors'][name]
    with payload.open('rb') as stream:
        stream.seek(item['offset']); data = stream.read(item['bytes'])
        if item['encoding']:
            stream.seek(item['scale_offset']); sb = stream.read(item['scale_bytes'])
    assert hashlib.sha256(data).hexdigest() == item['sha256']
    if item['encoding']:
        assert hashlib.sha256(sb).hexdigest() == item['scale_sha256']
        return np.frombuffer(data, '<i1').reshape(item['shape']).astype('<f8'), np.frombuffer(sb, '<f4')
    return np.frombuffer(data, '<f4').reshape(item['shape'])
wi_norm = tensor('decoder.block.11.layer.2.layer_norm.weight')
final_norm = tensor('decoder.final_layer_norm.weight')
hw, hs = tensor('lm_head.weight'); assert hw.shape == (32128, 768) and len(hs) == 32128
epsilon = np.float32(export['original_config']['layer_norm_epsilon'])
import base64
from decimal import Decimal, localcontext
FIELDS=('kl','tv','margin_source','margin_candidate','delta_min','delta_max','pmax_source','pmax_candidate','fisher_half','nll_source','nll_candidate','delta_nll','error_energy','reference_energy','route_probability','range_KL_bound')
TOYS=[([0.,1.,-1.],[0.,1.+2.**-10,-1.-2.**-10],1),([10000.,9999.,-10000.],[10016.,10015.,-9984.],0),([0.,0.,0.],[0.,0.,0.],2),([1000.,999.,-1000.],[999.,1000.,-999.],1),([0.,-1.,-2.],[0.,-1.+2.**-20,-2.-2.**-20],2),([0.,0.,-1.],[0.,2.**-10,-1.],0)]
def oracle(source,candidate,label):
    with localcontext() as cc:
        cc.prec=100
        a,z=[list(map(Decimal.from_float,v)) for v in (source,candidate)]
        pa=[(v-max(a)).exp() for v in a];pz=[(v-max(z)).exp() for v in z]
        la=max(a)+sum(pa).ln();lz=max(z)+sum(pz).ln();pa=[v/sum(pa) for v in pa];pz=[v/sum(pz) for v in pz]
        dd=[v-u for u,v in zip(a,z)];mu=sum(p*d for p,d in zip(pa,dd))
        values=[sum(p*((u-la)-(v-lz)) for p,u,v in zip(pa,a,z)),sum(abs(p-q) for p,q in zip(pa,pz))/2,
                sorted(a)[-1]-sorted(a)[-2],sorted(z)[-1]-sorted(z)[-2],min(dd),max(dd),max(pa),max(pz),sum(p*(d-mu)**2 for p,d in zip(pa,dd))/2,
                la-a[label],lz-z[label],(lz-z[label])-(la-a[label]),Decimal(0),Decimal(0),Decimal(0),(max(dd)-min(dd))**2/8]
        return np.array(list(map(float,values)),dtype='<f8')
controls=[json.loads(v) for v in (SOURCE_OUT/'controls.stdout.log').read_text().splitlines()]
assert len(controls)==8
for k,toy in enumerate(TOYS):
    expect=oracle(*toy);actual=np.array(controls[k]['metrics']);assert controls[k]['control']==k
    assert np.all(np.abs(actual-expect)<=1e-11+1e-10*np.abs(expect))
assert controls[6]=={'SHA_controls':[hashlib.sha256(b'').hexdigest(),hashlib.sha256(b'abc').hexdigest()]}
assert controls[7]['control_terminal'] and controls[7]['integer_fixtures']==13 and controls[7]['worker_physical_cores']==6
assert [v['actual_mask'] for v in controls[7]['worker_affinity']]==[1,4,16]
gates['independent_integer13_Decimal100digits6_softmax_toys_and_SHA_controls']=True
ctype=np.dtype([('qid','<u8'),('batch','<u4'),('position','<u4'),('pre','<f4',(768,)),('head_codes','<i2',(768,)),('head_alpha','<f4')]);assert ctype.itemsize==4628
context_path=ROOT/'results/native_expert_scaling/meth476_switch_last_layer_context/context.bin'
with context_path.open('rb') as f:assert f.read(16)==struct.pack('<8sII',b'M476CTX1',7993,4628)
contexts=np.memmap(context_path,dtype=ctype,mode='r',offset=16,shape=(7993,))
dtype=np.dtype([('qid','<u8'),('context_row','<u4'),('book','<u2'),('case','u1'),('mode','u1'),('expert','<u2'),('flags','<u2'),
                ('position','<u4'),('argmax_source','<u4'),('argmax_candidate','<u4'),('label','<u4'),('source_ties','<u4'),('candidate_ties','<u4'),('reserved','<u4')]
               +[(name,'<f8') for name in FIELDS]+[('head_codes','<i2',(768,)),('head_alpha','<f4'),('candidate_sha','V32')]);assert dtype.itemsize==1748
with (SOURCE_OUT/'observables.bin').open('rb') as f:assert f.read(16)==struct.pack('<8sII',b'M477OBS1',6649,1748)
assert (SOURCE_OUT/'observables.bin').stat().st_size==11622468
obs=np.memmap(SOURCE_OUT/'observables.bin',dtype=dtype,mode='r',offset=16,shape=(6649,))
qpath=ROOT/'results/native_expert_scaling/meth472_switch_private_input_probe/query_inputs.npy'
queries=np.load(qpath,mmap_mode='r',allow_pickle=False);reference=np.load(qpath.with_name('reference_functions.npy'),mmap_mode='r',allow_pickle=False);candidate=np.load(qpath.with_name('candidate_functions.npy'),mmap_mode='r',allow_pickle=False)
assert queries.shape==(19962,) and queries.dtype.itemsize==4732 and reference.shape==candidate.shape==(19962,768)
dev_codes={e:set() for e in range(128)}
for qi in np.flatnonzero(queries['role']==0):dev_codes[int(queries['expert'][qi])].add(queries['code_sha'][qi:qi+1].tobytes())
ready=set(b['fixed_ready_IDs']);assert len(ready)==107
cohort=json.loads((DOC/'meth467_switch_rust_query_manifest.json').read_bytes());labels={}
for book in range(64,128):
    item=cohort['items'][book];assert item['book']==book and item['role']=='diagnostic_validation'
    ex=base64.b64decode(item['excerpt_token_ids_base64_le_i32'],validate=True);assert hashlib.sha256(ex).hexdigest()==item['excerpt_token_ids_sha256'];ids=np.frombuffer(ex,'<i4')
    for ca,case in enumerate(item['cases']):
        original=ids[case['excerpt_token_start']:case['excerpt_token_start']+32].tolist();assert original==case['original_window_ids']
        assert case['span_starts']==[3,10,17,24]
        target=[v for k,start in enumerate((3,10,17,24)) for v in (32099-k,*original[start:start+2])]+[32095,1]
        removed={start+q for start in (3,10,17,24) for q in (0,1)}
        source=[32099-(3,10,17,24).index(p) if p in (3,10,17,24) else original[p] for p in range(32) if p not in removed or p in (3,10,17,24)]+[1]
        assert target==case['target_ids'] and source==case['source_ids'] and case['decoder_ids']==[0]+target[:-1]
        assert hashlib.sha256(struct.pack('<14i',*target)).hexdigest()==case['target_ids_sha256']
        labels[(book,ca)]=np.asarray(target,dtype='<u4')
native_rows=[json.loads(line) for line in (SOURCE_OUT/'native.stdout.log').read_text().splitlines()];assert len(native_rows)==608
jobs=(SOURCE_OUT/'jobs.bin').read_bytes();assert jobs[:8]==b'M477JOB1';batches,nsource,ncandidate,qo,fo,co=struct.unpack_from('<3I3Q',jobs,8)
assert (batches,nsource,ncandidate,qo,fo,co)==(608,7993,6649,448,128,128)
job_cursor=44;cx=ox=0;head_count=0;all_val=[];errors=np.empty((6649,16),'<f8');rescale=np.float32(1/math.sqrt(768))
for batch,item in enumerate(b['last_layer_batches']):
    t,s,kind=item['t'],item['s'],item['kind'];ctx=contexts[cx:cx+t]
    fields=struct.unpack_from('<6I3Q',jobs,job_cursor);job_cursor+=48
    assert fields==(kind,item['book'],item['case'],item['mode'],s,t,item['query_start'],cx,item['ledger_base'])
    for key in ('whole_path','control_path'):
        size=struct.unpack_from('<I',jobs,job_cursor)[0];job_cursor+=4;assert jobs[job_cursor:job_cursor+size].decode()==item[key];job_cursor+=size
    job_labels=np.empty(t,'<u4');job_flags=np.empty(t,'<u4')
    for pos in range(t):job_labels[pos],job_flags[pos]=struct.unpack_from('<2I',jobs,job_cursor);job_cursor+=8
    assert ctx['batch'].tolist()==[batch]*t and ctx['position'].tolist()==list(range(t))
    whole=Path(item['whole_path']).read_bytes();nr=6*(s+t);assert whole[:36]==struct.pack('<8s7I',b'SWR32O01',s,t,768,12,12,32128,nr)
    de_offset=36+4*14*s*768;lo_offset=de_offset+4*t*14*768;rr_offset=lo_offset+4*t*32128;assert len(whole)==rr_offset+nr*12
    snapshots=np.frombuffer(whole,'<f4',t*14*768,de_offset).reshape(t,14,768);l0=np.frombuffer(whole,'<f4',t*32128,lo_offset).reshape(t,32128)
    rr=np.frombuffer(whole,np.dtype([('expert','<i4'),('accepted','<i4'),('p','<f4')]),nr,rr_offset);ix=179+6*np.arange(t);prob=rr['p'][ix]
    if kind:
        qids=np.arange(item['query_start'],item['query_start']+t);qr=queries[qids];orows=obs[ox:ox+t]
        assert ctx['qid'].tolist()==qids.tolist() and orows['qid'].tolist()==qids.tolist() and orows['context_row'].tolist()==list(range(cx,cx+t))
        assert np.all(qr['role']==1) and qr['book'].tolist()==[item['book']]*t and qr['case'].tolist()==[item['case']]*t and qr['mode'].tolist()==[item['mode']]*t
        assert orows['book'].tolist()==qr['book'].tolist() and orows['case'].tolist()==qr['case'].tolist() and orows['mode'].tolist()==qr['mode'].tolist() and orows['expert'].tolist()==qr['expert'].tolist() and orows['position'].tolist()==list(range(t))
        assert orows['reserved'].tolist()==[0]*t and qr['expert'].tolist()==rr['expert'][ix].tolist() and qr['index'].tolist()==ix.tolist() and qr['ledger_record'].tolist()==(item['ledger_base']+ix).tolist()
        exact(prob,qr['probability']);exact(norm(ctx['pre'],wi_norm,epsilon),qr['input']);down=reference[qids]
        flags=np.array([int(int(row['expert']) in ready)|(2 if qr['code_sha'][k:k+1].tobytes() not in dev_codes[int(row['expert'])] else 0)|(16 if row['expert'] in (25,37) else 0) for k,row in enumerate(qr)],'<u4')
        if item['mode']==0:flags|=4;flags[[1,2,4,5,7,8,10,11]]|=8;targets=labels[(item['book'],item['case'])]
        else:targets=np.full(t,2**32-1,'<u4')
        assert flags.tolist()==job_flags.tolist() and targets.tolist()==job_labels.tolist() and targets.tolist()==orows['label'].tolist()
        all_val.extend(qids.tolist())
    else:
        gd=Path(item['control_path']).read_bytes();assert len(gd)==32+t*52264 and gd[:32]==struct.pack('<8s6I',b'SWFUN001',768,3072,32128,128,11,1)
        def golden(off,typ,width):return np.stack([np.frombuffer(gd,typ,width,32+k*52264+off) for k in range(t)])
        exact(ctx['pre'],golden(16,'<f4',768));down=golden(38424,'<f4',768)
        assert np.all(ctx['qid']==np.uint64(2**64-1)) and np.all(job_flags==0) and np.all(job_labels==np.uint32(2**32-1))
    post=(ctx['pre']+prob[:,None]*down).astype('<f4');exact(post,snapshots[:,12,:]);final=norm(post,final_norm,epsilon);exact(final,snapshots[:,13,:])
    source_input=(final*rescale).astype('<f4');qc,alpha=quant(source_input);exact(qc,ctx['head_codes']);exact(alpha,ctx['head_alpha']);actual_source=scaled(qc,hw,hs,alpha);exact(actual_source,l0);head_count+=t*32128
    if kind:
        cf=candidate[qids];fallback=np.flatnonzero((flags&1)==0)
        exact(cf[fallback],down[fallback])
        cpost=(ctx['pre']+prob[:,None]*cf).astype('<f4');ci=(norm(cpost,final_norm,epsilon)*rescale).astype('<f4');cq,ca=quant(ci)
        exact(cq,orows['head_codes']);exact(ca,orows['head_alpha']);l1=scaled(cq,hw,hs,ca);head_count+=t*32128
        for pos in range(t):assert hashlib.sha256(l1[pos].tobytes()).digest()==orows['candidate_sha'][pos:pos+1].tobytes()
        exact(l1[fallback],l0[fallback])
        s64,c64=l0.astype('<f8'),l1.astype('<f8');arg0=np.argmax(l0,axis=1);arg1=np.argmax(l1,axis=1);ms=np.max(s64,axis=1);mc=np.max(c64,axis=1)
        es=np.exp(s64-ms[:,None]);ec=np.exp(c64-mc[:,None]);zs=es.sum(axis=1);zc=ec.sum(axis=1);p=es/zs[:,None];q=ec/zc[:,None]
        delta=c64-s64;lo=delta.min(axis=1);hi=delta.max(axis=1);mu=(p*delta).sum(axis=1);center=delta-mu[:,None]
        kl=(mc-ms)+np.log(zc)-np.log(zs)-mu;small=(hi-lo)<=.5
        cm=(p[small]*center[small]).sum(axis=1);curv=(p[small]*(np.expm1(center[small])-center[small])).sum(axis=1);kl[small]=np.log1p(cm+curv)-cm
        tv=.5*np.abs(p-q).sum(axis=1);fisher=.5*(p*center*center).sum(axis=1);margin0=ms-np.partition(s64,-2,axis=1)[:,-2];margin1=mc-np.partition(c64,-2,axis=1)[:,-2]
        ties0=np.count_nonzero(l0==ms[:,None],axis=1);ties1=np.count_nonzero(l1==mc[:,None],axis=1)
        assert arg0.tolist()==orows['argmax_source'].tolist() and arg1.tolist()==orows['argmax_candidate'].tolist() and ties0.tolist()==orows['source_ties'].tolist() and ties1.tolist()==orows['candidate_ties'].tolist()
        flags|=np.where(ties0==1,32,0).astype('<u4');flags|=np.where(ties1==1,64,0).astype('<u4');flags|=np.where(arg0!=arg1,128,0).astype('<u4');cert=(ties0==1)&(margin0>hi-lo);flags|=np.where(cert,256,0).astype('<u4')
        assert np.all(arg0[cert]==arg1[cert]) and flags.tolist()==orows['flags'].tolist()
        n0=np.zeros(t,'<f8');n1=np.zeros(t,'<f8')
        if item['mode']==0:n0=ms+np.log(zs)-s64[np.arange(t),targets];n1=mc+np.log(zc)-c64[np.arange(t),targets]
        energy=np.sum((cf.astype('<f8')-down.astype('<f8'))**2,axis=1);ref_energy=np.sum(down.astype('<f8')**2,axis=1)
        expected=np.column_stack((kl,tv,margin0,margin1,lo,hi,1/zs,1/zc,fisher,n0,n1,n1-n0,energy,ref_energy,prob.astype('<f8'),(hi-lo)**2/8))
        actual=np.column_stack([orows[name] for name in FIELDS]);diff=np.abs(actual-expected);assert np.isfinite(expected).all() and np.all(diff<=1e-11+1e-10*np.abs(expected))
        assert np.all(actual[:,0]>=-1e-11) and np.all(actual[:,0]<=actual[:,15]+1e-11+1e-10*np.abs(actual[:,15]))
        errors[ox:ox+t]=diff;ox+=t
    cx+=t
    assert native_rows[batch]=={'batch':batch,'source_positions':cx,'candidate_positions':ox}
    guard()
    if batch%96==95:checkpoint(completed_batches=batch+1,source_positions=cx,candidate_positions=ox);print(json.dumps({'completed_batches':batch+1,'source_positions':cx,'candidate_positions':ox,'seconds':time.monotonic()-START}),flush=True)
assert job_cursor==len(jobs) and cx==7993 and ox==6649 and head_count==(7993+6649)*32128==470418176
assert all_val==np.flatnonzero(queries['role']==1).tolist() and len(set(all_val))==6649
assert np.count_nonzero(obs['flags']&4)==3584 and np.count_nonzero(obs['flags']&8)==2048
gates['ALL7993_source_vectors_norm_codes_logits_BYTE_exact']=True
gates['ALL6649_candidate_A16_codes_scales_BYTE_exact_full32128_logits_SHA_exact']=True
gates['ALL6649_true_labels_ready_novel_flags_margins_F64_metrics_independent_tolerance']=True
gates['complete608_batches_jobs_observations_exact_EOF_correct470418176_cardinality']=True
np.save(OUT/'metric_rederivation_absolute_error.npy',errors,allow_pickle=False)
def summ(indices):
    v=obs[indices];n=len(v)
    if not n:return {'count':0}
    r={'count':n,'argmax_changed':int(np.count_nonzero(v['flags']&128)),'unique_source':int(np.count_nonzero(v['flags']&32)),'margin_certified':int(np.count_nonzero(v['flags']&256)),'negative_raw_KL':int(np.count_nonzero(v['kl']<0))}
    for name in ('kl','tv','margin_source','delta_range','fisher_half'):
        values=v['delta_max']-v['delta_min'] if name=='delta_range' else np.maximum(v['kl'],0) if name=='kl' else v[name]
        percent=np.quantile(values,[.5,.95,.99],method='linear');r[name]={'mean':float(values.mean()),'p50':float(percent[0]),'p95':float(percent[1]),'p99':float(percent[2]),'max':float(values.max())}
    for name,bit in (('teacher_all_label',4),('teacher_masked_content_label',8)):
        a=v[(v['flags']&bit)!=0]
        if len(a):r[name]={'count':len(a),'source_mean_NLL':float(a['nll_source'].mean()),'candidate_mean_NLL':float(a['nll_candidate'].mean()),'mean_delta_NLL':float(a['delta_nll'].mean()),'source_label_argmax_correct':int(np.count_nonzero(a['argmax_source']==a['label'])),'candidate_label_argmax_correct':int(np.count_nonzero(a['argmax_candidate']==a['label']))}
    den=float(np.sum(v['reference_energy']));num=float(np.sum(v['error_energy']));r['pooled_FFN_RMS']=math.sqrt(num/den) if den>0 else None;r['reference_energy']=den;return r
selectors={'all':np.arange(6649),'teacher':np.flatnonzero(obs['mode']==0),'natural':np.flatnonzero(obs['mode']==1)}
for yes,no,bit in (('ready','fallback',1),('novel_code','seen_code',2),('dominant25_37','other_IDs',16)):
    for flag,name in ((True,yes),(False,no)):
        selection=((obs['flags']&bit)!=0)==flag;selectors[name]=np.flatnonzero(selection)
        for mode,label in ((0,'teacher'),(1,'natural')):selectors[name+'_'+label]=np.flatnonzero(selection&(obs['mode']==mode))
summaries={'views':{k:summ(v) for k,v in selectors.items()},'books':{str(book):summ(np.flatnonzero(obs['book']==book)) for book in range(64,128)},'experts':{str(e):{name:summ(np.flatnonzero((obs['expert']==e)&(np.ones(6649,dtype=bool) if mode is None else obs['mode']==mode))) for mode,name in ((None,'all'),(0,'teacher'),(1,'natural'))} for e in range(128)}}
assert summaries['views']['teacher']['count']==3584 and summaries['views']['natural']['count']==3065 and sum(v['count'] for v in summaries['books'].values())==6649 and sum(v['all']['count'] for v in summaries['experts'].values())==6649
gates['all_uniformquery_book128ID_mode_ready_fallback_novel_dominant_reporting_denominators']=True
for rel,r in b['preserved_unrelated_files'].items():assert digest(ROOT/rel)==r['sha256']
assert subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).splitlines()==b['tracked_status']
engine=ROOT/'benchmarks/phase60/engine.c';head(engine);assert digest(engine)==b['original_engine']['sha256']
assert not ({'torch','transformers','tensorflow','scipy','sklearn','pandas','pyarrow','tokenizers','meth477_observable_math','meth477_switch_head_observable'}&set(sys.modules))
gates['original_exit2_missing_endworker_readback_preserved_no_native_reexecution_resources_and_scope']=True
phase='report_retention'
report={'experiment':'METH477-R1 independent complete saved-head witness recovery','source_binding_sha256':BIND_SHA,'first_failure_sha256':digest(FAIL),'first_failure_retention_sha256':b['records']['RETENTION_477_FIRST_FAILURE_20261005.json']['sha256'],
        'source477_parent_exit':1,'source477_native_exit':2,'source477_end_worker_readback':'unavailable; unchanged source failure before terminal worker_json',
        'correct_head_cardinality':470418176,'original_wrong_head_cardinality':470422176,'native_commands':0,'whole_model_commands':0,'cached_head_vectors_rederived':14642,
        'start_utc':datetime.fromtimestamp(parent.create_time()).astimezone().isoformat(),'end_utc':datetime.now().astimezone().isoformat(),'process_instance':{'pid':os.getpid(),'create_time_unix':parent.create_time(),'executable':sys.executable},
        'scientific_sources':{str(Path(__file__).resolve()):digest(__file__),str(PROTO):digest(PROTO)},'gates':gates,'summaries':summaries,
        'metric_absolute_difference_max':{name:float(errors[:,k].max()) for k,name in enumerate(FIELDS)},
        'decision':'COMPLETE_HEAD_OBSERVABLE_WITNESSES_RECOVERED_NO_PROMOTION_pending_scalar_reporting_retention_audit',
        'scope':'Descriptive consumed-source-prefix calibration only; original477 still failed, old472 still localFAIL; no fitting/cohort narrowing/native replay/changed own-state/global quality/speed/DRAM/useful-n/multiple-family claim',
        'preserved_daemons':daemons,'resource':{'seconds_before_raw':time.monotonic()-START,'OS_peak_bytes':peak,'file_bytes_hashed':hashed,'hard_seconds':600,'hard_peak_bytes':2<<30,'hard_new_bytes':8<<20},
        'output_inventory':[{'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='progress.jsonl']}
with RAW.open('xb') as f:f.write((json.dumps(report,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
guard();raw_sha=digest(RAW);checkpoint(terminal=True,raw_sha256=raw_sha,OS_peak_bytes=peak)
print(json.dumps({'raw':str(RAW),'raw_sha256':raw_sha,'gates':gates,'all':summaries['views']['all'],'natural':summaries['views']['natural'],'seconds':time.monotonic()-START,'OS_peak_bytes':peak}),flush=True)
progress.close();faulthandler.disable();fatal.close()
