"""One frozen output-tile16 permutation/primal/cost experiment, no format sweep."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
OUT=ROOT/'results/native_expert_scaling/meth456_switch_output_tile_cost'
CPU=Path(__file__).with_name('meth456_switch_output_tile_cpu.c');PROTOCOL=DOC/'METH_456_SWITCH_OUTPUT_TILE_COST_PROTOCOL_20261005.md'
PARENT_SHA='049e8ea6ee63f0b2efa6204e37aaaf70842bf0185c1c814720e77fba8a2d07a5'
RETENTION_SHA='314b79a61a94b9de308b12d979a5fc520014ad856a6c6198a85f4a1f7f92e056'
REPAIR_SHA='dcbf35cdb2d326872401e2663920c26bcc2fbaea56bdddc0b999b2e94fee1f52'
STRIDE=3689472;HEADER=4096;N=336
ARMS=('original_I8','closed454_direct_diagnostic','tiled16_I4_sparse_I8')
WIRE=np.dtype([('query','<u4'),('arm','<u4'),('wrong','<u4'),('id','<u4'),('nonzero','<u4'),('wi_scale','<f4'),('wo_scale','<f4'),
    ('basis','<f4',(768,)),('raw','<f4',(3072,)),('down','<f4',(768,)),('wi_codes','<i2',(768,)),('wo_codes','<i2',(3072,))])
assert WIRE.itemsize==26140

def committed(path):
    p=Path(path);rel=p.resolve().relative_to(ROOT).as_posix()
    assert p.read_bytes()==subprocess.check_output(['git','-c','core.autocrlf=false','cat-file','--filters',f'--path={rel}',f'HEAD:{rel}'],cwd=ROOT),('physical_HEAD',rel)
def write(path,value):
    Path(path).write_bytes((json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').replace('\n','\r\n').encode('utf-8'))
def jobs(result):
    own={os.getpid(),*(p.pid for p in psutil.Process().parents())}
    for p in psutil.process_iter(['name','cmdline']):
        if p.pid in own:continue
        name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
        if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():result.setdefault('preserved_daemons',[]).append(p.pid);continue
        assert not(name.startswith('python') or ('meth' in name and name.endswith('.exe'))),('concurrent_job',p.pid,name)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();numeric=None;peak=hashed=0;stage='fresh_bindings'
    result={'experiment':'METH456-exact-output-tile16-WI-permutation-and-native-cost','native_commands':{},'helper_sha256':{}}
    def guard():
        nonlocal peak
        mi=psutil.Process().memory_info();peak=max(peak,mi.rss,getattr(mi,'peak_wset',0));elapsed=time.monotonic()-start
        assert peak<=512<<20 and elapsed<=900 and ((numeric is None and elapsed<=300) or (numeric is not None and time.monotonic()-numeric<=600)),'parent512MiB_admission300_numeric600_total900'
        assert not OUT.exists() or sum(p.stat().st_size for p in OUT.glob('*') if p.is_file())<=600<<20,'outputs600MiB'
    def sha(path):
        nonlocal hashed
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            while b:=f.read(4<<20):h.update(b);hashed+=len(b);guard()
        return h.hexdigest()
    try:
        result['git_head_before_execution']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        assert (ROOT/'.gitattributes').read_bytes()==subprocess.check_output(['git','show','HEAD:.gitattributes'],cwd=ROOT)
        for p in (Path(__file__),CPU,PROTOCOL):committed(p);result['helper_sha256'][str(p)]=sha(p)
        p=DOC/'meth455_switch_component_cost_result.json';committed(p);assert sha(p)==PARENT_SHA;parent=json.loads(p.read_bytes())
        assert all(parent['apparatus_gates'].values()) and all(all(v.values()) for v in parent['diagnostic_admissibility_gates'].values()) and parent['decision']=='prepare_ONE_NEW_exact_output_tiled_WI_format'
        for name,expected in [('RETENTION_455_20261005.json',RETENTION_SHA),('METH_455_COMMAND_PROVENANCE_REPAIR_1_20261005.json',REPAIR_SHA)]:committed(DOC/name);assert sha(DOC/name)==expected
        repair=json.loads((DOC/'METH_455_COMMAND_PROVENANCE_REPAIR_1_20261005.json').read_bytes());assert all(repair['gates'].values()) and repair['unchanged_raw']['sha256']==PARENT_SHA
        for path,expected in parent['helper_sha256'].items():committed(path);assert sha(path)==expected;result['helper_sha256'][path]=expected
        for name,expected in parent['retained_record_sha256'].items():committed(DOC/name);assert sha(DOC/name)==expected
        for item in parent['output_inventory']:
            p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
        reused={}
        for item in parent['reused_output_inventory']:
            p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];reused[p.name]=p
        a=parent['artifacts']['128'];assert Path(a['payload']).stat().st_size==a['bytes'] and sha(a['payload'])==a['sha256'] and sha(a['manifest'])==a['manifest_sha256']
        engine=ROOT/'benchmarks/phase60/engine.c';committed(engine);assert sha(engine)==parent['preserved_engine_sha256']
        for path,expected in parent['preserved_original_binary_sha256'].items():assert sha(path)==expected
        for path,expected in parent['preserved_numerical_runtime_sha256'].items():assert sha(path)==expected
        compiler=Path(parent['compile']['argv'][0]);assert sha(compiler)==parent['compile']['compiler_sha256'];dll=reused['libomp.dll'];assert sha(dll)==parent['compile']['runtime_sha256']
        result.update(retained_record_sha256=parent['retained_record_sha256']|{'meth455_switch_component_cost_result.json':PARENT_SHA,'RETENTION_455_20261005.json':RETENTION_SHA,'METH_455_COMMAND_PROVENANCE_REPAIR_1_20261005.json':REPAIR_SHA},
            artifacts=parent['artifacts'],preserved_engine_sha256=parent['preserved_engine_sha256'],preserved_original_binary_sha256=parent['preserved_original_binary_sha256'],
            preserved_numerical_runtime_sha256=parent['preserved_numerical_runtime_sha256'],reused_output_inventory=parent['reused_output_inventory'],diagnosis_output_inventory=parent['output_inventory'])
        jobs(result);psutil.Process().cpu_affinity([0]);assert np.__version__=='2.4.6' and psutil.disk_usage(str(ROOT)).free>=2<<30
        initial={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in (Path(a['payload']),Path(a['manifest']),reused['bank.bin'],reused['trace.bin'],reused['primal.bin'])}
        numeric=time.monotonic();OUT.mkdir(parents=True);result['admission']={'seconds':numeric-start,'bytes_hashed':hashed};print(json.dumps({'admission':result['admission']}),flush=True)
        stage='one_exact_streaming_permutation';bank=OUT/'bank.bin'
        with reused['bank.bin'].open('rb') as source,bank.open('wb') as target:
            h=source.read(HEADER);assert h[:8]==b'MC454B01' and struct.unpack_from('<4I',h,8)==(768,3072,128,4096)
            signs=source.read(768);assert len(signs)==768;new=bytearray(HEADER);new[:8]=b'MC456B01';struct.pack_into('<5I',new,8,768,3072,128,4096,16);target.write(new);target.write(signs)
            for expert in range(128):
                data=source.read(STRIDE);assert len(data)==STRIDE
                wi=np.frombuffer(data,dtype='u1',count=1179648).reshape(192,16,12,32)
                scales=np.frombuffer(data,dtype='<f4',count=36864,offset=1179648).reshape(192,16,12)
                target.write(wi.transpose(0,2,3,1).tobytes());target.write(scales.transpose(0,2,1).tobytes());target.write(data[1327104:]);guard()
            assert source.read(1)==b''
        assert bank.stat().st_size==472257280
        with reused['bank.bin'].open('rb') as source,bank.open('rb') as target:
            source.seek(HEADER);target.seek(HEADER);assert source.read(768)==target.read(768)==signs
            for expert in range(128):
                old=source.read(STRIDE);new=target.read(STRIDE);assert len(old)==len(new)==STRIDE
                wi=np.frombuffer(new,dtype='u1',count=1179648).reshape(192,12,32,16)
                scales=np.frombuffer(new,dtype='<f4',count=36864,offset=1179648).reshape(192,12,16)
                assert wi.transpose(0,3,1,2).tobytes()==old[:1179648] and scales.transpose(0,2,1).tobytes()==old[1179648:1327104] and new[1327104:]==old[1327104:]
                assert all((HEADER+768+expert*STRIDE+offset)%64==0 for offset in (0,1179648,1327104,3686400));guard()
            assert source.read(1)==target.read(1)==b''
        result['permutation']={'all128_inverse_bytes_exact':True,'WI_packed_shape':[192,12,32,16],'WI_scale_shape':[192,12,16],'signs_WO_codes_WO_scales_byte_identical':True,
            'mapped_bytes':bank.stat().st_size,'nominal_bytes_without_header':bank.stat().st_size-HEADER,'original_I8_nominal_bank_bytes':605945856,'stored_nominal_ratio':(bank.stat().st_size-HEADER)/605945856,'new_coefficient_roundings':0}
        env=os.environ.copy();env.update(parent['runtime']['native_environment']);env['PATH']=str(compiler.parent)+os.pathsep+env.get('PATH','')
        result['runtime']={'numpy':np.__version__,'controller_affinity':[0],'native_environment':parent['runtime']['native_environment'],'GPU':False,'BLAS_arithmetic_used':False}
        def run(label,argv,timeout):
            actual=list(argv);result['native_commands'][label]={'argv':list(actual),'timeout_seconds':timeout};before=time.monotonic()
            try:q=subprocess.run(actual,cwd=ROOT,env=env,capture_output=True,timeout=min(timeout,max(1,600-(time.monotonic()-numeric))))
            except subprocess.TimeoutExpired as error:
                (OUT/(label+'.stdout.log')).write_bytes(error.stdout or b'');(OUT/(label+'.stderr.log')).write_bytes(error.stderr or b'');result['native_commands'][label]['terminal']='subprocess_timeout_killed_and_waited';raise
            (OUT/(label+'.stdout.log')).write_bytes(q.stdout);(OUT/(label+'.stderr.log')).write_bytes(q.stderr);result['native_commands'][label].update(exit_code=q.returncode,seconds=time.monotonic()-before);guard();return q
        stage='compile_frozen456';binary=OUT/'meth456_switch_output_tile_cpu.exe';argv=[str(compiler),'-O3','-std=c11','-march=x86-64-v3','-fno-fast-math','-ffp-contract=off','-fopenmp',str(CPU),'-lpsapi','-o',str(binary)]
        q=run('compile',argv,120);assert q.returncode==0,q.stderr.decode(errors='replace');shutil.copyfile(dll,OUT/'libomp.dll')
        result['compile']={'argv':list(argv),'compiler_sha256':sha(compiler),'runtime_sha256':sha(OUT/'libomp.dll'),'binary_sha256':sha(binary)}
        stage='negative_reserved_nibble';q=run('negative_nibble',[str(binary),'--bad-nibble'],30);assert q.returncode==2 and b'tiled_minus8_nibble' in q.stderr
        stage='C_primal_before_timing';primal=OUT/'primal.bin';base=[str(binary),a['manifest'],str(reused['bank.bin']),str(bank),str(reused['trace.bin']),str(reused['primal.bin'])]
        q=run('primal',[base[0],'--qualify',*base[1:],str(primal)],240);assert q.returncode==0,q.stderr.decode(errors='replace');result['native_primal']=json.loads(q.stdout);assert result['native_primal']['tiny_qualified']
        with primal.open('rb') as f:h=f.read(24)
        assert h[:8]==b'MC456Q01' and struct.unpack_from('<4I',h,8)==(336,768,3072,1680) and primal.stat().st_size==43915224
        gold=np.memmap(reused['primal.bin'],dtype=WIRE,mode='r',offset=24,shape=(1680,));new=np.memmap(primal,dtype=WIRE,mode='r',offset=24,shape=(1680,))
        for i in range(1680):assert new[i].tobytes()==gold[i].tobytes(),('all1680_record_fields_BYTE_exact454',i)
        del new,gold;guard();result['C_primal']={'original_records':336,'closed_direct_diagnostic_records':672,'NEW_tiled_correct_and_wrong_ID_records':672,'all1680_basis_raw_down_codes_scales_IDs_nonzero_BYTE_exact454':True,'negative_reserved_nibble_exit2':True}
        print(json.dumps({'C_primal':result['C_primal'],'native_seconds':result['native_primal']['native_seconds']}),flush=True)
        stage='C_whole_selected_FFN_cost';timings=OUT/'timings.jsonl';q=run('timing',[base[0],'--bench',*base[1:],str(timings)],300);assert q.returncode==0,q.stderr.decode(errors='replace');result['native_timing']=json.loads(q.stdout)
        rows=[json.loads(line) for line in timings.read_text(encoding='utf-8').splitlines()];expected=[(rep,((rep+2)%3+slot)%3,i) for rep in range(-2,5) for slot in range(3) for i in range(336)]
        assert len(rows)==7056 and [(r['repetition'],r['arm'],r['query']) for r in rows]==expected
        books=np.array(parent['data']['books']);ids=parent['data']['selected_ids'];values=np.empty((5,3,336))
        for r in rows:
            rep,arm,i=(r[k] for k in ('repetition','arm','query'));assert r['id']==ids[i] and r['book']==int(books[i]) and np.isfinite(r['seconds']) and r['seconds']>0
            if rep>=0:values[rep,arm,i]=r['seconds']
        metrics={}
        for arm,label in enumerate(ARMS):
            v=values[:,arm,:];metrics[label]={'mean_seconds':float(np.mean(v)),'median_seconds':float(np.median(v)),'p95_seconds':float(np.quantile(v,.95)),'max_seconds':float(np.max(v)),
                'per_sweep_seconds':np.sum(v,axis=1).tolist(),'per_book_mean_seconds':{str(b):float(np.mean(v[:,books==b])) for b in range(18,24)}}
        original=metrics[ARMS[0]];tiled=metrics[ARMS[2]];tiled['mean_ratio_to_original']=tiled['mean_seconds']/original['mean_seconds'];tiled['p95_ratio_to_original']=tiled['p95_seconds']/original['p95_seconds']
        tiled['per_book_mean_ratio_to_original']={b:tiled['per_book_mean_seconds'][b]/v for b,v in original['per_book_mean_seconds'].items()}
        gates={'mean_cost_ratio_le0_80':tiled['mean_ratio_to_original']<=.80,'every_book_mean_ratio_le1_00':all(v<=1. for v in tiled['per_book_mean_ratio_to_original'].values()),
            'p95_cost_ratio_le1_00':tiled['p95_ratio_to_original']<=1.,'nominal_stored_bank_ratio_le0_80':result['permutation']['stored_nominal_ratio']<=.80}
        result['cost']=metrics;result['feasibility_gates']=gates
        result['apparatus_gates']={'fresh455_455R1_454_retention_helpers_complete_outputs_original_payload_runtime_compiler':True,'all128_tiled_inverse_codes_scales_signs_WO_bytes_exact':True,
            'tiny2025_pairs_all16_lanes_mixed_scalar_I64_prior454_suite_qualified':True,'reserved_minus8_native_fault_exit2':True,'all1680_C_record_fields_BYTE_exact454':True,
            'all7056_timed_outputs_BYTE_exact454_and_fixed_trace_order':True,'one_physical_worker_mask1_in_both_native_children':all(len(n['worker_affinity'])==1 and n['worker_affinity'][0]['actual_mask']==1 for n in (result['native_primal'],result['native_timing'])),
            'distinct_copied_primal_and_timing_command_records':result['native_commands']['primal']['argv'][1]=='--qualify' and result['native_commands']['timing']['argv'][1]=='--bench'}
        result['candidate_PASS']=all(gates.values()) and all(result['apparatus_gates'].values())
        for path,state in initial.items():assert state==(Path(path).stat().st_size,Path(path).stat().st_mtime_ns)
        nativepeak=max(n['native_peak_bytes'] for n in (result['native_primal'],result['native_timing']));assert peak+nativepeak<=(5<<29),'conservative_total2_5GiB'
        result['data']={k:parent['data'][k] for k in ('keys','books','selected_ids','consumed_positions','distinct_selected_IDs')}|{'arms':ARMS,'warm_sweeps_per_arm':2,'measured_sweeps_per_arm':5,'timed_records':7056,'whole336_trace_per_arm_before_next_arm':True}
        result['output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.glob('*')) if p.is_file()]
        result['resource']={'seconds':time.monotonic()-start,'admission_seconds':numeric-start,'numeric_seconds':time.monotonic()-numeric,'parent_peak_bytes':peak,'native_peak_bytes':nativepeak,
            'conservative_aggregate_peak_bytes':peak+nativepeak,'file_bytes_hashed':hashed,'output_bytes':sum(p['bytes'] for p in result['output_inventory']),'GPU':False,'optimizer_updates':0,'new_coefficient_roundings':0}
        result['decision']='tiled16_primal_cost_PASS_prepare_NEW_all_bank_geometry_composition' if result['candidate_PASS'] else 'tiled16_fixed_recipe_FAIL_close_without_tile_or_format_sweep_reassess_transfer_economics'
        result['scope']='One real128 bank11, consumed fixed original normalized inputs/IDs. Full signed basis/A16/WI/ReLU/WO scan/indices/columns/scales/output charged, unchanged original mv baseline, one worker. Exact physical permutation of existing453 coefficients; no new local prediction gain. No whole-model f/rate/fresh quality/DRAM/router useful-n or cross-family100B evidence. Closed454 direct arm diagnostic only.'
        jobs(result);guard();write(args.out,result);print(json.dumps({'apparatus':result['apparatus_gates'],'feasibility':gates,'cost':metrics,'decision':result['decision'],'resource':result['resource'],'sha256':sha(args.out)}),flush=True)
    except BaseException as error:
        result.update(failure_stage=stage,error=repr(error),seconds=time.monotonic()-start,parent_peak_bytes=peak,file_bytes_hashed=hashed)
        if OUT.exists():
            partial=[]
            for p in sorted(OUT.glob('*')):
                if p.is_file():
                    with p.open('rb') as f:partial.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.file_digest(f,'sha256').hexdigest()})
            result['partial_output_inventory']=partial
        write(args.out.with_suffix('.failure.json'),result);raise

if __name__=='__main__':main()
