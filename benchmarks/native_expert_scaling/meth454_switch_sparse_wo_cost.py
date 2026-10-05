"""One frozen C primal/cost screen; actual trace sweeps, no whole-rate claim."""
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
import torch
from threadpoolctl import threadpool_info,threadpool_limits
import meth453_switch_sparse_wo_pilot as P

M,R,H=P.M,P.R,P.H
CPU=Path(__file__).with_name('meth454_switch_sparse_wo_cpu.c')
PROTOCOL=M.DOC/'METH_454_SWITCH_SPARSE_WO_COST_PROTOCOL_20261005.md'
OUT=M.ROOT/'results/native_expert_scaling/meth454_switch_sparse_wo_cost'
PARENT_SHA='63d0fca69ad180b5024e1152b3d2fd9e13699fcfc3efd662850e011387ba3bbf'
SOURCE374_SHA='4601be80b787341f6d60e01f7219c1cd4cd71c0451f835d25e31ded3c468f4b1'
RETENTION_SHA='75c7ee7e36d7632d40c8f384308efec79e0ca833f7954c971dafddec04603d93'
FIELDS=(('wi_packed',1179648,'u1',(3072,384)),('wi_scales',147456,'<f4',(3072,12)),
        ('wo_columns',2359296,'i1',(3072,768)),('wo_scales',3072,'<f4',(768,)))
WIRE=np.dtype([('query','<u4'),('arm','<u4'),('wrong','<u4'),('id','<u4'),('nonzero','<u4'),
    ('wi_scale','<f4'),('wo_scale','<f4'),('basis','<f4',(768,)),('raw','<f4',(3072,)),
    ('down','<f4',(768,)),('wi_codes','<i2',(768,)),('wo_codes','<i2',(3072,))])
assert WIRE.itemsize==26140


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();numeric=None;peak=hashed=0;stage='bindings';result={'experiment':'METH454-C-direct-and-pair-LUT-WI-exact-column-WO-cost','native_commands':{}}
    def guard():
        nonlocal peak
        mi=psutil.Process().memory_info();peak=max(peak,mi.rss,getattr(mi,'peak_wset',0));elapsed=time.monotonic()-start
        assert peak<=3<<30 and elapsed<=900 and ((numeric is None and elapsed<=300) or (numeric is not None and time.monotonic()-numeric<=600)),'parent3GiB_admission300_numeric600_total900'
        assert not OUT.exists() or sum(p.stat().st_size for p in OUT.glob('*') if p.is_file())<=600<<20,'outputs600MiB'
    def sha(path):
        nonlocal hashed
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            while b:=f.read(4<<20):h.update(b);hashed+=len(b);guard()
        return h.hexdigest()
    try:
        result['helper_sha256']={}
        for p in (Path(__file__),CPU,PROTOCOL):M.committed(p);result['helper_sha256'][str(p)]=sha(p)
        p=M.DOC/'meth453_switch_sparse_wo_result.json';M.committed(p);assert sha(p)==PARENT_SHA;parent=json.loads(p.read_bytes());assert all(parent['apparatus_gates'].values()) and all(parent['feasibility_gates'].values())
        p=M.DOC/'RETENTION_453_20261005.json';M.committed(p);assert sha(p)==RETENTION_SHA
        for path,expected in parent['helper_sha256'].items():M.committed(path);assert sha(path)==expected;result['helper_sha256'][path]=expected
        files={}
        for item in parent['output_inventory']:assert sha(item['path'])==item['sha256'];files[Path(item['path']).name]=Path(item['path'])
        p=M.DOC/'meth374_switch_physical_workers_contract_result.json';M.committed(p);assert sha(p)==SOURCE374_SHA;original374=json.loads(p.read_bytes())
        assert all(original374['gates'].values()) and original374['model_and_driver_math_exact356_by_source_reversal']
        for name,expected in original374['source_sha256'].items():
            p=CPU.parent/name;M.committed(p);assert sha(p)==expected;result['helper_sha256'][str(p)]=expected
        compiler=Path(original374['compile']['argv'][0]);assert sha(compiler)==original374['compile']['compiler_sha256'];dll=Path(original374['compile']['argv'][-1]).parent/'libomp.dll';assert sha(dll)==original374['compile']['runtime_sha256']
        engine=M.ROOT/'benchmarks/phase60/engine.c';M.committed(engine);assert sha(engine)==parent['preserved_engine_sha256'];result['preserved_engine_sha256']=parent['preserved_engine_sha256']
        for path,expected in parent['preserved_original_binary_sha256'].items():assert sha(path)==expected
        result['preserved_original_binary_sha256']=parent['preserved_original_binary_sha256']
        own={os.getpid(),*(p.pid for p in psutil.Process().parents())}
        for proc in psutil.process_iter(['name','cmdline']):
            if proc.pid in own:continue
            name=(proc.info['name'] or '').lower();argv=proc.info['cmdline'] or []
            if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():result.setdefault('preserved_daemons',[]).append(proc.pid);continue
            assert not(name.startswith('python') or ('meth' in name and name.endswith('.exe'))),('concurrent_job',proc.pid,name)
        psutil.Process().cpu_affinity([0]);torch.set_num_threads(1);torch.set_num_interop_threads(1)
        assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6' and psutil.disk_usage(str(M.ROOT)).free>=2<<30
        name,expected=R.C.U.EXPORT[128];p=M.DOC/name;M.committed(p);assert sha(p)==expected;export=json.loads(p.read_bytes());a=export['artifact'];assert a==parent['artifacts']['128']
        assert sha(a['payload'])==a['sha256'] and sha(a['manifest'])==a['manifest_sha256'];R.B.read_manifest(a['manifest'],export['original_config'],export['tensors'],Path(a['payload']))
        result['artifacts']={'128':a};initial=(Path(a['payload']).stat().st_size,Path(a['payload']).stat().st_mtime_ns)
        result['retained_record_sha256']={'meth453_switch_sparse_wo_result.json':PARENT_SHA,'RETENTION_453_20261005.json':RETENTION_SHA,'meth374_switch_physical_workers_contract_result.json':SOURCE374_SHA}
        numeric=time.monotonic();OUT.mkdir(parents=True);result['admission']={'seconds':numeric-start,'bytes_hashed':hashed};print(json.dumps({'admission':result['admission']}),flush=True)
        env=os.environ.copy();env.update(original374['runtime_environment']);env['OMP_NUM_THREADS']='1';env['PATH']=str(compiler.parent)+os.pathsep+env.get('PATH','')
        def run(label,argv,timeout):
            result['native_commands'][label]={'argv':argv,'timeout_seconds':timeout};before=time.monotonic()
            try:q=subprocess.run(argv,cwd=M.ROOT,env=env,capture_output=True,timeout=min(timeout,max(1,600-(time.monotonic()-numeric))))
            except subprocess.TimeoutExpired as error:
                (OUT/(label+'.stdout.log')).write_bytes(error.stdout or b'');(OUT/(label+'.stderr.log')).write_bytes(error.stderr or b'');result['native_commands'][label]['terminal']='subprocess_timeout_killed_and_waited';raise
            (OUT/(label+'.stdout.log')).write_bytes(q.stdout);(OUT/(label+'.stderr.log')).write_bytes(q.stderr)
            result['native_commands'][label].update(exit_code=q.returncode,seconds=time.monotonic()-before);guard();return q
        with threadpool_limits(limits=1),torch.no_grad():
            result['runtime']={'torch':torch.__version__,'numpy':np.__version__,'CPU_affinity':[0],'BLAS':threadpool_info(),'GPU':False,'native_environment':original374['runtime_environment']|{'OMP_NUM_THREADS':'1'}}
            paths={p['filepath'] for p in result['runtime']['BLAS']}|{str(M.ROOT/'.venv/Lib/site-packages/torch/lib/torch_cpu.dll'),str(M.ROOT/'.venv/Lib/site-packages/torch/_C.cp312-win_amd64.pyd')}
            result['runtime']['binary_sha256']={p:sha(p) for p in sorted(paths)}
            for path,expected in parent['runtime']['binary_sha256'].items():assert result['runtime']['binary_sha256'][path]==expected
            stage='exact_wire_export';payload=OUT/'bank.bin';tracepath=OUT/'trace.bin'
            with np.load(files['bank.npz'],allow_pickle=False) as q:bank={k:q[k].copy() for k in q.files}
            with payload.open('wb') as f:
                f.write(b'MC454B01'+struct.pack('<4I',768,3072,128,4096)+bytes(4096-24));f.write(bank['wi_signs'].tobytes())
                for e in range(128):
                    for key,size,dtype,shape in FIELDS:
                        v=bank[key][e];assert v.nbytes==size and v.dtype==np.dtype(dtype) and v.shape==shape;f.write(v.tobytes())
                    guard()
            assert payload.stat().st_size==472257280;blob=np.memmap(payload,mode='r',dtype='u1');R.C.exact(blob[4096:4864].view('i1'),bank['wi_signs'])
            for e in range(128):
                off=4864+e*3689472
                for key,size,dtype,shape in FIELDS:R.C.exact(blob[off:off+size].view(dtype).reshape(shape),bank[key][e]);off+=size
            del blob,bank
            with np.load(files['original_prefixes.npz'],allow_pickle=False) as q:rows=q['rows'].copy();books=q['books'].copy();keys=q['keys'].tolist()
            assert rows.shape==(336,) and books.tolist()==parent['data']['books'] and keys==parent['data']['keys']
            td=np.dtype([('input','<f4',(768,)),('id','<u4'),('book','<i4')]);trace=np.empty(336,td);trace['input']=rows['input'];trace['id']=rows['expert'];trace['book']=books
            tracepath.write_bytes(b'MC454T01'+struct.pack('<5I',768,3072,128,336,11)+trace.tobytes());assert tracepath.stat().st_size==1034908
            basis=H.ActivationBasis(H.fixed_signs(768,'WI'),256);expected_basis=np.stack([basis.apply(x) for x in rows['input']]);expected_quant=[R.C.quant(x) for x in expected_basis]
            result['wire_export']={'all128_saved_bytes_recovery_exact':True,'bank_bytes':payload.stat().st_size,'trace_bytes':tracepath.stat().st_size,'expert_stride':3689472,'shared_signs_offset':4096,'expert0_offset':4864,'all_offsets_multiple64':True,'no_reencoding':True,'explicit_basis_calls':basis.calls,'explicit_basis_max_relative_error':basis.max_relative_error}
            stage='compile';binary=OUT/'meth454_switch_sparse_wo_cpu.exe';shutil.copyfile(dll,OUT/'libomp.dll');assert sha(OUT/'libomp.dll')==original374['compile']['runtime_sha256']
            argv=[str(compiler),'-O3','-std=c11','-march=x86-64-v3','-fno-fast-math','-ffp-contract=off','-fopenmp',str(CPU),'-lpsapi','-o',str(binary)]
            q=run('compile',argv,120);assert q.returncode==0,q.stderr.decode(errors='replace');result['compile']={'argv':argv,'compiler_sha256':original374['compile']['compiler_sha256'],'runtime_sha256':sha(OUT/'libomp.dll'),'binary_sha256':sha(binary)}
            stage='negative_reserved_nibble';q=run('negative_nibble',[str(binary),'--bad-nibble'],30);assert q.returncode==2 and b'candidate_minus8_nibble' in q.stderr
            stage='C_primal_before_timing';primal=OUT/'primal.bin';q=run('primal',[str(binary),'--qualify',a['manifest'],str(payload),str(tracepath),str(primal),str(primal)],240);assert q.returncode==0,q.stderr.decode(errors='replace')
            native=json.loads(q.stdout);assert native['tiny_qualified'] and native['column_alignment']==64;result['native_primal']=native
            data=primal.read_bytes();assert data[:8]==b'MC454Q01' and struct.unpack_from('<4I',data,8)==(336,768,3072,1680) and len(data)==43915224
            records=np.frombuffer(data,dtype=WIRE,offset=24);state={}
            for control in ('WI4_WO8_original_ID','WI4_WO8_ID_plus1'):
                with np.load(files[control+'_states.npz'],allow_pickle=False) as q:state[control]={k:q[k].copy() for k in ('up_raw','down','actual_function_ids')}
            with np.load(files['WO_quant_inputs.npz'],allow_pickle=False) as q:wo={k:q[k].copy() for k in q.files if k!='correct_per_expert_active_column_union'}
            seen=set()
            for rec in records:
                i,arm,wrong,identity=[int(rec[k]) for k in ('query','arm','wrong','id')];assert 0<=i<336 and 0<=arm<3 and wrong in (0,1) and not(arm==0 and wrong);key=(arm,wrong,i);assert key not in seen;seen.add(key)
                assert identity==(int(rows['expert'][i])+wrong)%128
                if arm==0:
                    for field,gold in [('raw','up_raw'),('down','down'),('basis','input'),('wi_codes','wi_codes'),('wo_codes','wo_codes'),('wi_scale','wi_scale'),('wo_scale','wo_scale')]:R.C.exact(rec[field],rows[gold][i])
                else:
                    control='WI4_WO8_ID_plus1' if wrong else 'WI4_WO8_original_ID';assert identity==int(state[control]['actual_function_ids'][i])
                    R.C.exact(rec['raw'],state[control]['up_raw'][i]);R.C.exact(rec['down'],state[control]['down'][i]);R.C.exact(rec['basis'],expected_basis[i]);R.C.exact(rec['wi_codes'],expected_quant[i][1]);R.C.exact(rec['wi_scale'],expected_quant[i][0])
                    R.C.exact(rec['wo_codes'],wo[control+'_WO_A16_codes'][i]);R.C.exact(rec['wo_scale'],wo[control+'_WO_A16_scales'][i])
                assert int(rec['nonzero'])==int(np.count_nonzero(rec['wo_codes']))
            assert len(seen)==1680;result['C_primal']={'original_functions':336,'candidate_functions_each_kernel':672,'all1680_states_codes_scales_IDs_exact':True,'negative_reserved_nibble_exit2':True};guard()
            print(json.dumps({'C_primal':result['C_primal'],'native_primal_seconds':native['native_seconds']}),flush=True)
            stage='C_actual_trace_cost';timings=OUT/'timings.jsonl';q=run('timing',[str(binary),'--bench',a['manifest'],str(payload),str(tracepath),str(primal),str(timings)],300);assert q.returncode==0,q.stderr.decode(errors='replace');result['native_timing']=json.loads(q.stdout)
            measured=[json.loads(line) for line in timings.read_text(encoding='utf-8').splitlines()];assert len(measured)==7*3*336
            expected_order=[(rep,((rep+2)%3+slot)%3,i) for rep in range(-2,5) for slot in range(3) for i in range(336)]
            assert [(v['repetition'],v['arm'],v['query']) for v in measured]==expected_order,'whole_trace_rotating_arm_order'
            values=np.empty((5,3,336),np.float64);counts=set()
            for item in measured:
                rep,arm,i=item['repetition'],item['arm'],item['query'];assert -2<=rep<5 and 0<=arm<3 and 0<=i<336 and (rep,arm,i) not in counts;counts.add((rep,arm,i))
                assert item['book']==int(books[i]) and item['id']==int(rows['expert'][i]) and np.isfinite(item['seconds']) and item['seconds']>0
                if rep>=0:values[rep,arm,i]=item['seconds']
            assert len(counts)==7056;labels=('original_I8','direct_I4_sparse_I8','pair_LUT_I4_sparse_I8');metrics={}
            for arm,label in enumerate(labels):
                v=values[:,arm,:];metrics[label]={'mean_seconds':float(np.mean(v)),'median_seconds':float(np.median(v)),'p95_seconds':float(np.quantile(v,.95)),'max_seconds':float(np.max(v)),
                    'per_sweep_seconds':np.sum(v,axis=1).tolist(),'per_book_mean_seconds':{str(b):float(np.mean(v[:,books==b])) for b in range(18,24)}}
            original=metrics[labels[0]];gates={}
            for label in labels[1:]:
                v=metrics[label];v['mean_ratio_to_original']=v['mean_seconds']/original['mean_seconds'];v['p95_ratio_to_original']=v['p95_seconds']/original['p95_seconds']
                v['per_book_mean_ratio_to_original']={b:v['per_book_mean_seconds'][b]/t for b,t in original['per_book_mean_seconds'].items()}
                gates[label]={'mean_cost_ratio_le0_80':v['mean_ratio_to_original']<=.80,'every_book_mean_ratio_le1_00':all(t<=1. for t in v['per_book_mean_ratio_to_original'].values()),'p95_cost_ratio_le1_00':v['p95_ratio_to_original']<=1.}
            result['cost']=metrics;result['feasibility_gates']=gates;result['candidate_PASS']={label:all(v.values()) for label,v in gates.items()}
            result['apparatus_gates']={'fresh453_374_retention_source_runtime_compiler_complete_files':True,'all128_real_wire_bytes_inverse_alignment_faults_exact':True,'C_tiny_signed2025_tables98304_sparse_extrema_zero_single_Walsh_qualified':True,
                'all336_original_C_states_A16_BYTE_exact':True,'all1344_candidate_C_states_basis_A16_BYTE_exact453':True,'reserved_minus8_nibble_native_fault_detected':True,
                'all7056_timed_outputs_byte_exact_primal_and_fixed_real_trace':True,'one_physical_worker_affinity_verified_primal_and_timing':all(n['worker_affinity'][0]['actual_mask']==1 and len(n['worker_affinity'])==1 for n in (result['native_primal'],result['native_timing']))}
        assert initial==(Path(a['payload']).stat().st_size,Path(a['payload']).stat().st_mtime_ns)
        nativepeak=max(result['native_primal']['native_peak_bytes'],result['native_timing']['native_peak_bytes']);assert peak+nativepeak<=4<<30,'aggregate_conservative_peak4GiB'
        result['data']={'keys':keys,'books':books.tolist(),'selected_ids':rows['expert'].tolist(),'consumed_positions':336,'distinct_selected_IDs':len(set(rows['expert'].tolist())),'warm_sweeps_per_arm':2,'measured_sweeps_per_arm':5,'whole336_trace_per_arm_before_next_arm':True}
        result['output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.glob('*')) if p.is_file()]
        result['resource']={'seconds':time.monotonic()-start,'admission_seconds':numeric-start,'numeric_seconds':time.monotonic()-numeric,'parent_peak_bytes':peak,'native_peak_bytes':nativepeak,'conservative_aggregate_peak_bytes':peak+nativepeak,'file_bytes_hashed':hashed,'output_bytes':sum(p['bytes'] for p in result['output_inventory']),'optimizer_updates':0,'new_coefficient_roundings':0,'GPU':False}
        result['decision']='C_primal_cost_PASS_prepare_NEW_all_bank_composition' if all(result['apparatus_gates'].values()) and any(result['candidate_PASS'].values()) else 'C_local_cost_FAIL_close_fixed_kernel_recipes_before_all_bank_export'
        result['scope']='One real128 finalbank, consumed fixed original inputs/routes. Full selected-FFN Walsh/quantizers/ReLU/scan/indices/direct_or_LUT_build_gathers/reductions/scales/output charged. One-worker original qualified C mv baseline. No whole-core/router/head/prefill/accepted-rate/hardwareDRAM/new n/cross-family100B proof. No C engine source edits.'
        guard();M.write(args.out,result);print(json.dumps({'apparatus':result['apparatus_gates'],'candidate_PASS':result['candidate_PASS'],'cost':result['cost'],'decision':result['decision'],'resource':result['resource'],'sha256':M.digest(args.out)}),flush=True)
    except BaseException as error:
        result.update(failure_stage=stage,error=repr(error),seconds=time.monotonic()-start,parent_peak_bytes=peak,file_bytes_hashed=hashed)
        if OUT.exists():result['partial_output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':M.digest(p)} for p in sorted(OUT.glob('*')) if p.is_file()]
        M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
