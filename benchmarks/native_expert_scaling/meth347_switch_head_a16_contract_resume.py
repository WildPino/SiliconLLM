"""Actual HEAD A16/core A8 contract; consumed composition, no quality promotion."""
import os
os.environ.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
import argparse
import gc
import inspect
import json
from pathlib import Path
import shutil
import struct
import subprocess
import time
import numpy as np
import psutil
import torch
import transformers
from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration
import meth324_switch_reference as M
import meth328_switch_native_contract as B
import meth336_switch_w8a8_contract as C
import meth345_switch_head_a16_reference as R

OUT=M.ROOT/'results/native_expert_scaling/meth347_switch_head_a16_contract_resume'
SOURCE=M.ROOT/'benchmarks/native_expert_scaling/meth345_switch_head_a16.c'
ENTRY=M.ROOT/'benchmarks/native_expert_scaling/meth345_switch_head_a16_entry.c'
PROTOCOL=M.DOC/'METH_347_SWITCH_HEAD_A16_CONTRACT_RESUME_PROTOCOL_20261003.md'
BASELINES={
    336:('meth336_switch_w8a8_contract_result.json','9ad429178abce3a8451162dff788480e5d87eb2b5fcc8537ed84edffcccbe046'),
    338:('meth338_switch_tensor_recovery_result.json','19ef2f987444d17396beea8e489d2fee341d64b61cff00714521ed806b407a7c'),
    341:('meth341_switch_batched_prefill_result.json','a73b41e01867915f0b0c5b2e1b1add2000bd97acdda329c1edecf98f9db8607c'),
    343:('meth343_switch_fresh_prediction_result.json','a4838903958be5f6bb567c1dedd802a17e6b27e2d97c4f490210139f296cbadf'),
    344:('meth344_switch_head_attribution_result.json','5936c80f3cd02914547e5baa029ce3d5d9c6067519b96d3727b960c97186ce98')}


def primitive_inputs(path):
    rng=np.random.default_rng(345345);cases=[]
    for cols in (8,15,16,17,31,32,33,768,3072,4096):
        weights=rng.integers(-127,128,size=(3,cols),dtype=np.int16).astype(np.int8)
        weights[0]=127;weights[1]=127;weights[1,::2]=-127
        x=rng.normal(size=(1,cols)).astype(np.float32);x[0,0]=32767
        x[0,1:7]=[.5,1.5,-.5,-1.5,2.5,-2.5]
        cases.append((weights,np.array([.001,1,3],dtype=np.float32),x))
    for value in (32767,-32767,0):
        cases.append((np.full((3,4096),127,dtype=np.int8),np.ones(3,dtype=np.float32),np.full((1,4096),value,dtype=np.float32)))
    with path.open('xb') as f:
        f.write(b'SWI8D001');f.write(struct.pack('<I',len(cases)))
        for w,scale,x in cases:
            f.write(struct.pack('<II',*w.shape));f.write(w.tobytes());f.write(scale.tobytes());f.write(x.tobytes())
    return cases


def primitive_result(path,cases):
    data=path.read_bytes();assert data[:8]==b'SW16R001' and struct.unpack_from('<I',data,8)[0]==len(cases)
    offset=12;records=[]
    for w,s,x in cases:
        rows,cols=struct.unpack_from('<II',data,offset);offset+=8;assert (rows,cols)==w.shape
        y=np.frombuffer(data,dtype='<f4',count=rows,offset=offset);offset+=rows*4
        codes=np.frombuffer(data,dtype='<i2',count=cols,offset=offset);offset+=cols*2
        scale=struct.unpack_from('<f',data,offset)[0];offset+=4
        dots=np.frombuffer(data,dtype='<i8',count=rows,offset=offset);offset+=rows*8
        ey,ed,ec,es=R.projection(x,w,s)
        scalar=np.array([sum(int(a)*int(b) for a,b in zip(row,ec[0])) for row in w],dtype=np.int64)
        checks={'codes_exact':np.array_equal(codes,ec[0]),'scale_exact':scale==float(es[0,0]),
                'i64_and_scalar_dot_exact':np.array_equal(dots,ed[0]) and np.array_equal(dots,scalar),
                'f32_output_exact':np.array_equal(y,ey[0])}
        assert all(checks.values()),checks
        if x[0,0]==32767 and x[0,1]==.5:assert codes[1:7].tolist()==[0,2,0,-2,2,-2]
        records.append({'rows':rows,'cols':cols,'checks':{k:bool(v) for k,v in checks.items()},'maximum_abs_dot':int(np.abs(dots).max())})
    assert offset==len(data) and max(r['maximum_abs_dot'] for r in records)==17045131264
    return {'cases':records,'passed':True,'input_cases':len(cases),'output_sha256':M.digest(path)}


def exact(actual,expected):
    checks={name:bool(np.array_equal(a,b)) for name,a,b in zip(('encoder','decoder','logits','routes'),actual,expected)}
    assert all(checks.values()),checks
    return checks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();stage='bindings';maximum=0
    result={'experiment':'METH-347-unchanged345-head-A16-contract-apparatus-resume','commands':[],'tiny_cases':[],'cases':[],'long_cases':[],'consumed_cases':[]}
    def guard(child=None):
        nonlocal maximum
        rss=psutil.Process().memory_info().rss
        if child is not None:
            try:rss+=psutil.Process(child.pid).memory_info().rss
            except psutil.NoSuchProcess:pass
        maximum=max(maximum,rss)
        if rss>16<<30 or time.monotonic()-start>1200:
            if child is not None and child.poll() is None:child.kill();child.wait()
            raise RuntimeError('contract_resource_guard_20min_16GiB')
    try:
        for p in (Path(__file__),SOURCE,ENTRY,PROTOCOL,B.ENGINE,Path(M.__file__),Path(B.__file__),Path(C.__file__),Path(R.__file__),Path(R.R.__file__),Path(R.R.R.__file__),C.BRIDGE):M.committed(p)
        records={}
        for n,(name,sha) in BASELINES.items():
            p=M.DOC/name;M.committed(p);assert M.digest(p)==sha;records[n]=json.loads(p.read_text(encoding='utf-8'))
        assert all(records[336]['gates'].values()) and all(records[338]['gates'].values()) and all(records[341]['gates'].values()) and all(records[344]['gates'].values())
        assert not records[343]['gates']['original_top1_agreement_ge0p95']
        cohort_path=M.DOC/'meth342_switch_fresh_span_manifest.json';M.committed(cohort_path)
        assert M.digest(cohort_path)==records[343]['manifest342_sha256']
        cohort_record=json.loads(cohort_path.read_text(encoding='utf-8'))
        assert transformers.__version__=='4.57.6' and torch.__version__=='2.6.0+cu124'
        assert M.digest(inspect.getfile(SwitchTransformersForConditionalGeneration))=='5acb527d50a07e4bce3a5dec4db4f0c476cb4a01e8eb05219f26607034a12a83'
        assert M.digest(C.BRIDGE)==C.BRIDGE_SHA and M.digest(B.COMPILER)==B.COMPILER_SHA and M.digest(B.DLL)==B.DLL_SHA
        result.update({'controller_sha256':M.digest(__file__),'source_sha256':M.digest(SOURCE),'entry_sha256':M.digest(ENTRY),'engine_sha256':M.digest(B.ENGINE),'reference_sha256':M.digest(R.__file__),'protocol_sha256':M.digest(PROTOCOL),'baseline_sha256':{str(n):v[1] for n,v in BASELINES.items()}})
        OUT.mkdir(parents=True);shutil.copyfile(B.DLL,OUT/'libomp.dll');binary=OUT/'meth345_switch_head_a16.exe'
        command=[str(B.COMPILER),'-O3','-std=c11','-march=x86-64-v3','-fno-fast-math','-ffp-contract=off','-fopenmp','-DSILICON_SWITCH_HEAD_A16',str(B.ENGINE),'-o',str(binary)]
        compiled=subprocess.run(command,capture_output=True,timeout=120);(OUT/'compile.stdout.log').write_bytes(compiled.stdout);(OUT/'compile.stderr.log').write_bytes(compiled.stderr)
        assert compiled.returncode==0,compiled.stderr.decode(errors='replace')
        result['compile']={'argv':command,'binary_sha256':M.digest(binary),'compiler_sha256':B.COMPILER_SHA,'runtime_sha256':B.DLL_SHA}
        env=os.environ.copy();env.pop('OMP_PROC_BIND',None);env.update({'OMP_WAIT_POLICY':'PASSIVE','KMP_AFFINITY':'none'})
        def run(argv,label,threads=1):
            env['OMP_NUM_THREADS']=str(threads)
            with (OUT/(label+'.stdout.log')).open('wb') as out,(OUT/(label+'.stderr.log')).open('wb') as err:
                child=subprocess.Popen([str(binary),*argv],stdout=out,stderr=err,env=env)
                while child.poll() is None:guard(child);time.sleep(.25)
            result['commands'].append({'argv':[str(binary),*argv],'returncode':child.returncode,'stdout_sha256':M.digest(OUT/(label+'.stdout.log')),'stderr_sha256':M.digest(OUT/(label+'.stderr.log'))})
            assert child.returncode==0,(label,child.returncode)
            guard()
        stage='integer_primitive';inp=OUT/'head_integer_cases.bin';cases=primitive_inputs(inp);answer=OUT/'head_integer_results.bin';run(['--head-int-dot',str(inp),str(answer)],'head_primitive');result['integer_primitive']=primitive_result(answer,cases)
        torch.set_num_threads(1);bridge=json.loads(C.BRIDGE.read_text(encoding='utf-8'))
        for capacity in (1,64):
            stage=f'tiny_capacity{capacity}';torch.manual_seed(328);old=next(v for v in bridge['cases'] if v['capacity']==capacity)
            model=SwitchTransformersForConditionalGeneration(SwitchTransformersConfig(**old['tiny_config'])).eval()
            original=B.export(model,OUT/f'tiny_original{capacity}');assert original['weights_sha256']==old['artifact']['weights_sha256']
            entries,artifact=C.tiny_export(model,OUT/f'tiny_target{capacity}')
            with R.compact_reference(model,entries,artifact['payload']) as mode:expected=B.torch_reference(model,torch.tensor([[2,3,4,5,6,7]]),torch.tensor([[0,8,9,10]]))
            assert mode.calls['head_a16_integer_projection']==4 and mode.calls['integer_projection']>0 and mode.calls['bmm']>0 and mode.calls['softmax']>0
            output=OUT/f'tiny{capacity}.bin';argv=[artifact['manifest'],'2,3,4,5,6,7','0,8,9,10',str(output),'0'];run(argv,f'tiny{capacity}')
            checks=exact(B.read_output(output),expected);faults=[]
            for number in (1,2,3,4,5):
                if number==4 and capacity==64:continue
                output=OUT/f'tiny{capacity}.fault{number}.bin';run([*argv[:3],str(output),str(number)],f'tiny{capacity}.fault{number}')
                error=B.relative(B.read_output(output)[2],expected[2]);assert error>1e-4
                faults.append({'fault':number,'logit_relative_l2':error,'detected':True})
            arrays=OUT/f'tiny{capacity}.reference.npz';np.savez(arrays,**dict(zip(('encoder','decoder','logits','routes'),expected)))
            result['tiny_cases'].append({'capacity':capacity,'same_original328':True,'artifact':artifact,'exact':checks,'calls':dict(mode.calls),'faults':faults,'reference_sha256':M.digest(arrays),'passed':True})
            del model;gc.collect();guard()
        stage='target_identity';recovered=records[338];artifact=recovered['artifact'];payload=Path(artifact['payload']);spec=Path(artifact['manifest']);before=(payload.stat().st_size,payload.stat().st_mtime_ns)
        import hashlib
        h=hashlib.sha256()
        with payload.open('rb') as f:
            while block:=f.read(4<<20):h.update(block);guard()
        assert h.hexdigest()==artifact['sha256'] and before[0]==artifact['bytes'] and M.digest(spec)==artifact['manifest_sha256'];result['artifact']=artifact
        entries=recovered['tensors'];cfg=SwitchTransformersConfig(**recovered['original_config'])
        with torch.device('meta'):model=SwitchTransformersForConditionalGeneration(cfg)
        result['reference_model']=R.R.target_control_model(model,entries,payload)
        for i,control in enumerate(records[336]['cases']):
            stage=f'complete_independent_case{i}'
            with R.compact_reference(model,entries,payload) as mode:expected=B.torch_reference(model,torch.tensor([control['source_ids']]),torch.tensor([control['decoder_ids']]))
            assert mode.calls['head_a16_integer_projection']==len(control['decoder_ids'])
            oldpath=C.OUT/f'case{i}.bin';assert M.digest(oldpath)==control['native_sha256'];old=B.read_output(oldpath)
            assert all(np.array_equal(expected[k],old[k]) for k in (0,1,3)),'head_change_affected_upstream_reference'
            checks=[];hashes=[]
            for threads in (1,6):
                for profile in (0,1):
                    label=f'case{i}.t{threads}.p{profile}';prefix=OUT/label
                    run([str(spec),','.join(map(str,control['source_ids'])),','.join(map(str,control['decoder_ids'])),str(prefix),str(threads),str(profile),'0','1','0'],label,threads)
                    output=Path(str(prefix)+'.0.bin');checks.append(exact(B.read_output(output),expected));hashes.append(M.digest(output))
            assert len(set(hashes))==1
            result['cases'].append({'source_ids':control['source_ids'],'decoder_ids':control['decoder_ids'],'exact':checks,'native_sha256':hashes[0],'reference_calls':dict(mode.calls),'passed':True})
        del model;gc.collect()
        for i,fixture in enumerate(records[341]['fixtures']):
            stage=f'long_composition{i}';previous=next(r for r in records[341]['measurements'] if r['source_length']==fixture['source_length'] and r['threads']==6)
            oldpath=M.ROOT/f'results/native_expert_scaling/meth341_switch_batched_prefill/cost.s{fixture["source_length"]}.t6.-1.bin'
            assert M.digest(oldpath)==previous['rows'][0]['output_sha256'];old=B.read_output(oldpath);expected=(old[0],old[1],R.head_from_states(old[1][:,-1],entries,payload),old[3])
            hashes=[]
            for threads in (1,6):
                label=f'long.s{fixture["source_length"]}.t{threads}';prefix=OUT/label
                run([str(spec),','.join(map(str,fixture['source_ids'])),','.join(map(str,fixture['decoder_ids'])),str(prefix),str(threads),'0','0','1','0'],label,threads)
                output=Path(str(prefix)+'.0.bin');exact(B.read_output(output),expected);hashes.append(M.digest(output))
            assert len(set(hashes))==1;result['long_cases'].append({**fixture,'native_sha256':hashes[0],'core_routes_exact341':True,'head_independent_a16_exact':True,'passed':True})
        for bi,book in enumerate(records[343]['books']):
            for ci,case in enumerate(book['cases']):
                stage=f'consumed_book{bi}_case{ci}';stem=f'book{bi}.case{ci}'
                oldpath=M.ROOT/'results/native_expert_scaling/meth343_switch_fresh_prediction'/(stem+'.0.bin');assert M.digest(oldpath)==case['native_output_sha256'];old=B.read_output(oldpath)
                record=next(r for r in records[344]['cases'] if r['book']==bi and r['case']==ci)
                arraypath=M.ROOT/'results/native_expert_scaling/meth344_switch_head_attribution'/(stem+'.head_counterfactuals.npz');assert M.digest(arraypath)==record['arrays_sha256']
                with np.load(arraypath) as saved:head=saved['native_a16'].copy()
                cohort=cohort_record['items'][bi]['cases'][ci]
                prefix=OUT/stem;run([str(spec),','.join(map(str,cohort['source_ids'])),','.join(map(str,cohort['decoder_ids'])),str(prefix),'6','0','0','1','0'],stem,6)
                output=Path(str(prefix)+'.0.bin');checks=exact(B.read_output(output),(old[0],old[1],head,old[3]))
                result['consumed_cases'].append({'book':bi,'case':ci,'exact':checks,'native_sha256':M.digest(output),'old343_sha256':case['native_output_sha256'],'counterfactual344_sha256':record['arrays_sha256']})
            print(json.dumps({'completed_book':bi,'consumed_cases':len(result['consumed_cases'])}),flush=True)
        assert before==(payload.stat().st_size,payload.stat().st_mtime_ns)
        result['gates']={'head_integer_scalar_extremes_exact':True,'same328_tiny_nine_faults_exact':True,'fresh_unchanged_payload_manifest_identity':True,'complete_independent_engineering_reference_exact':True,'long_cost_fixtures_exact_head_only':True,'all96_consumed_core_route_exact_and_head344_exact':len(result['consumed_cases'])==96}
        result['resource']={'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum,'end_rss_bytes':psutil.Process().memory_info().rss}
        result['decision']='new_execution_profile_numeric_eligible_for_separate_actual_CPU_cost_then_NEW_quality'
        result['scope']='HEAD A16 ONLY with fixed serialized338 target and core A8. Consumed343/344 cases confirm composition; no untouched quality or generation/rate/useful larger-n claim. Original343 failure unchanged.'
        guard();M.write(args.out,result);print(json.dumps({'gates':result['gates'],'resource':result['resource'],'sha256':M.digest(args.out)}),flush=True)
    except BaseException as error:
        result.update({'stage':stage,'error':repr(error),'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum});M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
