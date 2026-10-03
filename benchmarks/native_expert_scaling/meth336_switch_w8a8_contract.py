"""Complete compact target C vs serialized I8/I64 reference; no quality/rate gate."""
import argparse
import gc
import inspect
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
import transformers
from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration
import meth324_switch_reference as M
import meth327_switch_tensor_binding as T
import meth328_switch_native_contract as B
import meth335_switch_w8a8_export as E
import meth336_switch_integer_reference as R

SOURCE=M.ROOT/'benchmarks/native_expert_scaling/meth336_switch_w8a8.c'
ENTRY=M.ROOT/'benchmarks/native_expert_scaling/meth336_switch_w8a8_entry.c'
PROTOCOL=M.DOC/'METH_336_SWITCH_W8A8_CONTRACT_PROTOCOL_20261003.md'
EXPORTED=M.DOC/'meth338_switch_tensor_recovery_result.json'
EXPORTED_SHA='19ef2f987444d17396beea8e489d2fee341d64b61cff00714521ed806b407a7c'
ORIGINAL_DIAGNOSTIC=M.DOC/'meth331_switch_router_diagnostic_result.json'
ORIGINAL_DIAGNOSTIC_SHA='668a01a42276ee51955a9cde930f224cbb57180016bc7e88f1696226738948a1'
ORIGINAL_ARRAYS=M.ROOT/'results/native_expert_scaling/meth331_switch_router_diagnostic'
OUT=M.ROOT/'results/native_expert_scaling/meth336_switch_w8a8_contract'
BRIDGE=M.DOC/'meth328_switch_native_contract_result.json'
BRIDGE_SHA='08f04429c9081266c8bad29aaa3c7ffe0f2534c4484e39e3498e5f1f1ea9f330'


def tiny_export(model,path):
    path.mkdir();entries={};aliases={};payload=path/'weights.bin'
    with payload.open('xb') as stream:
        for name,tensor in sorted(model.state_dict().items()):
            entries[name]=E.emit_tensor(stream,name,tensor.detach().numpy().copy(),aliases)
    spec=path/'manifest.bin';E.manifest(model.config.to_dict(),entries,payload,spec)
    return entries,{'payload':str(payload),'payload_sha256':M.digest(payload),'manifest':str(spec),'manifest_sha256':M.digest(spec)}


def primitive_inputs(path):
    rng=np.random.default_rng(336);cases=[]
    for cols in (8,15,16,17,31,32,33,768,3072,4096):
        weights=rng.integers(-127,128,size=(3,cols),dtype=np.int16).astype(np.int8)
        weights[0]=127;weights[1]=127;weights[1,::2]=-127
        values=rng.normal(size=(1,cols)).astype(np.float32);values[0,0]=127.
        values[0,1:7]=[.5,1.5,-.5,-1.5,2.5,-2.5]
        scales=np.array([.001,1.,3.],dtype=np.float32)
        cases.append((weights,scales,values))
    cases.append((np.full((3,4096),127,dtype=np.int8),np.ones(3,dtype=np.float32),np.full((1,4096),127,dtype=np.float32)))
    cases.append((np.full((3,768),127,dtype=np.int8),np.ones(3,dtype=np.float32),np.zeros((1,768),dtype=np.float32)))
    with path.open('xb') as stream:
        stream.write(b'SWI8D001');stream.write(struct.pack('<I',len(cases)))
        for weights,scales,values in cases:
            stream.write(struct.pack('<II',*weights.shape));stream.write(weights.tobytes());stream.write(scales.tobytes());stream.write(values.tobytes())
    return cases


def primitive_result(path,cases):
    raw=path.read_bytes();assert raw[:8]==b'SWI8R001' and struct.unpack_from('<I',raw,8)[0]==len(cases)
    offset=12;entries=[]
    for weights,scales,values in cases:
        rows,cols=struct.unpack_from('<II',raw,offset);offset+=8;assert (rows,cols)==weights.shape
        actual=np.frombuffer(raw,dtype='<f4',count=rows,offset=offset).copy();offset+=rows*4
        actual_codes=np.frombuffer(raw,dtype=np.int8,count=cols,offset=offset).copy();offset+=cols
        actual_scale=struct.unpack_from('<f',raw,offset)[0];offset+=4
        dots=np.frombuffer(raw,dtype='<i4',count=rows,offset=offset).copy();offset+=rows*4
        expected,expected_dots,codes,activation_scales=R.integer_projection(values,weights,scales)
        checks={'exact_codes':bool(np.array_equal(actual_codes,codes[0])),
                'exact_activation_scale':actual_scale==float(activation_scales[0,0]),
                'exact_integer_dots':bool(np.array_equal(dots,expected_dots[0])),
                'exact_scaled_f32_outputs':bool(np.array_equal(actual,expected[0]))}
        entries.append({'rows':rows,'cols':cols,'checks':checks,'maximum_integer_abs':int(np.abs(dots.astype(np.int64)).max()),'passed':all(checks.values())})
    assert offset==len(raw)
    return {'cases':entries,'passed':all(v['passed'] for v in entries),'output_sha256':M.digest(path)}


def compare(actual,expected,d):
    pooled=[B.relative(a,b) for a,b in zip(actual[:3],expected[:3])]
    states=[B.relative(a,b) for a,b in zip(actual[0],expected[0])]
    states.extend(B.relative(a,b) for a,b in zip(actual[1].reshape(-1,d),expected[1].reshape(-1,d)))
    choices=bool(np.array_equal(actual[3][:,:2],expected[3][:,:2]))
    probability=float(np.abs(actual[3][:,2]-expected[3][:,2]).max())
    greedy=bool(np.array_equal(actual[2].argmax(-1),expected[2].argmax(-1)))
    return {'pooled_encoder_decoder_logit_relative_l2':pooled,'state_errors':states,'maximum_state_error':max(states),
            'exact_route_choice_capacity':choices,'maximum_probability_abs_error':probability,'exact_greedy_top1':greedy,
            'passed':max(pooled+states)<=1e-4 and choices and probability<=1e-6 and greedy}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();stage='bindings';maximum_rss=0
    result={'experiment':'METH-336-complete-W8A8-native-integer-contract','commands':[],'tiny_cases':[],'cases':[]}
    def guard():
        nonlocal maximum_rss
        rss=psutil.Process().memory_info().rss;maximum_rss=max(maximum_rss,rss)
        assert rss<=32<<30 and time.monotonic()-start<=1200,'reference_resource_guard'
    try:
        for path in (Path(__file__),SOURCE,ENTRY,PROTOCOL,EXPORTED,ORIGINAL_DIAGNOSTIC,T.HEADERS,T.ACQUISITION,E.BOUND,E.QUALIFIED,BRIDGE,B.ENGINE,
                     Path(M.__file__),Path(T.__file__),Path(B.__file__),Path(E.__file__),Path(R.__file__),Path(R.R.__file__)):
            M.committed(path)
        exported=json.loads(EXPORTED.read_text());qualified=json.loads(E.QUALIFIED.read_text());bridge=json.loads(BRIDGE.read_text())
        assert M.digest(EXPORTED)==EXPORTED_SHA and M.digest(ORIGINAL_DIAGNOSTIC)==ORIGINAL_DIAGNOSTIC_SHA
        diagnostic_record=json.loads(ORIGINAL_DIAGNOSTIC.read_text())
        assert all(exported['gates'].values()) and M.digest(E.QUALIFIED)==E.QUALIFIED_SHA
        assert M.digest(T.HEADERS)==T.HEADERS_SHA and M.digest(T.ACQUISITION)==json.loads(E.BOUND.read_text())['acquisition_sha256']
        assert M.digest(BRIDGE)==BRIDGE_SHA and M.digest(E.BOUND)==E.BOUND_SHA
        assert transformers.__version__=='4.57.6' and torch.__version__=='2.6.0+cu124'
        assert M.digest(inspect.getfile(SwitchTransformersForConditionalGeneration))=='5acb527d50a07e4bce3a5dec4db4f0c476cb4a01e8eb05219f26607034a12a83'
        assert M.digest(B.COMPILER)==B.COMPILER_SHA and M.digest(B.DLL)==B.DLL_SHA
        result.update({'controller_sha256':M.digest(__file__),'native_source_sha256':M.digest(SOURCE),'entry_sha256':M.digest(ENTRY),
                       'engine_sha256':M.digest(B.ENGINE),'protocol_sha256':M.digest(PROTOCOL),'recovered338_sha256':M.digest(EXPORTED),'original331_diagnostic_sha256':M.digest(ORIGINAL_DIAGNOSTIC),
                       'integer_reference_sha256':M.digest(R.__file__),'model':exported['model'],'revision':exported['revision']})
        OUT.mkdir(parents=True);shutil.copyfile(B.DLL,OUT/'libomp.dll');binary=OUT/'meth336_switch_w8a8.exe'
        command=[str(B.COMPILER),'-O3','-std=c11','-march=x86-64-v3','-fno-fast-math','-ffp-contract=off','-fopenmp',
                 '-DSILICON_SWITCH_W8A8_REFERENCE',str(B.ENGINE),'-lbcrypt','-o',str(binary)]
        stage='compile';compiled=subprocess.run(command,capture_output=True,timeout=120)
        (OUT/'compile.stdout.log').write_bytes(compiled.stdout);(OUT/'compile.stderr.log').write_bytes(compiled.stderr)
        assert compiled.returncode==0,compiled.stderr.decode(errors='replace')
        result['compile']={'argv':command,'binary_sha256':M.digest(binary),'compiler_sha256':B.COMPILER_SHA,'runtime_sha256':B.DLL_SHA}
        env=os.environ.copy();env.pop('OMP_PROC_BIND',None);env.update({'OMP_NUM_THREADS':'1','OMP_WAIT_POLICY':'PASSIVE','KMP_AFFINITY':'none'})
        def run(argv,label):
            nonlocal maximum_rss
            child_start=time.monotonic()
            with (OUT/(label+'.stdout.log')).open('wb') as stdout,(OUT/(label+'.stderr.log')).open('wb') as stderr:
                child=subprocess.Popen([str(binary),*argv],stdout=stdout,stderr=stderr,env=env)
                while child.poll() is None:
                    try:
                        rss=psutil.Process().memory_info().rss+psutil.Process(child.pid).memory_info().rss;maximum_rss=max(maximum_rss,rss)
                        if rss>32<<30 or time.monotonic()-start>1200:child.kill();child.wait();raise RuntimeError('native_resource_guard')
                    except psutil.NoSuchProcess:pass
                    time.sleep(.25)
            result['commands'].append({'argv':argv,'returncode':child.returncode,'seconds':time.monotonic()-child_start,
                                       'stdout_sha256':M.digest(OUT/(label+'.stdout.log')),'stderr_sha256':M.digest(OUT/(label+'.stderr.log'))})
            assert child.returncode==0,(label,child.returncode)
        stage='adversarial_integer_primitive';primitive=OUT/'integer_cases.bin';cases=primitive_inputs(primitive);answer=OUT/'integer_results.bin'
        run(['--int-dot',str(primitive),str(answer)],'integer_primitive');result['integer_primitive']=primitive_result(answer,cases)
        assert result['integer_primitive']['passed'],'integer_primitive_before_model'
        torch.set_num_threads(1)
        for capacity in (1,64):
            stage=f'tiny_capacity{capacity}';torch.manual_seed(328);original=next(v for v in bridge['cases'] if v['capacity']==capacity)
            tiny=SwitchTransformersForConditionalGeneration(SwitchTransformersConfig(**original['tiny_config'])).eval()
            uncompressed=B.export(tiny,OUT/f'tiny_original{capacity}');assert uncompressed['weights_sha256']==original['artifact']['weights_sha256']
            entries,artifact=tiny_export(tiny,OUT/f'tiny_compact{capacity}')
            source=torch.tensor([[2,3,4,5,6,7]]);decoder=torch.tensor([[0,8,9,10]])
            with R.compact_reference(tiny,entries,artifact['payload']) as mode:expected=B.torch_reference(tiny,source,decoder)
            calls=dict(mode.calls);assert calls.get('integer_projection',0)>0 and calls.get('mm',0)>0 and calls.get('bmm',0)>0 and calls.get('softmax',0)>0
            output=OUT/f'tiny_capacity{capacity}.bin';argv=[artifact['manifest'],'2,3,4,5,6,7','0,8,9,10',str(output),'0'];run(argv,f'tiny_capacity{capacity}')
            measurement=compare(B.read_output(output),expected,8);faults=[]
            for number in (1,2,3,4,5):
                if number==4 and capacity==64:continue
                fault=OUT/f'tiny_capacity{capacity}.fault{number}.bin';run([*argv[:3],str(fault),str(number)],f'tiny_capacity{capacity}_fault{number}')
                error=B.relative(B.read_output(fault)[2],expected[2]);faults.append({'fault':number,'logit_relative_l2':error,'detected':error>1e-4})
            arrays=OUT/f'tiny_capacity{capacity}.matched_reference.npz';np.savez(arrays,encoder=expected[0],decoder=expected[1],logits=expected[2],routes=expected[3])
            entry={'capacity':capacity,'artifact':artifact,'same328_original_weights':True,'comparison':measurement,'calls':calls,'faults':faults,
                   'reference_sha256':M.digest(arrays),'passed':measurement['passed'] and all(v['detected'] for v in faults)}
            result['tiny_cases'].append(entry);print(json.dumps({'tiny_capacity':capacity,'comparison':measurement,'passed':entry['passed']}),flush=True)
            del tiny;gc.collect();guard()
        assert all(v['passed'] for v in result['tiny_cases']),'tiny_contract_before_source'
        stage='fresh_complete_target_hash';artifact=exported['artifact'];payload=Path(artifact['payload']);spec=Path(artifact['manifest'])
        assert payload.stat().st_size==artifact['bytes'] and T.file_digest(payload,start)==artifact['sha256'] and M.digest(spec)==artifact['manifest_sha256']
        stage='all_actual_native_target_tensors';audit=OUT/'all_native_target_tensors.tsv';run(['--audit',str(spec),str(audit)],'target_audit');seen=set()
        for line in audit.read_text().splitlines():
            name,encoding,dims,rows,cols,size,sha,scalesize,scalesha=line.split('\t');expected=exported['tensors'][name]
            shape=[int(rows)] if int(dims)==1 else [int(rows),int(cols)]
            assert shape==expected['shape'] and int(encoding)==expected['encoding'] and int(size)==expected['bytes'] and sha==expected['sha256']
            assert int(scalesize)==expected['scale_bytes'] and scalesha==(expected['scale_sha256'] or '-') and name not in seen;seen.add(name)
        assert seen==set(exported['tensors']) and len(seen)==6392
        result['all_native_target_bytes']={'names':6392,'passed':True,'audit_sha256':M.digest(audit),'full_payload_sha256':artifact['sha256'],'all_codes_and_scales_exact335':True}
        stage='official_architecture_actual_target_controls'
        cfg=SwitchTransformersConfig(**exported['original_config'])
        with torch.device('meta'):model=SwitchTransformersForConditionalGeneration(cfg)
        result['reference_model']=R.target_control_model(model,exported['tensors'],payload)
        guard()
        for index,control in enumerate(qualified['cases']):
            stage=f'complete_compact_case{index}';source=torch.tensor([control['source_ids']]);decoder=torch.tensor([control['decoder_ids']])
            with R.compact_reference(model,exported['tensors'],payload) as mode:expected=B.torch_reference(model,source,decoder)
            calls=dict(mode.calls);assert calls.get('integer_projection',0)>0 and calls.get('mm',0)>0 and calls.get('bmm',0)>0 and calls.get('softmax',0)>0
            guard();source_csv=','.join(map(str,source[0].tolist()));decoder_csv=','.join(map(str,decoder[0].tolist()))
            output=OUT/f'case{index}.bin';argv=[str(spec),source_csv,decoder_csv,str(output),'0'];run(argv,f'case{index}')
            native=B.read_output(output);measurement=compare(native,expected,768)
            fault=OUT/f'case{index}.zero_head.bin';run([*argv[:3],str(fault),'5'],f'case{index}_zero_head');fault_error=B.relative(B.read_output(fault)[2],expected[2])
            arrays=OUT/f'case{index}.matched_reference.npz';np.savez(arrays,encoder=expected[0],decoder=expected[1],logits=expected[2],routes=expected[3])
            original_path=ORIGINAL_ARRAYS/f'case{index}.reference_arrays.npz'
            assert M.digest(original_path)==diagnostic_record['cases'][index]['reference_array_sha256']
            with np.load(original_path) as saved:original=tuple(saved[name].copy() for name in ('encoder','decoder','logits','routes'))
            diagnostic={'pooled_relative_l2':[B.relative(a,b) for a,b in zip(native[:3],original[:3])],
                'changed_route_choices':int(np.sum(native[3][:,0]!=original[3][:,0])),
                'maximum_probability_abs_difference':float(np.abs(native[3][:,2]-original[3][:,2]).max()),
                'greedy_ids_original':original[2].argmax(-1).tolist(),'greedy_ids_target':native[2].argmax(-1).tolist(),'consumed_control_only_no_quality_gate':True,'cached_original331_reference_sha256':M.digest(original_path)}
            entry={'source_ids':source[0].tolist(),'decoder_ids':decoder[0].tolist(),'comparison':measurement,'reference_calls':calls,
                   'reference_sha256':M.digest(arrays),'native_sha256':M.digest(output),'zero_head_fault_error':fault_error,
                   'original_donor_diagnostic':diagnostic,'passed':measurement['passed'] and fault_error>1e-4}
            result['cases'].append(entry);print(json.dumps({'case':index,'comparison':measurement,'original_donor_diagnostic':diagnostic,'passed':entry['passed']}),flush=True)
        result['gates']={'adversarial_integer_primitive':result['integer_primitive']['passed'],
                         'same328_tiny_nine_faults':all(v['passed'] for v in result['tiny_cases']),
                         'all_native6392_target_code_scale_bytes':True,'complete_compact_integer_reference':all(v['passed'] for v in result['cases'])}
        result['resource']={'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum_rss,'end_rss_bytes':psutil.Process().memory_info().rss}
        result['decision']='eligible_for_separate_actual_complete_CPU_cost_before_quality' if all(result['gates'].values()) else 'unchanged_compact_integer_contract_not_qualified'
        result['scope']='Full real14.664B source all-bank compact C vs independent serialized I64/scaling reference, original F32 controls, correctly rounded F64-exp to F32. Original donor consumed diagnostics are not heldout quality. No LUT/RAM-useful-n or accepted>=50 claim. Original329/330/332 failures unchanged; future NEW quality PRIMARY unmodified donor.'
        guard();M.write(args.out,result);print(json.dumps({'sha256':M.digest(args.out),'gates':result['gates'],'resource':result['resource'],'decision':result['decision']}),flush=True)
    except BaseException as error:
        result.update({'stage':stage,'error':repr(error),'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum_rss})
        M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
