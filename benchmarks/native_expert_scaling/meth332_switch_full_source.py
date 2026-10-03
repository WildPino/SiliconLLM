"""Every native source tensor and full14.7B Switch forward vs official reference."""
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
import zipfile
import numpy as np
import psutil
import torch
import transformers
from transformers import AutoTokenizer,SwitchTransformersConfig,SwitchTransformersForConditionalGeneration
import meth324_switch_reference as M
import meth327_switch_tensor_binding as T
import meth328_switch_native_contract as B

ARITHMETIC=M.ROOT/'benchmarks/native_expert_scaling/meth332_switch_f64_mv.c'
PROTOCOL=M.DOC/'METH_332_SWITCH_F64_MV_PROTOCOL_20261003.md'
BOUND=M.DOC/'meth327_switch_tensor_binding_result.json'
BOUND_SHA='e72fc17b5527dc34df5c00ed7c44498b2ee4fc5d338ed8b130fff101a30c20a7'
NATIVE=M.ROOT/'benchmarks/native_expert_scaling/meth332_switch_source_binding.c'
BRIDGE=M.DOC/'meth328_switch_native_contract_result.json'
BRIDGE_SHA='08f04429c9081266c8bad29aaa3c7ffe0f2534c4484e39e3498e5f1f1ea9f330'
BASELINE=M.DOC/'meth329_switch_full_source_result.json'
BASELINE_SHA='e76373359a81c00ffcbe4aa6d8684c83e17972c4eccbf0440d9c354c55cb419d'
STABLE=M.DOC/'meth330_switch_full_source_result.json'
STABLE_SHA='b28dc0f5a8a289be7aa0c23270d3abba2f7b67d07904dc576153618e0bc46745'
DIAGNOSTIC=M.DOC/'meth331_switch_router_diagnostic_result.json'
DIAGNOSTIC_SHA='668a01a42276ee51955a9cde930f224cbb57180016bc7e88f1696226738948a1'
OLD_RMS=M.ROOT/'benchmarks/native_expert_scaling/meth330_switch_stable_rms.c'
OUT=M.ROOT/'results/native_expert_scaling/meth332_switch_full_source'
TEXTS=(('The capital of France is <extra_id_0>.','<extra_id_0> Paris <extra_id_1>'),
       ('The scientist measured <extra_id_0> after the experiment.','<extra_id_0> the temperature <extra_id_1>'))


def source_manifest(config,header,bound,path):
    entries=[]
    for file_index,shard in enumerate(header['shards']):
        source=T.SOURCE/shard['name']
        with zipfile.ZipFile(source) as archive,source.open('rb') as stream:
            names=[v.filename for v in archive.infolist() if v.filename.endswith('/data.pkl')];assert len(names)==1
            prefix=names[0][:-len('data.pkl')]
            for name,record in sorted(header['actual_tensor_headers'].items()):
                if record['shard']!=shard['name']:continue
                storage=archive.getinfo(prefix+'data/'+record['storage']['key'])
                assert storage.compress_type==zipfile.ZIP_STORED
                stream.seek(storage.header_offset);raw=stream.read(30);values=struct.unpack('<I5H3I2H',raw)
                assert values[0]==0x04034b50 and values[3]==0
                offset=storage.header_offset+30+values[-2]+values[-1]+record['offset']*4
                shape=record['shape'];assert len(shape) in (1,2)
                elements=int(np.prod(shape));assert elements*4==bound['tensors'][name]['bytes']
                assert record['offset']*4+elements*4<=storage.file_size and offset%4==0
                entries.append((name,file_index,shape,offset,elements))
    assert len(entries)==6392 and len({v[0] for v in entries})==6392
    values=(config.d_model,config.d_ff,config.num_heads,config.d_kv,config.num_layers,config.num_decoder_layers,
        config.num_experts,config.expert_capacity,config.vocab_size,config.relative_attention_num_buckets,
        config.relative_attention_max_distance,config.encoder_sparse_step,config.decoder_sparse_step)
    with path.open('xb') as stream:
        stream.write(b'SWF32A01');stream.write(struct.pack('<13IfII',*values,config.layer_norm_epsilon,len(header['shards']),len(entries)))
        for shard in header['shards']:B.pack_string(stream,str((T.SOURCE/shard['name']).resolve()))
        for name,file_index,shape,offset,elements in entries:
            B.pack_string(stream,name);stream.write(struct.pack('<4I2Q',file_index,len(shape),shape[0],shape[1] if len(shape)==2 else 1,offset,elements))
    return {'path':str(path),'sha256':M.digest(path),'bytes':path.stat().st_size,'tensor_names':len(entries),
            'weight_encoding':'original_all_F32_unchanged_readonly_ZIP_storage_offsets','source_weight_copy_bytes':0}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();stage='bindings';maximum_rss=0
    result={'experiment':'METH-332-F64-MV-real-Switch256-native-source','cases':[],'commands':[]}
    try:
        for path in (Path(__file__),NATIVE,ARITHMETIC,B.SOURCE,OLD_RMS,B.ENGINE,PROTOCOL,BOUND,BRIDGE,BASELINE,STABLE,DIAGNOSTIC,T.HEADERS,T.ACQUISITION,
                     Path(M.__file__),Path(T.__file__),Path(B.__file__)):M.committed(path)
        assert M.digest(BOUND)==BOUND_SHA and M.digest(BRIDGE)==BRIDGE_SHA
        assert M.digest(BASELINE)==BASELINE_SHA
        assert M.digest(STABLE)==STABLE_SHA and M.digest(DIAGNOSTIC)==DIAGNOSTIC_SHA
        insertion='static double dot64(const float *a,const float *b,int n){__m256d x=_mm256_setzero_pd(),y=_mm256_setzero_pd();int i=0;for(;i+8<=n;i+=8){x=_mm256_add_pd(x,_mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(a+i)),_mm256_cvtps_pd(_mm_loadu_ps(b+i))));y=_mm256_add_pd(y,_mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(a+i+4)),_mm256_cvtps_pd(_mm_loadu_ps(b+i+4))));}double values[4];_mm256_storeu_pd(values,_mm256_add_pd(x,y));double sum=0.;for(int j=0;j<4;j++)sum+=values[j];for(;i<n;i++)sum+=(double)a[i]*(double)b[i];return sum;}\n'
        marker='static void mv(const float *w,const float *x,float *y,int rows,int cols){'
        old='y[row]=dot(w+(size_t)row*cols,x,cols);'
        assert OLD_RMS.read_text().count(marker)==1 and OLD_RMS.read_text().count(old)==1
        assert ARITHMETIC.read_text()==OLD_RMS.read_text().replace(marker,insertion+marker).replace(old,'y[row]=(float)dot64(w+(size_t)row*cols,x,cols);')
        bound=json.loads(BOUND.read_text());bridge=json.loads(BRIDGE.read_text());assert bound['passed'] and all(bridge['gates'].values())
        assert M.digest(T.HEADERS)==T.HEADERS_SHA
        assert transformers.__version__=='4.57.6' and torch.__version__=='2.6.0+cu124'
        assert M.digest(inspect.getfile(SwitchTransformersForConditionalGeneration))=='5acb527d50a07e4bce3a5dec4db4f0c476cb4a01e8eb05219f26607034a12a83'
        header=next(v for v in json.loads(T.HEADERS.read_text())['models'] if v['model']==bound['model'])
        acquisition=json.loads(T.ACQUISITION.read_text());assert acquisition['passed'] and M.digest(T.ACQUISITION)==bound['acquisition_sha256']
        for item in acquisition['files']:
            path=Path(item['path']);assert path.stat().st_size==item['bytes'] and item['complete']
            assert T.file_digest(path,start)==item['sha256']
        cfg=SwitchTransformersConfig(**json.loads((T.SOURCE/'config.json').read_text()))
        assert (cfg.d_model,cfg.d_ff,cfg.num_layers,cfg.num_decoder_layers,cfg.num_experts,cfg.expert_capacity,cfg.vocab_size)==(768,3072,12,12,256,64,32128)
        assert cfg.tie_word_embeddings and not cfg.router_bias and cfg.dense_act_fn=='relu' and cfg.router_dtype=='float32'
        result.update({'controller_sha256':M.digest(__file__),'native_source_sha256':M.digest(NATIVE),'arithmetic_source_sha256':M.digest(ARITHMETIC),
            'engine_sha256':M.digest(B.ENGINE),'protocol_sha256':M.digest(PROTOCOL),'source_binding_sha256':BOUND_SHA,
            'native_tiny_bridge_sha256':BRIDGE_SHA,'model':bound['model'],'revision':bound['revision'],'original_config':cfg.to_dict()})
        result['preserved329_sha256']=BASELINE_SHA
        result['preserved330_sha256']=STABLE_SHA
        result['diagnostic331_sha256']=DIAGNOSTIC_SHA
        result['sole_arithmetic_change']='F64 product/sum for every matrix-vector row then F32 output, prior330 stable RMS preserved; attention dot unchanged F32'
        OUT.mkdir(parents=True);spec=OUT/'source_manifest.bin';result['native_manifest']=source_manifest(cfg,header,bound,spec)
        assert M.digest(B.COMPILER)==B.COMPILER_SHA and M.digest(B.DLL)==B.DLL_SHA
        shutil.copyfile(B.DLL,OUT/'libomp.dll');binary=OUT/'meth332_switch_source_binding.exe'
        command=[str(B.COMPILER),'-O3','-std=c11','-march=x86-64-v3','-fno-fast-math','-ffp-contract=off','-fopenmp',
            '-DSILICON_SWITCH_F64_MV_SOURCE',str(B.ENGINE),'-lbcrypt','-o',str(binary)]
        compiled=subprocess.run(command,capture_output=True,timeout=120)
        (OUT/'compile.stdout.log').write_bytes(compiled.stdout);(OUT/'compile.stderr.log').write_bytes(compiled.stderr)
        assert compiled.returncode==0,compiled.stderr.decode(errors='replace')
        result['compile']={'argv':command,'binary_sha256':M.digest(binary),'compiler_sha256':B.COMPILER_SHA,'runtime_sha256':B.DLL_SHA}
        env=os.environ.copy();env.pop('OMP_PROC_BIND',None);env.update({'OMP_NUM_THREADS':'1','OMP_WAIT_POLICY':'PASSIVE','KMP_AFFINITY':'none'})
        def guard():
            nonlocal maximum_rss
            rss=psutil.Process().memory_info().rss;maximum_rss=max(maximum_rss,rss)
            assert time.monotonic()-start<=1200 and rss<=32<<30
        def run(argv,label):
            nonlocal maximum_rss
            child_start=time.monotonic()
            with (OUT/(label+'.stdout.log')).open('wb') as stdout,(OUT/(label+'.stderr.log')).open('wb') as stderr:
                child=subprocess.Popen([str(binary),*argv],stdout=stdout,stderr=stderr,env=env)
                while child.poll() is None:
                    try:
                        rss=psutil.Process().memory_info().rss+psutil.Process(child.pid).memory_info().rss
                        maximum_rss=max(maximum_rss,rss)
                        if rss>32<<30 or time.monotonic()-start>1200:child.kill();child.wait();raise RuntimeError('native_resource_guard')
                    except psutil.NoSuchProcess:pass
                    time.sleep(.25)
            result['commands'].append({'argv':argv,'returncode':child.returncode,'seconds':time.monotonic()-child_start,
                'stdout_sha256':M.digest(OUT/(label+'.stdout.log')),'stderr_sha256':M.digest(OUT/(label+'.stderr.log'))})
            assert child.returncode==0,(label,child.returncode)
        stage='stable_rms_tiny_controls'
        result['tiny_cases']=[]
        torch.set_num_threads(1)
        for capacity in (1,64):
            torch.manual_seed(328)
            original=next(v for v in bridge['cases'] if v['capacity']==capacity)
            tiny=SwitchTransformersForConditionalGeneration(SwitchTransformersConfig(**original['tiny_config'])).eval()
            tiny_artifact=B.export(tiny,OUT/f'tiny_capacity{capacity}')
            assert tiny_artifact['weights_sha256']==original['artifact']['weights_sha256']
            tiny_source=torch.tensor([[2,3,4,5,6,7]]);tiny_decoder=torch.tensor([[0,8,9,10]])
            expected=B.torch_reference(tiny,tiny_source,tiny_decoder)
            output=OUT/f'tiny_capacity{capacity}.bin'
            run([tiny_artifact['spec'],'2,3,4,5,6,7','0,8,9,10',str(output),'0'],f'tiny_capacity{capacity}')
            actual=B.read_output(output)
            errors=[B.relative(a,b) for a,b in zip(actual[:3],expected[:3])]
            state_errors=[B.relative(a,b) for a,b in zip(actual[0],expected[0])]+[B.relative(a,b) for a,b in zip(actual[1].reshape(-1,8),expected[1].reshape(-1,8))]
            route_exact=bool(np.array_equal(actual[3][:,:2],expected[3][:,:2]))
            probability_error=float(np.max(np.abs(actual[3][:,2]-expected[3][:,2])))
            entry={'capacity':capacity,'artifact':tiny_artifact,'same328_weight_bytes':True,'errors':errors,
                'maximum_state_error':max(state_errors),'exact_choices_capacity':route_exact,'probability_maxabs':probability_error,
                'passed':max(errors+state_errors)<=1e-4 and route_exact and probability_error<=1e-6 and bool(np.array_equal(actual[2].argmax(-1),expected[2].argmax(-1)))}
            result['tiny_cases'].append(entry)
            del tiny;gc.collect()
        stage='all_native_tensor_hashes'
        audit=OUT/'all_native_tensor_hashes.tsv';run(['--audit',str(spec),str(audit)],'audit')
        rows={}
        for line in audit.read_text().splitlines():
            name,dimensions,rows_count,cols_count,bytes_count,sha=line.split('\t');expected=bound['tensors'][name]
            shape=[int(rows_count)] if int(dimensions)==1 else [int(rows_count),int(cols_count)]
            assert shape==expected['shape'] and int(bytes_count)==expected['bytes'] and sha==expected['sha256']
            assert name not in rows;rows[name]=sha
        assert set(rows)==set(bound['tensors']) and len(rows)==6392
        result['all_native_tensor_audit']={'passed':True,'names':len(rows),'tsv_sha256':M.digest(audit),
            'all_actual_native_mapped_bytes_exact327':True,'working_set_trim_interval_tensors':64}
        stage='official_full_source_model'
        torch.set_num_threads(1)
        with torch.device('meta'):model=SwitchTransformersForConditionalGeneration(cfg)
        seen=set()
        for shard in header['shards']:
            state=torch.load(T.SOURCE/shard['name'],map_location='cpu',weights_only=True,mmap=True)
            assert not (seen&set(state));seen.update(state)
            incompatible=model.load_state_dict(state,strict=False,assign=True);assert not incompatible.unexpected_keys
            del state;gc.collect();guard()
        assert seen==set(bound['tensors']) and all(p.device.type=='cpu' and p.dtype==torch.float32 for p in model.parameters())
        model.tie_weights();model.requires_grad_(False);model.eval()
        assert sum(p.numel() for p in model.parameters())==14664154368
        result['official_reference_loaded']={'all_source_names':len(seen),'unique_parameters':14664154368,'no_meta_remaining':True,
            'weights_only':True,'mmap':True,'model_code_sha256':M.digest(inspect.getfile(type(model)))}
        tokenizer=AutoTokenizer.from_pretrained(T.SOURCE,local_files_only=True,use_fast=True,trust_remote_code=False)
        assert tokenizer.is_fast and tokenizer.pad_token_id==0 and tokenizer.eos_token_id==1
        assert tokenizer.convert_tokens_to_ids('<extra_id_0>')==32099 and tokenizer.convert_tokens_to_ids('<extra_id_1>')==32098
        result['tokenizer_control']={'class':type(tokenizer).__name__,'tokens':len(tokenizer),'pad':0,'eos':1,'extra_id_0':32099,'extra_id_1':32098,'local_only':True}
        for index,(prompt,completion) in enumerate(TEXTS):
            stage=f'engineering_case{index}'
            source=tokenizer(prompt,return_tensors='pt')['input_ids'];decoder=torch.tensor([[0]+tokenizer.encode(completion,add_special_tokens=False)])
            assert source.shape[1]<=64 and decoder.shape[1]<=64
            reference=B.torch_reference(model,source,decoder);guard()
            baseline=OUT/f'case{index}.output.bin'
            source_csv=','.join(map(str,source[0].tolist()));decoder_csv=','.join(map(str,decoder[0].tolist()))
            run([str(spec),source_csv,decoder_csv,str(baseline),'0'],f'case{index}')
            native=B.read_output(baseline)
            errors=[B.relative(a,b) for a,b in zip(native[:3],reference[:3])]
            encoder_errors=[B.relative(a,b) for a,b in zip(native[0],reference[0])]
            decoder_errors=[B.relative(a,b) for a,b in zip(native[1].reshape(-1,768),reference[1].reshape(-1,768))]
            choices=bool(np.array_equal(native[3][:,:2],reference[3][:,:2]));probability_error=float(np.max(np.abs(native[3][:,2]-reference[3][:,2])))
            top1=bool(np.array_equal(native[2].argmax(-1),reference[2].argmax(-1)))
            fault=OUT/f'case{index}.zero_head.bin';run([str(spec),source_csv,decoder_csv,str(fault),'5'],f'case{index}_zero_head')
            fault_error=B.relative(B.read_output(fault)[2],reference[2])
            case={'prompt':prompt,'forced_decoder_text':completion,'source_ids':source[0].tolist(),'decoder_ids':decoder[0].tolist(),
                'encoder_decoder_logit_relative_l2':errors,'encoder_state_errors':encoder_errors,'decoder_state_errors':decoder_errors,
                'exact_route_choice_and_capacity':choices,'maximum_selected_probability_abs_error':probability_error,
                'greedy_top1_equal':top1,'reference_greedy_top1_ids':reference[2].argmax(-1).tolist(),
                'reference_dropped_routes':int(np.sum(reference[3][:,1]==0)),'native_route_rows':native[3].tolist(),
                'zero_head_fault_relative_l2':fault_error,'native_output_sha256':M.digest(baseline),
                'passed':max(errors+encoder_errors+decoder_errors)<=1e-4 and choices and probability_error<=1e-6 and top1 and fault_error>1e-4}
            result['cases'].append(case);print(json.dumps({'case':index,'errors':errors,'choice':choices,'probability_error':probability_error,'top1':top1,'passed':case['passed']}),flush=True)
        result['gates']={'all_native_actual_source_tensors':result['all_native_tensor_audit']['passed'],
            'same328_tiny_stable_rms_controls':all(v['passed'] for v in result['tiny_cases']),
            'complete_actual_source_native_controls':all(v['passed'] for v in result['cases'])}
        result['resource']={'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum_rss,
            'end_controller_rss_bytes':psutil.Process().memory_info().rss}
        result['decision']='actual_source_native_bridge_eligible_for_separate_compact_precision_and_quality_cost' if all(result['gates'].values()) else 'unchanged_full_source_bridge_not_qualified'
        result['scope']='Complete original14.664B pretrained bank availability and actual C mapped weights/selected native cached forward. Two consumed engineering controls, no heldout/generation usefulness/task quality claim; F32 alone is not final efficient representation. Native subprocess times are assay/resource records, not accepted rate benchmarks.'
        guard();M.write(args.out,result);print(json.dumps({'sha256':M.digest(args.out),'gates':result['gates'],'resource':result['resource'],'decision':result['decision']}),flush=True)
    except BaseException as error:
        result.update({'error':repr(error),'stage':stage,'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_rss_bytes':maximum_rss})
        M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
