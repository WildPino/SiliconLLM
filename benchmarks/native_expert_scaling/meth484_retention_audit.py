"""Independent complete byte/geometry/budget verifier; never imports METH484 main."""
import argparse
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
RAW = DOC / 'meth484_shared_integer_eligibility_result.json'
RET = DOC / 'RETENTION_484_20261006.json'


def write(path, data):
    with path.open('xb') as f:
        f.write((json.dumps(data, indent=2, allow_nan=False) + '\n').replace('\n','\r\n').encode())


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def foreign():
    own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}; daemons = []
    for p in psutil.process_iter(['name','cmdline']):
        if p.pid in own:
            continue
        name = (p.info['name'] or '').lower(); args = p.info['cmdline'] or []
        if name == 'pythonw.exe' and len(args) == 2 and Path(args[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
            daemons.append(p.pid); continue
        assert not (name.startswith(('python','clang')) or (name.startswith('meth') and name.endswith('.exe'))), ('foreign_job',p.pid,name)
    return daemons


def audit():
    ap = argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); arg = ap.parse_args()
    assert arg.out.resolve() == RET.resolve() and not RET.exists() and not RET.with_suffix('.failure.json').exists() and sys.flags.optimize == 0
    started = time.monotonic(); peak = hashed = 0; stage = 'frozen_inputs'
    report = {'experiment':'METH484 independent ALL-descriptor eligibility admission', 'start_utc':stamp(),
              'process_instance':{'pid':os.getpid(),'create_time_unix':psutil.Process().create_time()},'commands':[]}
    def guard():
        nonlocal peak
        m = psutil.Process().memory_info(); peak = max(peak,m.rss,getattr(m,'peak_wset',0))
        assert peak <= 256<<20 and time.monotonic()-started <= 60
    def digest(path):
        nonlocal hashed
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            while b:=f.read(1<<20):
                h.update(b);hashed+=len(b);guard()
        return h.hexdigest()
    try:
        binding_path=DOC/'meth484_source_binding.json'; binding=json.loads(binding_path.read_bytes()); raw=json.loads(RAW.read_bytes())
        assert digest(binding_path)==raw['binding_sha256'] and all(raw['gates'].values()) and not RAW.with_suffix('.failure.json').exists()
        assert sys.version==binding['runtime']['python'] and Path(sys.executable).resolve()==Path(binding['runtime']['executable']).resolve() and psutil.__version__==binding['runtime']['psutil_version']
        for v in binding['helpers']+binding['catalog']+binding['source_records']+binding['runtime']['files']+[binding['parent'],binding['retention']]:
            assert Path(v['path']).stat().st_size==v['bytes'] and digest(v['path'])==v['sha256']
        assert not subprocess.check_output(['git','diff','HEAD','--',str(binding_path.relative_to(ROOT)),str(Path(__file__).relative_to(ROOT))],cwd=ROOT)
        parent=json.loads(Path(binding['parent']['path']).read_bytes()); retained=json.loads(Path(binding['retention']['path']).read_bytes())
        assert all(parent['apparatus_gates'].values()) and all(retained['gates'].values()) and retained['raw']['sha256']==binding['parent']['sha256'];del retained
        report['preserved_daemons_before']=foreign();report['raw']={'path':str(RAW),'bytes':RAW.stat().st_size,'sha256':digest(RAW)}
        winpath=ROOT/'results/native_expert_scaling/meth484_windows_terminal.json'; win=json.loads(winpath.read_bytes())
        assert win['query_available'] and not win['matching_scientific_events'] and len(win['instances'])==1
        instance=win['instances'][0];assert instance['pid']==raw['process_instance']['pid'] and instance['create_time_unix']==raw['process_instance']['create_time_unix']
        assert datetime.datetime.fromisoformat(instance['start_utc'])==datetime.datetime.fromisoformat(raw['start_utc'])
        assert datetime.datetime.fromisoformat(instance['end_utc'])==datetime.datetime.fromisoformat(raw['end_utc'])
        hits=[];expression=re.compile(r'cublas|CUDA_R_8I|CUBLAS_COMPUTE_32I|GemmEx|(?:q0|q1|q2).*16384',re.I)
        for entry in binding['catalog']:
            for line,textline in enumerate(Path(entry['path']).read_text(encoding='utf-8').splitlines(),1):
                if expression.search(textline):hits.append({'path':entry['relative'],'line':line,'text':textline[:600]})
        assert hits==raw['backend_name_catalog']['hits'] and raw['backend_name_catalog']['scope']==binding['catalog_scope']
        report['main_windows']={'path':str(winpath),'sha256':digest(winpath),'actual_controller_instance_exact':True}
        report['sources']={};stage='ALL_binary_descriptors'
        for sb in binding['sources']:
            n=sb['n']; result=raw['sources'][str(n)];payload=sb['payload'];stat=Path(payload['path']).stat()
            assert [stat.st_size,stat.st_mtime_ns]==[payload['bytes'],payload['mtime_ns']]==parent['sources'][str(n)]['artifact_stat_before']
            assert digest(sb['manifest']['path'])==sb['manifest']['sha256'] and result['payload_SHA_refreshed'] is False and result['payload_values_read'] is False
            stream=io.BytesIO(Path(sb['manifest']['path']).read_bytes());assert stream.read(8)==b'SWI8A001'
            header=struct.unpack('<13IfII',stream.read(64));keys=('d','ff','heads','dk','enc','dec','n','capacity','vocab','buckets','distance','encstep','decstep')
            c=dict(zip(keys,header[:13]));c['epsilon']=header[13];assert c==result['config'] and c['n']==n and header[14]==1
            d,ff=c['d'],c['ff'];assert d==768 and ff==3072 and c['enc']==c['dec']==12
            assert c['heads']*c['dk']==d and c['buckets']==32 and c['distance']==128 and 0<c['epsilon']<1 and c['capacity']>0 and c['vocab']<=65536
            def text():
                length,=struct.unpack('<I',stream.read(4));return stream.read(length).decode('utf-8')
            assert Path(text()).resolve()==Path(payload['path']).resolve()
            observed={};segments={};sum_gpu=sum_pad=sum_scales=0;max_gpu=max_cpu=0;counts={}
            for expected in result['tensors']:
                name=text(); values=struct.unpack('<5I3Q',stream.read(44));f,dim,m,k,enc,off,scale,num=values
                base={'name':name,'file':f,'dims':dim,'rows':m,'cols':k,'encoding':enc,'code_offset':off,'scale_offset':scale,'elements':num}
                assert name not in observed and len(name.encode())<192 and f==0 and dim in (1,2) and m*k==num and 0<num<1<<31 and enc in (0,1)
                assert off%64==0 and off+num*(1 if enc else 4)<=stat.st_size
                if enc:
                    assert dim==2 and k<=4096 and scale%64==0 and scale+4*m<=stat.st_size
                else:
                    assert scale==0
                batch=1
                if name in ('shared.weight','encoder.embed_tokens.weight','decoder.embed_tokens.weight'):
                    category='F32_embedding_alias';shape=(c['vocab'],d,2,0)
                elif name=='lm_head.weight':
                    category='shared_head';shape=(c['vocab'],d,2,1)
                elif name.endswith('final_layer_norm.weight'):
                    assert name in ('encoder.final_layer_norm.weight','decoder.final_layer_norm.weight');category='F32_control';shape=(d,1,1,0)
                else:
                    match=re.fullmatch(r'(encoder|decoder)\.block\.(\d+)\.layer\.(\d+)\.(.+)',name);assert match,name
                    stack,index,layer,suffix=match.groups();index=int(index);layer=int(layer)
                    assert index<c['enc' if stack=='encoder' else 'dec']
                    fflayer=1 if stack=='encoder' else 2; step=c['encstep' if stack=='encoder' else 'decstep'];sparse=step>0 and(index%step==1 or step==1)
                    if suffix=='layer_norm.weight':
                        assert layer in ((0,1) if stack=='encoder' else (0,1,2));category='F32_control';shape=(d,1,1,0)
                    elif suffix=='SelfAttention.relative_attention_bias.weight':
                        assert index==layer==0;category='F32_control';shape=(c['buckets'],c['heads'],2,0)
                    elif re.fullmatch(r'(SelfAttention|EncDecAttention)\.[qkvo]\.weight',suffix):
                        kind,organ,_=suffix.split('.');assert(layer==0 if kind=='SelfAttention' else stack=='decoder' and layer==1)
                        category='shared_attention';shape=(d,d,2,1)
                        batch=256 if ((stack=='encoder' and organ in 'qkv') or(kind=='EncDecAttention' and organ in 'kv')) else 1
                    elif suffix=='mlp.router.classifier.weight':
                        assert layer==fflayer and sparse;category='CPU_F32_router';shape=(n,d,2,0)
                    elif suffix.startswith('mlp.experts.'):
                        ex=re.fullmatch(r'mlp\.experts\.expert_(\d+)\.(wi|wo)\.weight',suffix);assert ex and int(ex[1])<n and sparse and layer==fflayer
                        category='CPU_expert';shape=(ff,d,2,1) if ex[2]=='wi' else(d,ff,2,1)
                    else:
                        assert suffix in ('mlp.wi.weight','mlp.wo.weight') and layer==fflayer and not sparse
                        category='shared_dense_FFN';shape=(ff,d,2,1) if suffix=='mlp.wi.weight' else(d,ff,2,1)
                assert(m,k,dim,enc)==shape
                base.update(category=category,max_legal_queries_per_operator=batch)
                if category.startswith('shared_'):
                    pm=8*math.ceil(m/8);pd=8*math.ceil(k/8)
                    b={'padded_rows':pm,'padded_cols':pd,'GPU_weight_bytes':pm*pd,'weight_padding_bytes':pm*pd-m*k,
                       'GPU_operator_scratch_bytes':batch*(4*pd+16*pm),'CPU_operator_arrays_bytes':batch*(10*pd+28*pm+4),
                       'H2D_per_query_bytes':4*pd,'D2H_per_query_bytes':16*pm,'CPU_row_scales_bytes':4*m,
                       'I32_partial_bound':pd*128*127,'I64_reconstruction_intermediate_bound':pd*(128*127*129+128*2*16384),'original_dot_bound':k*128*32767}
                    assert b['I32_partial_bound']<1<<31 and b['I64_reconstruction_intermediate_bound']<1<<63 and b['original_dot_bound']<1<<53
                    base['integer_backend_budget']=b;sum_gpu+=b['GPU_weight_bytes'];sum_pad+=b['weight_padding_bytes'];sum_scales+=4*m
                    max_gpu=max(max_gpu,b['GPU_operator_scratch_bytes']);max_cpu=max(max_cpu,b['CPU_operator_arrays_bytes'])
                assert base==expected,('descriptor_or_budget',name);observed[name]=base;counts[category]=counts.get(category,0)+1
                for kind,offset,size in [('value',off,num*(1 if enc else 4))]+([('row_scale',scale,4*m)]if enc else[]):
                    segments.setdefault((offset,size),[]).append((name,category,kind))
                guard()
            assert len(observed)==header[15] and list(observed)==sorted(observed) and not stream.read(1)
            sparse={};dense={}
            for stack,layers,step in(('encoder',c['enc'],c['encstep']),('decoder',c['dec'],c['decstep'])):
                sparse[stack]=[l for l in range(layers) if step>0 and(l%step==1 or step==1)]
                dense[stack]=[l for l in range(layers) if l not in sparse[stack]]
            assert sparse==result['sparse_layers'] and dense==result['dense_layers']
            banks=sum(map(len,sparse.values()))
            assert counts=={'F32_embedding_alias':3,'F32_control':4+2*c['enc']+3*c['dec'],
                            'CPU_F32_router':banks,'CPU_expert':2*n*banks,'shared_head':1,
                            'shared_attention':4*c['enc']+8*c['dec'],'shared_dense_FFN':2*sum(map(len,dense.values()))}
            groups={};alias=[];end=0;padding=0
            for(offset,size),users in sorted(segments.items()):
                assert offset>=end;padding+=offset-end;end=offset+size
                assert len({u[1:] for u in users})==1
                cat=users[0][1];group=groups.setdefault(cat,{'value_bytes':0,'row_scale_bytes':0,'tensor_names':counts[cat]})
                group['value_bytes' if users[0][2]=='value' else 'row_scale_bytes']+=size
                if len(users)>1:
                    assert cat=='F32_embedding_alias';alias.append({'offset':offset,'bytes':size,'names':sorted(u[0]for u in users)})
            padding+=stat.st_size-end
            budget={'payload_bytes':stat.st_size,'unique_component_bytes':sum(v[1]for v in segments),'file_alignment_or_trailing_bytes':padding,
                    'groups':groups,'alias_groups':alias,'shared_tensor_count':sum(v for k,v in counts.items()if k.startswith('shared_')),
                    'GPU_shared_weights_bytes':sum_gpu,'GPU_weight_padding_bytes':sum_pad,'GPU_max_operator_scratch_bytes':max_gpu,
                    'CPU_max_operator_arrays_bytes':max_cpu,'CPU_shared_row_scale_bytes':sum_scales}
            assert budget==result['byte_budget'] and sum_gpu<=1<<30 and max_gpu<=16<<20 and len(alias)==1
            assert budget['unique_component_bytes']+padding==stat.st_size
            stage='ALL_case_and_cost_algebra';source=parent['sources'][str(n)];summary=source['mode_summaries']['1'];whole=result['whole_cost']
            full=summary['ALL_case_mean_full_seconds'];eligible=sum(summary['matrix_mean_seconds_by_kind'][kind]for kind in('dense_control_matrices','head_matrix'));fraction=eligible/full
            assert whole['inherited_profile1_aggregate_sum_of_96_case_means_seconds']==full and whole['inherited_dense_plus_head_aggregate_seconds']==eligible
            assert whole['eligible_measured_fraction']==fraction and whole['remaining_measured_fraction']==1-fraction
            for constraint in whole['constraints']:
                rate=summary['accepted_'+constraint['metric']+'_rate_mean'];ratio=rate/constraint['target_accepted_IDs_per_second']
                assert constraint['inherited_rate']==rate and constraint['required_whole_cost_ratio']==ratio
                assert constraint['maximum_component_multiplier_at_h0']==(ratio-1+fraction)/fraction or math.isclose(constraint['maximum_component_multiplier_at_h0'],(ratio-1+fraction)/fraction,rel_tol=1e-14,abs_tol=1e-14)
                assert constraint['maximum_additional_overhead_fraction_at_r0']==ratio-(1-fraction)
            ae=len(result['dense_layers']['encoder']);ad=len(result['dense_layers']['decoder']);le,ld=c['enc'],c['dec']
            assert len(result['all_96_cases_including_rejected'])==len(source['cases'])==96
            sums={k:0 for k in('eligible_logical_query_maps','proposed_integer_GEMM_invocations','H2D_digit_bytes','D2H_four_I32_partial_bytes')}
            for published,case in zip(result['all_96_cases_including_rejected'],source['cases'],strict=True):
                s=published['source_tokens'];t=case['generated_tokens'];assert s==29 and published['generated_tokens']==t and published['book']==case['book'] and published['case']==case['case']
                expected_calls=[(4*le+2*ae+2*ld)*s,(6*ld+2*ad)*t]
                for rows in case['modes'].values():
                    assert len(rows)==4
                    for row in rows:
                        assert row['source_tokens']==s and row['actual_generated_tokens']==t
                        assert [row['counters'][i][0]['calls']for i in(0,1)]==expected_calls
                        for phase,code_bytes,scale_bytes in ((0,((4*le+2*ld)*d*d+2*ae*d*ff)*s,((4*le+2*ld)*d+ae*(d+ff))*4*s),
                                                           (1,(6*ld*d*d+2*ad*d*ff)*t,(6*ld*d+ad*(d+ff))*4*t)):
                            ctr=row['counters'][phase][0];assert(ctr['code_bytes'],ctr['scale_bytes'],ctr['f32_bytes'])==(code_bytes,scale_bytes,0)
                        ctr=row['counters'][1][3];assert tuple(ctr[k]for k in('calls','code_bytes','scale_bytes','f32_bytes'))==(t,t*c['vocab']*d,t*c['vocab']*4,0)
                charged={'eligible_logical_query_maps':sum(expected_calls)+t,
                         'proposed_integer_GEMM_invocations':(3*le+2*ld)+(le+2*ae)*s+(6*ld+2*ad+1)*t,
                         'H2D_digit_bytes':4*((4*le+2*ld)*d*s+ae*(d+ff)*s+6*ld*d*t+ad*(d+ff)*t+d*t),
                         'D2H_four_I32_partial_bytes':16*(((4*le+2*ld)*d+ae*(d+ff))*s+(6*ld*d+ad*(d+ff)+c['vocab'])*t)}
                assert all(published[k]==v for k,v in charged.items())
                for k,v in charged.items():sums[k]+=v
            assert sums==result['aggregate_logical_transfer_and_invocations'];report['sources'][str(n)]={'ALL_tensor_names':len(observed),'ALL_case_rows':768,'budget':budget,'charged':sums};guard()
        assert report['sources']['128']['budget']['GPU_shared_weights_bytes']==report['sources']['256']['budget']['GPU_shared_weights_bytes']
        # Exhaustive scalar integer identity, independent of GPU/matrix/source values; no old numerical control replay.
        for signed in range(-32768,32768):
            u=signed%65536;q0=u%128;q1=(u//128)%128;q2=u//16384-(4 if signed<0 else 0)
            assert 0<=q0<=127 and 0<=q1<=127 and -2<=q2<=1 and signed==q0+128*q1+16384*q2
        assert 512*128*32767<1<<31
        report['signed16_identity_scalar_values_audited']=65536
        report['preserved_daemons_after']=foreign();guard();report['end_utc']=stamp()
        report['gates']={'all_frozen_runtime_sources_and_original_records_exact':True,'ALL_binary_descriptors_categories_aliases_byte_and_scratch_budgets_exact':True,
                         'ALL1536_inherited_case_rows_cost_and_transfer_algebra_exact':True,'signed16_identity_and_overflow_bounds_exact':True,
                         'main_actual_Windows_instance_available_zero_matching_faults':True,'zero_payload_native_GPU_solver_model_replays':True}
        report['decision']='COMPLETE_METADATA_ELIGIBILITY_INDEPENDENTLY_ADMITTED_NO_SPEED_QUALITY_OR_DRIVER_PROOF'
        report['resource']={'seconds_excluding_imports':time.monotonic()-started,'peak_working_set_bytes':peak,'bytes_hashed':hashed};write(RET,report)
        print(json.dumps({'sha256':digest(RET),'gates':report['gates'],'resource':report['resource'],
                          'terminal_peak_working_set_bytes':peak,'terminal_seconds':time.monotonic()-started,'decision':report['decision']}),flush=True)
    except BaseException as error:
        report.update(stage=stage,error=repr(error),end_utc=stamp(),resource={'seconds_excluding_imports':time.monotonic()-started,'peak_working_set_bytes':peak,'bytes_hashed':hashed})
        write(RET.with_suffix('.failure.json'),report);raise


if __name__=='__main__':
    audit()
