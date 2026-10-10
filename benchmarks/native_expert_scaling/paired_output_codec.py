"""One FIT-only output-weighted paired codec plus original final readout control."""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];BROOT=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(BROOT))
from original_packed_capacity import SITE,extent,sha,write
from original_falcon_whole_recovery import memory_reader
from source_cached_fit_capture import decode,save
sys.path.insert(0,str(SITE))
SD,D,K,V=2048,256,255,65537


def prepare(a):
    assert not a.out.exists() and not a.directory.exists();a.directory.mkdir();start=time.monotonic()
    engine=ROOT/'benchmarks/phase60/engine.c';text=engine.read_text();pieces=[];functions=[]
    for name in ('hsum256','dotf','matvec','rmsnorm'):
        match=re.search(r'^static[^\n]*\b'+name+r'\(',text,re.M);assert match
        pos=text.index('{',match.start());depth=1;end=pos+1
        while depth:depth+=(text[end]=='{')-(text[end]=='}');end+=1
        body=text[match.start():end];pieces.append(body);functions.append(dict(name=name,original_body_sha256=hashlib.sha256(body.encode()).hexdigest()))
    source=a.directory/'paired_original_readout.c';dll=a.directory/'paired_original_readout.dll'
    boiler='''#include <stddef.h>
#include <math.h>
#include <immintrin.h>
#define D 256
#define V 65537
#define OMP_PFOR
'''
    exports='''
__declspec(dllexport) int codec_abi(int which){ return which?V:D; }
__declspec(dllexport) int codec_readout(const float* head,const float* states,int count,float* normalized,float* scores){
    if(count<1||count>256)return 1;
    float gamma[D];for(int i=0;i<D;i++)gamma[i]=1.0f;
    for(int j=0;j<count;j++){
        rmsnorm(states+(size_t)j*D,gamma,normalized+(size_t)j*D);
        matvec(head,normalized+(size_t)j*D,scores+(size_t)j*V,V,D);
        for(int i=0;i<V;i++)if(!isfinite(scores[(size_t)j*V+i]))return 2;
    }return 0;
}
'''
    with source.open('x',encoding='utf-8',newline='\n') as f:f.write(boiler+'\n\n'.join(pieces)+exports)
    compiler=Path(json.loads((DOC/'chatbot_hybrid_engine_probe_binding_20261009.json').read_bytes())['compiler'])
    command=[str(compiler),'-O3','-mavx2','-mfma','-march=znver2','-shared',str(source.resolve()),'-o',str(dll.resolve()),'-lm']
    proc=subprocess.run(command,capture_output=True,timeout=45);log=a.directory/'compile.log';log.write_bytes(proc.stdout+proc.stderr)
    assert proc.returncode==0,log.read_text(errors='replace')
    write(a.out,dict(schema='PAIRED_OUTPUT_NATIVE_PREPARATION_V1',source=extent(source),dll=extent(dll),engine=extent(engine),functions=functions,
        compiler=extent(compiler),command=command,exit_code=proc.returncode,log=extent(log),seconds=time.monotonic()-start,
        scope='Compile-only preparation: four original bodies unchanged, OMP_PFOR empty, F32 norm/head; no model input or readout call.'))
    print(json.dumps(dict(preparation=str(a.out),seconds=time.monotonic()-start,dll_sha256=sha(dll))),flush=True)


def bind(a):
    assert not a.binding.exists()
    parent_path=DOC/'source_cached_fit_binding_20261010.json';parent=json.loads(parent_path.read_bytes())
    result_path=DOC/'source_cached_fit_result_20261010.json';result=json.loads(result_path.read_bytes())
    audit_path=DOC/'source_cached_fit_stored_adjudication_20261010.json';audit=json.loads(audit_path.read_bytes())
    terminal_path=audit_path.with_suffix('.terminal.json');terminal=json.loads(terminal_path.read_bytes())
    assert audit['decision']==result['decision']=='PAIRED_CACHED_FIT_PASS' and audit['result']==extent(result_path)
    assert terminal['exit_code']==0 and terminal['error'] is None and terminal['result']==extent(audit_path)
    prep=json.loads(a.native_preparation.read_bytes());assert prep['exit_code']==0 and prep['engine']==extent(ROOT/'benchmarks/phase60/engine.c')
    compiler=Path(prep['compiler']['path']);files=[Path(x['path']) for x in parent['inputs']]+[parent_path,result_path,audit_path,terminal_path,
        Path(__file__),a.protocol,a.native_preparation,Path(prep['source']['path']),Path(prep['dll']['path']),Path(prep['log']['path']),Path(prep['engine']['path']),compiler,
        BROOT/'original_latent_readout_attribution.py']
    files += [p for p in compiler.parent.iterdir() if p.name in ('clang-21.exe','ld.lld.exe','lld.exe','libclang-cpp.dll','libLLVM.dll')]
    source_terminal=json.loads(result_path.with_suffix('.terminal.json').read_bytes())
    files += [result_path.with_suffix('.terminal.json')]+[Path(x['path']) for x in source_terminal['outputs']]
    inputs=[extent(path) for path in dict.fromkeys(files)];by_path={item['path']:item for item in inputs}
    for item in parent['inputs']+source_terminal['outputs']:assert by_path[item['path']]==item
    for key in ('source','dll','log','compiler'):assert by_path[prep[key]['path']]==prep[key]
    assert len(result['records'])==24 and result['labels']==4422 and all(rec['split']=='FIT' for rec in result['records'])
    write(a.binding,dict(schema='PAIRED_OUTPUT_CODEC_BINDING_V1',inputs=inputs,records=result['records'],fields=parent['fields'],source=parent['source'],
        native=prep,source_custody=extent(audit_path),rank=K,source_dimension=SD,target_dimension=D,vocab=V,multiplier=.01953125,epsilon=1e-5,
        weighting='Each case1/24; each of its m labels1/(24*m); uncentered feature second moment.',
        covariance_cut_relative=1e-12,PSD_negative_relative=1e-10,discarded_covariance_energy_relative=1e-8,
        rho=1.,gain_rule='2*max_FIT_norm(phi)/sqrt(256)',head_chunk=4096,label_chunk=16,
        source_label_gate=dict(mean_KL=.01,mean_disagreement=.01,case_KL=.05,case_disagreement=.05),
        numeric_gates=dict(algebra_relative=1e-8,stat_absolute=1e-9,metric_absolute=1e-10,real_score_absolute=1e-8,native_score_absolute=1e-4,
            native_norm_relative=1e-5,native_mean_KL_to_real=1e-6,native_case_disagreement_to_real=.01,surrogate_relative=1e-7),
        capture_limits=dict(seconds=900,reserve_seconds=30,OS_bytes=6<<30,GPU_allocated_bytes=4<<30,GPU_reserved_bytes=5<<30,output_bytes=5<<30,log_bytes=8<<20),
        audit_limits=dict(seconds=900,OS_bytes=3<<30,output_bytes=2<<20,log_bytes=8<<20),
        expected=dict(cases=24,labels=4422,score_bytes=3477655368),
        scope='One rank255 FIT pair and original F32 norm/head injected control. No source generation/history/optimizer/full-engine or DEV/RESERVED use.'))
    print(json.dumps(dict(binding=str(a.binding),sha256=sha(a.binding),inputs=len(inputs),bytes=sum(x['bytes'] for x in inputs))),flush=True)


def relative(actual,expected):
    import numpy as np
    return math.sqrt(float(np.sum((actual-expected)**2))/max(float(np.sum(expected**2)),1e-300))


def quality(records,gates):
    from original_latent_readout_attribution import aggregate
    agg=aggregate(records)
    flags=dict(mean_KL=agg['case_KL']<=gates['mean_KL'],mean_disagreement=agg['case_disagreement']<=gates['mean_disagreement'],
        every_case_KL=all(r['KL']<=gates['case_KL'] for r in records),every_case_disagreement=all(r['disagreement_rate']<=gates['case_disagreement'] for r in records),
        domains=all(row['case_KL']<=gates['case_KL'] and row['case_disagreement']<=gates['case_disagreement'] for row in agg['domains'].values()))
    return agg,flags


def measurements(teacher,scores,rec):
    import numpy as np
    from original_latent_readout_attribution import row_metrics
    values=[row_metrics(q,p) for q,p in zip(teacher,scores)];kl,wrong,ent,uniform,delta=map(list,zip(*values))
    return dict(id=rec['id'],split=rec['split'],domain=rec['domain'],labels=len(kl),KL=float(np.mean(kl)),disagreement=sum(wrong),
        disagreement_rate=float(np.mean(wrong)),teacher_entropy=float(np.mean(ent)),uniform_KL=float(np.mean(uniform)),
        KL_per_label=kl,disagreement_per_label=wrong,entropy_per_label=ent,uniform_KL_per_label=uniform,max_metric_delta=max(delta))


def capture(a):
    import numpy as np
    import psutil
    import torch
    b=json.loads(a.binding.read_bytes());assert sha(a.binding)==a.binding_sha
    assert np.__version__=='2.4.6' and torch.__version__=='2.6.0+cu124'
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6);torch.set_num_interop_threads(1)
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.directory.mkdir(exist_ok=False);stage='FIT_statistics';current=None;completed=[];native_attempted=native_completed=real_labels=0
    artifacts={}
    def guard():
        lim=b['capture_limits'];assert time.monotonic()-start<lim['seconds']-lim['reserve_seconds'],'worker reserve'
        assert proc.memory_info().peak_wset<=lim['OS_bytes'] and not proc.children(recursive=True)
        assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'] and torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes']
        assert sum(path.stat().st_size for path in a.directory.iterdir() if path.is_file())<=lim['output_bytes']
    def persist(key,value):artifacts[key]=save(a.directory,key+'.'+('f32' if value.dtype==np.float32 else 'f64'),np.asarray(value))
    try:
        C=np.zeros((SD,SD),'f8');qbar=np.zeros(V,'f8')
        for rec in b['records']:
            current=rec['id'];m=rec['labels'];ff=decode(np.fromfile(rec['normalized']['path'],dtype='<u2').reshape(m,SD))
            qs=decode(np.fromfile(rec['logits']['path'],dtype='<u2').reshape(m,V));assert np.isfinite(ff).all() and np.isfinite(qs).all()
            C+=(ff.T@ff)/(24*m);qs-=qs.max(axis=1,keepdims=True);qq=np.exp(qs);qq/=qq.sum(axis=1,keepdims=True);qbar+=qq.sum(axis=0)/(24*m);guard()
        assert abs(float(qbar.sum())-1)<=1e-12 and np.all(qbar>=0)
        C=(C+C.T)/2;persist('covariance',C);persist('qbar',qbar)
        print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start)),flush=True)
        stage='output_metric';hf=b['fields']['lm_head.weight'];assert hf['shape']==[V,SD]
        source_head=np.memmap(Path(b['source'])/'model.safetensors',mode='r',dtype='<u2',offset=hf['offset'],shape=(V,SD))
        Gt=torch.zeros((SD,SD),device='cuda',dtype=torch.float64);mut=torch.zeros(SD,device='cuda',dtype=torch.float64);blocks=0
        for first in range(0,V,b['head_chunk']):
            last=min(first+b['head_chunk'],V);mh=torch.from_numpy(decode(source_head[first:last])*b['multiplier']).to('cuda')
            qt=torch.from_numpy(qbar[first:last].copy()).to('cuda');weighted=mh*qt.sqrt()[:,None]
            Gt+=weighted.T@weighted;mut+=mh.T@qt;blocks+=1
            del mh,qt,weighted;torch.cuda.synchronize();guard()
        G=(Gt-torch.outer(mut,mut)).cpu().numpy();mu=mut.cpu().numpy();G=(G+G.T)/2
        del Gt,mut;persist('output_metric',G);persist('head_mean',mu)
        stage='rank_pair';evalC,QC=np.linalg.eigh(C);persist('covariance_eigenvalues',evalC);persist('covariance_eigenvectors',QC)
        assert evalC[-1]>0 and evalC[0]>=-b['PSD_negative_relative']*evalC[-1]
        positive=np.maximum(evalC,0);keep=positive>positive[-1]*b['covariance_cut_relative'];assert int(keep.sum())>=K
        discarded=np.where(keep,0,positive);discarded_energy=float(discarded.sum()/positive.sum())
        assert discarded_energy<=b['discarded_covariance_energy_relative']
        root=(QC*np.sqrt(np.where(keep,positive,0)))@QC.T
        inverse=(QC*np.where(keep,1/np.sqrt(np.maximum(positive,np.finfo('f8').tiny)),0))@QC.T
        Cdiscard=(QC*discarded)@QC.T;A=root@G@root;A=(A+A.T)/2
        persist('covariance_root',root);persist('covariance_discarded',Cdiscard);persist('rank_metric',A)
        evalA,QA=np.linalg.eigh(A);persist('rank_eigenvalues',evalA);persist('rank_eigenvectors',QA)
        assert evalA[-1]>0 and evalA[0]>=-b['PSD_negative_relative']*evalA[-1]
        assert evalA[-K]>evalA[-1]*b['covariance_cut_relative']
        U=QA[:,-K:][:,::-1].copy()
        for j in range(K):
            if U[int(np.argmax(abs(U[:,j]))),j]<0:U[:,j]*=-1
        encoder=U.T@inverse;transport=root@U;decoder=np.empty((V,K),'f8')
        for first in range(0,V,b['head_chunk']):
            last=min(first+b['head_chunk'],V);mh=torch.from_numpy(decode(source_head[first:last])*b['multiplier']).to('cuda')
            decoder[first:last]=(mh@torch.from_numpy(transport).to('cuda')).cpu().numpy();del mh;guard()
        persist('rank_basis',U);persist('encoder',encoder);persist('transport',transport);persist('decoder',decoder)
        whitening=relative(encoder@C@encoder.T,np.eye(K));pair_delta=relative(encoder@root,U.T)
        expected_surrogate=float(np.sum(G*Cdiscard.T)+np.sum(evalA[:-K]));total_surrogate=float(np.sum(G*C.T))
        phis=[];maxnorm=0.
        for rec in b['records']:
            ff=decode(np.fromfile(rec['normalized']['path'],dtype='<u2').reshape(rec['labels'],SD));phi=ff@encoder.T
            assert np.isfinite(phi).all();maxnorm=max(maxnorm,float(np.sqrt(np.sum(phi*phi,axis=1)).max()));phis.append(phi);guard()
        gain=2*maxnorm/math.sqrt(D);assert math.isfinite(gain) and gain>0
        head32=np.zeros((V,D),'<f4');head32[:,:K]=decoder*(gain*math.sqrt(b['rho']**2+b['epsilon']));assert np.isfinite(head32).all()
        persist('head',head32);persist('final_norm',np.ones(D,'<f4'))
        calibration=dict(gain=gain,rho=b['rho'],max_FIT_phi_norm=maxnorm,covariance_rank=int(keep.sum()),covariance_discarded_energy_relative=discarded_energy,
            covariance_smallest=float(evalC[0]),covariance_largest=float(evalC[-1]),covariance_retained_condition=float(positive[-1]/positive[keep].min()),
            rank_smallest=float(evalA[0]),rank_largest=float(evalA[-1]),whitening_relative=whitening,encoder_root_relative=pair_delta,
            expected_surrogate=expected_surrogate,total_source_surrogate=total_surrogate,metric_blocks=blocks,decoder_fit_blocks=blocks,
            eigenvector_sign='Largest absolute coordinate positive; first coordinate breaks ties.',
            optimum_scope='Constrained covariance support; discarded metric trace plus rank-metric tail, not a global KL optimum.')
        write(a.directory/'calibration.json',calibration);print(json.dumps(dict(stage='pair_frozen_before_FIT_scores',**calibration,seconds=time.monotonic()-start)),flush=True)
        dll=ctypes.CDLL(b['native']['dll']['path']);assert dll.codec_abi(0)==D and dll.codec_abi(1)==V
        pointer=ctypes.POINTER(ctypes.c_float);dll.codec_readout.argtypes=[pointer,pointer,ctypes.c_int,pointer,pointer];dll.codec_readout.restype=ctypes.c_int
        def ptr(array):assert array.dtype==np.float32 and array.flags.c_contiguous;return array.ctypes.data_as(pointer)
        kt=torch.from_numpy(decoder).to('cuda');stage='FIT_injected_readouts';direct_surrogate=0.
        real_records=[];native_records=[];precision_records=[];max_norm=max_score_delta=0.;payload=0
        for rec,phi in zip(b['records'],phis):
            current=rec['id'];m=rec['labels'];case_start=time.monotonic();guard()
            ff=decode(np.fromfile(rec['normalized']['path'],dtype='<u2').reshape(m,SD))
            residual=ff-phi@transport.T;case_surrogate=float(np.sum((residual@G)*residual)/m);direct_surrogate+=case_surrogate/24
            u=np.empty((m,D),'f8');u[:,:K]=phi/gain
            radius=D*b['rho']**2-np.sum(u[:,:K]**2,axis=1);assert np.all(radius>=0)
            u[:,-1]=np.sqrt(radius);u32=u.astype('<f4');norm32=np.empty_like(u32);scores32=np.empty((m,V),'<f4');scores64=np.empty((m,V),'<f8')
            case_files=dict(phi=save(a.directory,current+'.phi.f64',phi),state=save(a.directory,current+'.state.f32',u32))
            with torch.inference_mode():
                for first in range(0,m,b['label_chunk']):
                    last=min(first+b['label_chunk'],m)
                    scores64[first:last]=(torch.from_numpy(phi[first:last]).to('cuda')@kt.T).cpu().numpy();real_labels+=last-first;guard()
            native_attempted+=m;assert dll.codec_readout(ptr(head32),ptr(u32),m,ptr(norm32),ptr(scores32))==0;native_completed+=m
            case_files.update(normalized=save(a.directory,current+'.normalized.f32',norm32),real_scores=save(a.directory,current+'.real.f64',scores64),
                native_scores=save(a.directory,current+'.native.f32',scores32))
            payload+=case_files['real_scores']['bytes']+case_files['native_scores']['bytes']
            assert np.isfinite(scores64).all() and np.isfinite(norm32).all()
            ideal_norm=u32.astype('f8')/np.sqrt(np.mean(u32.astype('f8')**2,axis=1,keepdims=True)+b['epsilon'])
            norm_delta=relative(norm32.astype('f8'),ideal_norm);score_delta=float(np.max(np.abs(scores32.astype('f8')-scores64)))
            max_norm=max(max_norm,norm_delta);max_score_delta=max(max_score_delta,score_delta)
            teacher=decode(np.fromfile(rec['logits']['path'],dtype='<u2').reshape(m,V))
            real=measurements(teacher,scores64,rec);native=measurements(teacher,scores32,rec);precision=measurements(scores64,scores32,rec)
            real_records.append(real);native_records.append(native);precision_records.append(precision)
            row=dict(id=current,labels=m,domain=rec['domain'],split='FIT',source_normalized=rec['normalized'],source_logits=rec['logits'],
                real=real,native=native,precision=precision,native_norm_relative=norm_delta,max_native_to_real_score_delta=score_delta,
                surrogate=case_surrogate,minimum_carrier_radius_squared=float(radius.min()),seconds=time.monotonic()-case_start,**case_files)
            case_path=a.directory/(current+'.json');write(case_path,row);completed.append(dict(**row,case_record=extent(case_path)))
            print(json.dumps(dict(stage='case_complete',id=current,completed=len(completed),real_KL=real['KL'],native_KL=native['KL'],
                real_disagreement=real['disagreement_rate'],native_disagreement=native['disagreement_rate'],seconds=time.monotonic()-start)),flush=True)
            del scores32,scores64,teacher;guard()
        assert payload==b['expected']['score_bytes'] and real_labels==native_completed==4422
        real_agg,real_flags=quality(real_records,b['source_label_gate']);native_agg,native_flags=quality(native_records,b['source_label_gate'])
        from original_latent_readout_attribution import aggregate
        precision_agg=aggregate(precision_records)
        surrogate_delta=abs(direct_surrogate-expected_surrogate)/max(abs(total_surrogate),1e-300)
        g=b['numeric_gates'];numeric_flags=dict(whitening=whitening<=g['algebra_relative'],encoder_root=pair_delta<=g['algebra_relative'],
            surrogate_identity=surrogate_delta<=g['surrogate_relative'],native_norm=max_norm<=g['native_norm_relative'],
            native_scores=max_score_delta<=g['native_score_absolute'],native_mean_KL_to_real=precision_agg['case_KL']<=g['native_mean_KL_to_real'],
            native_every_case_disagreement_to_real=all(row['disagreement_rate']<=g['native_case_disagreement_to_real'] for row in precision_records))
        decision='PAIRED_CODEC_NUMERIC_FAIL' if not all(numeric_flags.values()) else (
            'PAIRED_CODEC_FIT_PASS_PENDING_DEV' if all(real_flags.values()) and all(native_flags.values()) else 'PAIRED_CODEC_FIT_QUALITY_FAIL')
        result=dict(schema='PAIRED_OUTPUT_CODEC_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,decision=decision,artifacts=artifacts,
            calibration=calibration,records=completed,real_aggregate=real_agg,native_aggregate=native_agg,precision_aggregate=precision_agg,
            real_quality_flags=real_flags,native_quality_flags=native_flags,numeric_flags=numeric_flags,
            direct_surrogate=direct_surrogate,surrogate_relative_delta=surrogate_delta,max_native_norm_relative=max_norm,max_native_to_real_score_delta=max_score_delta,
            FIT_labels=4422,linear_rank_pairs_fit=1,source_model_instances=0,source_generations=0,source_history_forwards=0,source_LM_head_calls=0,
            real_compact_head_labels=real_labels,native_final_readout_labels=native_completed,native_final_readout_labels_attempted=native_attempted,
            native_full_engine_calls=0,optimizer_updates=0,DEV_queries=0,reserved_queries=0,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),worker_OS_peak_snapshot=proc.memory_info().peak_wset,
            seconds=time.monotonic()-start,quality_admission=False,speed_admission=False,useful_large_n_admission=False,physical_DRAM_bytes=None,
            scope='FIT-only injected rank255 paired output test and original F32 norm/head; causal core/experts and held-out chatbot still missing.')
        write(a.out,result);print(json.dumps(dict(stage='complete',decision=decision,numeric_flags=numeric_flags,seconds=time.monotonic()-start)),flush=True)
    except BaseException as error:
        write(a.directory/'first_fault.json',dict(stage=stage,id=current,error=repr(error),completed_ids=[row['id'] for row in completed],
            native_final_readout_labels_attempted=native_attempted,native_final_readout_labels_completed=native_completed,real_compact_head_labels=real_labels,
            seconds=time.monotonic()-start));raise


def audit_metrics(teacher,student,stored,tolerance):
    import numpy as np
    values=[]
    for q,p in zip(teacher,student):
        q=q.astype('f8');p=p.astype('f8');assert np.isfinite(q).all() and np.isfinite(p).all()
        q-=q.max();p-=p.max();lq=q-float(np.logaddexp.reduce(q));lp=p-float(np.logaddexp.reduce(p));prob=np.exp(lq)
        loss=float(np.dot(prob,lq-lp));entropy=-float(np.dot(prob,lq))
        values.append((loss,int(np.argmax(q)!=np.argmax(p)),entropy,math.log(V)-entropy))
    keys=('KL_per_label','disagreement_per_label','entropy_per_label','uniform_KL_per_label');worst=0.
    for column,key in enumerate(keys):
        actual=np.asarray([row[column] for row in values]);expected=np.asarray(stored[key]);assert actual.shape==expected.shape
        worst=max(worst,float(np.max(abs(actual-expected))))
    assert worst<=tolerance
    for key,column in (('KL',0),('disagreement_rate',1),('teacher_entropy',2),('uniform_KL',3)):
        worst=max(worst,abs(float(np.mean([row[column] for row in values]))-stored[key]))
    assert stored['disagreement']==sum(row[1] for row in values) and worst<=tolerance
    return worst


def audit(a):
    import numpy as np
    import psutil
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));start=time.monotonic()
    b=json.loads(a.binding.read_bytes());r=json.loads(a.source_result.read_bytes());t=json.loads(a.source_result.with_suffix('.terminal.json').read_bytes());g=b['numeric_gates']
    assert t['exit_code']==0 and t['error'] is None and t['result']==extent(a.source_result)
    assert r['binding_sha256']==sha(a.binding)==t['binding_sha256'] and t['inputs_before_after_exact']
    for item in b['inputs']+t['outputs']:assert extent(item['path'])==item
    def guard():
        assert time.monotonic()-start<b['audit_limits']['seconds']-15 and proc.memory_info().peak_wset<=b['audit_limits']['OS_bytes']
    def array(item,shape,kind='f8'):
        dtype=np.dtype(kind);assert item['shape']==list(shape) and item['dtype']==str(dtype) and item['bytes']==math.prod(shape)*dtype.itemsize
        assert extent(item['path'])=={key:item[key] for key in ('path','bytes','sha256')}
        return np.memmap(item['path'],mode='r',dtype=dtype,shape=shape)
    ar=r['artifacts'];C=array(ar['covariance'],(SD,SD));qbar=array(ar['qbar'],(V,));G=array(ar['output_metric'],(SD,SD));mu=array(ar['head_mean'],(SD,))
    evalC=array(ar['covariance_eigenvalues'],(SD,));QC=array(ar['covariance_eigenvectors'],(SD,SD))
    root=array(ar['covariance_root'],(SD,SD));discard=array(ar['covariance_discarded'],(SD,SD));A=array(ar['rank_metric'],(SD,SD))
    evalA=array(ar['rank_eigenvalues'],(SD,));QA=array(ar['rank_eigenvectors'],(SD,SD));U=array(ar['rank_basis'],(SD,K))
    encoder=array(ar['encoder'],(K,SD));transport=array(ar['transport'],(SD,K));decoder=array(ar['decoder'],(V,K));head32=array(ar['head'],(V,D),'f4')
    head64=head32.astype('f8')
    assert np.array_equal(array(ar['final_norm'],(D,),'f4'),np.ones(D,'f4'))
    cal=r['calibration'];directory=Path(ar['encoder']['path']).parent;assert json.loads((directory/'calibration.json').read_bytes())==cal
    eigen_checks=dict(covariance_residual=relative(C@QC,QC*evalC),covariance_orthogonality=relative(QC.T@QC,np.eye(SD)),
        rank_residual=relative(A@QA,QA*evalA),rank_orthogonality=relative(QA.T@QA,np.eye(SD)),
        root_support=relative(root@root,C-discard),metric_pullback=relative(root@G@root,A),transport=relative(root@U,transport))
    assert all(value<=g['algebra_relative'] for value in eigen_checks.values());guard()
    positive=np.maximum(evalC,0);keep=positive>positive[-1]*b['covariance_cut_relative']
    assert int(keep.sum())==cal['covariance_rank'] and evalC[0]>=-b['PSD_negative_relative']*evalC[-1]
    canonical=QA[:,-K:][:,::-1].copy()
    for j in range(K):
        if canonical[int(np.argmax(abs(canonical[:,j]))),j]<0:canonical[:,j]*=-1
    assert np.array_equal(U,canonical)
    inverse_coeff=np.where(keep,1/np.sqrt(np.maximum(positive,np.finfo('f8').tiny)),0)
    eigen_checks['encoder_definition']=relative(encoder@QC,(U.T@QC)*inverse_coeff)
    assert eigen_checks['encoder_definition']<=g['algebra_relative']
    assert relative(discard,(QC*np.where(keep,0,positive))@QC.T)<=g['algebra_relative'] or not np.any(positive[~keep])
    hf=b['fields']['lm_head.weight'];source=np.memmap(Path(b['source'])/'model.safetensors',mode='r',dtype='<u2',offset=hf['offset'],shape=(V,SD))
    decoder_delta=0.
    for first in range(0,V,b['head_chunk']):
        last=min(first+b['head_chunk'],V);expected=(decode(source[first:last])*b['multiplier'])@transport
        decoder_delta=max(decoder_delta,float(np.max(abs(expected-decoder[first:last]))))
        assert np.array_equal(head32[first:last,:K],(decoder[first:last]*(cal['gain']*math.sqrt(b['rho']**2+b['epsilon']))).astype('f4'))
        assert np.all(head32[first:last,-1]==0);guard()
    assert decoder_delta<=g['real_score_absolute']
    coordinates=[(i,(37*i+17)%SD) for i in range(32)];Cw=np.zeros(32,'f8');qcheck=np.zeros(V,'f8')
    for rec in b['records']:
        m=rec['labels'];ff=decode(np.fromfile(rec['normalized']['path'],dtype='<u2').reshape(m,SD))
        for n,(i,j) in enumerate(coordinates):Cw[n]+=math.fsum(float(x)*float(y) for x,y in zip(ff[:,i],ff[:,j]))/(24*m)
        qs=decode(np.fromfile(rec['logits']['path'],dtype='<u2').reshape(m,V));qs-=qs.max(axis=1,keepdims=True)
        qs-=np.logaddexp.reduce(qs,axis=1)[:,None];qcheck+=np.exp(qs).sum(axis=0)/(24*m);guard()
    qbar_delta=float(np.max(abs(qbar-qcheck)));cov_delta=max(abs(C[i,j]-Cw[n]) for n,(i,j) in enumerate(coordinates))
    assert max(qbar_delta,cov_delta)<=g['stat_absolute']
    metric_delta=mean_delta=0.
    for i,j in coordinates:
        wi=decode(source[:,i])*b['multiplier'];wj=decode(source[:,j])*b['multiplier']
        mi=math.fsum(float(q)*float(w) for q,w in zip(qcheck,wi));mj=math.fsum(float(q)*float(w) for q,w in zip(qcheck,wj))
        actual=math.fsum(float(q)*float(x)*float(y) for q,x,y in zip(qcheck,wi,wj))-mi*mj
        metric_delta=max(metric_delta,abs(actual-float(G[i,j])));mean_delta=max(mean_delta,abs(mi-float(mu[i])),abs(mj-float(mu[j])))
    assert max(metric_delta,mean_delta)<=g['stat_absolute'];guard()
    assert [row['id'] for row in r['records']]==[rec['id'] for rec in b['records']]
    counts=0;max_phi=maxnorm=max_score_delta=max_metric_delta=0.;real_delta=native_linear_delta=direct_surrogate=0.;rows=[]
    for row,rec in zip(r['records'],b['records']):
        stored=json.loads(Path(row['case_record']['path']).read_bytes());assert {key:value for key,value in row.items() if key!='case_record'}==stored
        assert extent(row['case_record']['path'])==row['case_record'];m=rec['labels'];counts+=m
        assert row['source_normalized']==rec['normalized'] and row['source_logits']==rec['logits'] and row['split']=='FIT'
        phi=array(row['phi'],(m,K));u32=array(row['state'],(m,D),'f4');norm32=array(row['normalized'],(m,D),'f4')
        p64=array(row['real_scores'],(m,V));p32=array(row['native_scores'],(m,V),'f4')
        ff=decode(np.fromfile(rec['normalized']['path'],dtype='<u2').reshape(m,SD));assert relative(ff@encoder.T,phi)<=g['algebra_relative']
        max_phi=max(max_phi,float(np.sqrt(np.sum(phi*phi,axis=1)).max()))
        expected_state=np.empty((m,D),'f8');expected_state[:,:K]=phi/cal['gain']
        radius=D*b['rho']**2-np.sum(expected_state[:,:K]**2,axis=1);assert np.all(radius>=0);expected_state[:,-1]=np.sqrt(radius)
        assert np.array_equal(u32,expected_state.astype('f4'))
        ideal_norm=u32.astype('f8')/np.sqrt(np.mean(u32.astype('f8')**2,axis=1,keepdims=True)+b['epsilon'])
        norm_delta=relative(norm32.astype('f8'),ideal_norm);maxnorm=max(maxnorm,norm_delta)
        assert abs(norm_delta-row['native_norm_relative'])<=g['stat_absolute']
        for first in range(0,m,b['label_chunk']):
            last=min(first+b['label_chunk'],m);recon=phi[first:last]@decoder.T
            real_delta=max(real_delta,float(np.max(abs(recon-p64[first:last]))))
            native_recon=norm32[first:last].astype('f8')@head64.T
            native_linear_delta=max(native_linear_delta,float(np.max(abs(native_recon-p32[first:last]))));guard()
        score_delta=float(np.max(abs(p32.astype('f8')-p64)));max_score_delta=max(max_score_delta,score_delta)
        assert abs(score_delta-row['max_native_to_real_score_delta'])<=g['stat_absolute']
        teacher=decode(np.fromfile(rec['logits']['path'],dtype='<u2').reshape(m,V))
        deltas=[audit_metrics(teacher,p64,row['real'],g['metric_absolute']),audit_metrics(teacher,p32,row['native'],g['metric_absolute']),
            audit_metrics(p64,p32,row['precision'],g['metric_absolute'])];max_metric_delta=max(max_metric_delta,*deltas)
        residual=ff-phi@transport.T;surrogate=float(np.sum((residual@G)*residual)/m);direct_surrogate+=surrogate/24
        assert abs(surrogate-row['surrogate'])<=g['stat_absolute'];rows.append(dict(id=row['id'],labels=m,metric_absolute_delta=max(deltas)))
        guard();print(json.dumps(dict(stage='stored_case_audited',id=row['id'],completed=len(rows),seconds=time.monotonic()-start)),flush=True)
    assert real_delta<=g['real_score_absolute'] and native_linear_delta<=g['native_score_absolute']
    assert abs(cal['gain']-2*max_phi/math.sqrt(D))<=g['stat_absolute']
    whitening=relative(encoder@C@encoder.T,np.eye(K));pair_delta=relative(encoder@root,U.T)
    expected_surrogate=float(np.sum(G*discard.T)+np.sum(evalA[:-K]));total_surrogate=float(np.sum(G*C.T))
    surrogate_delta=abs(direct_surrogate-expected_surrogate)/max(abs(total_surrogate),1e-300)
    for name,value in (('whitening_relative',whitening),('encoder_root_relative',pair_delta),('expected_surrogate',expected_surrogate),('total_source_surrogate',total_surrogate)):
        assert abs(value-cal[name])<=g['stat_absolute']
    assert abs(direct_surrogate-r['direct_surrogate'])<=g['stat_absolute'] and abs(surrogate_delta-r['surrogate_relative_delta'])<=g['stat_absolute']
    real_agg,real_flags=quality([row['real'] for row in r['records']],b['source_label_gate']);native_agg,native_flags=quality([row['native'] for row in r['records']],b['source_label_gate'])
    from original_latent_readout_attribution import aggregate
    precision=aggregate([row['precision'] for row in r['records']]);assert real_agg==r['real_aggregate'] and native_agg==r['native_aggregate'] and precision==r['precision_aggregate']
    numeric_flags=dict(whitening=whitening<=g['algebra_relative'],encoder_root=pair_delta<=g['algebra_relative'],surrogate_identity=surrogate_delta<=g['surrogate_relative'],
        native_norm=maxnorm<=g['native_norm_relative'],native_scores=max_score_delta<=g['native_score_absolute'],
        native_mean_KL_to_real=precision['case_KL']<=g['native_mean_KL_to_real'],
        native_every_case_disagreement_to_real=all(row['precision']['disagreement_rate']<=g['native_case_disagreement_to_real'] for row in r['records']))
    decision='PAIRED_CODEC_NUMERIC_FAIL' if not all(numeric_flags.values()) else (
        'PAIRED_CODEC_FIT_PASS_PENDING_DEV' if all(real_flags.values()) and all(native_flags.values()) else 'PAIRED_CODEC_FIT_QUALITY_FAIL')
    assert numeric_flags==r['numeric_flags'] and real_flags==r['real_quality_flags'] and native_flags==r['native_quality_flags'] and decision==r['decision']
    assert counts==r['FIT_labels']==r['real_compact_head_labels']==r['native_final_readout_labels']==r['native_final_readout_labels_attempted']==4422
    assert r['linear_rank_pairs_fit']==1 and all(r[key]==0 for key in ('source_model_instances','source_generations','source_history_forwards','source_LM_head_calls','native_full_engine_calls','optimizer_updates','DEV_queries','reserved_queries'))
    assert t['elapsed_seconds']<=b['capture_limits']['seconds'] and t['worker_OS_peak_through_exit']+t['launcher_OS_peak_snapshot']<=b['capture_limits']['OS_bytes']
    assert r['GPU_allocated_peak']<=b['capture_limits']['GPU_allocated_bytes'] and r['GPU_reserved_peak']<=b['capture_limits']['GPU_reserved_bytes']
    native_source=Path(b['native']['source']['path']).read_text();engine=Path(b['native']['engine']['path']).read_text()
    for item in b['native']['functions']:
        for text in (native_source,engine):
            match=re.search(r'^static[^\n]*\b'+item['name']+r'\(',text,re.M);assert match
            pos=text.index('{',match.start());depth=1;end=pos+1
            while depth:depth+=(text[end]=='{')-(text[end]=='}');end+=1
            assert hashlib.sha256(text[match.start():end].encode()).hexdigest()==item['original_body_sha256']
    write(a.out,dict(schema='PAIRED_OUTPUT_CODEC_AUDIT_V1',result=extent(a.source_result),binding=extent(a.binding),decision=decision,numeric_flags=numeric_flags,
        real_quality_flags=real_flags,native_quality_flags=native_flags,complete_input_output_hashes=True,original_four_native_bodies_exact=True,
        eigen_checks=eigen_checks,decoder_full_reconstruction_absolute_delta=decoder_delta,covariance_scalar_witnesses=32,output_metric_scalar_witnesses=32,
        covariance_witness_absolute_delta=cov_delta,qbar_all_v_absolute_delta=qbar_delta,output_metric_witness_absolute_delta=metric_delta,head_mean_witness_absolute_delta=mean_delta,
        real_scores_full_reconstruction_absolute_delta=real_delta,native_scores_F64_linear_reference_absolute_delta=native_linear_delta,max_metric_absolute_delta=max_metric_delta,
        labels=counts,records=rows,stored_linear_head_labels_reconstructed=counts*2,full_decoder_factorizations_verified=1,
        source_generations=0,source_history_forwards=0,native_binary_calls=0,optimizer_updates=0,new_pairs_fit=0,
        seconds=time.monotonic()-start,scope='Complete stored numeric/metric/decision/hash audit; CPU F64 arithmetic checks both stored heads, no native/history replay or refit.'))
    print(json.dumps(dict(audit=str(a.out),decision=decision,seconds=time.monotonic()-start)),flush=True)


def launch(a):
    import psutil
    proc=psutil.Process();proc.cpu_affinity([11]);own={proc.pid,*(p.pid for p in proc.parents())}
    for p in psutil.process_iter(['name','cmdline']):
        name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
        if p.pid in own:continue
        if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():continue
        assert not name.startswith(('python','native','original_engine','clang','lld','packed_original')),('overlap',p.pid,name)
    terminal=a.out.with_suffix('.terminal.json');log=a.out.with_suffix('.worker.log')
    assert not any(path.exists() for path in (a.out,terminal,log)) and (a.audit or not a.directory.exists())
    assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes());lim=b['audit_limits' if a.audit else 'capture_limits']
    began=time.monotonic();reader=memory_reader();peak=0
    for item in b['inputs']:assert extent(item['path'])==item
    assert time.monotonic()-began<lim['seconds'],'prehash deadline'
    extra=[]
    if a.audit:extra=[extent(a.source_result),extent(a.source_result.with_suffix('.terminal.json'))]
    command=[sys.executable,'-I','-S','-B','-X','utf8',str(Path(__file__).resolve()),'--audit-worker' if a.audit else '--capture-worker',
        '--binding',str(a.binding.resolve()),'--binding-sha',a.binding_sha,'--freeze',a.freeze,'--out',str(a.out.resolve())]
    command += ['--source-result',str(a.source_result.resolve())] if a.audit else ['--directory',str(a.directory.resolve())]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1' if a.audit else '6',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',
             CUDA_VISIBLE_DEVICES='' if a.audit else '0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    receipt=dict(freeze=a.freeze,binding_sha256=a.binding_sha,command=command,launcher_pid=proc.pid,limits=lim)
    with log.open('xb') as output:
        child=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,env=env,creationflags=8)
        receipt.update(worker_pid=child.pid,worker_creation_time=psutil.Process(child.pid).create_time());fault=None
        try:
            while child.poll() is None:
                peak=max(peak,reader(child))
                assert time.monotonic()-began<=lim['seconds'],'deadline'
                assert peak+proc.memory_info().peak_wset<=lim['OS_bytes'],'OS cap'
                assert log.stat().st_size<=lim['log_bytes'],'log cap'
                try:assert not psutil.Process(child.pid).children(recursive=True),'unexpected child'
                except psutil.NoSuchProcess:pass
                time.sleep(.1)
            assert child.returncode==0,log.read_text(errors='replace')[-5000:]
            for item in b['inputs']+extra:assert extent(item['path'])==item
            outputs=[] if a.audit else [extent(path) for path in sorted(a.directory.iterdir()) if path.is_file()]
            assert sum(item['bytes'] for item in outputs)+a.out.stat().st_size<=lim['output_bytes']
            if not a.audit:
                r=json.loads(a.out.read_bytes());assert r['GPU_allocated_peak']<=lim['GPU_allocated_bytes'] and r['GPU_reserved_peak']<=lim['GPU_reserved_bytes']
            assert time.monotonic()-began<=lim['seconds'],'seal deadline'
            receipt.update(inputs_before_after_exact=True,result=extent(a.out),outputs=outputs,extra_inputs=extra)
        except BaseException as error:
            fault=repr(error)
            if child.poll() is None:child.kill();child.wait()
        finally:
            peak=max(peak,reader(child));receipt.update(exit_code=child.returncode,error=fault,elapsed_seconds=time.monotonic()-began,
                worker_OS_peak_through_exit=peak,launcher_OS_peak_snapshot=proc.memory_info().peak_wset,log=extent(log))
            write(terminal,receipt)
    assert child.returncode==0 and fault is None,fault
    print(json.dumps(dict(terminal=str(terminal),seconds=receipt['elapsed_seconds'],result_sha256=receipt['result']['sha256'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('prepare','bind','audit','capture-worker','audit-worker'):p.add_argument('--'+key,action='store_true')
    for key in ('binding','out','directory','source-result','protocol','native-preparation'):p.add_argument('--'+key,type=Path)
    for key in ('binding-sha','freeze'):p.add_argument('--'+key)
    a=p.parse_args()
    if a.prepare:prepare(a)
    elif a.bind:bind(a)
    elif a.capture_worker:capture(a)
    elif a.audit_worker:audit(a)
    else:launch(a)
