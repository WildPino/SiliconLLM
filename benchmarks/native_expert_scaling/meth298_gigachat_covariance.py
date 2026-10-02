#!/usr/bin/env python3
"""Source-bound GigaChat full routed-input covariance screen at fixed rank192."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time

import numpy as np
import psutil
import scipy.linalg
from safetensors import safe_open
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
OUT=ROOT/'results/native_expert_scaling'
BASE=Path('C:/Users/giosa/AppData/Local/Temp/siliconllm-llama-mtp-5b335f4')
BUILD=BASE/'build-cpu'
TOOLCHAIN=Path('C:/Users/giosa/AppData/Local/Microsoft/WinGet/Packages/MartinStorsjo.LLVM-MinGW.MSVCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/llvm-mingw-20251216-msvcrt-x86_64')
COMPILER=TOOLCHAIN/'bin/c++.exe'
HOOK=ROOT/'benchmarks/native_expert_scaling/meth298_capture_hook.h'
BASE_CPP=BASE/'tools/imatrix/imatrix.cpp'
BASE_SHA='49c71316c66717d8daccc90bf16c17bf4589239903eabe386d273d657777b9f3'
MODEL=ROOT/'benchmarks/donor_adaptation/density/results/strat01_gigachat_source_binding_v1/GigaChat3.1-10B-A1.8B-source-bf16.gguf'
MODEL_SHA='fabc8056f57e230ae9e6aadceb45f4abf9e5d7031fbfe8eca2671d871ef35d47'
SOURCE=ROOT/'benchmarks/donor_adaptation/density/results/strat01_gigachat_source_189fff27'
PRIOR=DOC/'meth180_gigachat_expert_rank_result.json'
PRIOR_SHA='ecc9a3432a25a4a7fc0f2881c421947aeb097b7a4cf32a746ebcdefa10081b8d'
DIAG=DOC/'meth181_gigachat_diagonal_rank_result.json'
DIAG_SHA='5b226744cde42404a4811ff32740a6b744328db0d697d3acafaf8880c0f3d7b0'
DOMAINS={
 'fit':('meth05_gigachat_calib_interleaved_ids.txt','3d611152b4a145f2219a34e7ad5bc2e8929a9a600321272e9e258b92924c250a',106,
        'meth05_gigachat_bf16_interleaved_106chunks.gguf','ef5de87fc4fa0d1382ed2988e59e9472fb1d7fb2c9bed0e8edcced6ff9bd4f9c'),
 'test':('meth06_cyrillic_source_ids.txt','7f5333d1d52a5c94ad0026ff0f38cc022a3aeeecf1b884085b56abe8c795c524',125,
         'meth06_cyrillic_bf16_125chunks.gguf','5c529f9009f0f241dfc50d0a1f598b0e183c137693a951f04963dc729a344d08')}
LIBS=['common/libllama-common.a','src/libllama.a','ggml/src/ggml.a',
      'ggml/src/ggml-cpu.a','ggml/src/ggml-base.a','common/libllama-common-base.a',
      'vendor/cpp-httplib/libcpp-httplib.a']
RANK=192

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    assert not Path(path).exists(),str(path)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
def binding(path):return {'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path)}

def build():
    assert digest(BASE_CPP)==BASE_SHA
    original=BASE_CPP.read_text(encoding='utf-8')
    replacements=[
      ('#if defined(_MSC_VER)', '#include "'+HOOK.as_posix()+'"\n\n#if defined(_MSC_VER)'),
      ('                e.counts[ex]++;',
       '                GGML_ASSERT(src1->type == GGML_TYPE_F32);\n'
       '                m298_capture(wname, int(ex), int(idx), int(row), int(ne0), x);\n'
       '                e.counts[ex]++;'),
      ('            if (llama_decode(ctx, batch)) {',
       '            m298_chunk = i; m298_batch = j;\n'
       '            if (llama_decode(ctx, batch)) {'),
      ('    g_collector.save_imatrix();\n\n    LOG("\\n");',
       '    g_collector.save_imatrix();\n    m298_finish();\n\n    LOG("\\n");')]
    modified=original
    for old,new in replacements:
        assert modified.count(old)==1,(old,modified.count(old))
        modified=modified.replace(old,new)
    reverse=modified
    for old,new in reversed(replacements):reverse=reverse.replace(new,old)
    assert reverse==original
    cpp=OUT/'meth298_imatrix_capture.cpp';exe=OUT/'meth298_imatrix_capture.exe'
    assert not cpp.exists() and not exe.exists()
    cpp.write_text(modified,encoding='utf-8',newline='\n')
    inc=[BASE/'common',BASE/'vendor',BASE/'include',BASE/'ggml/include']
    cmd=[str(COMPILER),'-O3','-DNDEBUG','-std=c++17','-D_WIN32_WINNT=0x0A00',
         '-DGGML_USE_CPU','-DLLAMA_SUBPROCESS','-D_CRT_SECURE_NO_WARNINGS','-pthread',
         *['-I'+str(p) for p in inc],str(cpp),'-o',str(exe),
         *[str(BUILD/p) for p in LIBS[:5]],'-pthread',
         str(TOOLCHAIN/'x86_64-w64-mingw32/lib/libomp.dll.a'),
         *[str(BUILD/p) for p in LIBS[5:]],'-lws2_32','-lpthread',
         '-lkernel32','-luser32','-lgdi32','-lwinspool','-lshell32','-lole32',
         '-loleaut32','-luuid','-lcomdlg32','-ladvapi32']
    start=time.monotonic();env=os.environ.copy()
    env['PATH']=str(TOOLCHAIN/'bin')+os.pathsep+env.get('PATH','')
    p=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=300)
    (OUT/'meth298_capture_build.stdout.log').write_text(p.stdout,encoding='utf-8')
    (OUT/'meth298_capture_build.stderr.log').write_text(p.stderr,encoding='utf-8')
    assert p.returncode==0,p.stderr[-5000:]
    # Pin EVERY existing header potentially included in this isolated compile.
    headers={str(p.relative_to(BASE)):binding(p) for d in inc for p in d.rglob('*')
             if p.is_file() and p.suffix in ('.h','.hpp','.inl')}
    result={'experiment':'METH-298-capture-apparatus-build','source_revision':
      subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip(),
      'base':binding(BASE_CPP),'hook':binding(HOOK),'controller':binding(Path(__file__)),
      'generated_source':binding(cpp),'executable':binding(exe),
      'compiler':binding(COMPILER),'libraries':{p:binding(BUILD/p) for p in LIBS},
      'openmp_library':binding(TOOLCHAIN/'x86_64-w64-mingw32/lib/libomp.dll.a'),
      'headers':headers,'command':cmd,'seconds':time.monotonic()-start,
      'exact_original_after_reversing_four_capture_insertions':True}
    write(DOC/'meth298_capture_build.json',result)
    print(json.dumps({'build_seconds':result['seconds'],'headers':len(headers),
                      'executable_sha256':result['executable']['sha256']}),flush=True)

def validate_build():
    manifest=read(DOC/'meth298_capture_build.json')
    def check(b):assert binding(Path(b['path']))==b,b['path']
    for k in ('base','hook','controller','generated_source','executable','compiler','openmp_library'):check(manifest[k])
    for b in (*manifest['libraries'].values(),*manifest['headers'].values()):check(b)
    for path in (Path(__file__),HOOK,DOC/'METH_298_GIGACHAT_FULL_COVARIANCE_PROTOCOL_20261002.md',DOC/'meth298_capture_build.json'):
        rel=path.relative_to(ROOT).as_posix()
        blob=subprocess.check_output(['git','-c','core.autocrlf=false','cat-file','--filters','--path='+rel,'HEAD:'+rel],cwd=ROOT)
        assert blob==path.read_bytes(),str(path)+' not frozen'
    return manifest

def collect(domain):
    start=time.monotonic();stage='bindings';p=None
    destination=DOC/f'meth298_{domain}_capture_result.json'
    assert not destination.exists()
    cap=OUT/f'meth298_{domain}_inputs.bin';matrix=OUT/f'meth298_{domain}_imatrix.gguf'
    assert not cap.exists() and not matrix.exists()
    try:
        build_info=validate_build()
        assert digest(MODEL)==MODEL_SHA
        ids_name,ids_sha,chunks,old_name,old_sha=DOMAINS[domain]
        ids=OUT/ids_name
        assert digest(ids)==ids_sha
        values=[int(v) for v in ids.read_text(encoding='utf-8').split()]
        assert len(values)>=chunks*512 and all(0<=v<128256 for v in values)
        assert digest(OUT/old_name)==old_sha
        env=os.environ.copy()
        env['PATH']=str(TOOLCHAIN/'bin')+os.pathsep+env.get('PATH','')
        env['SILICON_IMATRIX_SOURCE_IDS']=str(ids)
        env['SILICON_M298_CAPTURE']=str(cap)
        cmd=[build_info['executable']['path'],'-m',str(MODEL),'-f',str(ids),'-o',str(matrix),
             '--chunks',str(chunks),'--ctx-size','512','--batch-size','128',
             '--threads','6','--threads-batch','6','--gpu-layers','0',
             '--no-ppl','--output-frequency','1000']
        peak=0;child_start=time.monotonic();stage='source_graph_capture'
        with (OUT/f'meth298_{domain}.stdout.log').open('w',encoding='utf-8') as out,(OUT/f'meth298_{domain}.stderr.log').open('w',encoding='utf-8') as err:
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=out,stderr=err)
            proc=psutil.Process(p.pid)
            while p.poll() is None:
                try:peak=max(peak,proc.memory_info().rss)
                except psutil.NoSuchProcess:pass
                elapsed=time.monotonic()-child_start
                if elapsed>55*60 or peak>50*(1<<30):
                    p.terminate();p.wait(timeout=20)
                    raise RuntimeError('capture wall/RSS stop')
                time.sleep(2)
        assert p.returncode==0,'capture exit '+str(p.returncode)
        stage='capture_audit'
        arrays,coords,record_count=parse_capture(cap,chunks)
        # Same pinned GGUF, split, context and input order: old diagonal moments
        # are an independent closure guard, not a fit target or a relaxed gate.
        from benchmarks.native_expert_scaling.meth181_gigachat_diagonal_rank import Imatrix,ORGANS
        old=Imatrix(OUT/old_name,old_sha,chunks)
        new=Imatrix(matrix,digest(matrix),chunks)
        organ_names=list(ORGANS)
        audits=[]
        for (layer,expert,organ),x in arrays.items():
            w=x.shape[1];ov,on=old.moment(layer,expert,organ_names[organ],w)
            nv,nn=new.moment(layer,expert,organ_names[organ],w)
            assert len(x)==on==nn and len(x)>=384
            # Replicate original F32 row-order accumulation (not a BLAS sum).
            acc=np.zeros(w,dtype=np.float32)
            for row in x:acc+=row*row
            cv=acc.astype(np.float64)/len(x)
            assert np.array_equal(cv,nv),'capture/new-imatrix mismatch'
            rel=float(np.max(np.abs(nv-ov)/ov));assert rel<=1e-4,(layer,expert,organ,rel)
            audits.append({'layer':layer,'expert':expert,'organ':organ,'count':len(x),
                           'max_relative_old_moment_error':rel})
        for layer in (1,13,25):
            for expert in (0,32,63):
                assert coords[layer,expert,0]==coords[layer,expert,1]
                assert np.array_equal(arrays[layer,expert,0],arrays[layer,expert,1])
        result={'experiment':'METH-298-'+domain+'-real-routed-input-capture',
          'source_model':binding(MODEL),'ids':binding(ids),'chunks':chunks,
          'source_ids_used':chunks*512,'trailing_source_ids_unused':len(values)-chunks*512,
          'build_manifest_sha256':digest(DOC/'meth298_capture_build.json'),
          'capture':binding(cap),'new_imatrix':binding(matrix),'old_imatrix':binding(OUT/old_name),
          'records':record_count,'audits':audits,'gates':{'all27_present_counts384':True,
          'source_diagonal_count_and_moment_closure':True,'captured_diagonal_exact':True,
          'gate_up_inputs_exact':True,'coordinates_unique_finite_footer_eof':True},
          'command':cmd,'child_exit_code':p.returncode,'child_seconds':time.monotonic()-child_start,
          'peak_child_rss_bytes':peak,'total_seconds':time.monotonic()-start,
          'decision':'capture_qualified_for_frozen_full_covariance_screen',
          'scope':'BF16 GGUF source graph,not source-HF bit-equivalence or full-model quality. No candidate factors or spectra inspected in collection.'}
        write(destination,result)
        print(json.dumps({'domain':domain,'records':record_count,'seconds':result['total_seconds'],
                          'peak_rss':peak,'result_sha256':digest(destination)}),flush=True)
    except BaseException as e:
        if p and p.poll() is None:p.terminate();p.wait(timeout=20)
        write(destination.with_suffix('.failure.json'),{'stage':stage,'error':str(e),
          'elapsed_seconds':time.monotonic()-start,'capture':binding(cap) if cap.exists() else None})
        raise

def parse_capture(path,chunks):
    parts={(l,e,o):[] for l in (1,13,25) for e in (0,32,63) for o in range(3)}
    coords={k:[] for k in parts};seen={k:set() for k in parts};count=0
    with path.open('rb') as f:
        assert f.read(8)==b'M298ACT1'
        while True:
            raw=f.read(32);assert len(raw)==32,'missing capture footer'
            l,o,e,c,b,s,t,w=struct.unpack('<8I',raw)
            if l==0xffffffff:
                assert (o,e,c,b,s,t,w)==(0,)*7
                footer=f.read(8);assert len(footer)==8 and struct.unpack('<Q',footer)[0]==count
                assert not f.read(1);break
            key=l,e,o;assert key in parts
            assert c<chunks and b<4 and s<4 and t<128 and w==(1280 if o==2 else 1536)
            coordinate=(c,b,t);assert coordinate not in seen[key];seen[key].add(coordinate)
            raw=f.read(w*4);assert len(raw)==w*4
            x=np.frombuffer(raw,dtype='<f4').copy();assert np.isfinite(x).all()
            parts[key].append(x);coords[key].append((c,b,s,t));count+=1
    assert all(parts.values())
    return {k:np.stack(v) for k,v in parts.items()},coords,count

def factors(matrix,sqrt_v):
    u,s,vh=scipy.linalg.svd(np.asarray(matrix*sqrt_v[None,:],dtype=np.float64),
      full_matrices=False,check_finite=False,lapack_driver='gesdd')
    return ((u[:,:RANK]*s[:RANK])@vh[:RANK])/sqrt_v[None,:]

def score():
    start=time.monotonic();rows=[];stage='bindings'
    destination=DOC/'meth298_gigachat_full_covariance_result.json';assert not destination.exists()
    try:
        validate_build();assert digest(PRIOR)==PRIOR_SHA and digest(DIAG)==DIAG_SHA
        prior=read(PRIOR);diag=read(DIAG)
        assert digest(SOURCE/'model.safetensors.index.json')==prior['source_index_sha256']
        assert digest(SOURCE/'config.json')==prior['source_config_sha256']
        domain_data={};cap_bind={}
        for d,(_,_,chunks,_,_) in DOMAINS.items():
            r=read(DOC/f'meth298_{d}_capture_result.json')
            assert all(r['gates'].values()) and r['decision']=='capture_qualified_for_frozen_full_covariance_screen'
            path=Path(r['capture']['path']);assert binding(path)==r['capture']
            domain_data[d]=parse_capture(path,chunks)[0];cap_bind[d]=r['capture']
        for name,b in prior['source_shards'].items():assert binding(SOURCE/name)['sha256']==b['sha256']
        stage='fixed_rank192_factor_screen'
        with threadpool_limits(limits=6):
            for ref in prior['rows']:
                l,e,organ=ref['layer'],ref['expert'],ref['projection'];o=('gate_proj','up_proj','down_proj').index(organ)
                with safe_open(SOURCE/ref['source_shard'],framework='pt',device='cpu') as source:
                    w=source.get_tensor(f'model.layers.{l}.mlp.experts.{e}.{organ}.weight')
                assert w.dtype==torch.bfloat16 and hashlib.sha256(w.view(torch.uint16).numpy().tobytes()).hexdigest()==ref['tensor_bf16_sha256']
                w=w.float().numpy().astype(np.float64);assert list(w.shape)==ref['shape']
                x=domain_data['fit'][l,e,o].astype(np.float64);xt=domain_data['test'][l,e,o].astype(np.float64)
                y=x@w.T;yt=xt@w.T
                # Uncentered second moment: no centering/intercept/new bias.
                cov=y.T@y
                eig,u=scipy.linalg.eigh(cov,check_finite=False,driver='evd')
                assert eig[0]>=-1e-8*eig[-1] and eig[-1]>0
                q=u[:,-RANK:];full=q@(q.T@w)
                ordinary=factors(w,np.ones(w.shape[1]))
                sqrt_v=np.sqrt(np.mean(x*x,axis=0));assert (sqrt_v>0).all()
                diagonal=factors(w,sqrt_v)
                def energy(a,xx,yy):
                    error=xx@(w-a).T
                    return 1-float(np.sum(error*error)/np.sum(yy*yy))
                fit=energy(full,x,y);test=energy(full,xt,yt)
                eigenergy=float(np.sum(eig[-RANK:])/np.sum(eig))
                assert abs(fit-eigenergy)<=1e-8
                ordinary_test=energy(ordinary,xt,yt);diagonal_test=energy(diagonal,xt,yt)
                te=scipy.linalg.eigvalsh(yt.T@yt,check_finite=False,driver='evd')
                oracle=float(np.sum(te[-RANK:])/np.sum(te))
                assert test<=oracle+1e-8 and ordinary_test<=oracle+1e-8 and diagonal_test<=oracle+1e-8
                row={'layer':l,'expert':e,'projection':organ,'source_tensor_sha256':ref['tensor_bf16_sha256'],
                  'fit_count':len(x),'test_count':len(xt),'fit_full_covariance_energy':fit,
                  'fit_eigenvalue_energy':eigenergy,'test_full_covariance_energy':test,
                  'test_ordinary_energy':ordinary_test,'test_diagonal_energy':diagonal_test,
                  'test_oracle_energy':oracle,'full_minus_ordinary':test-ordinary_test,
                  'full_minus_diagonal':test-diagonal_test,
                  'rank192_factor_elements':RANK*sum(w.shape),'source_elements':w.size}
                rows.append(row)
                elapsed=time.monotonic()-start;rss=psutil.Process().memory_info().rss
                assert elapsed<=20*60 and rss<=12*(1<<30),'score resource stop'
                print(json.dumps({'completed':[l,e,organ],'test_full_energy':test,'seconds':elapsed}),flush=True)
        assert len(rows)==27
        energies=[r['test_full_covariance_energy'] for r in rows]
        gates={'median_test_energy95':statistics.median(energies)>=.95,
          'every_test_energy90':min(energies)>=.90,
          'no_ordinary_loss_over_half_point':min(r['full_minus_ordinary'] for r in rows)>=-.005,
          'no_diagonal_loss_over_half_point':min(r['full_minus_diagonal'] for r in rows)>=-.005}
        result={'experiment':'METH-298-GigaChat10B-full-routed-input-covariance-rank192',
          'source_revision':prior['source_revision'],'prior180_sha256':PRIOR_SHA,'prior181_sha256':DIAG_SHA,
          'capture_bindings':cap_bind,'rank':RANK,'rows':rows,'gates':gates,
          'summary':{'min_test_energy':min(energies),'median_test_energy':statistics.median(energies),
                     'max_test_energy':max(energies),'count':len(rows)},
          'seconds':time.monotonic()-start,'end_rss_bytes':psutil.Process().memory_info().rss,
          'controller_sha256':digest(Path(__file__)),
          'decision':'eligible_for_factor_precision_nonlinear_quality_cost_gate' if all(gates.values()) else 'reject_fixed_full_covariance_rank192_export',
          'scope':'27 real selected expert projections; full uncentered input covariance; distinct calibration domains previously used181. Exact FP64 factors,not quantized/native/full-model quality,useful additional n or50tok/s proof.'}
        write(destination,result);print(json.dumps({'gates':gates,'summary':result['summary'],'result_sha256':digest(destination)}),flush=True)
    except BaseException as e:
        write(destination.with_suffix('.failure.json'),{'stage':stage,'error':str(e),
          'partial_rows':rows,'seconds':time.monotonic()-start});raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('build','fit','test','score'))
    args=ap.parse_args()
    if args.mode=='build':build()
    elif args.mode=='score':score()
    else:collect(args.mode)

if __name__=='__main__':main()
