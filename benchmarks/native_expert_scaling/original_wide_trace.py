"""One diagnostic replay of actual26, with persisted original-operator witnesses.

No optimizer update, source call, initialization or speed admission. Observer
identity against retained head bytes is a prerequisite for interpreting traces.
"""
import argparse,gc,json,math,os,re,struct,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';sys.path.insert(0,str(B))
from original_packed_capacity import SITE,extent,sha,write,raw
from original_falcon_whole_recovery import check_inputs,memory_reader
sys.path.insert(0,str(SITE))

def layout(np):
    fields=[(k,'<f4',(256,)) for k in ('pre','core_norm','core','post_core','ff_norm','moe','post_moe')]
    fields += [('score','<f4',(1152,)),('prob','<f4',(1152,)),('ids','<i4',(8,)),('mass','<f4',(8,)),
        ('xq','i1',(256,)),('sa','<f4'),('gint','<i4',(8,128)),('uint','<i4',(8,128)),('dint','<i4',(8,256)),
        ('g','<f4',(8,128)),('u','<f4',(8,128)),('h','<f4',(8,128)),('hq','i1',(8,128)),('sh','<f4',(8,))]
    return np.dtype(fields)

def instrument(header,np):
    dt=layout(np);decl=[]
    for name in dt.names:
        field=dt.fields[name][0];base,shape=field.subdtype if field.subdtype else (field,())
        ctype={'f':'float','i':'int8_t' if base.itemsize==1 else 'int32_t'}[base.kind]
        decl.append(ctype+' '+name+''.join(f'[{n}]' for n in shape)+';')
    prefix='typedef struct {'+' '.join(decl)+'} Trace;\n'
    prefix+=f'_Static_assert(sizeof(Trace)=={dt.itemsize},"trace ABI");static Trace tr[L];\n'
    pairs=[
        ('    float mx=-1e30f; for(int e=0;e<E;e++) if(rp[e]>mx)mx=rp[e];','memcpy(tr[l].score,rp,E*4);'),
        ('    topk_sel(rp,E,KTOP,idx,wv);','memcpy(tr[l].prob,rp,E*4);'),
        ('    memcpy(actual_ids[l],idx,sizeof(idx)); memcpy(actual_mass[l],wv,sizeof(wv));','memcpy(tr[l].ids,idx,sizeof(idx));memcpy(tr[l].mass,wv,sizeof(wv));'),
        ('    float sa=0; if(mlp_lut){ sa=quant_i8(xn,D,g_xq); build_lut_t3(g_xq,TUP,g_lut); }','tr[l].sa=sa;memcpy(tr[l].xq,g_xq,D);'),
        ('            MV_ROWS(egate_cd[l],g_lut,g_S,e*HID_E,HID_E,MPAD_GU,TUP);','memcpy(tr[l].gint[j],g_S,HID_E*4);'),
        ('            MV_ROWS(eup_cd[l],g_lut,g_S,e*HID_E,HID_E,MPAD_GU,TUP);','memcpy(tr[l].uint[j],g_S,HID_E*4);memcpy(tr[l].g[j],gg,HID_E*4);'),
        ('            float sh=quant_i8(he,HID_E,g_hq); build_lut_t3(g_hq,TDE,g_lutd);','memcpy(tr[l].h[j],he,HID_E*4);tr[l].sh[j]=sh;memcpy(tr[l].hq[j],g_hq,HID_E);'),
        ('            MV_ROWS(eWd_cd[l]+(size_t)e*NPLANES(TDE)*MPAD_D,g_lutd,g_Sd,0,D,MPAD_D,TDE);','memcpy(tr[l].dint[j],g_Sd,D*4);'),
        ('    for(int l=0;l<L;l++){','memcpy(tr[l].pre,x,D*4);'),
        ('        rmsnorm(x, is_swa[l]?swa.norm:ssm[l].norm, xn);','memcpy(tr[l].core_norm,xn,D*4);'),
        ('        rmsnorm(x, mlp_n2[l], xn);','memcpy(tr[l].core,tmp,D*4);memcpy(tr[l].post_core,x,D*4);memcpy(tr[l].ff_norm,xn,D*4);'),
        ('        for(int i=0;i<D;i++) x[i]+=tmp[i];','memcpy(tr[l].moe,tmp,D*4);memcpy(tr[l].post_moe,x,D*4);')]
    observed=header
    for anchor,code in pairs:
        assert observed.count(anchor)==1,anchor
        # Router score/prob copies must be BEFORE the anchor that mutates them.
        marker='/*TRACE_BEGIN*/'+code+'/*TRACE_END*/'
        if anchor.startswith(('    float mx=','    topk_sel')):observed=observed.replace(anchor,marker+anchor)
        else:observed=observed.replace(anchor,anchor+marker)
    anchor='float uu=(float)g_S[i]*sa*eup_sc[l][e*HID_E+i];'
    assert observed.count(anchor)==1
    observed=observed.replace(anchor,anchor+'/*TRACE_BEGIN*/tr[l].u[j][i]=uu;/*TRACE_END*/')
    stripped=re.sub(r'/\*TRACE_BEGIN\*/.*?/\*TRACE_END\*/','',observed,flags=re.S)
    assert stripped==header,'observer stripping must recover all original body bytes'
    return prefix,observed,dict(original_sha256=__import__('hashlib').sha256(header.encode()).hexdigest(),observer_count=len(pairs)+1,strip_identity=True)

def bind(a):
    import numpy as np,numpy._core._multiarray_umath as ext,psutil,shutil
    parent=DOC/'original_wide_bridge_result_20261009.json';pr=json.loads(parent.read_bytes())
    terminal=parent.with_suffix('.terminal.json');pt=json.loads(terminal.read_bytes())
    assert pt['exit_code']==0 and pt['result_sha256']==sha(parent)
    prior=DOC/'original_wide_bridge_binding_20261009.json';old=json.loads(prior.read_bytes())
    nativeold=ROOT/'results/native_expert_scaling/original_native_envelope_finish_20261009'
    ns=ROOT/'results/native_expert_scaling/original_wide_bridge_20261009'
    compiler=Path(json.loads((DOC/'original_native_envelope_binding_20261009.json').read_bytes())['compiler'])
    files=[Path(__file__),B/'original_wide_trace_main.c',B/'original_wide_learner.py',B/'original_tensor_learner.py',B/'original_falcon_learner.py',
        B/'original_packed_capacity.py',B/'original_falcon_whole_recovery.py',B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py',
        DOC/'ORIGINAL_WIDE_TRACE_PROTOCOL_20261009.md',parent,terminal,prior,Path(pr['checkpoint']['path']),Path(pr['packed']['path']),
        nativeold/'native_envelope.c',nativeold/'original_packed_bodies.h',ns/'after.f32',ns/'after.routes',ns/'GPU_after.f32',ns/'requests.u32',
        ROOT/'benchmarks/phase60/engine.c',Path(sys.executable),compiler,SITE/'numpy/__init__.py',Path(ext.__file__),SITE/'psutil/__init__.py']
    files += [SITE/'torch'/p for p in ('__init__.py','_C.cp312-win_amd64.pyd','lib/torch_cpu.dll','lib/torch_cuda.dll','cuda/__init__.py')]
    files += [p for p in compiler.parent.iterdir() if p.name in ('clang-21.exe','ld.lld.exe','lld.exe','libclang-cpp.dll','libLLVM.dll')]
    assert shutil.disk_usage(ROOT).free>=4<<30
    write(a.out,dict(schema='ORIGINAL_WIDE_TRACE_BINDING_V1',python=str(Path(sys.executable).resolve()),worker_path=str(Path(__file__).resolve()),
        parent=extent(parent),checkpoint=pr['checkpoint'],packed=pr['packed'],GPU_heads=extent(ns/'GPU_after.f32'),native_heads=extent(ns/'after.f32'),native_routes=extent(ns/'after.routes'),
        queries=extent(ns/'requests.u32'),wrapper=extent(nativeold/'native_envelope.c'),header=extent(nativeold/'original_packed_bodies.h'),compiler=str(compiler.resolve()),
        sequence=old['sequences'][1],trace_bytes=1507*6*layout(np).itemsize,criteria=dict(observer_head_bytes_exact=True,observer_native_routes_exact=True,relative_RMS=1e-4),
        limits=dict(seconds=900,reserve_seconds=90,OS_bytes=20<<30,GPU_allocated_bytes=6<<30,GPU_reserved_bytes=7<<30,output_bytes=2<<30),
        inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in files)],scope='One full-FIT diagnostic replay actual26;no update/source/T4/speed/quality admission. Principal binaries only.'))
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(files),trace_bytes=1507*6*layout(np).itemsize)),flush=True)

def launch(a):
    import psutil
    start=time.monotonic();assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes());check_inputs(b)
    proc=psutil.Process();proc.cpu_affinity([11]);own={proc.pid,*(p.pid for p in proc.parents())}
    for p in psutil.process_iter(['name','cmdline']):
        name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
        if p.pid in own:continue
        if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():continue
        assert not name.startswith(('python','gcc','clang','engine','packed_original','native_')),('overlap',p.pid,name)
    assert not a.directory.exists() and not a.out.exists()
    log=a.out.with_suffix('.worker.log');term=a.out.with_suffix('.terminal.json');reader=memory_reader();peak=0;worker=None;seen={}
    record=dict(freeze=a.freeze,binding_sha256=a.binding_sha,launcher_pid=proc.pid)
    try:
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        argv=[b['python'],'-I','-S','-B','-X','utf8',b['worker_path'],'--worker','--binding',str(a.binding.resolve()),'--binding-sha',a.binding_sha,'--freeze',a.freeze,'--directory',str(a.directory.resolve()),'--out',str(a.out.resolve())]
        with log.open('xb') as f:
            worker=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,env=env,creationflags=8)
            record.update(worker_pid=worker.pid,worker_creation_time=psutil.Process(worker.pid).create_time(),command=argv);offset=0
            while worker.poll() is None:
                peak=max(peak,reader(worker));assert time.monotonic()-start<=b['limits']['seconds'] and peak+proc.memory_info().peak_wset<=b['limits']['OS_bytes']
                assert log.stat().st_size<=4<<20
                try:
                    for c in psutil.Process(worker.pid).children(recursive=True):
                        try:seen[(c.pid,c.create_time())]=c;name=c.name().lower();exe=Path(c.exe().removeprefix('\\\\?\\')).resolve()
                        except psutil.NoSuchProcess:continue
                        assert name in ('clang.exe','clang-21.exe','ld.lld.exe','lld.exe','native_trace.exe','conhost.exe'),name
                        if name=='native_trace.exe':assert exe==a.directory.resolve()/name
                        elif name=='conhost.exe':assert exe==Path('C:/Windows/System32/conhost.exe').resolve()
                        else:assert exe.parent==Path(b['compiler']).parent
                except psutil.NoSuchProcess:pass
                with log.open('rb') as s:
                    s.seek(offset);chunk=s.read();end=chunk.rfind(b'\n')+1
                    if end:print(chunk[:end].decode('utf8',errors='replace'),end='',flush=True);offset+=end
                time.sleep(.1)
        peak=max(peak,reader(worker));record.update(exit_code=worker.returncode,worker_OS_peak_through_exit=peak,launcher_OS_peak_snapshot=proc.memory_info().peak_wset)
        assert worker.returncode==0,log.read_text(errors='replace')[-4000:]
        check_inputs(b);r=json.loads(a.out.read_bytes());assert r['observer_identity'] and r['source_calls']==r['optimizer_updates']==0
        assert peak+r['max_direct_child_OS_peak']+proc.memory_info().peak_wset<=b['limits']['OS_bytes']
        outputs=[extent(p) for p in sorted(a.directory.iterdir()) if p.is_file()];assert sum(p['bytes'] for p in outputs)<=b['limits']['output_bytes']
        record.update(result_sha256=sha(a.out),elapsed_seconds=time.monotonic()-start,output_files=outputs,resource_gates=True,decision=r['decision']);write(term,record)
        print(json.dumps(dict(terminal=str(term),exit_code=worker.returncode,seconds=record['elapsed_seconds'])),flush=True)
    except BaseException as e:
        for (pid,created),c in reversed(list(seen.items())):
            try:
                if c.is_running() and c.create_time()==created:c.kill()
            except psutil.NoSuchProcess:pass
        if worker is not None:
            if worker.poll() is None:worker.kill();worker.wait()
            peak=max(peak,reader(worker));record.update(exit_code=worker.returncode,worker_OS_peak_through_exit=peak)
        record.update(fault=repr(e),elapsed_seconds=time.monotonic()-start);write(a.out.with_suffix('.launcher_failure.json'),record);raise

def worker(a):
    import numpy as np,psutil,torch
    from original_wide_learner import SourceLearner,tensor,source
    from torch.nn import functional as F
    start=time.monotonic();assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes());a.directory.mkdir(exist_ok=False)
    assert (np.__version__,psutil.__version__,torch.__version__)==('2.4.6','7.2.2','2.6.0+cu124')
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
    reader=memory_reader();children=[];max_child=0;stage='startup'
    def guard():
        lim=b['limits'];assert time.monotonic()-start<=lim['seconds']-lim['reserve_seconds']
        assert proc.memory_info().peak_wset+max_child<=lim['OS_bytes']
        assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'] and torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes']
        assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=lim['output_bytes']
    def event(**kw):guard();print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,**kw)),flush=True)
    def run(argv,name):
        nonlocal max_child
        t=time.monotonic();peak=0
        with (a.directory/(name+'.log')).open('xb') as f:
            child=subprocess.Popen([str(x) for x in argv],stdout=f,stderr=subprocess.STDOUT,creationflags=8);created=psutil.Process(child.pid).create_time()
            try:
                while child.poll() is None:peak=max(peak,reader(child));max_child=max(max_child,peak);guard();time.sleep(.05)
                peak=max(peak,reader(child));max_child=max(max_child,peak)
            except BaseException:
                if child.poll() is None:child.kill();child.wait()
                peak=max(peak,reader(child));max_child=max(max_child,peak);raise
            finally:
                rec=dict(command=[str(x) for x in argv],pid=child.pid,creation_time=created,exit_code=child.returncode,held_OS_peak=peak,seconds=time.monotonic()-t)
                children.append(rec);write(a.directory/(name+'.receipt.json'),rec)
        assert child.returncode==0,(name,(a.directory/(name+'.log')).read_text(errors='replace'));event(child=name,receipt=rec)
    try:
        stage='compile_observer';dt=layout(np);prefix,header,body_identity=instrument(Path(b['header']['path']).read_text(),np)
        wrapper=Path(b['wrapper']['path']).read_text();wrapper=wrapper[:wrapper.index('/* Consumer only.')]
        anchor='#include "original_packed_bodies.h"';assert wrapper.count(anchor)==1
        wrapper=wrapper.replace(anchor,prefix+'#include "trace_bodies.h"')
        raw(a.directory/'trace_bodies.h',header.encode());raw(a.directory/'trace_native.c',(wrapper+(B/'original_wide_trace_main.c').read_text()).encode())
        write(a.directory/'trace_layout.json',dict(dtype=dt.descr,record_bytes=dt.itemsize,shape=[1507,6],body_identity=body_identity))
        exe=a.directory/'native_trace.exe';run([b['compiler'],'-O3','-mavx2','-mfma','-march=znver2','-DDN=1024','-DDTR=48',a.directory/'trace_native.c','-I',a.directory,'-o',exe,'-lm'],'compile')
        stage='GPU_actual26';state=torch.load(b['checkpoint']['path'],map_location='cpu',weights_only=True,mmap=True)
        assert state['schema']=='ORIGINAL_WIDE_STATE_V1' and state['updates']==26
        model=SourceLearner(device='cuda');model.load_state_dict(state['model']);del state;gc.collect()
        obs=np.zeros((1507,6),dtype=dt);context={};last_event=0.
        def array(x):return x.detach().cpu().numpy()
        def prehook(l):
            def hook(module,args):context.update(l=l,phase=0);obs['pre'][:,l]=array(args[0][0])
            return hook
        def posthook(l):
            def hook(module,args,out):obs['post_moe'][:,l]=array(out[0])
            return hook
        def bankpre(l):
            def hook(module,args):context.update(l=l,phase=0);obs['ff_norm'][:,l]=array(args[0][0])
            return hook
        def bankpost(l):
            def hook(module,args,out):
                obs['moe'][:,l]=array(out[0]);ri,rm=module.last_routes
                assert np.array_equal(obs['ids'][:,l],array(ri)) and np.array_equal(obs['mass'][:,l],array(rm))
            return hook
        def routerhook(l):
            def hook(module,args,out):
                prob=out.softmax(-1);ids=torch.argsort(prob,dim=-1,descending=True,stable=True)[:,:8];mass=prob.gather(-1,ids);mass=mass/mass.sum(-1,keepdim=True)
                obs['score'][:,l]=array(out);obs['prob'][:,l]=array(prob);obs['ids'][:,l]=array(ids);obs['mass'][:,l]=array(mass)
                context['positions']={e:p for e,p in source.groups(ids)}
            return hook
        for l,block in enumerate(model.layers):
            block.register_forward_pre_hook(prehook(l));block.register_forward_hook(posthook(l));block.bank.register_forward_pre_hook(bankpre(l));block.bank.register_forward_hook(bankpost(l));block.bank.router.register_forward_hook(routerhook(l))
            oldcore=block.core
            def core(x,l=l,oldcore=oldcore,w=block.organs):
                obs['core_norm'][:,l]=array(tensor.norm(x,w['norm'])[0]);y=oldcore(x)
                obs['core'][:,l]=array(y[0]);obs['post_core'][:,l]=array((x+y)[0]);return y
            block.core=core
            def progress(mode,count,total,l=l):
                nonlocal last_event
                guard()
                if time.monotonic()-last_event>15:event(operation=mode,site=l,completed=count,total=total);last_event=time.monotonic()
            block.bank.progress=progress
        def traced_linear(x,master,grad=False):
            assert not grad
            q,s=source.quant_weight(master);q=q.to(x.device);s=s.to(x.device);xe,qx,scale=source.aq63(x)
            integer=qx@q.T;exact=(integer*scale)*s
            l=context['l'];phase=context['phase']%3;context['phase']+=1;e=master.storage_offset()//master.numel()
            positions=context['positions'][e].numpy();token=positions//8;rank=positions%8
            if phase==0:
                obs['xq'][token,l]=array(qx).astype('i1');obs['sa'][token,l]=array(scale[:,0]);obs['gint'][token,l,rank]=array(integer).astype('<i4');obs['g'][token,l,rank]=array(exact)
            elif phase==1:obs['uint'][token,l,rank]=array(integer).astype('<i4');obs['u'][token,l,rank]=array(exact)
            else:
                obs['h'][token,l,rank]=array(x);obs['hq'][token,l,rank]=array(qx).astype('i1');obs['sh'][token,l,rank]=array(scale[:,0]);obs['dint'][token,l,rank]=array(integer).astype('<i4')
            return exact,None
        source.linear=traced_linear
        with torch.no_grad():heads=model(torch.tensor([b['sequence']['student_input_ids']],device='cuda'))[0];torch.cuda.synchronize()
        raw(a.directory/'GPU_trace.bin',obs.tobytes());raw(a.directory/'GPU_heads.f32',array(heads).astype('<f4').tobytes());del heads,model;gc.collect();torch.cuda.empty_cache()
        assert sha(a.directory/'GPU_heads.f32')==b['GPU_heads']['sha256'],'observer GPU head identity'
        event(GPU_head_bit_identity=True,trace_bytes=obs.nbytes)
        stage='native_observation';run([exe,b['packed']['path'],b['queries']['path'],a.directory/'native_trace.bin',a.directory/'native_heads.f32'],'native')
        stage='stored_adjudication';native=np.memmap(a.directory/'native_trace.bin',dtype=dt,mode='r',shape=(1507,6))
        assert (a.directory/'native_trace.bin').stat().st_size==b['trace_bytes']
        oldheads=np.memmap(b['native_heads']['path'],dtype='<f4',mode='r',shape=(3352,65537));newheads=np.memmap(a.directory/'native_heads.f32',dtype='<f4',mode='r',shape=(1507,65537))
        for t in range(0,1507,32):assert np.array_equal(newheads[t:t+32].view('<u4'),oldheads[357+t:357+min(t+32,1507)].view('<u4')),'observer native head identity'
        rd=np.dtype([('ids','<i4',(8,)),('mass','<f4',(8,))]);oldroutes=np.memmap(b['native_routes']['path'],dtype=rd,mode='r',shape=(3352,6))[357:1864]
        assert np.array_equal(native['ids'],oldroutes['ids']) and np.array_equal(native['mass'].view('<u4'),oldroutes['mass'].view('<u4'))
        float_metrics={}
        for name in ('pre','core_norm','core','post_core','ff_norm','moe','post_moe','score','prob','g','u','h'):
            x=obs[name].astype('f8').reshape(1507,6,-1);y=native[name].astype('f8').reshape(1507,6,-1);assert np.isfinite(x).all() and np.isfinite(y).all()
            rel=np.sqrt(np.sum((x-y)**2,-1)/np.maximum(np.sum(y*y,-1),1e-24));bad=np.argwhere(rel>1e-4)
            float_metrics[name]=dict(worst=float(rel.max()),failed_token_sites=len(bad),first=bad[0].tolist() if len(bad) else None)
        def differences(name):
            diff=obs[name]!=native[name];pair=diff.reshape(1507,6,-1).any(-1);locations=np.argwhere(pair)
            return dict(coordinates=int(diff.sum()),token_sites=int(pair.sum()),locations=locations.tolist(),first_coordinate=np.argwhere(diff)[0].tolist() if diff.any() else None)
        discrete={name:differences(name) for name in ('ids','xq','hq','gint','uint','dint')}
        route_set=np.any(np.sort(obs['ids'],axis=-1)!=np.sort(native['ids'],axis=-1),axis=-1)
        same_route=np.all(obs['ids']==native['ids'],axis=-1);same_input=np.all(obs['xq']==native['xq'],axis=-1)
        ints={name:int(np.any(obs[name]!=native[name],axis=(-1,-2))[same_route&same_input].sum()) for name in ('gint','uint')}
        mass_delta=float(np.max(np.abs(obs['mass'].astype('f8')-native['mass'].astype('f8'))))
        witnesses=[]
        for t,l in np.argwhere(np.any(obs['xq']!=native['xq'],axis=-1))[:12]:
            for c in np.flatnonzero(obs['xq'][t,l]!=native['xq'][t,l])[:4]:
                witnesses.append(dict(position=int(t),site=int(l),coordinate=int(c),GPU_input=float(obs['ff_norm'][t,l,c]),native_input=float(native['ff_norm'][t,l,c]),
                    GPU_scale=float(obs['sa'][t,l]),native_scale=float(native['sa'][t,l]),GPU_code=int(obs['xq'][t,l,c]),native_code=int(native['xq'][t,l,c]),
                    GPU_scaled=float(np.float32(obs['ff_norm'][t,l,c]*np.float32(1/obs['sa'][t,l]))),native_scaled=float(np.float32(native['ff_norm'][t,l,c]*np.float32(1/native['sa'][t,l])))))
        detail=dict(float_metrics=float_metrics,discrete=discrete,route_set_mismatch_locations=np.argwhere(route_set).tolist(),ordered_route_mismatch_locations=discrete['ids']['locations'],
            conditional_same_route_same_input_integer_mismatch_sites=ints,mass_max_delta=mass_delta,first_AQ_witnesses=witnesses,
            scope='Per-operation observations on each runtime trajectory;integer comparisons are conditional on matching ordered routes/input codes. No forced-route or counterfactual inference.')
        write(a.directory/'trace_adjudication.json',detail)
        result=dict(schema='ORIGINAL_WIDE_TRACE_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,decision='ACTUAL26_OBSERVERS_IDENTICAL_DIVERGENCE_LOCALIZED',
            observer_identity=True,body_identity=body_identity,source_calls=0,optimizer_updates=0,trace=detail,children=children,
            worker_OS_peak_snapshot=proc.memory_info().peak_wset,max_direct_child_OS_peak=max_child,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.monotonic()-start,scope='Diagnostic original math witnesses;no quality/speed/useful-n/DRAM admission.')
        write(a.out,result);stage='complete';event(decision=result['decision'],ordered_ID_mismatch_calls=discrete['ids']['token_sites'],set_ID_mismatch_calls=int(route_set.sum()),AQ_input_mismatch_sites=discrete['xq']['token_sites'],integer_condition=ints)
    except BaseException as e:
        write(a.directory/'first_fault.json',dict(stage=stage,error=repr(e),children=children,elapsed_seconds=time.monotonic()-start,
            worker_OS_peak_snapshot=proc.memory_info().peak_wset,max_direct_child_OS_peak=max_child,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--bind',action='store_true');m.add_argument('--launch',action='store_true');m.add_argument('--worker',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.bind:bind(a)
    elif a.launch:launch(a)
    else:worker(a)
