"""Frozen384 recurrent channel latents with full stateless source skip.

No new scan, model forward, fit or native call. F64 controlled operator arms,
explicit BF16 input/output rounding; original F32 assay is a precision control.
"""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; B=ROOT/'benchmarks/native_expert_scaling'; DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'; SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
from native_head_gpu_preflight import gpu_metadata
START=time.monotonic(); CLOCK=time.perf_counter()


def prepare(a):
    old_path=DOC/'original_falcon_balanced_heads_result_20261009.json'; old=json.loads(old_path.read_bytes())
    parent_path=DOC/'original_falcon_balanced_heads_binding_20261009.json'; parent=json.loads(parent_path.read_bytes())
    capture=json.loads(Path(parent['capture']).read_bytes())
    gb=json.loads((DOC/'native_head_gpu_preflight_binding_20261012.json').read_bytes())
    paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',B/'native_head_gpu_preflight.py',
           DOC/'RECURRENT_ONLY_CHANNEL_SKIP_PROTOCOL_20261012.md',old_path,parent_path,Path(parent['capture']),Path(parent['source_weights']),Path(parent['source_config'])]
    for rec in capture['records']:
        for site in parent['sites']:
            paths += [Path(rec['sites'][site]['fields'][k]['path']) for k in ['x','y','gate','output']]
    paths += [Path(s['basis']['path']) for s in old['selections']]
    paths += [Path(m['latent']['path']) for r in old['records'] for m in r['measurements'].values()]
    runtime=[e for e in gb['inputs'] if e['path'].startswith(str(Path(sys.executable).parent)) or e['path'].startswith(str(SITE.resolve())) or e['path'].startswith('C:\\Windows\\System32')]
    inputs=list({e['path']:e for e in runtime+[extent(p) for p in paths]}.values())
    b=dict(schema='RECURRENT_ONLY_CHANNEL_SKIP_BINDING_V1',python=gb['python'],gpu=gb['gpu'],versions=gb['versions'],parent=parent,old=old,capture=capture,
           numeric=dict(source_output_relative=1e-4,old_control_metric_absolute=1e-4,precision_bridge_relative=1e-3,algebra_relative=1e-10,audit_output_relative=1e-4,audit_metric_absolute=1e-4,audit_energy_relative=1e-8),
           criteria={k:parent['criteria'][k] for k in ['DEV_output_mean','DEV_case_worst','DEV_domain_mean_worst','DEV_centered_mean']},
           limits=dict(seconds=900,audit_seconds=1200,OS_bytes=6<<30,GPU_allocated_bytes=2<<30,GPU_reserved_bytes=3<<30,output_bytes=3<<30),inputs=inputs,
           scope='All48 source-injected histories/sites0,12,23/two frozen384 maps and latents. F64 controlled recurrent/skip decomposition and jointly normalized BF16 outputs;full source generators/gates retained. No new scans/fit/forward/native/chatbot/n/50 admission.')
    assert gpu_metadata()==b['gpu']; write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.perf_counter()-CLOCK)),flush=True)


def aggregate(rows,b):
    aggregates={};domains={};gates={}
    for site in b['parent']['sites']:
        sk=str(site);aggregates[sk]={};domains[sk]={}
        for split in ['FIT','DEV']:
            sr=[r for r in rows if r['site']==site and r['split']==split];assert len(sr)==24
            aggregates[sk][split]={};domains[sk][split]={}
            for layout in b['parent']['layouts']:
                aggregates[sk][split][layout]={};domains[sk][split][layout]={}
                for arm in ['projected_skip','full_skip']:
                    aggregates[sk][split][layout][arm]={};domains[sk][split][layout][arm]={}
                    for mode in b['parent']['modes']:
                        keys=sr[0]['arms'][layout][arm][mode]
                        aa={k:dict(mean=sum(r['arms'][layout][arm][mode][k] for r in sr)/24,worst=max(r['arms'][layout][arm][mode][k] for r in sr)) for k in keys}
                        dr={d:{k:dict(mean=sum(r['arms'][layout][arm][mode][k] for r in sr if r['domain']==d)/2,worst=max(r['arms'][layout][arm][mode][k] for r in sr if r['domain']==d)) for k in keys} for d in sorted({r['domain'] for r in sr})}
                        aggregates[sk][split][layout][arm][mode]=aa;domains[sk][split][layout][arm][mode]=dr
        gates[sk]={}
        for layout in b['parent']['layouts']:
            gates[sk][layout]={}
            for arm in ['projected_skip','full_skip']:
                gates[sk][layout][arm]={}
                for mode in b['parent']['modes']:
                    aa=aggregates[sk]['DEV'][layout][arm][mode];dr=domains[sk]['DEV'][layout][arm][mode];c=b['criteria']
                    gates[sk][layout][arm][mode]=dict(mean=aa['output_relative_RMS']['mean']<=c['DEV_output_mean'],case_worst=aa['output_relative_RMS']['worst']<=c['DEV_case_worst'],domain_mean_worst=max(d['output_relative_RMS']['mean'] for d in dr.values())<=c['DEV_domain_mean_worst'],centered_mean=aa['centered_output_relative_RMS']['mean']<=c['DEV_centered_mean'])
    return aggregates,domains,gates


def run(a):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np,psutil,torch
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and gpu_metadata()==b['gpu']
    assert not a.directory.exists() and not a.out.exists()
    assert dict(torch=torch.__version__,numpy=np.__version__,psutil=psutil.__version__,cuda=torch.version.cuda)==b['versions']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and all(os.environ.get(k)=='6' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();torch.cuda.reset_peak_memory_stats();stream=None;rows=[];stage='startup'
    receipt=dict(schema='RECURRENT_ONLY_CHANNEL_SKIP_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        lim=b['limits'];assert time.monotonic()-START<=lim['seconds'] and proc.memory_info().peak_wset<=lim['OS_bytes'] and not proc.children(recursive=True)
        assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'] and torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes']
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir())<=lim['output_bytes']
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    def raw(e):
        if e['dtype']=='BF16':z=(np.fromfile(e['path'],'<u2').astype('<u4')<<16).view('<f4')
        else:assert e['dtype']=='F32';z=np.fromfile(e['path'],'<f4')
        z=z.reshape(e['shape']);assert np.isfinite(z).all();return torch.from_numpy(z).to('cuda')
    def error(x,y):return float(torch.linalg.vector_norm(x.double()-y.double())/torch.linalg.vector_norm(y.double()).clamp_min(1e-12))
    def metrics(op,ref,label):return dict(output_relative_RMS=error(op,ref),centered_output_relative_RMS=error(op.double()-op.double().mean(0),ref.double()-ref.double().mean(0)),label_output_relative_RMS=error(op[label],ref[label]))
    try:
        check();a.directory.mkdir();get,stream=tensor_reader(b['parent']['source_weights'],np);eps=json.loads(Path(b['parent']['source_config']).read_bytes())['rms_norm_eps']
        oldrows={(r['site'],r['id']):r for r in b['old']['records']}
        with torch.inference_mode():
            for packet in b['old']['selections']:
                site=packet['site'];prefix=f'model.layers.{site}.mamba.';dd=torch.from_numpy(get(prefix+'D')).float().cuda();weight=torch.from_numpy(get(prefix+'norm.weight')).float().cuda();ow=torch.from_numpy(get(prefix+'out_proj.weight')).float().cuda()
                with np.load(packet['basis']['path'],allow_pickle=False) as z:maps={name:([torch.from_numpy(z[f'{name}_V_{h}'].astype('<f4')).cuda() for h in range(48)],[torch.from_numpy(z[f'{name}_W_{h}'].astype('<f4')).cuda() for h in range(48)]) for name in b['parent']['layouts']}
                for rec in b['capture']['records']:
                    stage=f'{site}/{rec["id"]}';guard();f=rec['sites'][site]['fields'];n=rec['history'];oldrec=oldrows[(site,rec['id'])]
                    xx=raw(f['x']).reshape(n,48,64);yy=raw(f['y']);gate=raw(f['gate']);ref=raw(f['output']);label=torch.tensor(rec['positions'],device='cuda');sg32=torch.nn.functional.silu(gate);u32=yy*sg32
                    inv32=torch.rsqrt(u32.square().mean(-1,keepdim=True)+eps);sourceop=torch.nn.functional.linear((u32*inv32*weight).bfloat16(),ow.bfloat16())
                    source_error=error(sourceop,ref);assert source_error<=b['numeric']['source_output_relative']
                    sg=gate.double()*torch.sigmoid(gate.double());source_inv=torch.rsqrt((yy.double()*sg).square().mean(-1,keepdim=True)+eps)
                    arms={};decomposition={};controls={};outputs={}
                    for name in b['parent']['layouts']:
                        v,w=maps[name];lat=raw(oldrec['measurements'][name]['latent']);offsets=packet['variants'][name]['offsets'];yold=torch.empty_like(xx,dtype=torch.float64);ynew=torch.empty_like(yold);skiperr=torch.empty_like(yold);old32=torch.empty_like(xx)
                        for h in range(48):
                            sl=slice(offsets[h],offsets[h+1]);xm=xx[:,h].double()@w[h].double();skip=dd[h].double()*xx[:,h].double()
                            yold[:,h]=lat[:,sl].double()@v[h].double().T;ynew[:,h]=(lat[:,sl].double()-dd[h].double()*xm)@v[h].double().T+skip
                            skiperr[:,h]=dd[h].double()*(xm@v[h].double().T-xx[:,h].double());old32[:,h]=lat[:,sl]@v[h].T
                        yo=yold.reshape(n,3072);yn=ynew.reshape(n,3072);es=skiperr.reshape(n,3072);eo=yo-yy.double();er=yn-yy.double()
                        identity=error(eo,er+es);assert identity<=b['numeric']['algebra_relative']
                        decomposition[name]=dict(old_error_energy=float(eo.square().sum()),recurrent_error_energy=float(er.square().sum()),skip_error_energy=float(es.square().sum()),recurrent_skip_inner_product=float((er*es).sum()),source_y_energy=float(yy.double().square().sum()),identity_relative=identity,new_y_relative_RMS=error(yn,yy),old_y_relative_RMS=error(yo,yy))
                        arms[name]={};controls[name]={}
                        for arm,y in [('projected_skip',yo),('full_skip',yn)]:
                            arms[name][arm]={};ug=y*sg
                            for mode in b['parent']['modes']:
                                inv=source_inv if mode=='full_source_denominator' else torch.rsqrt(ug.square().mean(-1,keepdim=True)+eps)
                                inp=(ug*inv*weight.double()).bfloat16().double();op=(inp@ow.double().T).bfloat16();arms[name][arm][mode]=metrics(op,ref,label)
                                outputs[f'{name}__{arm}__{mode}']=op.cpu().view(torch.uint16).numpy().astype('<u2')
                                if arm=='projected_skip':
                                    ug32=old32.reshape(n,3072)*sg32;i32=inv32 if mode=='full_source_denominator' else torch.rsqrt(ug32.square().mean(-1,keepdim=True)+eps)
                                    oldop=torch.nn.functional.linear((ug32*i32*weight).bfloat16(),ow.bfloat16());mm=metrics(oldop,ref,label);prior=oldrec['arms'][name][mode]
                                    delta=max(abs(mm[k]-prior[k]) for k in mm);bridge=error(op,oldop)
                                    assert delta<=b['numeric']['old_control_metric_absolute'] and bridge<=b['numeric']['precision_bridge_relative'],('old control',stage,name,mode,delta,bridge)
                                    controls[name][mode]=dict(original_F32_metrics=mm,original_metric_max_delta=delta,F64_BF16_precision_bridge_relative=bridge)
                        del lat,yold,ynew,skiperr,old32,yo,yn,es,eo,er,ug,inp,op,oldop,ug32
                    path=a.directory/f'{rec["id"]}.site{site:02d}.npz';np.savez(path,**outputs)
                    rows.append(dict(id=rec['id'],site=site,split=rec['split'],domain=rec['domain'],history=n,source_reconstruction_relative=source_error,arms=arms,decomposition=decomposition,controls=controls,output=extent(path)))
                    print(json.dumps(dict(stage=stage,completed=len(rows),seconds=time.perf_counter()-CLOCK)),flush=True);del xx,yy,gate,ref,sg,sg32,u32,sourceop,source_inv,outputs;torch.cuda.empty_cache()
        stream.close();stream=None;check();guard();assert len(rows)==144
        aggregates,domains,gates=aggregate(rows,b);pass_any=any(all(all(gates[str(s)][name]['full_skip'][mode].values()) for s in b['parent']['sites']) for name in b['parent']['layouts'] for mode in b['parent']['modes'])
        r=dict(schema='RECURRENT_ONLY_CHANNEL_SKIP_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,records=rows,aggregates=aggregates,domains=domains,gates=gates,decision='LOCAL_FULL_SKIP_BUDGET_PASS_PENDING_AUDIT' if pass_any else 'LOCAL_FULL_SKIP_BUDGET_FAIL_PENDING_AUDIT',before_after_hashes_exact=True,source_forwards=0,new_scans=0,optimizer_updates=0,native_calls=0,quality_admission=False,speed_admission=False,recurrent_state_F32_bytes_per_site=dict(source=3072*256*4,projected=384*256*4),output_bytes=sum(r['output']['bytes'] for r in rows),seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),scope=b['scope'])
        write(a.out,r);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(decision=r['decision'],result=extent(a.out))),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),traceback=traceback.format_exc());
        if a.directory.exists():write(a.directory/'first_fault.json',dict(stage=stage,error=repr(e),completed_records=rows))
        raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved());write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    for name in ['binding','directory','out']:p.add_argument('--'+name,type=Path)
    for name in ['binding-sha','freeze']:p.add_argument('--'+name)
    a=p.parse_args();prepare(a) if a.prepare else run(a)
