"""Fixed FIT-only whitened point: all48 ideal and actual compact scans."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
from native_head_gpu_preflight import gpu_metadata
START=time.monotonic();CLOCK=time.perf_counter()


def prepare(a):
    rp=DOC/'whitened_recurrence_result_20261012.json';ap=DOC/'whitened_recurrence_adjudication_20261012.json';r=json.loads(rp.read_bytes());ad=json.loads(ap.read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(rp);pb=json.loads(Path(r['binding']['path']).read_bytes());prior=json.loads(Path(pb['prior']['path']).read_bytes());oldb=json.loads(Path(prior['binding']['path']).read_bytes());capture=json.loads(Path(pb['capture']['path']).read_bytes())['records'];assert (r['recommended_curve_coordinate'],r['recommended_curve_alpha'])==('whitened',.1);point=r['curves']['whitened']['best_FIT_point'];sp=DOC/'whitened_candidate_storage_20261012.json';storage=json.loads(sp.read_text(encoding='utf-8-sig'))
    paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',B/'native_head_gpu_preflight.py',B/'original_falcon_mixed_history.py',B/'original_falcon_channel_transport.py',B/'original_falcon_whole_recovery.py',B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py',SITE/'transformers/models/falcon_h1/modeling_falcon_h1.py',DOC/'WHITENED_CANDIDATE_PROTOCOL_20261012.md',sp,rp,ap,Path(r['binding']['path']),Path(pb['capture']['path']),Path(pb['parent']['source_weights']),Path(pb['parent']['source_config']),Path(pb['selected']['path']),Path(point['parameters']['path']),Path(pb['prior']['path'])]
    for q in capture:paths += [Path(e['path']) for e in q['sites'][12]['fields'].values()]
    runtime=[e for e in pb['inputs'] if e['path'].startswith(str(Path(sys.executable).parent)) or e['path'].startswith(str(SITE.resolve())) or e['path'].startswith('C:\\Windows\\System32')];paths += [ROOT/p for p in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']]
    b=dict(schema='WHITENED_CANDIDATE_BINDING_V1',python=pb['python'],gpu=pb['gpu'],versions=pb['versions'],parent=pb['parent'],capture=pb['capture'],preflight=extent(rp),preflight_adjudication=extent(ap),baseline=pb['selected'],candidate=point['parameters'],ranks=oldb['packet']['variants']['adaptive384']['ranks'],offsets=oldb['packet']['variants']['adaptive384']['offsets'],coordinate='whitened',alpha=.1,expected_FIT_loss=point['FIT_loss'],prior=pb['prior'],output_directory=storage['new_output_directory'],numeric=dict(biorthogonality=1e-8,curve_scalar_absolute=1e-8,scan_relative=1e-4,projected_input_relative=1e-6,audit_vector_relative=1e-9,audit_scalar_relative=1e-8,audit_BF16_relative=1e-4),criteria=dict(DEV_output_mean=.1,DEV_case_worst=.2,DEV_domain_mean_worst=.2,DEV_centered_mean=.1),limits=dict(seconds=900,audit_seconds=1200,OS_bytes=6<<30,GPU_allocated_bytes=4<<30,GPU_reserved_bytes=5<<30,output_bytes=2<<30),inputs=list({e['path']:e for e in runtime+[extent(p) for p in paths]}.values()),scope='Fixed physical whitened alpha.1 FIT-only point,all48 source-injected histories/site12,ideal and actual newF32 scans/fullskip/source state256/gate/generator/conv retained. No new search/gradient/source forward/native/model/n/50 admission.')
    assert gpu_metadata()==b['gpu'];write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.perf_counter()-CLOCK)),flush=True)


def run(a):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np,psutil,torch
    from original_falcon_mixed_history import scan
    from transformers.models.falcon_h1 import modeling_falcon_h1 as code
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and gpu_metadata()==b['gpu'] and str(a.directory.resolve())==str(Path(b['output_directory']).resolve()) and not a.directory.exists() and not a.out.exists();assert dict(torch=torch.__version__,numpy=np.__version__,psutil=psutil.__version__,cuda=torch.version.cuda)==b['versions'];assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and all(os.environ.get(k)=='6' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']);proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();torch.cuda.reset_peak_memory_stats();stream=None;rows=[];stage='startup'
    receipt=dict(schema='WHITENED_CANDIDATE_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        l=b['limits'];assert time.monotonic()-START<=l['seconds'] and proc.memory_info().peak_wset<=l['OS_bytes'] and not proc.children(recursive=True);assert torch.cuda.max_memory_allocated()<=l['GPU_allocated_bytes'] and torch.cuda.max_memory_reserved()<=l['GPU_reserved_bytes']
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir())<=l['output_bytes']
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    def raw(e):
        z=(np.fromfile(e['path'],'<u2').astype('<u4')<<16).view('<f4') if e['dtype']=='BF16' else np.fromfile(e['path'],'<f4');z=z.reshape(e['shape']).astype('f8');assert np.isfinite(z).all();return torch.from_numpy(z).cuda()
    def error(x,y):return float(torch.linalg.vector_norm(x.double()-y.double())/torch.linalg.vector_norm(y.double()).clamp_min(1e-30))
    def metrics(op,ref,pos):return dict(output_relative_RMS=error(op,ref),centered_output_relative_RMS=error(op.double()-op.double().mean(0),ref.double()-ref.double().mean(0)),label_output_relative_RMS=error(op[pos],ref[pos]))
    try:
        check();a.directory.mkdir();capture=json.loads(Path(b['capture']['path']).read_bytes())['records'];pre=json.loads(Path(b['preflight']['path']).read_bytes());prefit={q['id']:q for q in pre['curves']['whitened']['best_FIT_point']['FIT_cases']};prior=json.loads(Path(b['prior']['path']).read_bytes());priorrows={q['id']:q for q in prior['ideal']};get,stream=tensor_reader(b['parent']['source_weights'],np);p='model.layers.12.mamba.';dd=torch.from_numpy(get(p+'D')).cuda();alog=torch.from_numpy(get(p+'A_log')).float().cuda();ow=torch.from_numpy(get(p+'out_proj.weight')).cuda();gamma=torch.from_numpy(get(p+'norm.weight')).cuda();F=ow*gamma[None,:];eps=json.loads(Path(b['parent']['source_config']).read_bytes())['rms_norm_eps'];points=[]
        for e in [b['baseline'],b['candidate']]:
            with np.load(e['path'],allow_pickle=False) as z:points.append(([torch.from_numpy(z[f'V_{h}'].copy()).cuda() for h in range(48)],[torch.from_numpy(z[f'W_{h}'].copy()).cuda() for h in range(48)]))
        V,W=points[1];ranks=b['ranks'];offs=b['offsets'];assert sum(ranks)==384
        for h in range(48):assert float((W[h].T@V[h]-torch.eye(ranks[h],device='cuda',dtype=torch.float64)).abs().max())<=b['numeric']['biorthogonality']
        with torch.inference_mode():
            for rec in capture:
                stage=rec['id'];guard();n=rec['history'];f=rec['sites'][12]['fields'];x=raw(f['x']).reshape(n,48,64);y=raw(f['y']);gate=raw(f['gate']);g=gate*torch.sigmoid(gate);R=y.reshape(n,48,64)-dd[None,:,None]*x;skip=dd[None,:,None]*x;su=y*g;teacher=(su/torch.sqrt(su.square().mean(-1,keepdim=True)+eps))@F.T;den=teacher.square().sum();pos=torch.tensor(rec['positions'],device='cuda');ideal={};arrays={}
                for name,(v,w) in zip(['baseline','candidate'],points):
                    Y=torch.stack([(R[:,h]@w[h])@v[h].T+skip[:,h] for h in range(48)],dim=1).reshape(n,3072);ug=Y*g;op=(ug/torch.sqrt(ug.square().mean(-1,keepdim=True)+eps))@F.T;loss=float((op-teacher).square().sum()/den);ideal[name]=dict(loss=loss,relative_RMS=loss**.5,**metrics(op,teacher,pos));arrays[name+'_ideal_output']=op.cpu().numpy()
                assert abs(ideal['baseline']['relative_RMS']-priorrows[rec['id']]['selected']['relative_RMS'])<=b['numeric']['curve_scalar_absolute']
                if rec['split']=='FIT':assert abs(ideal['candidate']['loss']-prefit[rec['id']]['loss'])<=b['numeric']['curve_scalar_absolute']
                lat=torch.empty((n,384),device='cuda');px=torch.empty_like(lat);xx=x.float();bb=raw(f['B']).float();cc=raw(f['C']).float();dt=raw(f['delta']).bfloat16()
                for rank in sorted(set(ranks)):
                    heads=[h for h in range(48) if ranks[h]==rank];w=torch.stack([W[h].float() for h in heads]);xm=torch.einsum('thc,hcr->thr',xx[:,heads],w);ym=scan(torch,code,xm,bb,cc,dt[:,heads],alog[heads],dd[heads].float())
                    for j,h in enumerate(heads):sl=slice(offs[h],offs[h+1]);lat[:,sl]=ym[:,j];px[:,sl]=xm[:,j]
                expected=torch.empty_like(lat,dtype=torch.float64);rexpected=torch.empty_like(expected);sk=torch.empty_like(expected);Y=torch.empty((n,48,64),device='cuda',dtype=torch.float64)
                for h in range(48):sl=slice(offs[h],offs[h+1]);w=W[h].float().double();v=V[h].float().double();expected[:,sl]=y.reshape(n,48,64)[:,h]@w;rexpected[:,sl]=R[:,h]@w;sk[:,sl]=dd[h]*px[:,sl].double();Y[:,h]=(lat[:,sl].double()-sk[:,sl])@v.T+skip[:,h]
                pe=error(lat,expected);re=error(lat.double()-sk,rexpected);assert pe<=b['numeric']['scan_relative'] and re<=b['numeric']['scan_relative'];ug=Y.reshape(n,3072)*g;source_inv=torch.rsqrt(su.square().mean(-1,keepdim=True)+eps);machine={};ref=raw(f['output'])
                for mode,inv in [('full_source_denominator',source_inv),('reconstructed_denominator',torch.rsqrt(ug.square().mean(-1,keepdim=True)+eps))]:
                    op=(((ug*inv*gamma).bfloat16().double())@ow.T).bfloat16();machine[mode]=metrics(op,ref,pos);arrays[mode]=op.cpu().view(torch.uint16).numpy().astype('<u2')
                arrays['latent']=lat.cpu().numpy().astype('f4');arrays['projected_input']=px.cpu().numpy().astype('f4');path=a.directory/(rec['id']+'.npz');np.savez(path,**arrays);rows.append(dict(id=rec['id'],split=rec['split'],domain=rec['domain'],history=n,ideal=ideal,machine=machine,full_scan_relative=pe,recurrent_scan_relative=re,file=extent(path)));print(json.dumps(dict(stage=stage,complete=len(rows),seconds=time.perf_counter()-CLOCK)),flush=True);torch.cuda.empty_cache()
        stream.close();stream=None;check();guard();assert len(rows)==48;ideal_aggregates={};machine_aggregates={};gates={}
        for split in ['FIT','DEV']:
            sr=[q for q in rows if q['split']==split];assert len(sr)==24;ideal_aggregates[split]={name:{key:sum(q['ideal'][name][key] for q in sr)/24 for key in sr[0]['ideal'][name]} for name in ['baseline','candidate']}
        for mode in ['full_source_denominator','reconstructed_denominator']:
            sr=[q for q in rows if q['split']=='DEV'];m=sum(q['machine'][mode]['output_relative_RMS'] for q in sr)/24;w=max(q['machine'][mode]['output_relative_RMS'] for q in sr);c=sum(q['machine'][mode]['centered_output_relative_RMS'] for q in sr)/24;d=max(sum(q['machine'][mode]['output_relative_RMS'] for q in sr if q['domain']==domain)/2 for domain in {q['domain'] for q in sr});machine_aggregates[mode]=dict(mean=m,worst=w,centered_mean=c,worst_domain_mean=d);gates[mode]=dict(mean=m<=.1,worst=w<=.2,centered=c<=.1,domain=d<=.2)
        assert abs(ideal_aggregates['FIT']['candidate']['loss']-b['expected_FIT_loss'])<=b['numeric']['curve_scalar_absolute'];result=dict(schema='WHITENED_CANDIDATE_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,records=rows,ideal_aggregates=ideal_aggregates,machine_aggregates=machine_aggregates,gates=gates,decision='FIXED_LOCAL_CANDIDATE_COMPLETE_PENDING_AUDIT',before_after_hashes_exact=True,optimizer_updates=0,new_packed_scans=48,source_forwards=0,native_calls=0,quality_admission=False,speed_admission=False,output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()),seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),scope=b['scope']);write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(ideal=ideal_aggregates,machine=machine_aggregates,result=extent(a.out))),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),traceback=traceback.format_exc());
        if a.directory.exists():write(a.directory/'first_fault.json',dict(stage=stage,error=repr(e),records=rows))
        raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved());write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    for name in ['binding','directory','out']:p.add_argument('--'+name,type=Path)
    for name in ['binding-sha','freeze']:p.add_argument('--'+name)
    a=p.parse_args();prepare(a) if a.prepare else run(a)
