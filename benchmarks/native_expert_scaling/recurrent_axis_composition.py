"""Fixed source-state96 and accepted channel384 maps: actual joint SSD scans."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
from native_head_gpu_preflight import gpu_metadata
START=time.monotonic();CLOCK=time.perf_counter()
MODES=['full_source_denominator','reconstructed_denominator']


def prepare(a):
    oldpath=DOC/'recurrent_output_bases_result_20261012.json';adpath=DOC/'recurrent_output_bases_adjudication_20261012.json';old=json.loads(oldpath.read_bytes());ad=json.loads(adpath.read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(oldpath);pb=json.loads(Path(old['binding']['path']).read_bytes());packet=next(q for q in old['selections'] if q['site']==23);assert all(all(old['gates']['23']['adaptive384']['full_skip'][mode].values()) for mode in MODES)
    statepath=DOC/'original_falcon_recurrent_projection_finish_result_20261009.json';statead=DOC/'original_falcon_recurrent_projection_audit_result_20261009.json';sr=json.loads(statepath.read_bytes());sa=json.loads(statead.read_bytes());assert sa['decision']=='STORED_IDENTITIES_AND_AGGREGATES_VERIFIED';sp=next(q for q in sr['bases'] if q['site']==23);capture=json.loads(Path(pb['capture']['path']).read_bytes())['records'];storage=DOC/'recurrent_axis_composition_storage_20261013.json';st=json.loads(storage.read_text(encoding='utf-8-sig'));oldrows=[q for q in old['records'] if q['site']==23]
    paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',B/'native_head_gpu_preflight.py',B/'original_falcon_mixed_history.py',B/'original_falcon_channel_transport.py',B/'original_falcon_whole_recovery.py',B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py',SITE/'transformers/models/falcon_h1/modeling_falcon_h1.py',DOC/'RECURRENT_AXIS_COMPOSITION_PROTOCOL_20261013.md',storage,oldpath,adpath,Path(old['binding']['path']),statepath,statead,Path(pb['capture']['path']),Path(pb['parent']['source_weights']),Path(pb['parent']['source_config']),Path(packet['basis']['path']),Path(sp['basis']['path'])]
    paths += [Path(q['output']['path']) for q in oldrows]
    for q in capture:paths += [Path(e['path']) for e in q['sites'][23]['fields'].values()]
    runtime=[e for e in pb['inputs'] if e['path'].startswith(str(Path(sys.executable).parent)) or e['path'].startswith(str(SITE.resolve())) or e['path'].startswith('C:\\Windows\\System32')]
    paths += [ROOT/p for p in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']]
    b=dict(schema='RECURRENT_AXIS_COMPOSITION_BINDING_V1',python=pb['python'],gpu=pb['gpu'],versions=pb['versions'],parent=pb['parent'],capture=pb['capture'],channel_report=extent(oldpath),channel_adjudication=extent(adpath),state_report=extent(statepath),state_adjudication=extent(statead),channel_basis=packet['basis'],state_basis=sp['basis'],ranks=packet['variants']['adaptive384']['ranks'],offsets=packet['variants']['adaptive384']['offsets'],prior_outputs=[q['output'] for q in oldrows],site=23,channel_width=384,state_width=96,modes=MODES,output_directory=st['new_output_directory'],numeric=dict(biorthogonality=1e-8,scan_relative=1e-4,projected_relative=1e-6,audit_output_relative=1e-4,audit_scalar_relative=1e-8,baseline_metric_absolute=1e-10),criteria=dict(mean=.1,worst=.2,centered=.1,domain=.2),limits=dict(seconds=900,audit_seconds=1200,OS_bytes=6<<30,GPU_allocated_bytes=4<<30,GPU_reserved_bytes=5<<30,output_bytes=2<<30),inputs=list({e['path']:e for e in runtime+[extent(p) for p in paths]}.values()),scope='Fixed FIT site23 adaptive384 channel maps and dual96 state maps/no refit or DEV selection. All48 source-injected actualF32 scans/fullskip/gate/source gen/conv retained. Controlled F64+BF16 output,not full F32 source precision/native/causal/model/n/50 admission.')
    assert gpu_metadata()==b['gpu'];write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.perf_counter()-CLOCK)),flush=True)


def run(a):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np,psutil,torch
    from original_falcon_mixed_history import scan
    from transformers.models.falcon_h1 import modeling_falcon_h1 as code
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and gpu_metadata()==b['gpu'] and str(a.directory.resolve())==str(Path(b['output_directory']).resolve()) and not a.directory.exists() and not a.out.exists();assert dict(torch=torch.__version__,numpy=np.__version__,psutil=psutil.__version__,cuda=torch.version.cuda)==b['versions'];assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and all(os.environ.get(k)=='6' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']);proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();torch.cuda.reset_peak_memory_stats();stream=None;rows=[];stage='startup'
    receipt=dict(schema='RECURRENT_AXIS_COMPOSITION_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        l=b['limits'];assert time.monotonic()-START<=l['seconds'] and proc.memory_info().peak_wset<=l['OS_bytes'] and not proc.children(recursive=True);assert torch.cuda.max_memory_allocated()<=l['GPU_allocated_bytes'] and torch.cuda.max_memory_reserved()<=l['GPU_reserved_bytes']
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir())<=l['output_bytes']
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    def raw(e):
        z=(np.fromfile(e['path'],'<u2').astype('<u4')<<16).view('<f4') if e['dtype']=='BF16' else np.fromfile(e['path'],'<f4');z=z.reshape(e['shape']).astype('f8');assert np.isfinite(z).all();return torch.from_numpy(z).cuda()
    def rel(x,y):return float(torch.linalg.vector_norm(x.double()-y.double())/torch.linalg.vector_norm(y.double()).clamp_min(1e-30))
    def metrics(op,ref,pos):return dict(output_relative_RMS=rel(op,ref),centered_output_relative_RMS=rel(op.double()-op.double().mean(0),ref.double()-ref.double().mean(0)),label_output_relative_RMS=rel(op[pos],ref[pos]))
    try:
        check();a.directory.mkdir();old=json.loads(Path(b['channel_report']['path']).read_bytes());oldrows={q['id']:q for q in old['records'] if q['site']==23};capture=json.loads(Path(b['capture']['path']).read_bytes())['records'];get,stream=tensor_reader(b['parent']['source_weights'],np);p='model.layers.23.mamba.';dd=torch.from_numpy(get(p+'D')).cuda();alog=torch.from_numpy(get(p+'A_log')).float().cuda();gamma=torch.from_numpy(get(p+'norm.weight')).cuda();ow=torch.from_numpy(get(p+'out_proj.weight')).cuda();eps=json.loads(Path(b['parent']['source_config']).read_bytes())['rms_norm_eps']
        with np.load(b['channel_basis']['path'],allow_pickle=False) as z:
            vm=[z[f'adaptive384_V_{h}'].copy() for h in range(48)];wm=[z[f'adaptive384_W_{h}'].copy() for h in range(48)]
        with np.load(b['state_basis']['path'],allow_pickle=False) as z:vs=z['V'].copy();ws=z['W'].copy()
        ranks=b['ranks'];offs=b['offsets'];assert sum(ranks)==384 and vs.shape==ws.shape==(256,96) and np.max(np.abs(ws.T@vs-np.eye(96)))<=b['numeric']['biorthogonality']
        for h in range(48):assert vm[h].shape==wm[h].shape==(64,ranks[h]) and np.max(np.abs(wm[h].T@vm[h]-np.eye(ranks[h])))<=b['numeric']['biorthogonality']
        v=[torch.from_numpy(q).float().cuda() for q in vm];w=[torch.from_numpy(q).float().cuda() for q in wm];vs=torch.from_numpy(vs).float().cuda();ws=torch.from_numpy(ws).float().cuda()
        with torch.inference_mode():
            for rec in capture:
                stage=rec['id'];guard();n=rec['history'];f=rec['sites'][23]['fields'];x=raw(f['x']).reshape(n,48,64);y=raw(f['y']);gate=raw(f['gate']);g=gate*torch.sigmoid(gate);skip=dd[None,:,None]*x;bb=raw(f['B']).float();cc=raw(f['C']).float();dt=raw(f['delta']).bfloat16();bp=bb@ws;cp=cc@vs;ys=scan(torch,code,x.float(),bp,cp,dt,alog,dd.float()).reshape(n,48,64);guard();lat=torch.empty((n,384),device='cuda');px=torch.empty_like(lat)
                for rank in sorted(set(ranks)):
                    heads=[h for h in range(48) if ranks[h]==rank];wg=torch.stack([w[h] for h in heads]);xm=torch.einsum('thc,hcr->thr',x.float()[:,heads],wg);ym=scan(torch,code,xm,bp,cp,dt[:,heads],alog[heads],dd[heads].float())
                    for j,h in enumerate(heads):sl=slice(offs[h],offs[h+1]);px[:,sl]=xm[:,j];lat[:,sl]=ym[:,j]
                projected=torch.cat([ys[:,h].double()@w[h].double() for h in range(48)],dim=1);commutation=rel(lat,projected);assert commutation<=b['numeric']['scan_relative']
                prior=oldrows[rec['id']]
                with np.load(prior['output']['path'],allow_pickle=False) as z:oldlat=torch.from_numpy(z['adaptive384__latent'].copy()).cuda();oldpx=torch.from_numpy(z['adaptive384__projected_input'].copy()).cuda();baselinebits={mode:z['adaptive384__full_skip__'+mode].copy() for mode in MODES}
                input_control=rel(px,oldpx);assert input_control<=b['numeric']['projected_relative'];yc=torch.cat([(oldlat[:,offs[h]:offs[h+1]].double()-dd[h]*oldpx[:,offs[h]:offs[h+1]].double())@v[h].double().T+skip[:,h] for h in range(48)],dim=1);yj=torch.cat([(lat[:,offs[h]:offs[h+1]].double()-dd[h]*px[:,offs[h]:offs[h+1]].double())@v[h].double().T+skip[:,h] for h in range(48)],dim=1);source_inv=torch.rsqrt((y*g).square().mean(-1,keepdim=True)+eps);ref=raw(f['output']);pos=torch.tensor(rec['positions'],device='cuda');mm={};arrays=dict(state96_B=bp.cpu().numpy().astype('f4'),state96_C=cp.cpu().numpy().astype('f4'),state96_latent=ys.reshape(n,3072).cpu().numpy().astype('f4'),joint96_latent=lat.cpu().numpy().astype('f4'),projected_input=px.cpu().numpy().astype('f4'))
                for arm,Y in [('channel384_state256',yc),('channel3072_state96',ys.reshape(n,3072).double()),('channel384_state96',yj)]:
                    ug=Y*g;mm[arm]={}
                    for mode in MODES:
                        inv=source_inv if mode=='full_source_denominator' else torch.rsqrt(ug.square().mean(-1,keepdim=True)+eps);op=(((ug*inv*gamma).bfloat16().double())@ow.T).bfloat16();bits=op.cpu().view(torch.uint16).numpy().astype('<u2');mm[arm][mode]=metrics(op,ref,pos)
                        if arm=='channel384_state256':
                            assert np.array_equal(bits,baselinebits[mode]);assert max(abs(mm[arm][mode][k]-prior['arms']['adaptive384']['full_skip'][mode][k]) for k in mm[arm][mode])<=b['numeric']['baseline_metric_absolute']
                        else:arrays[arm+'__'+mode]=bits
                # Actual error interaction is retained; no linear quality-additivity assumption.
                ec=yc-y;es=ys.reshape(n,3072).double()-y;ej=yj-y;interaction=ej-ec-es;energies={key:float(q.square().sum()) for key,q in [('channel_error',ec),('state_error',es),('joint_error',ej),('interaction',interaction)]};energies.update(channel_state_inner=float((ec*es).sum()),channel_interaction_inner=float((ec*interaction).sum()),state_interaction_inner=float((es*interaction).sum()),source_y=float(y.square().sum()))
                ep=energies;identity=abs(ep['joint_error']-(ep['channel_error']+ep['state_error']+ep['interaction']+2*(ep['channel_state_inner']+ep['channel_interaction_inner']+ep['state_interaction_inner'])))/max(ep['joint_error'],1e-30);assert identity<=1e-10
                path=a.directory/(rec['id']+'.npz');np.savez(path,**arrays);rows.append(dict(id=rec['id'],split=rec['split'],domain=rec['domain'],history=n,commutation_relative=commutation,prior_input_relative=input_control,prior_output=prior['output'],metrics=mm,energy=energies,energy_identity_relative=identity,file=extent(path)));print(json.dumps(dict(stage=stage,complete=len(rows),seconds=time.perf_counter()-CLOCK)),flush=True);torch.cuda.empty_cache()
        stream.close();stream=None;check();guard();assert len(rows)==48;aggregates={};gates={}
        for arm in rows[0]['metrics']:
            aggregates[arm]={};gates[arm]={}
            for mode in MODES:
                aggregates[arm][mode]={}
                for split in ['FIT','DEV']:
                    sr=[q for q in rows if q['split']==split];assert len(sr)==24;vals=[q['metrics'][arm][mode]['output_relative_RMS'] for q in sr];c=sum(q['metrics'][arm][mode]['centered_output_relative_RMS'] for q in sr)/24;dm={d:sum(q['metrics'][arm][mode]['output_relative_RMS'] for q in sr if q['domain']==d)/2 for d in sorted({q['domain'] for q in sr})};aa=dict(mean=sum(vals)/24,worst=max(vals),centered_mean=c,domain_means=dm,worst_domain_mean=max(dm.values()));aggregates[arm][mode][split]=aa
                    if split=='DEV':gates[arm][mode]=dict(mean=aa['mean']<=.1,worst=aa['worst']<=.2,centered=c<=.1,domain=aa['worst_domain_mean']<=.2)
        costs=dict(source_state_coordinates=3072*256,joint_state_coordinates=384*96,state_coordinate_ratio=(3072*256)/(384*96),F32_source_state_bytes=4*3072*256,F32_joint_state_bytes=4*384*96,original_engine_state_coordinates=512*96,source_feature_channels_retained=3072,source_gate_channels_retained=3072,source_heads_retained=48,source_conv_width_retained=4)
        r=dict(schema='RECURRENT_AXIS_COMPOSITION_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,records=rows,aggregates=aggregates,gates=gates,algebraic_state_costs=costs,before_after_hashes_exact=True,new_scans=96,basis_fits=0,source_forwards=0,optimizer_updates=0,native_calls=0,quality_admission=False,speed_admission=False,decision='COMPLETE_PENDING_INDEPENDENT_AUDIT',output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()),seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),scope=b['scope']);write(a.out,r);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(aggregates=aggregates,gates=gates,result=extent(a.out))),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),stage=stage,traceback=traceback.format_exc())
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
