"""Approximate BF16 module inputs from source gates;all144 independent field replays."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic();CLOCK=time.perf_counter();FIELDS=['gate','x','B','C','delta']


def bf(x,np):
    q=np.asarray(x,dtype='<f4');assert np.isfinite(q).all();u=q.view('<u4');return ((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype('<u2')


def decode(x,np):assert x.dtype==np.dtype('<u2');return (x.astype('<u4')<<16).view('<f4').astype('f8')


def replay(H,W,cw,cb,db,mu,minimum,maximum,np):
    # Promoted exact linear contractions plus explicit source BF16 boundaries.
    projected=decode(bf(H@W.T,np),np);scaled=decode(bf(projected*mu,np),np);n=len(H);bc=scaled[:,3072:6656];conv=np.broadcast_to(cb,(n,3584)).copy()
    for tap in range(4):
        lag=3-tap
        if lag==0:conv+=bc*cw[:,tap]
        else:conv[lag:]+=bc[:-lag]*cw[:,tap]
    conv=decode(bf(conv,np),np);activated=decode(bf(conv*np.exp(-np.logaddexp(0.,-conv)),np),np);d=decode(bf(scaled[:,6656:]+db,np),np);soft=np.where(d>20,d,np.logaddexp(0.,d));soft=decode(bf(soft,np),np);delta=np.clip(soft,minimum,np.inf if maximum is None else maximum)
    return dict(gate=bf(scaled[:,:3072],np),x=bf(activated[:,:3072],np),B=bf(activated[:,3072:3328],np),C=bf(activated[:,3328:],np),delta=bf(delta,np))


def prepare(a):
    import numpy as np,psutil,torch,transformers
    from transformers import FalconH1Config
    from transformers.modeling_utils import local_torch_dtype
    from transformers.models.falcon_h1.modeling_falcon_h1 import compute_mup_vector
    assert not a.out.exists();priorpath=DOC/'gate_input_operator_result_20261013.json';adpath=DOC/'gate_input_operator_adjudication_20261013.json';prior=json.loads(priorpath.read_bytes());ad=json.loads(adpath.read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(priorpath) and all(q['recovery_preflight_eligible'] for q in prior['records']);pb=json.loads(Path(prior['binding']['path']).read_bytes());wbpath=DOC/'full_channel_native_binding_20261013.json';wb=json.loads(wbpath.read_bytes());capture=json.loads(Path(wb['capture']['path']).read_bytes())['records'];cfg=json.loads(Path(pb['source_config']).read_bytes());config=FalconH1Config(**cfg);old=torch.get_default_dtype()
    with local_torch_dtype(torch.bfloat16):mu=compute_mup_vector(config)
    assert mu.dtype==torch.bfloat16 and mu.shape==(1,1,6704) and torch.get_default_dtype()==old and not torch.cuda.is_initialized();mu_bits=mu[0,0].view(torch.uint16).numpy().astype('<u2');sections=[3072,3072,256,256,48];expected=np.repeat(np.array(cfg['ssm_multipliers']),sections);assert np.array_equal(mu_bits,bf(expected,np));effective=decode(mu_bits,np)
    # Independent CPU primitive witnesses,not original CUDA equivalence.
    tx=torch.arange(14,dtype=torch.float32).reshape(1,2,7).to(torch.bfloat16)/4;tw=torch.tensor([1,2,3,4,4,3,2,1],dtype=torch.bfloat16).reshape(2,1,4)/16;tb=torch.tensor([.125,-.125],dtype=torch.bfloat16);tc=torch.nn.functional.conv1d(tx,tw,tb,padding=3,groups=2)[...,:7];manual=np.broadcast_to(tb.float().numpy().astype('f8'),(7,2)).copy();xx=tx[0].float().numpy().T.astype('f8');ww=tw[:,0].float().numpy().astype('f8')
    for tap in range(4):
        lag=3-tap
        if lag:manual[lag:]+=xx[:-lag]*ww[:,tap]
        else:manual+=xx*ww[:,tap]
    assert np.array_equal(tc[0].T.contiguous().view(torch.uint16).numpy(),bf(manual,np));z=torch.tensor([-12,-4,-2,-.5,0,.5,2,4,12,32],dtype=torch.bfloat16);zz=z.float().numpy().astype('f8');assert np.array_equal(torch.nn.functional.silu(z).view(torch.uint16).numpy(),bf(zz*np.exp(-np.logaddexp(0.,-zz)),np));assert np.array_equal(torch.nn.functional.softplus(z).view(torch.uint16).numpy(),bf(np.where(zz>20,zz,np.logaddexp(0.,zz)),np))
    minimum,maximum=config.time_step_limit;maximum=None if maximum==float('inf') else maximum;storage=DOC/'gate_input_recovery_storage_20261013.json';st=json.loads(storage.read_text(encoding='utf-8-sig'));paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',DOC/'GATE_INPUT_RECOVERY_PROTOCOL_20261013.md',storage,priorpath,adpath,Path(prior['binding']['path']),wbpath,Path(wb['capture']['path']),Path(pb['source_weights']),Path(pb['source_config']),Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),Path(np._core._multiarray_umath.__file__),Path(np.linalg._umath_linalg.__file__),Path(psutil.__file__),Path(torch.__file__),Path(torch._C.__file__),Path(transformers.__file__),SITE/'transformers/modeling_utils.py',SITE/'transformers/models/falcon_h1/modeling_falcon_h1.py',SITE/'transformers/models/falcon_h1/configuration_falcon_h1.py',B/'original_falcon_recurrent_capture.py']
    paths += [Path(q['file']['path']) for q in prior['records']];nr=Path(np.__file__).resolve().parent;pr=Path(psutil.__file__).resolve().parent;tr=Path(torch.__file__).resolve().parent;paths += list((nr.parent/'numpy.libs').glob('*.dll'))+list(pr.glob('*.pyd'))+[tr/'lib'/q for q in ['torch_cpu.dll','c10.dll']]
    for rec in capture:
        for site in [0,12,23]:paths += [Path(rec['sites'][site]['fields'][k]['path']) for k in FIELDS]
    inputs=[extent(q) for q in dict.fromkeys(paths)]
    b=dict(schema='GATE_INPUT_RECOVERY_BINDING_V1',python=str(Path(sys.executable).resolve()),versions=dict(numpy=np.__version__,psutil=psutil.__version__,torch=torch.__version__,transformers=transformers.__version__),source_weights=pb['source_weights'],source_config=pb['source_config'],operator_report=extent(priorpath),operator_adjudication=extent(adpath),capture=wb['capture'],sites=[0,12,23],fields=FIELDS,semantics=dict(module_input='AFTER source ssm_in_multiplier,BEFORE in_proj',input_dtype='BF16',MuP_dtype='BF16',MuP_bits=[int(q) for q in mu_bits],MuP_effective_section_values=[float(effective[sum(sections[:i])]) for i in range(5)],in_proj_bias=False,conv_channels=3584,conv_kernel=4,conv_padding=3,conv_groups=3584,conv_bias=True,conv_orientation='tap0=t-3,tap3=current',activation='SiLU BF16 input/output',delta_add_dtype='BF16',delta_softplus_beta=1,delta_softplus_threshold=20,delta_min=minimum,delta_max=maximum,model_init_dtype_context_witness=True,capture_gate_dtype_assert_witness=True,CPU_primitive_controls_bit_exact=True,CUDA_reduction_equivalence_admitted=False,GPU_initialized=False),policy=dict(recovery='H64=(gate / effective_BF16_gate_multiplier) @ P.T,P=solve(G.T@G,G.T)',candidate='BF16_RNE(H64)',unrounded_H='stored diagnostic,no alternative selection',rank_or_ridge_search=False,DEV_selection=False),numeric=dict(inverse_left_identity_max=1e-10,audit_inverse_relative=1e-10,audit_H_relative=1e-9,audit_vector_relative=1e-6,audit_scalar_relative=1e-8),consistency=dict(each_case_each_field_relative_RMS=.01),limits=dict(seconds=900,audit_seconds=1200,OS_bytes=6<<30,output_bytes=3<<30),output_directory=st['new_output_directory'],inputs=inputs,scope='OFFLINE approximate source in_proj input from supplied BF16 gates/fixed full-rank trained matrix,all48 histories/3sites,OTHER x/B/C/delta generator replay/promoted+rounded semantics. No true input bits/source model/causal encoder/compact core/n/50 admission.')
    assert not torch.cuda.is_initialized();write(a.out,b);print(json.dumps(dict(binding=extent(a.out),semantics_summary=b['semantics']['MuP_effective_section_values'],seconds=time.perf_counter()-CLOCK)),flush=True)


def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil'] and not a.directory.exists() and not a.out.exists() and str(a.directory.resolve())==str(Path(b['output_directory']).resolve());proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None;rows=[];maps=[];stage='startup';receipt=dict(schema='GATE_INPUT_RECOVERY_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
        if a.directory.exists():assert sum(q.stat().st_size for q in a.directory.iterdir())<=b['limits']['output_bytes']
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    def raw(e):assert e['dtype']=='BF16';return decode(np.fromfile(e['path'],'<u2').reshape(e['shape']),np)
    def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    def mm(x,y,pos):return dict(relative_RMS=rel(x,y),centered_relative_RMS=rel(x-x.mean(0),y-y.mean(0)),label_relative_RMS=rel(x[pos],y[pos]))
    try:
        check();a.directory.mkdir();get,stream=tensor_reader(b['source_weights'],np);capture=json.loads(Path(b['capture']['path']).read_bytes())['records'];operator=json.loads(Path(b['operator_report']['path']).read_bytes());ops={q['site']:q for q in operator['records']};mu=decode(np.array(b['semantics']['MuP_bits'],'u2'),np);m0=mu[0]
        for site in b['sites']:
            stage=f'map/{site}';guard();p=f'model.layers.{site}.mamba.';W=get(p+'in_proj.weight');cw=get(p+'conv1d.weight')[:,0];cb=get(p+'conv1d.bias');db=get(p+'dt_bias');G=W[:3072]
            with np.load(ops[site]['file']['path'],allow_pickle=False) as z:gram=z['Gram'].copy()
            assert W.shape==(6704,2048) and cw.shape==(3584,4) and cb.shape==(3584,) and db.shape==(48,);P=np.linalg.solve(gram,G.T);identity=float(np.max(np.abs(P@G-np.eye(2048))));assert identity<=b['numeric']['inverse_left_identity_max'];mp=a.directory/f'site{site:02d}.recovery_map.npz';np.savez(mp,P=P);maps.append(dict(site=site,file=extent(mp),left_identity_max=identity))
            for rec in capture:
                stage=f'{site}/{rec["id"]}';guard();n=rec['history'];fields=rec['sites'][site]['fields'];gate=raw(fields['gate']);H=(gate/m0)@P.T;Hbits=bf(H,np);candidate=decode(Hbits,np);pred=replay(candidate,W,cw,cb,db,mu,b['semantics']['delta_min'],b['semantics']['delta_max'],np);metrics={};flags={};arrays=dict(H_unrounded=H,H_candidate_BF16=Hbits)
                for key in FIELDS:
                    expected=raw(fields[key]);decoded=decode(pred[key],np);metrics[key]=mm(decoded,expected,rec['positions']);flags[key]=metrics[key]['relative_RMS']<=.01;arrays['predicted_'+key]=pred[key]
                linear_residual=H@G.T-gate/m0;normal=float(np.linalg.norm(linear_residual@G)/(max(np.linalg.norm(gate/m0)*np.linalg.norm(G),1e-30)));roundrel=rel(candidate,H);path=a.directory/(rec['id']+f'.site{site:02d}.npz');np.savez(path,**arrays);rows.append(dict(id=rec['id'],site=site,split=rec['split'],domain=rec['domain'],history=n,file=extent(path),metrics=metrics,consistency_flags=flags,normal_equation_residual_relative=normal,BF16_candidate_vs_unrounded_relative=roundrel));print(json.dumps(dict(stage=stage,complete=len(rows),flags=flags,seconds=time.perf_counter()-CLOCK)),flush=True)
        stream.close();stream=None;check();guard();assert len(rows)==144;aggregates={}
        for site in b['sites']:
            aggregates[str(site)]={}
            for split in ['FIT','DEV']:
                rr=[q for q in rows if q['site']==site and q['split']==split];assert len(rr)==24;aggregates[str(site)][split]={}
                for key in FIELDS:
                    vals=[q['metrics'][key]['relative_RMS'] for q in rr];dm={d:float(np.mean([q['metrics'][key]['relative_RMS'] for q in rr if q['domain']==d])) for d in {q['domain'] for q in rr}};aggregates[str(site)][split][key]=dict(mean=float(np.mean(vals)),worst=max(vals),domain_means=dm,worst_domain_mean=max(dm.values()),failed_cases=sum(not q['consistency_flags'][key] for q in rr))
        independent=all(all(q['consistency_flags'][k] for k in ['x','B','C','delta']) for q in rows);success=all(all(q['consistency_flags'].values()) for q in rows);r=dict(schema='GATE_INPUT_RECOVERY_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,recovery_maps=maps,records=rows,aggregates=aggregates,before_after_hashes_exact=True,source_forwards=0,native_calls=0,optimizer_updates=0,true_hidden_bits_recovered=0,approximate_module_inputs=144,calibration_consistency_all_independent_fields=independent,calibration_consistency_all_fields=success,decision='COMPLETE_PENDING_FULL_AUDIT',output_bytes=sum(q.stat().st_size for q in a.directory.iterdir()),seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope=b['scope']);write(a.out,r);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(result=extent(a.out),consistent=success,aggregates=aggregates)),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),stage=stage,traceback=traceback.format_exc())
        if a.directory.exists():write(a.directory/'first_fault.json',dict(stage=stage,error=repr(e),records=rows))
        raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    for name in ['binding','directory','out']:p.add_argument('--'+name,type=Path)
    for name in ['binding-sha','freeze']:p.add_argument('--'+name)
    a=p.parse_args()
    if a.run:run(a)
    else:
        import psutil
        proc=psutil.Process();receipt=dict(schema='GATE_INPUT_RECOVERY_PREPARATION_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
        try:prepare(a);receipt.update(exit_code=0,error=None,result=extent(a.out))
        except BaseException as e:
            import traceback
            receipt.update(exit_code=1,error=repr(e),traceback=traceback.format_exc());raise
        finally:
            receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.preparation_terminal.json'),receipt)
