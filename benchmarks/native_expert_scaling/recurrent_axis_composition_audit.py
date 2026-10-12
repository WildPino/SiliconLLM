"""Independent complete CPU state96 scans, fixed axis maps and all BF16 outputs."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic();CLOCK=time.perf_counter()


def main(a):
    import numpy as np,psutil
    re=extent(a.report);te=extent(a.terminal);assert re['sha256']==a.report_sha and te['sha256']==a.terminal_sha;r=json.loads(a.report.read_bytes());terminal=json.loads(a.terminal.read_bytes());assert terminal['exit_code']==0 and terminal['result']==re and extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil'] and not a.out.exists();proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None;stage='startup';rows=[];maximum={};seqwords=0;outputwords=0;projectedwords=0
    receipt=dict(schema='RECURRENT_AXIS_COMPOSITION_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False);sources=[extent(Path(__file__)),extent(B/'falcon_auxiliary_s6_transport.py'),extent(DOC/'RECURRENT_AXIS_COMPOSITION_AUDIT_PROTOCOL_20261013.md')]
    def guard():assert time.monotonic()-START<=b['limits']['audit_seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
    def check():
        for e in b['inputs']+[r['binding'],re,te]+sources+[q['file'] for q in r['records']]:guard();assert extent(e['path'])==e,e['path']
    def load(e):
        with np.load(e['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def raw(e):
        z=(np.fromfile(e['path'],'<u2').astype('<u4')<<16).view('<f4') if e['dtype']=='BF16' else np.fromfile(e['path'],'<f4');z=z.reshape(e['shape']).astype('f8');assert np.isfinite(z).all();return z
    def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    def array_check(x,y,tol,key):
        assert np.isfinite(x).all() and np.isfinite(y).all();err=rel(x,y);maximum[key]=max(maximum.get(key,0.),err);assert err<=tol,(stage,key,err)
    def scalar(x,y):
        err=float(abs(x-y)/max(abs(y),1.));maximum['scalar']=max(maximum.get('scalar',0.),err);assert err<=b['numeric']['audit_scalar_relative'],(stage,x,y,err)
    def decode(z):assert z.dtype==np.dtype('<u2');return (z.astype('<u4')<<16).view('<f4').astype('f8')
    def bf16(z):
        f=np.asarray(z,dtype='<f4');assert np.isfinite(f).all();u=f.view('<u4');return ((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype('<u2')
    def metrics(op,ref,pos):return dict(output_relative_RMS=rel(op,ref),centered_output_relative_RMS=rel(op-op.mean(0),ref-ref.mean(0)),label_output_relative_RMS=rel(op[pos],ref[pos]))
    try:
        check();get,stream=tensor_reader(b['parent']['source_weights'],np);p='model.layers.23.mamba.';dd=get(p+'D');aa=-np.exp(get(p+'A_log'));ow=get(p+'out_proj.weight');gamma=get(p+'norm.weight');eps=json.loads(Path(b['parent']['source_config']).read_bytes())['rms_norm_eps'];capture=json.loads(Path(b['capture']['path']).read_bytes())['records'];captures={q['id']:q for q in capture};prior=json.loads(Path(b['channel_report']['path']).read_bytes());priorrows={q['id']:q for q in prior['records'] if q['site']==23};packet=next(q for q in prior['selections'] if q['site']==23);assert packet['basis']==b['channel_basis'] and packet['variants']['adaptive384']['ranks']==b['ranks'] and packet['variants']['adaptive384']['offsets']==b['offsets'];stateprior=json.loads(Path(b['state_report']['path']).read_bytes());assert next(q for q in stateprior['bases'] if q['site']==23)['basis']==b['state_basis']
        channel=load(b['channel_basis']);statebasis=load(b['state_basis']);vm=[channel[f'adaptive384_V_{h}'] for h in range(48)];wm=[channel[f'adaptive384_W_{h}'] for h in range(48)];vs=statebasis['V'];ws=statebasis['W'];ranks=b['ranks'];offs=b['offsets'];assert sum(ranks)==384 and offs==[sum(ranks[:h]) for h in range(49)] and vs.shape==ws.shape==(256,96) and np.max(np.abs(ws.T@vs-np.eye(96)))<=b['numeric']['biorthogonality'];assert len(r['records'])==48 and {q['id'] for q in r['records']}==set(captures)
        for h in range(48):assert vm[h].shape==wm[h].shape==(64,ranks[h]) and np.max(np.abs(wm[h].T@vm[h]-np.eye(ranks[h])))<=b['numeric']['biorthogonality']
        v=[q.astype('f4').astype('f8') for q in vm];w=[q.astype('f4').astype('f8') for q in wm];vs=vs.astype('f4').astype('f8');ws=ws.astype('f4').astype('f8');ids=np.repeat(np.arange(48),ranks)
        for item in r['records']:
            stage=item['id'];guard();rec=captures[item['id']];n=rec['history'];f=rec['sites'][23]['fields'];x=raw(f['x']).reshape(n,48,64);y=raw(f['y']);gate=raw(f['gate']);g=gate*np.exp(-np.logaddexp(0.,-gate));skip=dd[None,:,None]*x;bb=raw(f['B']);cc=raw(f['C']);dt=raw(f['delta']);z=load(item['file']);assert len(z)==9 and (item['split'],item['domain'],item['history'])==(rec['split'],rec['domain'],n);pr=priorrows[rec['id']];assert item['prior_output']==pr['output'];old=load(pr['output']);px=z['projected_input'].astype('f8');lat=z['joint96_latent'].astype('f8');ys=z['state96_latent'].astype('f8');bp=z['state96_B'].astype('f8');cp=z['state96_C'].astype('f8')
            for key in ['projected_input','joint96_latent','state96_latent','state96_B','state96_C']:assert z[key].dtype==np.dtype('<f4')
            assert px.shape==lat.shape==(n,384) and ys.shape==(n,3072) and bp.shape==cp.shape==(n,96);array_check(bp,bb@ws,b['numeric']['projected_relative'],'state_B');array_check(cp,cc@vs,b['numeric']['projected_relative'],'state_C');expectedpx=np.concatenate([x[:,h]@w[h] for h in range(48)],axis=1);array_check(px,expectedpx,b['numeric']['projected_relative'],'channel_input');projectedwords+=bp.size+cp.size+px.size
            # Direct recurrence, no source SSD helper: independent state update for ALL channels.
            for key,xscan,hids,target in [('source96',x.reshape(n,3072),np.repeat(np.arange(48),64),ys),('joint96',px,ids,lat)]:
                st=np.zeros((len(hids),96));direct=np.empty_like(target)
                for t in range(n):
                    if t%64==0:guard()
                    delta=dt[t,hids];st*=np.exp(delta*aa[hids])[:,None];st+=(delta*xscan[t])[:,None]*bp[t][None,:];direct[t]=np.sum(st*cp[t][None,:],axis=1)+dd[hids]*xscan[t]
                array_check(target,direct,b['numeric']['scan_relative'],'sequential_'+key);seqwords+=target.size
            projected=np.concatenate([ys.reshape(n,48,64)[:,h]@w[h] for h in range(48)],axis=1);comm=rel(lat,projected);assert comm<=b['numeric']['scan_relative'];scalar(item['commutation_relative'],comm);priorpx=old['adaptive384__projected_input'].astype('f8');priorlat=old['adaptive384__latent'].astype('f8');inputcontrol=rel(px,priorpx);assert inputcontrol<=b['numeric']['projected_relative'];scalar(item['prior_input_relative'],inputcontrol);yc=np.concatenate([(priorlat[:,offs[h]:offs[h+1]]-dd[h]*priorpx[:,offs[h]:offs[h+1]])@v[h].T+skip[:,h] for h in range(48)],axis=1);yj=np.concatenate([(lat[:,offs[h]:offs[h+1]]-dd[h]*px[:,offs[h]:offs[h+1]])@v[h].T+skip[:,h] for h in range(48)],axis=1);sourceinv=1/np.sqrt(np.mean((y*g)**2,axis=1,keepdims=True)+eps);ref=raw(f['output']);mm={}
            for arm,Y in [('channel384_state256',yc),('channel3072_state96',ys),('channel384_state96',yj)]:
                ug=Y*g;mm[arm]={}
                for mode in b['modes']:
                    inv=sourceinv if mode=='full_source_denominator' else 1/np.sqrt(np.mean(ug**2,axis=1,keepdims=True)+eps);inp=decode(bf16(ug*inv*gamma));op=np.empty((n,2048))
                    for lo in range(0,n,128):guard();hi=min(lo+128,n);op[lo:hi]=decode(bf16(inp[lo:hi]@ow.T))
                    bits=old['adaptive384__full_skip__'+mode] if arm=='channel384_state256' else z[arm+'__'+mode];assert bits.shape==(n,2048);actual=decode(bits);array_check(actual,op,b['numeric']['audit_output_relative'],'output_'+arm+'_'+mode);outputwords+=actual.size;mm[arm][mode]=metrics(actual,ref,rec['positions'])
                    for key,value in mm[arm][mode].items():scalar(item['metrics'][arm][mode][key],value)
                    if arm=='channel384_state256':assert max(abs(mm[arm][mode][k]-pr['arms']['adaptive384']['full_skip'][mode][k]) for k in mm[arm][mode])<=b['numeric']['baseline_metric_absolute']
            ec=yc-y;es=ys-y;ej=yj-y;interaction=ej-ec-es;energy={key:float(np.sum(q**2)) for key,q in [('channel_error',ec),('state_error',es),('joint_error',ej),('interaction',interaction)]};energy.update(channel_state_inner=float(np.sum(ec*es)),channel_interaction_inner=float(np.sum(ec*interaction)),state_interaction_inner=float(np.sum(es*interaction)),source_y=float(np.sum(y**2)))
            for key,value in energy.items():scalar(item['energy'][key],value)
            en=energy;err=abs(en['joint_error']-(en['channel_error']+en['state_error']+en['interaction']+2*(en['channel_state_inner']+en['channel_interaction_inner']+en['state_interaction_inner'])))/max(en['joint_error'],1e-30);assert err<=1e-10 and item['energy_identity_relative']<=1e-10;rows.append(dict(id=rec['id'],split=rec['split'],domain=rec['domain'],metrics=mm));print(json.dumps(dict(stage=stage,complete=len(rows),seconds=time.perf_counter()-CLOCK)),flush=True)
        for arm in r['aggregates']:
            for mode in b['modes']:
                for split in ['FIT','DEV']:
                    sr=[q for q in rows if q['split']==split];assert len(sr)==24;vals=[q['metrics'][arm][mode]['output_relative_RMS'] for q in sr];c=float(np.mean([q['metrics'][arm][mode]['centered_output_relative_RMS'] for q in sr]));dm={d:float(np.mean([q['metrics'][arm][mode]['output_relative_RMS'] for q in sr if q['domain']==d])) for d in {q['domain'] for q in sr}};actual=r['aggregates'][arm][mode][split];fresh=dict(mean=float(np.mean(vals)),worst=max(vals),centered_mean=c,worst_domain_mean=max(dm.values()))
                    for key,value in fresh.items():scalar(actual[key],value)
                    assert set(dm)==set(actual['domain_means'])
                    for key,value in dm.items():scalar(actual['domain_means'][key],value)
                    if split=='DEV':assert r['gates'][arm][mode]==dict(mean=fresh['mean']<=.1,worst=fresh['worst']<=.2,centered=c<=.1,domain=fresh['worst_domain_mean']<=.2)
        expectedcost=dict(source_state_coordinates=3072*256,joint_state_coordinates=384*96,state_coordinate_ratio=(3072*256)/(384*96),F32_source_state_bytes=4*3072*256,F32_joint_state_bytes=4*384*96,original_engine_state_coordinates=512*96,source_feature_channels_retained=3072,source_gate_channels_retained=3072,source_heads_retained=48,source_conv_width_retained=4);assert r['algebraic_state_costs']==expectedcost and r['new_scans']==96 and all(r[k]==0 for k in ['basis_fits','optimizer_updates','source_forwards','native_calls']);stream.close();stream=None;check();guard();ad=dict(schema='RECURRENT_AXIS_COMPOSITION_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,report=re,terminal=te,binding=r['binding'],cases_verified=48,full_sequential_scan_words_verified=seqwords,all_output_words_verified=outputwords,projected_words_verified=projectedwords,maximum=maximum,before_after_hashes_exact=True,decision='INDEPENDENT_STORED_PASS',seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope='All fixed map identities,source3072x96 and joint384x96 states/every timestep,B/C/input projections,commutation,all six BF16 outputs per case incl prior baseline,error interaction energies/full metrics/domains/gates and algebraic state counts. No source F32/native/whole-model/n/50 admission.')
        write(a.out,ad);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(ad),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),stage=stage,traceback=traceback.format_exc());raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['report','terminal','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['report-sha','terminal-sha','freeze']:p.add_argument('--'+name,required=True)
    main(p.parse_args())
