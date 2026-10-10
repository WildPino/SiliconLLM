"""Saved-gradient grouped displacements, original exports and actual C descent."""
import argparse,gc,json,math,os,struct,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from original_packed_capacity import SITE,extent,sha,raw
sys.path.insert(0,str(SITE))
from original_categorical_causal_preflight import emit,array,read,tensor_info
from original_falcon_whole_recovery import memory_reader
from original_joint_history_recovery_audit import packed_fields,master_name
from original_categorical_loss_compile import decode,probabilities
V,D,E=65537,256,1152

def bind(a):
    import numpy as np,psutil,torch
    import original_wide_learner
    assert not a.binding.exists()
    bp=DOC/'original_categorical_causal_preflight_reviewed_binding_20261010.json';rp=DOC/'original_categorical_causal_preflight_result_20261010.json';ap=DOC/'original_categorical_causal_preflight_stored_adjudication_20261010.json'
    b,r,ar=[json.loads(p.read_bytes()) for p in (bp,rp,ap)]
    assert ar['decision']=='CAUSAL_BRIDGE_QUALIFIED_PENDING_TRAINING' and all(ar['numeric_flags'].values()) and ar['result']==extent(rp) and ar['binding']==extent(bp)
    for p in (rp,ap):
        t=json.loads(p.with_suffix('.terminal.json').read_bytes());assert t['exit_code']==0 and t['error'] is None and t['inputs_before_after_exact'] and t['result']==extent(p)
    files=[Path(__file__),a.protocol,bp,rp,ap,rp.with_suffix('.terminal.json'),ap.with_suffix('.terminal.json'),Path(b['source_state']['path']),Path(b['executable']['path']),Path(b['original_bodies']['source']['path']),Path(b['head']['path']),Path(b['final_norm']['path']),Path(b['record']['logits']['path']),*[Path(r['artifacts'][k]['path']) for k in ('gradients','packed','native_scores','native_routes','native_witness','query')]]
    for m in list(sys.modules.values()):
        for k in ('__file__','__cached__'):
            p=getattr(m,k,None)
            if isinstance(p,(str,os.PathLike)) and Path(p).is_file():files.append(Path(p))
    files += [Path(sys.executable),Path(sys.executable).parent/'python312.dll']+sorted((SITE/'torch/lib').glob('*.dll'))+sorted((SITE/'numpy.libs').glob('*.dll'))+sorted((SITE/'psutil').glob('*.pyd'))
    inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in files)]
    for item in (b['source_state'],b['head'],b['final_norm'],b['executable'],*r['artifacts'].values()):
        if Path(item['path']).resolve() in {Path(i['path']) for i in inputs}:assert extent(item['path'])=={k:item[k] for k in ('path','bytes','sha256')}
    assert (torch.__version__,np.__version__,psutil.__version__)==('2.6.0+cu124','2.4.6','7.2.2')
    emit(a.binding,dict(schema='ORIGINAL_CATEGORICAL_TRUST_STEP_BINDING_V1',inputs=inputs,runtime=dict(Torch=torch.__version__,NumPy=np.__version__,psutil=psutil.__version__,junction_resolved=True),parent_binding=extent(bp),parent_result=extent(rp),parent_audit=extent(ap),source_state=b['source_state'],gradient_source=r['artifacts']['gradients'],baseline=r['artifacts'],baseline_diagnostics=r['diagnostics'],source_parameters=r['parameters'],source_gradient_stats=r['gradient_stats'],record=b['record'],head=b['head'],final_norm=b['final_norm'],executable=b['executable'],fields=b['fields'],original_bodies=b['original_bodies'],
        alphas=[1e-4,1e-3,1e-2],RMS_radius_floor=1e-5,chunk_groups=32,group_stat_columns=['parameter_L2','gradient_L2','radius','actual_delta_L2','actual_delta_dot_gradient','moved_coefficients'],
        numeric=dict(group_relative=1e-9,aggregate_relative=1e-9,proposal_max_F32_ULP=1,relative_radius_slack=1e-6,metric_absolute=1e-8,routing_mass_defect=1e-6),selection=dict(weighted_KL_relative=.99,first_label_KL_relative=1.,quality_mean_KL=.05,quality_disagreement=.05),
        capture_limits=dict(seconds=1800,reserve_seconds=180,OS_bytes=24<<30,output_bytes=12<<30,log_bytes=8<<20,child_seconds=100),audit_limits=dict(seconds=1800,OS_bytes=20<<30,output_bytes=2<<20,log_bytes=8<<20),
        scope='Three independently changed models from SAME qualified saved gradient, not three sequential optimization steps. CPU F64 proposal/F32 masters/original exporter/C; no model forward/backward/Adam/GPU/source/DEV/RESERVED/T4 or baseline replay. Single FIT descent is not useful chatbot/generalization/speed admission.'))
    print(json.dumps(dict(binding_sha=sha(a.binding),inputs=len(inputs),bytes=sum(i['bytes'] for i in inputs))),flush=True)

def grouped(name,shape):
    return name=='embed' or '.bank.' in name

def proposal_blocks(name,old,gradient,alpha,b,independent=False):
    import numpy as np
    shape=tuple(old.shape);isgroup=grouped(name,shape);groups=shape[0] if isgroup else 1;dim=math.prod(shape)//groups
    p=old.numpy().reshape(groups,dim);g=gradient.numpy().reshape(groups,dim)
    for first in range(0,groups,b['chunk_groups']):
        last=min(first+b['chunk_groups'],groups);w=p[first:last].astype('f8');v=g[first:last].astype('f8');assert np.isfinite(w).all() and np.isfinite(v).all()
        if independent:
            pn=np.sqrt(np.sum(w.astype(np.longdouble)**2,axis=1)).astype('f8');gn=np.sqrt(np.sum(v.astype(np.longdouble)**2,axis=1)).astype('f8')
        else:pn=np.sqrt(np.sum(w*w,axis=1));gn=np.sqrt(np.sum(v*v,axis=1))
        radius=np.maximum(pn,b['RMS_radius_floor']*math.sqrt(dim));scale=np.divide(alpha*radius,gn,out=np.zeros_like(gn),where=gn>0)
        x=(w-scale[:,None]*v).astype('<f4');x[gn==0]=p[first:last][gn==0];delta=x.astype('f8')-w
        stats=np.column_stack((pn,gn,radius,np.sqrt(np.sum(delta*delta,axis=1)),np.sum(delta*v,axis=1),np.count_nonzero(delta,axis=1)))
        yield first,last,x,stats

def proposal(name,old,g,alpha,b):
    import numpy as np,torch
    shape=tuple(old.shape);groups=shape[0] if grouped(name,shape) else 1;dim=math.prod(shape)//groups;out=np.empty((groups,dim),dtype='<f4');rows=[]
    for first,last,x,s in proposal_blocks(name,old,g,alpha,b):out[first:last]=x;rows.append(s)
    return torch.from_numpy(out.reshape(shape)),np.concatenate(rows)

class View:
    e,v=E,V
    def __init__(self,values):self.values=values
    def parameters(self):return self.values.values()
    def tensor(self,name):return self.values[master_name(name)]

def packing(model,path,b):
    """Full original ABI, quantizer/scales, all-symbol drift and native witness."""
    import numpy as np,torch
    fields=packed_fields(path);assert fields=={k:{**v,'shape':tuple(v['shape'])} for k,v in b['fields'].items()};baseline=b['baseline']['packed']['path'];checked=0;changed=scale_changed=certified=cert_changed=0
    with Path(path).open('rb') as x,Path(baseline).open('rb') as y:assert x.read(11520)==y.read(11520)
    oldmodel=torch.load(b['source_state']['path'],weights_only=True,map_location='cpu',mmap=True)['model']
    for name,f in fields.items():
        data=np.memmap(path,dtype='<f4' if f['dtype']==1 else 'u1',mode='r',offset=f['offset'],shape=f['shape'])
        if name.endswith('_scale'):del data;continue
        p=model[master_name(name[:-5] if name.endswith('_code') else name)]
        if name.endswith('_code'):
            base=name[:-5];sf=fields[base+'_scale'];scales=np.memmap(path,dtype='<f4',mode='r',offset=sf['offset'],shape=sf['shape']);prior=np.memmap(baseline,dtype='u1',mode='r',offset=f['offset'],shape=f['shape']);oscale=np.memmap(baseline,dtype='<f4',mode='r',offset=sf['offset'],shape=sf['shape'])
            for first in range(0,E,32):
                last=min(first+32,E);w=p[first:last];s=w.abs().mean(-1).clamp_min(1e-5);q=(w/s.unsqueeze(-1)).round().clamp(-1,1).numpy().astype('i1');code=((q[...,0::2].astype('i2')+1)*3+q[...,1::2]+1).astype('u1')
                saved=data[first:last].transpose(0,2,1) if base.endswith('.down') else data[:,first*128:last*128].T.reshape(last-first,128,128)
                oldcode=prior[first:last].transpose(0,2,1) if base.endswith('.down') else prior[:,first*128:last*128].T.reshape(last-first,128,128)
                assert np.array_equal(code,saved) and np.array_equal(s.numpy().view('<u4'),scales[first:last].view('<u4'))
                oq=np.empty_like(q);oq[...,0::2]=oldcode//3-1;oq[...,1::2]=oldcode%3-1
                ow=oldmodel[master_name(base)][first:last];so=ow.abs().mean(-1).clamp_min(1e-5);originalq=(ow/so.unsqueeze(-1)).round().clamp(-1,1).numpy().astype('i1');assert np.array_equal(originalq,oq) and np.array_equal(so.numpy().view('<u4'),oscale[first:last].view('<u4'))
                wn=w.numpy().astype('f8');on=ow.numpy().astype('f8');ss=s.numpy().astype('f8')[...,None];sso=so.numpy().astype('f8')[...,None];margin=np.minimum(abs(on-.5*sso),abs(on+.5*sso));u=2.**-24;bound=abs(wn-on)+.5*abs(ss-sso)+u/(1-u)*np.maximum(ss,sso)
                mask=margin>bound;different=q!=oq;certified+=int(mask.sum());cert_changed+=int(np.count_nonzero(mask&different));changed+=int(different.sum());scale_changed+=int(np.count_nonzero(s.numpy().view('<u4')!=so.numpy().view('<u4')))
                del w,s,q,code,saved,oldcode,oq,ow,so,originalq,wn,on,ss,sso,margin,bound,mask,different
            assert data.min()>=0 and data.max()<=8;checked+=2;del scales,prior,oscale
        else:assert tuple(p.shape)==f['shape'] and np.array_equal(p.numpy().view('<u4'),data.view('<u4'));checked+=1
        del data,p
    assert checked==110
    values=[]
    for expert in (0,31,32,1023,1024,1151):
        for site in range(6):
            for organ,length in (('gate',256),('up',256),('down',128)):
                w=model[f'layers.{site}.bank.{organ}'][expert];s=w.abs().mean(-1).clamp_min(1e-5);q=(w/s.unsqueeze(-1)).round().clamp(-1,1).numpy().astype('i8');operand=np.arange(length,dtype='i8');operand=operand%127-63 if organ!='down' else (operand*7)%127-63;values.append(q@operand)
    witness=np.concatenate(values).astype('<i4').tobytes();assert len(witness)==73728
    return dict(all110_fields_exact=True,ternary_symbols=679477248,changed_symbols=changed,changed_scales=scale_changed,certified_unchanged_symbols=certified,changed_among_certified=cert_changed),witness

def metrics(b,scores,routes,independent):
    import numpy as np
    rec=b['record'];q,lq=probabilities(decode(np.fromfile(rec['logits']['path'],dtype='<u2').reshape(18,V)),independent);s=np.fromfile(scores,dtype='<f4').reshape(58,V);assert np.isfinite(s).all();p,lp=probabilities(s[rec['positions']].astype('f8'),independent);kl=np.sum(q*(lq-lp),axis=1);w=np.full(18,.5/17);w[0]=.5
    typ=np.dtype([('ids','<i4',(8,)),('mass','<f4',(8,))]);actual=np.fromfile(routes,dtype=typ).reshape(58,6);prior=np.fromfile(b['baseline']['native_routes']['path'],dtype=typ).reshape(58,6);ids,mass=actual['ids'],actual['mass'];assert ids.min()>=0 and ids.max()<E and (np.diff(np.sort(ids,axis=-1),axis=-1)>0).all() and np.isfinite(mass).all() and (mass>=0).all();defect=float(np.max(abs(mass.astype('f8').sum(-1)-1)))
    return dict(KL_per_label=kl.tolist(),mean_KL=float(kl.mean()),weighted_KL=float(kl@w),first_KL=float(kl[0]),disagreement=int(np.count_nonzero(s[rec['positions']].argmax(1)!=q.argmax(1))),route_ID_slots_changed=int(np.count_nonzero(ids!=prior['ids'])),route_sets_changed=int(np.count_nonzero(np.any(np.sort(ids,axis=-1)!=np.sort(prior['ids'],axis=-1),axis=-1))),route_mass_delta_max=float(np.max(abs(mass.astype('f8')-prior['mass'].astype('f8')))),route_mass_defect=defect,route_unions=[int(np.unique(ids[:,site]).size) for site in range(6)])

def selection(b,rows,base):
    best=min(rows,key=lambda x:(x['metrics']['weighted_KL'],x['alpha']));m=best['metrics'];flags=dict(weighted_descent=m['weighted_KL']<=b['selection']['weighted_KL_relative']*base['weighted_KL'],first_not_regressed=m['first_KL']<=b['selection']['first_label_KL_relative']*base['first_KL'],all_numeric=all(all(r['numeric_flags'].values()) for r in rows))
    return dict(alpha=best['alpha'],flags=flags,decision='GROUPED_DISCRETE_DESCENT_QUALIFIED' if all(flags.values()) else 'GROUPED_DISCRETE_DESCENT_NOT_QUALIFIED',quality_flags=dict(mean_KL=m['mean_KL']<=b['selection']['quality_mean_KL'],disagreement=m['disagreement']/18<=b['selection']['quality_disagreement']))

def capture(a):
    import numpy as np,psutil,torch
    from original_wide_learner import export
    b=json.loads(a.binding.read_bytes());assert sha(a.binding)==a.binding_sha;start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6);torch.set_num_interop_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES']=='';a.directory.mkdir(exist_ok=False);native_peak=0;stage='restore';rows=[];reader=memory_reader()
    def guard():assert time.monotonic()-start<b['capture_limits']['seconds']-b['capture_limits']['reserve_seconds'] and proc.memory_info().peak_wset+native_peak<=b['capture_limits']['OS_bytes']
    try:
        state=torch.load(b['source_state']['path'],map_location='cpu',weights_only=True,mmap=True);assert state['schema']=='ORIGINAL_DELTA_SEEDED_STATE_V1' and state['updates']==27 and not state['optimizer_partial_possible'];old=state['model'];grads=torch.load(b['gradient_source']['path'],map_location='cpu',weights_only=True,mmap=True);assert len(old)==92 and len(grads)==90 and set(grads)==set(old)-{'head','final_norm'}
        base=metrics(b,b['baseline']['native_scores']['path'],b['baseline']['native_routes']['path'],False);assert max(abs(np.asarray(base['KL_per_label'])-b['baseline_diagnostics']['native_KL']))<=b['numeric']['metric_absolute']
        for alpha in b['alphas']:
            stage=f'proposal_{alpha}';directory=a.directory/f'alpha_{alpha:g}';directory.mkdir();model={};stats={};art={};dot=0.;ideal=0.;maxrad=0.;moved=0
            print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start)),flush=True)
            for name,p in old.items():
                if name in ('head','final_norm'):model[name]=torch.from_numpy(read(b[name]).copy())
                else:
                    value,s=proposal(name,p,grads[name],alpha,b);model[name]=value;stats[name]=array(directory,name.replace('.','_')+'.group_stats',s);dot+=float(s[:,4].sum());ideal-=alpha*float((s[:,2]*s[:,1]).sum());maxrad=max(maxrad,float((s[:,3]/s[:,2]).max()));moved+=int(s[:,5].sum())
                guard()
            parameters={name:tensor_info(p) for name,p in model.items()};path=directory/'masters.pt';torch.save(dict(schema='ORIGINAL_CATEGORICAL_GROUPED_MASTERS_V1',alpha=alpha,binding_sha256=a.binding_sha,source_state=b['source_state'],gradient_source=b['gradient_source'],model=model,optimizer_updates=0,baseline_updates=27,saved_gradient_displacements=1),path);art['masters']=extent(path);stage=f'export_{alpha}';print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start)),flush=True);packed=directory/'candidate.packed';export(View(model),packed);art['packed']=extent(packed);quant,witness=packing(model,packed,b);raw(directory/'expected.witness',witness);art['expected_witness']=extent(directory/'expected.witness');guard()
            stage=f'native_{alpha}';query=directory/'query.u32';raw(query,Path(b['baseline']['query']['path']).read_bytes());art['query']=extent(query);paths=[directory/x for x in ('native.scores.f32','native.routes','native.witness','native.summary.json')];command=[b['executable']['path'],'--prefix',str(packed),str(query),*[str(p) for p in paths]];began=time.monotonic();peak=0
            with (directory/'native.log').open('xb') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=8);created=psutil.Process(child.pid).create_time()
                try:
                    while child.poll() is None:peak=max(peak,reader(child));native_peak=max(native_peak,peak);guard();assert time.monotonic()-began<b['capture_limits']['child_seconds'];time.sleep(.1)
                    peak=max(peak,reader(child));native_peak=max(native_peak,peak);assert child.returncode==0
                finally:
                    if child.poll() is None:child.kill();child.wait()
                    emit(directory/'native.receipt.json',dict(command=command,pid=child.pid,creation_time=created,exit_code=child.returncode,OS_peak=peak,seconds=time.monotonic()-began))
            for key,p in zip(('native_scores','native_routes','native_witness','native_summary'),paths):art[key]=extent(p)
            assert paths[2].read_bytes()==witness;met=metrics(b,paths[0],paths[1],False);flags=dict(group_radius=maxrad<=alpha+b['numeric']['relative_radius_slack'],actual_direction_descent=dot<0,quantization_certificate=quant['changed_among_certified']==0,route_mass=met['route_mass_defect']<=b['numeric']['routing_mass_defect'])
            row=dict(alpha=alpha,artifacts=art,group_stats=stats,parameters=parameters,actual_dot_gradient=dot,ideal_dot_gradient=ideal,max_relative_displacement=maxrad,moved_coefficients=moved,quantization=quant,metrics=met,numeric_flags=flags);emit(directory/'candidate.json',row);rows.append(row);print(json.dumps(dict(stage=stage,alpha=alpha,metrics=met,quantization=quant,actual_dot=dot,seconds=time.monotonic()-start)),flush=True);del model,parameters,stats,value,s;gc.collect();guard()
        chosen=selection(b,rows,base);emit(a.out,dict(schema='ORIGINAL_CATEGORICAL_TRUST_STEP_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,decision=chosen['decision'],selection=chosen,baseline_metrics=base,records=rows,native_OS_peak=native_peak,worker_OS_peak=proc.memory_info().peak_wset,seconds=time.monotonic()-start,proposals=3,parameter_displacements=3,native_histories=3,native_positions=174,paired_labels=54,model_forwards=0,backward_calls=0,optimizer_updates=0,source_calls=0,GPU_calls=0,DEV_queries=0,reserved_queries=0,T4_calls=0,quality_admission=False,speed_admission=False,scope=b['scope']));guard()
    except BaseException as error:emit(a.directory/'first_fault.json',dict(stage=stage,error=repr(error),completed=rows,seconds=time.monotonic()-start));raise

def ulp_order(x):
    import numpy as np
    signed=x.view('<i4').astype('i8');return np.where(signed<0,-2147483648-signed,signed)

def audit(a):
    import numpy as np,psutil,torch
    b=json.loads(a.binding.read_bytes());r=json.loads(a.source_result.read_bytes());t=json.loads(a.source_result.with_suffix('.terminal.json').read_bytes());assert sha(a.binding)==a.binding_sha==r['binding_sha256']==t['binding_sha256'] and r['freeze']==a.freeze==t['freeze'] and t['exit_code']==0 and t['error'] is None and t['inputs_before_after_exact'] and t['result']==extent(a.source_result)
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([1]);torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES']==''
    def guard():assert time.monotonic()-start<b['audit_limits']['seconds'] and proc.memory_info().peak_wset<=b['audit_limits']['OS_bytes'] and not proc.children(recursive=True)
    for item in b['inputs']+t['outputs']:assert extent(item['path'])==item;guard()
    by={i['path']:i for i in t['outputs']};old=torch.load(b['source_state']['path'],weights_only=True,map_location='cpu',mmap=True)['model'];grads=torch.load(b['gradient_source']['path'],weights_only=True,map_location='cpu',mmap=True);audited=[];maximum_metric=0.;maximum_ulp=0
    assert len(old)==92 and len(grads)==90 and set(grads)==set(old)-{'head','final_norm'}
    for name,value in grads.items():assert tensor_info(value)==b['source_gradient_stats'][name];guard()
    for name,value in old.items():
        if name not in ('head','final_norm'):assert tensor_info(value)==b['source_parameters'][name]
        guard()
    base=metrics(b,b['baseline']['native_scores']['path'],b['baseline']['native_routes']['path'],True);assert max(abs(np.asarray(base['KL_per_label'])-r['baseline_metrics']['KL_per_label']))<=b['numeric']['metric_absolute']
    assert [row['alpha'] for row in r['records']]==b['alphas']
    for row in r['records']:
        for item in list(row['artifacts'].values())+list(row['group_stats'].values()):assert by[item['path']]=={k:item[k] for k in ('path','bytes','sha256')}
        alpha=row['alpha'];saved=torch.load(row['artifacts']['masters']['path'],weights_only=True,map_location='cpu',mmap=True);model=saved['model'];assert saved['schema']=='ORIGINAL_CATEGORICAL_GROUPED_MASTERS_V1' and saved['alpha']==alpha and saved['binding_sha256']==a.binding_sha and saved['source_state']==b['source_state'] and saved['gradient_source']==b['gradient_source'] and saved['optimizer_updates']==0 and saved['baseline_updates']==27 and saved['saved_gradient_displacements']==1 and set(model)==set(old)
        dot=ideal=maxrad=0.;moved=0;max_group_error=0.;case_ulp=0
        for name,p in model.items():
            assert tensor_info(p)==row['parameters'][name]
            if name in ('head','final_norm'):assert np.array_equal(p.numpy().view('<u4'),read(b[name]).view('<u4'));continue
            saved_stats=read(row['group_stats'][name]);groups=p.shape[0] if grouped(name,p.shape) else 1;actual=p.numpy().reshape(groups,-1);assert saved_stats.shape==(groups,6)
            for first,last,predicted,ind_stats in proposal_blocks(name,old[name],grads[name],alpha,b,True):
                got=actual[first:last];case_ulp=max(case_ulp,int(np.max(abs(ulp_order(got)-ulp_order(predicted)))));pw=old[name].numpy().reshape(groups,-1)[first:last].astype('f8');gv=grads[name].numpy().reshape(groups,-1)[first:last].astype('f8');delta=got.astype('f8')-pw
                ind_stats[:,3]=np.sqrt(np.sum(delta.astype(np.longdouble)**2,axis=1)).astype('f8');ind_stats[:,4]=np.sum(delta.astype(np.longdouble)*gv.astype(np.longdouble),axis=1).astype('f8');ind_stats[:,5]=np.count_nonzero(delta,axis=1)
                reference=saved_stats[first:last];err=float(np.max(abs(ind_stats-reference)/np.maximum(1.,abs(reference))));max_group_error=max(max_group_error,err);assert err<=b['numeric']['group_relative'];dot+=float(ind_stats[:,4].sum());ideal-=alpha*float((ind_stats[:,2]*ind_stats[:,1]).sum());maxrad=max(maxrad,float((ind_stats[:,3]/ind_stats[:,2]).max()));moved+=int(ind_stats[:,5].sum());guard()
            guard()
        assert case_ulp<=b['numeric']['proposal_max_F32_ULP'] and moved==row['moved_coefficients']
        for value,key in ((dot,'actual_dot_gradient'),(ideal,'ideal_dot_gradient'),(maxrad,'max_relative_displacement')):assert abs(value-row[key])<=b['numeric']['aggregate_relative']*max(1.,abs(row[key]))
        quant,witness=packing(model,row['artifacts']['packed']['path'],b);assert quant==row['quantization'] and witness==Path(row['artifacts']['native_witness']['path']).read_bytes()==Path(row['artifacts']['expected_witness']['path']).read_bytes()
        query=Path(row['artifacts']['query']['path']).read_bytes();assert query==struct.pack('<I',58)+np.asarray(b['record']['student_input_ids'],dtype='<u4').tobytes();directory=Path(row['artifacts']['packed']['path']).parent;receipt_path=directory/'native.receipt.json';assert extent(receipt_path)==by[str(receipt_path.resolve())];receipt=json.loads(receipt_path.read_bytes());expected=[b['executable']['path'],'--prefix',row['artifacts']['packed']['path'],row['artifacts']['query']['path'],*[row['artifacts'][k]['path'] for k in ('native_scores','native_routes','native_witness','native_summary')]];assert receipt['command']==expected and receipt['exit_code']==0 and receipt['seconds']<=b['capture_limits']['child_seconds'] and receipt['OS_peak']<=r['native_OS_peak']
        met=metrics(b,row['artifacts']['native_scores']['path'],row['artifacts']['native_routes']['path'],True);metric_error=max(abs(np.asarray(met['KL_per_label'])-row['metrics']['KL_per_label']));maximum_metric=max(maximum_metric,float(metric_error));assert metric_error<=b['numeric']['metric_absolute']
        for k,v in met.items():
            if k=='KL_per_label':continue
            if isinstance(v,float):assert abs(v-row['metrics'][k])<=b['numeric']['metric_absolute']
            else:assert v==row['metrics'][k]
        flags=dict(group_radius=maxrad<=alpha+b['numeric']['relative_radius_slack'],actual_direction_descent=dot<0,quantization_certificate=quant['changed_among_certified']==0,route_mass=met['route_mass_defect']<=b['numeric']['routing_mass_defect']);assert flags==row['numeric_flags'];audited.append(dict(alpha=alpha,metrics=met,numeric_flags=flags,max_proposal_ULP=case_ulp,max_group_relative_error=max_group_error));maximum_ulp=max(maximum_ulp,case_ulp);print(json.dumps(dict(audited_alpha=alpha,seconds=time.monotonic()-start)),flush=True);del model,saved,p,got,pw,gv,delta;gc.collect();guard()
    chosen=selection(b,audited,base);assert chosen==r['selection'] and chosen['decision']==r['decision'];assert r['proposals']==r['native_histories']==3 and r['native_positions']==174 and r['paired_labels']==54 and all(r[k]==0 for k in ('model_forwards','backward_calls','optimizer_updates','source_calls','GPU_calls','DEV_queries','reserved_queries','T4_calls')) and not r['quality_admission'] and not r['speed_admission'];assert r['worker_OS_peak']+r['native_OS_peak']<=b['capture_limits']['OS_bytes'] and r['seconds']<b['capture_limits']['seconds']-b['capture_limits']['reserve_seconds'] and sum(i['bytes'] for i in t['outputs'])+a.source_result.stat().st_size<=b['capture_limits']['output_bytes'];guard()
    emit(a.out,dict(schema='ORIGINAL_CATEGORICAL_TRUST_STEP_AUDIT_V1',result=extent(a.source_result),binding=extent(a.binding),decision=chosen['decision'],selection=chosen,complete_input_output_hashes=True,all3_92_master_states_110_fields_verified=True,all90_gradient_source_tensors_verified=True,all_group_updates_longdouble_norms_verified=True,max_proposal_ULP=maximum_ulp,max_metric_delta=maximum_metric,records=audited,model_forwards=0,backward_calls=0,optimizer_updates=0,native_calls=0,GPU_calls=0,source_calls=0,seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,scope='Stored arrays/weights only: independent proposal arithmetic, complete original packing/witness checks and shifted logaddexp metrics; no original forward/backward/native replay.'))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('bind','capture-worker','audit-worker'):p.add_argument('--'+k,action='store_true')
    for k in ('binding','directory','out','source-result','protocol'):p.add_argument('--'+k,type=Path)
    for k in ('binding-sha','freeze'):p.add_argument('--'+k)
    a=p.parse_args()
    if a.bind:bind(a)
    elif a.capture_worker:capture(a)
    elif a.audit_worker:audit(a)
    else:raise RuntimeError('Use held CPU/native launcher')
