"""Finite full-history conversion geometry and data checks, without a runner."""
import gc,hashlib,json,math,struct,time
from pathlib import Path
from original_categorical_causal_preflight import array,emit,read,tensor_info,gradient_role
from original_categorical_loss_compile import probabilities,decode
from original_categorical_trust_step import View,proposal,proposal_blocks,ulp_order
from original_joint_history_recovery_audit import packed_fields,master_name
from original_master_transition import bits,digest,file_digest

V,D,E,L,K=65537,256,1152,6,8


def routes(item,length):
    import numpy as np
    dtype=np.dtype([('ids','<i4',(K,)),('mass','<f4',(K,))])
    assert item['bytes']==length*L*dtype.itemsize
    x=np.fromfile(item['path'],dtype=dtype).reshape(length,L)
    ids,mass=x['ids'],x['mass'];assert np.isfinite(mass).all()
    valid=bool(np.all((ids>=0)&(ids<E)) and np.all(np.diff(np.sort(ids,axis=-1),axis=-1)>0) and np.all(mass>=0))
    return ids,mass,valid,float(np.max(abs(mass.astype('f8').sum(-1)-1)))


def native_metrics(rec,native,independent=False):
    import numpy as np
    n=len(rec['positions']);length=len(rec['student_input_ids'])
    bits=np.fromfile(rec['logits']['path'],dtype='<u2').reshape(n,V)
    q,lq=probabilities(decode(bits),independent)
    s=np.fromfile(native['scores']['path'],dtype='<f4').reshape(length,V);assert np.isfinite(s).all()
    _,lp=probabilities(s[rec['positions']].astype('f8'),independent)
    kl=np.sum(q*(lq-lp),axis=1);weights=np.full(n,.5/(n-1));weights[0]=.5
    ids,mass,valid,defect=routes(native['routes'],length)
    return dict(KL_per_label=kl.tolist(),mean_KL=float(kl.mean()),weighted_KL=float(kl@weights),
        first_KL=float(kl[0]),disagreement=int(np.count_nonzero(s[rec['positions']].argmax(1)!=decode(bits).argmax(1))),
        labels=n,route_valid=valid,route_mass_defect=defect,
        route_unions=[int(np.unique(ids[:,site]).size) for site in range(L)])


def choose(rows,baseline,baseline_valid=True):
    """Diagnostic numeric failure records rejection; no quality-dependent abort."""
    eligible=[r for r in rows if r['metrics']['weighted_KL']<=.99*baseline['weighted_KL']
              and r['metrics']['first_KL']<=baseline['first_KL']]
    best=min(eligible or rows,key=lambda r:(r['metrics']['weighted_KL'],r['alpha']))
    flags=dict(weighted_descent=best['metrics']['weighted_KL']<=.99*baseline['weighted_KL'],
        first_not_regressed=best['metrics']['first_KL']<=baseline['first_KL'],
        all_numeric=baseline_valid and all(all(row['numeric_flags'].values()) for row in rows))
    return dict(alpha=best['alpha'],accepted=all(flags.values()),flags=flags)


def stream_counters(request,expected_reuse,expected_reset):
    """Stored original ABI counters; final head is after consuming the last ID."""
    ids=request['query_ids'];output=request['generated_ids'];n=len(ids);generated=len(output)
    assert 1<=n<=8192 and 1<=generated<=64 and all(type(i) is int and 0<=i<V for i in ids+output)
    assert request['input_ids']==n and request['reused_prefix_ids']==expected_reuse and request['state_reset']==expected_reset
    assert request['core_calls']==n-expected_reuse+generated and request['head_calls']==request['core_calls']
    assert request['cached_ids']==n+generated and not any(i in (11,228) for i in output[:-1])
    assert request['eos']==(output[-1] if output[-1] in (11,228) else 0)
    assert all(request[k] is None for k in ('core_seconds','mlp_seconds','head_seconds'))
    timing={k:request[k] for k in ('request_seconds','prefill_seconds','decode_seconds','forward_seconds','load_seconds','pipe_request_seconds')}
    assert all(math.isfinite(v) and v>=0 for v in timing.values()) and timing['decode_seconds']>0 and timing['pipe_request_seconds']>0
    assert timing['request_seconds']+1e-8>=timing['prefill_seconds']+timing['decode_seconds']
    assert timing['prefill_seconds']+timing['decode_seconds']+1e-8>=timing['forward_seconds']
    assert timing['pipe_request_seconds']+1e-6>=timing['request_seconds']
    return generated


def pack_check(old,model,baseline,path,fields,guard=lambda:None):
    """Complete original110 fields, all symbols/scales and stable-cell checks."""
    import numpy as np,torch
    expected={k:{**v,'shape':tuple(v['shape'])} for k,v in fields.items()}
    assert packed_fields(path)==packed_fields(baseline)==expected
    with Path(path).open('rb') as x,Path(baseline).open('rb') as y:assert x.read(11520)==y.read(11520)
    changed=scales_changed=certified=cert_changed=checked=0
    for name,f in expected.items():
        data=np.memmap(path,dtype='<f4' if f['dtype']==1 else 'u1',mode='r',offset=f['offset'],shape=f['shape'])
        if name.endswith('_scale'):del data;continue
        key=master_name(name[:-5] if name.endswith('_code') else name);p=model[key]
        if name.endswith('_code'):
            base=name[:-5];sf=expected[base+'_scale']
            ss=np.memmap(path,dtype='<f4',mode='r',offset=sf['offset'],shape=sf['shape'])
            prior=np.memmap(baseline,dtype='u1',mode='r',offset=f['offset'],shape=f['shape'])
            os=np.memmap(baseline,dtype='<f4',mode='r',offset=sf['offset'],shape=sf['shape'])
            for first in range(0,E,32):
                last=min(first+32,E);w=p[first:last];ow=old[key][first:last]
                s=w.abs().mean(-1).clamp_min(1e-5);so=ow.abs().mean(-1).clamp_min(1e-5)
                q=(w/s.unsqueeze(-1)).round().clamp(-1,1).numpy().astype('i1')
                oq=(ow/so.unsqueeze(-1)).round().clamp(-1,1).numpy().astype('i1')
                code=((q[...,0::2].astype('i2')+1)*3+q[...,1::2]+1).astype('u1')
                oldcode=((oq[...,0::2].astype('i2')+1)*3+oq[...,1::2]+1).astype('u1')
                got=data[first:last].transpose(0,2,1) if base.endswith('.down') else data[:,first*128:last*128].T.reshape(last-first,128,128)
                prev=prior[first:last].transpose(0,2,1) if base.endswith('.down') else prior[:,first*128:last*128].T.reshape(last-first,128,128)
                assert np.array_equal(code,got) and np.array_equal(oldcode,prev)
                assert np.array_equal(s.numpy().view('<u4'),ss[first:last].view('<u4'))
                assert np.array_equal(so.numpy().view('<u4'),os[first:last].view('<u4'))
                wn=w.numpy().astype('f8');on=ow.numpy().astype('f8');snew=s.numpy().astype('f8')[...,None];sold=so.numpy().astype('f8')[...,None]
                margin=np.minimum(abs(on-.5*sold),abs(on+.5*sold));u=2.**-24
                bound=abs(wn-on)+.5*abs(snew-sold)+u/(1-u)*np.maximum(snew,sold)
                mask=margin>bound;different=q!=oq;certified+=int(mask.sum());cert_changed+=int(np.count_nonzero(mask&different))
                changed+=int(different.sum());scales_changed+=int(np.count_nonzero(s.numpy().view('<u4')!=so.numpy().view('<u4')));guard()
            assert data.min()>=0 and data.max()<=8;checked+=2;del ss,prior,os
        else:
            assert tuple(p.shape)==f['shape'] and np.array_equal(p.numpy().view('<u4'),data.view('<u4'));checked+=1
        del data,p;guard()
    assert checked==110
    values=[]
    for expert in (0,31,32,1023,1024,1151):
        for site in range(L):
            for organ,length in (('gate',256),('up',256),('down',128)):
                w=model[f'layers.{site}.bank.{organ}'][expert];s=w.abs().mean(-1).clamp_min(1e-5)
                q=(w/s.unsqueeze(-1)).round().clamp(-1,1).numpy().astype('i8');operand=np.arange(length,dtype='i8')
                operand=operand%127-63 if organ!='down' else (operand*7)%127-63;values.append(q@operand)
    witness=np.concatenate(values).astype('<i4').tobytes();assert len(witness)==73728
    return dict(all110_fields_exact=True,ternary_symbols=679477248,changed_symbols=changed,
        changed_scales=scales_changed,certified_unchanged_symbols=certified,changed_among_certified=cert_changed),witness


def backward_case(model,masters,rec,target,directory,guard,progress,tracker=None):
    """One new history/backward at current masters, two shared-head state checks."""
    import numpy as np,torch
    directory=Path(directory);directory.mkdir(exist_ok=False);art={};calls=[0]*L;hooks=[]
    tracker={} if tracker is None else tracker
    tracker.update(history_started=0,history_completed=0,backward_started=0,backward_completed=0,state_checks_started=0,state_checks_completed=0,block_calls=calls)
    for site,layer in enumerate(model.layers):
        layer.bank.progress=lambda mode,n,total,site=site:progress(mode,n,total,site)
        hooks.append(layer.register_forward_pre_hook(lambda module,args,site=site:calls.__setitem__(site,calls[site]+1)))
    model.zero_grad(set_to_none=True);ids=torch.tensor([rec['student_input_ids']],device='cuda')
    tracker['history_started']=1
    f=model.forward_features(ids,target['positions'].tolist());tracker['history_completed']=1
    f.retain_grad();fd=f.double();guard()
    for key,ix,dtype in (('forward_route_ids',0,'<i4'),('forward_route_mass',1,'<f4')):
        art[key]=array(directory,key,np.stack([layer.bank.last_routes[ix].numpy() for layer in model.layers],axis=1).astype(dtype))
    m=torch.from_numpy(target['moments'].copy()).cuda()[None];c=torch.from_numpy(target['negative_entropy'].copy()).cuda()[None]
    weights=torch.from_numpy(target['loss_weights'].copy()).cuda()[None]
    z,compiled,loss=model.categorical_objective(fd,m,c,weights)
    source=torch.from_numpy(decode(np.fromfile(rec['logits']['path'],dtype='<u2').reshape(len(rec['positions']),V))).cuda()[None]
    lq=torch.log_softmax(source,-1);q=lq.exp();dense=(q*(lq-torch.log_softmax(z,-1))).sum(-1)
    tracker['state_checks_started']+=1
    gd=torch.autograd.grad((dense*weights).sum(),fd,retain_graph=True)[0];tracker['state_checks_completed']+=1
    tracker['state_checks_started']+=1
    gc=torch.autograd.grad(loss,fd,retain_graph=True)[0];tracker['state_checks_completed']+=1
    tracker['backward_started']=1;loss.backward();torch.cuda.synchronize();tracker['backward_completed']=1;guard()
    for key,ix,dtype in (('backward_route_ids',0,'<i4'),('backward_route_mass',1,'<f4')):
        art[key]=array(directory,key,np.stack([layer.bank.last_routes[ix].numpy() for layer in model.layers],axis=1).astype(dtype))
    for key,value in [('features',f),('GPU_logits',z),('direct_loss',dense),('compiled_loss',compiled),
                      ('direct_gradient',gd),('compiled_gradient',gc),('backward_feature_gradient',f.grad)]:
        art[key]=array(directory,key,value.detach().cpu().numpy())
    parameters={};stats={};grads={};roles={}
    for name,p in model.named_parameters():
        assert np.array_equal(p.detach().cpu().numpy().view('<u4'),masters[name].numpy().view('<u4'))
        parameters[name]=tensor_info(p)
        if p.requires_grad:
            assert p.grad is not None;stats[name]=tensor_info(p.grad);grads[name]=p.grad.detach().cpu()
            role=gradient_role(name);roles[role]=roles.get(role,0.)+stats[name]['squared_norm']
        else:assert name in ('head','final_norm') and p.grad is None
        guard()
    assert len(stats)==90
    for hook in hooks:hook.remove()
    return dict(artifacts=art,parameters=parameters,gradient_stats=stats,gradient_role_squared_norms=roles,
                block_forward_calls=calls,outer_history_forwards=1,whole_backward_calls=1,state_gradient_checks=2),grads


def bridge(rec,target,H,native,history,numeric,independent=False):
    """Readout checks at ANY full FIT history length, no old-point error bound."""
    import numpy as np
    art=history['artifacts'];f,z,dense,comp,gd,gc,gf=[read(art[k]) for k in
        ('features','GPU_logits','direct_loss','compiled_loss','direct_gradient','compiled_gradient','backward_feature_gradient')]
    n=len(rec['positions']);t=len(rec['student_input_ids']);assert f.shape==gd.shape==gc.shape==gf.shape==(1,n,D)
    assert z.shape==(1,n,V) and dense.shape==comp.shape==(1,n)
    bits=np.fromfile(rec['logits']['path'],dtype='<u2').reshape(n,V);q,lq=probabilities(decode(bits),independent);p,lp=probabilities(z[0],independent)
    sn=np.fromfile(native['scores']['path'],dtype='<f4').reshape(t,V)[rec['positions']].astype('f8');_,lpn=probabilities(sn,independent)
    h=read(H).astype('f8');m=read(target['moments']);c=read(target['negative_entropy']);w=np.full(n,.5/(n-1));w[0]=.5
    shift=z[0]-z[0].max(1,keepdims=True);ln=np.logaddexp.reduce(shift,axis=1) if independent else np.log(np.exp(shift).sum(1))
    cl=c+z[0].max(1)+ln-np.sum(m*f[0].astype('f8'),axis=1)
    dg=((p-q)@h)*w[:,None];cg=(p@h-m)*w[:,None];kl=np.sum(q*(lq-lp),axis=1);nk=np.sum(q*(lq-lpn),axis=1)
    nums=dict(loss=float(np.max(abs(dense-comp))),gradient=float(np.max(abs(gd-gc))),upstream=float(np.max(abs(gf-gc.astype('<f4')))),
        CPU_score=float(np.max(abs(f[0].astype('f8')@h.T-z[0]))),CPU_dense_loss=float(np.max(abs(dense[0]-kl))),
        CPU_compiled_loss=float(np.max(abs(comp[0]-cl))),CPU_direct_gradient=float(np.max(abs(gd[0]-dg))),
        CPU_compiled_gradient=float(np.max(abs(gc[0]-cg))),native_score=float(np.max(abs(z[0]-sn))),
        native_KL=float(np.max(abs(kl-nk))),native_argmax=int(np.count_nonzero(z[0].argmax(1)!=sn.argmax(1))))
    ci,cm,valid,cdef=routes(native['routes'],t);fi,fm,bi,bm=[read(art[k]) for k in
        ('forward_route_ids','forward_route_mass','backward_route_ids','backward_route_mass')]
    assert fi.shape==fm.shape==bi.shape==bm.shape==(t,L,K)
    nums.update(route_ID_disagreements=int(np.count_nonzero(fi!=ci)),route_mass=float(np.max(abs(fm.astype('f8')-cm))),
        GPU_mass_defect=float(np.max(abs(fm.astype('f8').sum(-1)-1))),C_mass_defect=cdef,
        checkpoint_ID_bits=int(np.count_nonzero(fi!=bi)),checkpoint_mass_bits=int(np.count_nonzero(fm.view('<u4')!=bm.view('<u4'))))
    flags=dict(loss=nums['loss']<=numeric['loss_absolute'],gradient=nums['gradient']<=numeric['gradient_absolute'],
        upstream=nums['upstream']<=numeric['F32_upstream_absolute'],CPU_score=nums['CPU_score']<=numeric['loss_absolute'],
        CPU_losses=max(nums['CPU_dense_loss'],nums['CPU_compiled_loss'])<=numeric['loss_absolute'],
        CPU_gradients=max(nums['CPU_direct_gradient'],nums['CPU_compiled_gradient'])<=numeric['gradient_absolute'],
        native_score=nums['native_score']<=numeric['GPU_native_score_absolute'],native_KL=nums['native_KL']<=numeric['GPU_native_KL_absolute'],
        native_argmax=nums['native_argmax']==0,route_ID=nums['route_ID_disagreements']==0,
        route_mass=nums['route_mass']<=numeric['routing_mass_absolute'],route_normalization=max(cdef,nums['GPU_mass_defect'])<=numeric['routing_mass_defect'],
        route_valid=valid and bool(np.all((fi>=0)&(fi<E)) and np.all(np.diff(np.sort(fi,axis=-1),axis=-1)>0) and np.all(fm>=0)),
        checkpoint_routes=nums['checkpoint_ID_bits']==nums['checkpoint_mass_bits']==0,
        checkpoint_blocks=history['block_forward_calls']==[2]*L,
        gradient_roles_nonzero=all(history['gradient_role_squared_norms'].get(k,0)>0 for k in ('embed','core','bank','router')))
    return dict(numeric=nums,numeric_flags=flags)


def decode_transition(source,index_path,guard=lambda:None):
    """Independent integer-byte decoder returning exact target CPU tensors."""
    import numpy as np,torch,zlib
    path=Path(index_path);index=json.loads(path.read_bytes());blob=path.parent/'transition.bin'
    assert index['schema']=='ORIGINAL_MASTER_TRANSITION_V1' and index['chunk_bytes']>0 and index['chunk_bytes']%4==0
    assert file_digest(blob)==index['blob_sha256'] and blob.stat().st_size==index['blob_bytes']
    assert [r['name'] for r in index['parameters']]==sorted(source)
    out={};offset=total=moved=0;counts=dict(copy=0,xor_zlib=0,raw=0)
    with blob.open('rb') as stream:
        for row in index['parameters']:
            old,shape=bits(source[row['name']]);assert row['shape']==shape and row['dtype']=='<f4' and row['raw_bytes']==old.size*4
            dest=np.empty(old.shape,dtype='<u4');pos=0;oh=hashlib.sha256();nh=hashlib.sha256()
            for c in row['chunks']:
                n=c['raw_bytes'];assert 0<n<=index['chunk_bytes'] and n%4==0 and pos+n<=old.size*4
                assert c['raw_offset']==pos and c['encoded_offset']==offset and 0<=c['encoded_bytes']<=n
                a=memoryview(old).cast('B')[pos:pos+n].tobytes();assert digest(a)==c['source_sha256']
                data=stream.read(c['encoded_bytes']);assert len(data)==c['encoded_bytes'] and digest(data)==c['encoded_sha256']
                codec=c['codec']
                if codec=='copy':assert not data;decoded=a
                elif codec=='raw':assert len(data)==n;decoded=data
                else:
                    assert codec=='xor_zlib' and 0<len(data)<n;z=zlib.decompressobj();delta=z.decompress(data,n+1)
                    assert z.eof and not z.unused_data and not z.unconsumed_tail and len(delta)==n
                    decoded=(int.from_bytes(a,'little')^int.from_bytes(delta,'little')).to_bytes(n,'little')
                assert digest(decoded)==c['target_sha256'];oh.update(a);nh.update(decoded)
                v=np.frombuffer(decoded,'<u4');changed=int(np.count_nonzero(old[pos//4:(pos+n)//4]!=v))
                assert changed==c['changed_words'];dest[pos//4:(pos+n)//4]=v
                pos+=n;offset+=len(data);total+=n;moved+=changed;counts[codec]+=1;guard()
            assert pos==row['raw_bytes'] and oh.hexdigest()==row['source_sha256'] and nh.hexdigest()==row['target_sha256']
            out[row['name']]=torch.from_numpy(dest.view('<f4').reshape(shape))
        assert not stream.read(1) and offset==index['blob_bytes']
    assert total==index['raw_bytes'] and moved==index['changed_words'] and counts==index['codec_counts']
    return out
