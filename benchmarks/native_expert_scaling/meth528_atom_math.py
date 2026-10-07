"""Prospective variable source atoms, own hidden quantizer and fixed decision."""
import math
import numpy as np

ARMS=('continuous_atoms','physical_atoms','physical_rotation','physical_old_fold')


def quant(x):
    x=np.asarray(x,'<f4'); maximum=np.max(np.abs(x),axis=1)
    scale=np.divide(maximum,np.float32(32767),dtype=np.float32);scale[maximum==0]=1
    assert np.all(scale>0) and np.isfinite(scale).all()
    codes=np.clip(np.rint(np.divide(x,scale[:,None],dtype=np.float32)),-32767,32767).astype('<i2')
    return codes,scale


def function(dots,alpha,si,wo,so,continuous=True):
    z=(dots.astype('<f8')*si.astype('<f8'))*alpha.astype('<f8')[:,None]; h=np.maximum(z,0)
    q,a=quant(h.astype('<f4')); d=q.astype('<f8')@wo.astype('<f8').T
    assert np.array_equal(d,d.astype('<i8')) and np.max(np.abs(d))<=3072*128*32767
    out=((d*so.astype('<f8'))*a.astype('<f8')[:,None]).astype('<f4')
    if continuous:
        weights=wo.astype('<f8')*so.astype('<f8')[:,None]
        return out,d.astype('<i8'),q,a,h@weights.T,h,np.abs(h)@np.abs(weights).T
    return out,d.astype('<i8'),q,a,None,h,None


def identity_bound(inputs,source_h32,wi,si,wo,so,u,f,fold,abs_atoms,abs_m):
    # The rounded original source h32 encloses its old h64 even for subnormals.
    hup=np.nextafter(np.nextafter(source_h32.astype('<f8')+2.**-149,np.inf)/(1.-2.**-24),np.inf)
    o=wo.astype('<f8')*so.astype('<f8')[:,None]; x=np.abs(inputs['q'].astype('<f8')*inputs['alpha'].astype('<f8')[:,None])
    source=hup@np.abs(o).T; old=x@np.abs(fold).T+hup[:,u]@np.abs(o[:,u]).T
    i=wi[f].astype('<f8')*si[f].astype('<f8')[:,None]
    absolute_fold=(x@np.abs(i).T)@np.abs(o[:,f]).T
    result=1e-10*(1+source+old+abs_atoms+abs_m)+5e-12*(np.sum(x,axis=1)[:,None]+absolute_fold)
    g=65536/((1<<53)-65536)
    return np.nextafter(result*(1+8*g),np.inf)


def metrics(continuous,negative,physical,old,ref,target,p,bound):
    oldr=old[:,0]-old[:,2]; P=old[:,0]-continuous
    ratio=np.max(np.abs(oldr-P-negative)/bound,axis=1);assert np.max(ratio)<=1
    blocks=[]
    for weighted in (False,True):
        truth=target.astype('<f8') if weighted else ref.astype('<f8'); factor=p.astype('<f8')[:,None] if weighted else 1.
        g=continuous*factor; pp=P*factor; mm=negative*factor; rr=oldr*factor
        a=np.multiply(physical[:,0],p[:,None],dtype=np.float32).astype('<f8') if weighted else physical[:,0].astype('<f8')
        rot=np.multiply(physical[:,1],p[:,None],dtype=np.float32).astype('<f8') if weighted else physical[:,1].astype('<f8')
        oldp=np.multiply(old[:,4].astype('<f4'),p[:,None],dtype=np.float32).astype('<f8') if weighted else old[:,4]
        bridge=old[:,0]*factor-truth; codec=a-g
        vectors=(truth,g-truth,a-truth,rot-truth,oldp-truth,pp,mm,codec,bridge,rr)
        energies=[np.sum(v*v,axis=1) for v in vectors]
        blocks.append(np.column_stack((*energies[:7],np.sum(pp*mm,axis=1),*energies[7:],ratio)))
    return np.column_stack(blocks)


def summarize(label,ids,data,prices):
    sums=np.sum(data[ids],axis=0) if len(ids) else np.zeros(24)
    out={**label,'count':len(ids),'sums':sums[:11].tolist()+sums[12:23].tolist(),
         'identity_ratio':float(np.max(data[ids,11])) if len(ids) else 0.,
         'selected_FFN_byte_ratio':float(np.mean(prices[ids]/4733952)) if len(ids) else None}
    for prefix,off in (('unweighted',0),('weighted',12)):
        den=float(sums[off]);out[prefix+'_source_energy']=den
        for j,arm in enumerate(ARMS,1):out[prefix+'_'+arm+'_RMS']=math.sqrt(float(sums[off+j])/den) if den else None
        for j,name in ((5,'P_energy'),(6,'negative_fold_energy'),(7,'P_negative_cross'),(8,'physical_codec_energy'),(9,'source_shadow_bridge_energy'),(10,'old_continuous_residual_energy')):
            out[prefix+'_'+name]=float(sums[off+j])
    return out


def eligibility(views):
    six=[v for v in views if v['kind']=='role'];assert len(six)==6
    rare=[v for v in views if v['kind']=='rare' and v['split']=='consumed_validation' and v['development_class'] in ('1..4','5..15')]
    consumed=next(v for v in views if v['kind']=='uid' and v['split']=='consumed_validation')
    ans={p+'_'+arm+'_ALL_six_RMS_1pct':all(v['count'] and v[p+'_'+arm+'_RMS'] is not None and v[p+'_'+arm+'_RMS']<=.01 for v in six)
        for p,arm in (('unweighted','continuous_atoms'),('unweighted','physical_atoms'),('weighted','physical_atoms'))}
    ans['ALL_nonempty_rare_consumed_physical_RMS_1pct']=all(not v['count'] or all(v[p+'_physical_atoms_RMS'] is not None and v[p+'_physical_atoms_RMS']<=.01 for p in ('unweighted','weighted')) for v in rare)
    ans['ALL_six_selected_FFN_bytes_including_scales_IDs_header_le75pct_source']=all(v['count'] and v['selected_FFN_byte_ratio']<=.75 for v in six)
    ans['consumed_AND_ALL_six_source_informed_physical_strictly_better_than_matched_rotation']=all(v['count'] and all(v[p+'_physical_atoms_RMS'] is not None and v[p+'_physical_rotation_RMS'] is not None and v[p+'_physical_atoms_RMS']<v[p+'_physical_rotation_RMS'] for p in ('unweighted','weighted')) for v in [consumed,*six])
    return ans


def controls():
    i=np.array([[1,0],[0,1],[1,1],[1,-1]],'<f8');o=np.array([[1,2,-3,4],[-1,3,1,2]],'<f8');rows=[]
    for x in ([2,1],[-2,1],[-1,-1],[0,0]):
        z=i@x; source=o@np.maximum(z,0); atoms=o[:,[0,3]]@np.maximum(z[[0,3]],0)
        fold=o[:,0]*max(z[0],0)+o[:,3]*z[3];p=o[:,[1,2]]@np.maximum(z[[1,2]],0);m=o[:,3]*max(-z[3],0)
        assert np.array_equal(source-atoms,p) and np.array_equal(source-fold,p+m)
        rows.append(dict(x=x,source=source.tolist(),atoms=atoms.tolist(),fold=fold.tolist(),P=p.tolist(),M=m.tolist()))
    q,a=quant([[32767,-32767,1.5,2.5,4.5,5.5,-1.5,-4.5]])
    assert q.tolist()==[[32767,-32767,2,2,4,6,-2,-4]] and a.tolist()==[1.]
    zero,za=quant(np.zeros((1,4),'<f4'));assert not zero.any() and za.tolist()==[1.]
    result=function(np.full((1,3072),32767,'<i8'),np.ones(1,'<f4'),np.ones(3072,'<f4'),np.full((768,3072),127,'i1'),np.ones(768,'<f4'))
    wide=3072*127*32767; assert wide>2**31 and np.all(result[1]==wide)
    return dict(piecewise=rows,tie_codes=q.tolist(),zero_alpha=1.,wide_WO_integer=wide,wide_WO_physical=float(np.float32(wide)))
