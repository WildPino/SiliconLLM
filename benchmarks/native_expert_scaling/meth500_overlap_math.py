"""ONE source-mask recipe: input-only regions, mandatory coverage and overlap."""
import numpy as np
C,H,D,B=4,3072,768,1536
def projection(q,alpha,w,scales):
    assert q.dtype==np.dtype('<i2') and w.dtype==np.dtype('i1')
    # All products/partial sums <2^53, so F64 GEMM is exact integer arithmetic.
    assert q.shape[1]*128*32767<2**53
    dots=q.astype('<f8')@w.astype('<f8').T
    assert np.array_equal(dots,np.rint(dots))
    return ((dots*scales.astype('<f8'))*alpha.astype('<f8')[:,None]).astype('<f4')
def quant(h):
    maxv=np.max(np.abs(h),axis=1);a=(maxv/np.float32(32767)).astype('<f4');a[maxv==0]=1
    assert np.all(a>0) and np.isfinite(a).all()
    return np.clip(np.rint(np.divide(h,a[:,None],dtype=np.float32)),-32767,32767).astype('<i2'),a
def unit(x):
    x=x.astype('<f8');norm=np.linalg.norm(x,axis=1)
    return np.divide(x,norm[:,None],out=np.zeros_like(x),where=norm[:,None]!=0)
def regions(x,uids):
    assert len(x)>0 and np.all(uids[:-1]<uids[1:])
    z=unit(x);chosen=[0];dist=np.sum((z-z[0])**2,axis=1)
    for _ in range(1,C):
        eligible=np.ones(len(z),bool);eligible[chosen]=False
        if eligible.any():
            k=int(np.argmax(np.where(eligible,dist,-np.inf)));chosen.append(k);dist=np.minimum(dist,np.sum((z-z[k])**2,axis=1))
        else:chosen.append(0)
    centres=z[chosen].copy();states=[centres.copy()]
    for _ in range(8):
        labels=np.argmax(z@centres.T,axis=1)
        for c in range(C):
            group=z[labels==c]
            if len(group):centres[c]=unit(group.mean(axis=0,keepdims=True))[0]
        states.append(centres.copy())
    return centres,np.argmax(z@centres.T,axis=1),chosen,np.asarray(states)
def masks(scores):
    assert scores.shape==(C,H) and np.all(scores>=0) and np.isfinite(scores).all()
    # Mandatory balanced partition: every donor atom is stored at least once.
    c,j=np.indices((C,H));order=np.lexsort((c.ravel(),j.ravel(),-scores.ravel()))
    own=np.full(H,-1,'<i4');count=np.zeros(C,'<i4');mask=np.zeros((C,H),bool)
    for k in order:
        a,b=divmod(int(k),H)
        if own[b]<0 and count[a]<H//C:own[b]=a;count[a]+=1;mask[a,b]=True
    assert np.all(own>=0) and np.all(count==H//C)
    for c in range(C):
        eligible=np.flatnonzero(~mask[c]);take=eligible[np.lexsort((eligible,-scores[c,eligible]))[:B-H//C]];mask[c,take]=True
    assert np.all(mask.sum(axis=1)==B) and np.all(mask.any(axis=0))
    return mask,own
def scores(h,labels,wo,so):
    energy=(wo.astype('<f8')*so.astype('<f8')[:,None])**2;norm=energy.sum(axis=0)
    global_score=(h.astype('<f8')**2).mean(axis=0)*norm
    result=np.empty((C,H),'<f8')
    for c in range(C):
        selected=h[labels==c]
        result[c]=(selected.astype('<f8')**2).mean(axis=0)*norm if len(selected) else global_score
    return result,global_score
def controls():
    q=np.array([[32767,32767],[-32767,32767]],'<i2');w=np.array([[-128,127],[127,127]],'i1')
    expected=((q.astype('<i8')@w.astype('<i8').T).astype('<f8')*np.array([.5,1.]))*np.array([1.,.25])[:,None]
    assert projection(q,np.array([1.,.25],'<f4'),w,np.array([.5,1.],'<f4')).tobytes()==expected.astype('<f4').tobytes()
    h=np.array([[0,0,0],[1,.5,0]],'<f4');qh,a=quant(h)
    assert qh.tolist()==[[0,0,0],[32767,16384,0]] and float(a[0])==1
    z=np.array([[1,0],[-1,0],[0,1],[0,-1]],'<f4');p,l,seeds,states=regions(z,np.arange(4))
    assert seeds==[0,1,2,3] and l.tolist()==[0,1,2,3] and states.shape==(9,4,2)
    mk,own=masks(np.zeros((C,H)))
    assert np.all(mk.sum(axis=1)==B) and np.all(mk.any(axis=0)) and np.bincount(own).tolist()==[768]*4
    return {'integer_projection_I64':True,'A16_zero_tie':True,'four_axis_input_regions':True,'mandatory_union_capacity_ties':True}
