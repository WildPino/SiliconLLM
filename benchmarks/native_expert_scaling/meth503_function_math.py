"""Actual child A16/I8 functions and one input centroid rule; fixed masks."""
import numpy as np
def quant(h):
    maxima=np.max(np.abs(h),axis=1) if h.shape[1] else np.zeros(len(h),'<f4')
    a=(maxima/np.float32(32767)).astype('<f4');a[maxima==0]=1
    assert np.isfinite(a).all() and np.all(a>0)
    return np.clip(np.rint(np.divide(h,a[:,None],dtype=np.float32)),-32767,32767).astype('<i2'),a
def project(q,a,w,s):
    assert q.dtype==np.dtype('<i2') and w.dtype==np.dtype('i1') and q.shape[1]*128*32767<2**53
    d=q.astype('<f8')@w.astype('<f8').T;assert np.array_equal(d,np.rint(d))
    return ((d*s.astype('<f8'))*a.astype('<f8')[:,None]).astype('<f4')
def unit(x):
    z=x.astype('<f8');n=np.linalg.norm(z,axis=1)
    return np.divide(z,n[:,None],out=np.zeros_like(z),where=n[:,None]>0)
def prototypes(x,labels,C):
    z=unit(x);pc=np.zeros((C,z.shape[1]),'<f8');count=np.bincount(labels,minlength=C)
    for c in range(C):
        if count[c]:pc[c]=unit(z[labels==c].mean(axis=0,keepdims=True))[0]
    return pc,count
def controls():
    q=np.array([[32767,32767],[-32767,32767]],'<i2');w=np.array([[-128,127],[127,127]],'i1');a=np.array([1,.25],'<f4');s=np.array([.5,1],'<f4')
    expected=((q.astype('<i8')@w.astype('<i8').T).astype('<f8')*s.astype('<f8'))*a.astype('<f8')[:,None]
    assert project(q,a,w,s).tobytes()==expected.astype('<f4').tobytes()
    q,a=quant(np.array([[0,0],[1,.5]],'<f4'));assert q.tolist()==[[0,0],[32767,16384]] and a[0]==1
    pc,count=prototypes(np.array([[1,0],[-1,0]],'<f4'),np.array([0,1]),3)
    assert count.tolist()==[1,1,0] and pc.tolist()==[[1,0],[-1,0],[0,0]]
    score=np.array([[0,1],[0,-1]],'<f8')@pc.T;score[:,count==0]=-np.inf;assert np.argmax(score,axis=1).tolist()==[0,0]
    return {'I64_exact_projection':True,'A16_zero_tie':True,'input_centroids_unsupported_child_and_lowest_ID_tie':True}
