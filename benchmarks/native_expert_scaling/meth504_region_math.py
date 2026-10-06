"""Source input-cone certificate. Cross products are Python integers, never I64."""
from functools import cmp_to_key
import numpy as np

def quant(h):
    maxv=np.max(np.abs(h),axis=1) if h.shape[1] else np.zeros(len(h),'<f4')
    a=(maxv/np.float32(32767)).astype('<f4');a[maxv==0]=1
    return np.clip(np.rint(np.divide(h,a[:,None],dtype=np.float32)),-32767,32767).astype('<i2'),a

def choose_seed(q,uids):
    z=q.astype('<f8');n=np.linalg.norm(z,axis=1)
    z=np.divide(z,n[:,None],out=np.zeros_like(z),where=n[:,None]>0)
    return int(uids[int(np.argmax(z@z.mean(axis=0)))])

def mask(d,R,B,A):
    keep=np.zeros(len(d),'u1');required=[j for j in range(len(d)) if R[j] and d[j]>=0]
    if A==0:return keep,{'status':'ZERO_CENTRE_UNSUPPORTED','A':0}
    if len(required)>B:return keep,{'status':'CENTRE_NONNEGATIVE_WIDTH_ABOVE_B','A':int(A),'required':len(required)}
    negative=[j for j in range(len(d)) if R[j] and d[j]<0]
    def cmp(j,k):
        a=int(d[j])**2*int(R[k]);b=int(d[k])**2*int(R[j])
        return (a>b)-(a<b) or (j>k)-(j<k)
    order=sorted(negative,key=cmp_to_key(cmp));take=min(B-len(required),len(order))
    keep[required+order[:take]]=1;omitted=order[take:]
    if not omitted:return keep,{'status':'ALL_NONZERO_ROWS_RETAINED','A':int(A),'required':len(required)}
    j=omitted[0];return keep,{'status':'ANGULAR_CONE','A':int(A),'required':len(required),'limiting_neuron':j,'n':int(d[j])**2,'r':int(R[j])}

def accepts(t,Q,c):
    if Q==0:return True
    if c['status']=='ALL_NONZERO_ROWS_RETAINED':return True
    if c['status']!='ANGULAR_CONE' or t<=0:return False
    A,n,r=int(c['A']),int(c['n']),int(c['r']);Q,t=int(Q),int(t)
    assert A*Q-t*t>=0
    return r*(A*Q-t*t)<n*Q

def controls():
    w=np.array([[1,0],[0,1],[-1,0],[-3,4],[-4,3],[0,0]],'i1');a=np.array([3,0],'<i2')
    d=w.astype('<i8')@a.astype('<i8');R=np.sum(w.astype('<i8')**2,axis=1)
    mk,c=mask(d,R,3,9);assert np.flatnonzero(mk).tolist()==[0,1,3] and c['n']==144 and c['r']==25 and c['limiting_neuron']==4
    assert accepts(9,9,c) and not accepts(9,25,c) and not accepts(-9,9,c) and accepts(0,0,c)
    # Exhaustive small two-dimensional certificate and complete FFN identity.
    so=np.array([.5,.25],'<f4');si=np.array([1,.5,.25,.125,.5,1],'<f4');wo=np.array([[1,2,3,4,5,6],[-1,0,1,-2,3,4]],'i1')
    certified=0
    for u in range(-6,7):
        for v in range(-6,7):
            q=np.array([[u,v]],'<i2');Q=u*u+v*v
            if not accepts(3*u,Q,c):continue
            dots=w.astype('<i8')@q[0].astype('<i8');assert np.all(dots[(mk==0)&(R>0)]<0) if Q else np.all(dots==0)
            h=np.maximum((dots.astype('<f8')*si).astype('<f4'),np.float32(0))[None,:]
            qh,scale=quant(h);qc,ac=quant(h[:,mk!=0]);assert qc.tobytes()==qh[:,mk!=0].tobytes() and ac.tobytes()==scale.tobytes()
            full=(((qh.astype('<i8')@wo.astype('<i8').T).astype('<f8')*so)*scale.astype('<f8')[:,None]).astype('<f4')
            small=(((qc.astype('<i8')@wo[:,mk!=0].astype('<i8').T).astype('<f8')*so)*ac.astype('<f8')[:,None]).astype('<f4');assert full.tobytes()==small.tobytes();certified+=1
    assert certified>1
    assert mask(np.array([1,0],'<i8'),np.array([1,1],'<i8'),1,1)[1]['status']=='CENTRE_NONNEGATIVE_WIDTH_ABOVE_B'
    A=Q=768*32767**2;t=0;r=768*128**2;n=(768*128*32767)**2
    assert (r*A*Q).bit_length()<104 and (n*Q).bit_length()<104 and r*A*Q>2**63
    assert choose_seed(np.array([[3,0],[3,0]],'<i2'),np.array([4,7]))==4
    return {'strict_boundary_orientation_zero_and_rational_mask':True,'exhaustive_tiny_negative_rows_hidden_scale_and_F_BYTE':True,'products_above_I64_below_104bits':True,'input_seed_lowest_UID_tie':True}
