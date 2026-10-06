"""New Gaussian even projection and literal physical prior encoding."""
import math
import numpy as np

def absolute_kernel(left,right):
    u=np.sqrt(np.sum(left*left,axis=1)); v=np.sqrt(np.sum(right*right,axis=1))
    denominator=u[:,None]*v[None,:]; dot=left@right.T
    cosine=np.divide(dot,denominator,out=np.zeros_like(dot),where=denominator!=0)
    assert np.isfinite(cosine).all() and np.max(np.abs(cosine))<=1+2e-12
    clipped=int(np.count_nonzero(np.abs(cosine)>1)); cosine=np.clip(cosine,-1,1)
    answer=denominator*(2/math.pi)*(np.sqrt(np.maximum(0,1-cosine*cosine))+cosine*np.arcsin(cosine))
    assert np.isfinite(answer).all(); return answer,clipped

def ternary_row(codes,scale):
    values=codes.astype('<i2'); total=int(np.sum(values.astype('<i8')**2))
    t=np.zeros(len(values),dtype='<i1')
    if not total or scale==0: return t,np.float32(1),0
    chosen=np.lexsort((np.arange(len(values)), -np.abs(values)))[:256]
    t[chosen]=np.sign(values[chosen]); nz=int(np.count_nonzero(t))
    s=np.float32(np.sum(np.abs(values[chosen]))/(nz*math.sqrt(total)))
    assert s>0 and np.isfinite(s); return t,s,nz

def packed_ternary(t):
    groups=t.astype('<i2').reshape(len(t),-1,4)+1
    return np.sum(groups*np.array([1,3,9,27]),axis=2).astype('u1')

def quant_rows(value):
    assert value.dtype==np.dtype('<f4') and np.isfinite(value).all()
    maxima=np.max(np.abs(value),axis=1)
    scale=(maxima/np.float32(127)).astype('<f4'); scale[maxima==0]=np.float32(1)
    assert np.all(scale>0) and np.isfinite(scale).all()
    q=np.clip(np.rint(np.divide(value,scale[:,None],dtype=np.float32)),-127,127).astype('<i1')
    return q,scale

def controls():
    wi=np.array([[1,-2,0,1],[0,1,2,-1],[-1,0,1,2]],'<f8')
    wo=np.array([[1,-1,2],[-2,1,1]],'<f8')/4
    x=np.array([[0,1,-1,2],[2,-1,1,0],[-2,1,-1,0]],'<f8').T
    direct=wo@np.maximum(wi@x,0); odd_even=.5*(wo@wi)@x+.5*wo@np.abs(wi@x)
    assert direct.tobytes()==odd_even.tobytes()
    vectors=np.array([[1,0],[-1,0],[0,1],[0,0],[3,0]],'<f8'); kernel,clip=absolute_kernel(vectors,vectors)
    assert abs(kernel[0,0]-1)<2e-15 and abs(kernel[0,1]-1)<2e-15
    assert abs(kernel[0,2]-2/math.pi)<2e-15 and kernel[3].tobytes()==np.zeros(5,'<f8').tobytes()
    assert abs(kernel[4,0]-3)<2e-15
    t=np.array([[0,0,0,0],[-1,-1,-1,-1],[1,1,1,1]],'<i1')
    packed=packed_ternary(t); assert packed.ravel().tolist()==[40,0,80]
    row=np.array([[127,1.5,2.5,-1.5,-2.5,0],[0,0,0,0,0,0]],'<f4')
    q,s=quant_rows(row); assert q.tolist()==[[127,2,2,-2,-2,0],[0]*6] and s.tolist()==[1,1]
    return {'dyadic_identity_values':direct.tolist(),'absolute_kernel_values':kernel.tolist(),
        'absolute_mean_unit':math.sqrt(2/math.pi),'constant_second_moment':1,
        'packed_group_codes':packed.ravel().tolist(),'quant_codes':q.tolist(),'quant_scales':s.tolist(),'correlation_clips':clip}
