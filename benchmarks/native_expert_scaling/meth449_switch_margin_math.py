"""Reference-required margin/information diagnostics, never a runtime selector."""
import numpy as np


def quant(x):
    assert x.dtype == np.float32 and x.ndim == 1 and np.isfinite(x).all()
    maximum = np.max(np.abs(x))
    scale = np.float32(1) if maximum == 0 else np.float32(maximum/np.float32(32767))
    assert np.isfinite(scale) and scale > 0
    return scale, np.clip(np.rint(x/scale),-32767,32767).astype(np.int16)


def native_rows(weights, scales, activation_scale, codes):
    assert weights.dtype == np.int8 and codes.dtype == np.int16
    integer = weights.astype(np.int64) @ codes.astype(np.int64)
    unrounded = (integer.astype(np.float64)*scales.astype(np.float64))*np.float64(activation_scale)
    return unrounded.astype(np.float32), unrounded


def logp(z):
    z = z.astype(np.float64)
    z = z-np.max(z)
    return z-np.log(np.sum(np.exp(z)))


def information(original,candidate):
    lp0,lpc = logp(original),logp(candidate)
    p,q = np.exp(lp0),np.exp(lpc)
    delta = candidate.astype(np.float64)-original.astype(np.float64)
    mean = np.dot(p,delta)
    kl = float(np.dot(p,lp0-lpc))
    pivot = float(np.max(delta))
    cumulant = float(np.log(np.dot(p,np.exp(delta-pivot)))+pivot-mean)
    assert abs(kl-cumulant)<=1e-10 and kl>=-1e-10
    variance = float(np.dot(p,(delta-mean)**2))
    tv = float(.5*np.sum(np.abs(p-q)))
    assert tv<=np.sqrt(max(kl,0)/2)+1e-10, 'Pinsker'
    return {'KL':kl,'KL_cumulant_identity':cumulant,'half_p_weighted_logit_variance':variance/2,
            'TV':tv,'centered_logit_Linf_best':float((np.max(delta)-np.min(delta))/2),
            'source_entropy':float(-np.dot(p,lp0)),'source_max_probability':float(np.max(p))},p


def contrast(original,candidate,a,b):
    o,c = original.astype(np.float64),candidate.astype(np.float64)
    margin = float(o[a]-o[b])
    d = float((c[b]-o[b])-(c[a]-o[a]))
    assert abs((c[a]-c[b])-(margin-d))<=1e-12
    return margin,d,float(c[a]-c[b])


def tiny_qualification():
    o=np.asarray([2.,1.,-1.],np.float32);c=np.asarray([1.5,1.75,-.5],np.float32)
    m,d,g=contrast(o,c,0,1);assert m==1 and d==1.25 and g==-.25
    r,_=information(o,c);shift,_=information(o,c+np.float32(8))
    assert abs(r['KL']-shift['KL'])<=1e-14
    assert abs(r['half_p_weighted_logit_variance']-shift['half_p_weighted_logit_variance'])<=1e-14
    assert r['centered_logit_Linf_best']==shift['centered_logit_Linf_best']
    tied=np.asarray([2.,2.,-1.],np.float32);assert np.argmax(tied)==0
    assert contrast(o,tied,0,1)==(1.,1.,0.)
    x=np.asarray([-32767.,-16384.,0.,16384.,32767.],np.float32)*np.float32(.25)
    s,q=quant(x);assert s==np.float32(.25) and np.array_equal(q,[-32767,-16384,0,16384,32767])
    s,q=quant(np.zeros(5,np.float32));assert s==1 and not q.any()
    weights=np.asarray([[1,-2,3,-4,5],[-1,2,-3,4,-5]],np.int8)
    codes=np.asarray([32767,-16384,8192,-4096,2048],np.int16)
    scales=np.asarray([.5,.25],np.float32)
    native,raw=native_rows(weights,scales,np.float32(.25),codes)
    manual=np.asarray([sum(int(w)*int(a) for w,a in zip(row,codes)) for row in weights],np.int64)
    assert np.array_equal(raw,manual.astype(np.float64)*scales.astype(np.float64)*.25)
    assert native.tobytes()==raw.astype(np.float32).tobytes()
    return {'pair_margin_flip_and_tie_exact':True,'KL_cumulant_constant_shift_variance_Pinsker_qualified':True,
            'A16_and_selected_head_I64_manual_extrema_qualified':True}
