"""Frozen original-coordinate A16/I8 arithmetic and ONE private input SVD."""
from fractions import Fraction
import math
import struct
import numpy as np


def exact(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), 'byte_exact'


def quant(x):
    assert x.dtype == np.dtype('<f4') and x.ndim == 2 and np.isfinite(x).all()
    maximum = np.max(np.abs(x), axis=1)
    with np.errstate(under='ignore'):
        alpha = (maximum / np.float32(32767)).astype('<f4')
    alpha[maximum == 0] = np.float32(1)
    assert np.isfinite(alpha).all() and np.all(alpha > 0)
    q = np.clip(np.rint(np.divide(x, alpha[:, None], dtype=np.float32)), -32767, 32767).astype('<i2')
    return q, alpha


def nearest_positive(value):
    if not value:
        return 0.
    bits = struct.unpack('<I', struct.pack('<f', float(value)))[0]
    candidates = []
    for k in (bits-1, bits, bits+1):
        if 0 <= k <= 0x7f7fffff:
            v = struct.unpack('<f', struct.pack('<I', k))[0]
            candidates.append((abs(Fraction.from_float(v)-value), k & 1, v))
    return min(candidates)[2]


def rational_quant(values):
    maximum = max(abs(float(v)) for v in values)
    alpha = nearest_positive(Fraction.from_float(maximum)/32767) if maximum else 1.
    q = []
    for v in values:
        divided = nearest_positive(Fraction.from_float(abs(float(v)))/Fraction.from_float(alpha))
        q.append(max(-32767, min(32767, round(divided)*(-1 if v < 0 else 1))))
    return q, alpha


def projection(q, alpha, w64, scales):
    # Function widths<=3072; cached native qualifiers also include width4096.
    # Every integer product/partial sum has absolute bound below2^53.
    assert q.dtype == np.dtype('<i2') and np.all(q != -32768)
    assert q.shape[1] <= 4096 and w64.shape[1] == q.shape[1]
    assert q.shape[1]*128*32767 < 2**53
    assert np.all(w64 == np.rint(w64)) and np.max(np.abs(w64)) <= 128
    dots = q.astype(np.float64) @ w64.T
    assert np.isfinite(dots).all() and np.all(dots == np.rint(dots))
    with np.errstate(under='ignore'):
        answer = ((dots * scales.astype(np.float64)[None, :]) * alpha.astype(np.float64)[:, None]).astype('<f4')
    assert np.isfinite(answer).all()
    return answer


def source_ffn(q, alpha, wi64, si, wo64, so):
    up = projection(q, alpha, wi64, si)
    h = np.where(up < 0, np.float32(0), up).astype('<f4')
    qh, ah = quant(h)
    return up, h, qh, ah, projection(qh, ah, wo64, so)


def factor_ffn(q, alpha, p32, a32, si, wo64, so):
    assert p32.dtype == a32.dtype == np.dtype('<f4')
    dots = (q.astype(np.float64) @ p32.astype(np.float64)) @ a32.astype(np.float64).T
    with np.errstate(under='ignore'):
        up = ((dots * si.astype(np.float64)[None, :]) * alpha.astype(np.float64)[:, None]).astype('<f4')
    assert np.isfinite(up).all()
    h = np.where(up < 0, np.float32(0), up).astype('<f4')
    qh, ah = quant(h)
    return up, h, qh, ah, projection(qh, ah, wo64, so)


def spectrum(x, guard):
    assert x.dtype == np.float64 and x.ndim == 2 and np.isfinite(x).all()
    energy = float(np.sum(x*x));assert energy > 0
    u, s, vt = np.linalg.svd(x, full_matrices=False);guard()
    assert np.isfinite(s).all() and np.all(s >= 0) and np.all(s[:-1] >= s[1:])
    for k in range(len(s)):
        if vt[k, int(np.argmax(np.abs(vt[k])))] < 0:
            vt[k] *= -1;u[:, k] *= -1
    recon = float(np.linalg.norm((u*s) @ vt-x)/math.sqrt(energy))
    orth = float(np.linalg.norm(vt @ vt.T-np.eye(len(s))))
    eig = float(np.linalg.norm(x.T @ (x @ vt.T)-vt.T*(s*s))/energy)
    balance = abs(float(np.sum(s*s))-energy)/energy
    assert recon <= 1e-10 and orth <= 1e-10*math.sqrt(len(s)) and eig <= 1e-10 and balance <= 1e-10
    tol = float(s[0]*max(x.shape)*np.finfo(np.float64).eps)
    stats = {'energy':energy,'reconstruction_relative':recon,'orthogonal_residual':orth,
             'eigen_relative':eig,'energy_relative_discrepancy':balance,
             'numerical_rank_tolerance':tol,'numerical_rank':int(np.sum(s > tol)),
             'singular_values':s.tolist()}
    guard()
    return s, vt, stats


def identity_and_spectral_controls(guard):
    wi = np.array([[-128,127,0],[1,-3,2],[7,0,-8]], dtype=np.float64)
    q = np.array([[-32767,32767,1],[0,0,0],[1,-1,32767]],dtype='<i2')
    alpha = np.array([.5,1,2.**-20],dtype='<f4');si=np.array([.01,.5,2.],dtype='<f4')
    dots = q.astype(np.int64) @ wi.astype(np.int64).T
    assert np.array_equal(q.astype(np.float64) @ wi.T,dots.astype(np.float64))
    p = np.eye(3,dtype='<f4');a=wi.astype('<f4')
    expected=projection(q,alpha,wi,si)
    candidate=(((q.astype(np.float64) @ p.astype(np.float64)) @ a.astype(np.float64).T)*si.astype(np.float64))*alpha.astype(np.float64)[:,None]
    exact(candidate.astype('<f4'),expected)
    # Full-native width extreme integer reduction is independent of the SVD.
    for width in (768,3072,4096):
        cq=np.full((2,width),32767,dtype='<i2');cq[1,1::2]*=-1
        cw=np.full((2,width),-128,dtype=np.float64);cw[1,1::2]=127
        assert np.array_equal(cq.astype(np.float64) @ cw.T,(cq.astype(np.int64) @ cw.astype(np.int64).T).astype(np.float64))
    for values in ([0.]*17,[32767.,.5,-.5,1.5,-1.5,2.5,-2.5],[.01,-.003,.005,.0007],
                   [1e-20,-2e-20,3e-20],[1e20,-2e20,3e20],[2.**-133,2.**-140,-2.**-140]):
        v=np.asarray([values],dtype='<f4');qv,av=quant(v);sq,sa=rational_quant(v[0])
        assert qv[0].tolist()==sq and float(av[0])==sa
    x=np.array([[1.,0.,0.],[0.,8.,0.],[-1.,0.,0.],[0.,-8.,0.]])/2
    s,vt,stats=spectrum(x,guard)
    assert stats['numerical_rank']==2 and abs(s[0]**2-32)<1e-12 and abs(s[1]**2-.5)<1e-12
    assert abs(vt[0,1])>=1-1e-12 and abs(vt[1,0])>=1-1e-12
    # Explicit alpha^2 control: equal code magnitudes have unequal decoded energy.
    c=np.array([[32767.,0.],[0.,32767.]])
    sc=np.array([1./32767,8./32767])
    cov=(c*sc[:,None]).T @ (c*sc[:,None])
    assert np.allclose(cov,np.diag([1.,64.]),rtol=1e-15,atol=1e-15)
    assert np.array_equal(c.T @ c,np.eye(2)*32767**2)
    return {'identity_same_integer_dot_scale_alpha_F32_exact':True,
            'extreme_native_widths_I64_F64_integer_dots_exact':True,
            'six_rational_quantizer_shapes_exact':True,'known_rank_energy_axes_and_alpha_squared_controls':True}


def weighted_rms(error_energy, reference_energy, weights=None):
    assert error_energy.shape == reference_energy.shape and len(error_energy)>0
    assert np.isfinite(error_energy).all() and np.isfinite(reference_energy).all()
    assert np.all(error_energy>=0) and np.all(reference_energy>=0)
    if weights is None:
        numerator=float(np.sum(error_energy));denominator=float(np.sum(reference_energy))
    else:
        assert weights.shape==error_energy.shape and np.all(weights>0)
        numerator=float(np.sum(weights*error_energy));denominator=float(np.sum(weights*reference_energy))
    rms=math.sqrt(numerator/denominator) if denominator else (0. if numerator==0 else None)
    return {'RMS_relative':rms,'weighted_error_energy':numerator,'weighted_reference_energy':denominator,
            'zero_reference_nonzero_error':denominator==0 and numerator>0,'count':len(error_energy)}


def within(stats, threshold):
    return stats['RMS_relative'] is not None and stats['RMS_relative']<=threshold
