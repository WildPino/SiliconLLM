"""Certify E*100000000 <= R for saved F32 logits, with exact fallback.

All F32 values and their nonzero differences/squares are normal-range F64.
Use a conservative directed-rounding bound, outward endpoints, and big integers
when the interval cannot decide. This changes no native tolerance.
"""
import math
import numpy as np


def units(bits):
    exponent=(bits>>23)&255
    assert exponent!=255
    mantissa=bits&0x7fffff
    if exponent:
        mantissa=(mantissa|0x800000)<<(exponent-1)
    return -mantissa if bits>>31 else mantissa


def certify(actual,reference):
    assert actual.ndim==reference.ndim==1 and actual.shape==reference.shape
    assert actual.dtype.kind==reference.dtype.kind=='f' and actual.itemsize==reference.itemsize==4
    n=actual.size
    assert 1<=n<=65537 and np.isfinite(actual).all() and np.isfinite(reference).all()
    c=actual.astype(np.float64);r=reference.astype(np.float64)
    delta=c-r
    error=float(np.square(delta).sum(dtype=np.float64))
    energy=float(np.square(r).sum(dtype=np.float64))
    assert math.isfinite(error) and math.isfinite(energy)
    if energy==0:
        return dict(RMS_gate=error==0,method='zero_energy',relative_RMS=0.0 if error==0 else None)
    if error==0:
        return dict(RMS_gate=True,method='zero_error',relative_RMS=0.0)
    # u=2^-52 covers each supported IEEE rounding direction; m*u<1/2.
    # gamma_m=m*u/(1-m*u) <= 2*m*u. Reference squares are exact (<=48 bits).
    ge=2*(n+2)*2.0**-52
    gr=2*max(n-1,0)*2.0**-52
    lower_error=float(np.nextafter(error/(1+ge),-np.inf))
    upper_error=float(np.nextafter(error/(1-ge),np.inf))
    lower_energy=float(np.nextafter(energy/(1+gr),-np.inf))
    upper_energy=float(np.nextafter(energy/(1-gr),np.inf))
    upper_test=float(np.nextafter(upper_error*100000000,np.inf))
    lower_test=float(np.nextafter(lower_error*100000000,-np.inf))
    common=dict(relative_RMS=math.sqrt(error/energy),error_interval=[lower_error,upper_error],
                energy_interval=[lower_energy,upper_energy],error_relative_bound=ge,reference_relative_bound=gr)
    if upper_test<=lower_energy:
        return dict(RMS_gate=True,method='bounded_F64',**common)
    if lower_test>upper_energy:
        return dict(RMS_gate=False,method='bounded_F64',**common)
    numerator=0;denominator=0
    for cb,rb in zip(actual.view(np.uint32).tolist(),reference.view(np.uint32).tolist(),strict=True):
        cv,rv=units(cb),units(rb)
        numerator+=(cv-rv)**2
        denominator+=rv*rv
    assert denominator>0
    return dict(RMS_gate=numerator*100000000<=denominator,method='exact_dyadic',
                squared_error_integer=str(numerator),reference_energy_integer=str(denominator),
                square_units='2^-298',**common)
