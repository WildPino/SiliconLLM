"""Meaningful algebraic verification of interval decisions against a ratio oracle.

No saved source/student/native outputs are replayed. Run after the live pilot.
"""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_falcon_usability import SITE,sha,write
sys.path.insert(0,str(SITE))


def main():
    start=time.monotonic()
    import numpy as np
    from chatbot_hybrid_logit_certificate import certify
    assert np.__version__=='2.4.6' and sys.byteorder=='little'
    def ratio_units(value):
        numerator,denominator=float(value).as_integer_ratio()
        return numerator*(1<<149)//denominator
    def oracle(actual,reference):
        c=[ratio_units(v) for v in actual];r=[ratio_units(v) for v in reference]
        numerator=sum((x-y)**2 for x,y in zip(c,r,strict=True))
        denominator=sum(y*y for y in r)
        return numerator*100000000<=denominator
    pairs=[]
    def add(label,c,r):
        pairs.append((label,np.array(c,dtype='<f4'),np.array(r,dtype='<f4')))
    add('exact_boundary',[10001],[10000])
    add('below_boundary',[np.nextafter(np.float32(10001),np.float32(-np.inf))],[10000])
    add('above_boundary',[np.nextafter(np.float32(10001),np.float32(np.inf))],[10000])
    add('both_zero',[0,-0.0],[0,0])
    add('zero_reference',[1],[0])
    smallest=np.array([1],dtype='<u4').view('<f4')[0]
    add('subnormal_identity',[smallest],[smallest])
    add('subnormal_difference',[smallest*2],[smallest])
    add('large_sign_change',[-np.finfo(np.float32).max],[np.finfo(np.float32).max])
    rng=np.random.default_rng(20261009)
    for index in range(100):
        n=int(rng.integers(1,129))
        raw=rng.integers(0,1<<32,size=n,dtype=np.uint32)
        raw[((raw>>23)&255)==255]&=np.uint32(0x807fffff)
        r=raw.view('<f4').copy()
        pairs.append((f'identity_{index}',r.copy(),r.copy()))
        c=np.nextafter(r,np.where(r>=0,np.float32(np.inf),np.float32(-np.inf))).astype('<f4')
        finite=np.isfinite(c)
        c[~finite]=r[~finite]
        pairs.append((f'neighbor_{index}',c,r.copy()))
        pairs.append((f'sign_{index}',-r,r.copy()))
    outcomes=[]
    for label,c,r in pairs:
        result=certify(c,r)
        assert result['RMS_gate']==oracle(c,r),label
        outcomes.append(dict(label=label,gate=result['RMS_gate'],method=result['method'],coordinates=c.size))
    assert outcomes[0]['gate'] and outcomes[0]['method']=='exact_dyadic'
    assert outcomes[1]['gate'] and not outcomes[2]['gate']
    report=dict(schema='HYBRID_LOGIT_CERTIFICATE_CHECK_V1',decision='ALGEBRAIC_CERTIFICATE_CHECK_PASS',
          checks=len(outcomes),coordinates=sum(r['coordinates'] for r in outcomes),outcomes=outcomes,
          oracle='F32->exact Python float.as_integer_ratio()->2^-149 integer units; separate from bit decoder',
          implementation_sha256=sha(ROOT/'benchmarks/native_expert_scaling/chatbot_hybrid_logit_certificate.py'),
          checker_sha256=sha(__file__),source_calls=0,student_calls=0,native_calls=0,
          elapsed_seconds=time.monotonic()-start)
    assert report['elapsed_seconds']<=30
    path=Path(sys.argv[1])
    write(path,report)
    print(json.dumps(dict(decision=report['decision'],checks=len(outcomes),seconds=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    main()
