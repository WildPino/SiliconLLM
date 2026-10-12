"""Reproducible real-arithmetic diagnostic of uniform pre-learning sign mixture.

This is not a donor forward, trained checkpoint or deployed top8 evaluation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
EXPECTED='d6b2464044e34db4f179d1a17101d37f7ad91274580637fe625d2e8bdea389a2'


def extent(p):
    p=Path(p).resolve();data=p.read_bytes()
    return dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def main(a):
    began=time.monotonic()
    initializer=ROOT/'benchmarks/native_expert_scaling/original_falcon_learner.py'
    before=extent(initializer);assert before['sha256']==EXPECTED
    # Two complete partitions, four sign quadrants, four layers, three neurons.
    rng=random.Random(120610);E=4*2*4*3;rows=[]
    signs=((1,1),(1,-1),(-1,1),(-1,-1))
    for case in range(100):
        neurons=[[(rng.uniform(-4,4),rng.uniform(-4,4),rng.uniform(-2,2))
                  for _ in range(3)] for _ in range(4)]
        terms=[];negative_terms=[];quadrant_errors=[]
        for layer in neurons:
            for g,u,d in layer:
                quadrant=math.fsum(sg*su*max(sg*g,0)*max(su*u,0) for sg,su in signs)
                quadrant_errors.append(abs(quadrant-g*u))
                for _ in range(2):
                    for sg,su in signs:
                        terms.append((1/E)*(sg*su*(E/16)*d)*max(sg*g,0)*max(su*u,0))
                        negative_terms.append((1/E)*(sg*su*(E/16)*d)*max(-sg*g,0)*max(-su*u,0))
        assert len(terms)==E
        actual=math.fsum(terms);negative=math.fsum(negative_terms)
        even_mean=math.fsum(.5*d*g*u for layer in neurons for g,u,d in layer)/4
        source_mean=math.fsum(d*g/(1+math.exp(-g))*u for layer in neurons for g,u,d in layer)/4
        rows.append(dict(case=case,uniform_mixture=actual,even_mean=even_mean,
            full_source_mean=source_mean,even_identity_delta=abs(actual-even_mean),
            sign_reversal_delta=abs(actual-negative),source_function_difference=abs(actual-source_mean),
            quadrant_delta=max(quadrant_errors)))
    assert extent(initializer)==before
    assert max(r['even_identity_delta'] for r in rows)<=1e-12
    assert max(r['sign_reversal_delta'] for r in rows)==0
    result=dict(schema='INITIALIZER_UNIFORM_QUADRANT_IDENTITY_REPRODUCIBLE_V2',
        initializer=before,source=extent(__file__),python=extent(sys.executable),
        command=[sys.executable,*sys.argv],deterministic_seed=120610,synthetic_cases=100,
        distribution='Each g,u uniform[-4,4],down uniform[-2,2];four layers x three neurons.',
        max_even_identity_delta=max(r['even_identity_delta'] for r in rows),
        max_quadrant_delta=max(r['quadrant_delta'] for r in rows),
        max_sign_reversal_delta=max(r['sign_reversal_delta'] for r in rows),
        max_source_function_difference=max(r['source_function_difference'] for r in rows),
        records=rows,initializer_before_after_exact=True,
        quantization=False,top8_runtime_evaluated=False,actual_donor_or_checkpoint_evaluated=False,
        seconds=time.monotonic()-began,
        scope='Uniform ALL-expert real PRE-learning mixture on SAME input equals mean source-even FFNs; '
            'not sequential four-block composition or learned checkpoint or capacity ceiling. '
            'Earlier one-off JSON preserved; this successor specifies full input distribution and source.')
    with a.out.open('x',encoding='utf8',newline='\n') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ('synthetic_cases','max_even_identity_delta',
        'max_sign_reversal_delta','max_source_function_difference','seconds')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
