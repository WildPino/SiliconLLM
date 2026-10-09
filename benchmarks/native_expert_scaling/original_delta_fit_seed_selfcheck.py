"""Bounded algebra verification; no checkpoint, model evaluation or source call."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
sys.path.insert(0,str(B));sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
import numpy as np
from original_delta_fit_seed import seed_from_fit
start=time.monotonic();rng=np.random.default_rng(261027)
x=rng.normal(size=(192,48));old=rng.normal(size=(4,48))*.01
seed,ledger=seed_from_fit(x,old,count=6)
vold=rng.normal(size=(9,4));vnew=np.zeros((9,6))
before=(x@old.T)@vold.T;after=before+(x@seed.astype('f8').T)@vnew.T
assert np.array_equal(before.view('u8'),after.view('u8'))
g=rng.normal(size=(192,9));dv=g.T@(x@seed.astype('f8').T)
assert np.count_nonzero(dv)==54
# All activation responses are already in the old drive image. Numerical
# residual noise must not be accepted as new information-bearing directions.
low=np.zeros_like(x);low[:,:4]=x[:,:4];basis=np.eye(4,48)
try:seed_from_fit(low,basis,count=6)
except AssertionError as e:assert str(e)=='new FIT rank';rank_deficient_rejected=True
else:raise AssertionError('spurious new FIT directions accepted')
result=dict(schema='ORIGINAL_DELTA_FIT_SEED_ALGEBRA_V1',seed=261027,numpy=np.__version__,
    helper_sha256=hashlib.sha256((B/'original_delta_fit_seed.py').read_bytes()).hexdigest(),
    data_sha256=hashlib.sha256(x.tobytes()+old.tobytes()).hexdigest(),new_rows_sha256=hashlib.sha256(seed.tobytes()).hexdigest(),
    zero_read_output_bit_identity=True,new_dV_nonzero=int(np.count_nonzero(dv)),rank_deficient_rejected=rank_deficient_rejected,ledger=ledger,
    elapsed_seconds=time.monotonic()-start,scope='Synthetic finite algebra qualification only;not actual26 calibration,checkpoint fork,model gradient,source quality or native preservation.')
out=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925/original_delta_fit_seed_algebra_20261009.json'
with out.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
