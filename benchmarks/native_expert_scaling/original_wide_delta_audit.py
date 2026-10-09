"""Stored packed delta-factor audit and exact algebraic zero-lock witness."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json,struct,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
import numpy as np
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
NS=ROOT/'results/native_expert_scaling/original_wide_bridge_20261009'
FIELD=struct.Struct('<64s6I2Q');start=time.monotonic()
def fields(path):
    with path.open('rb') as f:
        header=f.read(80);assert header[:8]==b'E4BPv001'
        cfg=struct.unpack('<16I',header[8:72]);assert cfg[6:8]==(1024,48) and cfg[-1]==110
        out={}
        for _ in range(110):
            row=FIELD.unpack(f.read(FIELD.size));name=row[0].split(b'\0')[0].decode()
            if name.endswith(('.x_proj','.dt_proj')):
                assert row[1]==1
                shape=row[3:3+row[2]]
                out[name]=np.memmap(path,dtype='<f4',mode='r',offset=row[-2],shape=shape)
    return out
checks=[]
for filename in ('initial_wide.packed','candidate_wide26.packed'):
    data=fields(NS/filename)
    for l in range(5):
        u=data[f'layers.{l}.x_proj'][16:48,:]
        v=data[f'layers.{l}.dt_proj'][:,16:48]
        assert u.shape==(32,1024) and v.shape==(1024,32)
        assert np.count_nonzero(u)==np.count_nonzero(v)==0
        checks.append(dict(artifact=filename,site=l,new_U_nonzero=0,new_V_nonzero=0))
# z = V U x. For any upstream derivative g, dV = g (U x)^T,
# dU = (V^T g) x^T. Both vanish at U=V=0, independent of softplus upstream.
# Small deterministic witness: single-factor initialization preserves z=0 and
# allows a nonzero first dV; it is not an actual model-gradient observation.
x=np.arange(1,10,dtype='f8');g=np.arange(1,8,dtype='f8')
u=np.zeros((4,9));v=np.zeros((7,4))
dead_dv=np.outer(g,u@x);dead_du=np.outer(v.T@g,x)
assert not np.any(dead_dv) and not np.any(dead_du)
seed=np.eye(4,9,dtype='f8')/8
assert np.array_equal(v@(seed@x),v@(u@x))
unlocked_dv=np.outer(g,seed@x)
assert np.count_nonzero(unlocked_dv)>0
out=dict(schema='ORIGINAL_WIDE_DELTA_ZERO_LOCK_V1',packed_checks=checks,
    frozen_learner='original_wide_learner.py',U='x_proj[16:48,:]',V='dt_proj[:,16:48]',
    proof='z_new=V U x; dV=g(Ux)^T; dU=(V^Tg)x^T. U=V=0 gives both derivatives identically zero. Zero first/second Adam moments and no decay preserve this subspace under subsequent updates.',
    scope='Proof applies to the frozen computation graph and deterministic continuation without a new initialization or optimizer perturbation. Packed snapshots confirm both factors zero before and after update26; checkpoint moments are runtime observations, not independently deserialized here.',
    active_delta_factor_rank_upper_bound=16,advertised_delta_coordinates=48,
    single_factor_algebra_witness=dict(output_unchanged=True,dead_dV_nonzero=int(np.count_nonzero(dead_dv)),dead_dU_nonzero=int(np.count_nonzero(dead_du)),seeded_dV_nonzero=int(np.count_nonzero(unlocked_dv))),
    proposed_correction='Explicitly fork actual26; initialize new U only with bounded independent FIT-informed directions, keep V=0/new moments0. Native pre-update parity must be measured; real new V gradients/changes must be observed. No quality gain follows from this witness.',
    new_predictions=0,new_optimizer_updates=0,elapsed_seconds=time.monotonic()-start)
with (DOC/'original_wide_delta_zero_lock_20261009.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out,indent=2))
