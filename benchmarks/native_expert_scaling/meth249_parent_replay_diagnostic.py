#!/usr/bin/env python3
"""Read-only parent replay: no child fitting or validation targets."""
import json
import time
import numpy as np
from safetensors import safe_open
import torch
import meth249_parent_anchored_latent_pilot as F

B,A,C,P,M,R,Q=F.B,F.A,F.C,F.P,F.M,F.R,F.Q
start=time.monotonic()
assert P.digest(F.D.PAIR)==F.D.PAIR_SHA
pair=json.loads(F.D.PAIR.read_text())
assert P.digest(F.D.ENCODED)==pair['checkpoint_sha256']
assert P.digest(R.CAPTURE)==pair['capture_sha256']
device=M.D.Q.setup();P.MAX_SECONDS=P.M17.MAX_SECONDS=120
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
values,controls=Q.load_layer(json.loads(Q.NATIVE.read_text()),device)
assert controls==pair['controls']
with safe_open(str(F.D.ENCODED),framework='pt',device='cpu') as archive:
    tensors={name:archive.get_tensor(name) for name in archive.keys() if not name.startswith('e160.')}
with np.load(R.CAPTURE,allow_pickle=False) as archive:
    x=torch.from_numpy(archive['x_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
    y=torch.from_numpy(archive['y_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
labels=R.assign_full(x,tensors['router.parents'].to(device))
energy=float(y.double().square().sum());rows=[]
for parent in range(16):
    chosen=labels==parent;phi=Q.features(x[chosen],values)
    base=C.get(tensors,'base',parent,device);factors=B.factor_get(tensors,'e16',parent,device)
    pred=B.encoded_value(phi,base,factors)
    observed=float((pred-y[chosen]).double().square().sum())
    old=next(r for r in pair['fit_rows'] if r['arm']=='e16' and r['cell']==parent)
    rows.append({'parent':parent,'observed_sse':observed,'old_sse':old['actual_factorized_sse'],
        'relative_discrepancy':observed/old['actual_factorized_sse']-1})
    P.budget(start,device)
observed=sum(r['observed_sse'] for r in rows)/energy
result={'experiment':'METH-249-parent-fit-replay-diagnostic','parent_result_sha256':F.D.PAIR_SHA,
    'rows':rows,'energy':energy,'observed_nmse':observed,
    'old_nmse':pair['fit_summary']['e16']['factorized_normalized_sse'],
    'relative_nmse_discrepancy':observed/pair['fit_summary']['e16']['factorized_normalized_sse']-1,
    'runtime':P.budget(start,device),'script_sha256':P.digest(F.Path(__file__)),
    'scope':'Unchanged actual parent fit inputs only;no new fits or validation targets.'}
out=P.DOC/'meth249_parent_replay_diagnostic.json';assert not out.exists()
out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result),flush=True)
