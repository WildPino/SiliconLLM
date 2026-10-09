"""Stored-only F64 projection/energy/hash adjudication. No model or CUDA calls."""
import os
os.environ['OPENBLAS_NUM_THREADS']='6'
os.environ['OMP_NUM_THREADS']='6'
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
import numpy as np

start=time.monotonic()
def read(path): return json.loads(Path(path).read_bytes())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
def energy(x):
    x=x.astype('f8'); mean=x.mean(0)
    return dict(total=float(np.sum(x*x)),mean=float(len(x)*np.sum(mean*mean)),centered=float(np.sum((x-mean)**2)))

rp=DOC/'original_history_boundaries_result_20261009.json';r=read(rp)
t=read(rp.with_suffix('.terminal.json'));bp=Path(t['command'][t['command'].index('--binding')+1]);b=read(bp)
assert sha(rp)==t['result_sha256'] and sha(bp)==r['binding_sha256']==t['binding_sha256']
for item in b['inputs']+t['output_files']:
    assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256'],item['path']
assert t['exit_code']==0 and t['resource_gates'] and r['cases']==r['source_base_forwards']==r['source_base_attempts']==48
assert r['exact_history_output_witnesses']==288 and r['source_parameter_identities_versions_unchanged']
assert r['LM_head_calls']==r['source_generations']==r['optimizer_updates']==r['native_calls']==r['reserved_queries']==0
P=np.fromfile(r['basis']['path'],dtype='<f4').reshape(2048,256).astype('f8')
gram=P.T@P;defect=float(np.max(np.abs(gram-np.eye(256))))
assert abs(defect-r['basis_orthogonality_max'])<1e-12 and defect<=b['criteria']['projection_orthogonality_max']
rows=[];raw=0;projected=0;max_projection=0.;max_energy_delta=0.;max_discarded_identity_delta=0.
for rec in r['records']:
    assert [item['boundary'] for item in rec['boundaries']]==b['boundaries'] and rec['exact_output_witnesses']==b['witness_sites']
    for item in rec['boundaries']:
        n=rec['history'];bits=np.fromfile(item['h']['path'],dtype='<u2').reshape(n,2048)
        h=(bits.astype('<u4')<<16).view('<f4').astype('f8');z=np.fromfile(item['z']['path'],dtype='<f4').reshape(n,256).astype('f8')
        assert np.isfinite(h).all() and np.isfinite(z).all()
        expected=h@P
        err=np.sqrt(np.sum((expected-z)**2,axis=1)/np.maximum(np.sum(expected*expected,axis=1),1e-24))
        worst=float(err.max());max_projection=max(max_projection,worst)
        assert worst<=b['criteria']['projection_relative_RMS'],(rec['id'],item['boundary'],worst)
        eh=energy(h);ez=energy(z)
        for actual,stored in ((eh,item['source_energy']),(ez,item['projected_energy'])):
            for key in actual:
                delta=abs(actual[key]-stored[key])/max(abs(actual[key]),1e-24)
                max_energy_delta=max(max_energy_delta,delta);assert delta<1e-10
            assert abs(actual['total']-actual['mean']-actual['centered'])/max(actual['total'],1e-24)<1e-10
        retained=ez['total']/max(eh['total'],1e-24);centered=ez['centered']/max(eh['centered'],1e-24)
        assert abs(retained-item['retained_energy_fraction'])<1e-12 and abs(centered-item['centered_retained_fraction'])<1e-12
        # Algebraically exact residual norm, including measured nonorthogonality
        # and F32 z error: ||h-zP^T||^2 = ||h||^2 -2<hP,z> + <z(P^TP),z>.
        exact_discard=eh['total']-2*float(np.sum(expected*z))+float(np.sum((z@gram)*z))
        estimate=item['discarded_energy_difference'];assert abs(estimate-eh['total']+ez['total'])<1e-7
        delta=abs(exact_discard-estimate)/max(eh['total'],1e-24)
        max_discarded_identity_delta=max(max_discarded_identity_delta,delta);assert exact_discard>=-1e-10*max(eh['total'],1e-24)
        raw+=item['h']['bytes'];projected+=item['z']['bytes']
        rows.append(dict(id=rec['id'],split=rec['split'],domain=rec['domain'],boundary=item['boundary'],
            positions=n,projection_worst_row_relative_RMS=worst,retained_fraction=retained,
            centered_retained_fraction=centered,source_mean_fraction=eh['mean']/max(eh['total'],1e-24),
            exact_discarded_fraction=exact_discard/max(eh['total'],1e-24)))
assert len(rows)==336 and raw==b['raw_payload_bytes'] and projected==b['projected_payload_bytes']
aggregates=[]
for split in ('FIT','DEV'):
    for boundary in b['boundaries']:
        group=[row for row in rows if row['split']==split and row['boundary']==boundary];assert len(group)==24
        domains=[]
        for domain in sorted({row['domain'] for row in group}):
            pair=[row for row in group if row['domain']==domain];assert len(pair)==2
            domains.append(dict(domain=domain,retained_mean=float(np.mean([row['retained_fraction'] for row in pair])),
                centered_retained_mean=float(np.mean([row['centered_retained_fraction'] for row in pair]))))
        aggregates.append(dict(split=split,boundary=boundary,cases=24,
            retained_mean=float(np.mean([row['retained_fraction'] for row in group])),
            retained_min=float(min(row['retained_fraction'] for row in group)),
            centered_retained_mean=float(np.mean([row['centered_retained_fraction'] for row in group])),
            centered_retained_min=float(min(row['centered_retained_fraction'] for row in group)),domains=domains))
out=dict(schema='ORIGINAL_HISTORY_BOUNDARIES_STORED_AUDIT_V1',decision='STORED_BOUNDARIES_PROJECTION_PASS',
    binding_sha256=r['binding_sha256'],result_sha256=t['result_sha256'],checked_inputs=len(b['inputs']),checked_outputs=len(t['output_files']),
    output_bytes=sum(item['bytes'] for item in t['output_files']),raw_payload_bytes=raw,projected_payload_bytes=projected,
    projection_rows=sum(row['positions'] for row in rows),projection_worst_row_relative_RMS=max_projection,
    energy_relative_delta_max=max_energy_delta,basis_orthogonality_max=defect,
    discarded_estimate_vs_exact_relative_delta_max=max_discarded_identity_delta,aggregates=aggregates,rows=rows,
    source_calls=0,optimizer_updates=0,native_calls=0,
    scope='All saved projection rows independently checked by CPU NumPy F64 hP;all energy summaries/hashes checked.'
        'Script elapsed only,not separately held family resources;energy fractions are not a chatbot preservation theorem.',
    elapsed_seconds=time.monotonic()-start)
path=DOC/'original_history_boundaries_stored_adjudication_20261009.json'
with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({key:value for key,value in out.items() if key not in ('rows','aggregates')},indent=2))
