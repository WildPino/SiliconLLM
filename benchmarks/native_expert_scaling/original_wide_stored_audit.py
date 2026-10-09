"""Independent stored-byte/endpoint audit; no model evaluation or optimizer step."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
import numpy as np
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
start=time.monotonic()
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
result_path=DOC/'original_wide_bridge_result_20261009.json'
r=read(result_path);t=read(DOC/'original_wide_bridge_result_20261009.terminal.json')
b=read(DOC/'original_wide_bridge_binding_20261009.json')
assert sha(result_path)==t['result_sha256']
assert sha(DOC/'original_wide_bridge_binding_20261009.json')==r['binding_sha256']==t['binding_sha256']
for item in b['inputs']+t['output_files']:
    assert Path(item['path']).stat().st_size==item['bytes'],item['path']
    assert sha(item['path'])==item['sha256'],item['path']
ns=ROOT/'results/native_expert_scaling/original_wide_bridge_20261009'
assert {p.name for p in ns.iterdir()}=={Path(x['path']).name for x in t['output_files']}
assert r['initial_export']['sha256']==b['initial_fixture']['sha256']
assert r['checkpoint']['bytes']==8652209850 and r['packed']['bytes']==520029440
assert r['initial_counter']==25 and r['final_counter']==r['durable_counter']==26 and r['new_updates']==r['optimizer_updates']==1
assert r['source_calls']==0 and not any(r[k] for k in ('quality_admission','speed_admission','useful_large_n_admission'))
assert (ns/'after.witness').read_bytes()==(ns/'expected_updated.witness').read_bytes()
assert (ns/'after.witness').stat().st_size==18432*4
assert t['exit_code']==0 and t['resource_gates'] and len(r['children'])==1 and r['children'][0]['exit_code']==0
assert r['transport']['changed_master_tensors']==45 and r['transport']['unchanged_master_tensors']==47
assert len(r['transport']['rows'])==92 and r['transport']['parameters']==721008128
assert sum(x['nonzero_new_coefficients'] for x in r['update']['new_read_changes'])==655360
assert (ns/'after.f32').stat().st_size==3352*65537*4
assert (ns/'GPU_after.f32').stat().st_size==1507*65537*4
assert (ns/'after.routes').stat().st_size==3352*6*64
native=np.memmap(ns/'after.f32',dtype='<f4',mode='r',shape=(3352,65537))
gpu=np.memmap(ns/'GPU_after.f32',dtype='<f4',mode='r',shape=(1507,65537))
rows=[];greedy=[]
for pos in range(1507):
    x=gpu[pos].astype('f8');y=native[357+pos].astype('f8')
    assert np.isfinite(x).all() and np.isfinite(y).all()
    value=math.sqrt(float(np.sum((x-y)**2))/max(float(np.sum(y*y)),1e-24))
    rows.append(value)
    if np.argmax(x)!=np.argmax(y):greedy.append(pos)
delta=max(abs(x-y['relative_RMS']) for x,y in zip(rows,r['native_numerical_comparison']['rows']))
assert delta<1e-12
failed=[i for i,v in enumerate(rows) if v>1e-4]
assert len(failed)==r['native_numerical_comparison']['failed_rows']==26
assert len(greedy)==r['native_numerical_comparison']['greedy_mismatch_rows']==1
route_dtype=np.dtype([('ids','<i4',(8,)),('mass','<f4',(8,))])
metrics=[];max_metric_delta=0.
for arm,path,routepath,key in [('before',b['before_logits']['path'],b['before_routes']['path'],'before_native_adopted'),('after',ns/'after.f32',ns/'after.routes','after_native')]:
    z=np.memmap(path,dtype='<f4',mode='r',shape=(3352,65537))
    rt=np.memmap(routepath,dtype=route_dtype,mode='r',shape=(3352,6));offset=0
    for rec,original in zip(b['sequences'],r[key]):
        teacher=np.memmap(rec['logits']['path'],dtype='<u2',mode='r',shape=(len(rec['positions']),65537))
        losses=[];dis=0
        for j,pos in enumerate(rec['positions']):
            q=(teacher[j].astype('<u4')<<16).view('<f4').astype('f8');q-=q.max();q-=math.log(float(np.exp(q).sum()))
            p=z[offset+pos].astype('f8');p-=p.max();p-=math.log(float(np.exp(p).sum()))
            losses.append(float(np.sum(np.exp(q)*(q-p))));dis+=int(np.argmax(p)!=np.argmax(q))
        n=len(rec['student_input_ids']);ids=rt['ids'][offset:offset+n];mass=rt['mass'][offset:offset+n]
        assert ids.min()>=0 and ids.max()<1152 and (np.diff(np.sort(ids,axis=-1),axis=-1)>0).all()
        assert np.isfinite(mass).all() and (mass>=0).all()
        for first in range(offset,offset+n,64):assert np.isfinite(z[first:min(first+64,offset+n)]).all()
        defect=float(np.max(np.abs(mass.astype('f8').sum(-1)-1)))
        unions=[int(np.unique(ids[:,l]).size) for l in range(6)]
        val=float(np.mean(losses));max_metric_delta=max(max_metric_delta,abs(val-original['KL']))
        assert dis==original['disagreement'] and unions==original['unions'] and defect==original['mass_defect']
        metrics.append(dict(arm=arm,id=rec['id'],KL=val,disagreement=dis,labels=len(losses),unions=unions,mass_defect=defect))
        offset+=n
assert max_metric_delta<1e-12
out=dict(schema='ORIGINAL_WIDE_STORED_AUDIT_V1',result_sha256=t['result_sha256'],binding_sha256=r['binding_sha256'],
    checked_inputs=len(b['inputs']),checked_outputs=len(t['output_files']),output_bytes=sum(x['bytes'] for x in t['output_files']),
    metrics=metrics,max_metric_delta=max_metric_delta,max_row_metric_delta=delta,
    failed_positions=failed,greedy_mismatch_positions=greedy,worst_position=int(np.argmax(rows)),worst_row_relative_RMS=max(rows),
    supervised_failed_positions=[i for i in failed if i in b['sequences'][1]['positions']],
    GPU_native_supervised_KL_delta=r['GPU_KL_after']-r['after_native'][1]['KL'],
    independently_rechecked_GPU_routes=False,reason='GPU routes remained in worker memory; raw GPU IDs/masses were not persisted. Four ID calls and mass delta are worker observations only.',
    independently_deserialized_checkpoint=False,scope='All input/output hashes and stored native/GPU heads; six native teacher metrics and native route validity. No new prediction, update, source call or numerical cause claim.',
    elapsed_seconds=time.monotonic()-start)
path=DOC/'original_wide_bridge_stored_adjudication_20261009.json'
with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(out,indent=2))
