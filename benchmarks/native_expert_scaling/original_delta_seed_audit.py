"""Stored-only independent seed/native/checkpoint qualification; no refit/update."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json,math,struct,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
import numpy as np
start=time.monotonic()
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
rpath=DOC/'original_delta_seed_finish_result_20261009.json';r=read(rpath);t=read(rpath.with_suffix('.terminal.json'))
bpath=DOC/'original_delta_seed_finish_binding_20261009.json';b=read(bpath)
assert sha(rpath)==t['result_sha256'] and sha(bpath)==r['binding_sha256']==t['binding_sha256']
for item in b['inputs']+t['output_files']:
    assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256'],item['path']
ns=Path(r['checkpoint']['path']).parent
assert r['initial_counter']==26 and r['final_counter']==r['durable_counter']==27 and r['new_updates']==r['optimizer_updates']==1 and r['source_calls']==0
assert t['exit_code']==0 and t['resource_gates'] and len(r['children'])==2 and all(c['exit_code']==0 for c in r['children'])
for ext,key in [('f32','before_logits'),('routes','before_routes'),('witness','before_witness')]:assert sha(ns/('initial.'+ext))==b[key]['sha256']
assert (ns/'after.witness').read_bytes()==(ns/'expected_updated.witness').read_bytes()
def fields(path):
    out={}
    with Path(path).open('rb') as f:
        table=f.read(80+110*104);assert table[:8]==b'E4BPv001'
        for i in range(110):
            v=struct.unpack('<64s6I2Q',table[80+104*i:80+104*(i+1)]);name=v[0].split(b'\0')[0].decode()
            out[name]=np.memmap(path,dtype='<f4' if v[1]==1 else 'u1',mode='r',offset=v[-2],shape=v[3:3+v[2]])
    return table,out
parent_table,parent=fields(b['parent_packed']['path']);initial_table,initial=fields(ns/'seeded26.packed');final_table,final=fields(r['packed']['path'])
assert parent_table==initial_table==final_table
with np.load(ns/'seed_rows.npz',allow_pickle=False) as saved:seeds={k:saved[k].copy() for k in saved.files}
unchanged=0;cross=[]
for name in parent:
    if name in [f'layers.{l}.x_proj' for l in range(5)]:
        l=int(name.split('.')[1]);u=seeds[f'site{l}']
        assert np.array_equal(initial[name][:16].view('<u4'),parent[name][:16].view('<u4')) and np.array_equal(initial[name][48:].view('<u4'),parent[name][48:].view('<u4'))
        assert np.array_equal(initial[name][16:48].view('<u4'),u.view('<u4')) and np.array_equal(final[name][16:48].view('<u4'),u.view('<u4'))
        X=np.fromfile(b['adopted_features'][l]['path'],dtype='<f4').reshape(1507,1024).astype('f8');old_drive=X@parent[name][:16].astype('f8').T;new_drive=X@u.astype('f8').T
        value=float(np.linalg.norm(old_drive.T@new_drive)/max(np.linalg.norm(old_drive)*np.linalg.norm(new_drive),1e-24))
        rank=int(np.linalg.matrix_rank(new_drive));assert rank==32 and value<=1e-6
        ledger=r['seed']['sites'][l];assert abs(value-ledger['FIT_response_cross_relative'])<1e-12
        assert np.linalg.norm(u.astype('f8'),axis=1).max()<=ledger['new_row_L2_limit']*(1+1e-6)
        cross.append(dict(site=l,new_FIT_rank=rank,cross_relative=value))
    else:assert np.array_equal(initial[name].view('u1'),parent[name].view('u1')),name;unchanged+=1
assert unchanged==105
rd=np.dtype([('ids','<i4',(8,)),('mass','<f4',(8,))]);gpu_rt=np.memmap(ns/'GPU_after.routes',dtype=rd,mode='r',shape=(1507,6));native_rt=np.memmap(ns/'after.routes',dtype=rd,mode='r',shape=(3352,6))
head=np.memmap(ns/'after.f32',dtype='<f4',mode='r',shape=(3352,65537));gpu=np.memmap(ns/'GPU_after.f32',dtype='<f4',mode='r',shape=(1507,65537));rows=[];greedy=[]
for pos in range(1507):
    x=gpu[pos].astype('f8');y=head[357+pos].astype('f8');value=math.sqrt(float(np.sum((x-y)**2))/max(float(np.sum(y*y)),1e-24));rows.append(value)
    if np.argmax(x)!=np.argmax(y):greedy.append(pos)
numeric=r['native_numerical_comparison'];failed=[i for i,v in enumerate(rows) if v>1e-4]
assert len(failed)==numeric['failed_rows'] and len(greedy)==numeric['greedy_mismatch_rows'] and abs(max(rows)-numeric['worst_row_relative_RMS'])<1e-12
assert max(abs(v-row['relative_RMS']) for v,row in zip(rows,numeric['rows']))<1e-12
nrt=native_rt[357:1864];ordered=int(np.any(gpu_rt['ids']!=nrt['ids'],axis=-1).sum());sets=int(np.any(np.sort(gpu_rt['ids'],axis=-1)!=np.sort(nrt['ids'],axis=-1),axis=-1).sum());mass=float(np.max(np.abs(gpu_rt['mass'].astype('f8')-nrt['mass'].astype('f8'))))
assert ordered==numeric['ID_mismatch_calls'] and sets==numeric['set_ID_mismatch_calls'] and mass==numeric['mass_max_delta']
metrics=[];offset=0;max_loss_delta=0.
for rec,original in zip(b['sequences'],r['after_native']):
    n=len(rec['student_input_ids']);teacher=np.memmap(rec['logits']['path'],dtype='<u2',mode='r',shape=(256,65537));losses=[];dis=0
    for j,pos in enumerate(rec['positions']):
        q=(teacher[j].astype('<u4')<<16).view('<f4').astype('f8');p=head[offset+pos].astype('f8');q-=q.max();p-=p.max()
        lq=q-math.log(float(np.exp(q).sum()));lp=p-math.log(float(np.exp(p).sum()));losses.append(float(np.sum(np.exp(lq)*(lq-lp))));dis+=int(np.argmax(q)!=np.argmax(p))
    ids=native_rt['ids'][offset:offset+n];m=native_rt['mass'][offset:offset+n]
    assert ids.min()>=0 and ids.max()<1152 and (np.diff(np.sort(ids,axis=-1),axis=-1)>0).all() and np.isfinite(m).all() and (m>=0).all()
    defect=float(np.max(np.abs(m.astype('f8').sum(-1)-1)));assert defect<=1e-6 and defect==original['mass_defect']
    for first in range(offset,offset+n,64):assert np.isfinite(head[first:min(first+64,offset+n)]).all()
    val=float(np.mean(losses));max_loss_delta=max(max_loss_delta,abs(val-original['KL']));assert dis==original['disagreement'];metrics.append(dict(id=rec['id'],KL=val,disagreement=dis));offset+=n
assert max_loss_delta<1e-12
import torch
torch.set_num_threads(1)
state=torch.load(r['checkpoint']['path'],map_location='cpu',weights_only=True,mmap=True)
assert state['schema']=='ORIGINAL_DELTA_SEEDED_STATE_V1' and state['updates']==27 and state['new_updates']==1 and not state['optimizer_partial_possible']
assert len(state['model'])==92 and sum(x.numel() for x in state['model'].values())==721008128
ids=state['optimizer']['param_groups'][0]['params'];names=list(state['model']);assert len(ids)==92
assert all(int(state['optimizer']['state'][i]['step'])==27 for i in ids)
disk=[]
for l in range(5):
    for key in ('x_proj','dt_proj'):assert np.array_equal(state['model'][f'layers.{l}.organs.{key}'].numpy().view('<u4'),final[f'layers.{l}.{key}'].view('<u4'))
    u_name=f'layers.{l}.organs.x_proj';v_name=f'layers.{l}.organs.dt_proj';u_st=state['optimizer']['state'][ids[names.index(u_name)]];v_st=state['optimizer']['state'][ids[names.index(v_name)]]
    assert not torch.count_nonzero(u_st['exp_avg'][16:48]) and not torch.count_nonzero(u_st['exp_avg_sq'][16:48])
    for field in ('exp_avg','exp_avg_sq'):assert torch.any(v_st[field][:,16:48]!=0,dim=0).all()
    v=state['model'][v_name].numpy();assert np.any(v[:,16:48]!=0,axis=0).all();disk.append(dict(site=l,new_V_nonzero=int(np.count_nonzero(v[:,16:48])),new_V_rank_F64=int(np.linalg.matrix_rank(v[:,16:48].astype('f8'))),all_V_rank_F64=int(np.linalg.matrix_rank(v.astype('f8')))))
out=dict(schema='ORIGINAL_DELTA_SEED_STORED_AUDIT_V1',result_sha256=t['result_sha256'],binding_sha256=r['binding_sha256'],checked_inputs=len(b['inputs']),checked_outputs=len(t['output_files']),
    new_output_bytes=sum(x['bytes'] for x in t['output_files']),old_output_bytes=sum(x['bytes'] for x in b['old_output_files']),combined_held_seconds=b['parent_family_held_seconds']+t['elapsed_seconds'],
    initial_native_identity=True,unchanged_initial_fields=unchanged,FIT_seed_cross=cross,checkpoint_deserialized_CPU=True,all_92_disk_Adam_steps=27,disk_new_V=disk,native_metrics=metrics,max_loss_delta=max_loss_delta,
    GPU_native_failed_positions=failed,greedy_mismatch_positions=greedy,ordered_ID_mismatches=ordered,set_ID_mismatches=sets,mass_delta=mass,
    scope='Stored-only hashes/native metrics/routes/seed geometry andCPU checkpoint deserialization;no new prediction/source/update/refit. Script elapsed only,not separately held family resources.',elapsed_seconds=time.monotonic()-start)
path=DOC/'original_delta_seed_stored_adjudication_20261009.json'
with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out,indent=2))
