"""Stored-byte audit and exact integer explanation of the first AQ divergence."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json,math,struct,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
sys.path.insert(0,str(B))
from original_wide_trace import layout,SITE,DOC
sys.path.insert(0,str(SITE))
import numpy as np
start=time.monotonic();NS=ROOT/'results/native_expert_scaling/original_wide_trace_20261009'
def read(p):return json.loads(Path(p).read_bytes())
def digest(p,offset=0,length=None):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        f.seek(offset)
        while length is None or length>0:
            chunk=f.read(min(8<<20,length) if length is not None else 8<<20)
            if not chunk:break
            h.update(chunk)
            if length is not None:length-=len(chunk)
    assert length is None or length==0
    return h.hexdigest()
rpath=DOC/'original_wide_trace_result_20261009.json';r=read(rpath);t=read(rpath.with_suffix('.terminal.json'))
bpath=DOC/'original_wide_trace_binding_20261009.json';b=read(bpath)
assert digest(rpath)==t['result_sha256'] and digest(bpath)==t['binding_sha256']==r['binding_sha256']
for row in b['inputs']+t['output_files']:
    assert Path(row['path']).stat().st_size==row['bytes'] and digest(row['path'])==row['sha256'],row['path']
dt=layout(np);g=np.memmap(NS/'GPU_trace.bin',dtype=dt,mode='r',shape=(1507,6));c=np.memmap(NS/'native_trace.bin',dtype=dt,mode='r',shape=(1507,6))
assert digest(NS/'GPU_heads.f32')==b['GPU_heads']['sha256']
assert digest(NS/'native_heads.f32')==digest(b['native_heads']['path'],357*65537*4,1507*65537*4)
for name in ('ids','xq','hq','gint','uint','dint'):
    differences=g[name]!=c[name];sites=np.any(differences.reshape(1507,6,-1),axis=-1);m=r['trace']['discrete'][name]
    assert int(differences.sum())==m['coordinates'] and np.argwhere(sites).tolist()==m['locations']
same_route=np.all(g['ids']==c['ids'],axis=-1);same_xq=np.all(g['xq']==c['xq'],axis=-1);same_hq=np.all(g['hq']==c['hq'],axis=(-1,-2))
conditional={name:int(np.any(g[name]!=c[name],axis=(-1,-2))[same_route&same_xq].sum()) for name in ('gint','uint')}
conditional['dint']=int(np.any(g['dint']!=c['dint'],axis=(-1,-2))[same_route&same_hq].sum())
assert conditional==dict(gint=0,uint=0,dint=0)
T,L,coord=82,0,43
assert np.argwhere(g['xq']!=c['xq'])[0].tolist()==[T,L,coord]
assert np.flatnonzero(g['xq'][T,L]!=c['xq'][T,L]).tolist()==[coord]
assert np.array_equal(g['ids'][T,L],c['ids'][T,L])
with Path(b['packed']['path']).open('rb') as f:
    header=f.read(80);assert header[:8]==b'E4BPv001';cfg=struct.unpack('<16I',header[8:72]);assert cfg[6:8]==(1024,48)
    fields={}
    for _ in range(cfg[-1]):
        row=struct.unpack('<64s6I2Q',f.read(104));name=row[0].split(b'\0')[0].decode();fields[name]=(row[3:3+row[2]],row[-2])
def field(name,dtype):
    shape,offset=fields[name];return np.memmap(b['packed']['path'],dtype=dtype,mode='r',shape=shape,offset=offset)
proof=[]
for organ,key in [('gate','gint'),('up','uint')]:
    code=field(f'layers.0.{organ}_code','u1')
    for rank,e in enumerate(g['ids'][T,L]):
        pairs=code[coord//2,int(e)*128:(int(e)+1)*128].astype('i4')
        weights=(pairs//3-1) if coord%2==0 else (pairs%3-1)
        predicted=(int(g['xq'][T,L,coord])-int(c['xq'][T,L,coord]))*weights
        observed=g[key][T,L,rank].astype('i4')-c[key][T,L,rank].astype('i4')
        assert np.array_equal(predicted,observed)
        proof.append(dict(organ=organ,expert=int(e),rank=rank,integer_delta_equals_single_weight_column=True,nonzero=int(np.count_nonzero(observed))))
relative={}
for key in ('pre','core_norm','core','post_core','ff_norm','moe','post_moe'):
    x=g[key][T,L].astype('f8');y=c[key][T,L].astype('f8');relative[key]=math.sqrt(float(np.sum((x-y)**2))/max(float(np.sum(y*y)),1e-24))
scale_info={}
for arm,data in [('GPU',g),('native',c)]:
    x=data['ff_norm'][T,L];amax=np.max(np.abs(x));idx=int(np.argmax(np.abs(x)))
    exact_div=np.float32(float(amax)/63.);multiply=np.float32(amax*np.float32(1/63.))
    scale_info[arm]=dict(max_coordinate=idx,amax=float(amax),observed_scale=float(data['sa'][T,L]),correctly_rounded_division_scale=float(exact_div),multiply_reciprocal_scale=float(multiply),
        observed_matches_division=bool(data['sa'][T,L]==exact_div),observed_matches_multiply=bool(data['sa'][T,L]==multiply))
gh=np.memmap(NS/'GPU_heads.f32',dtype='<f4',mode='r',shape=(1507,65537));ch=np.memmap(NS/'native_heads.f32',dtype='<f4',mode='r',shape=(1507,65537));bad=[];worst=0.;greedy=[]
for pos in range(1507):
    x=gh[pos].astype('f8');y=ch[pos].astype('f8');value=math.sqrt(float(np.sum((x-y)**2))/max(float(np.sum(y*y)),1e-24));worst=max(worst,value)
    if value>1e-4:bad.append(pos)
    if np.argmax(x)!=np.argmax(y):greedy.append(pos)
previous=read(DOC/'original_wide_bridge_stored_adjudication_20261009.json')
assert bad==previous['failed_positions'] and greedy==previous['greedy_mismatch_positions'] and abs(worst-previous['worst_row_relative_RMS'])<1e-12
discrete_tokens=sorted(set(np.argwhere(np.any(g['xq']!=c['xq'],axis=-1))[:,0].tolist())|set(np.argwhere(np.any(g['hq']!=c['hq'],axis=(-1,-2)))[:,0].tolist()))
result=dict(schema='ORIGINAL_WIDE_TRACE_STORED_AUDIT_V1',result_sha256=t['result_sha256'],binding_sha256=r['binding_sha256'],checked_inputs=len(b['inputs']),checked_outputs=len(t['output_files']),
    output_bytes=sum(x['bytes'] for x in t['output_files']),observer_heads_bit_identical=True,discrete_recomputation_exact=True,conditional_integer_mismatch_sites=conditional,
    first_input_AQ=dict(position=T,site=L,coordinate=coord,GPU_code=int(g['xq'][T,L,coord]),native_code=int(c['xq'][T,L,coord]),relative_RMS=relative,scale_analysis=scale_info,integer_weight_column_proofs=proof),
    failed_head_positions=bad,greedy_mismatch_positions=greedy,worst_head_RMS=worst,quant_mismatch_tokens=discrete_tokens,
    failed_head_positions_without_same_token_quant_mismatch=sorted(set(bad)-set(discrete_tokens)),
    scope='Stored fields only: all byte hashes, independent head/discrete audits, conditional integer checks and exact first single-code-flip weight-column identity. No new prediction/update/source call. No global proof that all floating errors originate from this first flip.',elapsed_seconds=time.monotonic()-start)
out=DOC/'original_wide_trace_stored_adjudication_20261009.json'
with out.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
