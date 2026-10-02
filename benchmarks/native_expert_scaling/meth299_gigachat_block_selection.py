#!/usr/bin/env python3
"""Real GigaChat nonlinear channel-block omission, with full-output greedy labels."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil
from safetensors import safe_open
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from benchmarks.native_expert_scaling import meth298_gigachat_covariance as C

DOC=C.DOC
CAP_SHA={'fit':'57cc46da176b6be16953ecfadb21c173d85bb0b8bb6dcc026ddba709fb9c0331',
         'test':'124610e4ddf83f2f8f5d58bc7465deaebc91183c2452614522cdb1ef4db8f4fa'}
SCREEN=DOC/'meth298_gigachat_full_covariance_result.json'
SCREEN_SHA='b0b9a4773b02dc6c309504f2eddc9a8dcc623d115d70c751df9d392219d00223'
PROTOCOL=DOC/'METH_299_GIGACHAT_NONLINEAR_BLOCK_PROTOCOL_20261002.md'
BLOCK=32;TOTAL=40;COUNTS=(5,10,20);BATCH=64

def budget(start):
    r={'seconds':time.monotonic()-start,'rss_bytes':psutil.Process().memory_info().rss}
    assert r['seconds']<=15*60 and r['rss_bytes']<=12*(1<<30),'M299 resource stop'
    return r

def stats(values):
    v=np.asarray(values,dtype=np.float64)
    assert v.size and np.isfinite(v).all() and (v>=0).all()
    return {'count':v.size,'median':float(np.median(v)),
            'p95':float(np.percentile(v,95,method='linear')),'max':float(v.max())}

def committed(path):
    rel=path.relative_to(ROOT).as_posix()
    b=subprocess.check_output(['git','-c','core.autocrlf=false','cat-file','--filters','--path='+rel,'HEAD:'+rel],cwd=ROOT)
    assert b==path.read_bytes(),str(path)+' not frozen'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();assert not args.out.exists()
    start=time.monotonic();stage='bindings';rows=[]
    torch.set_num_threads(6)
    try:
        for path in (Path(__file__),PROTOCOL,SCREEN,*[DOC/f'meth298_{d}_capture_result.json' for d in CAP_SHA]):committed(path)
        C.validate_build()
        assert C.digest(SCREEN)==SCREEN_SHA
        screen=C.read(SCREEN);assert screen['decision']=='reject_fixed_full_covariance_rank192_export'
        prior=C.read(C.PRIOR);assert C.digest(C.PRIOR)==C.PRIOR_SHA
        assert C.digest(C.SOURCE/'model.safetensors.index.json')==prior['source_index_sha256']
        assert C.digest(C.SOURCE/'config.json')==prior['source_config_sha256']
        arrays={};coords={};captures={}
        for d in CAP_SHA:
            p=DOC/f'meth298_{d}_capture_result.json';assert C.digest(p)==CAP_SHA[d]
            r=C.read(p);assert all(r['gates'].values())
            cap=Path(r['capture']['path']);assert C.binding(cap)==r['capture']
            data,co,n=C.parse_capture(cap,r['chunks']);assert n==r['records']
            arrays[d]={k:v for k,v in data.items() if k[2]==2}
            coords[d]={k:v for k,v in co.items() if k[2]==2}
            captures[d]=r['capture'];del data,co
        refs=[x for x in prior['rows'] if x['projection']=='down_proj']
        assert len(refs)==9
        stage='nonlinear_block_omission'
        with threadpool_limits(limits=6):
            for ref in refs:
                l,e=ref['layer'],ref['expert'];key=l,e,2
                with safe_open(C.SOURCE/ref['source_shard'],framework='pt',device='cpu') as source:
                    tensor=source.get_tensor(f'model.layers.{l}.mlp.experts.{e}.down_proj.weight')
                assert tensor.dtype==torch.bfloat16
                assert hashlib.sha256(tensor.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()==ref['tensor_bf16_sha256']
                w=tensor.float().numpy().astype(np.float64);assert w.shape==(1536,1280)
                tile_norm=np.array([np.sum(w[:,i*BLOCK:(i+1)*BLOCK]**2) for i in range(TOTAL)])
                static_order=np.argsort(-tile_norm,kind='stable')
                for d in CAP_SHA:
                    z=arrays[d][key];assert z.shape[1]==1280 and len(z)>=384
                    errors={mode+str(k):[] for mode in ('greedy','static') for k in COUNTS}
                    reference_norm=[];closure_max=0.
                    for lo in range(0,len(z),BATCH):
                        x=z[lo:lo+BATCH].astype(np.float64);n=len(x)
                        y=x@w.T
                        tiles=np.stack([x[:,j*BLOCK:(j+1)*BLOCK]@w[:,j*BLOCK:(j+1)*BLOCK].T for j in range(TOTAL)],axis=1)
                        norm=np.linalg.norm(y,axis=1);assert (norm>0).all()
                        closure=np.linalg.norm(tiles.sum(axis=1)-y,axis=1)/norm
                        closure_max=max(closure_max,float(closure.max()))
                        assert closure_max<=1e-12,'all40 tile/full down closure'
                        norms2=np.einsum('btd,btd->bt',tiles,tiles)
                        selected=np.zeros((n,TOTAL),dtype=bool);residual=y.copy()
                        for step in range(1,max(COUNTS)+1):
                            # Direct residual-norm decrease. No rescaling or refit.
                            decrease=2*np.einsum('btd,bd->bt',tiles,residual)-norms2
                            decrease[selected]=-np.inf
                            choice=np.argmax(decrease,axis=1)  # lowest tile-ID ties
                            assert not selected[np.arange(n),choice].any()
                            selected[np.arange(n),choice]=True
                            residual-=tiles[np.arange(n),choice]
                            if step in COUNTS:
                                errors['greedy'+str(step)].extend((np.linalg.norm(residual,axis=1)/norm).tolist())
                        assert (selected.sum(axis=1)==max(COUNTS)).all()
                        for k in COUNTS:
                            approximation=tiles[:,static_order[:k]].sum(axis=1)
                            errors['static'+str(k)].extend((np.linalg.norm(y-approximation,axis=1)/norm).tolist())
                        reference_norm.extend(norm.tolist());budget(start)
                    row={'domain':d,'layer':l,'expert':e,'source_down_tensor_sha256':ref['tensor_bf16_sha256'],
                         'states':len(z),'coordinates':coords[d][key],
                         'reference_output_l2':reference_norm,'relative_l2_by_state':errors,
                         'summary':{a:stats(v) for a,v in errors.items()},
                         'all40_full_projection_max_relative_l2':closure_max,
                         'static_tile_order':static_order.tolist()}
                    rows.append(row)
                    print(json.dumps({'completed':[d,l,e],'states':len(z),
                       'greedy20':row['summary']['greedy20'],'seconds':budget(start)['seconds']}),flush=True)
        assert len(rows)==18
        pooled={d:{a:stats([v for row in rows if row['domain']==d for v in row['relative_l2_by_state'][a]])
                   for a in rows[0]['summary']} for d in CAP_SHA}
        gates={}
        for k in COUNTS:
            gates[str(k)]={}
            for d in CAP_SHA:
                s=pooled[d]['greedy'+str(k)];static=pooled[d]['static'+str(k)]
                gates[str(k)][d+'_pooled_median1']=s['median']<=.01
                gates[str(k)][d+'_pooled_p95_5']=s['p95']<=.05
                gates[str(k)][d+'_half_static_median']=s['median']<=.5*static['median']
                gates[str(k)][d+'_every_expert_median1_p95_5']=all(
                    row['summary']['greedy'+str(k)]['median']<=.01 and row['summary']['greedy'+str(k)]['p95']<=.05
                    for row in rows if row['domain']==d)
        passing=[k for k in COUNTS if all(gates[str(k)].values())]
        result={'experiment':'METH-299-GigaChat-real-nonlinear32-channel-greedy-block-preservation',
                'source_revision':prior['source_revision'],'capture_bindings':captures,
                'prior_screen_sha256':SCREEN_SHA,'block_channels':BLOCK,'total_blocks_per_macro_expert':TOTAL,
                'retained_blocks':list(COUNTS),'rows':rows,'pooled':pooled,'gates':gates,
                'smallest_passing_blocks':min(passing) if passing else None,
                'runtime':budget(start),'script_sha256':C.digest(Path(__file__)),
                'diagnostic_only':True,'native_promotion_qualified':False,
                'decision':'eligible_for_separately_costed_input_only_selector' if passing else 'reject_fixed_greedy32_block_omission_at_half_width',
                'scope':'Full-activation/full-output oracle; not optimal combinatorial subset,no saved gate/up reads,input-only routing,native sparse speed or new capacity. Both calibration domains consumed,not full-model heldout quality.'}
        C.write(args.out,result);assert args.out.stat().st_size<=50_000_000
        print(json.dumps({'gates':gates,'pooled':pooled,'decision':result['decision'],
                          'result_sha256':C.digest(args.out)}),flush=True)
    except BaseException as error:
        C.write(args.out.with_suffix('.failure.json'),{'stage':stage,'error':str(error),'partial_rows':rows,
                       'seconds':time.monotonic()-start});raise

if __name__=='__main__':main()
