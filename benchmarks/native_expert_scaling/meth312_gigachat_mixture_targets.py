#!/usr/bin/env python3
"""Reconstruct full source MoE FFN targets on existing anchor-conditioned real inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits
import meth310_gigachat_regional_subspace as R
import meth300_gigachat_vector_lut_cost as H
C=R.C
P=R.P
DOC=C.DOC
PROTOCOL=DOC/'METH_312_GIGACHAT_MIXTURE_TARGETS_PROTOCOL_20261003.md'
OUT=C.OUT/'meth312_mixture_targets'
peak_checked=0

def budget(start):
    global peak_checked
    rss=psutil.Process().memory_info().rss;peak_checked=max(peak_checked,rss)
    elapsed=time.monotonic()-start
    assert elapsed<=20*60 and rss<=12*(1<<30),'M312 wall/RSS stop'
    return {'seconds':elapsed,'rss_bytes':rss,'maximum_checked_rss_bytes':peak_checked}

def deduplicate(arrays,coords,layer):
    rows={}
    for e in (0,32,63):
        for co,x in zip(coords[layer,e,0],arrays[layer,e,0]):
            key=co[0],co[1],co[3]
            if key in rows:assert np.array_equal(rows[key],x),'same-token source input differs'
            else:rows[key]=x
    ordered=sorted(rows);lookup={key:i for i,key in enumerate(ordered)}
    x=np.stack([rows[key] for key in ordered]);anchors={}
    for e in (0,32,63):
        indices=np.asarray([lookup[c,b,t] for c,b,s,t in coords[layer,e,0]],dtype=np.int64)
        slots=np.asarray([s for c,b,s,t in coords[layer,e,0]],dtype=np.int64)
        assert len(indices)==len(set(indices.tolist()))
        anchors[e]=(indices,slots,arrays[layer,e,2])
    return x,np.asarray(ordered,dtype=np.int32),anchors

def routing(x,router,bias):
    logits=(x.astype(np.float64)@router.astype(np.float64).T).astype(np.float32)
    probs=(np.float32(1)/(np.float32(1)+np.exp(-logits))).astype(np.float32)
    rank=(probs+bias[None,:]).astype(np.float32)
    chosen=np.argsort(-rank,axis=1,kind='stable')[:,:4].astype(np.int16)
    gates=probs[np.arange(len(x))[:,None],chosen].copy()
    denominator=np.maximum(np.sum(gates,axis=1,keepdims=True,dtype=np.float32),np.float32(6.103515625e-5))
    gates/=denominator
    assert np.isfinite(gates).all() and (gates>0).all() and np.max(np.abs(gates.sum(axis=1)-1))<=1e-6
    assert all(len(set(row.tolist()))==4 for row in chosen)
    return chosen,gates

def ff_values(x,gate,up,down):
    post=[];output=[]
    for lo in range(0,len(x),256):
        xx=torch.from_numpy(np.ascontiguousarray(x[lo:lo+256]))
        g=xx@gate.T;u=xx@up.T;z=torch.nn.functional.silu(g)*u;y=z@down.T
        post.append(z.numpy());output.append(y.numpy())
    z=np.concatenate(post);y=np.concatenate(output)
    assert np.isfinite(z).all() and np.isfinite(y).all()
    return z,y

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists()
    assert not OUT.exists(),'preserve existing target directory'
    start=time.monotonic();stage='bindings';result={'experiment':'METH-312-source-complete-MoE-FFN-target-reconstruction','cases':[],'payloads':[]}
    torch.set_num_threads(6)
    try:
        for path in (Path(__file__),PROTOCOL,Path(R.__file__),Path(P.__file__),Path(H.__file__)):P.committed(path)
        assert C.digest(Path(H.__file__))=='986abfee67cc7137bed3aecaf6a32c469d38c61d481b16859275476ce6e92ea4'
        assert C.digest(Path(R.__file__))=='8fed130998c32a3cd1ba342958bce2ab179e9834b6bc64b666c30ab8fb352719'
        C.validate_build();assert C.digest(C.PRIOR)==C.PRIOR_SHA
        prior=C.read(C.PRIOR);config_path=C.SOURCE/'config.json'
        assert C.digest(config_path)==prior['source_config_sha256']
        config=C.read(config_path)
        for k,v in {'hidden_size':1536,'n_routed_experts':64,'n_shared_experts':1,'num_experts_per_tok':4,
                    'moe_intermediate_size':1280,'topk_method':'noaux_tc','n_group':1,'topk_group':1,
                    'norm_topk_prob':True,'scoring_func':'sigmoid','routed_scaling_factor':1.0}.items():assert config[k]==v,k
        stage='fresh_full_BF16_source_hash';assert C.digest(H.SOURCE)==C.MODEL_SHA
        fields,tensors,header=H.header(H.SOURCE,21356264448)
        assert header['header_sha256']=='2e18041f5c90d897f4ab3f882887c63b797f736fec63b1147c8cd9e0d1bcbe08'
        result.update({'source_model':{'path':str(H.SOURCE),'bytes':H.SOURCE.stat().st_size,'full_sha256_freshly_checked':C.MODEL_SHA},
                       'header':header,'controller_sha256':C.digest(Path(__file__)),'protocol_sha256':C.digest(PROTOCOL),
                       'source_config_sha256':C.digest(config_path),'libraries':{'torch':torch.__version__,'numpy':np.__version__},
                       'capture_bindings':{}})
        arrays={};coords={}
        for d,sha in P.CAP_SHA.items():
            path=DOC/f'meth298_{d}_capture_result.json';P.committed(path);assert C.digest(path)==sha
            info=C.read(path);assert all(info['gates'].values())
            cap=Path(info['capture']['path']);assert C.binding(cap)==info['capture']
            data,co,n=C.parse_capture(cap,info['chunks']);assert n==info['records']
            for l in (1,13,25):
                for e in (0,32,63):
                    assert co[l,e,0]==co[l,e,1]==co[l,e,2] and np.array_equal(data[l,e,0],data[l,e,1])
            arrays[d]={k:v for k,v in data.items() if k[2] in (0,2)};coords[d]={k:v for k,v in co.items() if k[2] in (0,2)}
            result['capture_bindings'][d]=info['capture'];del data,co
        OUT.mkdir();archive_bytes=0
        with H.SOURCE.open('rb') as stream,threadpool_limits(limits=6):
            def load(name,shape,kind,layer=None,organ=None):
                t=tensors[name];assert tuple(t['shape'])==shape and t['type']==kind,name
                offset=header['header_bytes_read']+t['offset'];stream.seek(offset);raw=stream.read(t['bytes']);assert len(raw)==t['bytes']
                manifest={'name':name,'offset':offset,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
                result['payloads'].append(manifest)
                if kind=='F32':return np.frombuffer(raw,dtype='<f4').copy().reshape(tuple(reversed(shape)))
                bits=np.frombuffer(raw,dtype='<u2').copy().reshape(tuple(reversed(shape)))
                if layer is not None:
                    for e in (0,32,63):
                        ref=next(v for v in prior['rows'] if v['layer']==layer and v['expert']==e and v['projection']==organ)
                        assert hashlib.sha256(bits[e].tobytes()).hexdigest()==ref['tensor_bf16_sha256']
                return torch.from_numpy(bits).view(torch.bfloat16).float()
            for layer in (1,13,25):
                stage=f'layer{layer}_source_weights'
                prefix=f'blk.{layer}.'
                router=load(prefix+'ffn_gate_inp.weight',(1536,64),'F32');bias=load(prefix+'exp_probs_b.bias',(64,),'F32')
                gate=load(prefix+'ffn_gate_exps.weight',(1536,1280,64),'BF16',layer,'gate_proj')
                up=load(prefix+'ffn_up_exps.weight',(1536,1280,64),'BF16',layer,'up_proj')
                down=load(prefix+'ffn_down_exps.weight',(1280,1536,64),'BF16',layer,'down_proj')
                sg=load(prefix+'ffn_gate_shexp.weight',(1536,1280),'BF16')
                su=load(prefix+'ffn_up_shexp.weight',(1536,1280),'BF16')
                sd=load(prefix+'ffn_down_shexp.weight',(1280,1536),'BF16');budget(start)
                for domain in ('fit','test'):
                    stage=f'layer{layer}_{domain}_targets'
                    x,coordinates,anchors=deduplicate(arrays[domain],coords[domain],layer)
                    chosen,gates=routing(x,router,bias)
                    for e,(ix,slots,_) in anchors.items():assert np.array_equal(chosen[ix,slots],np.full(len(ix),e)),('captured source route slot mismatch',layer,domain,e)
                    parent=np.empty((len(x),4,1536),dtype=np.float32);controls=[]
                    for e in range(64):
                        indices,slots=np.nonzero(chosen==e)
                        if not len(indices):continue
                        post,out=ff_values(x[indices],gate[e],up[e],down[e]);parent[indices,slots]=out
                        if e in anchors:
                            ix,_,original=anchors[e];local=np.full(len(x),-1,dtype=np.int64);local[indices]=np.arange(len(indices));assert (local[ix]>=0).all()
                            actual=post[local[ix]].astype(np.float64);ref=original.astype(np.float64)
                            norm=np.linalg.norm(ref,axis=1);assert (norm>0).all()
                            rel=np.linalg.norm(actual-ref,axis=1)/norm;assert np.isfinite(rel).all()
                            controls.append({'expert':e,'rows':len(ix),'maximum_post_SwiGLU_relative_l2':float(rel.max()),'summary':P.stats(rel)})
                            assert float(rel.max())<=1e-4,('captured nonlinear reference mismatch',layer,domain,e,float(rel.max()))
                        budget(start)
                    _,shared=ff_values(x,sg,su,sd)
                    weighted=np.sum(parent.astype(np.float64)*gates.astype(np.float64)[:,:,None],axis=1)
                    full=(weighted+shared.astype(np.float64)).astype(np.float32)
                    assert np.isfinite(full).all() and np.isfinite(parent).all()
                    for i in (0,len(x)//2,len(x)-1):
                        independent=shared[i].astype(np.float64).copy()
                        for k in range(4):independent+=float(gates[i,k])*parent[i,k].astype(np.float64)
                        assert np.linalg.norm(independent-full[i].astype(np.float64))/np.linalg.norm(independent)<=1e-6
                    # A missing selected-parent contribution must change the full target.
                    omitted=(weighted-gates[:,0,None].astype(np.float64)*parent[:,0].astype(np.float64)+shared.astype(np.float64)).astype(np.float32)
                    full_norm=np.linalg.norm(full.astype(np.float64),axis=1);assert (full_norm>0).all()
                    fault=float(np.max(np.linalg.norm(omitted.astype(np.float64)-full.astype(np.float64),axis=1)/full_norm));assert fault>1e-6
                    archive=OUT/f'meth312_layer{layer}_{domain}.npz'
                    with archive.open('xb') as f:np.savez(f,input=x,coordinates=coordinates,selected_parents=chosen,gates=gates,
                        parent_outputs=parent,shared_output=shared,full_mixture_output=full)
                    with np.load(archive,allow_pickle=False) as check:
                        for key,value in {'input':x,'coordinates':coordinates,'selected_parents':chosen,'gates':gates,'parent_outputs':parent,'shared_output':shared,'full_mixture_output':full}.items():assert np.array_equal(check[key],value),key
                    archive_bytes+=archive.stat().st_size;assert archive_bytes<=4*(1<<30),'archive byte stop'
                    case={'layer':layer,'domain':domain,'unique_source_inputs':len(x),'captured_anchor_rows':sum(len(v[0]) for v in anchors.values()),
                          'selected_expert_counts':np.bincount(chosen.ravel(),minlength=64).tolist(),'nonlinear_capture_controls':controls,
                          'omitted_parent_fault_max_relative_l2':fault,'archive':C.binding(archive),'runtime':budget(start)}
                    result['cases'].append(case);print(json.dumps({'completed':[layer,domain],'unique_inputs':len(x),'archive_bytes':archive.stat().st_size,
                        'max_nonlinear_error':max(v['maximum_post_SwiGLU_relative_l2'] for v in controls),'seconds':case['runtime']['seconds']}),flush=True)
                    del parent,weighted,full,shared,omitted
                del gate,up,down,sg,su,sd
        assert len(result['cases'])==6
        result['gates']={'source_full_hash_and_payload_bindings':True,'all_captured_anchor_slots_match':True,
                         'all_original_post_SwiGLU_rows_relative_l2_1e_4':True,'sum_shared_controls_and_omission_fault':True,'all_archives_bit_exact_roundtrip':True}
        result['archive_total_bytes']=archive_bytes;result['runtime']=budget(start)
        result['decision']='targets_eligible_for_separately_frozen_complete_mixture_representation_diagnostic'
        result['scope']='Mathematical full source four-parent-plus-shared FFN on actual normalized source inputs, not captured native full-mixture bit parity. Inputs conditioned on anchor0/32/63 selection; both domains consumed and coverage biased. All64 source parents may participate, but no realized compact student, attention/full causal quality, useful extra n, native cost or same-artifact50tokens/s.'
        C.write(args.out,result);print(json.dumps({'decision':result['decision'],'gates':result['gates'],'archive_total_bytes':archive_bytes,
            'runtime':result['runtime'],'result_sha256':C.digest(args.out)}),flush=True)
    except BaseException as error:
        result.update({'failure_stage':stage,'error':repr(error),'seconds':time.monotonic()-start})
        C.write(args.out.with_suffix('.failure.json'),result);raise

if __name__=='__main__':main()
