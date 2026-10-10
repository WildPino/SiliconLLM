"""Small structural/algebraic gates for the unconsumed full-history runner."""
import argparse,ast,json,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from original_packed_capacity import SITE
sys.path.insert(0,str(SITE))
import numpy as np
import original_categorical_campaign_geometry as g
from original_master_transition import encode


def main(out):
    start=time.monotonic()
    for name in ('original_categorical_campaign.py','original_categorical_campaign_geometry.py','original_categorical_campaign_audit.py','original_categorical_campaign_launch.py'):
        ast.parse((B/name).read_text())
    baseline=dict(weighted_KL=10.,first_KL=1.)
    rows=[dict(alpha=.0001,metrics=dict(weighted_KL=8.,first_KL=2.),numeric_flags=dict(valid=True)),
          dict(alpha=.001,metrics=dict(weighted_KL=9.,first_KL=.5),numeric_flags=dict(valid=True))]
    choice=g.choose(rows,baseline);assert choice['accepted'] and choice['alpha']==.001
    assert not g.choose(rows,baseline,False)['accepted']
    rows[0]['metrics']=dict(weighted_KL=9.,first_KL=.5)
    assert g.choose(rows,baseline)['alpha']==.0001
    request=dict(query_ids=[1,2,3],generated_ids=[4,11],input_ids=3,reused_prefix_ids=0,state_reset=True,
        core_calls=5,head_calls=5,cached_ids=5,eos=11,core_seconds=None,mlp_seconds=None,head_seconds=None,
        request_seconds=.31,prefill_seconds=.1,decode_seconds=.2,forward_seconds=.29,load_seconds=.5,pipe_request_seconds=.4)
    assert g.stream_counters(request,0,True)==2
    try:g.stream_counters(dict(request,core_calls=4),0,True);raise RuntimeError('wrong core count accepted')
    except AssertionError:pass
    try:g.stream_counters(dict(request,eos=0),0,True);raise RuntimeError('wrong EOS accepted')
    except AssertionError:pass
    assert g.stream_counters(dict(request,reused_prefix_ids=2,state_reset=False,core_calls=3,head_calls=3),2,False)==2
    g.V,g.D,g.E,g.L,g.K=7,4,5,2,2
    with tempfile.TemporaryDirectory(prefix='campaign_preflight_') as temporary:
        directory=Path(temporary)
        source={'copy':np.array([0.,-0.],dtype='<f4'),'raw':np.zeros(64,dtype='<f4'),'compress':np.zeros(4096,dtype='<f4')}
        target={k:v.copy() for k,v in source.items()};target['compress'].view('<u4')[::32]=1
        target['raw']=np.random.default_rng(43).integers(0,2**32,64,dtype='<u4').view('<f4')
        index=encode(source,target,directory/'delta',256)
        reconstructed=g.decode_transition(source,directory/'delta'/'transition.json')
        assert all(np.array_equal(reconstructed[k].numpy().view('<u4'),v.view('<u4')) for k,v in target.items())
        assert all(index['codec_counts'][k]>0 for k in ('copy','xor_zlib','raw'))
        bad={k:v.copy() for k,v in source.items()};bad['copy'].view('<u4')[0]=1
        try:g.decode_transition(bad,directory/'delta'/'transition.json');raise RuntimeError('bad source accepted')
        except AssertionError:pass
        # Variable-length synthetic readout: no model, native or GPU execution.
        H=np.arange(28,dtype='<f4').reshape(7,4)/32;H[:,-1]=0
        features=np.asarray([[[.2,.3,.1,.9],[.4,-.1,.2,.8]]],dtype='<f4');z=features[0].astype('f8')@H.astype('f8').T
        teacher=np.asarray([[.5,.25,0,-.25,-.5,.75,.125],[-.25,.5,0,.25,-.5,.125,.75]],dtype='<f4')
        bits=(teacher.view('<u4')>>16).astype('<u2');teacher_path=directory/'teacher.bin';teacher_path.write_bytes(bits.tobytes())
        q,lq=g.probabilities(g.decode(bits),False);p,lp=g.probabilities(z,False);m=q@H.astype('f8');c=np.sum(q*lq,axis=1)
        direct=np.sum(q*(lq-lp),axis=1);comp=z.max(1)+np.log(np.exp(z-z.max(1,keepdims=True)).sum(1))-np.sum(m*features[0],axis=1)+c
        gd=((p-q)@H.astype('f8'))*.5;gc=(p@H.astype('f8')-m)*.5
        art={}
        for key,x in [('features',features),('GPU_logits',z[None]),('direct_loss',direct[None]),('compiled_loss',comp[None]),
                       ('direct_gradient',gd[None]),('compiled_gradient',gc[None]),('backward_feature_gradient',gc.astype('<f4')[None])]:
            art[key]=g.array(directory,key,x)
        ids=np.broadcast_to(np.array([1,2],dtype='<i4'),(3,2,2)).copy();mass=np.full((3,2,2),.5,dtype='<f4')
        for key,x in [('forward_route_ids',ids),('backward_route_ids',ids),('forward_route_mass',mass),('backward_route_mass',mass)]:art[key]=g.array(directory,key,x)
        scores=np.zeros((3,7),dtype='<f4');scores[1:]=z;score_path=directory/'native.f32';scores.tofile(score_path)
        dtype=np.dtype([('ids','<i4',(2,)),('mass','<f4',(2,))]);r=np.empty((3,2),dtype=dtype);r['ids']=ids;r['mass']=mass;route_path=directory/'native.routes';r.tofile(route_path)
        from original_packed_capacity import extent
        rec=dict(id='synthetic',positions=[1,2],student_input_ids=[0,1,2],logits=extent(teacher_path))
        native=dict(scores=extent(score_path),routes=extent(route_path));target_stats={k:g.array(directory,k,x) for k,x in [('moments',m),('negative_entropy',c)]}
        head=g.array(directory,'head',H);history=dict(artifacts=art,block_forward_calls=[2,2],gradient_role_squared_norms={k:1. for k in ('embed','core','bank','router')})
        numeric=dict(loss_absolute=1e-8,gradient_absolute=1e-8,F32_upstream_absolute=1e-6,GPU_native_score_absolute=1e-3,GPU_native_KL_absolute=1e-3,routing_mass_absolute=1e-5,routing_mass_defect=1e-6)
        ordinary=g.bridge(rec,target_stats,head,native,history,numeric,False)
        independent=g.bridge(rec,target_stats,head,native,history,numeric,True)
        assert all(ordinary['numeric_flags'].values()) and all(independent['numeric_flags'].values())
        changed_ids=ids.copy();changed_ids[2,1,1]=3;art['forward_route_ids']=g.array(directory,'fault_ids',changed_ids)
        assert not g.bridge(rec,target_stats,head,native,history,numeric,True)['numeric_flags']['route_ID']
    import torch
    result=dict(AST_PASS=True,constrained_feasible_selection_PASS=True,tie_smaller_alpha_PASS=True,baseline_numeric_rejection_PASS=True,
        independent_all3_transition_modes_PASS=True,wrong_source_rejected=True,variable_history_synthetic_F64_loss_gradient_PASS=True,
        route_ID_fault_detected=True,original_stream_counter_cache_EOS_PASS=True,core_counter_EOS_faults_rejected=True,
        Torch=torch.__version__,NumPy=np.__version__,seconds=time.monotonic()-start,
        actual_model_loads=0,history_forwards=0,backward_calls=0,native_calls=0,GPU_calls=0,
        scope='Synthetic structural/algebra checks only, not an actual full-history numerical qualification.')
    g.emit(out,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);main(parser.parse_args().out)
