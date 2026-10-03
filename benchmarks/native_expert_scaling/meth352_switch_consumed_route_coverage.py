"""Model-free descriptive coverage on consumed343 routes, no useful-n gate."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import psutil
import meth324_switch_reference as M

BASE=M.DOC/'meth343_switch_fresh_prediction_result.json'
BASE_SHA='a4838903958be5f6bb567c1dedd802a17e6b27e2d97c4f490210139f296cbadf'
COMPOSED=M.DOC/'meth347_switch_head_a16_contract_resume_result.json'
COMPOSED_SHA='0242039aef21bbb087ae7c973a4bfe718507d8da2f80b8ba796d5d5ee8792c45'
PROTOCOL=M.DOC/'METH_352_SWITCH_CONSUMED_ROUTE_COVERAGE_PROTOCOL_20261003.md'
DATA=M.ROOT/'results/native_expert_scaling/meth343_switch_fresh_prediction'


def summary(ids,prob,accepted):
    counts=np.bincount(ids,minlength=256);assert counts.shape==(256,)
    p=counts[counts>0]/counts.sum()
    return {'positions':int(counts.sum()),'distinct_selected':int((counts>0).sum()),
            'counts':counts.tolist(),'unused_indices':np.flatnonzero(counts==0).tolist(),
            'max_count':int(counts.max()),'max_over_mean_load':float(counts.max()/(counts.sum()/256)),
            'normalized_load_entropy':float(-(p*np.log(p)).sum()/np.log(256)),
            'uniform_iid_expected_distinct':float(256*(1-(1-1/256)**counts.sum())),
            'selected_probability_min_median_max':[float(prob.min()),float(np.median(prob)),float(prob.max())],
            'accepted_positions':int(accepted.sum())}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists()
    start=time.monotonic();result={'experiment':'METH-352-consumed343-route-coverage-descriptive','bindings':[]}
    def guard():
        assert time.monotonic()-start<=120 and psutil.Process().memory_info().rss<=1<<30,'coverage_resource_guard'
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            while block:=f.read(4<<20):h.update(block);guard()
        return h.hexdigest()
    try:
        for path in (Path(__file__),BASE,COMPOSED,PROTOCOL,Path(M.__file__)):M.committed(path)
        assert digest(BASE)==BASE_SHA and digest(COMPOSED)==COMPOSED_SHA
        base=json.loads(BASE.read_text(encoding='utf-8'));composed=json.loads(COMPOSED.read_text(encoding='utf-8'))
        assert not base['gates']['original_top1_agreement_ge0p95'] and all(composed['gates'].values())
        streams={name:{phase:[] for phase in ('encoder','decoder')} for name in ('original','native')}
        changes={'encoder':0,'decoder':0};positions={'encoder':0,'decoder':0}
        for bi,book in enumerate(base['books']):
            for ci,case in enumerate(book['cases']):
                stem=f'book{bi}.case{ci}';native=DATA/(stem+'.0.bin');original=DATA/(stem+'.original_reference.npz')
                assert digest(native)==case['native_output_sha256'] and digest(original)==case['original_reference_sha256']
                with native.open('rb') as f:
                    assert f.read(8)==b'SWR32O01';s,t,d,e,l,v,n=np.frombuffer(f.read(28),dtype='<u4').tolist()
                    assert (s,t,d,e,l,v,n)==(57,11,768,12,12,32128,408)
                    f.seek(36+4*((e+2)*s*d+t*(l+2)*d+t*v))
                    a=np.frombuffer(f.read(),dtype=np.dtype([('expert','<i4'),('accepted','<i4'),('probability','<f4')]))
                    target=np.column_stack([a[k] for k in ('expert','accepted','probability')]).astype(np.float64)
                with np.load(original) as arrays:source=arrays['routes'].copy()
                assert source.shape==target.shape==(408,3) and np.isfinite(source).all() and np.isfinite(target).all()
                splits={name:{'encoder':r[:6*s].reshape(6,s,3),'decoder':r[6*s:].reshape(t,6,3).transpose(1,0,2)} for name,r in (('original',source),('native',target))}
                for phase in ('encoder','decoder'):
                    o=splits['original'][phase];c=splits['native'][phase]
                    assert np.all((o[:,:,0]>=0)&(o[:,:,0]<256)) and np.all((c[:,:,0]>=0)&(c[:,:,0]<256))
                    changes[phase]+=int(np.sum(o[:,:,0]!=c[:,:,0]));positions[phase]+=o.shape[0]*o.shape[1]
                    for name in streams:streams[name][phase].append(splits[name][phase])
                result['bindings'].append({'book':bi,'case':ci,'native_sha256':case['native_output_sha256'],'original_sha256':case['original_reference_sha256']})
        result['banks']={}
        for name in streams:
            result['banks'][name]={}
            for phase in ('encoder','decoder'):
                routes=np.concatenate(streams[name][phase],axis=1)
                result['banks'][name][phase]=[{'block':2*i+1,**summary(row[:,0].astype(np.int64),row[:,2],row[:,1])} for i,row in enumerate(routes)]
        result['changed_choices']={phase:{'changed':changes[phase],'positions':positions[phase],'fraction':changes[phase]/positions[phase]} for phase in changes}
        result.update({'controller_sha256':M.digest(__file__),'protocol_sha256':M.digest(PROTOCOL),'baseline343_sha256':BASE_SHA,'composition347_sha256':COMPOSED_SHA,
                       'resource':{'main_seconds_excluding_imports':time.monotonic()-start,'end_rss_bytes':psutil.Process().memory_info().rss},
                       'decision':'descriptive_real256_bank_utilization_only_no_quality_or_larger_n_promotion',
                       'scope':'Consumed343 96x57 encoder/11 teacher-forced decoder paths;347 confirms head-only A16 has identical routes for these inputs. Load entropy is selection frequency entropy, not per-token router entropy. Uniform iid is a null comparison, not measured cache/DRAM locality. Visits/distinct weights do not prove causal capacity gain, all functions useful or larger-n quality.'})
        guard();M.write(args.out,result)
        print(json.dumps({'sha256':M.digest(args.out),'distinct_selected':{name:{phase:[row['distinct_selected'] for row in rows] for phase,rows in phases.items()} for name,phases in result['banks'].items()},'changes':result['changed_choices'],'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'error':repr(error),'seconds':time.monotonic()-start});M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
