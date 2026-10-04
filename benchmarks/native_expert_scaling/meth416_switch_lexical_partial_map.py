"""ONE rank767 identifiable lexical partial isometry, unchanged local gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import psutil
from threadpoolctl import threadpool_limits, threadpool_info
import meth324_switch_reference as M
import meth405_switch_full_basis_inputs as P
import meth404_switch_rank32_input_map as F

PROTOCOL=M.DOC/'METH_416_SWITCH_LEXICAL_PARTIAL_MAP_PROTOCOL_20261004.md'
RAW=M.DOC/'meth415_switch_lexical_spectra_result.json'
RAW_SHA='c60e3d6bd77b14a5085d59f9fe3704666bb0dcd9f3fc13b08fe96793029b2bba'
PAIRS=M.DOC/'meth405_switch_full_basis_inputs_result.json'
PAIRS_SHA='1fb5c222807da2ad8f8e52e8f3d2d5fc64664431edc2332070305fa46cb64788'
OUT=M.ROOT/'results/native_expert_scaling/meth416_switch_lexical_partial_map'
RANK=767


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();peak=0;stage='bindings';result={'experiment':'METH-416-ONE-fixed-rank767-lexical-partial-isometry-input-screen','rank':RANK,'banks':[]}

    def guard():
        nonlocal peak
        peak=max(peak,psutil.Process().memory_info().rss);assert peak<=1<<30 and time.monotonic()-start<=120,'partial_map_2min_1GiB'

    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block:=stream.read(4<<20):h.update(block);guard()
        return h.hexdigest()

    def partial_error(q):
        a=q@q.T;b=q.T@q
        return max(float(np.linalg.norm(a@a-a)/np.sqrt(RANK)),float(np.linalg.norm(b@b-b)/np.sqrt(RANK)))

    try:
        result['bindings']={}
        for path in (Path(__file__),PROTOCOL,RAW,PAIRS,Path(M.__file__),Path(P.__file__),Path(F.__file__),
                     M.ROOT/'benchmarks/native_expert_scaling/meth415_switch_lexical_spectra.py'):
            M.committed(path);result['bindings'][str(path)]=sha(path)
        assert sha(RAW)==RAW_SHA and sha(PAIRS)==PAIRS_SHA
        prior=json.loads(RAW.read_text(encoding='utf-8'));pairs=json.loads(PAIRS.read_text(encoding='utf-8'))
        assert all(prior['gates'].values()) and all(pairs['gates'].values()) and len(pairs['pairs'])==96
        assert prior['cross']['rank_at414_bound1e_6']==767 and all(v['source_rank_at1e_6']==768 for v in prior['sources'])
        result['frozen_HEAD']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        ancestors={p.pid for p in psutil.Process().parents()}|{os.getpid()}
        for process in psutil.process_iter(['pid','name','cmdline']):
            if process.pid in ancestors:continue
            name=(process.info['name'] or '').lower();argv=process.info['cmdline'] or []
            unrelated=(name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve())
            if unrelated:result.setdefault('unrelated_background_processes',[]).append({'pid':process.pid,'argv':argv});continue
            assert not (name.startswith('python') or ('meth' in name and name.endswith('.exe'))),('concurrent_model_job',process.pid,name)
        archive=prior['archive'];assert sha(archive['path'])==archive['sha256']
        with np.load(archive['path']) as saved:c=saved['cross_covariance'].copy();old_s=saved['cross_singular'].copy()
        assert c.shape==(768,768) and np.isfinite(c).all()
        result['retained_original_weight_covariance']={'archive':archive,'source_weight_identity':prior['sources'],
            'scope':'Fresh complete415 covariance archive SHA; source whole-payload and embedding hashes inherited415, NOT freshly reread416'}
        arrays={n:[[] for _ in range(12)] for n in (256,128)};keys=[]
        for index,pair in enumerate(pairs['pairs']):
            assert (pair['book'],pair['case'])==divmod(index,4)
            assert pair['prospective_split']==('development' if pair['book']<18 else 'validation')
            assert P.pairing_key(pair['source_ids'],pair['decoder_ids'])==pair['pairing_sha256'];keys.append(pair['pairing_sha256'])
            for n,path,expected in ((256,pair['source256_trace_path'],pair['source256_trace_sha256']),
                                    (128,pair['source128_capture']['trace_path'],pair['source128_capture']['trace_sha256'])):
                assert sha(path)==expected
                for bank,values in enumerate(P.read_trace(path,n)):arrays[n][bank].append(values)
        assert len(set(keys))==96;result['paired_input_keys']=keys;OUT.mkdir(parents=True)
        with threadpool_limits(limits=1):
            info=threadpool_info();assert all(v['num_threads']==1 for v in info)
            assert np.__version__==prior['runtime']['numpy'] and [(v['version'],v['architecture']) for v in info]==[(v['version'],v['architecture']) for v in prior['runtime']['BLAS']]
            result['runtime']={'numpy':np.__version__,'BLAS':info,'BLAS_DLL_sha256':[sha(v['filepath']) for v in info]}
            assert result['runtime']['BLAS_DLL_sha256']==prior['runtime']['BLAS_DLL_sha256']
            truth=np.asarray([[0,0,-1,0],[1,0,0,0],[0,0,0,1],[0,-1,0,0]],dtype=np.float64)
            tiny=np.diag([4.,3.,2.,0.])@truth;tu,ts,tv=np.linalg.svd(tiny,full_matrices=False);tq=tu[:,:3]@tv[:3]
            expected=np.diag([1.,1.,1.,0.])@truth;assert np.max(np.abs(tq-expected))<=1e-12
            a=tq@tq.T;b=tq.T@tq;assert np.linalg.norm(a@a-a)+np.linalg.norm(b@b-b)<=1e-12 and abs(np.trace(b)-3)<=1e-12
            bad=tq.copy();bad[0,0]+=.125;bb=bad.T@bad;assert np.linalg.norm(bb@bb-bb)>.001
            wu,ws,wv=np.linalg.svd(np.roll(tiny,1,axis=1),full_matrices=False)
            assert np.linalg.norm(wu[:,:3]@wv[:3]-expected)>.1,'wrong_coordinate_anchor_control'
            stage='ONE_rank767_source_covariance_fit';u,s,vt=np.linalg.svd(c,full_matrices=False)
            assert np.max(np.abs(s-old_s))/s[0]<=1e-10
            assert s[RANK-1]/s[0]>=1e-6 and s[RANK]/s[0]<1e-6,'same_identifiable_subspace_bound_no_relaxation'
            q=u[:,:RANK]@vt[:RANK];q32=q.astype(np.float32);error64=partial_error(q);error32=partial_error(q32.astype(np.float64))
            trace64=float(np.trace(q.T@q));certificate=abs(float(np.sum(q*c))-float(s[:RANK].sum()))/float(s[:RANK].sum())
            assert error64<=1e-10 and error32<=1e-6 and abs(trace64-RANK)<=1e-8 and certificate<=1e-10
            file=OUT/'ONE_rank767_lexical_partial_map.npz';np.savez(file,matrix=q32,matrix64=q,left_basis=u[:,:RANK],right_basis=vt[:RANK],singular=s)
            with np.load(file) as saved:
                assert np.array_equal(saved['matrix'],q32) and np.array_equal(saved['matrix64'],q)
            result['map']={'direction':'row input256 @ rank767 partial map -> source128 coordinates','rank':RANK,'discarded_directions':1,
                'weakest_included_singular_ratio':float(s[RANK-1]/s[0]),'discarded_singular_ratio':float(s[RANK]/s[0]),
                'F64_partial_isometry_idempotence_error':error64,'F32_partial_isometry_idempotence_error':error32,'F64_projection_trace':trace64,
                'rank767_optimal_trace_certificate_relative_error':certificate,'path':str(file),'sha256':sha(file),'archive_bytes':file.stat().st_size,
                'shared_F32_matrix_bytes':768*768*4,'full_matrix_active_coefficients_per_bank':768*768,'ALL12_full_matrix_active_coefficients':12*768*768}
            for bank in range(12):
                stage=f'partial_input_bank_{bank}';positions=4*(29 if bank<6 else 14);cut=18*positions
                x=np.concatenate(arrays[256][bank]).astype(np.float64);y=np.concatenate(arrays[128][bank]).astype(np.float64);ym=y[:cut].mean(axis=0)
                prediction=x[cut:]@q;factor=(x[cut:]@u[:,:RANK])@vt[:RANK]
                factor_error=float(np.linalg.norm(factor-prediction)/max(np.linalg.norm(prediction),1e-30));assert factor_error<=1e-10
                prediction32=x[cut:].astype(np.float32)@q32;assert np.isfinite(prediction32).all()
                sensitivity=float(np.linalg.norm(prediction32-prediction)/max(np.linalg.norm(prediction),1e-30))
                measured=F.metrics(prediction32.astype(np.float64),y[cut:],ym);control=F.metrics(np.broadcast_to(ym,y[cut:].shape),y[cut:],ym)
                assert abs(control['squared_error_over_validation_mean_only_error']-1)<=1e-12
                eligible=bool(measured['relative_L2_quantiles'][2]<=.25 and measured['relative_L2_quantiles'][3]<=.5
                    and measured['squared_error_over_validation_mean_only_error']<=.5 and sensitivity<=1e-5)
                result['banks'].append({'stack':'encoder' if bank<6 else 'decoder','bank':bank%6,'development_mean_only_positions':cut,
                    'validation_positions':6*positions,'validation_F32_map':measured,'validation_F64_map':F.metrics(prediction,y[cut:],ym),
                    'validation_mean_only':control,'validation_identity':F.metrics(x[cut:],y[cut:],ym),
                    'F64_factor_vs_full_map_relative_L2':factor_error,'F32_vs_F64_map_relative_L2':sensitivity,'input_eligibility':eligible})
                print(json.dumps({'bank':bank,'input_eligibility':eligible,'validation':measured}),flush=True);guard()
        result['gates']={'fresh_full415_source_covariance_archive_and96_trace_hashes_keys_split_exact':True,'same_NumPy_BLAS_DLL_and_one_thread':True,
            'ONE_fixed767_identifiable_partial_map_bound_not_relaxed':True,'known_partial_map_wrong_anchor_nonisometry_controls':True,
            'F64_F32_projection_idempotence_trace_and_rank_optimal_certificate':True,'map_archive_readback_exact':True,
            'all12_F64_factor_full_map_numeric_bridges':True,'all12_mean_only_controls_exact_and_predictions_finite':True}
        result['input_eligibility_ALL12']=all(v['input_eligibility'] for v in result['banks'])
        result['decision']='eligible_only_for_new_actual_function_response_output_alignment_protocol' if result['input_eligibility_ALL12'] else 'reject_this_rank767_partial_map_and_close_lexical_weight_bridge_branch_before384_selector_model'
        result['resource']={'main_seconds':time.monotonic()-start,'maximum_checked_rss_bytes':peak,'output_bytes':file.stat().st_size}
        assert result['resource']['output_bytes']<=32<<20;guard()
        result['scope']='ONE fixed rank767 source-lexical partial isometry, unsupported direction explicitly discarded, no768 full rotation retry/complement/centering/anchor/rank/hyperparameter sweep. Weights-only415 covariance SHA freshly verified, source original weight hashes inherited415 and not freshly reread. Six consumed validation books/all4 cases at12 banks, development18 only mean-only comparison, no document/activation fit. Same .25/.50/.50 and1e-5 local gates, source1e-6 included-space conditioning unchanged. New loss of one source direction cannot inherit model quality. No source function/output map/384 selector/artifact/usefulness/accepted-rate/DRAM/10x/~100B/additional-family or new network/native/model inference/new quality data/GPU/T4 claim. Closed old interfaces remain immutable.'
        M.write(args.out,result);print(json.dumps({'sha256':sha(args.out),'decision':result['decision'],'input_eligibility_ALL12':result['input_eligibility_ALL12'],'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'failure_stage':stage,'error':repr(error),'seconds':time.monotonic()-start,'maximum_checked_rss_bytes':peak})
        M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
