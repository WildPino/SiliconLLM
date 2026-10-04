"""ONE source-weight-anchored orthogonal input bridge; not a384 model."""
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
import meth368_switch_bank_manifest as B
import meth401_pretrained_bank_union_applicability as U
import meth405_switch_full_basis_inputs as P
import meth404_switch_rank32_input_map as F

PROTOCOL = M.DOC/'METH_414_SWITCH_LEXICAL_ORTHOGONAL_MAP_PROTOCOL_20261004.md'
OUT = M.ROOT/'results/native_expert_scaling/meth414_switch_lexical_orthogonal_map'
INPUTS = {
    401: ('meth401_pretrained_bank_union_applicability_result.json','d35bf7f82b064593ff8568379fc8106b100d8337c1b860395e1bc6bd83d4e160'),
    405: ('meth405_switch_full_basis_inputs_result.json','1fb5c222807da2ad8f8e52e8f3d2d5fc64664431edc2332070305fa46cb64788'),
    406: ('meth406_switch_full_ridge_input_map_result.json','f1220d8c7eaa4fcc0d0749952c9eeec49cfebb937f679a4969f3599110cd6fd0'),
    326: ('meth326_switch_acquisition_result.json','39bac2bda18cc6660da77bba8256d1e21fcad5b6fa70796d343b7e139d480711'),
    378: ('meth378_switch_base128_acquisition_result.json','89add8d24541423dfa61827f785bf125c2062cf1aa6976214f03da94b7c582b7'),
}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();peak=0;hashed=0;stage='bindings';fit_start=None
    result={'experiment':'METH-414-one-global-lexical-weight-orthogonal-input-bridge','banks':[]}

    def guard():
        nonlocal peak
        peak=max(peak,psutil.Process().memory_info().rss)
        assert peak<=2<<30 and time.monotonic()-start<=600,'lexical_main_10min_2GiB'
        if fit_start is not None:assert time.monotonic()-fit_start<=120,'lexical_fit_evaluation_2min'

    def sha(path):
        nonlocal hashed
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block:=stream.read(4<<20):h.update(block);hashed+=len(block);guard()
        return h.hexdigest()

    def orthogonal_error(q):return float(np.linalg.norm(q.T@q-np.eye(q.shape[0]))/np.sqrt(q.shape[0]))

    try:
        result['bindings']={}
        for path in (Path(__file__),PROTOCOL,Path(M.__file__),Path(B.__file__),Path(U.__file__),Path(P.__file__),Path(F.__file__)):
            M.committed(path);result['bindings'][str(path)]=sha(path)
        result['frozen_HEAD']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        records={}
        for number,(name,expected) in INPUTS.items():
            path=M.DOC/name;M.committed(path);assert sha(path)==expected
            records[number]=json.loads(path.read_text(encoding='utf-8'))
            if 'gates' in records[number]:assert all(records[number]['gates'].values())
        result['input_sha256']={str(k):v[1] for k,v in INPUTS.items()}
        assert not records[406]['input_eligibility_ALL12']
        ancestors={p.pid for p in psutil.Process().parents()}|{os.getpid()}
        for process in psutil.process_iter(['pid','name','cmdline']):
            if process.pid in ancestors:continue
            name=(process.info['name'] or '').lower();argv=process.info['cmdline'] or []
            unrelated=(name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve())
            if unrelated:result.setdefault('unrelated_background_processes',[]).append({'pid':process.pid,'argv':argv});continue
            assert not (name.startswith('python') or ('meth' in name and name.endswith('.exe'))),('concurrent_model_job',process.pid,name)
        tokenizers=[]
        for number in (326,378):
            side={v['name']:v for v in records[number]['files']};hashes={}
            for name in ('spiece.model','tokenizer.json','special_tokens_map.json'):
                hashes[name]=sha(side[name]['path']);assert hashes[name]==side[name]['sha256']
            tokenizers.append(hashes)
        assert tokenizers[0]==tokenizers[1];result['tokenizer_sha256']=tokenizers
        arrays={n:[[] for _ in range(12)] for n in (256,128)};keys=[]
        pairs=records[405]['pairs'];assert len(pairs)==96
        for index,pair in enumerate(pairs):
            assert (pair['book'],pair['case'])==divmod(index,4)
            assert pair['prospective_split']==('development' if pair['book']<18 else 'validation')
            assert P.pairing_key(pair['source_ids'],pair['decoder_ids'])==pair['pairing_sha256'];keys.append(pair['pairing_sha256'])
            for n,path,expected in ((256,pair['source256_trace_path'],pair['source256_trace_sha256']),
                                    (128,pair['source128_capture']['trace_path'],pair['source128_capture']['trace_sha256'])):
                assert sha(path)==expected
                for bank,values in enumerate(P.read_trace(path,n)):arrays[n][bank].append(values)
        assert len(set(keys))==96;result['paired_input_keys']=keys
        exports={};initial={};anchors={};valid_rows={};result['source_weight_bindings']={}
        for n,(name,expected) in U.EXPORT.items():
            stage=f'fresh_source_payload_{n}';path=M.DOC/name;M.committed(path);assert sha(path)==expected
            export=json.loads(path.read_text(encoding='utf-8'));assert all(export['gates'].values())
            a=export['artifact'];payload=Path(a['payload']);initial[n]=(payload.stat().st_size,payload.stat().st_mtime_ns)
            assert initial[n][0]==a['bytes'] and sha(payload)==a['sha256'] and sha(a['manifest'])==a['manifest_sha256']
            B.read_manifest(a['manifest'],export['original_config'],export['tensors'],payload)
            row=export['tensors']['shared.weight'];assert row['shape']==[32128,768] and row['encoding']==0 and row['bytes']==98697216
            assert row['source_sha256']==row['sha256'] and row['offset']+row['bytes']<=a['bytes']
            for alias in ('encoder.embed_tokens.weight','decoder.embed_tokens.weight'):
                other=export['tensors'][alias];assert (other['shape'],other['encoding'],other['offset'],other['bytes'],other['sha256'])==(row['shape'],row['encoding'],row['offset'],row['bytes'],row['sha256'])
            with payload.open('rb') as stream:stream.seek(row['offset']);data=stream.read(row['bytes'])
            assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
            anchor=np.frombuffer(data,dtype='<f4').reshape(32128,768).astype(np.float64);del data
            assert np.isfinite(anchor).all();norm=np.linalg.norm(anchor,axis=1);valid=norm>0
            anchor[valid]/=norm[valid,None];anchors[n]=anchor;valid_rows[n]=valid
            exports[n]=export;result['source_weight_bindings'][str(n)]={'artifact':a,'source_export_sha256':expected,
                'embedding_tensor':row,'finite_F32_positions':32128*768,'nonzero_embedding_rows':int(valid.sum())}
            print(json.dumps({'stage':stage,'seconds':time.monotonic()-start}),flush=True);guard()
        OUT.mkdir(parents=True);stage='ONE_weight_orthogonal_fit';fit_start=time.monotonic()
        with threadpool_limits(limits=1):
            result['runtime']={'numpy':np.__version__,'BLAS':threadpool_info(),'arithmetic':'F64 unit-row lexical covariance/SVD; F32 no-intercept input map'}
            assert all(v['num_threads']==1 for v in threadpool_info())
            # Tiny independent known signed-permutation and mismatched-anchor controls.
            tiny=np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,2,3,4],[4,-1,2,3],[2,3,-4,1]],dtype=np.float64)
            tiny/=np.linalg.norm(tiny,axis=1)[:,None]
            truth=np.asarray([[0,0,-1,0],[1,0,0,0],[0,0,0,1],[0,-1,0,0]],dtype=np.float64);target=tiny@truth
            tu,ts,tv=np.linalg.svd(tiny.T@target,full_matrices=False);tq=tu@tv
            assert np.max(np.abs(tq-truth))<=1e-12 and orthogonal_error(tq)<=1e-12
            wu,ws,wv=np.linalg.svd(tiny.T@np.roll(target,1,axis=0),full_matrices=False)
            assert np.linalg.norm(tiny@(wu@wv)-target)/np.linalg.norm(target)>.1,'mismatched_token_anchors_detected'
            bad=tq.copy();bad[0,0]+=.125;assert orthogonal_error(bad)>.01,'nonorthogonal_fault_detected'
            valid=valid_rows[256]&valid_rows[128];assert valid.sum()>=768
            covariance=anchors[256][valid].T@anchors[128][valid];u,singular,vt=np.linalg.svd(covariance,full_matrices=False);q=u@vt;guard()
            assert np.isfinite(q).all() and len(singular)==768 and singular[0]>0
            spectrum=float(singular[-1]/singular[0]);assert spectrum>=1e-6,'full_lexical_cross_covariance_identifiable'
            reconstruction=float(np.linalg.norm((u*singular)@vt-covariance)/np.linalg.norm(covariance))
            orth64=orthogonal_error(q);score=float(np.sum(q*covariance));certificate=abs(score-float(singular.sum()))/max(abs(float(singular.sum())),1e-30)
            assert reconstruction<=1e-10 and orth64<=1e-10 and certificate<=1e-10
            q32=q.astype(np.float32);orth32=orthogonal_error(q32.astype(np.float64));assert orth32<=1e-6
            source_error=float(np.linalg.norm(anchors[256][valid]@q-anchors[128][valid])/np.linalg.norm(anchors[128][valid]))
            result['ONE_shared_lexical_map']={'direction':'row-input256 @ Q -> source128 coordinates','all_vocabulary_rows':32128,'nonzero_paired_anchor_rows':int(valid.sum()),
                'min_singular_over_max':spectrum,'covariance_reconstruction_relative_L2':reconstruction,'Procrustes_optimal_trace_certificate_relative_error':certificate,
                'F64_orthogonality_normalized_Frobenius':orth64,'F32_orthogonality_normalized_Frobenius':orth32,
                'embedding_unit_row_relative_L2':source_error,'no_activation_or_document_fit':True,'active_coefficients_per_bank':768*768,
                'same_shared_map_active_coefficients_ALL12':12*768*768,'shared_F32_map_parameter_bytes':768*768*4}
            map_path=OUT/'ONE_global_lexical_orthogonal_map.npz';np.savez(map_path,matrix=q32,matrix64=q,singular=singular,cross_covariance=covariance)
            with np.load(map_path) as restored:assert np.array_equal(restored['matrix'],q32) and np.array_equal(restored['matrix64'],q)
            result['ONE_shared_lexical_map'].update({'map_path':str(map_path),'map_sha256':sha(map_path),'map_archive_bytes':map_path.stat().st_size})
            for bank in range(12):
                stage=f'input_validation_bank_{bank}';positions=4*(29 if bank<6 else 14);cut=18*positions
                x=np.concatenate(arrays[256][bank]).astype(np.float64);y=np.concatenate(arrays[128][bank]).astype(np.float64);ym=y[:cut].mean(axis=0)
                prediction=x[cut:]@q;prediction32=x[cut:].astype(np.float32)@q32
                assert np.isfinite(prediction32).all()
                sensitivity=float(np.linalg.norm(prediction32-prediction)/max(np.linalg.norm(prediction),1e-30))
                measured=F.metrics(prediction32.astype(np.float64),y[cut:],ym);mean_only=F.metrics(np.broadcast_to(ym,y[cut:].shape),y[cut:],ym)
                assert abs(mean_only['squared_error_over_validation_mean_only_error']-1)<=1e-12
                eligible=bool(measured['relative_L2_quantiles'][2]<=.25 and measured['relative_L2_quantiles'][3]<=.5
                    and measured['squared_error_over_validation_mean_only_error']<=.5 and sensitivity<=1e-5)
                result['banks'].append({'stack':'encoder' if bank<6 else 'decoder','bank':bank%6,'development_mean_only_positions':cut,
                    'validation_positions':6*positions,'same_global_weight_only_map':True,'validation_F32_map':measured,
                    'validation_F64_map':F.metrics(prediction,y[cut:],ym),'validation_mean_only':mean_only,'validation_identity':F.metrics(x[cut:],y[cut:],ym),
                    'F32_vs_F64_prediction_relative_L2':sensitivity,'input_eligibility':eligible})
                print(json.dumps({'bank':bank,'input_eligibility':eligible,'validation':measured}),flush=True);guard()
        for n,export in exports.items():
            payload=Path(export['artifact']['payload']);assert initial[n]==(payload.stat().st_size,payload.stat().st_mtime_ns)
        result['gates']={'both_complete_qualified_payload_hashes_and_manifest_records_fresh_exact':True,'source_F32_embedding_segments_finite_exact_original_aliases':True,
            'same_tokenizer_vocabulary_control_bytes_exact':True,'all96_paired_trace_hashes_keys_and18_6_split_exact':True,
            'ONE_weight_only_shared_no_intercept_orthogonal_map':True,'full_lexical_covariance_identifiable_and_optimal_SVD_certificate':True,
            'tiny_known_rotation_wrong_anchor_and_nonorthogonal_controls':True,'F32_map_readback_orthogonality_all_predictions_finite':True,
            'validation_mean_only_control_exact':True,'BLAS_one_thread':True}
        result['input_eligibility_ALL12']=all(v['input_eligibility'] for v in result['banks'])
        result['decision']='eligible_only_for_new_actual_function_response_and_output_alignment_protocol' if result['input_eligibility_ALL12'] else 'reject_this_global_lexical_weight_orthogonal_interface_before384_selector_or_model'
        result['resource']={'main_seconds':time.monotonic()-start,'fit_and_evaluation_seconds':time.monotonic()-fit_start,
            'maximum_checked_rss_bytes':peak,'fresh_hashed_file_bytes':hashed,'actual_source_embedding_read_bytes':2*98697216,'output_bytes':sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())}
        assert result['resource']['output_bytes']<=32<<20;guard()
        result['scope']='ONE global768 orthogonal basis from ALL common nonzero unit-normalized32128 source embedding weight rows ONLY, no per-bank activation regression/intercept/hyperparameter search/new document fitting. Same-token rows freshly exact F32 originals in existing complete qualified I8 exports. Six consumed validation books/all4 cases from405 evaluated at all12 banks; development18 used ONLY mean-only comparison, not map fit. Same .25/.50/.50 and1e-5 local gates as404/406. Weight/token ID alignment is not hidden/function/output alignment, useful384 selector/model/quality/rate/physical DRAM/10x/~100B/another-family proof. No native/model inference/export mutation/new network/new quality data/GPU/T4. Prior closed identity/PCR/full-ridge inputs remain closed; no broad nonlinear/other-weight method impossibility.'
        M.write(args.out,result);print(json.dumps({'sha256':sha(args.out),'decision':result['decision'],'input_eligibility_ALL12':result['input_eligibility_ALL12'],'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'failure_stage':stage,'error':repr(error),'seconds':time.monotonic()-start,'maximum_checked_rss_bytes':peak,'bytes_hashed':hashed})
        M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
