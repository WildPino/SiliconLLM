"""Diagnostic spectra after414 precondition failure; NO new alignment map."""
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

PROTOCOL=M.DOC/'METH_415_SWITCH_LEXICAL_SPECTRA_PROTOCOL_20261004.md'
PREVIOUS=M.DOC/'meth414_switch_lexical_orthogonal_map_result.failure.json'
PREVIOUS_SHA='711dff37952336b5d553f607e9086a0992efae6cb831caf40ef274de966e3e07'
OUT=M.ROOT/'results/native_expert_scaling/meth415_switch_lexical_spectra'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();peak=0;hashed=0;stage='bindings';analysis_start=None
    result={'experiment':'METH-415-original-lexical-source-and-cross-spectral-inventory','sources':[]}

    def guard():
        nonlocal peak
        peak=max(peak,psutil.Process().memory_info().rss)
        assert peak<=2<<30 and time.monotonic()-start<=600,'spectra_10min_2GiB'
        if analysis_start is not None:assert time.monotonic()-analysis_start<=120,'spectra_analysis_2min'

    def sha(path):
        nonlocal hashed
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block:=stream.read(4<<20):h.update(block);hashed+=len(block);guard()
        return h.hexdigest()

    try:
        result['bindings']={}
        for path in (Path(__file__),PROTOCOL,PREVIOUS,Path(M.__file__),Path(B.__file__),Path(U.__file__),
                     M.ROOT/'benchmarks/native_expert_scaling/meth414_switch_lexical_orthogonal_map.py',
                     M.DOC/'METH_414_SWITCH_LEXICAL_ORTHOGONAL_MAP_PROTOCOL_20261004.md'):
            M.committed(path);result['bindings'][str(path)]=sha(path)
        assert sha(PREVIOUS)==PREVIOUS_SHA;previous=json.loads(PREVIOUS.read_text(encoding='utf-8'))
        assert previous['error']=="AssertionError('full_lexical_cross_covariance_identifiable')" and previous['banks']==[]
        result['frozen_HEAD']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        ancestors={p.pid for p in psutil.Process().parents()}|{os.getpid()}
        for process in psutil.process_iter(['pid','name','cmdline']):
            if process.pid in ancestors:continue
            name=(process.info['name'] or '').lower();argv=process.info['cmdline'] or []
            unrelated=(name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve())
            if unrelated:result.setdefault('unrelated_background_processes',[]).append({'pid':process.pid,'argv':argv});continue
            assert not (name.startswith('python') or ('meth' in name and name.endswith('.exe'))),('concurrent_model_job',process.pid,name)
        anchors={};initial={};grams={};spectra={};statistics={};archives={}
        for n,(name,expected) in U.EXPORT.items():
            stage=f'source{n}_original_embeddings';path=M.DOC/name;M.committed(path);assert sha(path)==expected
            export=json.loads(path.read_text(encoding='utf-8'));assert all(export['gates'].values())
            a=export['artifact'];assert a==previous['source_weight_bindings'][str(n)]['artifact'];payload=Path(a['payload'])
            initial[n]=(payload.stat().st_size,payload.stat().st_mtime_ns)
            assert initial[n][0]==a['bytes'] and sha(payload)==a['sha256'] and sha(a['manifest'])==a['manifest_sha256']
            B.read_manifest(a['manifest'],export['original_config'],export['tensors'],payload)
            row=export['tensors']['shared.weight'];assert row==previous['source_weight_bindings'][str(n)]['embedding_tensor']
            assert row['shape']==[32128,768] and row['encoding']==0 and row['bytes']==98697216 and row['sha256']==row['source_sha256']
            with payload.open('rb') as stream:stream.seek(row['offset']);data=stream.read(row['bytes'])
            assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
            anchor=np.frombuffer(data,dtype='<f4').reshape(32128,768).astype(np.float64);del data
            assert np.isfinite(anchor).all();norm=np.linalg.norm(anchor,axis=1);valid=norm>0
            assert valid.sum()==previous['source_weight_bindings'][str(n)]['nonzero_embedding_rows']==32128
            statistics[n]={'source_n':n,'finite_F32_positions':32128*768,'nonzero_rows':int(valid.sum()),
                'original_row_norm_quantiles':np.quantile(norm,[0,.05,.5,.95,1]).tolist(),
                'exact_zero_columns':np.where(np.all(anchor==0,axis=0))[0].tolist(),
                'artifact':a,'embedding_tensor':row}
            anchor[valid]/=norm[valid,None];anchors[n]=anchor
            print(json.dumps({'stage':stage,'seconds':time.monotonic()-start}),flush=True);guard()
        OUT.mkdir(parents=True);analysis_start=time.monotonic();stage='diagnostic_source_and_cross_spectra'
        with threadpool_limits(limits=1):
            info=threadpool_info();assert all(v['num_threads']==1 for v in info)
            assert [(v['version'],v['architecture']) for v in info]==[(v['version'],v['architecture']) for v in previous['runtime']['BLAS']]
            result['runtime']={'numpy':np.__version__,'BLAS':info,'BLAS_DLL_sha256':[sha(v['filepath']) for v in info]}
            assert np.__version__==previous['runtime']['numpy']
            known=np.diag([4.,2.,1.,0.]);assert np.array_equal(np.linalg.svd(known,compute_uv=False),[4.,2.,1.,0.])
            bad=known.copy();bad[3,3]=.5;assert not np.array_equal(np.linalg.svd(bad,compute_uv=False),[4.,2.,1.,0.]),'rank_fault_control'
            for n in (256,128):
                gram=anchors[n].T@anchors[n];values,vectors=np.linalg.eigh(gram);assert values[-1]>0 and values[0]>=-1e-10*values[-1]
                reconstruction=float(np.linalg.norm((vectors*values)@vectors.T-gram)/np.linalg.norm(gram));assert reconstruction<=1e-10
                relative=np.sqrt(np.maximum(values[::-1],0)/values[-1]);rank=int(np.count_nonzero(relative>=1e-6))
                statistics[n].update({'normalized_source_singular_ratios_from_Gram':relative.tolist(),'source_rank_at1e_6':rank,
                    'source_Gram_reconstruction_relative_L2':reconstruction,'Gram_sqrt_roundoff_boundary':'F64 Gram squares singular ratios; <=about1e-8 source ratios not resolved by this method'})
                result['sources'].append(statistics[n]);grams[n]=gram;archives[f'source{n}_Gram']=gram;archives[f'source{n}_ratios']=relative;guard()
            c=anchors[256].T@anchors[128];u,s,vt=np.linalg.svd(c,full_matrices=False);assert s[0]>0 and np.isfinite(s).all()
            reverse=np.linalg.svd(c.T,compute_uv=False);blocked=np.zeros_like(c)
            for begin in range(0,32128,1024):blocked+=anchors[256][begin:begin+1024].T@anchors[128][begin:begin+1024];guard()
            sb=np.linalg.svd(blocked,compute_uv=False)
            reconstruction=float(np.linalg.norm((u*s)@vt-c)/np.linalg.norm(c))
            blocked_error=float(np.linalg.norm(blocked-c)/np.linalg.norm(c));reverse_error=float(np.max(np.abs(reverse-s))/s[0]);spectrum_error=float(np.max(np.abs(sb-s))/s[0])
            assert max(reconstruction,blocked_error,reverse_error,spectrum_error)<=1e-10
            relative=s/s[0];rank=int(np.count_nonzero(relative>=1e-6));assert relative[-1]<1e-6,'414_full_identifiability_failure_reproduced'
            result['cross']={'all768_singular_values':s.tolist(),'all768_singular_ratios':relative.tolist(),'rank_at414_bound1e_6':rank,
                'minimum_over_maximum':float(relative[-1]),'maximum_singular':float(s[0]),'minimum_singular':float(s[-1]),
                'reconstruction_relative_L2':reconstruction,'independent1024row_blocked_covariance_relative_L2':blocked_error,
                'reverse_spectrum_max_error_over_largest':reverse_error,'blocked_spectrum_max_error_over_largest':spectrum_error,
                '414_minimum_ratio_gate_reproduced_FAIL':True}
            archives.update({'cross_covariance':c,'cross_singular':s,'cross_ratios':relative})
            file=OUT/'lexical_source_and_cross_spectra.npz';np.savez(file,**archives)
            with np.load(file) as saved:
                assert set(saved.files)==set(archives)
                for key,value in archives.items():assert np.array_equal(saved[key],value)
            result['archive']={'path':str(file),'sha256':sha(file),'bytes':file.stat().st_size,'keys':list(archives)}
        for n in (256,128):
            payload=Path(statistics[n]['artifact']['payload']);assert initial[n]==(payload.stat().st_size,payload.stat().st_mtime_ns)
        result['decision']='source_lexical_rank_limits_full_weight_anchor_basis' if any(v['source_rank_at1e_6']<768 for v in result['sources']) else 'cross_source_lexical_basis_not_identifiable_despite_individual_full_source_rank'
        result['gates']={'both_complete_payload_manifest_embedding_source_identity_fresh_exact':True,'all_nonzero_finite_unit_row_anchors_exact414':True,
            'same_NumPy_BLAS_one_thread_runtime':True,'tiny_known_spectrum_and_rank_fault_detected':True,
            'both_source_Gram_eigen_reconstructions_pass':True,'cross_SVD_reconstruction_and_blocked_reverse_spectrum_controls_pass':True,
            '414_full_lexical_identifiability_failure_reproduced':True,'all_full_spectra_archive_readback_exact':True,'no_new_map_inference_or_quality_data':True}
        result['resource']={'main_seconds':time.monotonic()-start,'analysis_seconds':time.monotonic()-analysis_start,'maximum_checked_rss_bytes':peak,
            'fresh_hashed_file_bytes':hashed,'actual_embedding_segment_read_bytes':2*98697216,'output_bytes':result['archive']['bytes']}
        assert result['resource']['output_bytes']<=32<<20;guard()
        result['scope']='Diagnostic original-F32 lexical spectra after retained414 rank precondition failure. ALL32128 nonzero unit rows/source, same weighting and NumPy/BLAS, source Gram and cross full spectra. Gram squared-spectrum roundoff limits tiny source-rank interpretation, cross bound independently reproduced by blocked/reversed controls. No orthogonal map Q fit/inferred/saved, no input/function/output bridge/384 selector/model/export/quality/rate/physical DRAM/10x/~100B/another-family claim. Existing414 failure remains authoritative and no rank/gate relaxation. No new model/native inference/network/new quality data/GPU/T4.'
        M.write(args.out,result);print(json.dumps({'sha256':sha(args.out),'decision':result['decision'],'source_ranks':[v['source_rank_at1e_6'] for v in result['sources']],
            'cross_rank':rank,'cross_min_ratio':result['cross']['minimum_over_maximum'],'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'failure_stage':stage,'error':repr(error),'seconds':time.monotonic()-start,'maximum_checked_rss_bytes':peak,'bytes_hashed':hashed})
        M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
