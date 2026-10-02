#!/usr/bin/env python3
"""Fixed128 output-aware private rows; even-fit/odd-reserve then native cost."""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import time
import numpy as np
import torch
from torch.nn import functional as TF
from safetensors import safe_open
from huggingface_hub import hf_hub_download
import meth269_ffn_bf16_node_assay as A

P,L,R,Q,G=A.P,A.L,A.R,A.Q,A.G
ASSAY=P.DOC/'meth269_ffn_bf16_node_assay_result.json'
ASSAY_SHA='95f905587e74de27b8d7661f9b4a0160af824cdc331c9cf34c8aef3df5b5cc1f'
COVERAGE=P.DOC/'meth270_private_source_rows_result.json'
COVERAGE_SHA='b14fdd1b7c8867ff01ee36600c8d8903ab74f1abeb60cd4051dfe9b0e93308bd'
PRIOR=P.DOC/'meth252_private_feature_native_result.json'
PRIOR_SHA='7da69e9d44df64719fddbfb62a452a2b4afb3be7044511f1749d16305b6910ce'
NATIVE=P.DOC/'meth253_split_private_feature_native_result.json'
NATIVE_SHA='9262f98cd87021d32697bb79badc1f07d7480a558def3cce6c9e48fb7466611a'
FIXTURE=P.ROOT/'results/native_expert_scaling/meth252_private_feature_fixture.bin'
PRIVATE,BYTES=128,337596452
BASE_MEAN=.0001972897928984215
BASE_MAX=.00030542892636731267
COMPILE_FLAGS=['-O3','-mavx2','-mssse3','-mfma','-fopenmp','-std=c11','-Wall','-Wextra']


def raw_tensor(t):
    return (t.view(torch.uint16) if t.dtype==torch.bfloat16 else t).cpu().contiguous().numpy().tobytes()


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=True)+'\n',encoding='utf-8')


class SelectionStop(Exception):
    pass


def subset_measure(reference,value,indices):
    reference=reference[indices];value=value[indices]
    assert reference.shape==value.shape==(128,896) and torch.isfinite(value).all()
    error=(reference.float()-value.float()).square().sum(1)
    energy=reference.float().square().sum(1);assert (energy>0).all()
    return {'squared_error':float(error.sum()),'reference_energy':float(energy.sum()),
        'energy_normalized_squared_error':float(error.sum()/energy.sum()),
        'mean_state_relative_squared_error':float((error/energy).mean())}


def output_aware_score(x,reference,old_value,phi,v,wd,fit):
    # Decode the fixed linear readout; source/operator parameters never change.
    down=v['down.q'].float()*v['down.scale'][:,None]
    index=v['down.ids'].long()
    assert torch.count_nonzero(down.gather(1,index))==0
    down.scatter_add_(1,index,v['down.escape'].float())
    down+=v['residual.left'].float() @ v['residual.right'].float()
    decoded=((TF.linear(phi,down)+v['bias'])+v['residual.bias'])
    error=torch.linalg.vector_norm((decoded-old_value).double(),dim=1)
    energy=torch.linalg.vector_norm(old_value.double(),dim=1)
    assert (energy>0).all() and float((error/energy).max())<=1e-5
    private=G.L.silu_lookup(TF.linear(x.float(),wd['gate'].float()))*TF.linear(x.float(),wd['up'].float())
    delta=(private[fit]-phi[fit]).double()
    residual=(reference[fit].float()-old_value[fit]).double()
    normalizer=reference[fit].float().double().square().sum(1)
    matrix=down.double()
    projected=TF.linear(residual,matrix.T.contiguous())
    score=((2*delta*projected-delta.square()*matrix.square().sum(0)[None,:])/normalizer[:,None]).mean(0)
    assert score.shape==(4864,) and torch.isfinite(score).all()
    return score,float((error/energy).max())


def main():
    ap=argparse.ArgumentParser()
    for name in ('binary','exe','check','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    assert all(not p.exists() for p in (args.binary,args.exe,args.check,args.out))
    assert not args.out.with_suffix('.failure.json').exists()
    args.binary.parent.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(args.binary.parent).free>=3*1024**3
    start=time.monotonic();stage='bindings';segments=[];selection=[];rows=[];native_rows=[]
    gates={};result={};device=None
    def partial():
        dump(args.out.with_suffix('.partial.json'),{'stage':stage,'selection':selection,
            'layers':rows,'segments':segments,'gates':gates,'seconds':time.monotonic()-start})
    try:
        for path,sha in ((COVERAGE,COVERAGE_SHA),(ASSAY,ASSAY_SHA),(PRIOR,PRIOR_SHA),(NATIVE,NATIVE_SHA),
                         (Q.D.CORE,Q.D.CORE_SHA),(Q.D.EXPORT,Q.D.EXPORT_SHA),
                         (G.C.VECTORS,G.C.VECTORS_SHA)):
            assert P.digest(path)==sha,(str(path),'hash mismatch')
        assay=json.loads(ASSAY.read_text(encoding='utf-8'))
        coverage=json.loads(COVERAGE.read_text(encoding='utf-8'))
        assert coverage['decision']=='stop_fixed128_coverage_at_BF16_fidelity_no_native_or_full_archive'
        assert P.digest(Path(__file__).with_name('meth270_private_source_rows.py'))==coverage['script_sha256']
        prior=json.loads(PRIOR.read_text(encoding='utf-8'))
        native=json.loads(NATIVE.read_text(encoding='utf-8'))
        export=json.loads(Q.D.EXPORT.read_text(encoding='utf-8'))
        assert all(assay['gates'].values()) and all(native['gates'].values())
        assert P.digest(Path(A.__file__))==assay['script_sha256']
        assert assay['summary']['candidate_existing_fp32']['mean_state_relative_squared_error']==BASE_MEAN
        assert assay['summary']['candidate_existing_fp32']['maximum_layer_energy_normalized_squared_error']==BASE_MAX
        assert P.digest(FIXTURE)==prior['binary']['sha256']==native['fixture_sha256']
        assert P.digest(Path(R.__file__))==export['script_sha256']
        for path,sha in export['helper_sha256'].items():assert P.digest(Path(path))==sha
        original_c=Path(__file__).with_name('meth253_split_private_feature_cpu.c')
        native_c=Path(__file__).with_name('meth271_output_aware_rows_cpu.c')
        include_c=native_c.with_name('meth182_group64_ffn_cpu.c')
        assert P.digest(original_c)==native['source_code_sha256']
        assert P.digest(include_c)==native['included_meth182_source_sha256']
        # Only constant, experiment labels and output header differ from253.
        old_c=original_c.read_text(encoding='utf-8')
        expected_c=old_c.replace('METH-253','METH-271').replace('#define PRIVATE_UNITS 32','#define PRIVATE_UNITS 128')
        expected_c=expected_c.replace('usage: meth253','usage: meth271').replace('M253OUT1','M271OUT1')
        expected_c=expected_c.replace('private_units\\\":32','private_units\\\":128')
        assert native_c.read_text(encoding='utf-8')==expected_c
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA
        vectors=G.C.VECTORS.read_bytes()
        assert struct.unpack_from('<8s4I',vectors)==(b'M125HX01',24,256,896,1280)
        states=np.frombuffer(vectors,dtype='<u2',offset=24).reshape(256,24,896)
        device=G.Q.M.D.Q.setup();P.MAX_SECONDS=P.M17.MAX_SECONDS=10*60
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        fit=torch.arange(0,256,2,device=device);reserve=torch.arange(1,256,2,device=device)
        oracles=[];source_hashes=[]
        stage='even_fit_output_aware_selection_BF16_scoring_and_fixture_export'
        with torch.inference_mode(),safe_open(str(source),framework='pt',device='cpu') as donor,\
             safe_open(str(Q.D.CORE),framework='pt',device='cpu') as saved,\
             FIXTURE.open('rb') as original,args.binary.open('xb') as output:
            assert original.read(32)==struct.pack('<8s6I',b'M252PF01',24,896,4864,32,32,32)
            output.write(struct.pack('<8s6I',b'M252PF01',24,896,4864,32,32,PRIVATE))
            def write(name,li,raw):
                segments.append({'layer':li,'name':name,'offset':output.tell(),
                    'bytes':len(raw),'sha256':P.M17.sha(raw)})
                output.write(raw)
            table=prior['segments'][0];assert table['name']=='silu_table'
            original.seek(table['offset']);table_raw=original.read(table['bytes'])
            assert P.M17.sha(table_raw)==table['sha256']==P.M17.sha(raw_tensor(saved.get_tensor('ffn.silu_table')))
            write('silu_table',None,table_raw);G.L.TABLE=saved.get_tensor('ffn.silu_table').to(device)
            for li in range(24):
                v={}
                for segment in prior['segments']:
                    if segment['layer']!=li:continue
                    name=segment['name'];original.seek(segment['offset']);raw=original.read(segment['bytes'])
                    t=saved.get_tensor(f'ffn.{li}.{name}')
                    assert P.M17.sha(raw)==segment['sha256'] and raw_tensor(t)==raw
                    v[name]=t
                    if not name.startswith('private.'):write(name,li,raw)
                assert len(v)==15
                w={name:donor.get_tensor(f'model.layers.{li}.mlp.{name}_proj.weight') for name in ('gate','up','down')}
                assert all(t.dtype==torch.bfloat16 for t in w.values())
                hashes={f'model.layers.{li}.mlp.{name}_proj.weight':P.M17.sha(raw_tensor(t)) for name,t in w.items()}
                assert hashes==prior['source_tensor_hashes'][li]['sha256']
                source_hashes.append({'layer':li,'sha256':hashes})
                wd={name:t.to(device) for name,t in w.items()};vd={name:t.to(device) for name,t in v.items()}
                score=torch.zeros(4864,dtype=torch.float64,device=device)
                for name in ('gate','up'):
                    decoded=vd[name+'.q'].float()*vd[name+'.scale'][:,None]
                    score+=(wd[name].float().double()-decoded.double()).square().sum(1)
                order=torch.argsort(score,descending=True,stable=True)
                old_ids=order[:32].sort().values.cpu()
                assert torch.equal(old_ids.to(torch.uint16),v['private.ids'])
                for name in ('gate','up'):assert torch.equal(w[name][old_ids],v['private.'+name])
                x=torch.from_numpy(states[:,li].copy()).view(torch.bfloat16).to(device)
                reference=TF.linear(TF.silu(TF.linear(x,wd['gate']))*TF.linear(x,wd['up']),wd['down'])
                phi=G.private_features(x.float(),vd);old_fp32=G.readout(phi,vd)
                old_value=old_fp32.bfloat16();baseline=A.measure(reference,old_value)
                assert baseline==assay['layers'][li]['methods']['candidate_existing_fp32']
                scores,readout_error=output_aware_score(x,reference,old_fp32,phi,vd,wd,fit)
                candidates=scores.clone();candidates[old_ids.to(device)]=-torch.inf
                positive=int((candidates>0).sum())
                if positive<96:
                    selection.append({'layer':li,'positive_additional_units':positive,'scores':scores.cpu().tolist()})
                    raise SelectionStop(f'layer{li}: only{positive} positive additional units')
                chosen=torch.argsort(candidates,descending=True,stable=True)[:96].cpu()
                ids=torch.cat((old_ids,chosen)).sort().values
                assert set(old_ids.tolist()).issubset(ids.tolist()) and len(set(ids.tolist()))==PRIVATE
                selection.append({'layer':li,'ids':ids.tolist(),'old32_ids':old_ids.tolist(),
                    'additional_selected_ids':chosen.tolist(),'positive_additional_units':positive,
                    'all_unit_output_scores':scores.cpu().tolist(),
                    'sum_individual_predicted_reduction':float(scores[chosen.to(device)].sum()),
                    'decoded_readout_max_relative_l2':readout_error,'old32_retained':True,
                    'additional_rows':96,'source_row_identity':True})
                v['private.ids']=ids.to(torch.uint16)
                for name in ('gate','up'):v['private.'+name]=w[name][ids].contiguous()
                for name in ('private.ids','private.gate','private.up'):write(name,li,raw_tensor(v[name]))
                vd={name:t.to(device) for name,t in v.items()}
                value=G.readout(G.private_features(x.float(),vd),vd)
                measure=A.measure(reference,value.bfloat16())
                assert torch.isfinite(value).all()
                oracles.append(value[:16].cpu().numpy().astype(np.float64))
                rows.append({'layer':li,'states':256,'baseline_exact269':True,
                    'baseline':baseline,'private128':measure,
                    'even_fit_baseline':subset_measure(reference,old_value,fit),
                    'odd_reserve_baseline':subset_measure(reference,old_value,reserve),
                    'even_fit_private128':subset_measure(reference,value.bfloat16(),fit),
                    'odd_reserve_private128':subset_measure(reference,value.bfloat16(),reserve)})
                partial();print(json.dumps({'layer':li,'completed_layers':len(rows)}),flush=True)
                del w,wd,v,vd,x,reference,old_value,value,score,decoded,t,phi,old_fp32,scores,candidates
                P.budget(start,device)
        assert len(segments)==361 and args.binary.stat().st_size==BYTES
        old_segments={(r['layer'],r['name']):r for r in prior['segments']}
        unchanged=[r for r in segments if not r['name'].startswith('private.')]
        assert len(unchanged)==289 and all((r['bytes'],r['sha256'])==
            (old_segments[(r['layer'],r['name'])]['bytes'],old_segments[(r['layer'],r['name'])]['sha256']) for r in unchanged)
        with args.binary.open('rb') as check:
            assert check.read(32)==struct.pack('<8s6I',b'M252PF01',24,896,4864,32,32,PRIVATE)
            for s in segments:assert check.tell()==s['offset'] and P.M17.sha(check.read(s['bytes']))==s['sha256']
            assert not check.read(1)
        cells=[r['private128'] for r in rows]
        mean=float(np.mean([c['mean_state_relative_squared_error'] for c in cells]))
        maximum=max(c['energy_normalized_squared_error'] for c in cells)
        reserve_mean=float(np.mean([r['odd_reserve_private128']['mean_state_relative_squared_error'] for r in rows]))
        reserve_baseline=float(np.mean([r['odd_reserve_baseline']['mean_state_relative_squared_error'] for r in rows]))
        reserve_max=max(r['odd_reserve_private128']['energy_normalized_squared_error'] for r in rows)
        reserve_max_baseline=max(r['odd_reserve_baseline']['energy_normalized_squared_error'] for r in rows)
        gates={'all_input_and_frozen_helper_bindings':True,'all289_nonprivate_segments_byte_exact':True,
            'all361_segments_readback_exact':True,'all_source_rows_exact_old32_subset':True,
            'all6144_baselines_exact269_and_outputs_finite':True,
            'mean_state_error_at_most_85percent_baseline':mean<=.85*BASE_MEAN,
            'maximum_layer_energy_error_no_worse':maximum<=BASE_MAX,
            'odd_reserve_mean_at_most85percent_baseline':reserve_mean<=.85*reserve_baseline,
            'odd_reserve_maximum_layer_energy_no_worse':reserve_max<=reserve_max_baseline}
        result={'experiment':'METH-271-fixed128-output-aware-source-rows',
            'coverage_stop_sha256':COVERAGE_SHA,'assay_sha256':ASSAY_SHA,'prior_sha256':PRIOR_SHA,'native_prior_sha256':NATIVE_SHA,
            'artifact_sha256':Q.D.CORE_SHA,'export_sha256':Q.D.EXPORT_SHA,
            'source_sha256':P.M57.MODEL_SHA,'source_revision':P.M42.REV,
            'vectors_sha256':G.C.VECTORS_SHA,'helper_sha256':export['helper_sha256'],
            'script_sha256':P.digest(Path(__file__)),'assay_script_sha256':P.digest(Path(A.__file__)),
            'native_source_sha256':P.digest(native_c),'included_meth182_source_sha256':P.digest(include_c),
            'source_tensor_hashes':source_hashes,'segments':segments,'selection':selection,'layers':rows,
            'binary':{'path':str(args.binary),'bytes':BYTES,'sha256':P.digest(args.binary),
                'extra_bytes_vs252':BYTES-prior['binary']['bytes']},
            'summary':{'mean_state_relative_squared_error':mean,'mean_ratio_to269':mean/BASE_MEAN,
                'odd_reserve_mean_state_relative_squared_error':reserve_mean,
                'odd_reserve_mean_ratio_to_original32':reserve_mean/reserve_baseline,
                'odd_reserve_maximum_layer_energy_error':reserve_max,
                'odd_reserve_baseline_maximum_layer_energy_error':reserve_max_baseline,
                'energy_normalized_squared_error':sum(c['squared_error'] for c in cells)/sum(c['reference_energy'] for c in cells),
                'maximum_layer_energy_normalized_squared_error':maximum},
            'gates':gates,'selection_scoring_export_runtime':P.budget(start,device),
            'scope':'Consumed6144 source states; Even-fit/odd-reserve consumed source precision, no new learned expert n. Component pass does not override267, establish full quality/rate/RAM scaling or family transfer.'}
        if not all(gates.values()):
            result['decision']='stop_fixed128_output_aware_at_fidelity_or_reserve_no_native_or_full_archive'
        else:
            stage='native_compile_and_same253_check_timing';partial();native_start=time.monotonic()
            # All GPU work finishes before launching the six-thread CPU component.
            torch.cuda.synchronize(device)
            command=['clang',*COMPILE_FLAGS,str(native_c),'-o',str(args.exe),'-lm','-lpsapi']
            compile_run=subprocess.run(command,capture_output=True,text=True,check=True,timeout=60)
            run=subprocess.run([str(args.exe),str(args.binary),str(G.C.VECTORS),str(args.check)],
                capture_output=True,text=True,check=True,timeout=180)
            native_seconds=time.monotonic()-native_start;assert native_seconds<=5*60
            timing=json.loads(run.stdout)
            assert all(timing[k]==v for k,v in (('threads',6),('tokens',256),('layers',24),
                ('hidden',4864),('residual_rank',32),('private_units',PRIVATE),('weight_bytes',BYTES)))
            assert len(timing['pass_ms_per_token'])==3
            assert timing['peak_working_set_bytes']<=20*1024**3
            raw=args.check.read_bytes()
            assert len(raw)==20+16*24*896*4 and struct.unpack_from('<8s3I',raw)==(b'M271OUT1',16,24,896)
            actual=np.frombuffer(raw,dtype='<f4',offset=20).reshape(16,24,896)
            assert np.isfinite(actual).all()
            for li,expected in enumerate(oracles):
                errors=np.linalg.norm(actual[:,li].astype(np.float64)-expected,axis=1)/np.linalg.norm(expected,axis=1)
                assert np.isfinite(errors).all()
                native_rows.extend({'token':t,'layer':li,'relative_l2':float(e)} for t,e in enumerate(errors))
            errors=[r['relative_l2'] for r in native_rows]
            gates['native_numeric_original_limits']=statistics.median(errors)<=1e-4 and max(errors)<=5e-4
            gates['native_component_median_at_most10ms']=statistics.median(timing['pass_ms_per_token'])<=10
            result.update({'native_rows':native_rows,'timing':timing,'compile_command':command,
                'compiler_version':subprocess.run(['clang','--version'],capture_output=True,text=True,check=True,timeout=10).stdout,
                'compile_stderr':compile_run.stderr,'native_seconds':native_seconds,
                'executable_sha256':P.digest(args.exe),'native_output_sha256':P.digest(args.check)})
            result['summary'].update({'median_native_relative_l2':statistics.median(errors),
                'maximum_native_relative_l2':max(errors),'median_ms_per_token':statistics.median(timing['pass_ms_per_token'])})
            result['decision']='private128_output_aware_component_pass_requires_new_full_archive_and_quality' if all(gates.values()) else 'stop_fixed128_output_aware_at_native_numeric_or_cost'
        dump(args.out,result);print(json.dumps({k:result[k] for k in ('decision','summary','gates')}),flush=True)
    except SelectionStop as failure:
        partial();dump(args.out,{'experiment':'METH-271-fixed128-output-aware-source-rows',
            'decision':'stop_output_aware_selector_fewer96_positive_additional_units',
            'reason':str(failure),'coverage_stop_sha256':COVERAGE_SHA,'script_sha256':P.digest(Path(__file__)),
            'selection':selection,'layers':rows,'segments':segments,'runtime':P.budget(start,device),
            'partial_fixture_sha256':P.digest(args.binary),'scope':'Consumed component selector stop; no full archive/native promotion.'})
        print(json.dumps({'decision':'stop_output_aware_selector_fewer96_positive_additional_units','reason':str(failure)}),flush=True)
    except BaseException as failure:
        partial();dump(args.out.with_suffix('.failure.json'),{'stage':stage,
            'error':type(failure).__name__+': '+str(failure),'completed_layers':len(rows),
            'seconds':time.monotonic()-start,'result_before_failure':result})
        raise


if __name__=='__main__':main()
