#!/usr/bin/env python3
"""Preserve fitted parent center values in source-slope child priors."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
import torch
import meth241_source_precision_replay as D

C,P,M,R,Q=D.C,D.P,D.M,D.R,D.Q
DIAG=P.DOC/'meth241_source_precision_replay_result.json'
DIAG_SHA='33568b9dc99a1ba93d43346db58a97865815e35cb8c669ea8656d44dd70b1980'
VALUE_RESULT=P.DOC/'meth242_bf16_value_prior_result.json'
VALUE_SHA='a3bb1e94ff8decc40363b904c0b39a3d583d13d57a721e69fd1e6d913ea5ddfd'


def main():
    ap=argparse.ArgumentParser()
    for key in ('checkpoint','out'): ap.add_argument('--'+key,required=True,type=Path)
    args=ap.parse_args(); assert not args.checkpoint.exists() and not args.out.exists()
    args.checkpoint.parent.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(args.checkpoint.parent).free>=4*1024**3
    start,stage=time.monotonic(),'bindings'; audits=[]; rows=[]
    try:
        assert P.digest(D.PAIR)==D.PAIR_SHA and P.digest(DIAG)==DIAG_SHA
        pair=json.loads(D.PAIR.read_text()); diag=json.loads(DIAG.read_text())
        assert P.digest(VALUE_RESULT)==VALUE_SHA
        value_result=json.loads(VALUE_RESULT.read_text())
        assert value_result['decision']=='stop_source_value_only_precision_correction_as_count_solution'
        assert diag['decision']=='precision_matched_source_replay_qualifies_new_prior_method'
        assert P.digest(D.SNAPSHOT)==pair['checkpoint_sha256']
        assert P.digest(R.CAPTURE)==pair['capture_sha256'] and P.digest(Q.NATIVE)==Q.NATIVE_SHA
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA
        device=M.D.Q.setup(); P.MAX_SECONDS=P.M17.MAX_SECONDS=10*60
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(True)
        native=json.loads(Q.NATIVE.read_text()); values,controls=Q.load_layer(native,device)
        assert controls==pair['controls']
        with safe_open(str(D.SNAPSHOT),framework='pt',device='cpu') as archive:
            tensors={name:archive.get_tensor(name) for name in archive.keys()}
        matrices=[]; source_hashes={}
        with safe_open(str(source),framework='pt',device='cpu') as archive:
            for organ in ('gate','up','down'):
                name=f'model.layers.{M.LAYER}.mlp.{organ}_proj.weight'; w=archive.get_tensor(name)
                assert w.dtype==torch.bfloat16
                source_hashes[name]=P.M17.sha(w.view(torch.uint16).numpy().tobytes())
                matrices.append(w.to(device))
        assert source_hashes==pair['source_tensor_sha256']
        parents=tensors['router.parents'].to(device); children=tensors['router.children'].to(device)
        with np.load(R.CAPTURE,allow_pickle=False) as archive:
            xf=torch.from_numpy(archive['x_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
            yf=torch.from_numpy(archive['y_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
        lp=R.assign_full(xf,parents); lc=torch.empty_like(lp)
        for parent in range(16):
            mask=lp==parent; lc[mask]=parent*10+R.assign_full(xf[mask],children[parent])
        assert torch.bincount(lp,minlength=16).tolist()==pair['counts16']
        assert torch.bincount(lc,minlength=160).tolist()==pair['counts160']
        route=json.loads(Q.S.ROUTE_RESULT.read_text()); assert P.digest(Q.S.ROUTE_RESULT)==Q.S.ROUTE_SHA
        for labels,key in ((lp,'parent_labels_sha256'),(lc,'leaf_labels_sha256')):
            assert P.M17.sha(labels.cpu().numpy().astype('<i4').tobytes())==route['fit'][key]
        # Copy only the two bias arrays; every other tensor is retained byte-for-byte.
        old_prior_bias=tensors['child_prior.bias']; old_fit_bias=tensors['e160.bias']
        tensors['child_prior.bias']=old_prior_bias.clone(); tensors['e160.bias']=old_fit_bias.clone()
        stage='fitted_parent_center_values_and_exact_centered_fit_intercept_transport'
        fit_sse={arm:0. for arm in ('parent_prior','e16','child_prior','e160')}
        energy_fit=float(yf.double().square().sum())
        for parent in range(16):
            mask=lp==parent; phi=Q.features(xf[mask],values); y=yf[mask]; local=lc[mask]
            for arm in ('parent_prior','e16'):
                fit_sse[arm]+=float((C.readout(phi,C.get(tensors,arm,parent,device))-y).double().square().sum())
            for child in range(10):
                leaf=parent*10+child; center=children[parent,child]
                prior=C.get(tensors,'child_prior',leaf,device)
                center_phi=Q.features(center,values)
                source_value=C.readout(center_phi,C.get(tensors,'e16',parent,device))
                bp=(source_value-Q.L.mixed_linear(center_phi,*prior[:4])).float()
                delta=bp.double()-old_prior_bias[leaf].to(device).double()
                n=pair['counts160'][leaf]; gamma=n/(n+C.TAU)
                bf=(old_fit_bias[leaf].to(device).double()+(1-gamma)*delta).float()
                tensors['child_prior.bias'][leaf].copy_(bp.cpu()); tensors['e160.bias'][leaf].copy_(bf.cpu())
                chosen=local==leaf; z=phi[chosen]; yc=y[chosen]
                prior_new=C.get(tensors,'child_prior',leaf,device); fit_new=C.get(tensors,'e160',leaf,device)
                stored_point=float((C.readout(center_phi,prior_new).double()-source_value.double()).norm()/source_value.double().norm().clamp_min(1e-12))
                # Independently replay the defining centered-fit mean/intercept equation.
                zd=z.double(); mean=zd.mean(0); wp=C.decode(prior_new).double(); wf=C.decode(fit_new).double()
                mr=(yc.double()-(zd@wp.T+bp.double())).mean(0)
                expected=wp@mean+bp.double()+gamma*mr
                mean_error=float(((wf@mean+bf.double())-expected).norm()/yc.double().mean(0).norm().clamp_min(1e-12))
                assert stored_point<=1e-5 and mean_error<=1e-5
                audits.append({'child':leaf,'states':n,'intercept_shrink':gamma,
                    'fitted_parent_center_relative_l2':stored_point,'centered_mean_relative_l2':mean_error,
                    'hierarchical_prior_bias_change_l2':float(delta.norm()),
                    'fitted_bias_change_l2':float((bf.double()-old_fit_bias[leaf].to(device).double()).norm())})
                for arm,coeff in (('child_prior',prior_new),('e160',fit_new)):
                    fit_sse[arm]+=float((C.readout(z,coeff)-yc).double().square().sum())
                P.budget(start,device)
            args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'audits':audits},indent=2)+'\n',encoding='utf-8')
        fit_nmse={arm:v/energy_fit for arm,v in fit_sse.items()}
        assert all(fit_nmse[arm]==pair['fit_normalized_sse'][arm] for arm in ('parent_prior','e16'))
        stage='physical_snapshot_readback_and_unchanged_weight_audit'
        save_file(tensors,str(args.checkpoint),metadata={'experiment':'METH-243','source_sha256':P.M57.MODEL_SHA,
            'parent_snapshot_sha256':pair['checkpoint_sha256'],'prior_samples':str(C.TAU),
            'change':'fitted-parent child center values; fixed FP32 source sensitivities'})
        assert args.checkpoint.stat().st_size<1700000000
        with safe_open(str(args.checkpoint),framework='pt',device='cpu') as archive, \
             safe_open(str(D.SNAPSHOT),framework='pt',device='cpu') as original:
            assert set(archive.keys())==set(tensors)==set(original.keys())
            unchanged=[]
            for name,t in tensors.items():
                assert torch.equal(archive.get_tensor(name),t)
                if name not in ('child_prior.bias','e160.bias'):
                    assert torch.equal(t,original.get_tensor(name)); unchanged.append(name)
        hashes={}
        for arm,count in (('e16',16),('e160',160)):
            hashes[arm]=[P.M17.sha(C.decode(C.get(tensors,arm,i,device)).cpu().numpy().tobytes()) for i in range(count)]
        assert hashes==pair['effective_function_hashes'] and len(set(hashes['e16']+hashes['e160']))==176
        gates={'all_source_capture_snapshot_route_bindings':True,'all_fit_labels_counts_exact':True,
            'only_two_bias_arrays_changed':True,'all_snapshot_tensors_readback_exact':True,
            'all160_parent_value_and_centered_mean_checks':len(audits)==160,
            'all176_effective_weights_distinct_and_unchanged':True,'unchanged_parent_fit_scores_exact':True}
        del matrices,xf,yf,phi,z,zd,wp,wf,y,yc,center_phi
        stage='single_consumed_validation_after_all_corrections'
        with np.load(R.CAPTURE,allow_pickle=False) as archive:
            xv=torch.from_numpy(archive['x_bf16'][M.FIT:].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
            yv=torch.from_numpy(archive['y_bf16'][M.FIT:].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
        pv=R.assign_full(xv,parents); cv=torch.empty_like(pv)
        for parent in range(16):
            mask=pv==parent
            if bool(mask.any()): cv[mask]=parent*10+R.assign_full(xv[mask],children[parent])
        rotated=cv//10*10+(cv%10+1)%10; hv=Q.features(xv,values)
        predictions={}
        for arm,bank,labels in (('parent_prior','parent_prior',pv),('e16','e16',pv),
            ('child_prior','child_prior',cv),('e160','e160',cv),('e160_rotated','e160',rotated)):
            predictions[arm]=C.predict_bank(hv,labels,tensors,bank,device); P.budget(start,device)
        for sequence in range(M.VALID):
            span=slice(sequence*M.SEQ,(sequence+1)*M.SEQ); energy=float(yv[span].double().square().sum())
            assert energy==pair['validation_rows'][sequence]['energy']
            sse={arm:float((pred[span]-yv[span]).double().square().sum()) for arm,pred in predictions.items()}
            assert all(sse[arm]==pair['validation_rows'][sequence]['sse'][arm] for arm in ('parent_prior','e16'))
            rows.append({'validation_sequence':sequence,'energy':energy,'sse':sse})
        energy=sum(r['energy'] for r in rows)
        summary={arm:{'sse':sum(r['sse'][arm] for r in rows),'normalized_sse':sum(r['sse'][arm] for r in rows)/energy} for arm in predictions}
        rng=np.random.default_rng(243244); draws=rng.integers(0,M.VALID,size=(10000,M.VALID))
        delta=np.asarray([r['sse']['e16']-r['sse']['e160'] for r in rows]); ey=np.asarray([r['energy'] for r in rows])
        gain=delta[draws].sum(1)/ey[draws].sum(1)
        summary['bootstrap']={'gain_p05':float(np.quantile(gain,.05)),'gain_p95':float(np.quantile(gain,.95)),
            'seed':243244,'draws':10000,'unit':'consumed_training_corpus_window'}
        gates.update({'unchanged_parent_validation_scores_exact':True,'complete_function_nmse_le_001':summary['e160']['normalized_sse']<=.01,
            'e160_sse_le_90_percent_e16':summary['e160']['sse']<=.9*summary['e16']['sse'],
            'e160_sse_le_90_percent_rotated':summary['e160']['sse']<=.9*summary['e160_rotated']['sse'],
            'e160_sse_le_90_percent_child_prior':summary['e160']['sse']<=.9*summary['child_prior']['sse'],
            'e160_sse_le_90_percent_parent_prior':summary['e160']['sse']<=.9*summary['parent_prior']['sse'],
            'paired_window_gain_p05_positive':float(np.quantile(gain,.05))>0})
        result={'experiment':'METH-243-fitted-parent-value-source-slope-hierarchical-E16-E160-pair',
            'pair_result_sha256':D.PAIR_SHA,'precision_result_sha256':DIAG_SHA,'value_only_result_sha256':VALUE_SHA,
            'input_snapshot_sha256':pair['checkpoint_sha256'],'source_sha256':P.M57.MODEL_SHA,
            'source_tensor_sha256':source_hashes,'capture_sha256':pair['capture_sha256'],
            'native_result_sha256':Q.NATIVE_SHA,'route_result_sha256':Q.S.ROUTE_SHA,'controls':controls,
            'prior_sample_equivalents':C.TAU,'counts16':pair['counts16'],'counts160':pair['counts160'],
            'additional_label_fit_solves':0,'parent_point_replays':160,'intercept_transport_audits':audits,
            'unchanged_tensor_names':unchanged,'effective_function_hashes':hashes,
            'fit_normalized_sse':fit_nmse,'summary':summary,'validation_rows':rows,'gates':gates,
            'checkpoint_sha256':P.digest(args.checkpoint),'checkpoint_bytes':args.checkpoint.stat().st_size,
            'script_sha256':P.digest(Path(__file__)),'runtime':P.budget(start,device),'torch_version':torch.__version__,
            'decision':'local_hierarchical_parent_value_count_pass_native_stored_bank_next' if all(gates.values()) else 'stop_fixed_parent_value_source_slope_hierarchy_as_count_solution',
            'scope':'Child priors conserve fitted parent center values rather than reset to original donor; all trained coefficients, parent biases, routing, lookup, ridge strength and FP32-source sensitivities unchanged. One-layer consumed development windows; no fresh/full model, routed C, large-RAM scaling, accepted rate or second donor.'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('decision','fit_normalized_sse','summary','gates','runtime')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),
            'audits':audits,'validation_rows':rows,'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__': main()
