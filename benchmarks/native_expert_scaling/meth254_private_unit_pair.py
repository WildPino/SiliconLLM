#!/usr/bin/env python3
"""Fit-only source-unit choices, same active32 private nonlinear functions."""
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
from torch.nn import functional as TF
import meth249_parent_anchored_latent_pilot as F

B,A,C,P,M,R,Q=F.B,F.A,F.C,F.P,F.M,F.R,F.Q
NATIVE=P.DOC/'meth253_split_private_feature_native_result.json'
NATIVE_SHA='9262f98cd87021d32697bb79badc1f07d7480a558def3cce6c9e48fb7466611a'
K=32


def greedy(delta,reference,target,weight):
    residual=reference.double()-target.double();initial=float(residual.square().sum())
    dot=(delta*(residual@weight)).sum(0)
    penalty=delta.square().sum(0)*weight.square().sum(0)
    ids=[];gains=[]
    for step in range(K):
        gain=-(2*dot+penalty)
        if ids:gain[ids]=-float('inf')
        selected=int(gain.argmax());best=float(gain[selected])
        if not np.isfinite(best) or best<=0:break
        ids.append(selected);gains.append(best)
        response=delta[:,selected,None]*weight[:,selected][None,:]
        residual=residual+response
        dot=dot+(delta.T@delta[:,selected])*(weight.T@weight[:,selected])
    final=float(residual.square().sum());closure=abs(final-(initial-sum(gains)))/max(initial,1e-12)
    assert torch.isfinite(residual).all() and closure<=1e-8
    return sorted(ids),target.double()+residual,{
        'selected_units':len(ids),'positive_gains':gains,'initial_oracle_sse':initial,'final_oracle_sse':final,'oracle_sse_closure_relative':closure}


def output(phi,x,base,factors,gate,up,ids):
    private=Q.L.silu_lookup(TF.linear(x,gate.float()))*TF.linear(x,up.float())
    patched=phi.clone();patched[:,ids.long()]=private
    return B.encoded_value(patched,base,factors)


def predict(phi,x,labels,tensors,bank,device):
    out=torch.empty((len(x),896),device=device)
    for index in range(16 if bank=='private16' else 160):
        chosen=labels==index
        if bool(chosen.any()):
            parent=index if bank=='private16' else index//10
            out[chosen]=output(phi[chosen],x[chosen],C.get(tensors,'base',parent,device),B.factor_get(tensors,'e16',parent,device),
                tensors[bank+'.gate'][index].to(device),tensors[bank+'.up'][index].to(device),tensors[bank+'.ids'][index].to(device))
    assert torch.isfinite(out).all();return out


def main():
    ap=argparse.ArgumentParser()
    for name in ('checkpoint','out'):ap.add_argument('--'+name,required=True,type=Path)
    args=ap.parse_args();assert not args.checkpoint.exists() and not args.out.exists()
    args.checkpoint.parent.mkdir(parents=True,exist_ok=True);assert shutil.disk_usage(args.checkpoint.parent).free>=2*1024**3
    start=time.monotonic();stage='bindings';fits=[];controls_fit=[];probe_rows=[];rows=[]
    try:
        for path,sha in ((NATIVE,NATIVE_SHA),(F.D.PAIR,F.D.PAIR_SHA),(A.D.PAIR,A.D.PAIR_SHA),(Q.NATIVE,Q.NATIVE_SHA),(Q.S.ROUTE_RESULT,Q.S.ROUTE_SHA)):
            assert P.digest(path)==sha
        native=json.loads(NATIVE.read_text());pair=json.loads(F.D.PAIR.read_text());original=json.loads(A.D.PAIR.read_text())
        assert all(native['gates'].values()) and native['decision']=='split_private_source_feature_operator_pass_freeze_matched_unit_selection'
        assert P.digest(F.D.ENCODED)==pair['checkpoint_sha256'] and P.digest(R.CAPTURE)==pair['capture_sha256']
        assert P.digest(A.D.SNAPSHOT)==original['checkpoint_sha256']
        source_path=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True));assert P.digest(source_path)==P.M57.MODEL_SHA
        device=M.D.Q.setup();P.MAX_SECONDS=P.M17.MAX_SECONDS=20*60
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        values,controls=Q.load_layer(json.loads(Q.NATIVE.read_text()),device);assert controls==pair['controls']
        source={};source_hashes={}
        with safe_open(str(source_path),framework='pt',device='cpu') as archive:
            for organ in ('gate','up','down'):
                name=f'model.layers.{M.LAYER}.mlp.{organ}_proj.weight';w=archive.get_tensor(name);assert w.dtype==torch.bfloat16
                source_hashes[name]=P.M17.sha(w.view(torch.uint16).numpy().tobytes());source[organ]=w
        assert source_hashes==original['source_tensor_sha256']
        with safe_open(str(F.D.ENCODED),framework='pt',device='cpu') as archive:
            tensors={name:archive.get_tensor(name) for name in archive.keys() if not name.startswith('e160.')}
        old_keys=set(tensors)
        for bank,n in (('private16',16),('private160',160)):
            tensors[bank+'.ids']=torch.empty((n,K),dtype=torch.uint16)
            for organ in ('gate','up'):tensors[bank+'.'+organ]=torch.empty((n,K,896),dtype=torch.bfloat16)
        parents=tensors['router.parents'].to(device);children=tensors['router.children'].to(device)
        with np.load(R.CAPTURE,allow_pickle=False) as archive:
            x=torch.from_numpy(archive['x_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
            y=torch.from_numpy(archive['y_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
        lp=R.assign_full(x,parents);lc=torch.empty_like(lp)
        for parent in range(16):
            chosen=lp==parent;lc[chosen]=parent*10+R.assign_full(x[chosen],children[parent])
        assert torch.bincount(lp,minlength=16).tolist()==pair['counts16'] and torch.bincount(lc,minlength=160).tolist()==pair['counts160']
        route=json.loads(Q.S.ROUTE_RESULT.read_text())
        for label,key in ((lp,'parent_labels_sha256'),(lc,'leaf_labels_sha256')):
            assert P.M17.sha(label.cpu().numpy().astype('<i4').tobytes())==route['fit'][key]
        source_gate=source['gate'].to(device).float();source_up=source['up'].to(device).float()
        shared_gate=values['gate.q'].cpu().float()*values['gate.scale'].cpu()[:,None]
        shared_up=values['up.q'].cpu().float()*values['up.scale'].cpu()[:,None]
        energy=float(y.double().square().sum());signatures=[];all_complete=True;stage='fixed32_positive_greedy_source_unit_selection'
        for parent in range(16):
            chosen=lp==parent;xx=x[chosen];yy=y[chosen];local=lc[chosen];phi=Q.features(xx,values)
            base=C.get(tensors,'base',parent,device);factors=B.factor_get(tensors,'e16',parent,device);reference=B.encoded_value(phi,base,factors)
            old=next(r for r in pair['fit_rows'] if r['arm']=='e16' and r['cell']==parent)
            old_sse=float((reference-yy).double().square().sum());assert old_sse==old['actual_factorized_sse']
            controls_fit.append({'parent':parent,'sse':old_sse})
            weight=C.decode(base).double()+factors[1].float().double()@factors[0].float().double()
            weight_hash=P.M17.sha(weight.cpu().numpy().tobytes());assert weight_hash==pair['decoded_FP64_coefficient_hashes']['e16'][parent]
            source_phi=Q.L.silu_lookup(TF.linear(xx,source_gate))*TF.linear(xx,source_up);delta=source_phi.double()-phi.double()
            predictions=[];probe_phi=phi[:64];probe_x=xx[:64]
            tasks=[('private16',parent,torch.ones(len(xx),dtype=torch.bool,device=device))]+[('private160',parent*10+c,local==parent*10+c) for c in range(10)]
            for bank,index,selected in tasks:
                ids,oracle,audit=greedy(delta[selected],reference[selected],yy[selected],weight)
                if len(ids)!=K:
                    fits.append({'bank':bank,'cell':index,'states':int(selected.sum()),'ids':ids,**audit});all_complete=False;break
                ids_cpu=torch.tensor(ids,dtype=torch.long);ids_gpu=ids_cpu.to(device)
                gate=source['gate'][ids_cpu].contiguous();up=source['up'][ids_cpu].contiguous()
                discrepancy=(gate.float()-shared_gate[ids_cpu]).square().sum(1)+(up.float()-shared_up[ids_cpu]).square().sum(1)
                assert bool((discrepancy>0).all())
                tensors[bank+'.ids'][index].copy_(ids_cpu.to(torch.uint16))
                tensors[bank+'.gate'][index].copy_(gate);tensors[bank+'.up'][index].copy_(up)
                actual=output(phi[selected],xx[selected],base,factors,gate.to(device),up.to(device),ids_gpu)
                relative=float((actual.double()-oracle).norm()/oracle.norm());assert relative<=1e-5
                fit_sse=float((actual-yy[selected]).double().square().sum())
                probe=output(probe_phi,probe_x,base,factors,gate.to(device),up.to(device),ids_gpu)
                predictions.append(probe.double())
                # Hash actual decoded full feature coefficients, not mask metadata.
                gg=shared_gate.clone();uu=shared_up.clone();gg[ids_cpu]=gate.float();uu[ids_cpu]=up.float()
                gh=P.M17.sha(gg.numpy().tobytes());uh=P.M17.sha(uu.numpy().tobytes())
                signature=P.M17.sha(bytes.fromhex(weight_hash+gh+uh));signatures.append(signature)
                fits.append({'bank':bank,'cell':index,'states':int(selected.sum()),'ids':ids,'actual_FP32_sse':fit_sse,
                    'oracle_actual_prediction_relative_l2':relative,'decoded_gate_sha256':gh,'decoded_up_sha256':uh,
                    'parent_coefficient_sha256':weight_hash,'function_parameter_signature':signature,**audit})
                P.budget(start,device)
            if not all_complete:break
            denominator=reference[:64].double().norm().clamp_min(1e-12)
            distances=[float((predictions[i]-predictions[j]).norm()/denominator) for i in range(11) for j in range(i)]
            probe_rows.append({'parent':parent,'common_fit_probe_states':len(probe_x),'minimum_pair_relative_l2':min(distances)})
            args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'fit_rows':fits,'probe_rows':probe_rows},indent=2)+'\n',encoding='utf-8')
        fit_summary={bank:sum(r['actual_FP32_sse'] for r in fits if r['bank']==bank and 'actual_FP32_sse' in r)/energy for bank in ('private16','private160')}
        gates={'source_native_capture_route_bindings':True,'all176_choices_have32_positive_gain_units':all_complete and len(fits)==176,
            'all176_function_parameters_distinct':len(signatures)==176 and len(set(signatures))==176,
            'all16_common_probe_min_relative_l2_gt_1e_minus_7':len(probe_rows)==16 and all(r['minimum_pair_relative_l2']>1e-7 for r in probe_rows)}
        summary={};checkpoint=None
        if all_complete:
            assert sum(r['sse'] for r in controls_fit)/energy==pair['fit_summary']['e16']['factorized_normalized_sse']
            stage='actual_source_unit_snapshot_readback';save_file(tensors,str(args.checkpoint),metadata={'experiment':'METH-254','private_units':str(K),
                'parent_snapshot_sha256':pair['checkpoint_sha256'],'source_sha256':P.M57.MODEL_SHA,'selector':'fit-only positive greedy source feature repair'})
            assert args.checkpoint.stat().st_size<140000000
            with safe_open(str(args.checkpoint),framework='pt',device='cpu') as archive,safe_open(str(F.D.ENCODED),framework='pt',device='cpu') as old:
                assert set(archive.keys())==set(tensors)
                for name,t in tensors.items():
                    assert torch.equal(archive.get_tensor(name),t)
                    if name in old_keys:assert torch.equal(t,old.get_tensor(name))
            checkpoint={'sha256':P.digest(args.checkpoint),'bytes':args.checkpoint.stat().st_size}
            gates.update({'all_old_parent_fields_coefficient_fit_scores_unchanged':True,'all_snapshot_fields_readback_exact':True,
                'fit_function_nmse_le_001':fit_summary['private160']<=.01,
                'fit_both_no_worse_than_reference':max(fit_summary.values())<=pair['fit_summary']['e16']['factorized_normalized_sse']*(1+1e-8),
                'fit_private160_no_worse_than_private16':fit_summary['private160']<=fit_summary['private16']*(1+1e-8)})
        if all(gates.values()):
            del x,y,xx,yy,phi,delta,source_phi,actual,oracle,reference,predictions,source_gate,source_up
            stage='consumed_validation_once_after_all_fit_gates'
            with np.load(R.CAPTURE,allow_pickle=False) as archive:
                x=torch.from_numpy(archive['x_bf16'][M.FIT:].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
                y=torch.from_numpy(archive['y_bf16'][M.FIT:].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
            pv=R.assign_full(x,parents);cv=torch.empty_like(pv)
            for parent in range(16):
                chosen=pv==parent
                if bool(chosen.any()):cv[chosen]=parent*10+R.assign_full(x[chosen],children[parent])
            rotated=cv//10*10+(cv%10+1)%10;phi=Q.features(x,values)
            predictions={'e16_reference':B.predict(phi,pv,tensors,'e16',device),'e16':predict(phi,x,pv,tensors,'private16',device),
                'e160':predict(phi,x,cv,tensors,'private160',device),'e160_rotated':predict(phi,x,rotated,tensors,'private160',device)}
            with safe_open(str(A.D.SNAPSHOT),framework='pt',device='cpu') as archive:
                priors={k:archive.get_tensor(k) for k in archive.keys() if k.startswith(('parent_prior.','child_prior.'))}
            for bank,labels in (('parent_prior',pv),('child_prior',cv)):predictions[bank]=C.predict_bank(phi,labels,priors,bank,device)
            for sequence in range(M.VALID):
                span=slice(sequence*M.SEQ,(sequence+1)*M.SEQ);ey=float(y[span].double().square().sum());assert ey==pair['validation_rows'][sequence]['energy']
                sse={bank:float((prediction[span]-y[span]).double().square().sum()) for bank,prediction in predictions.items()}
                assert all(sse['e16_reference' if bank=='e16' else bank]==pair['validation_rows'][sequence]['sse'][bank] for bank in ('e16','parent_prior','child_prior'))
                rows.append({'validation_sequence':sequence,'energy':ey,'sse':sse})
            ey=sum(r['energy'] for r in rows);summary={bank:{'sse':sum(r['sse'][bank] for r in rows),'normalized_sse':sum(r['sse'][bank] for r in rows)/ey} for bank in predictions}
            rng=np.random.default_rng(254255);draws=rng.integers(0,M.VALID,size=(10000,M.VALID))
            diff=np.asarray([r['sse']['e16']-r['sse']['e160'] for r in rows]);energies=np.asarray([r['energy'] for r in rows]);gain=diff[draws].sum(1)/energies[draws].sum(1)
            summary['bootstrap']={'gain_p05':float(np.quantile(gain,.05)),'gain_p95':float(np.quantile(gain,.95)),'seed':254255,'draws':10000}
            gates.update({'unchanged_parent_prior_validation_controls_exact':True,'function_nmse_le_001':summary['e160']['normalized_sse']<=.01,
                **{'e160_sse_le_90_percent_'+bank:summary['e160']['sse']<=.9*summary[bank]['sse'] for bank in ('e16','e160_rotated','e16_reference','parent_prior','child_prior')},
                'paired_window_gain_p05_positive':float(np.quantile(gain,.05))>0})
        result={'experiment':'METH-254-source-bound-private32-nonlinear-E16-E160-unit-selection','native_result_sha256':NATIVE_SHA,
            'parent_result_sha256':F.D.PAIR_SHA,'parent_snapshot_sha256':pair['checkpoint_sha256'],'source_sha256':P.M57.MODEL_SHA,
            'source_tensor_hashes':source_hashes,'capture_sha256':pair['capture_sha256'],'route_result_sha256':Q.S.ROUTE_SHA,'controls':controls,
            'counts16':pair['counts16'],'counts160':pair['counts160'],'fit_rows':fits,'common_fit_probe_rows':probe_rows,'fit_summary':fit_summary,
            'gates':gates,'summary':summary,'validation_rows':rows,'checkpoint':checkpoint,'runtime':P.budget(start,device),
            'script_sha256':P.digest(Path(__file__)),'torch_version':torch.__version__,
            'decision':'private_source_unit_count_pass_native_learned_bank_next' if all(gates.values()) else 'stop_this_fixed_positive_greedy_private32_unit_pair',
            'scope':'Same32 original BF16 nonlinear rows per active function,unchanged full actual parent readout. Fit-only source unit choices;common probes,consumed validation only after all prerequisites. No independent/full quality,n/DRAM/dynamic-native bank or accepted rate.'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:result[k] for k in ('decision','fit_summary','summary','gates','runtime')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),'fit_rows':fits,'probe_rows':probe_rows,'validation_rows':rows,'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__':main()
