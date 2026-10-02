#!/usr/bin/env python3
"""Continuous bounded source-response dictionary; matched active32 E16/E160."""
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
import meth254_private_unit_pair as F

B,A,C,P,M,R,Q=F.B,F.A,F.C,F.P,F.M,F.R,F.Q
DIAG=P.DOC/'meth255_feature_readout_composition_result.json'
DIAG_SHA='243f748287a50b88c4aeaeecaa362f7f8e7f742271c692a495d5497ac17691a2'
K=32


def box_solve(gram,cross,variance,lower,upper,initial=None,tau=C.TAU):
    gram=np.asarray(gram,dtype=np.float64);cross=np.asarray(cross,dtype=np.float64)
    variance=np.asarray(variance,dtype=np.float64);root=np.sqrt(variance)
    assert np.isfinite(gram).all() and np.isfinite(cross).all() and np.all(variance>0)
    symmetry=float(np.linalg.norm(gram-gram.T)/max(np.linalg.norm(gram),1e-12));assert symmetry<=1e-10
    matrix=(gram+gram.T)*.5/root[:,None]/root[None,:]+tau*np.eye(len(root))
    d=cross/root;lo=np.asarray(lower)*root;hi=np.asarray(upper)*root
    assert np.all(lo<=hi);condition=float(np.linalg.cond(matrix));assert np.isfinite(condition) and condition<=1e8
    beta=np.clip(np.zeros(len(root)) if initial is None else np.asarray(initial)*root,lo,hi)
    before=float(.5*beta@matrix@beta+d@beta);scale=max(float(np.max(np.abs(d))),1e-12)
    for sweep in range(256):
        for j in range(len(beta)):
            outside=matrix[j]@beta-matrix[j,j]*beta[j]
            beta[j]=np.clip(-(d[j]+outside)/matrix[j,j],lo[j],hi[j])
        gradient=matrix@beta+d;projected=gradient.copy()
        projected[beta==lo]=np.minimum(projected[beta==lo],0)
        projected[beta==hi]=np.maximum(projected[beta==hi],0)
        normal=float(np.max(np.abs(projected))/scale)
        if normal<=1e-7:break
    alpha=beta/root;after=float(.5*beta@matrix@beta+d@beta)
    assert normal<=1e-7 and np.isfinite(alpha).all() and after<=before+1e-10*max(abs(before),1)
    assert np.all(alpha>=np.asarray(lower)-1e-12) and np.all(alpha<=np.asarray(upper)+1e-12)
    return alpha,{'sweeps':sweep+1,'normalized_projected_KKT_infinity':normal,'normalized_system_condition':condition,
        'gram_symmetry_relative':symmetry,'regularized_quadratic_before':before,'regularized_quadratic_after':after}


def apparatus():
    alpha,audit=box_solve([[4,1],[1,3]],[-30,2],[2,1],[-1,-1],[1,1],tau=3)
    assert np.allclose(alpha,[1,-.5],rtol=0,atol=1e-10)
    delta,second=box_solve([[2,0],[0,0]],[-1,0],[.4,.8],[-1.8,-.8],[.2,1.2],tau=3)
    assert np.allclose(delta,[.2,0],rtol=0,atol=1e-10)
    return {'bounded_cross_coupled_known_solution':alpha.tolist(),'inactive_child_coordinate_known_solution':delta.tolist(),
        'controls':[audit,second]}


def choose(gram,cross,variance,eligible):
    chosen=[];alpha=np.empty(0);steps=[];diagonal=np.diag(gram)
    corr=cross.copy()
    for step in range(K):
        candidate=np.clip(-corr/(diagonal+C.TAU*variance),-1,1)
        gains=-(2*candidate*corr+candidate*candidate*(diagonal+C.TAU*variance))
        gains[~eligible]=-np.inf
        if chosen:gains[chosen]=-np.inf
        index=int(np.argmax(gains));gain=float(gains[index])
        if not np.isfinite(gain) or gain<=0:return chosen,alpha,steps
        previous={j:a for j,a in zip(chosen,alpha)};chosen=sorted(chosen+[index])
        ss=np.ix_(chosen,chosen);warm=np.asarray([previous.get(j,0.) for j in chosen])
        alpha,audit=box_solve(gram[ss],cross[chosen],variance[chosen],-np.ones(len(chosen)),np.ones(len(chosen)),warm)
        corr=cross+gram[:,chosen]@alpha
        steps.append({'added_source_unit':index,'fixed_other_coordinate_regularized_gain':gain,**audit})
    return chosen,alpha,steps


def statistics(delta,residual,weight):
    gram=(delta.T@delta)*(weight.T@weight)
    cross=(delta*(residual@weight)).sum(0)
    assert torch.isfinite(gram).all() and torch.isfinite(cross).all()
    return gram,cross


def private_delta(x,phi,gate,up,ids):
    private=Q.L.silu_lookup(TF.linear(x,gate.float()))*TF.linear(x,up.float())
    return private.double()-phi[:,ids.long()].double()


def evaluate(reference,delta,weight,alpha):
    return reference.double()+(delta*alpha[None,:])@weight.T


def predict(x,phi,labels,reference,tensors,bank,device):
    out=torch.empty_like(reference,dtype=torch.float64)
    for index in range(16 if bank=='e16' else 160):
        selected=labels==index
        if bool(selected.any()):
            parent=index if bank=='e16' else index//10
            delta=private_delta(x[selected],phi[selected],tensors['atoms.gate'][parent].to(device),
                tensors['atoms.up'][parent].to(device),tensors['atoms.ids'][parent].to(device))
            out[selected]=evaluate(reference[selected],delta,tensors['atoms.weight_FP64'][parent].to(device),
                tensors[bank+'.alpha'][index].to(device))
    assert torch.isfinite(out).all();return out


def main():
    ap=argparse.ArgumentParser()
    for name in ('checkpoint','out'):ap.add_argument('--'+name,required=True,type=Path)
    args=ap.parse_args();assert not args.checkpoint.exists() and not args.out.exists()
    args.checkpoint.parent.mkdir(parents=True,exist_ok=True);assert shutil.disk_usage(args.checkpoint.parent).free>=2*1024**3
    start=time.monotonic();stage='bindings';parents_audit=[];children_audit=[];rows=[];probe_rows=[]
    try:
        for path,sha in ((DIAG,DIAG_SHA),(F.NATIVE,F.NATIVE_SHA),(F.F.D.PAIR,F.F.D.PAIR_SHA),
                         (A.D.PAIR,A.D.PAIR_SHA),(Q.NATIVE,Q.NATIVE_SHA),(Q.S.ROUTE_RESULT,Q.S.ROUTE_SHA)):
            assert P.digest(path)==sha
        diagnosis=json.loads(DIAG.read_text());pair=json.loads(F.F.D.PAIR.read_text());original=json.loads(A.D.PAIR.read_text())
        assert diagnosis['decision']=='private_feature_composition_not_qualified_change_coupled_transfer'
        assert P.digest(F.F.D.ENCODED)==pair['checkpoint_sha256'] and P.digest(R.CAPTURE)==pair['capture_sha256']
        assert P.digest(A.D.SNAPSHOT)==original['checkpoint_sha256']
        checks=apparatus()
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
        with safe_open(str(F.F.D.ENCODED),framework='pt',device='cpu') as archive:
            tensors={name:archive.get_tensor(name) for name in archive.keys() if not name.startswith('e160.')}
        old_keys=set(tensors)
        tensors.update({'atoms.ids':torch.empty((16,K),dtype=torch.uint16),
            'atoms.gate':torch.empty((16,K,896),dtype=torch.bfloat16),'atoms.up':torch.empty((16,K,896),dtype=torch.bfloat16),
            'atoms.weight_FP64':torch.empty((16,896,K),dtype=torch.float64),'atoms.variance':torch.empty((16,K),dtype=torch.float64),
            'e16.alpha':torch.empty((16,K),dtype=torch.float64),'e160.alpha':torch.empty((160,K),dtype=torch.float64)})
        with np.load(R.CAPTURE,allow_pickle=False) as archive:
            x=torch.from_numpy(archive['x_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
            y=torch.from_numpy(archive['y_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
        router_parents=tensors['router.parents'].to(device);router_children=tensors['router.children'].to(device)
        lp=R.assign_full(x,router_parents);lc=torch.empty_like(lp)
        for parent in range(16):
            selected=lp==parent;lc[selected]=parent*10+R.assign_full(x[selected],router_children[parent])
        assert torch.bincount(lp,minlength=16).tolist()==pair['counts16'] and torch.bincount(lc,minlength=160).tolist()==pair['counts160']
        route=json.loads(Q.S.ROUTE_RESULT.read_text())
        for label,key in ((lp,'parent_labels_sha256'),(lc,'leaf_labels_sha256')):
            assert P.M17.sha(label.cpu().numpy().astype('<i4').tobytes())==route['fit'][key]
        source_gate=source['gate'].to(device).float();source_up=source['up'].to(device).float()
        shared_gate=values['gate.q'].float()*values['gate.scale'][:,None];shared_up=values['up.q'].float()*values['up.scale'][:,None]
        discrepancy=(source_gate.double()-shared_gate.double()).square().sum(1)+(source_up.double()-shared_up.double()).square().sum(1)
        energy=float(y.double().square().sum());signatures=[];complete=True;stage='bounded_matching_and_anchored_nonlinear_amplitude_fits'
        for parent in range(16):
            selected=lp==parent;xx=x[selected];yy=y[selected];local=lc[selected];phi=Q.features(xx,values)
            base=C.get(tensors,'base',parent,device);factors=B.factor_get(tensors,'e16',parent,device)
            reference=B.encoded_value(phi,base,factors);reference_sse32=float((reference-yy).double().square().sum())
            old=next(r for r in pair['fit_rows'] if r['arm']=='e16' and r['cell']==parent);assert reference_sse32==old['actual_factorized_sse']
            weight=C.decode(base).double()+factors[1].float().double()@factors[0].float().double()
            weight_hash=P.M17.sha(weight.cpu().numpy().tobytes());assert weight_hash==pair['decoded_FP64_coefficient_hashes']['e16'][parent]
            delta=(Q.L.silu_lookup(TF.linear(xx,source_gate))*TF.linear(xx,source_up)).double()-phi.double()
            residual=reference.double()-yy.double();gram,cross=statistics(delta,residual,weight)
            diagonal=gram.diag();eligible=(diagonal>0)&(discrepancy>0)
            mean_energy=diagonal/len(xx);floor=max(float(mean_energy[eligible].mean())*1e-4,1e-12)
            variance=mean_energy.clamp_min(floor)
            ids,provisional,steps=choose(gram.cpu().numpy(),cross.cpu().numpy(),variance.cpu().numpy(),eligible.cpu().numpy())
            if len(ids)!=K:
                parents_audit.append({'parent':parent,'selected_units':len(ids),'steps':steps});complete=False;break
            ids_cpu=torch.tensor(ids,dtype=torch.long);ids_gpu=ids_cpu.to(device)
            gate=source['gate'][ids_cpu].contiguous().to(device);up=source['up'][ids_cpu].contiguous().to(device)
            selected_weight=weight[:,ids_gpu].contiguous();actual_delta=private_delta(xx,phi,gate,up,ids_gpu)
            full_vs_stored=float(((actual_delta-delta[:,ids_gpu])@selected_weight.T).norm()/reference.double().norm());assert full_vs_stored<=1e-5
            g,c=statistics(actual_delta,residual,selected_weight)
            selected_variance=(g.diag()/len(xx)).clamp_min(floor)
            alpha,audit=box_solve(g.cpu().numpy(),c.cpu().numpy(),selected_variance.cpu().numpy(),-np.ones(K),np.ones(K),provisional)
            alpha_gpu=torch.from_numpy(alpha).to(device);parent_prediction=evaluate(reference,actual_delta,selected_weight,alpha_gpu)
            expected_sse=float(residual.square().sum())+2*float(c@alpha_gpu)+float(alpha_gpu@g@alpha_gpu)
            sse=float((parent_prediction-yy.double()).square().sum());assert abs(sse-expected_sse)/max(float(residual.square().sum()),1e-12)<=1e-8
            tensors['atoms.ids'][parent].copy_(ids_cpu.to(torch.uint16));tensors['atoms.gate'][parent].copy_(gate.cpu());tensors['atoms.up'][parent].copy_(up.cpu())
            tensors['atoms.weight_FP64'][parent].copy_(selected_weight.cpu());tensors['atoms.variance'][parent].copy_(selected_variance.cpu());tensors['e16.alpha'][parent].copy_(torch.from_numpy(alpha))
            parents_audit.append({'parent':parent,'selected_units':K,'ids':ids,'states':len(xx),'reference_FP32_sse':reference_sse32,
                'reference_FP64_sse':float(residual.square().sum()),'sse':sse,'dictionary_energy_floor':floor,
                'full_vs_stored_source_response_relative_l2':full_vs_stored,'matching_steps':steps,**audit})
            probes=[parent_prediction[:64]]
            gate_hash=P.M17.sha(gate.cpu().view(torch.uint16).numpy().tobytes());up_hash=P.M17.sha(up.cpu().view(torch.uint16).numpy().tobytes())
            def signature(a):
                coefficients=selected_weight*a[None,:];coefficients=torch.where(coefficients==0,torch.zeros_like(coefficients),coefficients)
                return P.M17.sha(bytes.fromhex(weight_hash+gate_hash+up_hash)+coefficients.cpu().numpy().tobytes())
            signatures.append(signature(alpha_gpu))
            for child in range(10):
                leaf=parent*10+child;mask=local==leaf;d=actual_delta[mask];target=yy[mask].double();pp=parent_prediction[mask]
                gc,cc=statistics(d,pp-target,selected_weight)
                adjustment,audit=box_solve(gc.cpu().numpy(),cc.cpu().numpy(),selected_variance.cpu().numpy(),-1-alpha,1-alpha)
                total=np.clip(alpha+adjustment,-1,1);a=torch.from_numpy(total).to(device)
                pred=evaluate(reference[mask],d,selected_weight,a)
                anchored=evaluate(pp,d,selected_weight,torch.from_numpy(adjustment).to(device))
                conservation=float((pred-anchored).norm()/pred.norm().clamp_min(1e-12));assert conservation<=1e-10
                initial_sse=float((pp-target).square().sum());observed=float((pred-target).square().sum())
                change=torch.from_numpy(adjustment).to(device)
                expected=initial_sse+2*float(cc@change)+float(change@gc@change)
                assert abs(observed-expected)/max(initial_sse,1e-12)<=1e-8
                tensors['e160.alpha'][leaf].copy_(torch.from_numpy(total));signatures.append(signature(a))
                probes.append(evaluate(reference[:64],actual_delta[:64],selected_weight,a))
                children_audit.append({'child':leaf,'states':int(mask.sum()),'sse':observed,'parent_sse_on_support':initial_sse,
                    'total_alpha_min':float(total.min()),'total_alpha_max':float(total.max()),'anchored_prediction_conservation_relative_l2':conservation,**audit})
                P.budget(start,device)
            denominator=reference[:64].double().norm().clamp_min(1e-12)
            minimum=min(float((probes[i]-probes[j]).norm()/denominator) for i in range(11) for j in range(i))
            probe_rows.append({'parent':parent,'common_fit_probe_states':len(reference[:64]),'minimum_pair_relative_l2':minimum})
            args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'parent_rows':parents_audit,'child_rows':children_audit,'probe_rows':probe_rows},indent=2)+'\n',encoding='utf-8')
            del gram,cross,g,c,delta,weight,phi,actual_delta,selected_weight,parent_prediction,probes;P.budget(start,device)
        gates={'source_parent_capture_route_diagnosis_bindings':True,'bounded_solver_known_controls':True,
            'all16_parent32_dictionaries_and160_child_fits_complete':complete and len(children_audit)==160,
            'all176_amplitude_weighted_function_parameters_distinct':len(signatures)==176 and len(set(signatures))==176,
            'all16_common_probe_min_relative_l2_gt_1e_minus_7':len(probe_rows)==16 and all(r['minimum_pair_relative_l2']>1e-7 for r in probe_rows)}
        checkpoint=None;fit_summary={};summary={}
        if complete:
            stage='continuous_snapshot_and_exact_original_parent_readback'
            save_file(tensors,str(args.checkpoint),metadata={'experiment':'METH-256','scope':'nondeployable FP64 source-response amplitudes',
                'parent_snapshot_sha256':pair['checkpoint_sha256'],'tau':str(C.TAU),'private_units':str(K)})
            assert args.checkpoint.stat().st_size<120000000
            with safe_open(str(args.checkpoint),framework='pt',device='cpu') as archive,safe_open(str(F.F.D.ENCODED),framework='pt',device='cpu') as previous:
                assert set(archive.keys())==set(tensors)
                for name,t in tensors.items():
                    assert torch.equal(archive.get_tensor(name),t) and torch.isfinite(t).all()
                    if name in old_keys:assert torch.equal(t,previous.get_tensor(name))
            assert sum(r['reference_FP32_sse'] for r in parents_audit)/energy==pair['fit_summary']['e16']['factorized_normalized_sse']
            ref=sum(r['reference_FP64_sse'] for r in parents_audit);pa=sum(r['sse'] for r in parents_audit);ch=sum(r['sse'] for r in children_audit)
            fit_summary={'original_FP32_reference_nmse':sum(r['reference_FP32_sse'] for r in parents_audit)/energy,
                'original_FP64_reference_nmse':ref/energy,'e16_nmse':pa/energy,'e160_nmse':ch/energy}
            checkpoint={'bytes':args.checkpoint.stat().st_size,'sha256':P.digest(args.checkpoint)}
            gates.update({'all_old_fields_coefficients_FP32_fit_controls_unchanged':True,'all_snapshot_fields_readback_exact':True,
                'all_bounded_normal_ledger_conservation_controls':True,'fit_function_nmse_le_001':ch/energy<=.01,
                'fit_e16_no_worse_than_original':pa<=ref*(1+1e-12),'fit_e160_no_worse_than_e16':ch<=pa*(1+1e-12)})
        if all(gates.values()):
            del x,y,xx,yy,source_gate,source_up;stage='consumed_validation_once_after_all_fit_gates'
            with np.load(R.CAPTURE,allow_pickle=False) as archive:
                x=torch.from_numpy(archive['x_bf16'][M.FIT:].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
                y=torch.from_numpy(archive['y_bf16'][M.FIT:].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
            pv=R.assign_full(x,router_parents);cv=torch.empty_like(pv)
            for parent in range(16):
                mask=pv==parent
                if bool(mask.any()):cv[mask]=parent*10+R.assign_full(x[mask],router_children[parent])
            rotated=cv//10*10+(cv%10+1)%10;phi=Q.features(x,values);reference=B.predict(phi,pv,tensors,'e16',device)
            predictions={'e16_reference':reference.double(),'e16':predict(x,phi,pv,reference,tensors,'e16',device),
                'e160':predict(x,phi,cv,reference,tensors,'e160',device),'e160_rotated':predict(x,phi,rotated,reference,tensors,'e160',device)}
            with safe_open(str(A.D.SNAPSHOT),framework='pt',device='cpu') as archive:
                priors={k:archive.get_tensor(k) for k in archive.keys() if k.startswith(('parent_prior.','child_prior.'))}
            for bank,labels in (('parent_prior',pv),('child_prior',cv)):predictions[bank]=C.predict_bank(phi,labels,priors,bank,device).double()
            for sequence in range(M.VALID):
                span=slice(sequence*M.SEQ,(sequence+1)*M.SEQ);ey=float(y[span].double().square().sum());assert ey==pair['validation_rows'][sequence]['energy']
                sse={bank:float((pred[span]-y[span].double()).square().sum()) for bank,pred in predictions.items()}
                control={bank:float((predictions['e16_reference' if bank=='e16' else bank][span].float()-y[span]).double().square().sum()) for bank in ('e16','parent_prior','child_prior')}
                assert all(control[bank]==pair['validation_rows'][sequence]['sse'][bank] for bank in control)
                rows.append({'validation_sequence':sequence,'energy':ey,'sse':sse,'unchanged_FP32_control_sse':control})
            ey=sum(r['energy'] for r in rows);summary={bank:{'sse':sum(r['sse'][bank] for r in rows),'normalized_sse':sum(r['sse'][bank] for r in rows)/ey} for bank in predictions}
            rng=np.random.default_rng(256257);draws=rng.integers(0,M.VALID,size=(10000,M.VALID))
            diff=np.asarray([r['sse']['e16']-r['sse']['e160'] for r in rows]);energies=np.asarray([r['energy'] for r in rows]);gain=diff[draws].sum(1)/energies[draws].sum(1)
            summary['bootstrap']={'gain_p05':float(np.quantile(gain,.05)),'gain_p95':float(np.quantile(gain,.95)),'seed':256257,'draws':10000}
            gates.update({'unchanged_original_parent_prior_validation_controls_exact':True,'function_nmse_le_001':summary['e160']['normalized_sse']<=.01,
                **{'e160_sse_le_90_percent_'+bank:summary['e160']['sse']<=.9*summary[bank]['sse'] for bank in ('e16','e160_rotated','e16_reference','parent_prior','child_prior')},
                'paired_window_gain_p05_positive':float(np.quantile(gain,.05))>0})
        result={'experiment':'METH-256-bounded-source-response-nonlinear-active32-E16-E160-continuous-pilot',
            'diagnosis_result_sha256':DIAG_SHA,'native_geometry_sha256':F.NATIVE_SHA,'parent_result_sha256':F.F.D.PAIR_SHA,
            'parent_snapshot_sha256':pair['checkpoint_sha256'],'source_sha256':P.M57.MODEL_SHA,'source_tensor_hashes':source_hashes,
            'capture_sha256':pair['capture_sha256'],'route_result_sha256':Q.S.ROUTE_SHA,'controls':controls,'tau':C.TAU,
            'counts16':pair['counts16'],'counts160':pair['counts160'],'solver_apparatus':checks,'parent_rows':parents_audit,
            'child_rows':children_audit,'probe_rows':probe_rows,'fit_summary':fit_summary,'function_parameter_signatures':signatures,
            'gates':gates,'summary':summary,'validation_rows':rows,'checkpoint':checkpoint,'runtime':P.budget(start,device),
            'script_sha256':P.digest(Path(__file__)),'torch_version':torch.__version__,
            'decision':'bounded_nonlinear_source_response_count_pass_freeze_native_blend_bank' if all(gates.values()) else 'stop_this_fixed_bounded_nonlinear_source_response_pair',
            'scope':'Actual parent protected,original source nonlinear difference atoms,same32 active atoms per parent/child;bounded amplitudes in FP64. Consumed development after fit gates. No native blending or learned bank/large-n/DRAM/full independent quality/accepted rate/family transfer.'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:result[k] for k in ('decision','fit_summary','summary','gates','runtime')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),'parent_rows':parents_audit,
            'child_rows':children_audit,'probe_rows':probe_rows,'validation_rows':rows,'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__':main()
