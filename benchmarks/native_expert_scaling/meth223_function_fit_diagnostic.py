#!/usr/bin/env python3
"""Saved-state fit/validation gap and coefficient precision diagnostic."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from safetensors import safe_open
import torch
from torch.nn import functional as F
import meth222_conditional_function_pilot as M

P=M.P
PREVIOUS=P.DOC/'meth222_conditional_function_pilot_result.json'
PREVIOUS_SHA='41858ede5147f7c98821987892096acb0aa7abf3611db4a1b849663e1d0ddb34'
CAPTURE=P.ROOT/'results/native_expert_scaling/meth222_layer12_training_states.npz'
CHECKPOINT=P.ROOT/'results/native_expert_scaling/meth222_layer12_function_cells.safetensors'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    assert not args.out.exists()
    start=time.monotonic()
    assert P.digest(PREVIOUS)==PREVIOUS_SHA
    old=json.loads(PREVIOUS.read_text(encoding='utf-8'))
    assert P.digest(CAPTURE)==old['capture']['sha256'] and P.digest(CHECKPOINT)==old['fit']['checkpoint_sha256']
    device=M.D.Q.setup()
    P.MAX_SECONDS=P.M17.MAX_SECONDS=5*60
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    with np.load(CAPTURE,allow_pickle=False) as archive:
        x=torch.from_numpy(archive['x_bf16'].reshape(-1,896)).view(torch.bfloat16).to(device).float()
        y=torch.from_numpy(archive['y_bf16'].reshape(-1,896)).view(torch.bfloat16).to(device).float()
    with safe_open(str(CHECKPOINT),framework='pt',device='cpu') as archive:
        saved={name:archive.get_tensor(name).to(device).float() for name in archive.keys()}
    cut=M.FIT*M.SEQ
    mx,my=x[:cut].double().mean(0),y[:cut].double().mean(0)
    xc,yc=x[:cut].double()-mx,y[:cut].double()-my
    gram=xc.T@xc/cut
    ridge=float(gram.trace()/896)*.001
    w=torch.linalg.solve(gram+ridge*torch.eye(896,dtype=torch.float64,device=device),(yc.T@xc/cut).T).T
    assert torch.equal(w.bfloat16().float(),saved['common.weight'])
    assert torch.equal(my.float()-saved['common.weight']@mx.float(),saved['common.bias'])
    wf=w.float(); bf=my.float()-wf@mx.float()
    del xc,yc,gram,w
    # Reproduce original fitting association, and original deployment association.
    q_all=F.linear(x,saved['projection.weight'],saved['projection.bias'])
    common_all=F.linear(x,saved['common.weight'],saved['common.bias'])
    summaries,records={},{'fit':[],'validation':[]}
    counts=torch.as_tensor(old['fit']['counts160'],device=device)
    for kind,offset,number in (('fit',0,M.FIT),('validation',cut,M.VALID)):
        span=slice(offset,offset+number*M.SEQ)
        if kind=='fit':
            qv,common=q_all[span],common_all[span]
        else:
            qv=F.linear(x[span],saved['projection.weight'],saved['projection.bias'])
            common=F.linear(x[span],saved['common.weight'],saved['common.bias'])
        parents=M.assign(qv,saved['router.parent'])
        if kind=='fit':
            cells=torch.empty_like(parents)
            for parent in range(M.PARENTS):
                mask=parents==parent
                cells[mask]=parent*M.CHILDREN+M.assign(qv[mask],saved['router.children'][parent])
            assert torch.bincount(cells,minlength=160).tolist()==old['fit']['counts160']
            assert torch.bincount(parents,minlength=16).tolist()==old['fit']['counts16']
        else:
            cells=parents*M.CHILDREN+((qv[:,None]-saved['router.children'][parents]).square().sum(-1)).argmin(-1)
        float_common=F.linear(x[span],wf,bf)
        totals={arm:0.0 for arm in ('common','common_fp32_weight','e16','e160')}
        energy=0.0
        support={name:{'states':0,'energy':0.0,'e16_sse':0.0,'e160_sse':0.0} for name in ('below65','65_to255','at_least256')}
        for seq in range(number):
            s=slice(seq*M.SEQ,(seq+1)*M.SEQ)
            target=y[span][s]
            features=torch.cat((qv[s],torch.ones((M.SEQ,1),device=device)),dim=1)
            predictions={'common':common[s],'common_fp32_weight':float_common[s]}
            predictions['e16']=common[s]+torch.bmm(saved['experts.e16'][parents[s]],features[:,:,None]).squeeze(-1)
            predictions['e160']=common[s]+torch.bmm(saved['experts.e160'][cells[s]],features[:,:,None]).squeeze(-1)
            per_state={arm:(prediction-target).double().square().sum(-1) for arm,prediction in predictions.items()}
            state_energy=target.double().square().sum(-1)
            row={'sequence':seq,'energy':float(state_energy.sum()),'sse':{arm:float(value.sum()) for arm,value in per_state.items()}}
            if kind=='validation':
                reference=old['validation_rows'][seq]
                assert row['energy']==reference['energy']
                for arm in ('common','e16','e160'):
                    assert row['sse'][arm]==reference['sse'][arm]
            for arm,value in row['sse'].items(): totals[arm]+=value
            energy+=row['energy']; records[kind].append(row)
            selected_counts=counts[cells[s]]
            for name,mask in (('below65',selected_counts<65),('65_to255',(selected_counts>=65)&(selected_counts<256)),('at_least256',selected_counts>=256)):
                support[name]['states']+=int(mask.sum())
                support[name]['energy']+=float(state_energy[mask].sum())
                support[name]['e16_sse']+=float(per_state['e16'][mask].sum())
                support[name]['e160_sse']+=float(per_state['e160'][mask].sum())
            P.budget(start,device)
        summaries[kind]={'energy':energy,'sse':totals,'normalized_sse':{arm:v/energy for arm,v in totals.items()},'support_strata':support}
    fit,val=summaries['fit'],summaries['validation']
    gap=val['sse']['e160']-val['sse']['e16']
    low=val['support_strata']['below65']
    diagnoses={'common_representation_needs_nonlinearity':fit['normalized_sse']['common_fp32_weight']>=.05 and
        val['normalized_sse']['common']-val['normalized_sse']['common_fp32_weight']<=.005,
        'larger_bank_fit_gain_but_validation_loss':fit['sse']['e160']<=.9*fit['sse']['e16'] and val['sse']['e160']>val['sse']['e16'],
        'validation_loss_concentrated_in_under65_support':gap>0 and low['states']>=.01*M.VALID*M.SEQ and
            low['e160_sse']-low['e16_sse']>=.5*gap}
    result={'experiment':'METH-223-saved-function-fit-diagnostic','prior_sha256':PREVIOUS_SHA,
        'capture_sha256':old['capture']['sha256'],'checkpoint_sha256':old['fit']['checkpoint_sha256'],
        'script_sha256':P.digest(Path(__file__)),'fit_counts_reconcile_exact':True,'all_validation_metrics_reconcile_exact':True,
        'common_fit_replay_bf16_weight_bias_exact':True,'summary':summaries,'diagnoses':diagnoses,
        'sequence_rows':records,'runtime':P.budget(start,device),
        'decision':'specify_nonlinear_common_and_regularized_route_function_learning' if diagnoses['common_representation_needs_nonlinearity'] else 'reassess_representation_from_saved_fit_gap',
        'scope':'Replayed fixed consumed training states only; no new fitting hyperparameters, route/coefficient changes, source inference, quality or native promotion'}
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'summary':summaries,'diagnoses':diagnoses,'runtime':result['runtime']}),flush=True)


if __name__=='__main__':
    main()
