#!/usr/bin/env python3
"""Fit-only compare the same private features with source versus fitted readout."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from safetensors import safe_open
import meth254_private_unit_pair as F
import meth252_private_feature_feasibility as S

B,A,C,P,M,R,Q=F.B,F.A,F.C,F.P,F.M,F.R,F.Q
FAILED=P.DOC/'meth254_private_unit_pair_result.json'
FAILED_SHA='453f03f31183e4da4d29fdf6b3aafefce9eef9ab7f44b18704348576ca1e8c35'
SOURCE_RESULT=P.DOC/'meth252_private_feature_native_result.json'
SOURCE_RESULT_SHA='7da69e9d44df64719fddbfb62a452a2b4afb3be7044511f1749d16305b6910ce'
FIXTURE=P.ROOT/'results/native_expert_scaling/meth252_private_feature_fixture.bin'


def source_layer(record,device):
    values={}
    with FIXTURE.open('rb') as file:
        for segment in record['segments']:
            if segment['layer'] not in (None,M.LAYER):continue
            file.seek(segment['offset']);raw=file.read(segment['bytes']);assert P.M17.sha(raw)==segment['sha256']
            name=segment['name']
            if name.startswith('private.'):
                t=torch.from_numpy(np.frombuffer(raw,dtype='<u2').copy())
                if name!='private.ids':t=t.reshape(32,896).view(torch.bfloat16)
            else:t=S.A.tensor_from_raw(name,raw)
            values[name]=t.to(device)
    assert torch.equal(values['silu_table'],Q.L.TABLE)
    return values


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args();assert not args.out.exists();start=time.monotonic();stage='bindings';rows=[]
    try:
        for path,sha in ((FAILED,FAILED_SHA),(SOURCE_RESULT,SOURCE_RESULT_SHA),(F.NATIVE,F.NATIVE_SHA),
                         (F.F.D.PAIR,F.F.D.PAIR_SHA),(Q.NATIVE,Q.NATIVE_SHA),(Q.S.ROUTE_RESULT,Q.S.ROUTE_SHA)):
            assert P.digest(path)==sha
        failed=json.loads(FAILED.read_text());source=json.loads(SOURCE_RESULT.read_text());native=json.loads(F.NATIVE.read_text());pair=json.loads(F.F.D.PAIR.read_text())
        assert failed['decision']=='stop_this_fixed_positive_greedy_private32_unit_pair' and failed['checkpoint'] is None and not failed['validation_rows']
        assert all(native['gates'].values()) and P.digest(FIXTURE)==source['binary']['sha256']==native['fixture_sha256']
        assert P.digest(F.F.D.ENCODED)==pair['checkpoint_sha256'] and P.digest(R.CAPTURE)==pair['capture_sha256']
        device=M.D.Q.setup();P.MAX_SECONDS=P.M17.MAX_SECONDS=10*60
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        shared,controls=Q.load_layer(json.loads(Q.NATIVE.read_text()),device);assert controls==pair['controls']
        values=source_layer(source,device)
        for name in ('gate.q','gate.scale','up.q','up.scale'):assert torch.equal(values[name],shared[name])
        with safe_open(str(F.F.D.ENCODED),framework='pt',device='cpu') as archive:
            tensors={k:archive.get_tensor(k) for k in archive.keys() if not k.startswith('e160.')}
        with np.load(R.CAPTURE,allow_pickle=False) as archive:
            x=torch.from_numpy(archive['x_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
            y=torch.from_numpy(archive['y_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
        labels=R.assign_full(x,tensors['router.parents'].to(device));assert torch.bincount(labels,minlength=16).tolist()==pair['counts16']
        route=json.loads(Q.S.ROUTE_RESULT.read_text());assert P.M17.sha(labels.cpu().numpy().astype('<i4').tobytes())==route['fit']['parent_labels_sha256']
        energy=float(y.double().square().sum());stage='same_private_features_four_readout_compositions_fit_only'
        for parent in range(16):
            chosen=labels==parent;xx=x[chosen];yy=y[chosen];phi=Q.features(xx,shared);private=S.private_features(xx,values)
            base=C.get(tensors,'base',parent,device);factors=B.factor_get(tensors,'e16',parent,device)
            effective=C.decode(base).double()+factors[1].float().double()@factors[0].float().double()
            assert P.M17.sha(effective.cpu().numpy().tobytes())==pair['decoded_FP64_coefficient_hashes']['e16'][parent]
            outputs={'learned_shared':B.encoded_value(phi,base,factors),'learned_private':B.encoded_value(private,base,factors),
                'source_shared':S.readout(phi,values),'source_private':S.readout(private,values)}
            assert all(torch.isfinite(t).all() for t in outputs.values())
            sse={bank:float((prediction-yy).double().square().sum()) for bank,prediction in outputs.items()}
            expected=next(r for r in pair['fit_rows'] if r['arm']=='e16' and r['cell']==parent);assert sse['learned_shared']==expected['actual_factorized_sse']
            rows.append({'parent':parent,'states':len(xx),'sse':sse,'private_feature_change_squared_norm':float((private.double()-phi.double()).square().sum())})
            P.budget(start,device)
        summary={bank:{'sse':sum(r['sse'][bank] for r in rows),'normalized_sse':sum(r['sse'][bank] for r in rows)/energy} for bank in outputs}
        assert summary['learned_shared']['normalized_sse']==pair['fit_summary']['e16']['factorized_normalized_sse']
        gates={'source_native_parent_capture_route_bindings':True,'all16_original_learned_parent_scores_coefficients_exact':True,
            'same32_source_private_rows_and_shared_feature_bytes':True,'coherent_source_private_nmse_le_001':summary['source_private']['normalized_sse']<=.01,
            'coherent_source_private_improves_source_shared_at_least_10_percent':summary['source_private']['sse']<=.9*summary['source_shared']['sse'],
            'coherent_source_private_no_worse_than_learned_shared':summary['source_private']['sse']<=summary['learned_shared']['sse']}
        result={'experiment':'METH-255-fit-only-source-private-feature-readout-composition','failed_pair_sha256':FAILED_SHA,
            'source_result_sha256':SOURCE_RESULT_SHA,'native_result_sha256':F.NATIVE_SHA,'parent_result_sha256':F.F.D.PAIR_SHA,
            'parent_snapshot_sha256':pair['checkpoint_sha256'],'fixture_sha256':source['binary']['sha256'],'source_sha256':source['source_sha256'],
            'capture_sha256':pair['capture_sha256'],'route_result_sha256':Q.S.ROUTE_SHA,'private_ids':values['private.ids'].cpu().tolist(),
            'rows':rows,'summary':summary,'gates':gates,'runtime':P.budget(start,device),'script_sha256':P.digest(Path(__file__)),
            'decision':'coherent_source_private_features_pass_freeze_unit_pair' if all(gates.values()) else 'private_feature_composition_not_qualified_change_coupled_transfer',
            'scope':'Same source-only fixed32 private rows;fit-only four-function composition diagnostic,no new fits or validation targets. No learned count/native bank/full independent quality/accepted rate.'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:result[k] for k in ('decision','summary','gates','runtime')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),'rows':rows,'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__':main()
