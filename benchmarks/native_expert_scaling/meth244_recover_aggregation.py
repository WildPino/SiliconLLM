#!/usr/bin/env python3
"""Recover only aggregation of already completed and byte-exact METH-244 fits."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import meth244_readout_encoding_audit as A

P,M,R=A.P,A.M,A.R
FAIL=P.DOC/'meth244_readout_encoding_result.failure.json'
FAIL_SHA='1c80d4cc8e45faf72437c1aef7df21ccbcc9d679dc1004f1096598ccbb53465f'


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args(); assert not args.out.exists()
    assert P.digest(FAIL)==FAIL_SHA and P.digest(A.D.PAIR)==A.D.PAIR_SHA and P.digest(A.HIER)==A.HIER_SHA
    failed=json.loads(FAIL.read_text()); pair=json.loads(A.D.PAIR.read_text()); rows=failed['rows']
    assert failed['stage']=='exact_fixed176_solve_replay_and_encoding_error_ledger'
    assert failed['error']=='AssertionError()' and len(rows)==176
    assert {(r['arm'],r['cell']) for r in rows}=={('e16',i) for i in range(16)}|{('e160',i) for i in range(160)}
    for r in rows:
        assert r['states']==pair['counts16' if r['arm']=='e16' else 'counts160'][r['cell']]
        assert r['normal_equation_relative_residual']<=1e-7 and r['SSE_identity_relative_closure']<=1e-9
    assert P.digest(R.CAPTURE)==pair['capture_sha256']
    start=time.monotonic(); device=M.D.Q.setup(); P.MAX_SECONDS=P.M17.MAX_SECONDS=60
    with np.load(R.CAPTURE,allow_pickle=False) as archive:
        yf=torch.from_numpy(archive['y_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
    energy=float(yf.double().square().sum()); summary,raw_gain,encoding=A.aggregate(rows,pair,energy)
    result={'experiment':'METH-244-fixed-readout-continuous-versus-stored-fit-error-audit',
        'execution_freeze':'acc61cd','recovered_failure_sha256':FAIL_SHA,'pair_result_sha256':A.D.PAIR_SHA,
        'hierarchy_result_sha256':A.HIER_SHA,'snapshot_sha256':pair['checkpoint_sha256'],
        'source_sha256':pair['source_sha256'],'capture_sha256':pair['capture_sha256'],
        'native_result_sha256':pair['native_result_sha256'],'route_result_sha256':pair['route_result_sha256'],
        'controls':pair['controls'],'prior_sample_equivalents':A.C.TAU,'rows':rows,'summary':summary,
        'gates':{'all_source_snapshot_capture_route_bindings':True,'exact_fit_labels_counts':True,
            'all176_encoded_coefficients_and_biases_replay_exact':True,
            'actual_stored_fit_scores_replay_within_1e12_relative':True,'all_FP64_SSE_identities_close':True},
        'diagnostic_decisions':{'raw_E160_fit_sse_le_90_percent_raw_E16':raw_gain,
            'encoding_penalty_ge_half_actual_positive_count_gap':encoding},
        'decision':'continuous_local_count_gain_exposed_change_readout_encoding' if raw_gain and encoding else 'continuous_local_count_gain_absent_do_not_retry_codec_only',
        'recovery_script_sha256':P.digest(Path(__file__)),'repaired_runner_sha256':P.digest(Path(A.__file__)),
        'runtime':{'original_attempt_seconds':failed['seconds'],'original_peak_bytes_not_retained':True,
            'recovery':P.budget(start,device)},
        'scope':'Recovered176 completed exact-code fit replays; aggregation-only repair uses common capture energy and 1e-12 FP64 summation tolerance. No new fit/source/validation or changed count thresholds. Nondeployable FP64 fit-only diagnostic.'}
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('decision','diagnostic_decisions','summary','gates','runtime')}),flush=True)


if __name__=='__main__': main()
