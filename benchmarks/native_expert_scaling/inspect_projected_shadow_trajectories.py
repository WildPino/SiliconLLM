"""Stored-only geometry at seven matched donor/shadow residual boundaries."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
START=time.monotonic()


def main(a):
    import numpy as np,psutil
    proc=psutil.Process();proc.cpu_affinity([0]);doc=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
    assert extent(a.audit)['sha256']==a.audit_sha
    audit=json.loads(a.audit.read_bytes());assert audit['all411_parameter_and_metric_fields_checked'] and audit['all_inputs_outputs_before_after_exact']
    assert extent(audit['report']['path'])==audit['report'];shadow=json.loads(Path(audit['report']['path']).read_bytes())
    source_path=doc/'original_history_boundaries_finish_result_20261009.json'
    source_audit=doc/'original_history_boundaries_stored_adjudication_20261009.json'
    source=json.loads(source_path.read_bytes());assert source['basis']==json.loads(Path(shadow['binding']['path']).read_bytes())['basis']
    prior=json.loads(source_audit.read_bytes());assert prior['decision']=='STORED_BOUNDARIES_PROJECTION_PASS'
    assert extent(source_path)['sha256']==prior['result_sha256']
    lookup={r['id']:r for r in source['records']}
    inputs=[extent(__file__),extent(a.audit),audit['report'],extent(source_path),extent(source_audit),source['basis']]
    for r in shadow['DEV']:
        s=lookup[r['id']];assert s['history']==r['trajectory'][0]['shape'][0]
        for target,original in zip(r['trajectory'],s['boundaries']):
            assert target['boundary']==original['boundary']
            inputs.extend(({k:target[k] for k in ('path','bytes','sha256')},original['z'],original['h']))
    def guard():
        assert time.monotonic()-START<=120 and proc.memory_info().peak_wset<=512<<20
        assert not proc.children(recursive=True)
    def check():
        for item in inputs:guard();assert extent(item['path'])=={k:item[k] for k in ('path','bytes','sha256')},item['path']
    check();p=np.fromfile(source['basis']['path'],'<f4').reshape(2048,256).astype('f8');inv=np.linalg.inv(p.T@p);rows=[]
    for r in shadow['DEV']:
        for target,original in zip(r['trajectory'],lookup[r['id']]['boundaries']):
            guard();n=target['shape'][0]
            z=np.fromfile(target['path'],'<f4').reshape(n,256).astype('f8')
            ref=np.fromfile(original['z']['path'],'<f4').reshape(n,256).astype('f8')
            raw=(np.fromfile(original['h']['path'],'<u2').astype('<u4')<<16).view('<f4').reshape(n,2048).astype('f8')
            assert np.isfinite(z).all() and np.isfinite(ref).all() and np.isfinite(raw).all()
            total=float(np.sum(raw*raw));projected=float(np.sum(ref*ref))
            assert abs(total-original['source_energy']['total'])<=1e-10*max(total,1)
            assert abs(projected-original['projected_energy']['total'])<=1e-10*max(projected,1)
            rel=float(np.linalg.norm(z-ref)/max(float(np.linalg.norm(ref)),1e-300))
            zc=z-z.mean(0);rc=ref-ref.mean(0)
            centered=float(np.linalg.norm(zc-rc)/max(float(np.linalg.norm(rc)),1e-300))
            den=np.linalg.norm(z,axis=1)*np.linalg.norm(ref,axis=1);valid=den>0
            cosine=float(np.mean(np.sum(z[valid]*ref[valid],axis=1)/den[valid])) if valid.any() else None
            fraction=float(np.sum((ref@inv)*ref)/max(total,1e-300))
            rows.append(dict(id=r['id'],boundary=target['boundary'],positions=n,relative_RMS=rel,
                centered_relative_RMS=centered,mean_position_cosine=cosine,
                shadow_to_source_coordinate_norm=float(np.linalg.norm(z)/max(float(np.linalg.norm(ref)),1e-300)),
                source_LS_projection_energy_fraction=fraction,
                source_total_energy=total,source_coordinate_energy=projected))
    check();guard()
    summary=[]
    for boundary in (0,4,8,12,16,20,24):
        items=[r for r in rows if r['boundary']==boundary];assert len(items)==24
        summary.append(dict(boundary=boundary,cases=24,
            relative_RMS_mean=math.fsum(r['relative_RMS'] for r in items)/24,
            relative_RMS_worst=max(r['relative_RMS'] for r in items),
            centered_relative_RMS_mean=math.fsum(r['centered_relative_RMS'] for r in items)/24,
            mean_position_cosine=math.fsum(r['mean_position_cosine'] for r in items)/24,
            source_LS_projection_energy_fraction_mean=math.fsum(r['source_LS_projection_energy_fraction'] for r in items)/24))
    result=dict(schema='PROJECTED_SHADOW_TRAJECTORY_GEOMETRY_V1',source=extent(__file__),audit=extent(a.audit),
        source_capture=extent(source_path),source_prior_audit=extent(source_audit),rows=rows,summary=summary,
        all168_matched_streams_before_after_exact=True,seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
        scope='Stored source BF16 full-prefill residuals/qualified rounded P coordinates versus new F32 shadow. '
            'Least-squares energy uses previously rounded/qualified source coordinates and actual inverse Gram: '
            'approximate energy observation, not exact capacity floor. No forward/fit/head/native/GPU replay; '
            'first observed boundary drift is not sole causal attribution or general impossibility.')
    write(a.out,result);print(json.dumps(dict(summary=summary,seconds=result['seconds'],OS_peak=result['OS_peak'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--audit-sha',required=True)
    p.add_argument('--out',type=Path,required=True);main(p.parse_args())
