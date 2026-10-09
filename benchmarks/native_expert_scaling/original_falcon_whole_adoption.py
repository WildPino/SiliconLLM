"""Stored-only aggregate/cost adjudication; no model, native or GPU calls."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_falcon_usability import sha,write


def main(a):
    raw=DOC/'original_falcon_whole_finish_result_20261009.json';terminal=raw.with_suffix('.terminal.json')
    r=json.loads(raw.read_bytes());t=json.loads(terminal.read_bytes())
    assert t['exit_code']==0 and sha(raw)==t['result_sha256'] and t['resource_gates']
    assert r['new_updates']==0 and r['inherited_updates']==24 and r['final_counter']==25
    binding=DOC/'original_falcon_whole_finish_binding_20261009.json';b=json.loads(binding.read_bytes())
    assert sha(binding)==t['binding_sha256']
    receipts=b['inputs']+t['output_files']
    for item in receipts:
        p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],p
    errors=[]
    for stage in ('before','after'):
        rows=r[stage];assert len(rows)==48 and len({i['id'] for i in rows})==48
        for row in rows:
            assert row['labels']==len(row['KL_per_label']) and all(math.isfinite(v) for v in row['KL_per_label'])
            assert abs(row['KL']-sum(row['KL_per_label'])/row['labels'])<1e-10
            assert abs(row['disagreement_rate']-row['disagreement']/row['labels'])<1e-12
        for split in ('FIT','DEV'):
            values=[i for i in rows if i['split']==split];assert len(values)==24
            agg=r[stage+'_aggregates'][split]
            for key,val in dict(case_KL=sum(i['KL'] for i in values)/24,
                case_disagreement=sum(i['disagreement_rate'] for i in values)/24,
                label_KL=sum(sum(i['KL_per_label']) for i in values)/sum(i['labels'] for i in values),
                label_disagreement=sum(i['disagreement'] for i in values)/sum(i['labels'] for i in values)).items():
                errors.append(abs(val-agg[key]));assert abs(val-agg[key])<1e-10,(stage,split,key)
            assert len(agg['domains'])==12
            for domain,ag in agg['domains'].items():
                v=[i for i in values if i['domain']==domain];assert len(v)==2
                assert abs(sum(i['KL'] for i in v)/2-ag['case_KL'])<1e-10
                assert abs(sum(i['disagreement_rate'] for i in v)/2-ag['case_disagreement'])<1e-12
    before=r['before_aggregates']['DEV'];after=r['after_aggregates']['DEV'];g=b['gates']
    flags=dict(absolute_DEV_KL=after['case_KL']<=g['DEV_case_KL'],absolute_DEV_disagreement=after['case_disagreement']<=g['DEV_case_disagreement'],
        all_domain_KL=all(v['case_KL']<=g['domain_case_KL'] for v in after['domains'].values()),
        all_domain_disagreement=all(v['case_disagreement']<=g['domain_case_disagreement'] for v in after['domains'].values()),
        relative_DEV_KL=after['case_KL']<=g['relative_DEV_KL']*before['case_KL'],
        relative_DEV_disagreement=after['case_disagreement']<=g['relative_DEV_disagreement']*before['case_disagreement'])
    assert flags==r['gates']
    expected=Path(b['expected_witness']['path']).read_bytes()
    witnesses=[i for i in t['output_files'] if i['path'].endswith('.witness')];assert len(witnesses)==48
    assert all(Path(i['path']).read_bytes()==expected for i in witnesses)
    oldfailure=json.loads(Path(b['old_failure']).read_bytes());combined=oldfailure['elapsed_seconds']+t['elapsed_seconds']
    assert combined<=3600 and len(r['children'])==48 and len(r['inherited_children'])==47
    for item in receipts:
        p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],p
    value=dict(r,schema='ORIGINAL_FALCON_WHOLE_RECOVERY_ADOPTED_RESULT_V1',combined_family_seconds=combined,
        raw_original_family_plus_finish_worker_seconds=r['combined_family_seconds'],finish_family_seconds=t['elapsed_seconds'],
        raw_result_sha256=sha(raw),metadata_adoption=True,aggregate_max_error=max(errors),adjudication_inputs=[
            dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)) for p in (Path(__file__),raw,terminal,binding,Path(b['old_failure']))],
        verified_recorded_inputs=len(b['inputs']),verified_finish_outputs=len(t['output_files']),
        adoption_new_optimizer_updates=0,adoption_new_predictions=0,adoption_GPU_calls=0,
        resource_scope='Original failure records retain missing GPU/launcher peaks;completion held terminal validates its own CPU family.',
        cost_correction='Raw combined_family_seconds includes only finish worker;canonical value uses both held family elapsed times,including launch/input/output hash overhead.')
    write(a.out,value)
    print(json.dumps(dict(result=str(a.out),sha256=sha(a.out),combined_seconds=combined,gates=flags)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
