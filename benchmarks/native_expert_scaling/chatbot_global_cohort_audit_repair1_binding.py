"""Only repair the failed reader cursor; retain original science and first fault."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once
from chatbot_directional_binding import entry


def main(args):
    assert not args.out.exists();doc=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';code=ROOT/'benchmarks/native_expert_scaling'
    prior=doc/'chatbot_global_cohort_audit_binding_20261008.json';assert sha(prior)=='9477336190584ce062edd6a3323c1a7ddc9e85352d36c511c53687bdabfd3651'
    b=json.loads(prior.read_bytes())
    for item in b['inputs']:
        p=Path(item['path']);assert str(p.resolve())==item['resolved_path'] and p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    rawpath=doc/'chatbot_global_cohort_audit_20261008.failure.json';assert sha(rawpath)=='58e86d105545181761aeedff76f3f0cd677f8bfdff17be640e8f5169cf39bdb4'
    termpath=doc/'chatbot_global_cohort_audit_20261008.launcher_failure.json';assert sha(termpath)=='8327f52ccd6262961e234ab1e6a754f4159036e46f8802dbc28126e7a5bd5502'
    log=doc/'chatbot_global_cohort_audit_20261008.worker.log';assert sha(log)=='c9af3176a36587c05304363cf3a3eb73e3e6561106a126caa6e1e88a5818d737'
    raw=json.loads(rawpath.read_bytes());term=json.loads(termpath.read_bytes())
    assert raw['fault_stage']=='complete_saved_NEW_case_wires' and raw['conversations']==[] and raw['new_original_BF16_full_forwards']==0
    assert term['actual_worker_exit_code']==1 and raw['process_instance']['pid']==term['worker_instance']['pid']
    original=code/'chatbot_global_cohort_audit.py';repair=code/'chatbot_global_cohort_audit_repair1.py'
    anchor="            with Path(current['x_path']).open('rb') as xs,Path(current['logits_path']).open('rb') as ls:"
    before=original.read_text(encoding='utf8');after=repair.read_text(encoding='utf8')
    assert before.count(anchor)==1 and after==before.replace(anchor,anchor+'\n                xs.seek(24);ls.seek(24)')
    b['job']['worker_path']=str(repair)
    paths=[prior,rawpath,termpath,log,repair,Path(__file__).resolve(),doc/'CHATBOT_GLOBAL_COHORT_AUDIT_REPAIR1_20261008.md']
    b['inputs'].extend(entry(p) for p in paths)
    b['unchanged_science_reader_repair']=dict(first_fault_path=str(rawpath),first_fault_SHA256=sha(rawpath),
        difference='Only seek both reopened streams past their already verified24-byte headers',
        original_complete_case_adoptions=0,source_or_student_query_replay=0,original_fault_not_promoted=True)
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
