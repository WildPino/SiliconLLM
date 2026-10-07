"""Adopt byte-qualified original forward prefixes; never recompute source outputs.

Full-generation completion is independent of calibration operand eligibility.
Keep the original ALL48/300s producer failure explicitly false. This tool only
checks saved bytes, their shape/IDs/journals and the original source provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_capture_cases import manifest


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def qualify(case):
    path=Path(case['binary_path']);frames=case['frames']
    assert frames and [f['step'] for f in frames]==list(range(len(frames)))
    assert [f['next_id'] for f in frames]==case['generated_ids']
    assert frames[0]['input_ids']==case['prompt_ids']
    rows=0;cursor=24
    with path.open('rb') as source:
        assert struct.unpack('<8s4I',source.read(24))==(b'QWCAP001',24,896,2,0)
        for step,frame in enumerate(frames):
            ids=case['prompt_ids'] if step==0 else [case['generated_ids'][step-1]]
            assert frame['input_ids']==ids and frame['shape']==[len(ids),24,2,896]
            assert frame['dtype']=='<u2 BF16 bits' and frame['order']=='C'
            assert frame['offset']==cursor and frame['bytes']==len(ids)*24*2*896*2
            raw=source.read(frame['bytes']);assert len(raw)==frame['bytes'] and hashlib.sha256(raw).hexdigest()==frame['sha256']
            rows+=len(ids);cursor+=len(raw)
        assert not source.read(1),'unqualified extra binary tail'
    journal=Path(case['journal_path'])
    assert [json.loads(line) for line in journal.read_bytes().splitlines()]==frames
    assert rows==len(case['prompt_ids'])+len(case['generated_ids'])-1
    assert hashlib.sha256(case['rendered_text'].encode('utf8')).hexdigest()==case['rendered_UTF8_SHA256']
    assert hashlib.sha256(struct.pack('<'+'I'*len(case['prompt_ids']),*case['prompt_ids'])).hexdigest()==case['prompt_U32LE_SHA256']
    return dict(binary_SHA256=sha(path),binary_bytes=path.stat().st_size,journal_SHA256=sha(journal),captured_rows=rows)


def main(args):
    assert not args.out.exists() and sha(args.report)==args.report_sha
    r=json.loads(args.report.read_bytes());spec=manifest();by_id={c['id']:c for c in spec['cases']}
    assert r['gates']['frozen_input_files'] and r['gates']['pure_original_source_and_tied_head'] and r['gates']['unrelated_Arrow_dataset_packages_absent']
    cases=list(r['conversations'])
    if 'pending_conversation' in r:
        pending=dict(r['pending_conversation']);identifier=pending['id']
        assert pending['frames'],'no captured pending forward'
        pending.update(binary_path=str((args.directory/(identifier+'.bf16.bin')).resolve()),
            journal_path=str((args.directory/(identifier+'.frames.jsonl')).resolve()),
            acquisition_status='complete_saved_forward_prefix_of_failed_generation',
            complete_requested_generation=False,termination='original300s_resource_fault_NOT_EOS_OR_MAX96',
            accepted_generated_ids=pending['generated_ids'])
        cases.append(pending)
    assert len({c['id'] for c in cases})==len(cases)
    for case in cases:
        original=by_id[case['id']]
        assert all(case[k]==original[k] for k in ('split','category','mode','messages'))
        receipt=qualify(case)
        for key in ('binary_SHA256','binary_bytes','journal_SHA256','captured_rows'):
            if key in case:assert case[key]==receipt[key]
        case.update(receipt)
        case.setdefault('acquisition_status','original_complete_generation')
        case.setdefault('complete_requested_generation',True)
    out=dict(schema='QWEN_ORIGINAL_FORWARD_PREFIX_ADOPTION_V1',source_report=str(args.report.resolve()),
        source_report_SHA256=args.report_sha,cases=cases,all_saved_frame_bytes_and_alignment_qualified=True,
        new_source_forwards=0,new_source_responses=0,original_ALL48_300s_capture_qualified=False,
        complete_requested_generations=sum(c['complete_requested_generation'] for c in cases),
        calibration_case_coverage=len(cases),fit_cases=sum(c['split']=='fit' for c in cases),
        development_cases=sum(c['split']=='development' for c in cases),
        missing_case_ids=[c['id'] for c in spec['cases'] if c['id'] not in {v['id'] for v in cases}],
        decision='RETAINED_ORIGINAL_OPERANDS_ELIGIBLE_FOR_CALIBRATION_NOT_FULL_GENERATION_OR_FINAL_QUALITY')
    raw=(json.dumps(out,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode('utf8')
    assert len(raw)<=2<<20
    with args.out.open('xb') as f:f.write(raw)
    print(json.dumps({k:out[k] for k in ('calibration_case_coverage','fit_cases','development_cases','complete_requested_generations','missing_case_ids','new_source_forwards')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--report-sha',required=True)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
