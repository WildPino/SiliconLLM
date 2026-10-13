"""Bind a full block audit only after original audit is authoritatively terminal."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
B = ROOT/'benchmarks/native_expert_scaling'
sys.path[:0] = [str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write


def main(args):
    import psutil
    tp = DOC/'ffn_remainder_oracle_adjudication_20261013.terminal.json'
    t = json.loads(tp.read_bytes())
    assert t['exit_code'] == 1
    if psutil.pid_exists(t['pid']): assert abs(psutil.Process(t['pid']).create_time()-t['creation_time']) > .001
    rp = DOC/'ffn_remainder_oracle_result_20261013.json'; pt = rp.with_suffix('.terminal.json')
    r,p = [json.loads(x.read_bytes()) for x in (rp,pt)]
    assert p['exit_code'] == 0 and p['result'] == extent(rp)
    assert len(r['records']) == 720 and extent(r['binding']['path']) == r['binding']
    out = DOC/'ffn_remainder_oracle_audit_v2_binding_20261013.json'; assert not out.exists()
    paths = [Path(__file__),B/'ffn_remainder_oracle_audit_v2.py',B/'ffn_remainder_oracle_close_v2.py',
             DOC/'FFN_REMAINDER_ORACLE_AUDIT_LAYOUT_20261013.md',tp,rp,pt,Path(r['binding']['path'])]
    inputs = [extent(x) for x in paths]
    binding = dict(schema='FFN_REMAINDER_ORACLE_AUDIT_V2_BINDING_V1',freeze=args.freeze,
        source=extent(Path(__file__)),auditor=extent(paths[1]),inputs=inputs,
        report=extent(rp),producer_terminal=extent(pt),original_terminal=extent(tp),
        original_process_absent=True,candidate_changed=False,quality_gates_changed=False,
        full_audit_predictions=720,projection='All4608 source columns partitioned into256-column blocks; '
        'unchanged independent routing,chain rule,remainder,literal IDs,case/domain/metric/hash checks.')
    write(out,binding); print(json.dumps(dict(binding=extent(out),original_terminal=extent(tp))),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--freeze',required=True); args = p.parse_args()
    import psutil
    start = time.monotonic(); proc = psutil.Process()
    out = DOC/'ffn_remainder_oracle_audit_v2_binding_20261013.json'
    receipt = dict(schema='FFN_REMAINDER_ORACLE_AUDIT_V2_PREPARATION_TERMINAL_V1',command=sys.orig_argv,
                   pid=proc.pid,creation_time=proc.create_time(),freeze=args.freeze)
    try: main(args); receipt.update(exit_code=0,error=None,result=extent(out))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(out.with_suffix('.preparation_terminal.json'),receipt)
