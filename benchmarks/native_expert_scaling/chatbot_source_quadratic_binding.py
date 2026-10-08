"""Bind FIRST factor extraction to only actually used sealed source operands."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once
from chatbot_directional_binding import entry,receipt
from chatbot_null_prior_kernel_binding import checked


def main(args):
    assert not args.out.exists();doc=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';code=ROOT/'benchmarks/native_expert_scaling'
    cbpath=doc/'chatbot_null_curvature_binding_20261008.json'
    assert sha(cbpath)=='5048f5070460ab79a7284670497c0ac0ba93fe0bc2ab2dd8289a100cca8429e4';cb=json.loads(cbpath.read_bytes())
    raw_path=doc/'chatbot_null_curvature_repair1_20261008.json'
    raw,term,rt=checked(raw_path,'466ccfc6f0e6308b9738fe3a9f6f1d0013d746b6934fc5f7483ef292d5c93bbb')
    assert raw['decision']=='NULL_CURVATURE_SUPPORTED_PRICE_ONE_CHANGED_REPRESENTATION' and all(raw['diagnostic_gates'].values())
    audit_path=doc/'chatbot_null_curvature_repair1_audit_20261008.json'
    ar,aterm,_=checked(audit_path,'f0294d84df1c6cfb92c6238ee9c12f79d8f11e400d780815cde5d7a92dbd24e1')
    assert ar['decision']=='SAVED_NULL_CURVATURE_INDEPENDENTLY_VERIFIED'
    closure=doc/'chatbot_null_curvature_repair1_terminal_20261008.json'
    assert sha(closure)=='435dfa47200964b546b94cb193b595aa9dd3b2f56d856dc9f5b8f7f47572cfc4'
    used=[receipt(raw['outputs'][m+'_'+n]['path'],rt) for m in ('source','core') for n in ('gx','ux','Gv','Uv','H')]
    for path in cb['core_paths'].values():
        item=next(i for i in cb['inputs'] if Path(i['path'])==Path(path));assert sha(path)==item['sha256']
    worker=code/'chatbot_source_quadratic.py'
    paths=[cbpath,cb['python'],Path(cb['python']).with_name('python312.dll'),raw_path,term,audit_path,aterm,closure,*used,
        cb['plan_path'],*cb['core_paths'].values(),worker,Path(__file__).resolve(),code/'chatbot_null_curvature.py',code/'chatbot_null_prior_kernel_binding.py',
        code/'chatbot_directional_binding.py',code/'chatbot_interaction_launch.py',code/'chatbot_directional_launch.py',
        doc/'CHATBOT_SOURCE_QUADRATIC_NEXT_20261008.md',doc/'CHATBOT_CURVATURE_PIPELINE_BUDGET_20261008.md',doc/'CHATBOT_SOURCE_QUADRATIC_PROTOCOL_20261008.md']
    b=dict(schema=cb['schema'],python=cb['python'],source_slices=cb['source_slices'],core_paths=cb['core_paths'],plan_path=cb['plan_path'],curvature_path=str(raw_path),
        inputs=[entry(p) for p in dict.fromkeys(paths)],capture_runtime_roots=cb['capture_runtime_roots'],interaction_view=cb['interaction_view'],
        job=dict(name='source_quadratic',worker_path=str(worker),result_schema='QWEN_SOURCE_QUADRATIC_RESULT_V1',
            accepted_decisions=['SOURCE_QUADRATIC_FACTORS_REQUIRE_FIRST_INDEPENDENT_AUDIT'],
            limits=dict(worker_seconds=180,family_seconds=300,OS_bytes=4<<30,output_bytes=64<<20,log_bytes=2<<20)))
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
