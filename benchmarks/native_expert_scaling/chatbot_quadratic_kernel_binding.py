"""Bind saved actual factors and only necessary original affine geometry bytes."""
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
    rbpath=doc/'chatbot_source_quadratic_binding_20261008.json';assert sha(rbpath)=='1032f072be75480ffb6fdf9b6db6e8db12cacbbe287c5fe7684b9f0dc22c9b71';rb=json.loads(rbpath.read_bytes())
    factor_path=doc/'chatbot_source_quadratic_20261008.json';fr,fterm,ft=checked(factor_path,'85577acc7a24677cde7b489974b932c97b7ee8244061c13fb41f54fd16288389')
    used=[receipt(fr['outputs'][n]['path'],ft) for n in ('R_BF16','T_BF16')]
    audit=doc/'chatbot_source_quadratic_audit_repair1_20261008.json';ar,aterm,_=checked(audit,'9c9852a00dabcc71276f5e61cad7fa06c13f8d9e1a0f85e4740b6a47ef2c228f')
    assert ar['decision']=='SOURCE_QUADRATIC_FACTORS_INDEPENDENTLY_VERIFIED'
    closure=doc/'chatbot_source_quadratic_terminal_20261008.json';assert sha(closure)=='2d878f9a2ce2f8b621c76fecfed5cdcd15e3062d5cc8b5a523c634d8f100f678'
    old_path=doc/'chatbot_kernel_20261008.json';old,oterm,ot=checked(old_path,'d63a24ec90d5c382f9c00655afcecc9f70ca1d43a1e1d876ac744afbd002aa9c')
    used.extend(receipt(old['outputs'][n]['path'],ot) for n in ('FIT_x','FIT_weights','FIT_W','mu','G'))
    oldaudit=doc/'chatbot_kernel_audit_20261008.json';_,oaterm,_=checked(oldaudit,'ef4bee138ac55c1ffc3f8f6ad0cbd9fbeed22226e10c3fab9bfa76c32c0ced73')
    worker=code/'chatbot_quadratic_kernel.py'
    paths=[rbpath,rb['python'],Path(rb['python']).with_name('python312.dll'),factor_path,fterm,audit,aterm,closure,old_path,oterm,oldaudit,oaterm,*used,
        worker,Path(__file__).resolve(),code/'chatbot_null_prior_kernel_binding.py',code/'chatbot_directional_binding.py',code/'chatbot_interaction_launch.py',code/'chatbot_directional_launch.py',
        doc/'CHATBOT_QUADRATIC_KERNEL_NEXT_20261008.md',doc/'CHATBOT_CURVATURE_PIPELINE_BUDGET_20261008.md',doc/'CHATBOT_QUADRATIC_KERNEL_PROTOCOL_20261008.md']
    b=dict(schema=rb['schema'],python=rb['python'],old_kernel_path=str(old_path),factor_path=str(factor_path),inputs=[entry(p) for p in dict.fromkeys(paths)],
        capture_runtime_roots=rb['capture_runtime_roots'],interaction_view=rb['interaction_view'],
        job=dict(name='quadratic_kernel',worker_path=str(worker),result_schema='QWEN_QUADRATIC_KERNEL_RESULT_V1',
            accepted_decisions=['ELIGIBLE_FOR_ONE_QUADRATIC_CONVEX_COMPILER','CLOSE_QUADRATIC_KERNEL_PREREQUISITE'],
            limits=dict(worker_seconds=180,family_seconds=300,OS_bytes=4<<30,output_bytes=512<<20,log_bytes=2<<20)))
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
