"""Freeze NEW conversion ledger or FIRST mechanical adapter qualification."""
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
    priorpath=doc/'chatbot_quadratic_fit_binding_20261008.json';assert sha(priorpath)=='41fce9b5cfb53524a90bb146bbde0a004a5eddeb1a76b4edad719513d704b138';prior=json.loads(priorpath.read_bytes())
    spec=code/'chatbot_compact_spec.json';assert sha(spec)=='a5850df0172d9f9152e8391a3588ed5cfe9fb184d049955a161e28b4786cd9d0'
    census=doc/'chatbot_qwen_operator_census_20261007.json';assert sha(census)=='a4293870cb38706fc60d94562d9579653ddff1a1387c366c307a56b4b262bbc2'
    budget=doc/'chatbot_compact_geometry_budget_20261007.json';assert sha(budget)=='4aad875680fc2fb0ea35e275d395c71a8542ced41ffca6e70fc613bb6dc5cd37'
    paths=[priorpath,prior['python'],Path(prior['python']).with_name('python312.dll'),spec,
        Path(__file__).resolve(),code/'chatbot_whole_transfer_model.py',code/'chatbot_compact_geometry.py',code/'chatbot_interaction_launch.py',
        code/'chatbot_directional_binding.py',code/'chatbot_directional_launch.py',code/'chatbot_null_prior_kernel_binding.py',
        doc/'CHATBOT_WHOLE_TRANSFER_PROTOCOL_20261008.md',doc/'CHATBOT_PIPELINE_TRANSFER_NEXT_20261008.md']
    fields={}
    if args.stage=='inventory':
        worker=code/'chatbot_whole_transfer_inventory.py';paths.extend([census,budget,worker])
        fields.update(census_path=str(census),old_budget_path=str(budget));runtime=prior['capture_runtime_roots'];view=prior['interaction_view']
        job=dict(name='whole_inventory',worker_path=str(worker),result_schema='QWEN_WHOLE_TRANSFER_INVENTORY_RESULT_V1',
            accepted_decisions=['E16_COMPLETE_IMPLEMENTATION_PREREQUISITE_E160_RESIDENT_ADAM_REJECTED'],
            limits=dict(worker_seconds=45,family_seconds=180,OS_bytes=512<<20,output_bytes=1<<20,log_bytes=2<<20))
    else:
        baselinepath=doc/'chatbot_joint_baseline_binding_20261007.json';assert sha(baselinepath)=='a3265dbeceaf7de82a903bf7735beba88628d7814be34fb1a46d459424b00b5b'
        base=json.loads(baselinepath.read_bytes())
        structural=doc/'chatbot_structural_fit_20261008.json';_,terminalpath,terminal=checked(structural,'b4dfe3a8cea4d76a43e66ff71f38dfd5fbcbb7163effbcad857d06fa1f5fc7c1')
        saved=ROOT/'results/native_expert_scaling/chatbot_structural_fit_20261008/E16.final_F32.pt';receipt(saved,terminal)
        assert sha(saved)=='8e04c2d59c61b5d822051ad04de107a808b0e2e5d3984b7599d485de5a5c3e32'
        fields['saved_block_path']=str(saved)
        prefixes=('numpy','psutil','torch','functorch','sympy','mpmath','networkx','typing_extensions','packaging','filelock','jinja2','markupsafe','fsspec')
        runtime=[v for v in base['capture_runtime_roots'] if v['view_name'].startswith(prefixes)]
        names={v['view_name'] for v in runtime};assert {'torch','torchgen','functorch','numpy','numpy.libs','psutil','typing_extensions.py'}<=names
        view=[v for v in base['interaction_view'] if v['view_name'].startswith(prefixes)]
        worker=code/'chatbot_whole_interface_probe.py'
        paths.extend([baselinepath,structural,terminalpath,saved,worker])
        job=dict(name='whole_interface',worker_path=str(worker),result_schema='QWEN_WHOLE_INTERFACE_PROBE_RESULT_V1',
            accepted_decisions=['WHOLE_INTERFACE_AND_OUTPUT_GRADIENT_MECHANICS_QUALIFIED'],
            limits=dict(worker_seconds=180,family_seconds=540,OS_bytes=4<<30,output_bytes=1<<20,log_bytes=2<<20))
    b=dict(schema=prior['schema'],python=prior['python'],spec_path=str(spec),**fields,
        inputs=[entry(p) for p in dict.fromkeys(paths)],capture_runtime_roots=runtime,interaction_view=view,job=job)
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']),runtime_roots=len(runtime))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('inventory','interface'),required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
