"""Refine prospective completion custody after observing actual seal IO cost."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
sys.path.insert(0,str(B))
from original_packed_capacity import extent,write,sha


def main(a):
    base=json.loads(a.base.read_bytes());assert base['schema']=='ORIGINAL_JOINT_HISTORY_RECOVERY_FINISH_BINDING_V1'
    parent=Path(base['parent_directory']);pb=json.loads(Path(base['parent_binding']['path']).read_bytes())
    omitted_source={pb['source_state']['path'],pb['source_packed']['path'],pb['before_GPU_heads']['path'],pb['before_GPU_routes']['path']}
    additional=[extent(a.base),extent(Path(__file__)),extent(ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925/ORIGINAL_JOINT_HISTORY_RECOVERY_FINISH_INPUT_CUSTODY_20261010.md')]
    all_inputs={i['path']:i for i in base['inputs']+additional};consumed=[]
    for item in all_inputs.values():
        path=Path(item['path']);name=path.name
        if path.parent==parent and not (name in ('B.actual051.pt','B.actual051.packed','A.arm_result.json','first_fault.json','initial27.auxiliary_DEV.json')
            or name.startswith('B.actual') and name.endswith('.milestone.json')
            or name.startswith('B.final.') and name.endswith('.aux.json')):continue
        if item['path'] in omitted_source:continue
        consumed.append(item)
    assert base['source_state'] in consumed and base['packed'] in consumed
    assert all(i in consumed for i in base['adopted_B_auxiliary'])
    assert len(consumed)<len(all_inputs)
    base.update(inputs=consumed,adjudication_inputs=list(all_inputs.values()),prospective_base_binding=extent(a.base),
        input_custody='Whole parent seal before launch; exact completion-consumed inputs before/after; whole seal at independent stored audit. '
            'Avoid repeated unused75GB IO; original scientific thresholds/call count/1200s limits unchanged.',
        measured_seal_read_bytes_s=100478662)
    write(a.out,base);print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),consumed_inputs=len(consumed),adjudication_inputs=len(all_inputs))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
