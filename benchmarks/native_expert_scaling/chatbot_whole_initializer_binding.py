"""Freeze actual source/capture/reused layer12 inputs for init or FIRST audit."""
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
    oldpath=doc/'chatbot_joint_baseline_binding_20261007.json'
    assert sha(oldpath)=='a3265dbeceaf7de82a903bf7735beba88628d7814be34fb1a46d459424b00b5b';old=json.loads(oldpath.read_bytes())
    adoption=Path(old['adoption_path']);assert sha(adoption)=='1b54b8fd3f82f93caa35c1aa05b26abfbce555f05165b1428dc4a14536cad7d9'
    adopted=json.loads(adoption.read_bytes());source=Path(old['source_weights'])
    source_entry=next(v for v in old['inputs'] if Path(v['path'])==source)
    assert sha(source)==source_entry['sha256'] and source.stat().st_size==source_entry['bytes']
    pilot=doc/'chatbot_joint_pilot_20261007.json';pilot_term=pilot.with_suffix('.terminal.json');pt=json.loads(pilot_term.read_bytes())
    assert pt['actual_worker_exit_code']==0 and sha(pilot)==pt['result_sha256']
    base=doc/'chatbot_joint_baseline_20261007.json';base_term=base.with_suffix('.terminal.json');bt=json.loads(base_term.read_bytes())
    assert bt['actual_worker_exit_code']==0 and sha(base)==bt['result_sha256']
    old_initial=receipt(ROOT/'results/native_expert_scaling/chatbot_joint_baseline_20261007/E16.initial.pt',bt)
    old_indices=receipt(old_initial.with_name('E16.initializer.json'),bt)
    assert sha(old_initial)=='d85c4b292fcc54d7ad8bd8777e5158354b6d043da62627ed4c04dc3de856822d'
    assert sha(old_indices)=='8e1c0409a85c7cc3840b7dcb972a22caa7620dfd6466890fe2cb098306bf1bd0'
    old_geometry=receipt(old['saved_geometry'],pt);old_routes=receipt(old['saved_E16_routes'],pt)
    old_exposure=receipt(old_routes.with_name('E16.exposure.json'),pt)
    spec=code/'chatbot_compact_spec.json';assert sha(spec)=='a5850df0172d9f9152e8391a3588ed5cfe9fb184d049955a161e28b4786cd9d0'
    paths=[oldpath,old['python'],Path(old['python']).with_name('python312.dll'),adoption,source,spec,
        pilot,pilot_term,base,base_term,old_initial,old_indices,old_geometry,old_routes,old_exposure,
        Path(__file__).resolve(),code/'chatbot_interaction_launch.py',code/'chatbot_directional_binding.py',code/'chatbot_directional_launch.py',
        code/'chatbot_null_prior_kernel_binding.py',code/'chatbot_compact_geometry.py',code/'chatbot_joint_pilot.py',
        doc/'CHATBOT_WHOLE_INITIALIZER_NEXT_20261008.md',doc/'CHATBOT_WHOLE_INITIALIZER_PROTOCOL_20261008.md']
    for case in adopted['cases']:
        for key in ('binary','journal'):
            path=Path(case[key+'_path']);assert sha(path)==case[key+'_SHA256'];paths.append(path)
    prefixes=('numpy','psutil','torch','functorch','sympy','mpmath','networkx','typing_extensions','packaging','filelock','jinja2','markupsafe','fsspec','safetensors')
    runtime=[v for v in old['capture_runtime_roots'] if v['view_name'].startswith(prefixes)]
    view=[v for v in old['interaction_view'] if v['view_name'].startswith(prefixes)]
    b=dict(schema='QWEN_DIRECTIONAL_BINDING_V1',python=old['python'],source_weights=str(source),adoption_path=str(adoption),
        spec_path=str(spec),old_initial_path=str(old_initial),old_indices_path=str(old_indices),
        old_geometry_path=str(old_geometry),old_routes_path=str(old_routes),old_exposure_path=str(old_exposure),
        capture_runtime_roots=runtime,interaction_view=view)
    if args.stage=='initialize':
        worker=code/'chatbot_whole_initializer.py';name='whole_initializer';schema='QWEN_WHOLE_INITIALIZER_RESULT_V1'
        decisions=['ALL24_SOURCE_INITIALIZED_REQUIRE_FIRST_INDEPENDENT_AUDIT','CLOSE_ALL24_INITIALIZER_ON_LAYER_SUPPORT']
        limits=dict(worker_seconds=600,family_seconds=900,OS_bytes=16<<30,GPU_allocated_bytes=8<<30,GPU_reserved_bytes=9<<30,output_bytes=1<<30,log_bytes=2<<20)
    else:
        assert args.result is not None and args.result_sha
        raw,term,terminal=checked(args.result,args.result_sha)
        assert raw['schema']=='QWEN_WHOLE_INITIALIZER_RESULT_V1' and all(raw['procedure_gates'].values())
        assert raw['decision'] in ('ALL24_SOURCE_INITIALIZED_REQUIRE_FIRST_INDEPENDENT_AUDIT','CLOSE_ALL24_INITIALIZER_ON_LAYER_SUPPORT')
        original=Path(terminal['command'][terminal['command'].index('--binding')+1])
        b['initializer_result_path']=str(args.result.resolve());paths.extend([args.result,term,original])
        for item in terminal['output_manifest']:paths.append(receipt(item['path'],terminal))
        worker=code/'chatbot_whole_initializer_audit.py';name='whole_initializer_audit';schema='QWEN_WHOLE_INITIALIZER_AUDIT_RESULT_V1'
        decisions=['ALL24_SOURCE_INITIALIZER_INDEPENDENTLY_VERIFIED','PARTIAL_INITIALIZER_SUPPORT_STOP_INDEPENDENTLY_VERIFIED']
        limits=dict(worker_seconds=300,family_seconds=600,OS_bytes=8<<30,output_bytes=1<<20,log_bytes=2<<20)
    paths.append(worker);b['inputs']=[entry(p) for p in dict.fromkeys(paths)]
    b['job']=dict(name=name,worker_path=str(worker),result_schema=schema,accepted_decisions=decisions,limits=limits)
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']),runtime_roots=len(runtime))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('initialize','audit'),required=True)
    p.add_argument('--result',type=Path);p.add_argument('--result-sha');p.add_argument('--out',type=Path,required=True);main(p.parse_args())
