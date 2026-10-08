"""Freeze actual source extents, coefficients, anchors and finite curvature test."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once
from chatbot_directional_binding import entry,receipt
from chatbot_null_prior_kernel_binding import checked


def main(args):
    assert not args.out.exists();doc=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';code=ROOT/'benchmarks/native_expert_scaling'
    runtime_path=doc/'chatbot_null_prior_kernel_binding_20261008.json'
    assert sha(runtime_path)=='2b5b63bb4b801066e74362e471c92f0ab203d407194fce7a3c473a9e3b042ee0'
    prior=json.loads(runtime_path.read_bytes())
    kernel_path=doc/'chatbot_null_prior_kernel_20261008.json'
    kernel,kterm,kt=checked(kernel_path,'7b84d405b1c86c65a9d3babaa0b898ca8264e5d67eefd789f997b47fb94c8fee')
    audit_path=doc/'chatbot_null_prior_kernel_audit_20261008.json'
    _,aterm,_=checked(audit_path,'7b73c93c93e94d4d0cab1d68a15285b5b75cc1ea74ff2ec77781fe148339f23d')
    projector_paths={n+'_path':str(receipt(kernel['outputs'][n]['path'],kt)) for n in ('P','Pi_N')}
    plan_path=doc/'chatbot_directional_plan_20261008.json'
    assert sha(plan_path)=='bcbb9a5921ece474870375c438c42a474b0f115b85f9e3a35366cf1916b165dc'
    correction=doc/'chatbot_directional_plan_provenance_20261008.json'
    corr=json.loads(correction.read_bytes());assert corr['result_sha256']==sha(plan_path) and corr['actual_source_freeze']=='ec197f9efdf1b38dfdc6f8a958e2203a354ee6c8'
    directional=doc/'chatbot_directional_jacobian_20261008.json'
    _,jterm,jt=checked(directional,'576f0c5a8495456fe6d4dac9b0055e8a32d964546029d00ddc4f89c337356b22')
    source_j=receipt(ROOT/'results/native_expert_scaling/chatbot_directional_jacobian_20261008/source_J_F64.npy',jt)
    compile_path=doc/'chatbot_coupled_compile_20261008.json'
    compiled,cterm,ct=checked(compile_path,'9569e690c5975d01e9b9f9e5bc454be64563e93a2fc6c9efa93648055ab93f66')
    core_paths={n:str(receipt(compiled['output_arrays']['shared_'+n+'_BF16']['path'],ct)) for n in ('g','u','b')}
    original_binding=doc/'chatbot_directional_jacobian_binding_20261008.json'
    source=Path(json.loads(original_binding.read_bytes())['source_weights'])
    census_path=doc/'chatbot_qwen_operator_census_20261007.json';census=json.loads(census_path.read_bytes())
    header_spec=next(v for v in census['inputs'] if Path(v['path']).resolve()==source.resolve())
    assert source.stat().st_size==header_spec['file_bytes'] and header_spec['offset']==0
    spec=dict(path=str(source),resolved_path=str(source.resolve()),file_bytes=source.stat().st_size,
        header_bytes=header_spec['bytes'],header_sha256=header_spec['sha256'],tensors=[],whole_file_SHA_recomputed=False)
    with source.open('rb') as stream:
        header=stream.read(spec['header_bytes']);assert hashlib.sha256(header).hexdigest()==spec['header_sha256']
        assert len(header)==struct.unpack('<Q',header[:8])[0]+8;fields=json.loads(header[8:])
        for name,shape in (('gate_proj',[4864,896]),('up_proj',[4864,896]),('down_proj',[896,4864])):
            full='model.layers.12.mlp.'+name+'.weight';field=fields[full]
            assert field['dtype']=='BF16' and field['shape']==shape and 'model.layers.12.mlp.'+name+'.bias' not in fields
            begin,end=field['data_offsets'];assert end-begin==2*math.prod(shape)
            stream.seek(len(header)+begin);raw=stream.read(end-begin);assert len(raw)==end-begin
            spec['tensors'].append(dict(name=full,shape=shape,offset=len(header)+begin,bytes=end-begin,sha256=hashlib.sha256(raw).hexdigest()))
    radius_path=doc/'chatbot_kernel_20261008.json';assert sha(radius_path)=='d63a24ec90d5c382f9c00655afcecc9f70ca1d43a1e1d876ac744afbd002aa9c'
    worker=code/'chatbot_null_curvature.py'
    paths=[runtime_path,prior['python'],Path(prior['python']).with_name('python312.dll'),kernel_path,kterm,audit_path,aterm,
        *projector_paths.values(),plan_path,plan_path.with_suffix('.terminal.json'),correction,directional,jterm,source_j,
        compile_path,cterm,*core_paths.values(),original_binding,census_path,radius_path,
        worker,Path(__file__).resolve(),code/'chatbot_null_prior_kernel_binding.py',code/'chatbot_interaction_launch.py',code/'chatbot_directional_binding.py',code/'chatbot_directional_launch.py',
        doc/'CHATBOT_NULL_CURVATURE_NEXT_20261008.md',doc/'CHATBOT_NULL_CURVATURE_PROTOCOL_20261008.md']
    b=dict(schema='QWEN_DIRECTIONAL_BINDING_V1',python=prior['python'],plan_path=str(plan_path),source_J_path=str(source_j),
        source_slices=spec,core_paths=core_paths,**projector_paths,radius_report_path=str(radius_path),
        inputs=[entry(p) for p in dict.fromkeys(paths)],capture_runtime_roots=prior['capture_runtime_roots'],interaction_view=prior['interaction_view'],
        job=dict(name='null_curvature',worker_path=str(worker),result_schema='QWEN_NULL_CURVATURE_RESULT_V1',
            accepted_decisions=['NULL_CURVATURE_SUPPORTED_PRICE_ONE_CHANGED_REPRESENTATION','CLOSE_NULL_CURVATURE_QUALIFICATION','NULL_CURVATURE_UNSUPPORTED_REASSESS_COVERAGE'],
            limits=dict(worker_seconds=180,family_seconds=300,OS_bytes=4<<30,output_bytes=16<<20,log_bytes=2<<20)))
    assert len(b['capture_runtime_roots'])==5
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']),source_header_and_payload_bytes=spec['header_bytes']+sum(v['bytes'] for v in spec['tensors']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
