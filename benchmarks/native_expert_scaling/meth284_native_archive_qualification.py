#!/usr/bin/env python3
"""Frozen direct-archive CPU operators versus actual complete276 reference."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time

import numpy as np
import psutil
import torch
import meth280_complete_i16_fresh_prediction as Q

P,R=Q.P,Q.R
ROOT=P.ROOT;DOC=P.DOC;ART=ROOT/'results/native_expert_scaling'
TASK=DOC/'meth283_complete_i16_piqa_result.json'
TASK_SHA='2ed1e54df9722c32287445eeb2b8d40590742b1893656c374cff6490912a7725'
VECTORS=ROOT/'benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth125_e1280_vectors.bin'
VECTORS_SHA='f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699'
CONTROL=ART/'meth274_i16_shared_input.check.bin'
CONTROL_SHA='0336e15c46e80f8d4d45816525e16cce8f34de44e635c04fac584f12257ca5ba'
FILES=['benchmarks/phase60/engine.c','benchmarks/native_expert_scaling/meth284_complete_core_cpu.c',
       'benchmarks/native_expert_scaling/meth284_source_operator.h',
       'benchmarks/native_expert_scaling/meth284_archive_catalog.h',
       'benchmarks/native_expert_scaling/meth284_archive_catalog.py',
       'benchmarks/native_expert_scaling/meth182_group64_ffn_cpu.c']
RECORD=np.dtype([('base','<f4',(896,)),('parents','<i4',(4,)),('children','<i4',(4,)),
                 ('aliases','<i4',(4,)),('gates','<f4',(4,)),('delta','<f4',(896,))])


def budget(start,device):
    r={'seconds':time.monotonic()-start,'rss_bytes':psutil.Process().memory_info().rss,
       'peak_gpu_bytes':torch.cuda.max_memory_allocated(device)}
    if r['seconds']>720 or r['rss_bytes']>20*(1<<30) or r['peak_gpu_bytes']>10.5*(1<<30):
        raise RuntimeError('METH-284 reference resource stop '+str(r))
    return r


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--exe',type=Path,default=ART/'meth284_complete_core_cpu.exe')
    ap.add_argument('--native',type=Path,default=ART/'meth284_operators.bin')
    ap.add_argument('--negative',type=Path,default=ART/'meth284_wrong_alias.bin')
    ap.add_argument('--reference',type=Path,default=ART/'meth284_gpu_conditional_reference.npz')
    args=ap.parse_args()
    assert all(not p.exists() for p in (args.out,args.out.with_suffix('.partial.json'),args.out.with_suffix('.failure.json'),args.native,args.negative,args.reference))
    start=time.monotonic();stage='bindings';layers=[];cpu_runs={}
    def partial():
        args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'layers':layers,
            'cpu_runs':cpu_runs,'seconds':time.monotonic()-start},indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
    try:
        for path,sha in ((TASK,TASK_SHA),(Q.CORE,Q.CORE_SHA),(Q.EXPORT,Q.EXPORT_SHA),
                         (VECTORS,VECTORS_SHA),(CONTROL,CONTROL_SHA)):
            assert P.digest(path)==sha,str(path)
        task=json.loads(TASK.read_text(encoding='utf-8'));assert all(task['gates'].values())
        assert task['artifact_sha256']==Q.CORE_SHA and task['decision']=='complete_I16_task_pass_requires_joint_native_quality_rate'
        composition=json.loads(Q.EXPORT.read_text(encoding='utf-8'))
        for path,sha in composition['helper_sha256'].items():assert P.digest(Path(path))==sha
        operator=(ROOT/FILES[2]).read_text(encoding='utf-8')
        original=(ROOT/'benchmarks/native_expert_scaling/meth274_i16_shared_input_cpu.c').read_text(encoding='utf-8').split('int main(int argc,char **argv) {')[0]
        assert operator==original
        device=R.G.Q.M.D.Q.setup();assert torch.get_num_threads()==6
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        stage='same_archive_GPU_loader';model,wrappers,proposal,load_record=R.load_stored(Q.CORE,device,Q.CORE_SHA)
        assert load_record['tensors_consumed']==725 and not load_record['source_weights_or_conditional_checkpoints_loaded']
        raw=np.fromfile(VECTORS,dtype='<u2',offset=24).reshape(256,24,896)
        expected={key:[] for key in ('parents','children','aliases','gates','delta')}
        stage='all6144_GPU_conditional_reference'
        with torch.inference_mode():
            for li,w in enumerate(wrappers):
                x=torch.from_numpy(raw[:,li].copy()).view(torch.bfloat16).to(device)
                c=R.conditional_controls(w,x)
                expected['parents'].append(c['parents'].cpu().numpy().astype('<i4'))
                expected['children'].append(c['children'].cpu().numpy().astype('<i4'))
                expected['aliases'].append(c['aliases'].cpu().numpy().astype('<i4'))
                expected['gates'].append(torch.softmax(c['scores'],dim=-1).to(torch.bfloat16).float().cpu().numpy())
                expected['delta'].append(c['conditional_contribution'].float().cpu().numpy())
                budget(start,device)
        expected={k:np.stack(v,axis=1) for k,v in expected.items()}
        np.savez(args.reference,**expected)
        torch.cuda.synchronize(device);gpu_runtime=budget(start,device)
        del model,wrappers,proposal,x,c;torch.cuda.empty_cache()
        assert args.reference.stat().st_size<48*(1<<20)
        # CPU numerical qualification only. The GPU job is terminal before this phase.
        stage='native_compile';native_start=time.monotonic();partial()
        command=['clang','-O3','-mavx2','-mssse3','-mfma','-fopenmp','-std=c11',
                 '-DSILICON_COMPLETE_I16',str(ROOT/FILES[0]),'-o',str(args.exe),'-lm','-lpsapi','-lbcrypt']
        compiled=subprocess.run(command,capture_output=True,text=True,check=True,timeout=60)
        for mode,path in (('operators',args.native),('operators_wrong_alias',args.negative)):
            stage=mode;remaining=180-(time.monotonic()-native_start);assert remaining>0
            run=subprocess.run([str(args.exe),str(Q.CORE),mode,str(VECTORS),str(path)],
                               capture_output=True,text=True,check=True,timeout=remaining)
            cpu_runs[mode]={'stdout':json.loads(run.stdout),'loader':json.loads(run.stderr)};partial()
            assert cpu_runs[mode]['loader']['fields_bound']==725 and not cpu_runs[mode]['loader']['fallback']
        control=np.fromfile(CONTROL,dtype='<f4',offset=20).reshape(256,24,896)
        def read(path):
            with path.open('rb') as f:assert struct.unpack('<8s4I',f.read(24))==(b'M284OP01',256,24,896,4)
            rows=np.fromfile(path,dtype=RECORD,offset=24);assert rows.size==6144
            assert path.stat().st_size==24+6144*RECORD.itemsize
            return rows.reshape(256,24)
        native=read(args.native);negative=read(args.negative)
        def compare(rows):
            rel=np.linalg.norm(rows['delta']-expected['delta'],axis=-1)/np.maximum(np.linalg.norm(expected['delta'],axis=-1),1e-12)
            parent_mismatch=child_mismatch=alias_mismatch=0;worst_gate=0
            for t in range(256):
                for li in range(24):
                    nr=rows[t,li];order=np.argsort(nr['parents']);eo=np.argsort(expected['parents'][t,li])
                    parent_mismatch+=int(not np.array_equal(nr['parents'][order],expected['parents'][t,li,eo]))
                    child_mismatch+=int(not np.array_equal(nr['children'][order],expected['children'][t,li,eo]))
                    alias_mismatch+=int(not np.array_equal(nr['aliases'][order],expected['aliases'][t,li,eo]))
                    worst_gate=max(worst_gate,float(np.max(np.abs(nr['gates'][order]-expected['gates'][t,li,eo]))))
            return {'source_base_FP32_bytes_exact274':rows['base'].tobytes()==control.tobytes(),
                    'parent_state_mismatches':parent_mismatch,'child_state_mismatches':child_mismatch,
                    'alias_state_mismatches':alias_mismatch,'gate_max_abs':worst_gate,
                    'conditional_relative_l2_median':float(np.median(rel)),
                    'conditional_relative_l2_max':float(np.max(rel)),
                    'conditional_BF16_value_mismatches':int(np.count_nonzero(rows['delta']!=expected['delta'])),
                    'finite':bool(np.isfinite(rows['base']).all() and np.isfinite(rows['delta']).all()),'relative_by_state':rel.tolist()}
        actual=compare(native);wrong=compare(negative)
        gates={'all725_exact_archive_no_fallback':True,'all6144_source_base_byte_exact274':actual['source_base_FP32_bytes_exact274'],
               'all6144_parent_child_alias_IDs_exact':all(actual[k]==0 for k in ('parent_state_mismatches','child_state_mismatches','alias_state_mismatches')),
               'BF16_gate_max_abs':actual['gate_max_abs']<=.002,
               'isolated_conditional_median_relative_l2':actual['conditional_relative_l2_median']<=1e-3,
               'isolated_conditional_max_relative_l2':actual['conditional_relative_l2_max']<=1e-2,
               'finite':actual['finite'],
               'wrong_lookup_preserves_all_other_controls':all(np.array_equal(native[k],negative[k]) for k in ('base','parents','children','aliases','gates')),
               'wrong_B_lookup_detected':wrong['conditional_relative_l2_median']>1e-3 or wrong['conditional_relative_l2_max']>1e-2}
        result={'experiment':'METH-284-actual-phase60-complete-archive-operator-qualification',
                'artifact_sha256':Q.CORE_SHA,'task_sha256':TASK_SHA,'vectors_sha256':VECTORS_SHA,
                'source_control_sha256':CONTROL_SHA,'catalog_sha256':P.digest(ROOT/FILES[3]),
                'source_sha256':{f:P.digest(ROOT/f) for f in FILES},'script_sha256':P.digest(Path(__file__)),
                'compile_command':command,'compile_stderr':compiled.stderr,'executable_sha256':P.digest(args.exe),
                'native_sha256':P.digest(args.native),'negative_sha256':P.digest(args.negative),'reference_sha256':P.digest(args.reference),
                'load_record':load_record,'cpu_runs':cpu_runs,'actual':actual,'wrong_alias_control':wrong,'gates':gates,
                'GPU_reference_runtime':gpu_runtime,'CPU_qualification_seconds':time.monotonic()-native_start,
                'seconds':time.monotonic()-start,'diagnostic_only':True,'native_promotion_qualified':False,
                'decision':'native_archive_operators_pass_requires_whole_model_qualification' if all(gates.values()) else 'native_archive_operators_fail_stop_before_whole_model',
                'scope':'Direct original276 archive in phase60 compile personality;6144 original states and causal wrong-B control. Whole attention/cache/logits/generation/task and rate not observed or qualified;large-n/DRAM/family goal remains.'}
        args.out.write_text(json.dumps(result,indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('decision','gates','seconds','CPU_qualification_seconds')}),flush=True)
    except BaseException as error:
        partial();args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':type(error).__name__+': '+str(error),'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8');raise


if __name__=='__main__':main()
