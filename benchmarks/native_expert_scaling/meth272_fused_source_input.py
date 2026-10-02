#!/usr/bin/env python3
"""Paired unchanged/fused input-dot CPU test on the fixed271 physical function."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import struct
import subprocess
import time
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
PRIOR=DOC/'meth271_output_aware_rows_result.json'
PRIOR_SHA='638594a650189ab45efa62c08be04a3e42b21f7eff00c52d372d5991077120c3'
FIXTURE=ROOT/'results/native_expert_scaling/meth271_output_aware_rows_fixture.bin'
VECTORS=ROOT/'benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth125_e1280_vectors.bin'
VECTORS_SHA='f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699'
SOURCE=Path(__file__).with_name('meth272_fused_source_input_cpu.c')
CONTROL=SOURCE.with_name('meth271_output_aware_rows_cpu.exe')
FLAGS=['-O3','-mavx2','-mssse3','-mfma','-fopenmp','-std=c11','-Wall','-Wextra']


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser()
    for name in ('exe','check','control-check','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();start=time.monotonic();stage='bindings';runs={}
    assert all(not p.exists() for p in (args.exe,args.check,args.control_check,args.out,args.out.with_suffix('.failure.json')))
    try:
        assert digest(PRIOR)==PRIOR_SHA
        prior=json.loads(PRIOR.read_text(encoding='utf-8'))
        assert prior['decision']=='stop_fixed128_output_aware_at_native_numeric_or_cost'
        assert all(v for k,v in prior['gates'].items() if k!='native_component_median_at_most10ms')
        assert digest(FIXTURE)==prior['binary']['sha256'] and digest(VECTORS)==VECTORS_SHA
        assert digest(CONTROL)==prior['executable_sha256']
        assert digest(CONTROL.with_suffix('.c'))==prior['native_source_sha256']
        include=SOURCE.with_name('meth182_group64_ffn_cpu.c')
        assert digest(include)==prior['included_meth182_source_sha256']
        previous=ROOT/'results/native_expert_scaling/meth271_output_aware_rows.check.bin'
        assert digest(previous)==prior['native_output_sha256']
        with FIXTURE.open('rb') as file:
            assert file.read(32)==struct.pack('<8s6I',b'M252PF01',24,896,4864,32,32,128)
            for s in prior['segments']:
                assert file.tell()==s['offset'] and hashlib.sha256(file.read(s['bytes'])).hexdigest()==s['sha256']
            assert not file.read(1)
        stage='compile_fused_kernel'
        command=['clang',*FLAGS,str(SOURCE),'-o',str(args.exe),'-lm','-lpsapi']
        compile_run=subprocess.run(command,capture_output=True,text=True,check=True,timeout=60)
        # This process does not import/use Torch/CUDA. Fixed order,one invocation/arm.
        for name,exe,check,magic in (('unchanged271',CONTROL,args.control_check,b'M271OUT1'),
                                    ('fused272',args.exe,args.check,b'M272OUT1')):
            stage='run_'+name
            run=subprocess.run([str(exe),str(FIXTURE),str(VECTORS),str(check)],capture_output=True,text=True,check=True,timeout=180)
            timing=json.loads(run.stdout)
            assert all(timing[k]==v for k,v in (('threads',6),('tokens',256),('layers',24),
                ('hidden',4864),('residual_rank',32),('private_units',128),('weight_bytes',337596452)))
            assert len(timing['pass_ms_per_token'])==3 and timing['peak_working_set_bytes']<=20*1024**3
            raw=check.read_bytes()
            assert len(raw)==20+16*24*896*4 and struct.unpack_from('<8s3I',raw)==(magic,16,24,896)
            assert np.isfinite(np.frombuffer(raw,dtype='<f4',offset=20)).all()
            runs[name]={'timing':timing,'output_sha256':digest(check),'payload_sha256':hashlib.sha256(raw[20:]).hexdigest(),
                'median_ms_per_token':statistics.median(timing['pass_ms_per_token'])}
            dump(args.out.with_suffix('.partial.json'),{'stage':stage,'runs':runs,'seconds':time.monotonic()-start})
            print(json.dumps({'arm':name,'timing':timing}),flush=True)
        old=previous.read_bytes()[20:]
        exact=args.control_check.read_bytes()[20:]==old==args.check.read_bytes()[20:]
        checksum=all(r['timing']['checksum']==prior['timing']['checksum'] for r in runs.values())
        control=runs['unchanged271']['median_ms_per_token'];fused=runs['fused272']['median_ms_per_token']
        gates={'all_fixed_fixture_source_prior_vector_bindings':True,'all361_segments_exact':True,
            'all384_outputs_bitwise_equal271_and_control':exact,'all256_timing_checksums_equal271':checksum,
            'inherited_source_BF16_fit_reserve_and_GPU_native_limits':exact and prior['gates']['native_numeric_original_limits'],
            'changed_kernel_absolute_median_at_most10ms':fused<=10,
            'changed_kernel_at_least5percent_faster_than_control':fused<=.95*control}
        runtime={'seconds':time.monotonic()-start,'rss_bytes':psutil.Process().memory_info().rss,'GPU_not_used':True}
        assert runtime['seconds']<=10*60 and runtime['rss_bytes']<=20*1024**3
        result={'experiment':'METH-272-fused-shared-input-dots-paired-native',
            'prior_result_sha256':PRIOR_SHA,'fixture_sha256':prior['binary']['sha256'],'vectors_sha256':VECTORS_SHA,
            'script_sha256':digest(Path(__file__)),'source_sha256':digest(SOURCE),'included_meth182_sha256':digest(include),
            'control_executable_sha256':digest(CONTROL),'executable_sha256':digest(args.exe),
            'compile_command':command,'compile_stderr':compile_run.stderr,'runs':runs,'gates':gates,'runtime':runtime,
            'summary':{'fused_to_control_ratio':fused/control,'bitwise_checked_elements':344064 if exact else 0},
            'decision':'fused_private128_component_pass_requires_full_archive_and_quality' if all(gates.values()) else 'stop_fixed_fused_input_kernel',
            'scope':'Same271 weights,selector and function. Paired consumed CPU component only; no new full artifact/generation/task/large-n/real dynamic routing/accepted>=50 or family proof.271 cost stop preserved;259 closed by267.'}
        dump(args.out,result);print(json.dumps({k:result[k] for k in ('decision','summary','gates','runtime')}),flush=True)
    except BaseException as failure:
        dump(args.out.with_suffix('.failure.json'),{'stage':stage,'error':type(failure).__name__+': '+str(failure),
            'runs':runs,'seconds':time.monotonic()-start});raise


if __name__=='__main__':main()
