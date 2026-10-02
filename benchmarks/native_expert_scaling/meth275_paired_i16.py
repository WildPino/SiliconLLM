#!/usr/bin/env python3
"""Lossless paired16/aligned input layout: full CPU conservation and paired cost."""
import argparse
import json
from pathlib import Path
import statistics
import struct
import subprocess
import time
import numpy as np
import psutil
import meth272_fused_source_input as A

PRIOR=A.DOC/'meth274_i16_shared_input_result.json'
PRIOR_SHA='aa2acf6452031d621568cb4712c1a976587755c398af2dc03274ac844cd451cc'
FIXTURE=A.ROOT/'results/native_expert_scaling/meth271_output_aware_rows_fixture.bin'
QUANTIZER=A.ROOT/'results/native_expert_scaling/meth274_i16_quantizer.bin'
SOURCE=Path(__file__).with_name('meth275_paired_i16_cpu.c')
CONTROL=SOURCE.with_name('meth274_i16_shared_input_cpu.exe')
BYTES=337596480


def main():
    ap=argparse.ArgumentParser()
    for name in ('binary','exe','check','control-check','timing-check','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();start=time.monotonic();stage='bindings';segments=[];unpack=[];runs={}
    assert all(not p.exists() for p in (args.binary,args.exe,args.check,args.control_check,
        args.timing_check,args.out,args.out.with_suffix('.failure.json')))
    def partial():
        A.dump(args.out.with_suffix('.partial.json'),{'stage':stage,'segments':segments,
            'unpack':unpack,'runs':runs,'seconds':time.monotonic()-start})
    try:
        assert A.digest(PRIOR)==PRIOR_SHA
        prior=json.loads(PRIOR.read_text(encoding='utf-8'))
        assert prior['decision']=='stop_fixed_I16_shared_operator_at_native_cost'
        assert all(v for k,v in prior['gates'].items() if k!='native_component_median_at_most10ms')
        assert A.digest(A.PRIOR)==A.PRIOR_SHA
        original=json.loads(A.PRIOR.read_text(encoding='utf-8'))
        assert A.digest(FIXTURE)==prior['fixture_sha256']==original['binary']['sha256']
        assert A.digest(QUANTIZER)==prior['quantizer_sha256']
        assert A.digest(A.VECTORS)==A.VECTORS_SHA==prior['vectors_sha256']
        assert A.digest(CONTROL)==prior['executable_sha256']
        assert A.digest(CONTROL.with_suffix('.c'))==prior['source_code_sha256']
        assert A.digest(CONTROL.with_name('meth274_i16_shared_input.py'))==prior['script_sha256']
        include=SOURCE.with_name('meth182_group64_ffn_cpu.c')
        assert A.digest(include)==prior['included_meth182_sha256']
        old_check=A.ROOT/'results/native_expert_scaling/meth274_i16_shared_input.check.bin'
        old_timing_check=A.ROOT/'results/native_expert_scaling/meth274_i16_shared_input.check.timing.bin'
        assert A.digest(old_check)==prior['native_all_output_sha256']
        assert A.digest(old_timing_check)==prior['timing_output_sha256']
        old={(r['layer'],r['name']):r for r in original['segments']}
        stage='reversible_paired16_fixture_export'
        args.binary.parent.mkdir(parents=True,exist_ok=True)
        with FIXTURE.open('rb') as source,args.binary.open('xb') as output:
            assert source.read(32)==struct.pack('<8s6I',b'M252PF01',24,896,4864,32,32,128)
            output.write(struct.pack('<8s6I',b'M275PK01',24,896,4864,32,32,128))
            def read(li,name):
                s=old[(li,name)];source.seek(s['offset']);raw=source.read(s['bytes'])
                assert A.hashlib.sha256(raw).hexdigest()==s['sha256'];return raw
            def write(li,name,raw):
                segments.append({'layer':li,'name':name,'offset':output.tell(),'bytes':len(raw),
                    'sha256':A.hashlib.sha256(raw).hexdigest()});output.write(raw)
            write(None,'silu_table',read(None,'silu_table'));write(None,'alignment_padding',bytes(28))
            for li in range(24):
                gate=read(li,'gate.q');up=read(li,'up.q')
                g=np.frombuffer(gate,dtype=np.int8).reshape(4864,56,16)
                u=np.frombuffer(up,dtype=np.int8).reshape(4864,56,16)
                packed=np.stack((g,u),axis=2)
                assert packed.shape==(4864,56,2,16)
                assert packed[:,:,0,:].copy().reshape(4864,896).tobytes()==gate
                assert packed[:,:,1,:].copy().reshape(4864,896).tobytes()==up
                assert output.tell()%64==0
                write(li,'input.paired_q',packed.tobytes())
                for name in ('gate.scale','up.scale'):write(li,name,read(li,name))
                for s in original['segments']:
                    if s['layer']==li and s['name'] not in ('gate.q','up.q','gate.scale','up.scale'):
                        write(li,s['name'],read(li,s['name']))
                unpack.append({'layer':li,'gate_sha256':old[(li,'gate.q')]['sha256'],
                    'up_sha256':old[(li,'up.q')]['sha256'],'both_unpacked_byte_exact':True})
                partial()
        assert args.binary.stat().st_size==BYTES and len(segments)==338
        unchanged=[s for s in segments if s['name'] not in ('input.paired_q','alignment_padding')]
        assert len(unchanged)==313 and all((s['bytes'],s['sha256'])==
            (old[(s['layer'],s['name'])]['bytes'],old[(s['layer'],s['name'])]['sha256']) for s in unchanged)
        with args.binary.open('rb') as file:
            assert file.read(32)==struct.pack('<8s6I',b'M275PK01',24,896,4864,32,32,128)
            for s in segments:
                assert file.tell()==s['offset'] and A.hashlib.sha256(file.read(s['bytes'])).hexdigest()==s['sha256']
            assert not file.read(1)
        stage='compile_paired_reader';partial()
        command=['clang',*A.FLAGS,str(SOURCE),'-o',str(args.exe),'-lm','-lpsapi']
        compilation=subprocess.run(command,capture_output=True,text=True,check=True,timeout=60)
        def run(name,exe,fixture,check,mode,weight_bytes):
            nonlocal stage
            stage=name
            invocation=[str(exe),str(fixture),str(A.VECTORS),str(check),str(QUANTIZER),mode]
            execution=subprocess.run(invocation,capture_output=True,text=True,check=True,timeout=180)
            timing=json.loads(execution.stdout);qualification=json.loads(execution.stderr)
            assert all(timing[k]==v for k,v in (('threads',6),('tokens',256),('layers',24),
                ('hidden',4864),('residual_rank',32),('private_units',128),('weight_bytes',weight_bytes)))
            assert timing['peak_working_set_bytes']<=20*1024**3
            assert qualification['mode']==mode
            runs[name]={'command':invocation,'timing':timing,'qualification':qualification,'output_sha256':A.digest(check)}
            partial();print(json.dumps({'stage':name,'timing':timing,'qualification':qualification}),flush=True)
            return runs[name]
        control=run('unchanged274_timing',CONTROL,FIXTURE,args.control_check,'timing',337596452)
        qualification=run('paired275_qualification',args.exe,args.binary,args.check,'qualify',BYTES)
        record=qualification['qualification']
        integer_exact=record['qualified_states']==6144 and record['integer_dots_checked']==59768832 and record['quantizer_values_checked']==5505024
        integer_exact=integer_exact and all(record[k]==0 for k in ('quantizer_mismatches','parameter_mismatches','integer_dot_mismatches'))
        all_output_exact=qualification['output_sha256']==prior['native_all_output_sha256']
        gates={'all_frozen_source_prior_vector_quantizer_executable_bindings':True,
            'all24_gate_up_matrices_unpack_byte_exact':True,'all313_nonpaired_segments_byte_unchanged':True,
            'all338_new_segments_readback_exact':True,'all_paired_layers_64byte_aligned':True,
            'all6144_CPU_GPU_quantizers_and_scalar_I64_dots_exact':integer_exact,
            'all6144_output_vectors_bitwise_equal274_including_header':all_output_exact,
            'unchanged_control384_check_exact274':control['output_sha256']==prior['timing_output_sha256']}
        if not all(gates.values()):
            decision='stop_paired_layout_qualification_no_changed_timing'
        else:
            changed=run('paired275_timing',args.exe,args.binary,args.timing_check,'timing',BYTES)
            ct,nt=control['timing'],changed['timing']
            assert len(ct['pass_ms_per_token'])==len(nt['pass_ms_per_token'])==3
            assert struct.unpack_from('<8s3I',args.timing_check.read_bytes())==(b'M274OUT1',16,24,896)
            gates['all384_timing_outputs_bitwise_equal274']=changed['output_sha256']==prior['timing_output_sha256']
            gates['all256_timing_checksum_equal274']=ct['checksum']==nt['checksum']==prior['native_runs']['timing']['timing']['checksum']
            cm=statistics.median(ct['pass_ms_per_token']);nm=statistics.median(nt['pass_ms_per_token'])
            gates['paired_kernel_absolute_median_at_most10ms']=nm<=10
            gates['paired_kernel_at_least5percent_faster_than_control']=nm<=.95*cm
            decision='paired_I16_layout_component_pass_requires_full_archive_and_quality' if all(gates.values()) else 'stop_fixed_paired_I16_layout_at_numeric_or_cost'
        runtime={'seconds':time.monotonic()-start,'rss_bytes':psutil.Process().memory_info().rss,'GPU_not_used':True}
        assert runtime['seconds']<=10*60 and runtime['rss_bytes']<=20*1024**3
        result={'experiment':'METH-275-paired16-aligned-I16-shared-input-layout','prior_sha256':PRIOR_SHA,
            'source_fixture_sha256':prior['fixture_sha256'],'vectors_sha256':A.VECTORS_SHA,
            'quantizer_sha256':prior['quantizer_sha256'],'segments':segments,'unpack':unpack,
            'binary':{'bytes':BYTES,'sha256':A.digest(args.binary),'extra_alignment_bytes':28},
            'script_sha256':A.digest(Path(__file__)),'source_code_sha256':A.digest(SOURCE),
            'included_meth182_sha256':A.digest(include),'runner_helper_sha256':A.digest(Path(A.__file__)),
            'control_executable_sha256':A.digest(CONTROL),'executable_sha256':A.digest(args.exe),
            'compile_command':command,'compile_stderr':compilation.stderr,'runs':runs,'gates':gates,'runtime':runtime,
            'decision':decision,'scope':'Same274 physical functions,paired layout+aligned reader only. Finite all6144 CPU output conservation; no new full artifact/fresh quality/useful n/RAM routing-LUT-real DRAM/accepted>=50/family proof. Earlier cost and267 semantic stops unchanged.'}
        if 'paired275_timing' in runs:
            result['summary']={'unchanged_median_ms_per_token':cm,'paired_median_ms_per_token':nm,'paired_to_control_ratio':nm/cm}
        A.dump(args.out,result);print(json.dumps({k:result[k] for k in ('decision','gates','runtime')}),flush=True)
    except BaseException as failure:
        partial();A.dump(args.out.with_suffix('.failure.json'),{'stage':stage,'error':type(failure).__name__+': '+str(failure),
            'seconds':time.monotonic()-start,'runs':runs,'subprocess_stdout':getattr(failure,'stdout',None),
            'subprocess_stderr':getattr(failure,'stderr',None)})
        raise


if __name__=='__main__':main()
