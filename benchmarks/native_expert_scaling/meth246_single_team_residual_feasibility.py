#!/usr/bin/env python3
"""Same physical rank32 functions, one OpenMP team rather than four."""
import argparse
import json
from pathlib import Path
import statistics
import struct
import subprocess
import time
import numpy as np
import psutil
import meth245_separate_residual_feasibility as A

P,C=A.P,A.C
PRIOR=P.DOC/'meth245_separate_residual_native_repair1_result.json'
PRIOR_SHA='7022e2063afef2339342ff515ab4788732b7d795b8b2c690c511b4f5a5e5d61d'
FIXTURE=P.ROOT/'results/native_expert_scaling/meth245_separate_residual_fixture_repair1.bin'


def main():
    ap=argparse.ArgumentParser()
    for name in ('exe','check','out'): ap.add_argument('--'+name,required=True,type=Path)
    args=ap.parse_args(); assert not args.check.exists() and not args.out.exists()
    start,stage=time.monotonic(),'bindings'
    try:
        assert P.digest(PRIOR)==PRIOR_SHA
        old=json.loads(PRIOR.read_text())
        assert old['decision']=='stop_this_fixed_rank32_separate_residual_operator'
        assert not old['gates']['native_component_cost']
        assert all(v for k,v in old['gates'].items() if k!='native_component_cost')
        assert P.digest(FIXTURE)==old['binary']['sha256'] and P.digest(C.VECTORS)==C.VECTORS_SHA
        previous=FIXTURE.with_suffix('.check.bin'); assert P.digest(previous)==old['output_sha256']
        with FIXTURE.open('rb') as file:
            assert file.read(28)==struct.pack('<8s5I',b'M245RS01',24,896,4864,32,32)
            for segment in old['segments']:
                assert file.tell()==segment['offset'] and P.M17.sha(file.read(segment['bytes']))==segment['sha256']
            assert file.read(1)==b''
        stage='new_single_team_kernel_one_invocation'
        run=subprocess.run([str(args.exe),str(FIXTURE),str(C.VECTORS),str(args.check)],capture_output=True,text=True,check=True,timeout=180)
        timing=json.loads(run.stdout)
        assert all(timing[name]==value for name,value in (('threads',6),('tokens',256),('layers',24),('hidden',4864),('residual_rank',32),('weight_bytes',326580256)))
        assert len(timing['pass_ms_per_token'])==3
        raw=args.check.read_bytes(); original=previous.read_bytes()
        assert len(raw)==len(original)==20+16*24*896*4
        assert struct.unpack_from('<8s3I',raw)==(b'M246OUT1',16,24,896)
        assert struct.unpack_from('<8s3I',original)==(b'M245OUT1',16,24,896)
        assert np.isfinite(np.frombuffer(raw,dtype='<f4',offset=20)).all()
        exact=raw[20:]==original[20:]
        checksum=timing['checksum']==old['timing']['checksum']
        gates={'source_fixture_vector_prior_bindings':True,'all289_segments_readback_exact':True,
            'all384_native_output_vectors_bitwise_equal_prior':exact,'all256_timing_checksum_equal_prior':checksum,
            'inherited_same_GPU_oracle_numeric_and_source_function_gates':exact and old['gates']['native_numeric'] and old['gates']['source_function_error'],
            'native_component_cost':statistics.median(timing['pass_ms_per_token'])<=10}
        runtime={'seconds':time.monotonic()-start,'rss_bytes':psutil.Process().memory_info().rss,
            'GPU_not_used':True,'native_peak_working_set_bytes':timing['peak_working_set_bytes']}
        assert runtime['seconds']<=6*60 and runtime['rss_bytes']<=20*1024**3
        result={'experiment':'METH-246-single-OpenMP-team-mixed-plus-rank32-residual-native-qualification',
            'prior_result_sha256':PRIOR_SHA,'fixture_sha256':old['binary']['sha256'],
            'source_sha256':old['source_sha256'],'vectors_sha256':C.VECTORS_SHA,
            'timing':timing,'gates':gates,'runtime':runtime,
            'summary':{**old['summary'],'median_ms_per_token':statistics.median(timing['pass_ms_per_token']),
                'bitwise_output_elements':16*24*896 if exact else 0},
            'output_sha256':P.digest(args.check),'output_payload_sha256':P.M17.sha(raw[20:]),
            'prior_payload_sha256':P.M17.sha(original[20:]),'executable_sha256':P.digest(args.exe),
            'source_code_sha256':P.digest(args.exe.with_suffix('.c')),'script_sha256':P.digest(Path(__file__)),
            'included_meth182_source_sha256':P.digest(args.exe.parent/'meth182_group64_ffn_cpu.c'),
            'decision':'single_team_rank32_operator_pass_freeze_learned_factorization' if all(gates.values()) else 'stop_this_fixed_single_team_rank32_kernel',
            'scope':'Identical source rank32 physical fixture/operator arithmetic;changed OpenMP execution organization only. No learned bank,n-scaled route/LUT/DRAM,new/full LLM quality or accepted full rate.'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('decision','timing','gates','runtime')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),
            'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__': main()
