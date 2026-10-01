#!/usr/bin/env python3
"""Same private nonlinear function, split shared/private execution stages."""
import argparse
import json
from pathlib import Path
import statistics
import struct
import subprocess
import time
import numpy as np
import psutil
import meth252_private_feature_feasibility as A

P,C=A.P,A.C
PRIOR=P.DOC/'meth252_private_feature_native_result.json'
PRIOR_SHA='7da69e9d44df64719fddbfb62a452a2b4afb3be7044511f1749d16305b6910ce'
FIXTURE=P.ROOT/'results/native_expert_scaling/meth252_private_feature_fixture.bin'


def main():
    ap=argparse.ArgumentParser()
    for name in ('exe','check','out'):ap.add_argument('--'+name,required=True,type=Path)
    args=ap.parse_args();assert not args.check.exists() and not args.out.exists()
    start=time.monotonic();stage='bindings'
    try:
        assert P.digest(PRIOR)==PRIOR_SHA;prior=json.loads(PRIOR.read_text())
        assert prior['decision']=='stop_this_fixed_private_source_feature_operator' and not prior['gates']['native_component_cost']
        assert all(v for k,v in prior['gates'].items() if k!='native_component_cost')
        assert P.digest(FIXTURE)==prior['binary']['sha256'] and P.digest(C.VECTORS)==C.VECTORS_SHA
        previous=FIXTURE.with_suffix('.check.bin');assert P.digest(previous)==prior['output_sha256']
        with FIXTURE.open('rb') as file:
            assert file.read(32)==struct.pack('<8s6I',b'M252PF01',24,896,4864,32,32,32)
            for r in prior['segments']:assert file.tell()==r['offset'] and P.M17.sha(file.read(r['bytes']))==r['sha256']
            assert file.read(1)==b''
        stage='split_private_feature_kernel_once'
        run=subprocess.run([str(args.exe),str(FIXTURE),str(C.VECTORS),str(args.check)],capture_output=True,text=True,check=True,timeout=180)
        timing=json.loads(run.stdout)
        assert all(timing[k]==v for k,v in (('threads',6),('tokens',256),('layers',24),('hidden',4864),('residual_rank',32),('private_units',32),('weight_bytes',329334308)))
        assert len(timing['pass_ms_per_token'])==3
        raw=args.check.read_bytes();old=previous.read_bytes();assert len(raw)==len(old)==20+16*24*896*4
        assert struct.unpack_from('<8s3I',raw)==(b'M253OUT1',16,24,896) and struct.unpack_from('<8s3I',old)==(b'M252OUT1',16,24,896)
        assert np.isfinite(np.frombuffer(raw,dtype='<f4',offset=20)).all()
        exact=raw[20:]==old[20:];checksum=timing['checksum']==prior['timing']['checksum']
        gates={'source_prior_fixture_vector_bindings':True,'all361_segments_readback_exact':True,
            'all384_native_output_vectors_bitwise_equal_prior':exact,'all256_timing_checksum_equal_prior':checksum,
            'inherited_same_GPU_oracle_numeric_and_source_fidelity':exact and prior['gates']['native_numeric'] and prior['gates']['source_function_error'],
            'native_component_cost':statistics.median(timing['pass_ms_per_token'])<=10}
        runtime={'seconds':time.monotonic()-start,'rss_bytes':psutil.Process().memory_info().rss,
            'GPU_not_used':True,'native_peak_working_set_bytes':timing['peak_working_set_bytes']}
        assert runtime['seconds']<=6*60 and runtime['rss_bytes']<=20*1024**3
        result={'experiment':'METH-253-split-shared-private-nonlinear-feature-native-qualification','prior_result_sha256':PRIOR_SHA,
            'fixture_sha256':prior['binary']['sha256'],'source_sha256':prior['source_sha256'],'vectors_sha256':C.VECTORS_SHA,
            'timing':timing,'gates':gates,'runtime':runtime,
            'summary':{**prior['summary'],'median_ms_per_token':statistics.median(timing['pass_ms_per_token']),
                'bitwise_output_elements':16*24*896 if exact else 0},
            'output_sha256':P.digest(args.check),'output_payload_sha256':P.M17.sha(raw[20:]),'prior_payload_sha256':P.M17.sha(old[20:]),
            'executable_sha256':P.digest(args.exe),'source_code_sha256':P.digest(args.exe.with_suffix('.c')),
            'included_meth182_source_sha256':P.digest(args.exe.parent/'meth182_group64_ffn_cpu.c'),'script_sha256':P.digest(Path(__file__)),
            'decision':'split_private_source_feature_operator_pass_freeze_matched_unit_selection' if all(gates.values()) else 'stop_this_fixed_split_private_source_feature_kernel',
            'scope':'Same original private BF1632 features/shared LUT/readout. Execution stages changed only;static source choices,not learned n/dynamic routing/DRAM/new or full quality/accepted rate/family transfer.'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:result[k] for k in ('decision','timing','gates','runtime')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__':main()
