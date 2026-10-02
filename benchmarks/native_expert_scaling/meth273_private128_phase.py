#!/usr/bin/env python3
"""Actual unchanged private128 operator phase diagnosis; no speed promotion."""
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

PHASES=('input_decode_team_entry','shared_gate_up_LUT','private_BF16_LUT',
        'concurrent_down_right','left_bias_finite','team_exit')
FUSED=A.DOC/'meth272_fused_source_input_result.json'
FUSED_SHA='7109476722266d7dcc37abbfbde30bb11ff2c8772cbf7abcc73a4570149b343f'
SOURCE=Path(__file__).with_name('meth273_private128_phase_cpu.c')


def main():
    ap=argparse.ArgumentParser()
    for name in ('exe','check','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();start=time.monotonic();stage='bindings'
    assert all(not p.exists() for p in (args.exe,args.check,args.out,args.out.with_suffix('.failure.json')))
    try:
        assert A.digest(A.PRIOR)==A.PRIOR_SHA and A.digest(FUSED)==FUSED_SHA
        prior=json.loads(A.PRIOR.read_text(encoding='utf-8'));fused=json.loads(FUSED.read_text(encoding='utf-8'))
        assert fused['decision']=='stop_fixed_fused_input_kernel'
        assert A.digest(Path(A.__file__))==fused['script_sha256']
        assert all(v for k,v in prior['gates'].items() if k!='native_component_median_at_most10ms')
        assert A.digest(A.FIXTURE)==prior['binary']['sha256'] and A.digest(A.VECTORS)==A.VECTORS_SHA
        assert A.digest(SOURCE.with_name('meth271_output_aware_rows_cpu.c'))==prior['native_source_sha256']
        include=SOURCE.with_name('meth182_group64_ffn_cpu.c')
        assert A.digest(include)==prior['included_meth182_source_sha256']
        old_check=A.ROOT/'results/native_expert_scaling/meth271_output_aware_rows.check.bin'
        assert A.digest(old_check)==prior['native_output_sha256']
        with A.FIXTURE.open('rb') as file:
            assert file.read(32)==struct.pack('<8s6I',b'M252PF01',24,896,4864,32,32,128)
            for s in prior['segments']:
                assert file.tell()==s['offset'] and A.hashlib.sha256(file.read(s['bytes'])).hexdigest()==s['sha256']
            assert not file.read(1)
        stage='compile_instrumented_kernel'
        command=['clang',*A.FLAGS,str(SOURCE),'-o',str(args.exe),'-lm','-lpsapi']
        compilation=subprocess.run(command,capture_output=True,text=True,check=True,timeout=60)
        stage='instrumented384_checks_and_three256_state_passes'
        run=subprocess.run([str(args.exe),str(A.FIXTURE),str(A.VECTORS),str(args.check)],
            capture_output=True,text=True,check=True,timeout=180)
        timing=json.loads(run.stdout);phases=json.loads(run.stderr)['phase_ms_per_token']
        assert all(timing[k]==v for k,v in (('threads',6),('tokens',256),('layers',24),
            ('hidden',4864),('residual_rank',32),('private_units',128),('weight_bytes',337596452)))
        assert len(timing['pass_ms_per_token'])==3 and np.shape(phases)==(3,6)
        assert np.isfinite(phases).all() and (np.array(phases)>=0).all()
        raw=args.check.read_bytes()
        assert len(raw)==20+16*24*896*4 and struct.unpack_from('<8s3I',raw)==(b'M271OUT1',16,24,896)
        exact=A.digest(args.check)==prior['native_output_sha256']
        checksum=timing['checksum']==prior['timing']['checksum']
        gates={'all_fixed_source_fixture_prior_vector_bindings':True,'all361_segments_exact':True,
            'all384_outputs_bitwise_exact_original271_including_header':exact,
            'all256_timing_checksum_exact271':checksum,'all_phase_intervals_finite_nonnegative':True}
        medians=[statistics.median(row[i] for row in phases) for i in range(6)]
        denominator=sum(medians);assert denominator>0
        private_boundary=(medians[0]+medians[2]+medians[5])/denominator
        matrix=(medians[1]+medians[3])/denominator
        if not all(gates.values()):decision='stop_instrumentation_numeric_guards_no_timing_interpretation'
        elif private_boundary>=.20:decision='inspect_private_stage_and_team_handoffs_next'
        elif matrix>=.80:decision='inspect_matrix_operator_layout_or_memory_execution_next'
        else:decision='phase_distribution_unresolved_before_next_optimization'
        runtime={'seconds':time.monotonic()-start,'rss_bytes':psutil.Process().memory_info().rss,'GPU_not_used':True}
        assert runtime['seconds']<=10*60 and runtime['rss_bytes']<=20*1024**3 and timing['peak_working_set_bytes']<=20*1024**3
        result={'experiment':'METH-273-actual-private128-instrumented-phase-diagnosis',
            'prior_sha256':A.PRIOR_SHA,'fused_stop_sha256':FUSED_SHA,'fixture_sha256':prior['binary']['sha256'],
            'vectors_sha256':A.VECTORS_SHA,'source_sha256':A.digest(SOURCE),'script_sha256':A.digest(Path(__file__)),
            'runner_helper_sha256':A.digest(Path(A.__file__)),'included_meth182_sha256':A.digest(include),
            'executable_sha256':A.digest(args.exe),'output_sha256':A.digest(args.check),
            'compile_command':command,'compile_stderr':compilation.stderr,'timing':timing,
            'phase_names':PHASES,'phase_ms_per_token':phases,'phase_medians_ms':dict(zip(PHASES,medians)),
            'summary':{'private_plus_entry_exit_share':private_boundary,'shared_plus_concurrent_down_right_share':matrix},
            'gates':gates,'runtime':runtime,'decision':decision,
            'scope':'Instrumented consumed CPU component; clock/single/barrier overhead retained. No uninstrumented cost gate,DRAM isolation,full archive/new quality/useful n/route-LUT/accepted>=50/family proof.271/272 and267 stops unchanged.'}
        A.dump(args.out,result);print(json.dumps({k:result[k] for k in ('decision','phase_medians_ms','summary','gates','runtime')}),flush=True)
    except BaseException as failure:
        A.dump(args.out.with_suffix('.failure.json'),{'stage':stage,'error':type(failure).__name__+': '+str(failure),
            'seconds':time.monotonic()-start});raise


if __name__=='__main__':main()
