#!/usr/bin/env python3
"""Exact signed-I8 biased storage/activation split: cost screen, not quality."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import meth303_compact_i8_preflight as M

PRIOR=M.DOC/'meth303_compact_i8_preflight_result.json'
PRIOR_SHA='396edaee9c201324ffcb35eb57cadfb915a4ea6899694ca9e1055d8c29dd968b'
SOURCE_SHA='eee66947fec515b1f7805a20bd7de2be7b92e665c8a0c4602d1a7a80e2fcdcf5'
DRIVER_SHA='3b0300ada9b645fbf4576dff301cfa93bc0635ab8254e66d080de83570ef4765'
CPU=M.ROOT/'benchmarks/native_expert_scaling/meth305_biased_i8_cpu.c'
INTRINSICS=M.TOOLCHAIN/'lib/clang/21/include/avx2intrin.h'

def unchanged_math():
    old=M.CPU.read_text(encoding='utf-8');new=CPU.read_text(encoding='utf-8')
    for begin,end in [('static void execute(void){','static Matrix *matrix('),
                      ('static void prepare(unsigned token){','static Matrix *matrix('),
                      ('static double decoded_head(unsigned row){','static void selftest(void){')]:
        assert old[old.index(begin):old.index(end)]==new[new.index(begin):new.index(end)],begin
    begin,end='static int quantize(', 'static int32_t idot('
    original=old[old.index(begin):old.index(end)].strip()
    current=new[new.index(begin):new.index('// Adjacent-pair high')].strip()
    assert current.replace('split_input(q);return 1;','return 1;')==original,'changed F32 quantizer'
    return True

def run(path):
    start=time.monotonic();stage='bindings';result={'experiment':'METH-305-exact-biased-I8-kernel-cost'}
    try:
        assert M.digest(PRIOR)==PRIOR_SHA and M.digest(M.CPU)==SOURCE_SHA and M.digest(Path(M.__file__))==DRIVER_SHA
        for rel,sha in M.PINS.items():assert M.digest(M.ROOT/rel)==sha,rel
        p=json.loads(PRIOR.read_text(encoding='utf-8'))
        assert p['decision']=='reject_unchanged_compact_I8_cost_before_teacher_collection_or_training'
        result.update({'prior_sha256':PRIOR_SHA,'immutable303_source_sha256':SOURCE_SHA,'immutable303_controller_sha256':DRIVER_SHA,
                       'controller_sha256':M.digest(Path(__file__)),'cpu_source_sha256':M.digest(CPU),'engine_sha256':M.digest(M.ENGINE),
                       'full_composition_routing_head_and_F32_quantizer_exact303':unchanged_math(),
                       'intrinsic_definition':{'path':str(INTRINSICS),'sha256':M.digest(INTRINSICS),
                                               'unsigned_first_signed_second_saturating_pair_intrinsic':'_mm256_maddubs_epi16'},
                       'ledger':M.catalogue(),'gates':{'complete_weight_560mb':True}})
        M.OUT=M.OUT/'meth305_biased_i8';M.OUT.mkdir(exist_ok=True)
        M.EXE=M.OUT/'meth305_biased_i8_cpu.exe';M.SPEC=M.OUT/'meth305_spec.bin'
        stage='source_and_compile';result['spec'],result['source_binding']=M.make_spec()
        assert result['spec']['sha256']==p['spec']['sha256'] and result['source_binding']==p['source_binding']
        assert not M.EXE.exists()
        command=[str(M.COMPILER),'-O3','-mavx2','-mssse3','-mfma','-fopenmp','-ffp-contract=off','-std=c11',
                 '-DSILICON_BIASED_I8_PREFLIGHT',str(M.ENGINE),'-o',str(M.EXE),'-lm','-lpsapi','-lbcrypt']
        c=subprocess.run(command,cwd=M.ROOT,capture_output=True,text=True,timeout=120)
        result['compile']={'command':command,'exit_code':c.returncode,'stdout':c.stdout,'stderr':c.stderr,
                           'compiler_sha256':M.digest(M.COMPILER),'libomp_sha256':M.digest(M.TOOLCHAIN/'bin/libomp.dll')}
        assert c.returncode==0,c.stderr
        result['executable_sha256']=M.digest(M.EXE);stage='native';M.native(result,result['ledger'])
        native=result['native'];checks=native['selftest']
        assert [x['output_route_hash'] for x in native['operators']]==[x['output_route_hash'] for x in p['native']['operators']]
        for key,value in p['native']['selftest'].items():assert checks[key]==value,key
        assert checks['exhaustive_weight_input_cases']==65025 and checks['mixed_adjacent_pairs']==5929
        assert checks['unsplit_saturation_negative'] and checks['row_bias_error_detected']
        for k in ('allocated_bytes','active_i8_coefficients','active_row_scale_bytes','stored_i8_coefficients','functions_per_layer'):
            assert native['ready'][k]==p['native']['ready'][k]
        result['gates'].update({'all30_output_route_hashes_exact303':True,'old_selftests_exact303':True,
                                'exhaustive_products_mixed_pairs_and_fault_controls':True,'source_weight_precision_unchanged':True})
        if not result['gates']['repeatability_1_10']:result['decision']='inconclusive_biased_I8_variation_stop_before_training'
        elif not result['gates']['all_operator_medians_14ms']:result['decision']='reject_unchanged_biased_I8_kernel_before_teacher_collection_or_training'
        else:result['decision']='eligible_only_for_separately_frozen_teacher_function_fit'
        result['scope']='Exact fixture representation/compute change; synthetic640-function bank/real source head/router/norms. No donor-trained child quality/useful n/causal inference/physical DRAM or accepted-rate claim.'
        result['total_seconds']=time.monotonic()-start;M.write(path,result)
        print(json.dumps({'decision':result['decision'],'gates':result['gates'],'rep_medians_ms':[v*1000 for v in native['rep_medians_seconds']],
                          'peak_rss_bytes':native['peak_rss_bytes'],'result_sha256':M.digest(path),'total_seconds':result['total_seconds']}),flush=True)
    except BaseException as error:
        result.update({'failure_stage':stage,'error':repr(error),'total_seconds':time.monotonic()-start})
        M.write(path.with_suffix('.failure.json'),result);raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists();run(args.out)
