"""First unrounded-bank novel observable, or its saved-only exact proof audit."""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once
from chatbot_joint_retained_audit import integer32


def integer64(bits):
    exponent=(bits>>52)&2047;mantissa=bits&((1<<52)-1);assert exponent!=2047
    value=mantissa if exponent==0 else ((1<<52)|mantissa)<<(exponent-1)
    return -value if bits>>63 else value


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_COUPLED_PROBE_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_source_or_student_J=0,new_encoded_response_evaluations=0)
    def guard():assert time.monotonic()-start<=45 and proc.memory_info().peak_wset<=512<<20 and not proc.children(recursive=True)
    try:
        assert sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name'] in ('coupled_probe','coupled_probe_audit')
        for item in b['inputs']:
            p=Path(item['path']);assert str(p.resolve())==item['resolved_path'] and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];guard()
        journal=json.loads(Path(b['journal_path']).read_bytes());assert len(journal)==64
        old=json.loads(Path(b['original_response_audit']).read_bytes())['response_audits']['F32']['development']['exact_necessary_failure']
        denominator=int(old['source_energy_integer'])<<1850
        args.directory.mkdir();numerator=0;errors=[];energies=[];gaps=[]
        if b['job']['name']=='coupled_probe':
            import numpy as np
            assert np.__version__=='2.4.6' and 'torch' not in sys.modules
            raw=json.loads(Path(b['compile_path']).read_bytes())
            def array(name):return np.load(raw['output_arrays'][name]['path'],mmap_mode='r',allow_pickle=False)
            matrices=array('affine_A_F64');bias=array('affine_bias_F64');encoded=array('novel_prefix64_F32')
            core=[]
            for name in ('shared_g','shared_u','shared_b'):
                bits=array(name+'_BF16');core.append((bits.astype('<u4')<<16).view('<f4').astype(np.float64))
            values=np.empty((64,896));core_values=np.empty_like(values)
            for i,item in enumerate(journal):
                with Path(item['binary_path']).open('rb') as f:f.seek(item['x_byte_offset']);data=f.read(3584)
                assert hashlib.sha256(data[:1792]).hexdigest()==item['x_SHA256'] and hashlib.sha256(data[1792:]).hexdigest()==item['y_SHA256']
                bits=np.frombuffer(data[:1792],dtype='<u2');x=(bits.astype('<u4')<<16).view('<f4').astype(np.float64)
                gate,up,down=core;g=gate@x;u=up@x;t=np.exp(-np.abs(g));sig=np.where(g>=0,1/(1+t),t/(1+t))
                core_values[i]=down@(g*sig*u)
                ids=item['selected_ids'];mass=np.asarray(struct.unpack('<4f',bytes.fromhex(item['selected_mass_F32_HEX'])),dtype=np.float64)
                values[i]=core_values[i]+mass@(matrices[ids]@x+bias[ids])
                ybits=struct.unpack('<896H',data[1792:]);target=struct.unpack('<896f',struct.pack('<896I',*(v<<16 for v in ybits)))
                pbits=struct.unpack('<896Q',values[i].tobytes());mult=item['original_multiplicity']
                numerator+=sum((integer64(p)-(integer32(v<<16)<<925))**2 for p,v in zip(pbits,ybits))*(2//mult)
                errors.append(math.fsum((float(p)-y)**2 for p,y in zip(values[i],target))/mult)
                energies.append(math.fsum(y*y for y in target)/mult)
                gaps.append(math.fsum((float(p)-float(e))**2 for p,e in zip(values[i],encoded[i]))/mult);guard()
            for name,value in (('prefix64_unrounded_bank_F64',values),('prefix64_shared_only_F64',core_values)):
                assert np.isfinite(value).all()
                path=args.directory/(name+'.npy')
                with path.open('xb') as f:np.save(f,value,allow_pickle=False)
                r.setdefault('outputs',{})[name]=dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path),shape=[64,896],dtype='<f8',C_order=True)
            r['new_unrounded_bank_novel_responses']=64;r['new_shared_only_F64_novel_values']=64
            failed=10000*numerator>denominator
            gap_ratio=math.sqrt(math.fsum(gaps)/math.fsum(errors));dominates=failed and gap_ratio<=.1
            r['decision']='UNROUNDED_FIELD_EXTENSION_FAILURE_DOMINATES_CODEC_ON_PREFIX' if dominates else ('UNROUNDED_FIELD_ALSO_FAILS_CODEC_CONTRIBUTION_UNRESOLVED' if failed else 'UNROUNDED_PREFIX_BOUND_INCONCLUSIVE_CODEC_REMAINS_OPEN')
            r.update(exact_prefix_error_integer=str(numerator),reused_FULL_source_integer_scaled=str(denominator),
                common_square_units='2^-2148; old BF16 source integer shifted1850 bits',
                necessarily_fails_1pct_RMS=failed,prefix_squared_error_over_FULL_energy=numerator/denominator,
                prefix_relative_RMS=math.sqrt(math.fsum(errors)/math.fsum(energies)),
                encoded_difference_RMS_over_unrounded_error_RMS=gap_ratio,
                encoded_difference_relative_RMS=math.sqrt(math.fsum(gaps)/math.fsum(energies)),
                scope='Same64 inputs/mass exact F64 promotion, stored BF16 core, UNROUNDED full affine coefficients; no new routing, codec rescue or whole quality')
        else:
            probe=json.loads(Path(b['probe_path']).read_bytes());saved=Path(probe['outputs']['prefix64_unrounded_bank_F64']['path'])
            with saved.open('rb') as f:
                import ast
                assert f.read(8)==b'\x93NUMPY\x01\x00'
                length=struct.unpack('<H',f.read(2))[0];header=ast.literal_eval(f.read(length).decode('ascii').strip())
                assert header=={'descr':'<f8','fortran_order':False,'shape':(64,896)}
                for item in journal:
                    row=struct.unpack('<896d',f.read(7168));assert all(math.isfinite(v) for v in row)
                    with Path(item['binary_path']).open('rb') as original:original.seek(item['x_byte_offset']+1792);ybits=struct.unpack('<896H',original.read(1792))
                    mult=item['original_multiplicity']
                    # Independent exact Python ratio, rather than IEEE bit decoder.
                    exact=[]
                    for p,v in zip(row,ybits):
                        n,d=p.as_integer_ratio();assert (1<<1074)%d==0
                        target=struct.unpack('<f',struct.pack('<I',v<<16))[0]
                        yn,yd=target.as_integer_ratio();assert (1<<1074)%yd==0
                        exact.append((n*((1<<1074)//d)-yn*((1<<1074)//yd))**2)
                    numerator+=sum(exact)*(2//mult);guard()
                assert not f.read(1)
            assert str(numerator)==probe['exact_prefix_error_integer'] and str(denominator)==probe['reused_FULL_source_integer_scaled']
            assert (10000*numerator>denominator)==probe['necessarily_fails_1pct_RMS']
            r.update(schema='QWEN_COUPLED_PROBE_AUDIT_RESULT_V1',decision='SAVED_UNROUNDED_PREFIX_EXACT_BOUND_INDEPENDENTLY_VERIFIED',
                exact_prefix_error_integer=str(numerator),reused_FULL_source_integer_scaled=str(denominator),
                necessarily_fails_1pct_RMS=10000*numerator>denominator,source_full_sum_replayed=False)
        r['procedure_gates'].update(only_bound_first64_original_bytes=True,original_source_energy_integer_reused=True,
            new_observable_or_independent_integer_decoder=True,no_donor_routing_optimizer_encoded_response_replay=True)
        r.update(elapsed_before_final_serialization=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],seconds=time.monotonic()-start)),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset)
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
