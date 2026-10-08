"""Independent NumPy/binary reconstruction of NEW saved J/probes, no model calls."""
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
from chatbot_directional_plan import geometry


def determinant(matrix,prime):
    matrix=[row[:] for row in matrix];value=1
    for col in range(len(matrix)):
        pivot=next(i for i in range(col,len(matrix)) if matrix[i][col])
        if pivot!=col:matrix[col],matrix[pivot]=matrix[pivot],matrix[col];value=-value
        v=matrix[col][col];value=value*v%prime
        for row in range(col+1,len(matrix)):
            factor=matrix[row][col]*pow(v,-1,prime)%prime
            for k in range(col,len(matrix)):matrix[row][k]=(matrix[row][k]-factor*matrix[col][k])%prime
    return value%prime


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_DIRECTIONAL_AUDIT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_source_or_student_values=0,new_jacobians=0,anchors=[])
    def guard():
        assert time.monotonic()-start<=90 and proc.memory_info().peak_wset<=1<<30
        assert not proc.children(recursive=True)
    def close(a,b):assert abs(a-b)<=1e-10+1e-10*abs(b),(a,b)
    try:
        assert sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['schema']=='QWEN_DIRECTIONAL_BINDING_V1' and b['job']['name']=='audit'
        for v in b['inputs']:
            assert str(Path(v['path']).resolve())==v['resolved_path'] and Path(v['path']).stat().st_size==v['bytes'] and sha(v['path'])==v['sha256'];guard()
        import numpy as np
        assert np.__version__=='2.4.6';args.directory.mkdir()
        raw=json.loads(Path(b['jacobian_path']).read_bytes());plan=json.loads(Path(b['plan_path']).read_bytes())
        directory=Path(b['jacobian_directory']);journal=[json.loads(v) for v in (directory/'anchors.jsonl').read_text().splitlines()]
        assert journal==raw['anchors'] and len(journal)==len(plan['anchors'])==32
        _,pv,_,_=geometry(Path(b['saved_geometry']));p=np.array(pv,dtype=np.float64)
        proof=plan['projection_rank_proof'];columns=proof['pivot_columns'];prime=proof['prime']
        minor=[]
        for row in pv:
            integers=[]
            for col in columns:
                numerator,denominator=row[col].as_integer_ratio()
                assert (1<<149)%denominator==0
                integers.append((numerator*((1<<149)//denominator))%prime)
            minor.append(integers)
        det=determinant(minor,prime);assert det==proof['minor_determinant_mod_prime'] and det!=0
        r['independent_minor']=dict(columns=columns,determinant_mod_prime=det,decoder='F32 Python exact integer ratio, not bit-exponent elimination',real_null_dimension=864)
        def array(name,shape):
            a=np.load(directory/name,mmap_mode='r',allow_pickle=False)
            assert a.shape==shape and a.dtype==np.dtype('<f8') and a.flags.c_contiguous
            assert a.offset+a.nbytes==(directory/name).stat().st_size
            return a
        q=array('selector_rowspace_Q_F64.npy',(896,32));directions=array('directions_F64.npy',(5,896))
        source=array('source_J_F64.npy',(32,896,896));student=array('student_J_F64.npy',(32,896,896));probes=array('probes_F64.npy',(32,5,2,4,896))
        orth=float(np.linalg.norm(q.T@q-np.eye(32)));row_error=float(np.linalg.norm(p-(p@q)@q.T)/np.linalg.norm(p))
        assert orth<=1e-10 and row_error<=1e-10;close(orth,raw['QR']['orthogonality_Frobenius']);close(row_error,raw['QR']['projection_relative_Frobenius'])
        expected=np.zeros((5,896))
        for i,col in enumerate((0,31,447,895)):expected[i,col]=1
        null=expected[3]-q@(q.T@expected[3]);expected[4]=null/np.linalg.norm(null)
        assert np.linalg.norm(expected-directions)<=1e-10 and np.max(np.abs(np.linalg.norm(directions,axis=1)-1))<=1e-10
        def decompose(a):
            projected=a@q;parallel=projected@q.T;remaining=a-parallel
            result={n:float(np.sum(v*v,dtype=np.float64)) for n,v in (('total',a),('parallel',projected),('null',remaining))}
            result['cross']=float(np.sum(parallel*remaining,dtype=np.float64))
            result['closure_relative']=abs(result['total']-result['parallel']-result['null'])/result['total']
            result['cross_relative']=abs(result['cross'])/result['total'];return result
        reconstructed=[];max_jv_gap=0.;max_fd_gap=0.;max_probe_fraction=0.
        for index,(record,anchor) in enumerate(zip(journal,plan['anchors'])):
            assert record['anchor_index']==index and record['x_SHA256']==anchor['x_SHA256'] and record['split']==anchor['split']
            key=bytes.fromhex(anchor['x_BF16_HEX']);assert hashlib.sha256(key).hexdigest()==anchor['x_SHA256']
            with Path(anchor['binary_path']).open('rb') as f:f.seek(anchor['x_byte_offset']);assert f.read(1792)==key
            for name,a in (('source',source),('student',student),('probes',probes)):
                frame=record['frames'][name];saved=a[index];assert np.isfinite(saved).all()
                assert frame['byte_offset']==a.offset+index*saved.nbytes and frame['bytes']==saved.nbytes
                assert frame['sha256']==hashlib.sha256(saved.tobytes(order='C')).hexdigest() and Path(frame['path'])==Path(a.filename)
            sj=np.array(source[index]);tj=np.array(student[index]);error=tj-sj
            se=decompose(sj);ee=decompose(error)
            for name,energy in (('source_energy',se),('error_energy',ee)):
                for field,value in energy.items():close(value,record['stats'][name][field])
                assert energy['closure_relative']<=1e-10 and energy['cross_relative']<=1e-10
            relative={n:math.sqrt(ee[n]/se[n]) for n in ('total','parallel','null')}
            for n,v in relative.items():close(v,record['stats']['relative_RMS'][n])
            close(se['null']/se['total'],record['stats']['source_null_energy_share'])
            for di,probe in enumerate(record['probes']):
                assert probe['direction_index']==di and len(probe['checks'])==2 and all(v>0 for v in probe['plus_minus_cell_margins'])
                h=probe['h'];difference=probe['max_score_slope_difference'];margin=anchor['interior_score_margin_F64']
                expected_h=min(2**-10,margin/(4*difference)) if difference else 2**-10
                assert h==expected_h and h>=1e-10
                for mi,j in enumerate((sj,tj)):
                    jv,fd,plus,minus=probes[index,di,mi]
                    jv_gap=float(np.linalg.norm(j@directions[di]-jv));fd_gap=float(np.linalg.norm((plus-minus)/(2*h)-fd))
                    assert jv_gap<=1e-10+1e-10*np.linalg.norm(jv) and fd_gap<=1e-10+1e-10*np.linalg.norm(fd)
                    discrepancy=float(np.linalg.norm(jv-fd));fd_norm=float(np.linalg.norm(fd));tolerance=1e-10+2e-5*fd_norm
                    assert discrepancy<=tolerance
                    check=probe['checks'][mi];assert check['model']==('source','student')[mi] and check['passed']
                    for v,w in ((discrepancy,check['discrepancy_norm']),(fd_norm,check['FD_norm']),(tolerance,check['tolerance'])):close(v,w)
                    max_jv_gap=max(max_jv_gap,jv_gap);max_fd_gap=max(max_fd_gap,fd_gap);max_probe_fraction=max(max_probe_fraction,discrepancy/tolerance)
            reconstructed.append(dict(split=anchor['split'],se=se,ee=ee,relative=relative));guard()
            r['anchors'].append(dict(anchor_index=index,source_and_error_metrics_match=True,full_frames_SHA_match=True,all_ten_directional_pairs_pass=True))
        pooled={}
        for split in ('fit','development'):
            group=[v for v in reconstructed if v['split']==split];assert len(group)==16
            se={n:math.fsum(v['se'][n] for v in group) for n in ('total','parallel','null')}
            ee={n:math.fsum(v['ee'][n] for v in group) for n in se}
            relative={n:math.sqrt(ee[n]/se[n]) for n in se};share=se['null']/se['total'];count=sum(v['relative']['total']>.1 for v in group)
            for name,values in (('source_energy',se),('error_energy',ee),('relative_RMS',relative)):
                for n,v in values.items():close(v,raw['pooled'][split][name][n])
            close(share,raw['pooled'][split]['source_null_energy_share']);assert count==raw['pooled'][split]['anchors_total_RMS_above_10_percent']
            pooled[split]=dict(source_energy=se,error_energy=ee,relative_RMS=relative,source_null_energy_share=share,anchors_total_RMS_above_10_percent=count)
        dev=pooled['development'];gates=dict(dev_total_J_RMS_above_10_percent=dev['relative_RMS']['total']>.1,
            dev_source_null_energy_at_least_half=dev['source_null_energy_share']>=.5,dev_null_J_RMS_above_10_percent=dev['relative_RMS']['null']>.1,
            dev_at_least_eight_total_RMS_above_10_percent=dev['anchors_total_RMS_above_10_percent']>=8)
        assert gates==raw['scientific_diagnostic_gates']
        decision=('DIRECTIONAL_DEFICIT_SUPPORTED_NEW_STRUCTURAL_FIT_JUSTIFIED' if all(gates.values()) else 'DIRECTIONAL_DEFICIT_UNSUPPORTED_CHANGE_REPRESENTATION')
        assert decision==raw['decision'] and raw['new_source_jacobians']==raw['new_student_jacobians']==32
        assert raw['new_source_surrogate_point_values']==raw['new_student_surrogate_point_values']==320
        r.update(decision='SAVED_DIRECTIONAL_DIAGNOSTIC_INDEPENDENTLY_VERIFIED',scientific_decision=decision,pooled=pooled,
            max_independent_Jv_gap=max_jv_gap,max_independent_FD_gap=max_fd_gap,max_probe_fraction_of_tolerance=max_probe_fraction,
            scope_limit='No re-evaluation of source/student formulas. Mass-J component energies not reconstructed: full mass-J not saved. Numerical cell margins retained, not re-evaluated.',
            elapsed_before_final_serialization=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        r['procedure_gates'].update(exact_bound_inputs=True,rank_minor_independent_decode=True,all32_original_anchor_bytes=True,
            all_complete_F64_C_layout_and_frame_SHAs=True,QR_and_directions=True,all_source_error_and_pooled_metrics=True,
            ALL320_saved_derivative_checks=True,same_frozen_scientific_decision=True,no_source_student_or_control_replay=True)
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],scientific_decision=decision,pooled=pooled)),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
