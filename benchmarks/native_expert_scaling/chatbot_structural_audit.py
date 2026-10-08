"""Saved-only independent scalar, NumPy matrix and coefficient-byte audit."""
import argparse
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once
from chatbot_joint_retained_audit import Descriptors,operands,npy_rows,integer32,near


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_STRUCTURAL_AUDIT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_source_student_or_gradient_evaluations=0)
    def guard():assert time.monotonic()-start<=90 and proc.memory_info().peak_wset<=1<<30 and not proc.children(recursive=True)
    def compare(a,b):
        assert a.keys()==b.keys()
        for n,v in a.items():
            if isinstance(v,dict):compare(v,b[n])
            elif v is None:assert b[n] is None
            elif isinstance(v,int):assert v==b[n]
            else:near(v,b[n])
    try:
        assert sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['schema']=='QWEN_DIRECTIONAL_BINDING_V1' and b['job']['name']=='structural_audit'
        for v in b['inputs']:
            assert str(Path(v['path']).resolve())==v['resolved_path'] and Path(v['path']).stat().st_size==v['bytes'] and sha(v['path'])==v['sha256'];guard()
        import numpy as np
        assert np.__version__=='2.4.6';args.directory.mkdir()
        raw=json.loads(Path(b['fit_result']).read_bytes());plan=json.loads(Path(b['plan_path']).read_bytes())
        baseline=json.loads(Path(b['baseline_result']).read_bytes())['arms'][0]['encodings']['F32']
        old_j=json.loads(Path(b['prior_jacobian_result']).read_bytes())
        original_audit=json.loads(Path(b['original_response_audit']).read_bytes())
        old_certificate=original_audit['response_audits']['F32']['development']['exact_necessary_failure']
        groups=operands(json.loads(Path(b['adoption_path']).read_bytes()),guard);r['responses']={}
        for split,g in groups.items():
            primary=[True]*len(g['keys']) if split=='fit' else g['novel'];errors=[];prefix=0;prefix_rows=0
            lcm=math.lcm(*(g['counts'][k] for k,keep in zip(g['keys'],primary) if keep))
            for i,(prediction,bits) in enumerate(npy_rows(Path(raw['responses'][split]['prediction_file']['path']),len(primary))):
                errors.append(math.fsum((v-y)**2 for v,y in zip(prediction,g['ys'][i])))
                if split=='development' and primary[i] and prefix_rows<64:
                    prefix+=sum((integer32(v)-integer32(y<<16))**2 for v,y in zip(bits,g['y_bits'][i]))*(lcm//g['counts'][g['keys'][i]]);prefix_rows+=1
                if i%128==0:guard()
            def metric(indices,weighted):
                if not indices:return dict(rows=0,squared_ratio=None,relative_RMS=None)
                weights=[1/g['counts'][g['keys'][i]] if weighted else 1 for i in indices]
                error=math.fsum(errors[i]*w for i,w in zip(indices,weights));source=math.fsum(g['energy'][i]*w for i,w in zip(indices,weights))
                return dict(rows=len(indices),error_energy=error,source_energy=source,squared_ratio=error/source,relative_RMS=math.sqrt(error/source))
            indices=[i for i,keep in enumerate(primary) if keep]
            metrics=dict(occurrences=metric(indices,False),unique_x_weighted=metric(indices,True),all_captured_occurrences=metric(list(range(len(primary))),False),
                generated=metric([i for i in indices if g['generated'][i]],True),prompt=metric([i for i in indices if not g['generated'][i]],True),
                categories={c:metric([i for i in indices if g['category'][i]==c],True) for c in sorted(set(g['category']))})
            compare(metrics,{n:raw['responses'][split][n] for n in metrics});r['responses'][split]=metrics
            if split=='development':
                assert len(indices)==old_certificate['source_novel_rows'] and lcm==old_certificate['weight_LCM'] and prefix_rows==64
                source_exact=int(old_certificate['source_energy_integer']);passed=10000*prefix>source_exact
                r['NEW_exact_prefix_failure_witness']=dict(prefix_rows=64,prefix_error_integer=str(prefix),source_energy_integer=str(source_exact),
                    source_energy='REUSED original byte-qualified independent exact proof, not recalculated',weight_LCM=lcm,
                    common_square_units='2^-298',strict_10000_prefix_error_gt_FULL_source=passed,inconclusive_if_false=not passed)
        del groups,g,errors
        directory=Path(b['fit_directory']);journal=[json.loads(v) for v in (directory/'anchors.jsonl').read_text().splitlines()]
        assert journal==raw['anchors'] and len(journal)==32
        def array(path,shape,dtype='<f8'):
            a=np.load(path,mmap_mode='r',allow_pickle=False);assert a.shape==shape and a.dtype==np.dtype(dtype) and a.flags.c_contiguous
            assert a.offset+a.nbytes==Path(path).stat().st_size;return a
        source=array(b['source_J_path'],(32,896,896));student=array(directory/'student_J_F64.npy',(32,896,896));probes=array(directory/'student_probes_F64.npy',(32,5,4,896))
        q=array(b['Q_path'],(896,32));directions=array(b['directions_path'],(5,896))
        labels=np.ascontiguousarray(source[:16],dtype='<f4')
        assert hashlib.sha256(labels.tobytes()).hexdigest()==raw['source_FIT_J_F32_targets']['sha256'];del labels
        errors=[];max_jv_gap=0.;max_fd_gap=0.;max_probe_fraction=0.
        for i,(record,a) in enumerate(zip(journal,plan['anchors'])):
            assert record['anchor_index']==i and record['split']==a['split'] and record['x_SHA256']==a['x_SHA256']
            for name,v in (('student',student),('probes',probes)):
                frame=record['frames'][name];value=v[i];assert np.isfinite(value).all()
                assert frame['byte_offset']==v.offset+i*value.nbytes and frame['bytes']==value.nbytes
                assert frame['sha256']==hashlib.sha256(value.tobytes(order='C')).hexdigest() and Path(frame['path'])==Path(v.filename)
            error=np.array(student[i])-source[i];projected=error@q;null=error-projected@q.T
            ee={n:float(np.sum(v*v,dtype=np.float64)) for n,v in (('total',error),('parallel',projected),('null',null))}
            closure=abs(ee['total']-ee['parallel']-ee['null'])/ee['total'];assert closure<=1e-10
            for n,value in ee.items():near(value,record['error_energy'][n])
            for n in ('total','parallel','null'):near(record['source_energy'][n],old_j['anchors'][i]['stats']['source_energy'][n])
            for n in ee:near(math.sqrt(ee[n]/record['source_energy'][n]),record['relative_RMS'][n])
            for di,check in enumerate(record['probes']):
                assert check['direction_index']==di and check['h']==old_j['anchors'][i]['probes'][di]['h'] and check['passed']
                jv,fd,plus,minus=probes[i,di];h=check['h']
                jv_gap=float(np.linalg.norm(student[i]@directions[di]-jv));fd_gap=float(np.linalg.norm((plus-minus)/(2*h)-fd))
                assert jv_gap<=1e-10+1e-10*np.linalg.norm(jv) and fd_gap<=1e-10+1e-10*np.linalg.norm(fd)
                discrepancy=float(np.linalg.norm(jv-fd));norm=float(np.linalg.norm(fd));tolerance=1e-10+2e-5*norm
                assert discrepancy<=tolerance
                for value,expected in ((discrepancy,check['discrepancy_norm']),(norm,check['FD_norm']),(tolerance,check['tolerance'])):near(value,expected)
                max_jv_gap=max(max_jv_gap,jv_gap);max_fd_gap=max(max_fd_gap,fd_gap);max_probe_fraction=max(max_probe_fraction,discrepancy/tolerance)
            errors.append(dict(split=a['split'],ee=ee,se=record['source_energy']));guard()
        pooled={}
        for split in ('fit','development'):
            records=[v for v in errors if v['split']==split];assert len(records)==16
            se={n:math.fsum(v['se'][n] for v in records) for n in ('total','parallel','null')};ee={n:math.fsum(v['ee'][n] for v in records) for n in se}
            pooled[split]=dict(source_energy=se,error_energy=ee,relative_RMS={n:math.sqrt(ee[n]/se[n]) for n in se});compare(pooled[split],raw['pooled'][split])
        updates=[json.loads(v) for v in (directory/'updates.jsonl').read_text().splitlines()];assert len(updates)==raw['completed_updates']==744
        visits=[0]*16
        for i,row in enumerate(updates):
            assert row['update']==i+1 and row['epoch']==i//31+1 and row['anchor_index']==i%16 and row['rows']==(5 if i%31==30 else 128)
            assert all(math.isfinite(row[n]) and row[n]>=0 for n in ('value_loss','J_loss','coefficient_prior','preclip_gradient_norm','elapsed_fit_seconds'))
            visits[i%16]+=1
        assert visits==raw['anchor_visits']==[47]*8+[46]*8
        for i,curve in enumerate(raw['curve']):
            rows=updates[i*31:(i+1)*31]
            near(math.fsum(v['value_loss']*v['rows'] for v in rows)/3845,curve['mean_response_minibatch_loss'])
            near(math.fsum(v['J_loss'] for v in rows)/31,curve['mean_J_minibatch_loss'])
        def tensors(path):
            with zipfile.ZipFile(path) as z:
                prefix=next(n[:-8] for n in z.namelist() if n.endswith('/data.pkl'));assert z.read(prefix+'byteorder')==b'little'
                descriptors=Descriptors(io.BytesIO(z.read(prefix+'data.pkl'))).load();result={}
                for name,desc in descriptors.items():
                    storage=desc['storage'];assert storage['dtype']=='FloatStorage' and desc['offset']==0
                    raw=z.read(prefix+'data/'+storage['key']);assert len(raw)==storage['count']*4
                    values=np.frombuffer(raw,dtype='<f4').reshape(desc['shape']);assert np.isfinite(values).all()
                    stride=tuple(v//4 for v in values.strides);assert desc['stride']==stride
                    result[name]=values
            return result
        final=tensors(directory/'E16.final_F32.pt');warm=tensors(b['warm_checkpoint'])
        for n in ('projection','parent_centers','child_centers','parent_norms','child_norms'):assert final[n].tobytes()==warm[n].tobytes()
        def leaf_hashes(values):
            hashes=[]
            for i in range(16):
                h=hashlib.sha256()
                for n in ('leaf_g','leaf_u','leaf_b'):
                    bits=values[n][i].view('<u4');bf=((bits+0x7fff+((bits>>16)&1))>>16).astype('<u2')
                    assert ((bf&0x7fff)<0x7f80).all();h.update(bf.tobytes(order='C'))
                hashes.append(h.hexdigest())
            return hashes
        hashes=leaf_hashes(final);warm_hashes=leaf_hashes(warm);assert hashes==raw['leaf_BF16_hashes']
        functions=dict(all_functions_distinct_BF16=len(set(hashes))==16,all_functions_updated_BF16=all(a!=b for a,b in zip(hashes,warm_hashes)))
        assert functions==raw['function_gates']
        dev=r['responses']['development'];fit=r['responses']['fit'];mechanism=dict(
            novel_response_improves_10pct=dev['unique_x_weighted']['relative_RMS']<=.9*baseline['development']['unique_x_weighted']['relative_RMS'],
            FIT_response_no_more_than_5pct_worse=fit['unique_x_weighted']['relative_RMS']<=1.05*baseline['fit']['unique_x_weighted']['relative_RMS'],
            ALL_categories_no_more_than_5pct_worse=all(v['relative_RMS']<=1.05*baseline['development']['categories'][n]['relative_RMS'] for n,v in dev['categories'].items()))
        mechanism.update({s+'_full_J_improves_10pct':pooled[s]['relative_RMS']['total']<=.9*old_j['pooled'][s]['relative_RMS']['total'] for s in pooled})
        absolute=dict(novel_response_RMS_le_1pct=dev['unique_x_weighted']['relative_RMS']<=.01,ALL_category_RMS_le_3pct=all(v['relative_RMS']<=.03 for v in dev['categories'].values()))
        assert mechanism==raw['mechanism_gates'] and absolute==raw['absolute_fidelity_gates']
        decision=('LOCAL_F32_STRUCTURAL_FIDELITY_AVAILABLE_ENCODING_NOT_ADMITTED' if all(mechanism.values()) and all(absolute.values()) and all(functions.values()) else 'CLOSE_ONE_STRUCTURAL_FIT_NO_WHOLE_PROMOTION')
        assert decision==raw['decision'] and raw['new_source_jacobians']==0 and raw['new_final_student_jacobians']==32 and raw['new_final_student_surrogate_point_values']==320
        r.update(decision='SAVED_FIRST_STRUCTURAL_FIT_INDEPENDENTLY_VERIFIED',scientific_decision=decision,mechanism_gates=mechanism,absolute_fidelity_gates=absolute,
            pooled=pooled,function_gates=functions,max_Jv_gap=max_jv_gap,max_FD_gap=max_fd_gap,max_probe_fraction=max_probe_fraction,
            scope_limit='No gradient/model/control/source-J evaluation. Per-selected-leaf response metrics and parameter-reference displacement not reconstructed.',
            elapsed_before_final_serialization=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        r['procedure_gates'].update(bound_inputs=True,ALL_primary_category_prompt_generated_and_occurrence_response_metrics=True,
            ALL32_new_J_and160_probes=True,ALL744_update_record_order_and_epoch_curve=True,F32_source_FIT_label_bytes=True,
            saved_router_buffer_BYTES_unchanged=True,BF16_leaf_hashes_independent_RNE=True,ALL_frozen_decisions_reconstructed=True)
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],scientific_decision=decision,mechanism_gates=mechanism,absolute_fidelity_gates=absolute)),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
