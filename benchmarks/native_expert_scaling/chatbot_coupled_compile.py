"""One full-input affine jet compiler and encoded-response necessary screen."""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once
from chatbot_joint_retained_audit import Descriptors,routes,integer32
from chatbot_directional_plan import rows


def main(args):
    import psutil
    started=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_COUPLED_COMPILE_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_source_jacobians=0,new_optimizer_steps=0,output_arrays={})
    def guard():
        assert time.monotonic()-started<=180 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
        if args.directory.exists():assert sum(p.stat().st_size for p in args.directory.iterdir() if p.is_file())<=512<<20
    try:
        assert sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='coupled_compile'
        for item in b['inputs']:
            p=Path(item['path']);assert str(p.resolve())==item['resolved_path'] and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];guard()
        import numpy as np
        assert np.__version__=='2.4.6' and 'torch' not in sys.modules;args.directory.mkdir()
        design=json.loads(Path(b['design_path']).read_bytes());arm=design['budget']['arms'][0]
        # Preserve original F32 router coefficients; this actual proposal prices
        # their extra two bytes before any new core/function observable.
        logical=arm['logical_coefficient_bytes_per_token']+2*arm['router_matrix_MAC']
        storage=arm['proposed_payload_bytes']+2*24*(896*32+16*32)
        r['budget']=dict(matrix_MAC=arm['matrix_MAC_per_token'],logical_bytes=logical,stored_bytes=storage,
            source_matrix_MAC=493961216,source_logical_bytes=988067328,
            router='Original F32 keys/norms; cached original selections/mass for this local test',
            core_and_affine_matrix_encoding='BF16',affine_bias_encoding='F32',
            complete_matrix_gate=5*arm['matrix_MAC_per_token']<=3*493961216,complete_logical_gate=5*logical<=3*988067328,
            physical_DRAM_and_rate='UNMEASURED')
        assert r['budget']['complete_matrix_gate'] and r['budget']['complete_logical_gate']
        def save(name,value,dtype='<f8'):
            a=np.ascontiguousarray(value,dtype=dtype);assert np.isfinite(a).all()
            path=args.directory/(name+'.npy')
            with path.open('xb') as f:np.save(f,a,allow_pickle=False)
            r['output_arrays'][name]=dict(path=str(path.resolve()),shape=list(a.shape),dtype=a.dtype.str,
                C_order=True,bytes=path.stat().st_size,sha256=sha(path));guard();return a
        def array(name):
            item=design['output_arrays'][name];assert sha(item['path'])==item['sha256']
            return np.load(item['path'],mmap_mode='r',allow_pickle=False)
        w=array('W');k=array('K');mu=array('mu');scale=array('scale');z=array('z');gamma_sel=array('selected_gamma');grad_sel=array('selected_gradient_x')
        q=np.load(b['Q_path'],mmap_mode='r',allow_pickle=False)
        plan=json.loads(Path(b['plan_path']).read_bytes());anchors=[a for a in plan['anchors'] if a['split']=='fit'];assert len(anchors)==16
        def decode(raw):
            a=np.frombuffer(raw,dtype='<u2');assert ((a&0x7fff)<0x7f80).all()
            return (a.astype('<u4')<<16).view('<f4')
        x=np.stack([decode(bytes.fromhex(a['x_BF16_HEX'])).astype(np.float64) for a in anchors]);ys=[]
        for a in anchors:
            with Path(a['binary_path']).open('rb') as f:f.seek(a['x_byte_offset']+1792);data=f.read(1792)
            assert len(data)==1792;ys.append(decode(data).astype(np.float64))
        y=np.stack(ys)
        def bf16(value):
            f32=np.asarray(value,dtype='<f4');assert np.isfinite(f32).all()
            bits=f32.view('<u4');encoded=((bits+0x7fff+((bits>>16)&1))>>16).astype('<u2')
            assert ((encoded&0x7fff)<0x7f80).all();return encoded
        core32=[]
        with zipfile.ZipFile(b['core_checkpoint']) as archive:
            prefix=next(n[:-8] for n in archive.namelist() if n.endswith('/data.pkl'))
            assert archive.read(prefix+'byteorder')==b'little'
            descriptors=Descriptors(io.BytesIO(archive.read(prefix+'data.pkl'))).load()
            for name,shape in (('shared_g',(512,896)),('shared_u',(512,896)),('shared_b',(896,512))):
                desc=descriptors[name];storage_desc=desc['storage'];assert storage_desc['dtype']=='FloatStorage' and desc['offset']==0 and desc['shape']==shape
                raw=archive.read(prefix+'data/'+storage_desc['key']);assert len(raw)==math.prod(shape)*4
                value=np.frombuffer(raw,dtype='<f4').reshape(shape);assert desc['stride']==tuple(t//4 for t in value.strides)
                encoded=save(name+'_BF16',bf16(value),'<u2');core32.append((encoded.astype('<u4')<<16).view('<f4'))
        core64=[a.astype(np.float64) for a in core32]
        def core_value_j(row,coeff,jacobian=False):
            gate,up,down=coeff;gx=gate@row;ux=up@row
            exp=np.exp(-np.abs(gx));sig=np.where(gx>=0,1/(1+exp),exp/(1+exp));silu=gx*sig
            value=down@(silu*ux)
            if not jacobian:return value
            derivative=sig+gx*sig*(1-sig)
            return value,down@((ux*derivative)[:,None]*gate+silu[:,None]*up)
        core_values=np.empty((16,896));core_j=np.empty((16,896,896))
        for i,row in enumerate(x):core_values[i],core_j[i]=core_value_j(row,core64,True);guard()
        save('core_FIT_values_F64',core_values);save('core_FIT_J_F64',core_j)
        r['new_shared_only_point_values']=16;r['new_shared_only_jacobians']=16
        source_j=np.load(b['source_J_path'],mmap_mode='r',allow_pickle=False)
        assert source_j.shape==(32,896,896) and source_j.dtype==np.dtype('<f8') and source_j.flags.c_contiguous
        residual_j=source_j[:16]-core_j
        perpendicular=residual_j-(residual_j@q)@q.T
        aperp=np.linalg.solve(w,perpendicular.reshape(16,-1)).reshape(16,896,896);guard()
        t=np.einsum('eod,id->ieo',aperp,x-mu,optimize=True)
        gamma=np.zeros((16,16,32))
        for i,a in enumerate(anchors):gamma[i,a['selected_parent_ids']]=gamma_sel[i]
        rhs=np.empty((16,33,896));rhs[:,0]=y-core_values-np.einsum('ie,ieo->io',w,t,optimize=True)
        row_j=(residual_j@q)*scale-np.einsum('ieo,ieq->ioq',t,gamma,optimize=True)
        rhs[:,1:]=row_j.transpose(0,2,1)
        coefficients=np.linalg.solve(k,rhs.reshape(528,896)).reshape(16,33,896);guard()
        c=coefficients[:,1:].transpose(0,2,1);matrices=aperp+(c/scale)@q.T
        bias=coefficients[:,0]-matrices@mu
        r['linear_solves']=dict(W_relative_residual=float(np.linalg.norm(w@aperp.reshape(16,-1)-perpendicular.reshape(16,-1))/np.linalg.norm(perpendicular)),
            K_relative_residual=float(np.linalg.norm(k@coefficients.reshape(528,896)-rhs.reshape(528,896))/np.linalg.norm(rhs)),
            A_perp_Q_relative=float(np.linalg.norm(aperp@q)/np.linalg.norm(aperp)),
            folded_matrix_max_absolute=float(np.max(np.abs(matrices))),folded_bias_max_absolute=float(np.max(np.abs(bias))))
        save('jet_RHS_F64',rhs);save('jet_coefficients_F64',coefficients)
        save('affine_A_F64',matrices);save('affine_bias_F64',bias)
        a_bf=save('affine_A_BF16',bf16(matrices),'<u2');b32=save('affine_bias_F32',bias,'<f4')
        a32=(a_bf.astype('<u4')<<16).view('<f4');a64=a32.astype(np.float64);bias64=b32.astype(np.float64)
        r['leaf_BF16_F32_hashes']=[hashlib.sha256(a_bf[e].tobytes()+b32[e].tobytes()).hexdigest() for e in range(16)]
        assert len(set(r['leaf_BF16_F32_hashes']))==16
        fit_value64=np.empty((16,896));fit_j64=np.empty((16,896,896));encoded_j64=np.empty_like(fit_j64);fit_value32=np.empty((16,896),dtype=np.float32)
        for i,row in enumerate(x):
            ids=anchors[i]['selected_parent_ids'];weights=w[i,ids];gradient=grad_sel[i]
            fields=matrices[ids]@row+bias[ids]
            fit_value64[i]=core_values[i]+weights@fields
            fit_j64[i]=core_j[i]+np.einsum('e,eod->od',weights,matrices[ids])+fields.T@gradient
            encoded_fields=a64[ids]@row+bias64[ids]
            encoded_j64[i]=core_j[i]+np.einsum('e,eod->od',weights,a64[ids])+encoded_fields.T@gradient
            f32fields=a32[ids]@row.astype(np.float32)+b32[ids]
            fit_value32[i]=core_value_j(row.astype(np.float32),core32)+np.asarray(anchors[i]['selected_mass_F32'],dtype=np.float32)@f32fields
            guard()
        j_energy=json.loads(Path(b['jacobian_result']).read_bytes())['pooled']['fit']['source_energy']['total']
        value_energy=math.fsum(float(v)**2 for v in y.ravel())
        r['FIT_constraints']=dict(source_value_energy=value_energy,reused_source_J_energy=j_energy,
            value_F64_relative_RMS=float(np.linalg.norm(fit_value64-y)/math.sqrt(value_energy)),
            J_F64_relative_RMS=float(np.linalg.norm(fit_j64-source_j[:16])/math.sqrt(j_energy)),
            encoded_value_F32_relative_RMS=float(np.linalg.norm(fit_value32.astype(np.float64)-y)/math.sqrt(value_energy)),
            encoded_smooth_J_F64_relative_RMS=float(np.linalg.norm(encoded_j64-source_j[:16])/math.sqrt(j_energy)))
        save('FIT_value_F64',fit_value64);save('FIT_value_F32',fit_value32,'<f4')
        save('FIT_J_F64',fit_j64);save('FIT_encoded_smooth_J_F64',encoded_j64)
        del fit_j64,encoded_j64,aperp,perpendicular,residual_j,t,a64,core_j
        numerical=all(r['linear_solves'][n]<=1e-8 for n in ('W_relative_residual','K_relative_residual','A_perp_Q_relative')) and all(r['FIT_constraints'][n]<=1e-8 for n in ('value_F64_relative_RMS','J_F64_relative_RMS'))
        r['constraint_gate']=numerical
        if not numerical:
            r['decision']='CLOSE_COUPLED_COMPILER_CONSTRAINTS'
        else:
            adoption=json.loads(Path(b['adoption_path']).read_bytes());groups=rows(adoption)
            saved=routes(Path(b['saved_E16_routes']),{s:len(g) for s,g in groups.items()})
            fit_keys={a['key'] for a in groups['fit']};dev=groups['development'];counts=Counter(a['key'] for a in dev)
            indices=[i for i,a in enumerate(dev) if a['key'] not in fit_keys];assert len(indices)==1495
            old=json.loads(Path(b['original_response_audit']).read_bytes())['response_audits']['F32']['development']['exact_necessary_failure']
            source_exact=int(old['source_energy_integer']);lcm=old['weight_LCM'];assert lcm==2
            predictions=[];journal=[];prefix_exact=0;prefix_error=0.;prefix_energy=0.
            def evaluate(index):
                item=dev[index];row=decode(item['key']);ids=saved['development']['ids'][index]
                weights=np.asarray(saved['development']['mass'][index],dtype=np.float32)
                prediction=core_value_j(row,core32)+weights@(a32[list(ids)]@row+b32[list(ids)])
                assert prediction.dtype==np.float32 and np.isfinite(prediction).all()
                with Path(item['binary_path']).open('rb') as f:f.seek(item['x_byte_offset']+1792);raw_y=f.read(1792)
                target=decode(raw_y);y_bits=np.frombuffer(raw_y,dtype='<u2');prediction_bits=prediction.view('<u4')
                multiplicity=counts[item['key']];assert lcm%multiplicity==0
                exact=sum((integer32(int(p))-integer32(int(v)<<16))**2 for p,v in zip(prediction_bits,y_bits))*(lcm//multiplicity)
                error=math.fsum((float(p)-float(v))**2 for p,v in zip(prediction,target))/multiplicity
                energy=math.fsum(float(v)**2 for v in target)/multiplicity
                metadata=dict(split_row_index=index,case_id=item['case_id'],category=adoption['cases'][item['case_index']]['category'],
                    local_input_row=item['local_input_row'],binary_path=item['binary_path'],x_byte_offset=item['x_byte_offset'],
                    x_SHA256=hashlib.sha256(item['key']).hexdigest(),y_SHA256=hashlib.sha256(raw_y).hexdigest(),
                    original_multiplicity=multiplicity,selected_ids=list(ids),selected_mass_F32_HEX=struct.pack('<4f',*weights).hex())
                guard();return prediction,metadata,exact,error,energy
            for index in indices[:64]:
                prediction,metadata,exact,error,energy=evaluate(index)
                predictions.append(prediction);journal.append(metadata);prefix_exact+=exact;prefix_error+=error;prefix_energy+=energy
            save('novel_prefix64_F32',np.stack(predictions),'<f4')
            write_once(args.directory/'novel_prefix64_journal.json',journal)
            failed=10000*prefix_exact>source_exact
            r['necessary_novel_failure']=dict(prefix_rows=64,total_novel_rows=1495,weight_LCM=lcm,
                common_square_units='2^-298',prefix_error_integer=str(prefix_exact),reused_FULL_source_energy_integer=str(source_exact),
                strict_integer_inequality='10000*NEW_prefix_error > reused_FULL_source_energy',necessarily_fails_1pct_RMS=failed,
                prefix_squared_error_over_FULL_energy=prefix_exact/source_exact,prefix_weighted_relative_RMS=math.sqrt(prefix_error/prefix_energy),
                complete_source_energy_recomputed=False,original_full_response_audit_sha256=sha(b['original_response_audit']))
            r['new_encoded_novel_responses']=64
            if failed:
                r['decision']='CLOSE_COUPLED_COMPILER_ENCODING_OR_NOVEL'
            else:
                # Continue only inconclusive prefix; preserve existing64 predictions.
                errors=[prefix_error];energies=[prefix_energy];category_error={};category_energy={}
                for index in indices[64:]:
                    prediction,metadata,_,error,energy=evaluate(index)
                    predictions.append(prediction);journal.append(metadata);errors.append(error);energies.append(energy)
                for prediction,metadata in zip(predictions,journal):
                    with Path(metadata['binary_path']).open('rb') as f:f.seek(metadata['x_byte_offset']+1792);target=decode(f.read(1792))
                    category=metadata['category'];mult=metadata['original_multiplicity']
                    category_error.setdefault(category,[]).append(math.fsum((float(p)-float(v))**2 for p,v in zip(prediction,target))/mult)
                    category_energy.setdefault(category,[]).append(math.fsum(float(v)**2 for v in target)/mult)
                all_rms=math.sqrt(math.fsum(errors)/math.fsum(energies))
                categories={c:math.sqrt(math.fsum(category_error[c])/math.fsum(category_energy[c])) for c in category_error}
                save('novel_ALL_F32',np.stack(predictions),'<f4');write_once(args.directory/'novel_ALL_journal.json',journal)
                r['novel_ALL']=dict(relative_RMS=all_rms,category_relative_RMS=categories)
                r['new_encoded_novel_responses']=1495
                r['decision']='ELIGIBLE_FOR_SEPARATE_NATIVE_COMPOSITION' if all_rms<=.01 and len(categories)==16 and all(v<=.03 for v in categories.values()) else 'CLOSE_COUPLED_COMPILER_ENCODING_OR_NOVEL'
        r['procedure_gates'].update(exact_used_inputs_and_original_source_J_reused=True,only_original16FIT_rhs=True,
            all_arrays_finite_C_order=True,complete_preserved_F32_router_cost=True,distinct16_encoded_coefficient_hashes=True,
            original_responses_controls_optimizer_not_replayed=True,no_Torch_GPU_download=True)
        r.update(elapsed_before_final_serialization=time.monotonic()-started,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            scope='One finite layer12 compiler; NumPy local F32 values and smooth F64 J, NOT native C parity/fresh chatbot/useful count/rate')
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],constraints=r['FIT_constraints'],necessary=r.get('necessary_novel_failure'),seconds=time.monotonic()-started)),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-started,OS_peak_snapshot=proc.memory_info().peak_wset)
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
