"""First saved-bank equation/encoding/response proof audit; no donor/core calls."""
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
from chatbot_joint_retained_audit import integer32,Descriptors,routes
from chatbot_directional_plan import rows


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_COUPLED_COMPILE_AUDIT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_source_or_shared_jacobians=0,new_encoded_function_responses=0)
    def guard():assert time.monotonic()-start<=90 and proc.memory_info().peak_wset<=2<<30 and not proc.children(recursive=True)
    try:
        assert sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='coupled_compile_audit'
        for item in b['inputs']:
            p=Path(item['path']);assert str(p.resolve())==item['resolved_path'] and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];guard()
        import numpy as np
        assert np.__version__=='2.4.6' and 'torch' not in sys.modules;args.directory.mkdir()
        raw=json.loads(Path(b['compile_path']).read_bytes());design=json.loads(Path(b['design_path']).read_bytes())
        plan=json.loads(Path(b['plan_path']).read_bytes());anchors=[a for a in plan['anchors'] if a['split']=='fit']
        def array(container,name):
            item=container['output_arrays'][name];path=Path(item['path']);assert sha(path)==item['sha256'] and path.stat().st_size==item['bytes']
            a=np.load(path,mmap_mode='r',allow_pickle=False)
            assert list(a.shape)==item['shape'] and a.dtype.str==item['dtype'] and a.flags.c_contiguous and np.isfinite(a).all()
            assert a.offset+a.nbytes==path.stat().st_size;return a
        w=array(design,'W');k=array(design,'K');mu=array(design,'mu');scale=array(design,'scale');gamma=array(design,'selected_gamma')
        gradient=array(design,'selected_gradient_x');q=np.load(b['Q_path'],mmap_mode='r',allow_pickle=False)
        a=array(raw,'affine_A_F64');bias=array(raw,'affine_bias_F64');coef=array(raw,'jet_coefficients_F64');rhs=array(raw,'jet_RHS_F64')
        core_j=array(raw,'core_FIT_J_F64');core_y=array(raw,'core_FIT_values_F64')
        source=np.load(b['source_J_path'],mmap_mode='r',allow_pickle=False)[:16]
        def decode(data):return np.asarray(struct.unpack('<896f',struct.pack('<896I',*(v<<16 for v in struct.unpack('<896H',data)))),dtype=np.float64)
        x=[];y=[]
        for item in anchors:
            x.append(decode(bytes.fromhex(item['x_BF16_HEX'])))
            with Path(item['binary_path']).open('rb') as f:f.seek(item['x_byte_offset']+1792);y.append(decode(f.read(1792)))
        x=np.stack(x);y=np.stack(y)
        # Recover null components from the folded full bank; producer used the
        # source residual first, not the final A. Check the independent identity.
        aperp=a-(a@q)@q.T;residual=source-core_j;target=residual-(residual@q)@q.T
        w_res=float(np.linalg.norm(w@aperp.reshape(16,-1)-target.reshape(16,-1))/np.linalg.norm(target))
        k_res=float(np.linalg.norm(k@coef.reshape(528,896)-rhs.reshape(528,896))/np.linalg.norm(rhs))
        chart_res=float(np.linalg.norm((a@q)*scale-coef[:,1:].transpose(0,2,1))/np.linalg.norm(coef[:,1:]))
        bias_res=float(np.linalg.norm(bias+a@mu-coef[:,0])/max(1.,np.linalg.norm(coef[:,0])))
        r['equation_residuals']=dict(W_from_folded_bank=w_res,K=k_res,chart_coefficients_from_full_bank=chart_res,folded_bias=bias_res)
        assert all(v<=1e-8 for v in r['equation_residuals'].values())
        # Reconstruct each RHS with scalar anchor/leaf sums, rather than einsum.
        expected_rhs=np.empty((16,33,896))
        for i,item in enumerate(anchors):
            ts=[aperp[e]@(x[i]-mu) for e in range(16)]
            expected_rhs[i,0]=y[i]-core_y[i]-sum(float(w[i,e])*ts[e] for e in range(16))
            row=(residual[i]@q)*scale
            for selected,e in enumerate(item['selected_parent_ids']):row-=ts[e][:,None]*gamma[i,selected][None,:]
            expected_rhs[i,1:]=row.T;guard()
        rhs_gap=float(np.linalg.norm(expected_rhs-rhs)/np.linalg.norm(rhs));assert rhs_gap<=1e-8
        r['equation_residuals']['independent_RHS_from_folded_null']=rhs_gap
        del aperp,residual,target,expected_rhs
        # Alternative nearest-even rounding: high-half/remainder predicate,
        # independently of the producer's unsigned rounding-bias addition.
        a_bits=np.asarray(a,dtype='<f4').view('<u4');high=a_bits>>16;low=a_bits&0xffff
        rounded=(high+((low>0x8000)|((low==0x8000)&((high&1)!=0)))).astype('<u2')
        encoded=array(raw,'affine_A_BF16');bias32=array(raw,'affine_bias_F32')
        assert np.array_equal(rounded,encoded) and np.array_equal(np.asarray(bias,dtype='<f4'),bias32)
        hashes=[hashlib.sha256(encoded[e].tobytes()+bias32[e].tobytes()).hexdigest() for e in range(16)]
        assert hashes==raw['leaf_BF16_F32_hashes'] and len(set(hashes))==16
        del a_bits,high,low,rounded
        with zipfile.ZipFile(b['core_checkpoint']) as archive:
            key=next(n[:-8] for n in archive.namelist() if n.endswith('/data.pkl'))
            descriptors=Descriptors(io.BytesIO(archive.read(key+'data.pkl'))).load()
            for name in ('shared_g','shared_u','shared_b'):
                desc=descriptors[name];original=archive.read(key+'data/'+desc['storage']['key'])
                bits=np.frombuffer(original,dtype='<u4').reshape(desc['shape']);high=bits>>16;low=bits&0xffff
                expected=(high+((low>0x8000)|((low==0x8000)&((high&1)!=0)))).astype('<u2')
                assert np.array_equal(expected,array(raw,name+'_BF16'))
        f64=array(raw,'FIT_value_F64');f32=array(raw,'FIT_value_F32');j64=array(raw,'FIT_J_F64');encoded_j=array(raw,'FIT_encoded_smooth_J_F64')
        value_energy=math.fsum(float(v)**2 for v in y.ravel());j_energy=raw['FIT_constraints']['reused_source_J_energy']
        metrics=dict(source_value_energy=value_energy,reused_source_J_energy=j_energy,
            value_F64_relative_RMS=math.sqrt(math.fsum((float(p)-float(v))**2 for p,v in zip(f64.ravel(),y.ravel()))/value_energy),
            encoded_value_F32_relative_RMS=math.sqrt(math.fsum((float(p)-float(v))**2 for p,v in zip(f32.ravel(),y.ravel()))/value_energy),
            J_F64_relative_RMS=float(np.linalg.norm(j64-source)/math.sqrt(j_energy)),
            encoded_smooth_J_F64_relative_RMS=float(np.linalg.norm(encoded_j-source)/math.sqrt(j_energy)))
        for n,value in metrics.items():assert abs(value-raw['FIT_constraints'][n])<=max(1e-12,1e-12*abs(value))
        r['FIT_metrics']=metrics
        # F64 saved field formula certificate, core values/J reused as operands.
        formula_value=0.;formula_j=0.
        for i,item in enumerate(anchors):
            ids=item['selected_parent_ids'];weights=w[i,ids]
            fields=a[ids]@x[i]+bias[ids]
            predicted=core_y[i]+sum(float(weight)*field for weight,field in zip(weights,fields))
            derivative=core_j[i]+sum(float(weight)*a[e] for weight,e in zip(weights,ids))
            for selected,field in enumerate(fields):derivative+=np.outer(field,gradient[i,selected])
            formula_value=max(formula_value,float(np.linalg.norm(predicted-f64[i])/max(1.,np.linalg.norm(f64[i]))))
            formula_j=max(formula_j,float(np.linalg.norm(derivative-j64[i])/max(1.,np.linalg.norm(j64[i]))));guard()
        assert formula_value<=1e-10 and formula_j<=1e-10
        r['saved_F64_field_formula_gaps']=dict(value=formula_value,J=formula_j)
        groups=rows(json.loads(Path(b['adoption_path']).read_bytes()));fit_keys={v['key'] for v in groups['fit']}
        dev=groups['development'];counts=Counter(v['key'] for v in dev)
        indices=[i for i,v in enumerate(dev) if v['key'] not in fit_keys];assert len(indices)==1495
        prefix=array(raw,'novel_prefix64_F32');assert prefix.shape==(64,896) and prefix.dtype==np.dtype('<f4')
        journal_path=Path(raw['output_arrays']['novel_prefix64_F32']['path']).with_name('novel_prefix64_journal.json')
        journal=json.loads(Path(journal_path).read_bytes());assert len(journal)==64
        cached=routes(Path(b['saved_E16_routes']),{s:len(v) for s,v in groups.items()})
        old=json.loads(Path(b['original_response_audit']).read_bytes())['response_audits']['F32']['development']['exact_necessary_failure']
        numerator=0;errors=[];energies=[]
        for ordinal,index in enumerate(indices[:64]):
            record=journal[ordinal];item=dev[index]
            assert (record['split_row_index'],record['case_id'],record['local_input_row'],record['x_byte_offset'])==(index,item['case_id'],item['local_input_row'],item['x_byte_offset'])
            assert hashlib.sha256(item['key']).hexdigest()==record['x_SHA256']
            assert record['selected_ids']==list(cached['development']['ids'][index])
            assert record['selected_mass_F32_HEX']==struct.pack('<4f',*cached['development']['mass'][index]).hex()
            with Path(item['binary_path']).open('rb') as f:f.seek(item['x_byte_offset']+1792);raw_y=f.read(1792)
            assert hashlib.sha256(raw_y).hexdigest()==record['y_SHA256']
            multiplicity=counts[item['key']];assert multiplicity==record['original_multiplicity']
            ybits=struct.unpack('<896H',raw_y);pbits=struct.unpack('<896I',prefix[ordinal].tobytes());target=decode(raw_y)
            numerator+=sum((integer32(p)-integer32(v<<16))**2 for p,v in zip(pbits,ybits))*(old['weight_LCM']//multiplicity)
            errors.append(math.fsum((float(p)-float(v))**2 for p,v in zip(prefix[ordinal],target))/multiplicity)
            energies.append(math.fsum(float(v)**2 for v in target)/multiplicity);guard()
        denominator=int(old['source_energy_integer']);failed=10000*numerator>denominator
        proof=raw['necessary_novel_failure']
        assert str(numerator)==proof['prefix_error_integer'] and str(denominator)==proof['reused_FULL_source_energy_integer']
        assert failed==proof['necessarily_fails_1pct_RMS'] and failed
        rms=math.sqrt(math.fsum(errors)/math.fsum(energies));assert abs(rms-proof['prefix_weighted_relative_RMS'])<=1e-12*max(1.,rms)
        r['exact_necessary_proof']=dict(prefix_error_integer=str(numerator),reused_FULL_source_energy_integer=str(denominator),
            full_1pct_RMS_necessarily_fails=True,prefix_weighted_relative_RMS=rms,source_denominator_recomputed=False)
        assert raw['decision']=='CLOSE_COUPLED_COMPILER_ENCODING_OR_NOVEL' and raw['new_encoded_novel_responses']==64
        arm=design['budget']['arms'][0];assert raw['budget']['logical_bytes']==arm['logical_coefficient_bytes_per_token']+2*arm['router_matrix_MAC']
        assert raw['budget']['stored_bytes']==arm['proposed_payload_bytes']+2*24*(896*32+16*32)
        r['procedure_gates'].update(first_bank_equation_and_F64_formula_certificates=True,independent_nearest_even_leaf_encoding=True,
            all_saved_FIT_error_metrics=True,all64_original_novel_prefix_positions_and_exact_failure=True,
            no_old_donor_core_or_encoded_response_replay=True,complete_preserved_F32_router_cost=True)
        r.update(decision='SAVED_COUPLED_COMPILER_FAILURE_INDEPENDENTLY_VERIFIED',elapsed_before_final_serialization=time.monotonic()-start,
            OS_peak_snapshot=proc.memory_info().peak_wset,ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],seconds=time.monotonic()-start)),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset)
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
