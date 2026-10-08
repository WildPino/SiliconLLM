"""FIRST independent saved-scalar and decomposed-curvature formula audit."""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once


def main(args):
    import psutil
    started=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_NULL_CURVATURE_AUDIT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_source_or_core_full_J=0,new_scientific_acquisitions=0,new_optimizer_calls=0)
    def guard():
        assert time.monotonic()-started<=180 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='null_curvature_audit'
        for item in b['inputs']:
            path=Path(item['path']);assert str(path.resolve())==item['resolved_path'] and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'];guard()
        import numpy as np
        assert np.__version__=='2.4.6' and 'torch' not in sys.modules and os.environ['OPENBLAS_NUM_THREADS']=='1'
        args.directory.mkdir();raw=json.loads(Path(b['curvature_path']).read_bytes())
        def load(path,shape,dtype='<f8'):
            value=np.load(path,mmap_mode='r',allow_pickle=False)
            assert value.dtype==np.dtype(dtype) and value.shape==shape and value.flags.c_contiguous and np.isfinite(value).all()
            assert value.offset+value.nbytes==Path(path).stat().st_size;return value
        def saved(name,shape):
            item=raw['outputs'][name];assert sha(item['path'])==item['sha256'] and Path(item['path']).stat().st_size==item['bytes']
            assert item['shape']==list(shape) and item['dtype']=='<f8' and item['C_order'];return load(item['path'],shape)
        def bfdecode(data,shape):
            # Independent scalar IEEE BF16 promotion rather than producer bit-view.
            numbers=[]
            for (bits,) in struct.iter_unpack('<H',data):
                assert bits&0x7fff<0x7f80
                numbers.append(struct.unpack('<f',struct.pack('<I',bits<<16))[0])
            return np.array(numbers,dtype='<f8').reshape(shape)
        spec=b['source_slices'];source_path=Path(spec['path'])
        def coefficients():
            assert source_path.stat().st_size==spec['file_bytes'] and str(source_path.resolve())==spec['resolved_path']
            arrays=[]
            with source_path.open('rb') as stream:
                header=stream.read(spec['header_bytes']);assert hashlib.sha256(header).hexdigest()==spec['header_sha256']
                assert len(header)==struct.unpack('<Q',header[:8])[0]+8;fields=json.loads(header[8:])
                for item in spec['tensors']:
                    field=fields[item['name']];begin,end=field['data_offsets']
                    assert field['shape']==item['shape'] and field['dtype']=='BF16' and len(header)+begin==item['offset'] and end-begin==item['bytes']
                    stream.seek(item['offset']);data=stream.read(item['bytes']);assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
                    arrays.append(bfdecode(data,tuple(item['shape'])))
            return arrays
        source=coefficients()
        core=[bfdecode(load(b['core_paths'][n],shape,'<u2').tobytes(),shape) for n,shape in zip(('g','u','b'),((512,896),(512,896),(896,512)))]
        plan=json.loads(Path(b['plan_path']).read_bytes());anchors=plan['anchors'];assert len(anchors)==32
        x=np.stack([bfdecode(bytes.fromhex(a['x_BF16_HEX']),(896,)) for a in anchors])
        for a in anchors:assert hashlib.sha256(bytes.fromhex(a['x_BF16_HEX'])).hexdigest()==a['x_SHA256']
        v=saved('directions',(4,896));p=load(b['P_path'],(32,896));pn=load(b['Pi_N_path'],(896,896))
        for j,index in enumerate((0,31,447,895)):
            norm=math.sqrt(math.fsum(float(z)**2 for z in pn[:,index]));assert norm>0
            assert np.linalg.norm(v[j]-pn[:,index]/norm)<=1e-12
            assert abs(math.fsum(float(z)**2 for z in v[j])-1)<=2e-12
            projected=[math.fsum(float(a)*float(z) for a,z in zip(row,v[j])) for row in p]
            assert math.sqrt(math.fsum(z*z for z in projected))/np.linalg.norm(p)<=1e-10
        h=2**-10;radius=json.loads(Path(b['radius_report_path']).read_bytes())['kernel']['radius'];rho=math.sqrt(896)*radius
        assert (raw['h'],raw['radius'],raw['rho'])==(h,radius,rho)
        def scalar_derivatives(values):
            sig=[];first=[];second=[]
            for z in np.asarray(values).flat:
                z=float(z);e=math.exp(-abs(z));s=1/(1+e) if z>=0 else e/(1+e)
                # a'=s*(1+z*(1-s)); a''=(2*s*(1-s))+z*s*(1-s)*(1-2*s)
                sig.append(s);first.append(s*(1+z*(1-s)));second.append(2*s*(1-s)+z*s*(1-s)*(1-2*s))
            shape=np.shape(values);return tuple(np.array(a).reshape(shape) for a in (sig,first,second))
        gaps={};hs={};grads={}
        def check(name,actual,want,tol=1e-11):
            absolute=float(np.linalg.norm(actual-want));relative=absolute/max(float(np.linalg.norm(actual)),float(np.linalg.norm(want)),1e-12)
            assert absolute<=1e-11+tol*max(float(np.linalg.norm(actual)),float(np.linalg.norm(want))), (name,absolute,relative)
            gaps[name]=dict(absolute=absolute,relative=relative)
        for name,(g,u,down) in (('source',source),('core',core)):
            width=len(g);gx=saved(name+'_gx',(32,width));ux=saved(name+'_ux',(32,width))
            gv=saved(name+'_Gv',(4,width));uv=saved(name+'_Uv',(4,width))
            check(name+'_gx',gx,x@g.T);check(name+'_ux',ux,x@u.T)
            check(name+'_Gv',gv,(g@v.T).T);check(name+'_Uv',uv,(u@v.T).T)
            hs[name]=saved(name+'_H',(32,4,896));grads[name]=saved(name+'_gradients',(32,4,2,896))
            maximum_H=0.;maximum_grad=0.
            for i in range(32):
                _,a1,a2=scalar_derivatives(gx[i])
                # Separate two down contractions, unlike fused producer hidden sum.
                expected=((down*(a2*ux[i])[None,:])@(gv*gv).T+(down*a1[None,:])@(2*gv*uv).T).T
                check(name+'_H_'+str(i),hs[name][i],expected)
                maximum_H=max(maximum_H,gaps[name+'_H_'+str(i)]['relative'])
                # Independent affine-state contraction for perturbed-gradient audit.
                # Nominal gx±hGv versus direct G(x±hv) differs only by F64 arithmetic;
                # tolerance is declared, not asserted as exact rounded equality.
                nominal_g=gx[i][None,None,:]+h*gv[:,None,:]*np.array([1.,-1.])[None,:,None]
                nominal_u=ux[i][None,None,:]+h*uv[:,None,:]*np.array([1.,-1.])[None,:,None]
                s,a1,_=scalar_derivatives(nominal_g)
                expected_grad=((nominal_u*a1*gv[:,None,:])@down.T)+((nominal_g*s*uv[:,None,:])@down.T)
                check(name+'_grad_'+str(i),grads[name][i],expected_grad,1e-9)
                maximum_grad=max(maximum_grad,gaps[name+'_grad_'+str(i)]['relative']);guard()
            r[name+'_maximum_H_relative_formula_gap']=maximum_H;r[name+'_maximum_gradient_relative_formula_gap']=maximum_grad
        source_j=load(b['source_J_path'],(32,896,896));jv=saved('source_Jv',(32,4,896))
        maximum_jv=0.
        for i in range(32):
            check('source_Jv_'+str(i),jv[i],v@source_j[i].T)
            maximum_jv=max(maximum_jv,gaps['source_Jv_'+str(i)]['relative'])
            for j in range(4):
                for output in (0,31,447,895):
                    scalar=math.fsum(float(a)*float(z) for a,z in zip(source_j[i,output],v[j]))
                    assert abs(scalar-jv[i,j,output])<=1e-12+1e-11*abs(scalar)
        def energy(a):return math.fsum(float(z)**2 for z in np.asarray(a).flat)
        records=[];pairs_passed=0;maximum_fraction=0.
        for i,anchor in enumerate(anchors):
            old=raw['anchors'][i];assert old['anchor_index']==i and old['x_SHA256']==anchor['x_SHA256'] and old['split']==anchor['split'] and old['case_id']==anchor['case_id']
            se=energy(hs['source'][i]);ce=energy(hs['core'][i]);re=energy(hs['source'][i]-hs['core'][i]);je=energy(jv[i])
            new=dict(source_curvature_energy=se,core_curvature_energy=ce,residual_curvature_energy=re,source_Jv_energy=je,
                residual_source_curvature_RMS=math.sqrt(re/se),chi=rho*math.sqrt(re/je),split=anchor['split'])
            for key in new:
                if key=='split':continue
                assert math.isclose(new[key],old[key],rel_tol=2e-12,abs_tol=1e-14),key
            assert len(old['checks'])==8
            for m,name in enumerate(('source','core')):
                fd=(grads[name][i,:,0]-grads[name][i,:,1])/(2*h)
                for j in range(4):
                    hn=math.sqrt(energy(hs[name][i,j]));fn=math.sqrt(energy(fd[j]));dn=math.sqrt(energy(hs[name][i,j]-fd[j]));tol=2e-5*max(1e-6,hn,fn)+2e-10
                    saved_check=old['checks'][4*m+j]
                    assert (saved_check['model'],saved_check['direction'],saved_check['passed'])==(name,j,dn<=tol)
                    for key,val in (('analytic_norm',hn),('FD_norm',fn),('discrepancy_norm',dn),('tolerance',tol)):
                        assert math.isclose(saved_check[key],val,rel_tol=2e-12,abs_tol=1e-14),key
                    pairs_passed+=int(dn<=tol);maximum_fraction=max(maximum_fraction,dn/tol)
            records.append(new)
        pooled={}
        for split in ('fit','development'):
            group=[a for a in records if a['split']==split];assert len(group)==16
            total={key:math.fsum(a[key] for a in group) for key in ('source_curvature_energy','core_curvature_energy','residual_curvature_energy','source_Jv_energy')}
            total.update(residual_source_curvature_RMS=math.sqrt(total['residual_curvature_energy']/total['source_curvature_energy']),chi=rho*math.sqrt(total['residual_curvature_energy']/total['source_Jv_energy']),anchors_chi_at_least_point1=sum(a['chi']>=.1 for a in group))
            for key,val in total.items():assert math.isclose(val,raw['pooled'][split][key],rel_tol=2e-12,abs_tol=1e-14)
            pooled[split]=total
        dev=pooled['development'];gates=dict(ALL256_Hessian_gradient_difference_checks=pairs_passed==256,positive_pooled_source_curvature_and_Jv=all(z[n]>1e-12 for z in pooled.values() for n in ('source_curvature_energy','source_Jv_energy')),
            novel_curvature_residual_RMS_at_least_point1=dev['residual_source_curvature_RMS']>=.1,novel_chi_at_least_point1=dev['chi']>=.1,novel_at_least_eight_anchors_chi_at_least_point1=dev['anchors_chi_at_least_point1']>=8)
        expected_decision='NULL_CURVATURE_SUPPORTED_PRICE_ONE_CHANGED_REPRESENTATION' if all(gates.values()) else ('CLOSE_NULL_CURVATURE_QUALIFICATION' if pairs_passed!=256 else 'NULL_CURVATURE_UNSUPPORTED_REASSESS_COVERAGE')
        assert raw['diagnostic_gates']==gates and raw['decision']==expected_decision
        # Recheck only used source bytes, without redoing derivative arithmetic.
        with source_path.open('rb') as stream:
            assert hashlib.sha256(stream.read(spec['header_bytes'])).hexdigest()==spec['header_sha256']
            for item in spec['tensors']:
                stream.seek(item['offset']);assert hashlib.sha256(stream.read(item['bytes'])).hexdigest()==item['sha256']
        r.update(decision='SAVED_NULL_CURVATURE_INDEPENDENTLY_VERIFIED',verified_producer_decision=expected_decision,
            pooled=pooled,diagnostic_gates=gates,ALL256_pairs_passed=pairs_passed,maximum_FD_discrepancy_over_tolerance=maximum_fraction,maximum_saved_Jv_relative_gap=maximum_jv,
            verification_scope='FIRST saved linear-node/formula/scalar audit; independent scalar BF16/sigmoid and two-term down contractions; no new scientific namespace or original forward/J')
        r['procedure_gates']=dict(bound_inputs_source_extents_before_after=True,independent_scalar_BF16_and_SiLU_derivatives=True,
            ALL_saved_H_and_perturbed_gradient_formulas=True,ALL32_source_Jv_reused_bytes=True,ALL_saved_anchor_pooled_and256_FD_decisions=True,
            no_original_or_producer_acquisition_replay=True,CPU_only_no_Torch=True)
        r.update(elapsed_before_final_serialization=time.monotonic()-started,OS_peak_snapshot=proc.memory_info().peak_wset,ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],verified=expected_decision,pairs=pairs_passed,maximum_FD_fraction=maximum_fraction)),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-started,OS_peak_snapshot=proc.memory_info().peak_wset,ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
