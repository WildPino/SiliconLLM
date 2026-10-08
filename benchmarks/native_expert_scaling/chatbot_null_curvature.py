"""First contracted selector-null Hessians; no original forward/J reacquisition."""
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


def source_slices(binding):
    """Verify only the qualified header and three actually used BF16 extents."""
    spec=binding['source_slices'];path=Path(spec['path'])
    assert str(path.resolve())==spec['resolved_path'] and path.stat().st_size==spec['file_bytes']
    values=[]
    with path.open('rb') as stream:
        header=stream.read(spec['header_bytes'])
        assert len(header)==spec['header_bytes'] and hashlib.sha256(header).hexdigest()==spec['header_sha256']
        assert struct.unpack('<Q',header[:8])[0]+8==len(header)
        fields=json.loads(header[8:])
        for item in spec['tensors']:
            field=fields[item['name']]
            assert field['dtype']=='BF16' and field['shape']==item['shape']
            begin,end=field['data_offsets']
            assert item['offset']==len(header)+begin and item['bytes']==end-begin==2*math.prod(item['shape'])
            stream.seek(item['offset']);raw=stream.read(item['bytes'])
            assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
            values.append(raw)
    return values


def main(args):
    import psutil
    started=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_NULL_CURVATURE_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_source_or_core_full_J=0,new_response_or_optimizer_calls=0,
        new_source_H_contractions=0,new_core_H_contractions=0,new_directional_gradients_at_perturbed_states=0,anchors=[],outputs={})
    streams={}
    def guard():
        assert time.monotonic()-started<=180 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
        if args.directory.exists():assert sum(p.stat().st_size for p in args.directory.iterdir() if p.is_file())<=16<<20
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='null_curvature'
        for item in b['inputs']:
            path=Path(item['path']);assert str(path.resolve())==item['resolved_path'] and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'];guard()
        import numpy as np
        assert np.__version__=='2.4.6' and 'torch' not in sys.modules and os.environ['OPENBLAS_NUM_THREADS']=='1'
        args.directory.mkdir()
        def load(path,shape,dtype='<f8'):
            value=np.load(path,mmap_mode='r',allow_pickle=False)
            assert value.shape==shape and value.dtype==np.dtype(dtype) and value.flags.c_contiguous and value.offset+value.nbytes==Path(path).stat().st_size
            assert np.isfinite(value).all();return value
        def decode(raw,shape):
            value=np.frombuffer(raw,dtype='<u2').reshape(shape);assert ((value&0x7fff)<0x7f80).all()
            return (value.astype('<u4')<<16).view('<f4').astype('<f8')
        def save(name,value):
            path=args.directory/(name+'.npy')
            with path.open('xb') as stream:np.save(stream,np.ascontiguousarray(value,dtype='<f8'),allow_pickle=False)
        def output(name,shape):
            stream=(args.directory/(name+'.npy')).open('xb')
            np.lib.format.write_array_header_2_0(stream,dict(descr='<f8',fortran_order=False,shape=shape));streams[name]=stream
        def append(name,value):
            value=np.ascontiguousarray(value,dtype='<f8');assert np.isfinite(value).all()
            stream=streams[name];stream.write(value.tobytes());stream.flush();os.fsync(stream.fileno())
        def sig(value):
            exp=np.exp(-np.abs(value));return np.where(value>=0,1/(1+exp),exp/(1+exp))
        def energy(value):return math.fsum(float(t)*float(t) for t in np.asarray(value).flat)
        plan=json.loads(Path(b['plan_path']).read_bytes());anchors=plan['anchors']
        assert len(anchors)==32 and [a['split'] for a in anchors]==['fit']*16+['development']*16
        p=load(b['P_path'],(32,896));pn=load(b['Pi_N_path'],(896,896));sj=load(b['source_J_path'],(32,896,896))
        directions=[];direction_records=[]
        for j in (0,31,447,895):
            v=np.array(pn[:,j]);norm=float(np.linalg.norm(v));assert math.isfinite(norm) and norm>0
            v/=norm;unit=abs(float(np.linalg.norm(v))-1);null=float(np.linalg.norm(p@v)/np.linalg.norm(p))
            assert unit<=1e-12 and null<=1e-10
            directions.append(v);direction_records.append(dict(coordinate=j,projected_norm=norm,unit_gap=unit,Pv_over_P_Frobenius=null))
        directions=np.stack(directions);save('directions',directions);r['directions']=direction_records
        shapes=((4864,896),(4864,896),(896,4864))
        source=[decode(raw,shape) for raw,shape in zip(source_slices(b),shapes)]
        core=[decode(load(b['core_paths'][n],shape,'<u2').tobytes(),shape) for n,shape in zip(('g','u','b'),((512,896),(512,896),(896,512)))]
        models={};h=2**-10
        for name,coeff in (('source',source),('core',core)):
            g,u,down=coeff;gv=directions@g.T;uv=directions@u.T
            save(name+'_Gv',gv);save(name+'_Uv',uv);models[name]=(g,u,down,gv,uv)
            for suffix,shape in (('gx',(32,len(g))),('ux',(32,len(u))),('H',(32,4,896)),('gradients',(32,4,2,896))):output(name+'_'+suffix,shape)
        output('source_Jv',(32,4,896))
        journal=(args.directory/'anchors.jsonl').open('xb');streams['journal']=journal
        radius=json.loads(Path(b['radius_report_path']).read_bytes())['kernel']['radius']
        assert radius==.7645380726589079;rho=math.sqrt(896)*radius;r.update(radius=radius,rho=rho,h=h)
        for index,anchor in enumerate(anchors):
            guard();raw=bytes.fromhex(anchor['x_BF16_HEX']);assert hashlib.sha256(raw).hexdigest()==anchor['x_SHA256']
            x=decode(raw,(896,));hs={};checks=[]
            for name,(g,u,down,gv,uv) in models.items():
                gx=g@x;ux=u@x;s=sig(gx);a1=s+gx*s*(1-s);a2=s*(1-s)*(2+gx*(1-2*s))
                hidden=a2[None,:]*ux[None,:]*gv*gv+2*a1[None,:]*gv*uv
                curvature=hidden@down.T
                points=x[None,None,:]+h*directions[:,None,:]*np.array([1.,-1.])[None,:,None]
                pg=points@g.T;pu=points@u.T;ps=sig(pg);pa1=ps+pg*ps*(1-ps)
                gradients=(pa1*pu*gv[:,None,:]+pg*ps*uv[:,None,:])@down.T
                fd=(gradients[:,0]-gradients[:,1])/(2*h)
                for j in range(4):
                    hn=float(np.linalg.norm(curvature[j]));fn=float(np.linalg.norm(fd[j]));dn=float(np.linalg.norm(curvature[j]-fd[j]))
                    tolerance=2e-5*max(1e-6,hn,fn)+2e-10
                    checks.append(dict(model=name,direction=j,analytic_norm=hn,FD_norm=fn,discrepancy_norm=dn,tolerance=tolerance,passed=dn<=tolerance))
                for suffix,value in (('gx',gx),('ux',ux),('H',curvature),('gradients',gradients)):append(name+'_'+suffix,value)
                hs[name]=curvature;r['new_'+('source' if name=='source' else 'core')+'_H_contractions']+=4
                r['new_directional_gradients_at_perturbed_states']+=8
            jv=(sj[index]@directions.T).T;append('source_Jv',jv)
            se=energy(hs['source']);ce=energy(hs['core']);re=energy(hs['source']-hs['core']);je=energy(jv)
            assert se>0 and je>0
            record=dict(anchor_index=index,split=anchor['split'],case_id=anchor['case_id'],x_SHA256=anchor['x_SHA256'],
                source_curvature_energy=se,core_curvature_energy=ce,residual_curvature_energy=re,source_Jv_energy=je,
                residual_source_curvature_RMS=math.sqrt(re/se),chi=rho*math.sqrt(re/je),checks=checks)
            journal.write((json.dumps(record,separators=(',',':'),allow_nan=False)+'\n').encode());journal.flush();os.fsync(journal.fileno())
            r['anchors'].append(record);print(json.dumps(dict(anchor=index,split=anchor['split'],chi=record['chi'],seconds=time.monotonic()-started)),flush=True)
        for stream in streams.values():stream.close()
        streams={};guard()
        assert r['new_source_H_contractions']==r['new_core_H_contractions']==128 and r['new_directional_gradients_at_perturbed_states']==512
        for path in sorted(args.directory.glob('*.npy')):
            value=np.load(path,mmap_mode='r',allow_pickle=False);assert np.isfinite(value).all() and value.offset+value.nbytes==path.stat().st_size
            r['outputs'][path.stem]=dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path),shape=list(value.shape),dtype='<f8',C_order=True)
        pooled={}
        for split in ('fit','development'):
            rows=[a for a in r['anchors'] if a['split']==split];assert len(rows)==16
            totals={n:math.fsum(a[n] for a in rows) for n in ('source_curvature_energy','core_curvature_energy','residual_curvature_energy','source_Jv_energy')}
            pooled[split]=dict(**totals,residual_source_curvature_RMS=math.sqrt(totals['residual_curvature_energy']/totals['source_curvature_energy']),
                chi=rho*math.sqrt(totals['residual_curvature_energy']/totals['source_Jv_energy']),anchors_chi_at_least_point1=sum(a['chi']>=.1 for a in rows))
        dev=pooled['development'];qualified=all(c['passed'] for a in r['anchors'] for c in a['checks'])
        gates=dict(ALL256_Hessian_gradient_difference_checks=qualified,positive_pooled_source_curvature_and_Jv=all(v[n]>1e-12 for v in pooled.values() for n in ('source_curvature_energy','source_Jv_energy')),
            novel_curvature_residual_RMS_at_least_point1=dev['residual_source_curvature_RMS']>=.1,novel_chi_at_least_point1=dev['chi']>=.1,novel_at_least_eight_anchors_chi_at_least_point1=dev['anchors_chi_at_least_point1']>=8)
        source_slices(b);guard()
        r.update(pooled=pooled,diagnostic_gates=gates,decision=('NULL_CURVATURE_SUPPORTED_PRICE_ONE_CHANGED_REPRESENTATION' if all(gates.values()) else ('CLOSE_NULL_CURVATURE_QUALIFICATION' if not qualified else 'NULL_CURVATURE_UNSUPPORTED_REASSESS_COVERAGE')),
            scope='Smooth BF16-coefficient F64 field, four approximate null directions; not rounded-native derivative, response bound or fresh chatbot quality')
        r['procedure_gates']=dict(bound_inputs_and_source_extents_before_after=True,original32_anchors_and_original_J_bytes=True,
            fixed_four_directions_qualified=True,FIRST128_source_and128_core_H=True,FIRST512_perturbed_directional_gradients=True,
            no_original_forward_J_response_feature_experiment_replay=True,CPU_only_no_Torch=True)
        r.update(elapsed_before_final_serialization=time.monotonic()-started,OS_peak_snapshot=proc.memory_info().peak_wset,ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],pooled=pooled)),flush=True)
    except BaseException as error:
        for stream in streams.values():stream.close()
        r.update(fault=repr(error),elapsed=time.monotonic()-started,OS_peak_snapshot=proc.memory_info().peak_wset,ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
