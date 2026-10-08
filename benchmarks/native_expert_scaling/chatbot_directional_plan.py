"""Freeze original-state anchors without observing any new source derivative."""
import argparse
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
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once
from chatbot_joint_retained_audit import Descriptors,routes


def buffer(archive,prefix,descriptor,shape):
    storage=descriptor['storage'];stride=[];cursor=1
    for dim in reversed(shape):stride.insert(0,cursor);cursor*=dim
    assert descriptor['offset']==0 and descriptor['shape']==shape and descriptor['stride']==tuple(stride)
    assert storage['dtype']=='FloatStorage' and storage['count']==cursor
    raw=archive.read(prefix+'data/'+storage['key']);assert len(raw)==cursor*4
    values=struct.unpack('<'+'f'*cursor,raw);assert all(math.isfinite(v) for v in values)
    return raw,values


def geometry(path):
    with zipfile.ZipFile(path) as archive:
        prefix=next(n[:-8] for n in archive.namelist() if n.endswith('/data.pkl'))
        assert archive.read(prefix+'byteorder')==b'little'
        tensors=Descriptors(io.BytesIO(archive.read(prefix+'data.pkl'))).load()
        raw,p=buffer(archive,prefix,tensors['projection'],(32,896))
        _,centers=buffer(archive,prefix,tensors['parent_centers'],(16,32))
        _,norms=buffer(archive,prefix,tensors['parent_norms'],(16,))
    return raw,[p[i*896:(i+1)*896] for i in range(32)],[centers[i*32:(i+1)*32] for i in range(16)],norms


def rank_proof(raw):
    prime=1000000007
    assert prime>2 and all(prime%k for k in range(3,math.isqrt(prime)+1,2))
    bits=struct.unpack('<28672I',raw);powers=[pow(2,i,prime) for i in range(254)]
    values=[]
    for v in bits:
        exponent=(v>>23)&255;mantissa=v&0x7fffff;assert exponent!=255
        coefficient=mantissa if not exponent else ((0x800000|mantissa)*powers[exponent-1])%prime
        values.append((-coefficient if v>>31 else coefficient)%prime)
    matrix=[values[i*896:(i+1)*896] for i in range(32)]
    permutation=list(range(32));columns=[];pivots=[];determinant=1;row=0
    for col in range(896):
        candidates=[i for i in range(row,32) if matrix[i][col]]
        if not candidates:continue
        index=candidates[0]
        if index!=row:
            matrix[row],matrix[index]=matrix[index],matrix[row];permutation[row],permutation[index]=permutation[index],permutation[row]
            determinant=-determinant
        pivot=matrix[row][col];determinant=determinant*pivot%prime;pivots.append(pivot);columns.append(col)
        inverse=pow(pivot,-1,prime)
        matrix[row][col:]=[(v*inverse)%prime for v in matrix[row][col:]]
        for index in range(row+1,32):
            factor=matrix[index][col]
            if factor:matrix[index][col:]=[(v-factor*w)%prime for v,w in zip(matrix[index][col:],matrix[row][col:])]
        row+=1
        if row==32:break
    assert row==32 and determinant%prime!=0
    return dict(prime=prime,prime_trial_division_qualified=True,integer_coefficient_units='2^-149',
        pivot_columns=columns,row_permutation=permutation,pivots=pivots,minor_determinant_mod_prime=determinant%prime,
        exact_real_row_rank=32,exact_real_nullspace_dimension=864)


def rows(adoption):
    groups={s:[] for s in ('fit','development')}
    for case_index,case in enumerate(adoption['cases']):
        assert sha(case['binary_path'])==case['binary_SHA256']
        provenance=[]
        for frame in case['frames']:
            provenance.extend(dict(frame_step=frame['step'],within_frame=i,input_token_id=token) for i,token in enumerate(frame['input_ids']))
        assert len(provenance)==case['captured_rows']
        with Path(case['binary_path']).open('rb') as f:
            assert struct.unpack('<8s4I',f.read(24))==(b'QWCAP001',24,896,2,0)
            for local_index in range(case['captured_rows']):
                offset=24+local_index*86016+12*3584;f.seek(offset);key=f.read(1792);assert len(key)==1792
                groups[case['split']].append(dict(case_id=case['id'],case_index=case_index,
                    local_input_row=local_index,binary_path=case['binary_path'],x_byte_offset=offset,
                    key=key,provenance=provenance[local_index]))
    return groups


def shadow(row,ids,mass,projection,centers,norms):
    bits=struct.unpack('<896H',row['key']);assert all((v&0x7fff)<0x7f80 for v in bits)
    x=struct.unpack('<896f',struct.pack('<896I',*(v<<16 for v in bits)))
    q=[math.fsum(a*b for a,b in zip(vector,x)) for vector in projection]
    scores=[2*math.fsum(a*b for a,b in zip(center,q))-norm for center,norm in zip(centers,norms)]
    selected=[scores[i] for i in ids];other=[s for i,s in enumerate(scores) if i not in ids]
    margin=min(selected)-max(other);maximum=max(selected)
    exponential=[math.exp(s-maximum) for s in selected];total=math.fsum(exponential)
    shadow_mass=[v/total for v in exponential];gap=max(abs(a-b) for a,b in zip(shadow_mass,mass))
    return margin,gap,shadow_mass


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_DIRECTIONAL_PLAN_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_derivative_values=0,anchors=[])
    def guard():
        assert time.monotonic()-start<=45 and proc.memory_info().peak_wset<=512<<20
        assert not proc.children(recursive=True)
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['schema']=='QWEN_DIRECTIONAL_BINDING_V1' and b['job']['name']=='plan'
        for item in b['inputs']:
            assert str(Path(item['path']).resolve())==item['resolved_path']
            assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256'],item['path'];guard()
        args.directory.mkdir()
        raw,projection,centers,norms=geometry(Path(b['saved_geometry']))
        r['projection_rank_proof']=rank_proof(raw);guard()
        adoption=json.loads(Path(b['adoption_path']).read_bytes());groups=rows(adoption)
        saved=routes(Path(b['saved_E16_routes']),{s:len(g) for s,g in groups.items()})
        fit_keys={row['key'] for row in groups['fit']}
        for split,group in groups.items():
            used=set()
            for parent in range(16):
                candidates=[]
                for index,(row,ids,mass) in enumerate(zip(group,saved[split]['ids'],saved[split]['mass'])):
                    if parent not in ids or row['key'] in used or (split=='development' and row['key'] in fit_keys):continue
                    candidates.append((-mass[ids.index(parent)],index))
                candidates.sort();accepted=None;rejected=[]
                for _,index in candidates:
                    row=group[index];ids=saved[split]['ids'][index];mass=saved[split]['mass'][index]
                    margin,gap,shadow_mass=shadow(row,ids,mass,projection,centers,norms)
                    if margin>1e-7 and gap<=1e-6:
                        accepted=dict(split=split,parent=parent,split_row_index=index,case_id=row['case_id'],case_index=row['case_index'],
                            local_input_row=row['local_input_row'],binary_path=row['binary_path'],x_byte_offset=row['x_byte_offset'],
                            provenance=row['provenance'],x_BF16_HEX=row['key'].hex(),x_SHA256=hashlib.sha256(row['key']).hexdigest(),
                            selected_parent_ids=list(ids),selected_mass_F32=list(mass),selected_mass_F32_HEX=struct.pack('<4f',*mass).hex(),
                            interior_score_margin_F64=margin,shadow_mass_max_absolute_gap=gap,selected_shadow_mass_F64=shadow_mass,
                            earlier_rejected_candidates=rejected)
                        used.add(row['key']);break
                    rejected.append(dict(split_row_index=index,margin=margin,mass_gap=gap))
                    guard()
                assert accepted is not None,('No eligible distinct interior anchor',split,parent)
                r['anchors'].append(accepted);guard()
        assert len(r['anchors'])==32 and len({a['x_SHA256'] for a in r['anchors']})==32
        r['procedure_gates'].update(all32_distinct_original_anchors=True,novel_development_absent_from_FIT=True,
            exact_rank32_minor_proof=True,ALL_interior_and_shadow_mass_prerequisites=True)
        r.update(decision='ANCHORS_FROZEN_FOR_FIRST_DIRECTIONAL_VALUES',elapsed_before_final_serialization=time.monotonic()-start,
            OS_peak_snapshot=proc.memory_info().peak_wset,ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        guard();write_once(args.out,r)
        print(json.dumps(dict(decision=r['decision'],anchors=32,elapsed=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
