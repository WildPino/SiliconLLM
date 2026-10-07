"""Independent scalar/descriptor audit of saved bytes, no scientific imports."""
import argparse
import ast
from collections import Counter,OrderedDict
import datetime as dt
import io
import json
import math
import os
from pathlib import Path
import pickle
import struct
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once


class Descriptors(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)==('collections','OrderedDict'):return OrderedDict
        if module=='torch' and name in ('LongStorage','FloatStorage'):return name
        if (module,name)==('torch._utils','_rebuild_tensor_v2'):return self.rebuild
        raise ValueError('Forbidden pickle class: '+module+'.'+name)
    def persistent_load(self,value):
        assert len(value)==5 and value[0]=='storage' and value[1] in ('LongStorage','FloatStorage')
        assert value[3]=='cpu' and isinstance(value[2],str) and isinstance(value[4],int)
        return dict(dtype=value[1],key=value[2],count=value[4])
    @staticmethod
    def rebuild(storage,offset,shape,stride,requires_grad,hooks,metadata=None):
        assert not requires_grad and not hooks and metadata is None
        return dict(storage=storage,offset=offset,shape=shape,stride=stride)


def routes(path,sizes):
    with zipfile.ZipFile(path) as archive:
        prefix=next(n[:-len('data.pkl')] for n in archive.namelist() if n.endswith('/data.pkl'))
        assert archive.read(prefix+'byteorder')==b'little'
        tensors=Descriptors(io.BytesIO(archive.read(prefix+'data.pkl'))).load()
        assert list(tensors)==['fit','development']
        result={}
        for split,n in sizes.items():
            assert list(tensors[split])==['ids','mass']
            result[split]={}
            for field,dtype,width,format_code in (('ids','LongStorage',8,'q'),('mass','FloatStorage',4,'f')):
                descriptor=tensors[split][field];storage=descriptor['storage']
                assert descriptor['offset']==0 and descriptor['shape']==(n,4) and descriptor['stride']==(4,1)
                assert storage['dtype']==dtype and storage['count']==n*4
                raw=archive.read(prefix+'data/'+storage['key']);assert len(raw)==n*4*width
                result[split][field+'_bytes']=raw
                result[split][field]=list(struct.iter_unpack('<4'+format_code,raw))
        return result


def operands(adoption,guard):
    result={s:dict(keys=[],ys=[],y_bits=[],case=[],category=[],generated=[]) for s in ('fit','development')}
    for case_index,case in enumerate(adoption['cases']):
        assert sha(case['binary_path'])==case['binary_SHA256']
        g=result[case['split']]
        with Path(case['binary_path']).open('rb') as f:
            assert struct.unpack('<8s4I',f.read(24))==(b'QWCAP001',24,896,2,0)
            for row in range(case['captured_rows']):
                f.seek(24+row*86016+12*3584)
                key=f.read(1792);raw_y=f.read(1792);assert len(key)==len(raw_y)==1792
                bits=struct.unpack('<896H',raw_y);assert all((v&0x7fff)<0x7f80 for v in bits)
                y=struct.unpack('<896f',struct.pack('<896I',*(v<<16 for v in bits)))
                g['keys'].append(key);g['ys'].append(y);g['y_bits'].append(bits)
                g['case'].append(case_index);g['category'].append(case['category'])
                g['generated'].append(row>=len(case['prompt_ids']))
        guard()
    fit_keys=set(result['fit']['keys'])
    for split,g in result.items():
        g['counts']=Counter(g['keys']);g['novel']=[key not in fit_keys for key in g['keys']]
        g['energy']=[math.fsum(v*v for v in y) for y in g['ys']]
    return result


def audit_exposure(directory,data,guard):
    sizes={s:len(g['keys']) for s,g in data.items()}
    saved={e:routes(directory/f'E{e}.routes_F32.pt',sizes) for e in (16,160)}
    for split in sizes:
        assert saved[16][split]['mass_bytes']==saved[160][split]['mass_bytes']
        assert all(tuple(v//10 for v in large)==small for small,large in zip(saved[16][split]['ids'],saved[160][split]['ids']))
    outcomes={}
    for e in (16,160):
        original=json.loads((directory/f'E{e}.exposure.json').read_bytes());eligible=True;failed=[]
        for leaf in range(e):
            leaf_good=True
            for split,g in data.items():
                ids=saved[e][split]['ids'];mass=saved[e][split]['mass']
                assert all(len(set(row))==4 and all(0<=v<e for v in row) for row in ids)
                assert all(all(math.isfinite(v) and 0<=v<=1 for v in row) and abs(math.fsum(row)-1)<=2e-6 for row in mass)
                selected=[i for i,row in enumerate(ids) if leaf in row and (split=='fit' or g['novel'][i])]
                count=len({g['keys'][i] for i in selected});cases=sorted({g['case'][i] for i in selected})
                expected=original['leaves'][str(leaf)][split]
                assert (len(selected),count,cases)==(expected['selected_occurrences'],expected['distinct_x'],expected['conversations'])
                leaf_good &= count >= (8 if split=='fit' else 4) and (split!='fit' or len(cases)>=2)
            eligible &= leaf_good
            if not leaf_good:failed.append(leaf)
            guard()
        assert eligible==original['eligible'];outcomes[str(e)]=dict(all_exact_exposure_fields_match=True,eligible=eligible,failed_leaf_ids=failed)
    return dict(arms=outcomes,all_parent_choice_and_mass_bytes_identical=True)


def npy_rows(path,n):
    with path.open('rb') as f:
        assert f.read(8)==b'\x93NUMPY\x01\x00'
        length=struct.unpack('<H',f.read(2))[0];header=ast.literal_eval(f.read(length).decode('latin1'))
        assert header==dict(descr='<f4',fortran_order=False,shape=(n,896))
        assert path.stat().st_size==10+length+n*896*4
        for _ in range(n):
            raw=f.read(896*4);assert len(raw)==896*4
            row=struct.unpack('<896f',raw);assert all(math.isfinite(v) for v in row)
            yield row,struct.unpack('<896I',raw)
        assert not f.read(1)


def integer32(bits):
    exponent=(bits>>23)&255;mantissa=bits&0x7fffff;assert exponent!=255
    value=mantissa if exponent==0 else (0x800000|mantissa)<<(exponent-1)
    return -value if bits>>31 else value


def near(a,b):
    assert math.isfinite(a) and math.isfinite(b)
    assert abs(a-b)<=max(1e-12,1e-12*max(abs(a),abs(b))),(a,b)


def predictions(path,g,split,expected,guard):
    primary=[True]*len(g['keys']) if split=='fit' else g['novel']
    errors=[];prefix_error=0;prefix_rows=0
    primary_counts=[g['counts'][key] for key,keep in zip(g['keys'],primary) if keep]
    lcm=math.lcm(*set(primary_counts));source_exact=0
    if split=='development':
        for keep,raw_y,key in zip(primary,g['y_bits'],g['keys']):
            if keep:source_exact+=sum(integer32(v<<16)**2 for v in raw_y)*(lcm//g['counts'][key])
        guard()
    for i,(prediction,prediction_bits) in enumerate(npy_rows(path,len(g['keys']))):
        errors.append(math.fsum((v-y)**2 for v,y in zip(prediction,g['ys'][i])))
        if split=='development' and primary[i] and prefix_rows<64:
            exact=sum((integer32(v)-integer32(y<<16))**2 for v,y in zip(prediction_bits,g['y_bits'][i]))
            prefix_error+=exact*(lcm//g['counts'][g['keys'][i]]);prefix_rows+=1
        if i%128==0:guard()
    def metric(indices,weighted):
        if not indices:return dict(rows=0,squared_ratio=None,relative_RMS=None)
        weights=[1/g['counts'][g['keys'][i]] if weighted else 1 for i in indices]
        error=math.fsum(errors[i]*w for i,w in zip(indices,weights));energy=math.fsum(g['energy'][i]*w for i,w in zip(indices,weights))
        return dict(rows=len(indices),error_energy=error,source_energy=energy,squared_ratio=error/energy,relative_RMS=math.sqrt(error/energy))
    indices=[i for i,keep in enumerate(primary) if keep]
    metrics=dict(occurrences=metric(indices,False),unique_x_weighted=metric(indices,True),
        all_captured_occurrences=metric(list(range(len(primary))),False),
        generated=metric([i for i in indices if g['generated'][i]],True),prompt=metric([i for i in indices if not g['generated'][i]],True),
        categories={c:metric([i for i in indices if g['category'][i]==c],True) for c in sorted(set(g['category']))})
    def compare(a,b):
        assert a.keys()==b.keys()
        for key,value in a.items():
            if isinstance(value,dict):compare(value,b[key])
            elif value is None:assert b[key] is None
            elif isinstance(value,int):assert value==b[key]
            else:near(value,b[key])
    compare(metrics,{k:expected[k] for k in metrics})
    certificate=None
    if split=='development':
        assert prefix_rows==64 and 10000*prefix_error>source_exact
        certificate=dict(prefix_novel_rows=64,source_novel_rows=len(indices),weight_LCM=lcm,
            source_energy_integer=str(source_exact),prefix_error_integer=str(prefix_error),
            common_square_units='2^-298; cancel in ratio',strict_integer_inequality='10000*prefix_error > FULL_source_energy',
            full_saved_1pct_RMS_criterion_necessarily_fails=True,prefix_SSE_over_FULL_source_energy=prefix_error/source_exact)
    return dict(all_scalar_metrics_match=True,metrics=metrics,exact_necessary_failure=certificate)


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_JOINT_RETAINED_AUDIT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        gates={},new_source_full_forwards=0,new_function_predictions=0,new_optimizer_steps=0)
    def guard():
        assert time.monotonic()-start<=120 and proc.memory_info().peak_wset<=512<<20
        assert not proc.children(recursive=True)
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['schema']=='QWEN_JOINT_RETAINED_AUDIT_BINDING_V1'
        for item in b['inputs']:
            assert str(Path(item['path']).resolve())==item['resolved_path']
            assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256'],item['path'];guard()
        args.directory.mkdir();adoption=json.loads(Path(b['adoption_path']).read_bytes())
        data=operands(adoption,guard)
        r['exposure']=audit_exposure(Path(b['exposure_directory']),data,guard)
        r['gates']['ALL_saved_parent_mass_and_exposure_fields']=True
        baseline=json.loads(Path(b['baseline_result']).read_bytes());arm=baseline['arms'][0]
        r['response_audits']={}
        for encoding,diagnostic in arm['encodings'].items():
            r['response_audits'][encoding]={}
            for split,g in data.items():
                receipt=diagnostic['prediction_files'][split];path=Path(receipt['path'])
                assert path.stat().st_size==receipt['bytes'] and sha(path)==receipt['sha256']
                r['response_audits'][encoding][split]=predictions(path,g,split,diagnostic[split],guard)
        r['gates'].update(ALL_four_scalar_prediction_metric_audits=True,BOTH_exact_1pct_failure_certificates=True)
        r['decision']='RETAINED_E160_EXPOSURE_AND_E16_FIDELITY_FAILURES_INDEPENDENTLY_VERIFIED_NO_WHOLE_PROMOTION'
        r.update(elapsed_before_final_serialization=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],elapsed=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
