"""Saved-only packed format/byte and exact-dyadic full-logit gate certificate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from chatbot_falcon_usability import SITE,sha,write
sys.path.insert(0,str(SITE))


def bind(a):
    r=json.loads(a.native_result.read_bytes())
    exported=json.loads((DOC/'chatbot_hybrid_export_result_20261008.json').read_bytes())
    pilot=DOC/'chatbot_hybrid_pilot_result_repair1_20261008.json'
    p=json.loads(pilot.read_bytes())
    files=[a.native_result,a.native_result.with_suffix('.terminal.json'),pilot,Path(r['model']['path']),
        Path(exported['manifest']['path']),Path(__file__),B/'chatbot_falcon_usability.py',
        B/'chatbot_falcon_usability_launch.py',DOC/'CHATBOT_HYBRID_NATIVE_AUDIT_PROTOCOL_20261009.md',Path(sys.executable)]
    files += [Path(v['path']) for v in r['native_outputs']]+[Path(r['queries']['path'])]
    files += [Path(v['logits']['path']) for v in p['after']['cases']]
    b=dict(schema='HYBRID_NATIVE_AUDIT_BINDING_V1',python=str(Path(sys.executable).resolve()),
        worker_path=str(Path(__file__).resolve()),native_result=str(a.native_result.resolve()),pilot=str(pilot.resolve()),
        export_manifest=str(Path(exported['manifest']['path']).resolve()),
        limits=dict(seconds=600,OS_bytes=2<<30,output_bytes=16<<20),
        runtime_binding_scope='Saved extents/code/Python; isolated NumPy version checked, no full native DLL tree rehash.',
        inputs=[dict(path=str(v.resolve()),bytes=v.stat().st_size,sha256=sha(v)) for v in dict.fromkeys(files)])
    write(a.out,b);print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(b['inputs']))),flush=True)


def units(bits):
    # Every finite IEEE754 F32 is an integer multiple of 2^-149.
    exponent=(bits>>23)&255
    assert exponent!=255
    m=bits&0x7fffff
    if exponent:m=(m|0x800000)<<(exponent-1)
    return -m if bits>>31 else m


def audit(a):
    start=time.monotonic();assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes());assert b['schema']=='HYBRID_NATIVE_AUDIT_BINDING_V1'
    a.directory.mkdir(exist_ok=False)
    import numpy as np
    import psutil
    assert np.__version__=='2.4.6' and psutil.__version__=='7.2.2'
    proc=psutil.Process();proc.cpu_affinity(list(range(6)))
    try:
        native=json.loads(Path(b['native_result']).read_bytes())
        pilot=json.loads(Path(b['pilot']).read_bytes())
        manifest=json.loads(Path(b['export_manifest']).read_bytes())
        model=Path(native['model']['path']);assert sha(model)==native['model']['sha256']
        with model.open('rb') as f:
            header=f.read(80);magic,*values=struct.unpack('<8s16IQ',header)
            assert magic==b'SLH1PK01' and values[:16]==[1,512,12,65537,72,8,128,768,256,48,16,4,128,(1<<5)|(1<<11),212,0x01020304]
            assert values[-1]==model.stat().st_size==425210736
            records=[struct.unpack('<64sII4IQQ',f.read(104)) for _ in range(212)]
            cursor=22128;codes=0;floats=0;names=[]
            for record,expected in zip(records,manifest['fields'],strict=True):
                name=record[0].split(b'\0')[0].decode();dtype,rank=record[1:3];dims=list(record[3:7]);offset,size=record[7:9]
                assert name==expected['name'] and dtype==expected['dtype'] and dims[:rank]==expected['shape']
                assert not any(dims[rank:]) and offset==cursor and size==expected['bytes']
                data=f.read(size);assert len(data)==size and hashlib.sha256(data).hexdigest()==expected['sha256']
                if dtype==2:
                    assert name.rsplit('.',1)[-1] in ('gate','up','down')
                    x=np.frombuffer(data,dtype=np.uint8);assert x.max()<=8;codes+=size
                else:
                    assert dtype==1 and name.rsplit('.',1)[-1] not in ('gate','up','down')
                    x=np.frombuffer(data,dtype='<f4');assert np.isfinite(x).all();floats+=size
                cursor+=size;names.append(name)
                del data,x
            assert f.read()==b'' and len(set(names))==212
            assert codes==84934656 and floats==340253952 and cursor==425210736
        outputs={Path(v['path']).name:Path(v['path']) for v in native['native_outputs']}
        raw=outputs['logits.bin'].read_bytes();assert struct.unpack_from('<4I',raw)==(0x314c4853,6,32,65537)
        actual=np.frombuffer(raw,offset=16,dtype='<f4').reshape(32,65537)
        certificates=[];row_index=0
        for case in pilot['after']['cases']:
            reference=np.fromfile(case['logits']['path'],dtype='<f4').reshape(case['logits']['shape'])
            for j,row in enumerate(reference):
                c=actual[row_index];expected=native['rows'][row_index]
                ci=c.view(np.uint32).tolist();ri=row.view(np.uint32).tolist()
                numerator=0;denominator=0
                for cb,rb in zip(ci,ri,strict=True):
                    cv,rv=units(cb),units(rb);numerator+=(cv-rv)**2;denominator+=rv*rv
                assert denominator>0
                passed=numerator*100000000<=denominator
                rms=math.sqrt(numerator/denominator)
                assert passed==expected['RMS_gate'] and abs(rms-expected['relative_logit_RMS'])<=1e-12
                assert int(c.argmax())==expected['C_ID'] and int(row.argmax())==expected['learner_ID']
                assert bool(c.argmax()==row.argmax())==expected['winner_equal']
                certificates.append(dict(id=case['id'],position=expected['position'],square_units='2^-298',
                    squared_error_integer=str(numerator),reference_energy_integer=str(denominator),
                    exact_test='error*100000000<=reference',RMS_gate=passed,relative_logit_RMS=rms,
                    C_ID=expected['C_ID'],learner_ID=expected['learner_ID']))
                row_index+=1
        assert row_index==32
        failed=sum(not r['RMS_gate'] for r in certificates)
        assert failed==19 and native['decision']=='NATIVE_PREFIX_PARITY_FAIL'
        assert native['winners_equal']==32 and native['gates']['all32_winners']
        dtype=np.dtype([(n,'<f4',512) for n in ('input','core_input','core_output','ff_input')]+[
            ('scores','<f4',72),('ids','<u4',8),('mass','<f4',8),('ff_output','<f4',512),('output','<f4',512)])
        trace=np.fromfile(outputs['trace.bin'],dtype=dtype);assert len(trace)==3132
        counts=[]
        for layer in range(12):
            chosen=trace['ids'][layer::12]
            count=np.bincount(chosen.flatten().astype(np.int64),minlength=72)
            assert count.sum()==261*8
            counts.append(dict(layer=layer,visits=count.tolist(),visited=int(np.count_nonzero(count)),
                               minimum=int(count.min()),maximum=int(count.max())))
        assert proc.memory_info().peak_wset<=2<<30 and time.monotonic()-start<=600
        report=dict(schema='HYBRID_NATIVE_AUDIT_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
            process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),decision='SAVED_NATIVE_AUDIT_PASS',
            certified_native_decision='NATIVE_PREFIX_PARITY_FAIL',RMS_pass_rows=32-failed,RMS_fail_rows=failed,
            all32_winners_equal=True,max_relative_logit_RMS=max(v['relative_logit_RMS'] for v in certificates),
            exact_certificates=certificates,packed_fields=212,packed_codes_bytes=codes,F32_payload_bytes=floats,
            expert_master_or_unpacked_in_model_bytes=0,router_visits=counts,elapsed_seconds=time.monotonic()-start,
            worker_OS_peak_snapshot=proc.memory_info().peak_wset,model_forwards=0,source_generations=0,training_updates=0,
            scope='Exact saved-logit gate and format bytes only; route exposure not useful capacity, no independent experimental replication.')
        write(a.out,report);print(json.dumps(dict(decision=report['decision'],certified_FAIL_rows=failed,seconds=report['elapsed_seconds'])),flush=True)
    except BaseException as error:
        write(a.directory/'first_failure.json',dict(fault=repr(error),elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('bind','audit'),default='audit')
    p.add_argument('--native-result',type=Path);p.add_argument('--binding',type=Path);p.add_argument('--binding-sha')
    p.add_argument('--freeze');p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();bind(a) if a.mode=='bind' else audit(a)
