"""Streamed format/route audit and certified whole-logit gates for a NEW cohort."""
import argparse
import hashlib
import json
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
    native=json.loads(a.native_result.read_bytes())
    exported=json.loads(a.export_result.read_bytes())
    learner=json.loads(a.learner_result.read_bytes())
    assert native['case_count']==160 and native['full_vocabulary_rows']==5746
    assert exported['model']==native['model'] and learner['decision']=='BALANCED_RECOVERY_ELIGIBLE'
    assert exported['checkpoint_sha256']==learner['checkpoint']['sha256']
    files=[a.native_result,a.native_result.with_suffix('.terminal.json'),a.export_result,
           a.export_result.with_suffix('.terminal.json'),a.learner_result,a.corpus,
           Path(native['model']['path']),Path(exported['manifest']['path']),Path(__file__),
           B/'chatbot_hybrid_logit_certificate.py',B/'chatbot_falcon_usability.py',
           B/'chatbot_falcon_usability_launch.py',Path(sys.executable),
           DOC/'CHATBOT_HYBRID_NATIVE_COHORT_AUDIT_PROTOCOL_20261009.md']
    assert json.loads(a.native_result.with_suffix('.terminal.json').read_bytes())['result_sha256']==sha(a.native_result)
    assert json.loads(a.native_result.with_suffix('.terminal.json').read_bytes())['exit_code']==0
    files += [Path(v['path']) for v in native['native_outputs']]
    files += [Path(native['queries']['path']),Path(native['queries']['path']).with_suffix('.json')]
    files += [Path(r['logits']['path']) for r in learner['after']['cases']]
    files += [Path(r['logits']['path']) for r in json.loads(a.corpus.read_bytes())['records']]
    write(a.out,dict(schema='HYBRID_NATIVE_COHORT_AUDIT_BINDING_V1',python=str(Path(sys.executable).resolve()),
          worker_path=str(Path(__file__).resolve()),native_result=str(a.native_result.resolve()),
          export_result=str(a.export_result.resolve()),learner_result=str(a.learner_result.resolve()),
          corpus=str(a.corpus.resolve()),limits=dict(seconds=600,OS_bytes=4<<30,output_bytes=16<<20),
          runtime_binding_scope='All saved NEW cohort/model/reference/trace/state packets and selected code/Python. NumPy F64 bounded arithmetic with exact integer fallback; no source/student/native rerun or full native DLL certificate.',
          inputs=[dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)) for p in dict.fromkeys(files)]))
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(files))),flush=True)


def worker(a):
    start=time.monotonic()
    assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes())
    assert b['schema']=='HYBRID_NATIVE_COHORT_AUDIT_BINDING_V1'
    assert Path(sys.executable).resolve()==Path(b['python']).resolve() and sys.version_info[:3]==(3,12,10)
    a.directory.mkdir(exist_ok=False)
    try:
        import numpy as np
        import psutil
        from chatbot_hybrid_logit_certificate import certify
        assert np.__version__=='2.4.6' and psutil.__version__=='7.2.2' and sys.byteorder=='little'
        proc=psutil.Process();proc.cpu_affinity(list(range(6)))
        def guard():
            assert time.monotonic()-start<=600 and proc.memory_info().peak_wset<=4<<30
            assert not proc.children(recursive=True)
        native=json.loads(Path(b['native_result']).read_bytes())
        exported=json.loads(Path(b['export_result']).read_bytes())
        manifest=json.loads(Path(exported['manifest']['path']).read_bytes())
        learner=json.loads(Path(b['learner_result']).read_bytes())
        corpus=json.loads(Path(b['corpus']).read_bytes())
        metadata=json.loads(Path(native['queries']['path']).with_suffix('.json').read_bytes())
        sources={r['id']:r for r in corpus['records']}
        references={r['id']:r for r in learner['after']['cases']}
        assert len(metadata)==len(sources)==len(references)==160
        assert {r['id'] for r in metadata}==sources.keys()==references.keys()
        assert sha(native['queries']['path'])==native['queries']['sha256']
        with Path(native['queries']['path']).open('rb') as f:
            assert struct.unpack('<4I',f.read(16))==(0x31514853,160,5746,65537)
            for case in metadata:
                n,t=struct.unpack('<2I',f.read(8))
                assert 1<=n<=512 and 1<=t<=n
                assert np.fromfile(f,dtype='<u4',count=n).tolist()==case['input_ids']
                assert np.fromfile(f,dtype='<u4',count=t).tolist()==case['positions']
            assert f.read()==b''
        model=Path(native['model']['path'])
        assert sha(model)==native['model']['sha256']==exported['model']['sha256']
        with model.open('rb') as f:
            magic,*header=struct.unpack('<8s16IQ',f.read(80))
            assert magic==b'SLH1PK01' and header[:16]==[1,512,12,65537,72,8,128,768,256,48,16,4,128,(1<<5)|(1<<11),212,0x01020304]
            assert header[-1]==model.stat().st_size==425210736
            descriptors=[struct.unpack('<64sII4IQQ',f.read(104)) for _ in range(212)]
            cursor=22128;code_bytes=0;float_bytes=0;names=[]
            for record,expected in zip(descriptors,manifest['fields'],strict=True):
                name=record[0].split(b'\0')[0].decode();dtype,rank=record[1:3]
                dims=list(record[3:7]);offset,size=record[7:9]
                assert 1<=rank<=3 and name==expected['name'] and dtype==expected['dtype']
                assert dims[:rank]==expected['shape'] and not any(dims[rank:]) and all(d>0 for d in dims[:rank])
                assert offset==cursor and size==expected['bytes']==int(np.prod(dims[:rank]))*(4 if dtype==1 else 1)
                data=f.read(size)
                assert len(data)==size and hashlib.sha256(data).hexdigest()==expected['sha256']
                if dtype==2:
                    assert name.rsplit('.',1)[-1] in ('gate','up','down')
                    assert np.frombuffer(data,dtype=np.uint8).max()<=8
                    code_bytes+=size
                else:
                    assert dtype==1 and name.rsplit('.',1)[-1] not in ('gate','up','down')
                    values=np.frombuffer(data,dtype='<f4');assert np.isfinite(values).all()
                    if name.endswith('_scale'):assert values.min()>=np.float32(1e-8)
                    float_bytes+=size
                    del values
                cursor+=size;names.append(name);del data;guard()
            assert f.read()==b'' and cursor==425210736 and len(set(names))==212
            assert code_bytes==84934656 and float_bytes==340253952
        outputs={Path(v['path']).name:Path(v['path']) for v in native['native_outputs']}
        for item in native['native_outputs']:
            assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256']
            guard()
        total_inputs=sum(len(r['input_ids']) for r in metadata)
        total_rows=sum(len(r['positions']) for r in metadata)
        assert total_inputs==15999 and total_rows==5746
        assert outputs['logits.bin'].stat().st_size==16+total_rows*65537*4
        certificates=[];methods={};row_index=0
        with outputs['logits.bin'].open('rb') as f:
            assert struct.unpack('<4I',f.read(16))==(0x314c4853,160,5746,65537)
            for case in metadata:
                source=sources[case['id']]
                assert case['input_ids']==source['student_input_ids'] and case['positions']==source['positions']
                reference=references[case['id']]['logits']
                ref=np.fromfile(reference['path'],dtype='<f4').reshape(reference['shape'])
                assert ref.shape==(len(case['positions']),65537)
                actual=np.fromfile(f,dtype='<f4',count=ref.size).reshape(ref.shape)
                for j,r in enumerate(ref):
                    c=actual[j];expected=native['rows'][row_index]
                    certificate=certify(c,r)
                    assert expected['id']==case['id'] and expected['position']==case['positions'][j]
                    assert certificate['RMS_gate']==expected['RMS_gate']
                    estimate=certificate['relative_RMS']
                    if estimate is not None:
                        assert abs(estimate-expected['relative_logit_RMS'])<=1e-12
                    c_id=int(c.argmax());r_id=int(r.argmax())
                    assert c_id==expected['C_ID'] and r_id==expected['learner_ID']
                    assert (c_id==r_id)==expected['winner_equal']
                    certificates.append(dict(id=case['id'],position=case['positions'][j],C_ID=c_id,learner_ID=r_id,**certificate))
                    methods[certificate['method']]=methods.get(certificate['method'],0)+1
                    row_index+=1
                guard()
            assert f.read()==b'' and row_index==5746
        dtype=np.dtype([(n,'<f4',512) for n in ('input','core_input','core_output','ff_input')]+[
            ('scores','<f4',72),('ids','<u4',8),('mass','<f4',8),('ff_output','<f4',512),('output','<f4',512)])
        assert dtype.itemsize==12640 and outputs['trace.bin'].stat().st_size==total_inputs*12*dtype.itemsize
        visits=np.zeros((12,72),dtype=np.int64);count=0;mass_error=0.0
        with outputs['trace.bin'].open('rb') as f:
            while True:
                trace=np.fromfile(f,dtype=dtype,count=1024)
                if not len(trace):break
                for name in dtype.names:assert np.isfinite(trace[name]).all()
                ids=np.argsort(-trace['scores'],axis=-1,kind='stable')[:,:8].astype('<u4')
                assert np.array_equal(ids,trace['ids'])
                chosen=np.take_along_axis(trace['scores'].astype(np.float64),ids.astype(np.int64),-1)
                mass=np.exp(chosen-chosen.max(-1,keepdims=True));mass/=mass.sum(-1,keepdims=True)
                mass_error=max(mass_error,float(np.abs(mass-trace['mass']).max()));assert mass_error<=1e-6
                layer_ids=(count+np.arange(len(trace)))%12
                for site in range(12):
                    visits[site]+=np.bincount(ids[layer_ids==site].flatten().astype(np.int64),minlength=72)
                count+=len(trace);guard()
        assert count==total_inputs*12 and (visits.sum(-1)==total_inputs*8).all()
        assert outputs['states.bin'].stat().st_size==160*9117696
        with outputs['states.bin'].open('rb') as f:
            while True:
                state=np.fromfile(f,dtype='<f4',count=1<<18)
                if not len(state):break
                assert np.isfinite(state).all();guard()
        failed=sum(not r['RMS_gate'] for r in certificates)
        winners=sum(r['C_ID']==r['learner_ID'] for r in certificates)
        certified='NATIVE_PREFIX_PARITY_PASS' if failed==0 and winners==5746 else 'NATIVE_PREFIX_PARITY_FAIL'
        assert certified==native['decision']
        assert native['winners_equal']==winners and native['gates']['all_full_logit_RMS']==(failed==0)
        assert native['gates']['all_winners_equal']==(winners==5746)
        guard()
        report=dict(schema='HYBRID_NATIVE_COHORT_AUDIT_RESULT_V1',decision='SAVED_NATIVE_COHORT_AUDIT_PASS',
             freeze=a.freeze,binding_sha256=a.binding_sha,process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
             certified_native_decision=certified,RMS_pass_rows=5746-failed,RMS_fail_rows=failed,
             certified_rows=5746,certificate_methods=methods,certificates=certificates,winners_equal=winners,
             packed_fields=212,packed_codes_bytes=code_bytes,F32_payload_bytes=float_bytes,
             expert_master_or_unpacked_in_model_bytes=0,router_records=count,router_visits=visits.tolist(),
             max_router_mass_abs_error=mass_error,elapsed_seconds=time.monotonic()-start,
             worker_OS_peak_snapshot=proc.memory_info().peak_wset,source_calls=0,student_calls=0,native_calls=0,
             scope='Saved format/state/routes and certified F32 logit predicate,not a source-quality/accepted-rate or useful-capacity admission. Direct source metrics remain the native producer observations.')
        write(a.out,report)
        print(json.dumps(dict(decision=report['decision'],certified=certified,methods=methods,seconds=report['elapsed_seconds'])),flush=True)
    except BaseException as error:
        write(a.directory/'first_failure.json',dict(fault=repr(error),elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--bind',action='store_true')
    p.add_argument('--native-result',type=Path)
    p.add_argument('--export-result',type=Path)
    p.add_argument('--learner-result',type=Path)
    p.add_argument('--corpus',type=Path)
    p.add_argument('--binding',type=Path)
    p.add_argument('--binding-sha')
    p.add_argument('--freeze')
    p.add_argument('--directory',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();bind(a) if a.bind else worker(a)
