"""Independent fsum metrics and scalar recurrence on ALL saved lift operands."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
BEGAN=time.monotonic()


def extent(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for z in iter(lambda:f.read(1<<20),b''):h.update(z)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())


def metric(a,b):
    pairs=list(zip(a.flat,b.flat));assert all(math.isfinite(float(x)) and math.isfinite(float(y)) for x,y in pairs)
    num=math.fsum((float(x)-float(y))**2 for x,y in pairs)
    den=math.fsum(float(y)**2 for x,y in pairs)
    return math.sqrt(num)/max(math.sqrt(den),1e-300),max(abs(float(x)-float(y)) for x,y in pairs)


def main(a):
    import numpy as np
    import psutil
    proc=psutil.Process();proc.cpu_affinity([0])
    assert extent(a.report)['sha256']==a.report_sha
    assert extent(a.terminal)['sha256']==a.terminal_sha
    r=json.loads(a.report.read_bytes());t=json.loads(a.terminal.read_bytes())
    assert r['schema']=='FALCON_AUXILIARY_S6_TRANSPORT_RESULT_V1'
    assert t['exit_code']==0 and t['error'] is None and t['inputs_before_after_exact']
    assert t['result']==extent(a.report) and t['binding']==r['binding'] and t['outputs']==r['outputs']
    assert extent(r['binding']['path'])==r['binding']
    b=json.loads(Path(r['binding']['path']).read_bytes())
    inputs=[extent(__file__),extent(a.report),extent(a.terminal),r['binding'],*b['inputs'],*r['outputs']]
    def guard():
        assert time.monotonic()-BEGAN<=120 and proc.memory_info().peak_wset<=512<<20
        assert not proc.children(recursive=True)
    def check():
        for item in inputs:guard();assert extent(item['path'])==item,item['path']
    check();rows=[];max_delta=0
    with (Path(b['source'])/'model.safetensors').open('rb') as weights:
        size=int.from_bytes(weights.read(8),'little');header=json.loads(weights.read(size));base=8+size
        def coefficients(key):
            q=header[key];lo,hi=q['data_offsets'];weights.seek(base+lo);raw=weights.read(hi-lo)
            assert q['dtype'] in ('BF16','F32')
            if q['dtype']=='BF16':
                return [struct.unpack('<f',(int.from_bytes(raw[i:i+2],'little')<<16).to_bytes(4,'little'))[0]
                        for i in range(0,len(raw),2)]
            return list(struct.unpack('<'+str(len(raw)//4)+'f',raw))
        for record,item in zip(r['records'],r['outputs']):
            guard();site=record['site'];assert Path(item['path']).name==f'site{site:02d}.npz'
            measured={}
            with np.load(item['path'],allow_pickle=False) as z:
                for key,width in [('x',3072),('B',256),('C',256),('delta',48),('gate',3072),
                                  ('y',3072),('denominator',1),('output',2048)]:
                    actual=z['lifted_'+key];source=z['source_'+key]
                    assert actual.dtype==source.dtype==np.dtype('float64')
                    assert actual.shape==source.shape==(13,width)
                    rel,abs_max=metric(actual,source)
                    old=record['components'][key]
                    max_delta=max(max_delta,abs(rel-old['relative_RMS']))
                    assert rel<=b['gates']['component_relative_RMS']
                    assert abs(abs_max-old['max_abs'])<=1e-12
                    measured[key]=dict(relative_RMS=rel,max_abs=abs_max)
                A=coefficients(f'model.layers.{site}.mamba.A_log')[0]
                D=coefficients(f'model.layers.{site}.mamba.D')[0]
                state=[0.0]*256;predictions=[]
                for token in range(13):
                    dt=float(z['source_delta'][token,0]);xx=float(z['source_x'][token,0])
                    decay=math.exp(-math.exp(A)*dt)
                    state=[decay*s+dt*xx*float(bb) for s,bb in zip(state,z['source_B'][token])]
                    predictions.append(math.fsum(s*float(cc) for s,cc in zip(state,z['source_C'][token]))+D*xx)
                witness=metric(np.array(predictions),z['source_y'][:,0])[0]
                assert witness<=b['gates']['scalar_relative']
                max_delta=max(max_delta,abs(witness-record['scalar_recurrence_relative']))
                ablations={}
                for field,label in [('ablated_dt_output','omitted_negative_dt_pair_output_relative'),
                                    ('no_norm_output','omitted_postgate_norm_output_relative')]:
                    value=metric(z[field],z['source_output'])[0]
                    max_delta=max(max_delta,abs(value-record[label]));ablations[label]=value
                rows.append(dict(site=site,components=measured,scalar_recurrence_relative=witness,**ablations))
    assert len(rows)==3 and max_delta<=1e-12;check();guard()
    out=dict(schema='FALCON_AUXILIARY_S6_STORED_AUDIT_V1',source=extent(__file__),
        report=extent(a.report),terminal=extent(a.terminal),command=sys.orig_argv,
        all3_sites8_operand_metrics_39_positions_complete=True,
        all_saved_source_and_lifted_operand_elements_checked=True,
        independent_scalar_recurrence_3sites_complete=True,full_inputs_outputs_before_after_exact=True,
        max_report_metric_delta=max_delta,records=rows,seconds=time.monotonic()-BEGAN,
        OS_peak=proc.memory_info().peak_wset,limits=dict(seconds=120,OS_bytes=512<<20),
        scope='Independent Python fsum aggregate metrics and scalar first-channel recurrence from '
            'actual raw source A/D. No generator replay or independent full coefficient-map audit, '
            'no whole C/BF16/chatbot/speed/generality qualification.')
    with a.out.open('x',encoding='utf8',newline='\n') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(sites=3,max_metric_delta=max_delta,seconds=out['seconds'],OS_peak=out['OS_peak'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True)
    p.add_argument('--report-sha',required=True);p.add_argument('--terminal',type=Path,required=True)
    p.add_argument('--terminal-sha',required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
