"""Complete saved metrics and independent source-to-F32 field relation audit.

Weighted input/output maps use a predeclared1e-6 sup-relative coefficient bound;
copied/selector/zero/control fields require exact F32 equality. No native parity.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from falcon_auxiliary_s6_transport import extent,tensor_reader,write
from falcon_auxiliary_s6_stored_audit import metric
START=time.monotonic()


def main(a):
    import numpy as np
    import psutil
    proc=psutil.Process();proc.cpu_affinity([0]);stream=None
    stage='identity';receipt=dict(schema='FALCON_PROJECTED_S6_STORED_AUDIT_TERMINAL_V1',
        source=extent(__file__),command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time())
    def guard():
        assert time.monotonic()-START<=120,'audit deadline'
        assert proc.memory_info().peak_wset<=1<<30,'audit OS cap'
        assert not proc.children(recursive=True),'unexpected child'
    try:
        assert extent(a.report)['sha256']==a.report_sha and extent(a.terminal)['sha256']==a.terminal_sha
        r=json.loads(a.report.read_bytes());t=json.loads(a.terminal.read_bytes())
        assert r['schema']=='FALCON_PROJECTED_S6_TRANSPORT_RESULT_V1'
        assert t['exit_code']==0 and t['error'] is None and t['inputs_before_after_exact']
        assert t['result']==extent(a.report) and t['binding']==r['binding'] and t['outputs']==r['outputs']
        assert extent(r['binding']['path'])==r['binding']
        b=json.loads(Path(r['binding']['path']).read_bytes())
        inputs=[extent(__file__),extent(a.report),extent(a.terminal),r['binding'],*b['inputs'],*r['outputs'],
                extent(ROOT/'benchmarks/native_expert_scaling/falcon_auxiliary_s6_stored_audit.py')]
        def check():
            for item in inputs:guard();assert extent(item['path'])==item,item['path']
        stage='before_hash';check()
        p=np.fromfile(b['basis']['path'],'<f4').reshape(2048,256).astype('<f8');G=p.T@p
        config=json.loads((Path(b['source'])/'config.json').read_bytes());eps=config['rms_norm_eps']
        get,stream=tensor_reader(Path(b['source'])/'model.safetensors',np)
        frames=[];max_delta=0;max_weight_delta=0
        for row in r['records']:
            stage=f"site{row['site']}";guard();site=row['site'];base=f'model.layers.{site}.'
            payload={Path(item['path']).name:item for item in r['outputs']}
            field=payload[f'site{site:02d}.coefficients.f32.npz'];frame=payload[f'site{site:02d}.frames.npz']
            with np.load(field['path'],allow_pickle=False) as packet:
                actual={key:packet[key] for key in packet.files}
            assert len(actual)==11 and all(value.dtype==np.dtype('float32') and np.isfinite(value).all() for value in actual.values())
            gamma=get(base+'input_layernorm.weight');master=get(base+'mamba.in_proj.weight')
            expected=np.zeros((7360,256),dtype='f8')
            # Independent direct donor maps, no compile_lift call or compiled full-D intermediate.
            def put(dst,lo,hi,mult):
                expected[dst:dst+hi-lo]=(master[lo:hi]@(gamma[:,None]*p))*(math.sqrt(8)*config['ssm_in_multiplier']*mult)
            mul=config['ssm_multipliers']
            put(3680,0,3072,mul[0]);put(0,3072,6144,mul[1])
            put(3072,6144,6400,mul[2]);put(3328,6400,6656,mul[3])
            put(3584,6656,6704,mul[4]);expected[3632:3680]=-expected[3584:3632]
            error=float(np.max(np.abs(actual['in_proj'].astype('f8')-expected))/max(float(np.max(np.abs(expected))),1e-300))
            assert error<=1e-6 and np.all(actual['in_proj'][6752:]==0)
            max_weight_delta=max(max_weight_delta,error);del master,expected
            out=get(base+'mamba.out_proj.weight');gamma_out=get(base+'mamba.norm.weight')
            expected_out=(p.T@out)*gamma_out[None,:]
            error_out=float(np.max(np.abs(actual['out_proj'][:,:3072].astype('f8')-expected_out))/max(float(np.max(np.abs(expected_out))),1e-300))
            assert error_out<=1e-6 and np.all(actual['out_proj'][:,3072:]==0)
            max_weight_delta=max(max_weight_delta,error_out);del out,expected_out
            selectors=np.zeros((560,3680),dtype='f4')
            selectors[np.arange(48),3584+np.arange(48)]=1
            selectors[np.arange(48),3632+np.arange(48)]=-1
            selectors[48+np.arange(256),3072+np.arange(256)]=1
            selectors[304+np.arange(256),3328+np.arange(256)]=1
            dt=np.zeros((3680,48),dtype='f4');dt[np.arange(3072),np.arange(3072)//64]=1
            conv=np.zeros((3680,4),dtype='f4');conv[:3584]=get(base+'mamba.conv1d.weight').reshape(3584,4);conv[3584:,-1]=1
            cb=np.zeros(3680,dtype='f4');cb[:3584]=get(base+'mamba.conv1d.bias')
            bias=np.zeros(3680,dtype='f4');bias[:3072]=np.repeat(get(base+'mamba.dt_bias'),64)
            A=get(base+'mamba.A_log');D=get(base+'mamba.D')
            decay=np.zeros((3680,256),dtype='f4');decay[:3072]=np.repeat(A,64)[:,None]
            skip=np.zeros(3680,dtype='f4');skip[:3072]=np.repeat(D,64)
            expected_control=dict(x_proj=selectors,dt_proj=dt,conv_w=conv,conv_b=cb,dt_b=bias,A_log=decay,Dskip=skip,
                                  input_metric=G.astype('f4'),input_epsilon=np.array(8*eps,dtype='f4'))
            for key,value in expected_control.items():assert np.array_equal(actual[key],value),key
            assert sum(value.nbytes for value in actual.values())==row['F32_package_bytes']
            measured={};rounded={};controls={}
            with np.load(frame['path'],allow_pickle=False) as z:
                assert z['z'].shape==z['target_norm'].shape==(13,256)
                norm=z['z']/np.sqrt(np.sum((z['z']@G)*z['z'],axis=1,keepdims=True)/256+8*eps)
                assert np.array_equal(norm,z['target_norm'])
                for key,width in [('x',3072),('B',256),('C',256),('delta',48),('gate',3072),('y',3072),('denominator',1),('output',256)]:
                    source=z['source_'+key];compact=z['compact_'+key];roundtrip=z['rounded_'+key]
                    assert source.shape==compact.shape==roundtrip.shape==(13,width)
                    rel,maximum=metric(compact,source);round_rel,_=metric(roundtrip,compact)
                    assert rel<=b['gates']['exact_relative_RMS'] and round_rel<=b['gates']['F32_coefficient_roundtrip_relative_RMS']
                    max_delta=max(max_delta,abs(rel-row['exact_components'][key]['relative_RMS']),
                                  abs(maximum-row['exact_components'][key]['max_abs']),
                                  abs(round_rel-row['F32_coefficient_roundtrip_components'][key]))
                    measured[key]=rel;rounded[key]=round_rel
                state=[0.0]*256;predictions=[]
                for pos in range(13):
                    dd=float(z['source_delta'][pos,0]);xx=float(z['source_x'][pos,0]);aa=math.exp(-math.exp(float(A[0]))*dd)
                    state=[aa*s+dd*xx*float(bb) for s,bb in zip(state,z['source_B'][pos])]
                    predictions.append(math.fsum(s*float(cc) for s,cc in zip(state,z['source_C'][pos]))+float(D[0])*xx)
                witness=metric(np.array(predictions),z['source_y'][:,0])[0]
                assert witness<=b['gates']['scalar_relative']
                max_delta=max(max_delta,abs(witness-row['scalar_recurrence_relative']))
                for field,label in [('identity_metric_output','identity_metric_output_relative'),
                                    ('wrong_epsilon_output','wrong_input_epsilon_output_relative'),
                                    ('diagonal_gamma_output','diagonal_gamma_output_relative')]:
                    value=metric(z[field],z['source_output'])[0];controls[label]=value
                    max_delta=max(max_delta,abs(value-row[label]))
            frames.append(dict(site=site,exact_metrics=measured,F32_coefficient_metrics=rounded,controls=controls,
                input_map_sup_relative=error,output_map_sup_relative=error_out,control_fields_F32_exact=True,
                scalar_recurrence_relative=witness))
            del actual,expected_control,selectors,decay
        stream.close();stream=None;stage='after_hash';check();guard()
        assert len(frames)==3 and max_delta<=1e-12
        result=dict(schema='FALCON_PROJECTED_S6_STORED_AUDIT_V1',source=extent(__file__),report=extent(a.report),
            terminal=extent(a.terminal),command=sys.orig_argv,records=frames,max_report_metric_delta=max_delta,
            max_independent_weight_map_sup_relative=max_weight_delta,weight_map_gate=1e-6,
            all39_positions8_fields_exact_and_roundtrip_metrics_checked=True,
            all_exported_parameter_fields_checked=True,full_inputs_outputs_before_after_exact=True,
            source_or_generator_or_model_or_native_or_optimizer_replay=False,
            seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
            scope='Independent direct source coefficient-map relations, exact control fields and fsum saved metrics; '
                'no independent generator prediction, F32 arithmetic/C parity, whole chatbot or speed qualification.')
        write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),stage=stage);raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--report-sha',required=True)
    p.add_argument('--terminal',type=Path,required=True);p.add_argument('--terminal-sha',required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
