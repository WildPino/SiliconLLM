"""Trained source S6 fields in D256 coordinates, with exact input norm metric.

This compiles restricted source operators, not the composed source chatbot.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from falcon_auxiliary_s6_transport import (
    SITE,extent,write,tensor_reader,source_operator,compile_lift,lifted_operator,relative,scalar_witness)
START=time.monotonic()


def prepare(a):
    old=DOC/'falcon_auxiliary_s6_transport_binding_20261012.json'
    assert extent(old)['sha256']=='e52b539f37193d402b685e57354689f1d935935dcce3d026b4ea9f59a4f2be71'
    b=json.loads(old.read_bytes())
    provenance=DOC/'original_history_boundaries_finish_result_20261009.json'
    p=json.loads(provenance.read_bytes())['basis']
    assert extent(p['path'])==p and p['sha256']=='3dd96e0fd782833f51c005109e7ba1f95a98996bac1f5babb6a43ab3833e465d'
    extra=[Path(__file__),old,provenance,Path(p['path']),DOC/'FALCON_PROJECTED_S6_TRANSPORT_PROTOCOL_20261012.md']
    inputs={item['path']:item for item in b['inputs']}
    for path in extra:inputs[str(path.resolve())]=extent(path)
    b.update(schema='FALCON_PROJECTED_S6_TRANSPORT_BINDING_V1',worker=str(Path(__file__).resolve()),
        basis=p,source_D=2048,target_D=256,rho=8,
        gates=dict(exact_relative_RMS=1e-10,F32_coefficient_roundtrip_relative_RMS=1e-5,scalar_relative=1e-10),
        limits=dict(seconds=300,OS_bytes=4<<30,output_bytes=128<<20),
        inputs=list(inputs.values()),
        input_origin='First13 existing FIT embeddings -> z=embedding@P, reference raw residual=Pz. '
            'All tested sites use injected subspace residuals, not actual intermediate histories.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)


def run(a):
    assert extent(a.binding)['sha256']==a.binding_sha
    b=json.loads(a.binding.read_bytes());assert b['schema']=='FALCON_PROJECTED_S6_TRANSPORT_BINDING_V1'
    assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    assert not a.out.exists() and not a.directory.exists() and not a.out.with_suffix('.terminal.json').exists()
    import numpy as np
    import psutil
    assert (np.__version__,psutil.__version__)==('2.4.6','7.2.2')
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    proc=psutil.Process();proc.cpu_affinity([0]);stage='before_hash';stream=None;rows=[]
    receipt=dict(schema='FALCON_PROJECTED_S6_TRANSPORT_TERMINAL_V1',binding=extent(a.binding),
        freeze=a.freeze,command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),
        timing_control_admitted=False,original_engine_modified=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'],'deadline'
        assert proc.memory_info().peak_wset<=b['limits']['OS_bytes'],'OS cap'
        assert not proc.children(recursive=True),'unexpected child'
    def check():
        for item in b['inputs']:guard();assert extent(item['path'])==item,item['path']
    try:
        check();a.directory.mkdir()
        p=np.fromfile(b['basis']['path'],dtype='<f4').reshape(2048,256).astype('<f8')
        metric=p.T@p;defect=float(np.max(np.abs(metric-np.eye(256))))
        assert defect<=1e-4 and np.linalg.eigvalsh(metric).min()>0
        c=json.loads((Path(b['source'])/'config.json').read_bytes())
        assert (c['hidden_size'],c['mamba_d_ssm'],c['mamba_d_state'],c['mamba_n_heads'],
                c['mamba_d_head'],c['mamba_n_groups'])==(2048,3072,256,48,64,1)
        assert c['hidden_act']=='silu' and c['mamba_rms_norm'] and not c['mamba_norm_before_gate']
        assert not c['mamba_proj_bias'] and not c['projectors_bias']
        get,stream=tensor_reader(Path(b['source'])/'model.safetensors',np)
        embed=get('model.embed_tokens.weight',b['input_ids'])*c['embedding_multiplier']
        z=embed@p;raw_source=z@p.T;eps=c['rms_norm_eps'];rho=8
        target_norm=z/np.sqrt(np.sum((z@metric)*z,axis=1,keepdims=True)/256+rho*eps)
        identity_norm=z/np.sqrt(np.mean(z*z,axis=1,keepdims=True)+rho*eps)
        wrong_eps_norm=z/np.sqrt(np.sum((z@metric)*z,axis=1,keepdims=True)/256+eps)
        keys=dict(in_='in_proj.weight',conv='conv1d.weight',conv_b='conv1d.bias',
                  dt_b='dt_bias',A='A_log',D='D',norm='norm.weight',out='out_proj.weight')
        for site in b['sites']:
            guard();stage=f'site{site}'
            base=f'model.layers.{site}.';gamma=get(base+'input_layernorm.weight')
            inp=raw_source/np.sqrt(np.mean(raw_source**2,axis=1,keepdims=True)+eps)*gamma
            w={('in' if k=='in_' else k):get(base+'mamba.'+name) for k,name in keys.items()}
            w['conv']=w['conv'].reshape(3584,4)
            source=source_operator(inp,w,c,np);source['output']=source['output']@p
            lifted=compile_lift(w,c,np)
            # Fold norm gamma BEFORE the input map; output projection AFTER source output map.
            compact_input=math.sqrt(rho)*((lifted['in_proj']*gamma)@p)
            diagonal_input=math.sqrt(rho)*(lifted['in_proj']@p)
            lifted['in_proj']=compact_input;lifted['out_proj']=p.T@lifted['out_proj']
            actual=lifted_operator(target_norm,lifted,c,np)
            metric_identity=lifted_operator(identity_norm,lifted,c,np)['output']
            wrong_eps=lifted_operator(wrong_eps_norm,lifted,c,np)['output']
            diagonal_lift={**lifted,'in_proj':diagonal_input}
            projected_gamma=(p*p*gamma[:,None]).sum(0)
            diagonal=lifted_operator(target_norm*projected_gamma,diagonal_lift,c,np)['output']
            package={key:value.astype('<f4') for key,value in lifted.items()}
            package['input_metric']=metric.astype('<f4')
            package['input_epsilon']=np.array(rho*eps,dtype='<f4')
            roundtrip={key:value.astype('<f8') for key,value in package.items() if key in lifted}
            f32_norm=z/np.sqrt(np.sum((z@package['input_metric'].astype('<f8'))*z,axis=1,keepdims=True)/256
                              +float(package['input_epsilon']))
            rounded=lifted_operator(f32_norm,roundtrip,c,np)
            exact={key:dict(relative_RMS=relative(actual[key],source[key],np),
                           max_abs=float(np.max(np.abs(actual[key]-source[key]))))
                   for key in ('x','B','C','delta','gate','y','denominator','output')}
            rounding={key:relative(rounded[key],actual[key],np) for key in exact}
            witness=scalar_witness(source,w,np)
            row=dict(site=site,positions=13,exact_components=exact,F32_coefficient_roundtrip_components=rounding,
                scalar_recurrence_relative=witness,
                identity_metric_output_relative=relative(metric_identity,source['output'],np),
                wrong_input_epsilon_output_relative=relative(wrong_eps,source['output'],np),
                diagonal_gamma_output_relative=relative(diagonal,source['output'],np),
                dense_coefficients=sum(value.size for value in lifted.values()),
                F32_package_bytes=sum(value.nbytes for value in package.values()),
                auxiliary_output_columns_exact_zero=bool(np.all(package['out_proj'][:,3072:]==0)))
            row['exact_gates']=all(v['relative_RMS']<=b['gates']['exact_relative_RMS'] for v in exact.values())
            row['F32_coefficient_gates']=all(v<=b['gates']['F32_coefficient_roundtrip_relative_RMS'] for v in rounding.values())
            row['scalar_gate']=witness<=b['gates']['scalar_relative'];rows.append(row)
            np.savez(a.directory/f'site{site:02d}.coefficients.f32.npz',**package)
            np.savez(a.directory/f'site{site:02d}.frames.npz',z=z,target_norm=target_norm,
                **{'source_'+key:source[key] for key in exact},
                **{'compact_'+key:actual[key] for key in exact},
                **{'rounded_'+key:rounded[key] for key in exact},
                identity_metric_output=metric_identity,wrong_epsilon_output=wrong_eps,diagonal_gamma_output=diagonal)
            print(json.dumps(dict(site=site,exact=row['exact_gates'],F32=row['F32_coefficient_gates'],
                diagonal_gamma_error=row['diagonal_gamma_output_relative'],seconds=time.monotonic()-START)),flush=True)
            del w,source,lifted,roundtrip,package,actual,rounded,compact_input,diagonal_input,diagonal_lift
        stream.close();stream=None;stage='after_hash';check();guard()
        outputs=[extent(path) for path in sorted(a.directory.iterdir())]
        assert sum(item['bytes'] for item in outputs)<=b['limits']['output_bytes']
        # Preserve all scientific failures: a completed negative experiment still gets a result.
        passed=all(row['exact_gates'] and row['F32_coefficient_gates'] and row['scalar_gate'] for row in rows)
        result=dict(schema='FALCON_PROJECTED_S6_TRANSPORT_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,
            decision='RESTRICTED_D256_OPERATOR_AND_F32_COEFFICIENTS_QUALIFIED' if passed else 'COMPLETED_NUMERIC_FAILURE',
            basis=b['basis'],basis_orthogonality_max=defect,input_metric='actual P^T P',
            source_D=2048,target_D=256,DN=3680,N=256,DT=48,primary_channels=3072,auxiliary_channels=608,
            input_epsilon=rho*eps,postgate_epsilon=eps,input_case=b['input_case'],input_ids=b['input_ids'],
            input_origin=b['input_origin'],records=rows,outputs=outputs,inputs_before_after_exact=True,
            dense_linear_MAC_per_operator=3*3680*256+(48+512)*3680+3680*48,
            naive_states_per_operator=3680*256,source_primary_states_per_operator=3072*256,
            seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
            optimizer_GPU_model_native_calls=0,engine_extension_implemented=False,
            F32_runtime_arithmetic_evaluated=False,composed_chatbot_evaluated=False,
            useful_chatbot_admission=False,speed_admission=False,generality_admission=False,
            scope='F64 restricted operator with actual pretrained coefficients and P256. '
                'F32 coefficient export promoted to F64 to isolate coefficient rounding; '
                'not F32 arithmetic or packed/native parity, not full donor projection fidelity '
                'or sequential24-layer behavior. Source feature/state widths remain costly.')
        write(a.out,result);receipt.update(exit_code=0,error=None,inputs_before_after_exact=True,result=extent(a.out),outputs=outputs)
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),stage=stage,completed_sites=rows);raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze')
    p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.prepare:prepare(a)
    else:assert all((a.binding,a.binding_sha,a.freeze,a.directory));run(a)
