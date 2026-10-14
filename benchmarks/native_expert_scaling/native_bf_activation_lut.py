"""Explicit activation backend adapter over the qualified paired/replay harness."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
import native_wide_readout as harness
import native_wide_readout_audit as stored_auditor
from native_24_cost import load,check_pins,parameter_sets
from row_state import interface
from falcon_auxiliary_s6_transport import extent,write
FAMILY=Path('E:/SiliconLLM/results/native_expert_scaling/native_bf_activation_lut_20261014')
SCOPE='Exact finite-BF conv/gate/delta lookup; original accepted I16 readout. Legacy arm key wide denotes activation candidate. Stored native compatibility/supplied-row cost, not final CPU model quality/useful50/DRAM/useful-n/family transfer.'
_BASE_SETUP=harness.setup
_TABLE_OWNERS=None
_PARAMETERS_OWNER=None
_BINDING=None


def cfunction(dll,name,args,result=ctypes.c_int):
    fn=getattr(dll,name);fn.argtypes=args;fn.restype=result;return fn


def tables(b,np):
    return tuple(np.memmap(e['file']['path'],dtype=e['dtype'],mode='r',shape=(65536,))
                 for e in b['activation_tables'])


def bind(api,b,owned):
    fn=cfunction(api[0],'act_bind',[ctypes.c_void_p]*3+[ctypes.c_uint])
    assert fn(*(x.ctypes.data for x in owned),b['FP_control'])==0


def exhaustive(b,np):
    global _TABLE_OWNERS
    owned=tables(b,np);code=np.arange(65536,dtype='u2');finite=(code&0x7f80)!=0x7f80
    assert int(finite.sum())==65280
    ref=ctypes.CDLL(b['reference_compile']['dll']['path'])
    ctrl=cfunction(ref,'ref_control',[],ctypes.c_uint)()
    assert ctrl==b['FP_control']
    expected=(np.empty(65536,'u2'),np.empty(65536,'f8'),np.empty(65536,'u2'))
    reference=cfunction(ref,'act_reference',[ctypes.c_int]+[ctypes.c_void_p]*4)
    assert reference(65536,code.ctypes.data,*(x.ctypes.data for x in expected))==0
    for a,z in zip(owned,expected,strict=True):harness.equal(np.ascontiguousarray(a[finite]),np.ascontiguousarray(z[finite]))
    api=interface(dict(compile=b['candidate_compile']))
    probe=cfunction(api[0],'act_probe',[ctypes.c_int]+[ctypes.c_void_p]*4)
    binding=cfunction(api[0],'act_bind',[ctypes.c_void_p]*3+[ctypes.c_uint])
    setter=cfunction(api[0],'act_set_control_for_test',[ctypes.c_uint],None)
    actual=(np.empty(65536,'u2'),np.empty(65536,'f8'),np.empty(65536,'u2'))
    cfunction(api[0],'act_unbind_for_test',[],None)()
    assert probe(65536,code.ctypes.data,*(x.ctypes.data for x in actual))==5
    tag=np.zeros(1,'u4');row=np.zeros((1,6704),'u2')
    unbound=api[1]();assert unbound
    try:
        assert api[4](unbound,1,tag.ctypes.data,row.ctypes.data,*([0]*10))==5
        from native_24_cost import snap
        zero=snap(api[5],unbound,np)
        assert int(zero['seen'][0])==0 and not zero['state'].any() and not zero['ring'].any()
    finally:api[2](unbound)
    assert binding(*(x.ctypes.data for x in owned),ctrl^0x8000)==6
    bind(api,b,owned)
    _TABLE_OWNERS=owned
    ctx=api[1]();assert ctx
    try:
        setter(ctrl^0x8000)
        assert probe(65536,code.ctypes.data,*(x.ctypes.data for x in actual))==5
        assert api[4](ctx,1,tag.ctypes.data,row.ctypes.data,*([0]*10))==5
    finally:setter(ctrl)
    try:
        from native_24_cost import snap
        final=snap(api[5],ctx,np)
        assert int(final['seen'][0])==0 and not final['state'].any() and not final['ring'].any()
    finally:api[2](ctx)
    assert probe(65536,code.ctypes.data,*(x.ctypes.data for x in actual))==0
    for a,z in zip(actual,expected,strict=True):harness.equal(a,z)
    return dict(finite_codes=65280,nonfinite_fallback_codes=256,table_words=3*65280,
        all_numeric_bits_exact=True,FP_control=ctrl,wrong_bind_rejected=True,
        changed_FP_probe_and_row_rejected_without_state_mutation=True,
        numeric_contract='Output/state bits only; libm errno/sticky exceptions excluded. Scalar reference/manual RNE separate, libm/compiler shared.')


def activation_setup(b):
    global _TABLE_OWNERS
    import numpy as np
    apis=_BASE_SETUP(b);owned=tables(b,np);bind(apis[1][0],b,owned);_TABLE_OWNERS=owned
    return apis


def activation_parameters(b,np):
    global _PARAMETERS_OWNER
    result=parameter_sets(b,np);_PARAMETERS_OWNER=result
    assert _TABLE_OWNERS is not None
    support=ctypes.CDLL(b['support']['path'])
    fn=cfunction(support,'cost_activation_bind',[ctypes.c_void_p]*4,None)
    fn(result[-1][0]['scale'].ctypes.data,*(x.ctypes.data for x in _TABLE_OWNERS))
    return result


def configure(b):
    global _BINDING
    _BINDING=b;harness.FAMILY=FAMILY;harness.SCOPE=SCOPE
    harness.setup=activation_setup;harness.parameter_sets=activation_parameters
    stored_auditor.setup=activation_setup;stored_auditor.parameter_sets=activation_parameters


def generated_source():
    original=(B/'row_state.c').read_text()
    replacements=[
        ('#include <fenv.h>', '#include <fenv.h>\n#include "bf_activation_consumer.h"'),
        ('if(fegetround()!=FE_TONEAREST)return 3;TokenState*s=p;',
         'if(fegetround()!=FE_TONEAREST)return 3;if(!act_valid())return 5;TokenState*s=p;'),
        ('double v=unbf(bf((float)value));double z=fmax(0.,-v)+log1p(exp(-fabs(v)));uint16_t b=bf((float)(v*exp(-z)));',
         'uint16_t b=act_conv(bf((float)value));'),
        ('double d=unbf(bf((float)((double)unbf(row[6656+h])+conv[2][h])));double z=d>20?d:fmax(d,0.)+log1p(exp(-fabs(d)));uint16_t b=bf((float)z);',
         'uint16_t b=act_delta(bf((float)((double)unbf(row[6656+h])+conv[2][h])));'),
        ('((double)gate[c]/(1.0+exp(-(double)gate[c])))','act_gate(row[c])')]
    result=original
    for old,new in replacements:assert result.count(old)==1;result=result.replace(old,new)
    restored=result
    for old,new in reversed(replacements):assert restored.count(new)==1;restored=restored.replace(new,old)
    assert restored==original and '#include "integer_readout.c"' in result
    return result,replacements


def prepare(a):
    import numpy as np,psutil
    assert not a.out.exists() and not FAMILY.exists()
    parent=DOC/'native_wide_readout_binding_v2_20261014.json';b=load(parent)
    closed=DOC/'native_wide_readout_session_closure_v2_20261014.json'
    assert load(closed)['decision']=='EXACT_CANDIDATE_ACCELERATION_REJECTED'
    new=[Path(__file__),B/'bf_activation_emit.c',B/'bf_activation_reference.c',B/'bf_activation_consumer.h',
         B/'bf_activation_cost_support.c',DOC/'NATIVE_BF_ACTIVATION_LUT_PROTOCOL_20261014.md']
    for p in new:
        blob=subprocess.check_output(['git','show',a.freeze+':'+p.relative_to(ROOT).as_posix()],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==p.read_bytes().replace(b'\r\n',b'\n')
    check_pins(b,lambda:None);pins=b['inputs']+[extent(p) for p in new]+[extent(parent),extent(closed)]
    FAMILY.mkdir();sources,changes=generated_source();src=FAMILY/'row_state_bf_lut.c'
    src.write_text(sources,encoding='utf8',newline='\n')
    compiler=b['compile']['command'][0]
    def compile(source,name,scalar=False):
        dll=FAMILY/(name+'.dll');log=FAMILY/(name+'.compile.log')
        cmd=[compiler,'-O3','-mavx2','-mfma','-march=znver2','-ffp-contract=off','-shared','-I',str(B)]
        if scalar:cmd+=['-fno-vectorize','-fno-slp-vectorize']
        cmd += [str(source),'-o',str(dll),'-lm']
        result=subprocess.run(cmd,capture_output=True,timeout=45,creationflags=0x08000000)
        log.write_bytes(result.stdout+result.stderr);assert result.returncode==0
        return dict(command=cmd,source=extent(source),dll=extent(dll),log=extent(log))
    candidate=compile(src,'row_state_bf_lut');emitter=compile(B/'bf_activation_emit.c','activation_emitter')
    reference=compile(B/'bf_activation_reference.c','activation_reference',True)
    support=compile(B/'bf_activation_cost_support.c','activation_cost_support')
    emit=ctypes.CDLL(emitter['dll']['path']);ctrl=cfunction(emit,'act_control',[],ctypes.c_uint)()
    table_arrays=(np.empty(65536,'u2'),np.empty(65536,'f8'),np.empty(65536,'u2'))
    fn=cfunction(emit,'act_emit',[ctypes.c_void_p]*3)
    assert fn(*(x.ctypes.data for x in table_arrays))==0
    descriptors=[]
    for name,dtype,arr in zip(('conv','gate','delta'),('u2','f8','u2'),table_arrays,strict=True):
        path=FAMILY/(name+'.table.bin');arr.tofile(path);descriptors.append(dict(role=name,dtype=dtype,file=extent(path)))
    artifacts=[extent(src)]+[e[k] for e in (candidate,emitter,reference,support) for k in ('dll','log')]
    artifacts += [e['file'] for e in descriptors]
    b.update(schema='NATIVE_BF_ACTIVATION_LUT_BINDING_V1',freeze=a.freeze,
        inherited_closed_wide_experiment=extent(closed),parent_binding=extent(parent),
        candidate_compile=candidate,emitter_compile=emitter,reference_compile=reference,
        support=support['dll'],activation_support_compile=support,
        activation_tables=descriptors,FP_control=ctrl,source_replacements=changes,
        adapter=dict(candidate_slot='wide',meaning='finite-BF activation lookup, ORIGINAL readout',
            table_flush='Once per arm after final parameter pointer, both arms, expected128 groups'),
        output_directory=str(FAMILY/'run'),limits=dict(seconds=2700,OS_bytes=2<<30,output_bytes=32<<20),
        inputs=list({e['path']:e for e in pins+artifacts}.values()),artifacts=artifacts,scope=SCOPE)
    validation=exhaustive(b,np);vp=FAMILY/'table_validation.json';write(vp,validation)
    b['artifacts'].append(extent(vp));b['inputs'].append(extent(vp));b['table_validation']=extent(vp)
    check_pins(b,lambda:None);write(a.out,b)
    print(json.dumps(dict(binding=extent(a.out),exhaustive=validation)),flush=True)


def run(a):
    import numpy as np
    b=load(a.binding);assert extent(a.binding)['sha256']==a.binding_sha
    assert exhaustive(b,np)==load(b['table_validation']['path'])
    configure(b);harness.run(a)
    support=ctypes.CDLL(b['support']['path']);groups=cfunction(support,'cost_activation_groups',[],ctypes.c_uint64)()
    assert groups==128,groups
    summary=dict(binding=extent(a.binding),report=extent(a.out),exhaustive=load(b['table_validation']['path']),
        activation_table_payload_bytes=786432,table_flush_groups=groups,scope=SCOPE)
    write(a.out.with_suffix('.activation.json'),summary)


def audit(a):
    import numpy as np
    r=load(a.report);b=load(r['binding']['path']);validation=exhaustive(b,np)
    assert validation==load(b['table_validation']['path'])
    supplemental=a.report.with_suffix('.activation.json');data=load(supplemental)
    assert data['report']==extent(a.report) and data['binding']==r['binding']
    assert data['table_flush_groups']==128 and data['activation_table_payload_bytes']==786432 and data['exhaustive']==validation
    configure(b);stored_auditor.main(a)
    write(a.out.with_suffix('.activation.json'),dict(report=extent(a.report),audit=extent(a.out),
        exhaustive=validation,producer_activation_record=extent(supplemental),table_flush_accounting_pass=True))


def close(a):
    import psutil
    r=load(a.report);b=load(r['binding']['path']);ad=load(a.audit)
    assert ad['decision']=='ALL_STORED_EXACT_AND_ACCOUNTING_PASS' and ad['report']==extent(a.report)
    assert ad['bridge_counts']['output_words']==1187954688 and ad['checked_site_records']==192
    commands=[]
    for path,suffix in ((Path(r['binding']['path']),'.preparation_terminal.json'),(a.report,'.terminal.json'),(a.audit,'.terminal.json')):
        t=load(path.with_suffix(suffix));assert t['exit_code']==0 and t['result']==extent(path)
        if psutil.pid_exists(t['pid']):assert abs(psutil.Process(t['pid']).create_time()-t['creation_time'])>.001
        commands.append(dict(terminal=extent(path.with_suffix(suffix)),pid=t['pid'],creation_time=t['creation_time'],matching_process_absent=True))
    check_pins(b,lambda:None)
    extras=[r['adversarial'],r['BF_ties']]
    expected={e['path']:e['bytes'] for e in b['artifacts']+extras}
    for e in extras:assert extent(e['path'])==e
    assert {str(p.resolve()):p.stat().st_size for p in FAMILY.rglob('*') if p.is_file()}==expected
    producer_extra=a.report.with_suffix('.activation.json');audit_extra=a.audit.with_suffix('.activation.json')
    pe,ae=load(producer_extra),load(audit_extra)
    assert pe['report']==extent(a.report) and ae['audit']==extent(a.audit)
    assert ae['producer_activation_record']==extent(producer_extra) and ae['table_flush_accounting_pass']
    write(a.out,dict(schema='NATIVE_BF_ACTIVATION_LUT_CLOSURE_V1',report=extent(a.report),audit=extent(a.audit),
        producer_activation=extent(producer_extra),audit_activation=extent(audit_extra),binding=r['binding'],
        commands=commands,family_files=len(expected),family_bytes=sum(expected.values()),
        all_declared_hashes_and_numeric_qualification_pass=True,decision=r['decision'],
        activation_tables_payload_bytes=786432,goal_status='ACTIVE_INCOMPLETE',scope=SCOPE))
    print(json.dumps(dict(closure=extent(a.out))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','run','audit','close'])
    for name in ('out','binding','report','audit'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--freeze');parser.add_argument('--binding-sha');a=parser.parse_args()
    import psutil
    proc=psutil.Process();start=time.monotonic();receipt=dict(schema='NATIVE_BF_ACTIVATION_LUT_TERMINAL_V1',
        phase=a.phase,command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time())
    try:
        assert not a.out.exists();{'prepare':prepare,'run':run,'audit':audit,'close':close}[a.phase](a)
        assert time.monotonic()-start<=2700 and proc.memory_info().peak_wset<=2<<30
        receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.preparation_terminal.json' if a.phase=='prepare' else '.terminal.json'),receipt)
