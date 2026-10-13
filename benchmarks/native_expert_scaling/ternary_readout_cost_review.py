"""Independent full observation ledger/source transform/custody review,not retiming."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(ROOT/'benchmarks/native_expert_scaling'),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
import numpy as np,psutil

def main():
    rp=DOC/'ternary_readout_cost_result_20261013.json';tp=DOC/'ternary_readout_cost_result_20261013.terminal.json';r=json.loads(rp.read_bytes());t=json.loads(tp.read_bytes());assert t['exit_code']==0 and t['result']==extent(rp) and extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes());assert r['all_measured_outputs_bit_exact'] and r['CLFLUSH_supported'] and r['cache_line_bytes']==64
    for e in b['inputs']:assert extent(e['path'])==e
    original=Path(b['source_transform']['original']['path']).read_text();statement=b['source_transform']['deleted_statement'];assert original.count(statement)==1;prefix=original.replace('API int tr_history(','API int tr_lut_only(').replace(statement,'');assert Path(b['compile']['source']['path']).read_text().startswith(prefix+'\n#include <cpuid.h>')
    assert len(b['samples'])==len(r['sample_checks'])==18 and all(q['all_raw_and_BF16_bits_exact'] for q in r['sample_checks']);assert [{k:q[k] for k in ['site','id','position']} for q in r['sample_checks']]==b['samples'];assert len(r['measurements'])==360
    expected={(s['site'],s['id'],s['position'],mode,rep,method) for s in b['samples'] for mode in b['modes'] for rep in range(5) for method in ['lut','dense']};seen=set()
    for q in r['measurements']:
        key=tuple(q[k] for k in ['site','id','position','mode','repetition','method']);assert key not in seen;seen.add(key);assert q['call_nanoseconds']>0 and q['flush_nanoseconds']>=0;assert q['source_bytes']==(7573504 if q['method']=='lut' else 25165824);assert (q['flush_nanoseconds']==0)==(q['mode']=='warm_repeat')
    assert seen==expected;summary={};ratios={}
    for site in [0,12,23]:
        summary[str(site)]={};ratios[str(site)]={}
        for mode in b['modes']:
            summary[str(site)][mode]={}
            for method in ['lut','dense']:
                vals=[q['call_nanoseconds'] for q in r['measurements'] if q['site']==site and q['mode']==mode and q['method']==method];assert len(vals)==30;summary[str(site)][mode][method]=dict(median_us=float(np.median(vals)/1000),minimum_us=min(vals)/1000,maximum_us=max(vals)/1000)
            ratios[str(site)][mode]=summary[str(site)][mode]['lut']['median_us']/summary[str(site)][mode]['dense']['median_us']
    assert summary==r['summary'];commands=[]
    for name in ['ternary_readout_cost_binding_20261013.preparation_terminal.json','ternary_readout_cost_result_20261013.terminal.json']:
        p=DOC/name;v=json.loads(p.read_bytes());assert v['exit_code']==0 and extent(v['result']['path'])==v['result'];live=False
        if psutil.pid_exists(v['pid']):live=abs(psutil.Process(v['pid']).create_time()-v['creation_time'])<1e-5
        assert not live;commands.append(dict(terminal=extent(p),pid=v['pid'],creation_time=v['creation_time'],exit_code=0,seconds=v['seconds'],OS_peak=v['OS_peak'],matching_process_absent=True))
    bytes_new=sum(e['bytes'] for e in b['inputs'] if Path(e['path']).parent.resolve()==Path(b['output_directory']).resolve());assert bytes_new<=b['limits']['output_bytes'];result=dict(schema='TERNARY_READOUT_COST_REVIEW_V1',source=extent(Path(__file__)),report=extent(rp),terminal=extent(tp),binding=r['binding'],all360_observations_reconciled=True,source_transform_verified=True,sample_output_parity='Producer measured-call assertions with archived raw/BF16 reference;not independent full wrapper-history reproduction.',median_LUT_over_dense=ratios,owned_commands=commands,compiled_family_bytes=bytes_new,before_after_hashes_exact=True,decision='REJECT_CURRENT_F64_LUT_AS_ACCELERATION',goal_status='ACTIVE_INCOMPLETE',scope='Full immutable timing ledger/source/sample/count/summary/custody review. No retiming/PMU DRAM measurement/new full-history wrapper audit/whole chatbot/useful-n/model rate. Original fully audited native data retained.');out=DOC/'ternary_readout_cost_review_20261013.json';write(out,result);print(json.dumps(dict(review=extent(out),ratios=ratios,commands=commands)),flush=True)

if __name__=='__main__':main()
