"""Full immutable LUT cost ledger and complete campaign custody, no retiming."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(ROOT/'benchmarks/native_expert_scaling'),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
import numpy as np,psutil

def main():
    rp=DOC/'exact_integer_lut_cost_result_20261013.json';tp=rp.with_suffix('.terminal.json');r=json.loads(rp.read_bytes());t=json.loads(tp.read_bytes());assert t['exit_code']==0 and t['result']==extent(rp) and extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes())
    assert r['all_measured_outputs_bit_exact'] and r['CLFLUSH_supported'] and r['cache_line_bytes']==64 and not r['model_rate_admitted'] and r['source_forwards']==r['recurrence_calls']==0
    for e in b['inputs']:assert extent(e['path'])==e
    parent=json.loads(Path(b['parent_report']['path']).read_bytes());ad=json.loads(Path(b['parent_adjudication']['path']).read_bytes());pb=json.loads(Path(parent['binding']['path']).read_bytes());ib=json.loads(Path(pb['parent_binding']['path']).read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==b['parent_report'] and ad['numeric_all_pass'] and ad['inherited_additional_all_pass'] and all(all(g.values()) for site in ad['inherited_donor_gates'].values() for g in site.values());assert b['lut_compile']==pb['compile'] and b['integer_compile']==ib['compile']
    assert len(b['samples'])==len(r['sample_checks'])==18 and all(q['all_input_integer_raw_and_BF16_bits_exact'] for q in r['sample_checks']);assert [{k:q[k] for k in ['site','id','position']} for q in r['sample_checks']]==b['samples'];assert len(r['measurements'])==360
    expected={(s['site'],s['id'],s['position'],mode,rep,method) for s in b['samples'] for mode in ['warm_repeat','source_flushed'] for rep in range(5) for method in ['lut','integer']};seen=set()
    for q in r['measurements']:
        key=tuple(q[k] for k in ['site','id','position','mode','repetition','method']);assert key not in seen;seen.add(key);assert q['call_nanoseconds']>0 and q['flush_nanoseconds']>=0 and q['source_bytes']==(9453568 if q['method']=='lut' else 12599296);assert (q['flush_nanoseconds']==0)==(q['mode']=='warm_repeat')
    assert seen==expected;summary={};ratios={}
    for site in [0,12,23]:
        summary[str(site)]={};ratios[str(site)]={}
        for mode in ['warm_repeat','source_flushed']:
            summary[str(site)][mode]={}
            for method in ['lut','integer']:
                vals=[q['call_nanoseconds'] for q in r['measurements'] if q['site']==site and q['mode']==mode and q['method']==method];assert len(vals)==30;summary[str(site)][mode][method]=dict(median_us=float(np.median(vals)/1000),minimum_us=min(vals)/1000,maximum_us=max(vals)/1000)
            ratios[str(site)][mode]=summary[str(site)][mode]['lut']['median_us']/summary[str(site)][mode]['integer']['median_us']
    assert summary==r['summary'];commands=[]
    names=['exact_integer_lut_binding_20261013.preparation_terminal.json','exact_integer_lut_result_20261013.terminal.json','exact_integer_lut_adjudication_20261013.terminal.json','exact_integer_lut_cost_binding_20261013.preparation_terminal.json','exact_integer_lut_cost_result_20261013.terminal.json']
    for name in names:
        p=DOC/name;v=json.loads(p.read_bytes());assert v['exit_code']==0 and extent(v['result']['path'])==v['result'];live=False
        if psutil.pid_exists(v['pid']):live=abs(psutil.Process(v['pid']).create_time()-v['creation_time'])<1e-5
        assert not live;commands.append(dict(terminal=extent(p),pid=v['pid'],creation_time=v['creation_time'],exit_code=0,seconds=v['seconds'],OS_peak=v['OS_peak'],matching_process_absent=True))
    files=[q['file'] for q in parent['records']]+[parent['controls'][0]['exhaustive_digits']]+parent['controls'][0]['exhaustive_tables']
    for e in files:assert extent(e['path'])==e
    total=sum(e['bytes'] for e in files);assert total==parent['output_bytes'];other=[p['file'] for p in pb['packets']]+[pb['compile']['dll'],pb['compile']['log']]
    for e in other:assert extent(e['path'])==e
    family=total+sum(e['bytes'] for e in other);assert family<=pb['limits']['output_bytes'];protected=[]
    for rel in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/nibble_pack.h','benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']:
        p=(ROOT/rel).resolve();e=next(e for e in pb['inputs'] if Path(e['path']).resolve()==p);assert extent(p)==e;protected.append(e)
    source=Path(pb['compile']['source']['path']).read_text();engine=(ROOT/'benchmarks/phase60/engine.c').read_text();literal=engine[engine.index('static inline void acc_add_i8x32('):engine.index('// P1 (BRIEF_P1_NIBBLE_PACKING')].strip();assert literal in source and '#include "../donor_adaptation/nibble_pack.h"' in source
    foreign=[]
    for p in psutil.process_iter(['pid','create_time','name','cmdline']):
        try:
            if 'daemon_workflow.py' in ' '.join(p.info['cmdline'] or []):foreign.append(p.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    new_cost_bytes=sum(p.stat().st_size for p in Path(b['output_directory']).rglob('*') if p.is_file());assert new_cost_bytes<=b['limits']['output_bytes'];values=[v for site in ratios.values() for v in site.values()];decision='REJECT_CURRENT_EXACT_BYTE_LUT_AS_ACCELERATION' if all(v>1 for v in values) else 'SAMPLED_MEDIAN_ADVANTAGE' if all(v<1 for v in values) else 'MIXED_SAMPLED_COST_BOUNDARY'
    result=dict(schema='EXACT_INTEGER_LUT_COST_REVIEW_CLOSURE_V1',source=extent(Path(__file__)),report=extent(rp),terminal=extent(tp),binding=r['binding'],parent_adjudication=b['parent_adjudication'],all360_observations_reconciled=True,same_full_audited_LUT_and_integer_DLLs=True,original_packed_header_and_literal_accumulator_verified=True,sample_output_parity='Producer verifies every warmup/measured input/scale/I64/raw/BF16 against complete audited archives;ledger reviewer does not rerun timed calls.',median_LUT_over_integer=ratios,logical_code_vector_load_bytes_per_input=37748736,owned_commands=commands,new_array_files=len(files),new_run_bytes=total,new_family_bytes=family,new_cost_family_bytes=new_cost_bytes,protected_files_unchanged=protected,foreign_publisher_snapshot=foreign,before_after_hashes_exact=True,decision=decision,goal_status='ACTIVE_INCOMPLETE',scope='Complete immutable observations/counts/summaries/source bytes/full custody. Exact representation qualified without new approximation. Logical code loads are not PMU DRAM bytes. No retiming/tail guarantee/selective useful-n/own-history chatbot/model-rate/cross-family admission.')
    out=DOC/'exact_integer_lut_cost_review_20261013.json';assert not out.exists();write(out,result);print(json.dumps(dict(review=extent(out),ratios=ratios,decision=decision,new_family_bytes=family,commands=commands)),flush=True)

if __name__=='__main__':main()
