"""Full immutable cost ledger and integer campaign custody; no retiming."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(ROOT/'benchmarks/native_expert_scaling'),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
import numpy as np,psutil

def main():
    rp=DOC/'integer_readout_cost_result_20261013.json';tp=rp.with_suffix('.terminal.json');r=json.loads(rp.read_bytes());t=json.loads(tp.read_bytes());assert t['exit_code']==0 and t['result']==extent(rp) and extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes())
    assert r['all_measured_outputs_bit_exact'] and r['CLFLUSH_supported'] and r['cache_line_bytes']==64 and not r['model_rate_admitted']
    for e in b['inputs']:assert extent(e['path'])==e
    parent=json.loads(Path(b['parent_report']['path']).read_bytes());ad=json.loads(Path(b['parent_adjudication']['path']).read_bytes());pb=json.loads(Path(parent['binding']['path']).read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==b['parent_report'] and ad['numeric_all_pass'] and ad['additional_all_pass'] and all(all(g.values()) for site in ad['donor_gates'].values() for g in site.values());assert b['integer_compile']==pb['compile']
    assert len(b['samples'])==len(r['sample_checks'])==18 and all(q['all_input_integer_raw_and_BF16_bits_exact'] for q in r['sample_checks']);assert [{k:q[k] for k in ['site','id','position']} for q in r['sample_checks']]==b['samples'];assert len(r['measurements'])==360
    expected={(s['site'],s['id'],s['position'],mode,rep,method) for s in b['samples'] for mode in ['warm_repeat','source_flushed'] for rep in range(5) for method in ['integer','dense']};seen=set()
    for q in r['measurements']:
        key=tuple(q[k] for k in ['site','id','position','mode','repetition','method']);assert key not in seen;seen.add(key);assert q['call_nanoseconds']>0 and q['flush_nanoseconds']>=0 and q['source_bytes']==(12599296 if q['method']=='integer' else 25165824);assert (q['flush_nanoseconds']==0)==(q['mode']=='warm_repeat')
    assert seen==expected;summary={};ratios={}
    for site in [0,12,23]:
        summary[str(site)]={};ratios[str(site)]={}
        for mode in ['warm_repeat','source_flushed']:
            summary[str(site)][mode]={}
            for method in ['integer','dense']:
                vals=[q['call_nanoseconds'] for q in r['measurements'] if q['site']==site and q['mode']==mode and q['method']==method];assert len(vals)==30;summary[str(site)][mode][method]=dict(median_us=float(np.median(vals)/1000),minimum_us=min(vals)/1000,maximum_us=max(vals)/1000)
            ratios[str(site)][mode]=summary[str(site)][mode]['integer']['median_us']/summary[str(site)][mode]['dense']['median_us']
    assert summary==r['summary'];commands=[]
    names=['integer_readout_binding_20261013.preparation_terminal.json','integer_readout_result_20261013.terminal.json','integer_readout_adjudication_20261013.terminal.json','integer_readout_adjudication_v2_20261013.terminal.json','integer_readout_adjudication_v3_20261013.terminal.json','integer_readout_cost_binding_20261013.preparation_terminal.json','integer_readout_cost_result_20261013.terminal.json']
    for index,name in enumerate(names):
        p=DOC/name;v=json.loads(p.read_bytes());assert v['exit_code']==(1 if index==3 else 0)
        if index!=3:assert extent(v['result']['path'])==v['result']
        else:assert 'TypeError' in v['traceback'] and not (DOC/'integer_readout_adjudication_v2_20261013.json').exists()
        live=False
        if psutil.pid_exists(v['pid']):live=abs(psutil.Process(v['pid']).create_time()-v['creation_time'])<1e-5
        assert not live;commands.append(dict(terminal=extent(p),pid=v['pid'],creation_time=v['creation_time'],exit_code=v['exit_code'],seconds=v['seconds'],OS_peak=v['OS_peak'],matching_process_absent=True))
    assert json.loads((DOC/'integer_readout_adjudication_20261013.json').read_bytes())['decision']=='INDEPENDENT_STORED_NUMERIC_FAILURE'
    files=[q['file'] for q in parent['records']]
    for e in files:assert extent(e['path'])==e
    total=sum(e['bytes'] for e in files);assert total==parent['output_bytes'];compiled=[pb['compile']['dll'],pb['compile']['log']]
    for e in compiled:assert extent(e['path'])==e
    family=total+sum(e['bytes'] for e in compiled);assert family<=pb['limits']['output_bytes'];protected=[]
    for rel in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']:
        p=(ROOT/rel).resolve();e=next(e for e in pb['inputs'] if Path(e['path']).resolve()==p);assert extent(p)==e;protected.append(e)
    foreign=[]
    for p in psutil.process_iter(['pid','create_time','name','cmdline']):
        try:
            if 'daemon_workflow.py' in ' '.join(p.info['cmdline'] or []):foreign.append(p.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    new_cost_bytes=sum(p.stat().st_size for p in Path(b['output_directory']).rglob('*') if p.is_file());assert new_cost_bytes<=b['limits']['output_bytes']
    result=dict(schema='INTEGER_READOUT_COST_REVIEW_CLOSURE_V1',source=extent(Path(__file__)),report=extent(rp),terminal=extent(tp),binding=r['binding'],parent_adjudication=b['parent_adjudication'],all360_observations_reconciled=True,same_full_audited_integer_DLL=True,sample_output_parity='Producer assertions verify every warmup/measured input/scale/int64/raw/BF16 against full audited archives;ledger reviewer does not rerun timed calls.',median_integer_over_dense=ratios,owned_commands=commands,new_array_files=len(files),new_run_bytes=total,new_family_bytes=family,new_cost_family_bytes=new_cost_bytes,protected_files_unchanged=protected,foreign_publisher_snapshot=foreign,before_after_hashes_exact=True,decision='COMPLETE_COST_COMPARISON',goal_status='ACTIVE_INCOMPLETE',scope='Full ledger/counts/summaries/source bytes and full campaign custody. V1 numeric diagnostic-denominator fault and V2 final hash-check variable shadow retained. No retiming, PMU DRAM bytes, ternary lookup execution, selective useful-n, own-history chatbot, model-rate or cross-family admission.')
    out=DOC/'integer_readout_cost_review_20261013.json';assert not out.exists();write(out,result);print(json.dumps(dict(review=extent(out),ratios=ratios,new_family_bytes=family,commands=commands)),flush=True)

if __name__=='__main__':main()
