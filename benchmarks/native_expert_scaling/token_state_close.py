"""Full native branch custody and immutable incremental-cost ledger review."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';sys.path[:0]=[str(ROOT/'benchmarks/native_expert_scaling'),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
import numpy as np,psutil

def main():
    rp=DOC/'token_state_cost_result_20261013.json';r=json.loads(rp.read_bytes());b=json.loads(Path(r['binding']['path']).read_bytes());assert extent(r['binding']['path'])==r['binding'] and r['all_measured_outputs_bit_exact'] and not r['model_rate_admitted'] and r['CLFLUSH_supported'] and r['cache_line_bytes']==64;ad=json.loads(Path(b['parent_adjudication']['path']).read_bytes());parent=json.loads(Path(b['parent_report']['path']).read_bytes());pb=json.loads(Path(b['parent_binding']['path']).read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==b['parent_report'] and ad['numeric_all_pass'] and ad['quality_all_pass'] and all(all(q.values()) for q in ad['donor_gates'].values());assert b['compile']==pb['compile']
    for e in b['inputs']:assert extent(e['path'])==e
    assert len(r['measurements'])==60 and len(b['samples'])==6;expected={(s['site'],s['id'],s['position'],mode,rep) for s in b['samples'] for mode in ['warm_ready','source_flushed'] for rep in range(5)};seen=set()
    for row in r['measurements']:
        key=tuple(row[k] for k in ['site','id','position','mode','repetition']);assert key not in seen;seen.add(key);rec=next(q for q in pb['records'] if q['id']==row['id']);assert row['token_ID']==rec['input_ids'][row['position']] and row['own_prefix_seen_before']==row['position'] and row['source_bytes']==12965728 and row['all_output_bits_exact'] and row['call_nanoseconds']>0 and (row['flush_nanoseconds']==0)==(row['mode']=='warm_ready')
    assert seen==expected;summary={}
    for mode in ['warm_ready','source_flushed']:
        v=[q['call_nanoseconds'] for q in r['measurements'] if q['mode']==mode];assert len(v)==30;summary[mode]=dict(median_us=float(np.median(v)/1000),minimum_us=min(v)/1000,maximum_us=max(v)/1000)
    assert summary==r['summary'];commands=[]
    for name in ['token_state_binding_20261013.preparation_terminal.json','token_state_result_20261013.terminal.json','token_state_adjudication_20261013.terminal.json','token_state_cost_binding_20261013.preparation_terminal.json','token_state_cost_result_20261013.terminal.json']:
        p=DOC/name;t=json.loads(p.read_bytes());assert t['exit_code']==0 and extent(t['result']['path'])==t['result'];live=False
        if psutil.pid_exists(t['pid']):live=abs(psutil.Process(t['pid']).create_time()-t['creation_time'])<1e-5
        assert not live;commands.append(dict(terminal=extent(p),pid=t['pid'],creation_time=t['creation_time'],exit_code=0,seconds=t['seconds'],OS_peak=t['OS_peak'],matching_process_absent=True))
    files=[q['file'] for q in parent['records']];compiled=[pb['compile']['dll'],pb['compile']['log']]
    for e in files+compiled:assert extent(e['path'])==e
    run=sum(e['bytes'] for e in files);assert run==parent['output_bytes'];family=run+sum(e['bytes'] for e in compiled);assert family<=pb['limits']['output_bytes'];protected=[]
    for rel in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']:
        p=(ROOT/rel).resolve();e=next(e for e in pb['inputs'] if Path(e['path']).resolve()==p);assert extent(p)==e;protected.append(e)
    foreign=[]
    for p in psutil.process_iter(['pid','create_time','name','cmdline']):
        try:
            if 'daemon_workflow.py' in ' '.join(p.info['cmdline'] or []):foreign.append(p.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    costbytes=sum(p.stat().st_size for p in Path(b['output_directory']).rglob('*') if p.is_file());assert costbytes<=b['limits']['output_bytes'];result=dict(schema='TOKEN_STATE_CLOSURE_COST_REVIEW_V1',source=extent(Path(__file__)),native_report=b['parent_report'],native_adjudication=b['parent_adjudication'],cost_report=extent(rp),cost_binding=r['binding'],same_audited_DLL=True,all60_records_reconciled=True,summary=summary,sample_output_parity='Producer verifies every warmup/measured archived output word/actual ID/count/ring;reviewer reconciles complete ledger without retiming.',owned_commands=commands,new_run_bytes=run,new_family_bytes=family,new_cost_family_bytes=costbytes,protected_files_unchanged=protected,foreign_publisher_snapshot=foreign,goal_status='ACTIVE_INCOMPLETE',scope='Initial native SSM branch complete/source calibration/temporal/output/custody and state-ready cost reviewed. Prefix setup/initialization/state warming outside clock,architectural source eviction not PMU DRAM bytes/cold storage I/O;no all-layer/chatbot/semantic-n/useful50 admission.');out=DOC/'token_state_session_closure_20261013.json';assert not out.exists();write(out,result);print(json.dumps(dict(closure=extent(out),new_family_bytes=family,summary=summary,commands=commands)),flush=True)

if __name__=='__main__':main()
