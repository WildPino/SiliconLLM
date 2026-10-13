"""Native p6 LUT correctness custody closure;cost remains a separate admission."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(ROOT/'benchmarks/native_expert_scaling'),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
import psutil

def main():
    rp=DOC/'ternary_readout_native_result_20261013.json';ap=DOC/'ternary_readout_native_adjudication_20261013.json';r=json.loads(rp.read_bytes());ad=json.loads(ap.read_bytes());b=json.loads(Path(r['binding']['path']).read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(rp) and ad['binding']==r['binding'] and extent(r['binding']['path'])==r['binding'];files=[q['file'] for q in r['records']]
    for e in files:assert extent(e['path'])==e
    total=sum(e['bytes'] for e in files);assert total==r['output_bytes'];compiled=[e for c in b['compile'] for e in [c['dll'],c['log']]]
    for e in compiled:assert extent(e['path'])==e
    family=total+sum(e['bytes'] for e in compiled);assert family<=b['limits']['output_bytes'];commands=[]
    for role,name in [('preparation','ternary_readout_native_binding_20261013.preparation_terminal.json'),('producer','ternary_readout_native_result_20261013.terminal.json'),('audit','ternary_readout_native_adjudication_20261013.terminal.json')]:
        p=DOC/name;t=json.loads(p.read_bytes());assert t['exit_code']==0 and extent(t['result']['path'])==t['result'];live=False
        if psutil.pid_exists(t['pid']):live=abs(psutil.Process(t['pid']).create_time()-t['creation_time'])<1e-5
        assert not live;(commands.append(dict(role=role,terminal=extent(p),pid=t['pid'],creation_time=t['creation_time'],exit_code=0,seconds=t['seconds'],OS_peak=t['OS_peak'],matching_process_absent=True)))
    assert ad['terminal']==commands[1]['terminal'];protected=[]
    for rel in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']:
        p=(ROOT/rel).resolve();expected=next(e for e in b['inputs'] if Path(e['path']).resolve()==p);assert extent(p)==expected;protected.append(expected)
    foreign=[]
    for p in psutil.process_iter(['pid','create_time','name','cmdline']):
        try:
            if 'daemon_workflow.py' in ' '.join(p.info['cmdline'] or []):foreign.append(p.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    result=dict(schema='TERNARY_READOUT_NATIVE_CLOSURE_V1',source=extent(Path(__file__)),report=extent(rp),adjudication=extent(ap),binding=r['binding'],owned_commands=commands,compiled_artifacts=compiled,all_owned_matching_processes_absent=True,new_array_files=len(files),new_run_bytes=total,new_family_bytes=family,protected_files_unchanged=protected,foreign_publisher_snapshot=foreign,goal_status='ACTIVE_INCOMPLETE',scope='Compiled selected-p6 LUT correctness/full original score bits qualified;producer scalar/table/mutation receipts distinct from independent all-history vector audit. No source/recurrence/GPU/T4/activation quantization/native timings/DRAM/OWN-history chatbot/useful-n/same useful50 admission. Three scientific commands terminal0/no owned job live.');out=DOC/'ternary_readout_native_session_closure_20261013.json';write(out,result);print(json.dumps(dict(closure=extent(out),commands=commands,new_family_bytes=family)),flush=True)

if __name__=='__main__':main()
