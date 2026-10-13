"""All24 native row-export and qualification family custody closure."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';sys.path[:0]=[str(ROOT/'benchmarks/native_expert_scaling'),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
import psutil

def main():
    rp=DOC/'row_state_result_20261013.json';ap=DOC/'row_state_adjudication_20261013.json';r=json.loads(rp.read_bytes());ad=json.loads(ap.read_bytes());b=json.loads(Path(r['binding']['path']).read_bytes());assert extent(r['binding']['path'])==r['binding'] and ad['report']==extent(rp) and ad['decision']=='INDEPENDENT_STORED_PASS' and ad['all_numeric_pass'];files=[q['file'] for q in r['records']+r['operator_controls']];params=[q['file'] for q in b['exports']];compiled=[b['compile']['dll'],b['compile']['log']]
    for e in files+params+compiled:assert extent(e['path'])==e
    run=sum(e['bytes'] for e in files);assert run==r['output_bytes'];family=run+sum(e['bytes'] for e in params+compiled);assert family<=b['limits']['output_bytes'];commands=[]
    for name in ['row_state_binding_20261013.preparation_terminal.json','row_state_result_20261013.terminal.json','row_state_adjudication_20261013.terminal.json']:
        p=DOC/name;t=json.loads(p.read_bytes());assert t['exit_code']==0 and extent(t['result']['path'])==t['result'];live=False
        if psutil.pid_exists(t['pid']):live=abs(psutil.Process(t['pid']).create_time()-t['creation_time'])<1e-5
        assert not live;commands.append(dict(terminal=extent(p),pid=t['pid'],creation_time=t['creation_time'],exit_code=0,seconds=t['seconds'],OS_peak=t['OS_peak'],matching_process_absent=True))
    protected=[]
    for rel in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']:
        p=(ROOT/rel).resolve();e=next(e for e in b['inputs'] if Path(e['path']).resolve()==p);assert extent(p)==e;protected.append(e)
    canonical=next(e for e in b['inputs'] if Path(e['path']).resolve()==Path(b['source_weights']).resolve());assert extent(canonical['path'])==canonical;foreign=[]
    for p in psutil.process_iter(['pid','create_time','name','cmdline']):
        try:
            if 'daemon_workflow.py' in ' '.join(p.info['cmdline'] or []):foreign.append(p.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    result=dict(schema='ROW_STATE_CLOSURE_V1',source=extent(Path(__file__)),report=extent(rp),adjudication=extent(ap),binding=r['binding'],owned_commands=commands,new_history_control_files=len(files),new_parameter_packets=len(params),new_run_bytes=run,new_family_bytes=family,protected_files_unchanged=protected,canonical_source_weights_unchanged=canonical,foreign_publisher_snapshot=foreign,goal_status='ACTIVE_INCOMPLETE',scope='Explicit contextual-row ABI/actual all24 source coefficient exports and arbitrary-input numeric qualification complete;real48 initial-layer archive identity qualified. No actual later21 trajectories/whole-chatbot/GPU/speed/useful-n claim. All3 commands closed0,no owned matching process live.');out=DOC/'row_state_session_closure_20261013.json';assert not out.exists();write(out,result);print(json.dumps(dict(closure=extent(out),new_family_bytes=family,commands=commands)),flush=True)

if __name__=='__main__':main()
