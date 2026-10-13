"""Complete vocabulary campaign custody, including completed numeric/quality failures."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(ROOT/'benchmarks/native_expert_scaling'),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
import psutil

def main():
    rp=DOC/'token_generator_compiler_result_20261013.json';ap=DOC/'token_generator_compiler_adjudication_20261013.json';r=json.loads(rp.read_bytes());ad=json.loads(ap.read_bytes());b=json.loads(Path(r['binding']['path']).read_bytes());assert extent(r['binding']['path'])==r['binding'] and ad['binding']==r['binding'] and ad['report']==extent(rp) and ad['decision'] in ['INDEPENDENT_STORED_PASS','INDEPENDENT_STORED_NUMERIC_FAILURE']
    files=list(r['tables'].values())+[r['parameters'],r['scalar_control']['file']]+[q['file'] for q in r['records']];assert len(files)==52
    for e in files:assert extent(e['path'])==e
    total=sum(e['bytes'] for e in files);assert total==r['output_bytes'] and total<=b['limits']['output_bytes'];actual={str(p.resolve()) for p in Path(b['output_directory']).iterdir()};assert actual=={str(Path(e['path']).resolve()) for e in files};commands=[]
    for name in ['token_generator_compiler_binding_20261013.preparation_terminal.json','token_generator_compiler_result_20261013.terminal.json','token_generator_compiler_adjudication_20261013.terminal.json']:
        p=DOC/name;t=json.loads(p.read_bytes());assert t['exit_code']==0 and extent(t['result']['path'])==t['result'];live=False
        if psutil.pid_exists(t['pid']):live=abs(psutil.Process(t['pid']).create_time()-t['creation_time'])<1e-5
        assert not live;commands.append(dict(terminal=extent(p),pid=t['pid'],creation_time=t['creation_time'],exit_code=0,seconds=t['seconds'],OS_peak=t['OS_peak'],matching_process_absent=True))
    assert ad['terminal']==commands[1]['terminal'];protected=[]
    for rel in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']:
        p=(ROOT/rel).resolve();e=next(e for e in b['inputs'] if Path(e['path']).resolve()==p);assert extent(p)==e;protected.append(e)
    source=next(e for e in b['inputs'] if Path(e['path']).resolve()==Path(b['source_weights']).resolve());assert extent(source['path'])==source
    foreign=[]
    for p in psutil.process_iter(['pid','create_time','name','cmdline']):
        try:
            if 'daemon_workflow.py' in ' '.join(p.info['cmdline'] or []):foreign.append(p.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    eligible=ad['numeric_all_pass'] and ad['H0_all_pass'] and ad['source_fields_all_pass'] and ad['incremental_fields_all_pass'];result=dict(schema='TOKEN_GENERATOR_COMPILER_CLOSURE_V1',source=extent(Path(__file__)),report=extent(rp),adjudication=extent(ap),binding=r['binding'],owned_commands=commands,new_files=len(files),new_family_bytes=total,all_matching_owned_processes_absent=True,protected_files_unchanged=protected,canonical_source_weights_unchanged=source,foreign_publisher_snapshot=foreign,eligible_for_native_composition=eligible,goal_status='ACTIVE_INCOMPLETE',scope='All complete vocabulary/control/calibration extents/resources/terminal PID+creation/canonical source/protected file custody. Full completed numeric or quality failures preserved,not waived. No native-state composition,cost,GPU/T4/RESERVED/model rate/useful-n admission.');out=DOC/'token_generator_compiler_session_closure_20261013.json';assert not out.exists();write(out,result);print(json.dumps(dict(closure=extent(out),eligible_for_native_composition=eligible,new_family_bytes=total,commands=commands)),flush=True)

if __name__=='__main__':main()
