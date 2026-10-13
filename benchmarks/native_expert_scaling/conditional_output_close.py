"""Close both fully audited conditional-bank and target-aware coverage jobs."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
import psutil


def main():
    reports=[]
    protected=[]
    for tag in ['function','coverage']:
        rp=DOC/f'conditional_output_{tag}_result_20261013.json'
        ap=DOC/f'conditional_output_{tag}_adjudication_20261013.json'
        r=json.loads(rp.read_bytes())
        ad=json.loads(ap.read_bytes())
        assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(rp)
        assert ad['binding']==r['binding'] and extent(r['binding']['path'])==r['binding']
        b=json.loads(Path(r['binding']['path']).read_bytes())
        files=[q['file'] for q in r['records']]
        if tag=='function':
            for m in r['models']:
                files+=[m['global_file']]+[q['file'] for q in m['banks']]
        for e in files:
            assert extent(e['path'])==e
        actual=sum(e['bytes'] for e in files)
        assert actual==r['output_bytes']<=b['limits']['output_bytes']
        reports.append(dict(kind=tag,report=extent(rp),adjudication=extent(ap),binding=r['binding'],
                            array_files=len(files),new_E_bytes=actual))
        for rel in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json',
                    'benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']:
            p=(ROOT/rel).resolve()
            expected=next(e for e in b['inputs'] if Path(e['path']).resolve()==p)
            assert extent(p)==expected
            if tag=='coverage':
                protected.append(expected)
    commands=[]
    for tag in ['function','coverage']:
        for role,name in [('preparation',f'conditional_output_{tag}_binding_20261013.preparation_terminal.json'),
                          ('producer',f'conditional_output_{tag}_result_20261013.terminal.json'),
                          ('audit',f'conditional_output_{tag}_adjudication_20261013.terminal.json')]:
            p=DOC/name
            t=json.loads(p.read_bytes())
            assert t['exit_code']==0 and extent(t['result']['path'])==t['result']
            live=False
            if psutil.pid_exists(t['pid']):
                live=abs(psutil.Process(t['pid']).create_time()-t['creation_time'])<1e-5
            assert not live,(tag,role,t['pid'])
            commands.append(dict(kind=tag,role=role,terminal=extent(p),pid=t['pid'],
                                 creation_time=t['creation_time'],exit_code=0,seconds=t['seconds'],
                                 OS_peak=t['OS_peak'],matching_process_absent=True))
    foreign=[]
    for p in psutil.process_iter(['pid','create_time','name','cmdline']):
        try:
            if 'daemon_workflow.py' in ' '.join(p.info['cmdline'] or []):
                foreign.append(p.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            pass
    out=DOC/'conditional_output_session_closure_20261013.json'
    result=dict(schema='CONDITIONAL_OUTPUT_SESSION_CLOSURE_V1',source=extent(Path(__file__)),
                reports=reports,owned_commands=commands,all_owned_matching_processes_absent=True,
                protected_files_unchanged=protected,foreign_publisher_snapshot=foreign,
                source_forwards=0,native_calls=0,optimizer_updates=0,source_generations=0,
                T4_allocations=0,GPU_jobs=0,new_ternary_expert_banks=0,
                goal_status='ACTIVE_INCOMPLETE',
                scope='Both complete independent audits PASS; all affine-bank quality gates remain failed. '
                      'Coverage oracle uses target and all experts and is not an admissible router. '
                      'No OWN-history chatbot, useful-n, native DRAM/timing or same-artifact50 admission. '
                      'All six scientific commands terminal0/no owned job live/no new mechanical fault.')
    write(out,result)
    print(json.dumps(dict(closure=extent(out),reports=reports,commands=commands)),flush=True)


if __name__=='__main__':
    main()
