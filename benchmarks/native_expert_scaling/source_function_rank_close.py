"""Custody closure of bounded rank probe including two retained mechanical faults."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(ROOT/'benchmarks/native_expert_scaling'),
              str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
import psutil

def main():
    rp=DOC/'source_function_rank_result_20261013.json'
    ap=DOC/'source_function_rank_adjudication_v2_20261013.json'
    r=json.loads(rp.read_bytes())
    ad=json.loads(ap.read_bytes())
    b=json.loads(Path(r['binding']['path']).read_bytes())
    assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(rp)
    assert ad['binding']==r['binding'] and extent(r['binding']['path'])==r['binding']
    specs=[('initial preparation','source_function_rank_binding_20261013.preparation_terminal.json',1),
           ('preparation','source_function_rank_binding_v2_20261013.preparation_terminal.json',0),
           ('producer','source_function_rank_result_20261013.terminal.json',0),
           ('audit serialization fault','source_function_rank_adjudication_20261013.terminal.json',1),
           ('complete V2 audit','source_function_rank_adjudication_v2_20261013.terminal.json',0)]
    commands=[]
    for role,name,code in specs:
        p=DOC/name
        t=json.loads(p.read_bytes())
        assert t['exit_code']==code
        live=False
        if psutil.pid_exists(t['pid']):
            live=abs(psutil.Process(t['pid']).create_time()-t['creation_time'])<1e-5
        assert not live,(role,t['pid'])
        if code==0:
            assert extent(t['result']['path'])==t['result']
        commands.append(dict(role=role,terminal=extent(p),pid=t['pid'],creation_time=t['creation_time'],
                             exit_code=code,seconds=t['seconds'],OS_peak=t['OS_peak'],
                             matching_process_absent=True))
    assert ad['terminal']==commands[2]['terminal']
    bytes_total=0
    for q in r['records']:
        assert extent(q['file']['path'])==q['file']
        bytes_total+=q['file']['bytes']
    assert bytes_total==r['output_bytes']<=b['limits']['output_bytes']
    protected=[]
    for relative in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json',
                     'benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']:
        p=(ROOT/relative).resolve()
        expected=next(e for e in b['inputs'] if Path(e['path']).resolve()==p)
        actual=extent(p)
        assert actual==expected
        protected.append(actual)
    foreign=[]
    for p in psutil.process_iter(['pid','create_time','name','cmdline']):
        try:
            if 'daemon_workflow.py' in ' '.join(p.info['cmdline'] or []):
                foreign.append(p.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            pass
    launch=json.loads((DOC/'source_function_rank_audit_launch_v2_20261013.json').read_text(encoding='utf-8-sig'))
    partial=extent(DOC/'source_function_rank_adjudication_20261013.json')
    assert partial['sha256']==launch['prior_partial_adjudication_sha256']
    closure=dict(schema='SOURCE_FUNCTION_RANK_CLOSURE_V1',source=extent(Path(__file__)),
                 report=extent(rp),adjudication=extent(ap),binding=r['binding'],owned_commands=commands,
                 all_owned_matching_processes_absent=True,partial_V1_adjudication=partial,
                 partial_V1_admissible=False,new_array_files=len(r['records']),new_E_bytes=bytes_total,
                 protected_files_unchanged=protected,foreign_publisher_snapshot=foreign,
                 source_forwards=0,native_calls=0,DEV_arrays_read=0,optimizer_updates=0,
                 GPU_jobs=0,T4_allocations=0,goal_status='ACTIVE_INCOMPLETE',
                 scope='Full finite FIT linear rank geometry qualified by complete V2 audit. '
                       'Initial schema fault and JSON bool fault/partial bytes retained; '
                       'partial JSON trailing whitespace explicitly preserved under scoped Git check. '
                       'No factor kernel, new local quality, OWN-history chatbot, timing/DRAM '
                       'or useful-n/model-speed admission. No owned job live.')
    out=DOC/'source_function_rank_session_closure_20261013.json'
    write(out,closure)
    print(json.dumps(dict(closure=extent(out),commands=commands,new_E_bytes=bytes_total)),flush=True)

if __name__=='__main__':
    main()
