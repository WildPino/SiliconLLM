"""Custody closure of source-only ternary precision assay and its full audit."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(ROOT/'benchmarks/native_expert_scaling'),
              str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
import psutil


def main():
    rp=DOC/'ternary_readout_precision_result_20261013.json'
    ap=DOC/'ternary_readout_precision_adjudication_20261013.json'
    r=json.loads(rp.read_bytes())
    ad=json.loads(ap.read_bytes())
    b=json.loads(Path(r['binding']['path']).read_bytes())
    assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(rp)
    assert ad['binding']==r['binding'] and extent(r['binding']['path'])==r['binding']
    assert ad['FIT_selected_planes']==r['FIT_selected_planes'] and ad['selected_DEV_all_pass']==r['selected_DEV_all_pass']
    files=[q['file'] for q in r['records']]+[e['file'] for m in r['models'] for e in m['arms']]
    for e in files:
        assert extent(e['path'])==e
    total=sum(e['bytes'] for e in files)
    assert total==r['output_bytes']<=b['limits']['output_bytes']
    commands=[]
    for role,name in [('preparation','ternary_readout_precision_binding_20261013.preparation_terminal.json'),
                      ('producer','ternary_readout_precision_result_20261013.terminal.json'),
                      ('audit','ternary_readout_precision_adjudication_20261013.terminal.json')]:
        p=DOC/name
        t=json.loads(p.read_bytes())
        assert t['exit_code']==0 and extent(t['result']['path'])==t['result']
        live=False
        if psutil.pid_exists(t['pid']):
            live=abs(psutil.Process(t['pid']).create_time()-t['creation_time'])<1e-5
        assert not live,(role,t['pid'])
        commands.append(dict(role=role,terminal=extent(p),pid=t['pid'],creation_time=t['creation_time'],
                             exit_code=0,seconds=t['seconds'],OS_peak=t['OS_peak'],matching_process_absent=True))
    assert ad['terminal']==commands[1]['terminal']
    protected=[]
    for rel in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json',
                'benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']:
        p=(ROOT/rel).resolve()
        expected=next(e for e in b['inputs'] if Path(e['path']).resolve()==p)
        assert extent(p)==expected
        protected.append(expected)
    foreign=[]
    for p in psutil.process_iter(['pid','create_time','name','cmdline']):
        try:
            if 'daemon_workflow.py' in ' '.join(p.info['cmdline'] or []):
                foreign.append(p.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            pass
    result=dict(schema='TERNARY_READOUT_PRECISION_CLOSURE_V1',source=extent(Path(__file__)),report=extent(rp),
                adjudication=extent(ap),binding=r['binding'],owned_commands=commands,all_owned_matching_processes_absent=True,
                new_array_files=len(files),new_E_bytes=total,protected_files_unchanged=protected,foreign_publisher_snapshot=foreign,
                FIT_selected_planes=r['FIT_selected_planes'],selected_DEV_all_pass=r['selected_DEV_all_pass'],
                source_forwards=0,native_calls=0,activation_quantizations=0,optimizer_updates=0,GPU_jobs=0,T4_allocations=0,
                goal_status='ACTIVE_INCOMPLETE',
                scope='Full source-weight-only ternary packing/numerical response assay fully audited; '
                      'BF16 output words archived,raw F64 metrics recomputed without archived raw arrays. '
                      'No native LUT/physical DRAM/OWN-history chatbot/useful conditional-n/same useful50 admission. '
                      'Three scientific commands terminal0/no owned run live/no new mechanical scientific fault.')
    out=DOC/'ternary_readout_precision_session_closure_20261013.json'
    write(out,result)
    print(json.dumps(dict(closure=extent(out),commands=commands,new_E_bytes=total)),flush=True)


if __name__=='__main__':
    main()
