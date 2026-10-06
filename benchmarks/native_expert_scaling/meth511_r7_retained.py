"""Operational reuse of 553 closed C calls; no model or numerical kernel call."""
import datetime
import json
from pathlib import Path
import psutil

FAULT_SHA = '1384c005600e796dfd1cf47936c084e0ca35f1144f18109dedca2d12bb166e1b'
BIND_SHA = 'aeb6a9fa6438a4ab63efc258583de08f3635332c6a3042f05ab5ed1d478bb1f7'

def expected_labels():
    labels=[]
    for bi in range(24):
        for ci in range(4):
            for arm in (['source','candidate'] if (4*bi+ci)%2==0 else ['candidate','source']):
                labels.extend(f'b{bi:02d}c{ci}.{arm}.{kind}' for kind in ['teacher','gen','profile'])
    return labels

def cpu(rows):
    return {(r['pid'],r['create_time_unix']):r['cpu_seconds'] for r in rows}

def closed(instance):
    try:assert abs(psutil.Process(instance['pid']).create_time()-instance['create_time_unix'])>=.002
    except psutil.NoSuchProcess:pass

def verify(prior):
    assert prior['native_calls']==553 and prior['model_calls']==0
    assert [c['label'] for c in prior['commands']]==expected_labels()[:553]
    baseline=cpu(prior['commands'][0]['idle_start'])
    for c in prior['commands']:
        assert c['returncode']==0 and c['descendants']==[] and c['timing_observations']
        assert cpu(c['idle_start'])==cpu(c['idle_end'])==baseline
        assert all(cpu(o['idle'])==baseline for o in c['timing_observations'])
        closed(c['process_instance'])
    closed(prior['process_instance'])
    assert 'idle_process_CPU_or_identity_change' in prior['traceback']
    assert "foreign,kept=self.scientific_processes(ancestors);assert not foreign,('foreign_before',foreign)" in prior['traceback']
    assert prior['OS_peak_bytes']<2<<30 and prior['native_OS_peak_bytes']<4<<30
    assert prior['seconds']+1100<2400

def bind(O,b,catalog,item,retain):
    binding=item(O.DOC/'meth511_r6_binding.json');assert binding['sha256']==BIND_SHA
    old=json.loads(Path(binding['path']).read_bytes())
    fault=item(O.DOC/'meth511_r6_native_result.failure.json');assert fault['sha256']==FAULT_SHA
    prior=json.loads(Path(fault['path']).read_bytes());assert prior['binding_sha256']==BIND_SHA;verify(prior)
    for r in old['catalog']:
        key=r['path'].lower()
        if key in catalog:assert catalog[key]==r
        else:retain(r);catalog[key]=r
    for r in old['scientific']:
        if r['path'] not in {v['path'] for v in b['scientific']}:b['scientific'].append(r)
    out=O.ROOT/'results/native_expert_scaling/meth511_r6_native';inventory=[]
    assert (out/'fatal.log').stat().st_size==0 and len(list(out.glob('*.native.json')))==92
    for p in sorted(out.iterdir()):
        assert p.is_file();retain({'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns})
        r=item(p);catalog[r['path'].lower()]=r;inventory.append(r)
    lookup={r['path']:r for r in inventory}
    for p in out.glob('*.native.json'):
        case=json.loads(p.read_bytes())
        for arm in ['source','candidate']:
            for rows in case['native'][arm].values():
                for row in rows:
                    assert lookup[row['wire']['path']]['sha256']==row['wire']['sha256']
    assert not (out/'b23c0.source.gen.stdout').exists()
    for r in [binding,fault]:catalog[r['path'].lower()]=r
    b['retained_native']={'binding':binding,'fault':fault,'completed_calls':553,
        'output_bytes':sum(r['bytes'] for r in inventory),'inventory':inventory,
        'original_resource':{k:prior[k] for k in ['seconds','OS_peak_bytes','native_OS_peak_bytes']},
        'resource_scope':'Original first-fault guard peaks, not a post-fault serialization peak.',
        'scope':'553 closed calls with constant idle counters; service tick occurred in preflight before call554. New23 calls use a separately rebound quiet interval. No old C call is repeated.'}

def initialize(ctx,b):
    receipt=b['retained_native'];ctx.exact(receipt['fault']);prior=json.loads(Path(receipt['fault']['path']).read_bytes());verify(prior)
    ctx.retained_commands={r['label']:r for r in prior['commands']};ctx.retained_seen=[]
    ctx.r['commands']=prior['commands'];ctx.r['retained_native_calls']=553
    ctx.r['retained_native_resource']=receipt['original_resource'];ctx.r['retained_native_scope']=receipt['scope']
    ctx.r['retained_native_failure_sha256']=FAULT_SHA
    ctx.r['retained_process_instances']=[{'label':'native_R6','pid':prior['process_instance']['pid'],
        'create_time_unix':prior['process_instance']['create_time_unix'],'start_utc':prior['started_utc'],'end_utc':ctx.r['started_utc']}]
    ctx.native_peak=max(ctx.native_peak,prior['native_OS_peak_bytes']);ctx.retained_output_bytes=receipt['output_bytes']

def prefix(ctx,label):
    return Path(ctx.retained_commands[label]['argv'][4]) if label in ctx.retained_commands else ctx.out/label

def reuse_or_run(ctx,argv,label):
    if label not in ctx.retained_commands:
        assert ctx.retained_seen==expected_labels()[:553]
        return ctx.run_native(argv,label)
    c=ctx.retained_commands[label];assert list(map(str,argv))==c['argv'];closed(c['process_instance'])
    assert label==expected_labels()[len(ctx.retained_seen)];ctx.retained_seen.append(label)
    p=Path(c['argv'][4]).parent/(label+'.stdout');rows=[json.loads(v) for v in p.read_text(encoding='utf8').splitlines()]
    assert (p.parent/(label+'.stderr')).stat().st_size==0
    ctx.log(retained_closed_command_reused=label);ctx.guard();return rows
