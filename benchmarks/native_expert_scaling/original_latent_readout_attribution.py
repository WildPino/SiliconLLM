"""Two stored h24P/actual51 decoder interventions; never instantiate a model."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from original_packed_capacity import SITE,extent,sha,write
sys.path.insert(0,str(SITE))
V,D=65537,256


def bind(a):
    import numpy as np
    import numpy._core._multiarray_umath as native_numpy
    from original_joint_history_recovery_audit import packed_fields
    result=DOC/'original_joint_history_recovery_finish_result_20261010.json'
    terminal=result.with_suffix('.terminal.json');r=json.loads(result.read_bytes());t=json.loads(terminal.read_bytes())
    audit=DOC/'original_joint_history_recovery_stored_adjudication_finish_20261010.json'
    receipt=audit.with_suffix('.receipt.json');ar=json.loads(audit.read_bytes());rr=json.loads(receipt.read_bytes())
    assert t['exit_code']==rr['exit_code']==0 and rr['error'] is None
    assert t['result_sha256']==sha(result)==ar['result']['sha256']
    assert rr['output']==extent(audit) and ar['end_hash_count']==1890
    assert rr['seconds']<=rr['limits']['seconds'] and rr['worker_OS_peak_through_exit']+rr['launcher_OS_peak_snapshot']<=rr['limits']['OS_bytes']
    for item in rr['inputs']:assert extent(item['path'])==item
    assert ar['sealed_input_metadata_unchanged'] and ar['decision']=='JOINT_HISTORY_B_NOT_SUPPORTED'
    pbpath=DOC/'original_joint_history_recovery_binding_20261009.json';pb=json.loads(pbpath.read_bytes())
    assert ar['binding']==extent(pbpath)
    records=[rec for rec in pb['records'] if rec['split']=='DEV'];assert len(records)==24
    assert sum(len(rec['positions']) for rec in records)==4386
    heads=[]
    files=[Path(__file__),B/'original_latent_readout_attribution_launch.py',
        B/'original_latent_readout_attribution_audit.py',B/'original_latent_readout_attribution_audit_launch.py',
        B/'original_joint_history_recovery_audit.py',
        B/'original_packed_capacity.py',B/'chatbot_falcon_usability.py',B/'original_falcon_whole_recovery.py',
        DOC/'ORIGINAL_LATENT_READOUT_ATTRIBUTION_PROTOCOL_20261010.md',
        DOC/'ORIGINAL_LATENT_READOUT_ATTRIBUTION_NEXT_20261010.md',result,terminal,audit,receipt,pbpath,
        Path(sys.executable),Path(native_numpy.__file__),SITE/'numpy/__init__.py',SITE/'psutil/__init__.py']
    files+=sorted((SITE/'numpy.libs').glob('*.dll'))
    for arm in r['arms']:
        fields=packed_fields(arm['final_packed']['path'])
        assert fields['head']['shape']==(V,D) and fields['final_norm']['shape']==(D,)
        assert fields['head']['dtype']==fields['final_norm']['dtype']==1
        heads.append(dict(name=arm['name'],packed=arm['final_packed'],head=fields['head'],
            final_norm=fields['final_norm'],native=arm['milestones'][-1]['native']['aggregate']))
        files.append(Path(arm['final_packed']['path']))
    targets={rec['id']:pb['boundary_targets'][rec['id']][-1] for rec in records}
    for rec in records:
        assert len(rec['student_input_ids'])*D*4==targets[rec['id']]['bytes']
        assert len(rec['positions'])*V*2==rec['logits']['bytes']
        assert len(set(rec['positions']))==len(rec['positions'])
        files.extend((Path(targets[rec['id']]['path']),Path(rec['logits']['path'])))
    binding=dict(schema='ORIGINAL_LATENT_READOUT_ATTRIBUTION_BINDING_V1',result=extent(result),
        parent_adjudication=extent(audit),parent_receipt=extent(receipt),records=records,targets=targets,heads=heads,
        dtype='float64',labels_per_arm=4386,score_bytes=2*4386*V*8,eps=1e-5,chunk_labels=16,
        gates=dict(reference_absolute=1e-10,native_KL=1.,native_disagreement=.20,domain_KL=2.,domain_disagreement=.35,
            good_KL_relative=.50,good_disagreement_relative=.80,poor_KL_relative=.80),
        limits=dict(seconds=900,OS_bytes=4<<30,output_bytes=5<<30,log_bytes=8<<20),
        inputs=[extent(p) for p in dict.fromkeys(files)],
        runtime=dict(Python=sys.version,NumPy=np.__version__,threads=6,principal_numpy_dlls=[str(p) for p in sorted((SITE/'numpy.libs').glob('*.dll'))]),
        scope='Exactly two h24P/actual51 decoder interventions on existing24DEV/4386 labels. No fit/map sweep/model/native/source/GPU calls. '
            'F64 full-V output retained losslessly. This diagnostic is not useful chatbot/speed/n/DRAM admission; inherited failures remain.')
    assert binding['score_bytes']<binding['limits']['output_bytes']
    write(a.out,binding);print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(binding['inputs']),score_bytes=binding['score_bytes'])),flush=True)


def row_metrics(teacher,student):
    import numpy as np
    q=teacher.astype('f8');p=student.astype('f8');assert q.shape==p.shape==(V,)
    assert np.isfinite(q).all() and np.isfinite(p).all()
    q-=q.max();p-=p.max()
    qsum=math.log(float(np.exp(q).sum()));psum=math.log(float(np.exp(p).sum()))
    qref=float(np.logaddexp.reduce(q));pref=float(np.logaddexp.reduce(p))
    primary_q=q-qsum;primary_p=p-psum;ref_q=q-qref;ref_p=p-pref
    loss=float(np.sum(np.exp(primary_q)*(primary_q-primary_p)))
    reference=float(np.dot(np.exp(ref_q),ref_q-ref_p))
    entropy=-float(np.sum(np.exp(primary_q)*primary_q))
    delta=max(abs(qsum-qref),abs(psum-pref),abs(loss-reference))
    assert delta<=1e-10,(delta,loss,reference)
    assert loss>=-1e-10 and entropy>=0
    return loss,int(np.argmax(student)!=np.argmax(teacher)),entropy,math.log(V)-entropy,delta


def aggregate(rows):
    def group(values):
        labels=sum(r['labels'] for r in values)
        return dict(cases=len(values),labels=labels,case_KL=sum(r['KL'] for r in values)/len(values),
            label_KL=sum(sum(r['KL_per_label']) for r in values)/labels,
            case_disagreement=sum(r['disagreement_rate'] for r in values)/len(values),
            label_disagreement=sum(r['disagreement'] for r in values)/labels,
            case_teacher_entropy=sum(r['teacher_entropy'] for r in values)/len(values),
            label_teacher_entropy=sum(sum(r['entropy_per_label']) for r in values)/labels,
            case_uniform_KL=sum(r['uniform_KL'] for r in values)/len(values),
            label_uniform_KL=sum(sum(r['uniform_KL_per_label']) for r in values)/labels)
    return dict(**group(rows),domains={domain:group([r for r in rows if r['domain']==domain]) for domain in sorted({r['domain'] for r in rows})})


def decision(arms,g):
    flags=[]
    for arm in arms:
        agg,native=arm['aggregate'],arm['native']
        screens=dict(KL=agg['case_KL']<=g['native_KL'],disagreement=agg['case_disagreement']<=g['native_disagreement'],
            domain_KL=all(r['case_KL']<=g['domain_KL'] for r in agg['domains'].values()),
            domain_disagreement=all(r['case_disagreement']<=g['domain_disagreement'] for r in agg['domains'].values()))
        good=agg['case_KL']<=g['good_KL_relative']*native['case_KL'] and agg['case_disagreement']<=g['good_disagreement_relative']*native['case_disagreement']
        poor=agg['case_KL']>=g['poor_KL_relative']*native['case_KL'] and not all(screens.values())
        flags.append(dict(name=arm['name'],target_readable_substantially_better=good,
            target_does_not_make_current_head_sufficient=poor,absolute_domain_screens=screens))
    assert len(flags)==2 and [x['name'] for x in flags]==['A','B']
    branch='HISTORY_FUNCTION_RECOVERY' if all(x['target_readable_substantially_better'] for x in flags) else (
        'TARGET_DECODER_COUPLING' if all(x['target_does_not_make_current_head_sufficient'] for x in flags) else 'MIXED_NO_UNIQUE_BOTTLENECK')
    return branch,flags


def worker(a):
    import numpy as np
    import psutil
    started=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(6)))
    assert sha(a.binding)==a.binding_sha;binding=json.loads(a.binding.read_bytes());a.directory.mkdir(exist_ok=False)
    assert np.__version__==binding['runtime']['NumPy']=='2.4.6'
    rows=[];stage='startup';arm='none';ref_peak=0.
    def guard():
        assert time.monotonic()-started<=binding['limits']['seconds'],'attribution worker deadline'
        assert proc.memory_info().peak_wset<=binding['limits']['OS_bytes'],'OS cap'
        assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=binding['limits']['output_bytes'],'output cap'
    try:
        for head in binding['heads']:
            arm=head['name'];stage='head_loading';guard();path=head['packed']['path'];hf=head['head'];nf=head['final_norm']
            mmap=np.memmap(path,dtype='<f4',mode='r',offset=hf['offset'],shape=tuple(hf['shape']))
            W=np.array(mmap,dtype='f8');del mmap
            gamma=np.fromfile(path,dtype='<f4',count=D,offset=nf['offset']).astype('f8')
            assert W.shape==(V,D) and gamma.shape==(D,) and np.isfinite(W).all() and np.isfinite(gamma).all()
            case_rows=[]
            for rec in binding['records']:
                stage='label_contraction';guard();n=len(rec['student_input_ids']);m=len(rec['positions'])
                z=np.memmap(binding['targets'][rec['id']]['path'],dtype='<f4',mode='r',shape=(n,D))
                target=np.array(z[rec['positions']],dtype='f8');del z
                assert np.isfinite(target).all()
                target/=np.sqrt(np.mean(target*target,axis=-1,keepdims=True)+binding['eps']);target*=gamma
                teacher=np.memmap(rec['logits']['path'],dtype='<u2',mode='r',shape=(m,V))
                scores=a.directory/(arm+'.'+rec['id']+'.scores.f64');losses=[];dis=0;ent=[];uniform=[];case_peak=0.
                with scores.open('xb') as stream:
                    for first in range(0,m,binding['chunk_labels']):
                        guard();last=min(first+binding['chunk_labels'],m);value=target[first:last]@W.T
                        assert value.shape==(last-first,V) and np.isfinite(value).all()
                        stream.write(value.astype('<f8',copy=False).tobytes())
                        for j in range(first,last):
                            q=(teacher[j].astype('<u4')<<16).view('<f4')
                            loss,bad,entropy,u,delta=row_metrics(q,value[j-first]);losses.append(loss);dis+=bad;ent.append(entropy);uniform.append(u);case_peak=max(case_peak,delta)
                        del value,q
                    stream.flush();os.fsync(stream.fileno())
                assert scores.stat().st_size==m*V*8
                row=dict(arm=arm,id=rec['id'],split='DEV',domain=rec['domain'],history=n,labels=m,
                    positions=rec['positions'],KL=float(np.mean(losses)),KL_per_label=losses,
                    disagreement=dis,disagreement_rate=dis/m,teacher_entropy=float(np.mean(ent)),entropy_per_label=ent,
                    uniform_KL=float(np.mean(uniform)),uniform_KL_per_label=uniform,max_reference_delta=case_peak,
                    scores=extent(scores),target=binding['targets'][rec['id']],teacher=rec['logits'])
                write(a.directory/(arm+'.'+rec['id']+'.case.json'),row);case_rows.append(row);rows.append(row)
                ref_peak=max(ref_peak,case_peak);del target,teacher;gc.collect();guard()
                print(json.dumps(dict(stage=stage,arm=arm,id=rec['id'],completed=len(case_rows),seconds=time.monotonic()-started)),flush=True)
            summary=dict(name=arm,records=case_rows,aggregate=aggregate(case_rows),native=head['native'])
            write(a.directory/(arm+'.arm.json'),summary);del W,gamma;gc.collect()
        arms=[json.loads((a.directory/(name+'.arm.json')).read_bytes()) for name in ('A','B')]
        branch,flags=decision(arms,binding['gates']);stage='complete';guard()
        assert sum(r['labels'] for r in rows)==2*4386
        result=dict(schema='ORIGINAL_LATENT_READOUT_ATTRIBUTION_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
            parent_adjudication=binding['parent_adjudication'],arms=arms,decision=branch,decision_flags=flags,
            reference_max_absolute_delta=ref_peak,new_label_score_bytes=sum(r['scores']['bytes'] for r in rows),
            label_head_contractions=2*4386,source_calls=0,model_calls=0,native_calls=0,optimizer_updates=0,GPU_calls=0,reserved_queries=0,
            quality_admission=False,speed_admission=False,useful_large_n_admission=False,physical_DRAM_bytes=None,
            elapsed_seconds=time.monotonic()-started,worker_OS_peak_snapshot=proc.memory_info().peak_wset,
            scope=binding['scope'],limits='Target injection tests compatibility of retained projected states with learned heads; '
                'neither optimal decoder nor deployable chatbot. Original training/audit deadlines, numeric/source/full-runtime/family gaps retained.')
        write(a.out,result);print(json.dumps(dict(stage='complete',decision=branch,seconds=result['elapsed_seconds'])),flush=True)
    except BaseException as error:
        write(a.directory/'first_fault.json',dict(stage=stage,arm=arm,error=repr(error),seconds=time.monotonic()-started,
            completed_cases=[r['scores'] for r in rows],completed_case_ids=[(r['arm'],r['id']) for r in rows],model_calls=0,optimizer_updates=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bind',action='store_true')
    parser.add_argument('--binding',type=Path);parser.add_argument('--binding-sha');parser.add_argument('--freeze')
    parser.add_argument('--directory',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.bind:bind(args)
    else:worker(args)
