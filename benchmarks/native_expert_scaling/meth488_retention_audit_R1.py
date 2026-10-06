"""Independent admission: no import of inquiry/checks, CUDA or compiled candidate."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import sys
import time
import traceback
import psutil
import numpy as np

ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';OUT=ROOT/'results/native_expert_scaling/meth488_shared_integer_backend'
RAW=DOC/'meth488_shared_integer_backend_result.json';RET=DOC/'RETENTION_488_R1_20261006.json'
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
    with p.open('xb') as f:f.write((json.dumps(v,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())

def audit():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();assert args.out.resolve()==RET.resolve() and not RET.exists() and not RET.with_suffix('.failure.json').exists() and sys.flags.optimize==0
    start=time.monotonic();peak=hashed=0;stage='binding'
    r={'experiment':'METH488-R1 independent retained controls and complete case prefix; no rate promotion','start_utc':stamp(),'process_instance':{'pid':os.getpid(),'create_time_unix':psutil.Process().create_time()},'commands':[],'gates':{}}
    def guard():
        nonlocal peak
        v=psutil.Process().memory_info();peak=max(peak,v.rss,getattr(v,'peak_wset',0));assert peak<=512<<20 and time.monotonic()-start<=900
    def digest(p):
        nonlocal hashed
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            while b:=f.read(4<<20):h.update(b);hashed+=len(b);guard()
        return h.hexdigest()
    def exact(v):assert Path(v['path']).stat().st_size==v['bytes'] and digest(v['path'])==v['sha256'],v['path']
    def wire(p,n):
        data=Path(p).read_bytes();assert data[:8]==b'SWR32O01';s,t,d,le,ld,v,nr=struct.unpack_from('<7I',data,8)
        assert s==29 and 1<=t<=64 and (d,le,ld,v,nr)==(768,12,12,32128,6*(s+t))
        enc=(le+2)*s*d;dec=t*(ld+2)*d;ln=t*v;end=36+4*(enc+dec+ln);assert len(data)==end+12*nr
        val=np.frombuffer(data,dtype='<f4',count=enc+dec+ln,offset=36);assert np.isfinite(val).all()
        route=np.frombuffer(data,dtype=[('e','<i4'),('a','<i4'),('p','<f4')],count=nr,offset=end)
        assert ((route['e']>=0)&(route['e']<n)).all() and np.isin(route['a'],[0,1]).all() and np.isfinite(route['p']).all() and ((route['p']>0)&(route['p']<=1)).all()
        return val[enc+dec:].reshape(t,v),{'source':s,'positions':t,'all_state_route_and_head_finite':True}
    def healthy(ids):
        fields=[];nextsentinel=32099;bad=False;closed=False
        for token in ids:
            if token==1:break
            if token>=32000:
                if token!=nextsentinel or nextsentinel<32095:bad=True
                else:
                    nextsentinel-=1
                    if token==32095:closed=True
                    else:fields.append([])
            elif token<=0 or not fields or closed:bad=True
            else:fields[-1].append(token)
        triples=[[(a[j],a[j+1],a[j+2]) for j in range(max(0,len(a)-2))] for a in fields];total=sum(map(len,triples));repeated=(total-sum(len(set(a)) for a in triples))/max(1,total)
        return not bad and closed and len(fields)==4 and all(fields) and ids[-1]==32095 and repeated<=.5
    try:
        wait_start=time.monotonic();r['isolation_wait_observations']=[]
        while True:
            own={os.getpid(),*(p.pid for p in psutil.Process().parents())};r['preserved_daemons']=[];foreign=[]
            for p in psutil.process_iter(['name','cmdline']):
                if p.pid in own:continue
                name=(p.info['name'] or '').lower();a=p.info['cmdline'] or []
                if name=='pythonw.exe' and len(a)==2 and Path(a[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():r['preserved_daemons'].append(p.pid);continue
                if name.startswith(('python','clang')) or (name.startswith('meth') and name.endswith('.exe')):foreign.append({'pid':p.pid,'name':name})
            if not foreign:break
            r['isolation_wait_observations'].append({'utc':stamp(),'foreign':foreign});guard();assert time.monotonic()-wait_start<=600,'audit_R1_initial_quiet600s'
            print(json.dumps({'awaiting_initial_audit_isolation':foreign,'wait_seconds':time.monotonic()-wait_start}),flush=True);time.sleep(1)
        r['isolation_wait_seconds']=time.monotonic()-wait_start
        extension=json.loads((DOC/'meth488_audit_r1_binding.json').read_bytes())
        for v in extension['files']:exact(v)
        r['extension_binding_sha256']=digest(DOC/'meth488_audit_r1_binding.json')
        first_audit=json.loads((DOC/'RETENTION_488_20261006.failure.json').read_bytes())
        first_window=json.loads((ROOT/'results/native_expert_scaling/meth488_audit_windows_terminal.json').read_bytes())
        assert first_window['query_available'] and not first_window['matching_scientific_events']
        assert first_window['instances'][0]['pid']==first_audit['process_instance']['pid'] and first_window['instances'][0]['create_time_unix']==first_audit['process_instance']['create_time_unix']
        r['gates']['first_actual_audit_failure_and_windows_preserved']=True
        failed=not RAW.exists();path=RAW.with_suffix('.failure.json') if failed else RAW;assert path.exists() and not (RAW.exists() and RAW.with_suffix('.failure.json').exists())
        raw=json.loads(path.read_bytes());binding=json.loads((DOC/'meth488_binding.json').read_bytes());assert digest(DOC/'meth488_binding.json')==raw['binding_sha256']
        r['raw']={'path':str(path),'sha256':digest(path),'bytes':path.stat().st_size};r['main_completed']=not failed
        assert sys.version==binding['runtime']['python'] and np.__version__==binding['runtime']['versions']['numpy'] and psutil.__version__==binding['runtime']['versions']['psutil']
        for v in binding['helpers']+binding['records']+binding['toolchain']+binding['cuda']+binding['runtime']['files']:exact(v)
        deriv=json.loads((DOC/'meth486_source_derivation.json').read_bytes());candidate=(ROOT/'benchmarks/native_expert_scaling/meth486_shared_integer_model.c').read_bytes().decode()
        for old,new in reversed(deriv['replacements']):assert candidate.count(new)==1;candidate=candidate.replace(new,old)
        assert hashlib.sha256(candidate.encode()).hexdigest()==deriv['source_sha256']
        engine=(ROOT/'benchmarks/phase60/engine.c').read_bytes();prefix=b'#ifdef SILICON_SWITCH_SHARED_INTEGER_GPU_R1\r\n#include "../native_expert_scaling/meth486_shared_integer_entry.c"\r\n#elif defined(SILICON_SWITCH_SHARED_INTEGER_GPU)'
        assert engine.startswith(prefix) and hashlib.sha256(b'#ifdef SILICON_SWITCH_SHARED_INTEGER_GPU'+engine[len(prefix):]).hexdigest()==deriv['engine_original_sha256']
        for s in binding['sources']:
            exact(s['manifest']);exact(s['payload'])
            for v in s['references']:
                for kind in ('teacher','natural','donor_teacher','donor_generation'):exact(v[kind])
        r['gates']['frozen_sources_runtime_full_payload_and_reference_digests']=True
        winpath=ROOT/'results/native_expert_scaling/meth488_windows_terminal.json';win=json.loads(winpath.read_bytes())
        assert win['query_available'] and not win['matching_scientific_events'];expected=[raw['process_instance']]
        for c in raw['commands']:
            if 'process_instance' in c:expected.append(c['process_instance'])
            expected.extend(c.get('descendant_process_peaks',[]))
        assert len(win['instances'])==len(expected)
        assert [(x['pid'],x['create_time_unix']) for x in win['instances']]==[(x['pid'],x['create_time_unix']) for x in expected]
        assert datetime.datetime.fromisoformat(win['instances'][0]['start_utc'])==datetime.datetime.fromisoformat(raw['start_utc']) and datetime.datetime.fromisoformat(win['instances'][0]['end_utc'])==datetime.datetime.fromisoformat(raw['end_utc'])
        r['main_windows']={'sha256':digest(winpath),'actual_PID_creation_and_ISO_bounds':True};r['gates']['actual_terminal_windows_instance_without_application_fault']=True
        stage='ALL retained output and command inventory';inv=raw.get('all_partial_files',[]) if failed else raw['output_inventory']
        unique={}
        for v in inv:
            exact(v)
            if v['path'] in unique:assert v==unique[v['path']]
            unique[v['path']]=v
        if OUT.exists():assert set(unique)=={str(p) for p in OUT.iterdir() if p.is_file()}
        for c in raw['commands']:
            assert 'process_instance' in c and c['end_utc']>=c['start_utc']
            assert c['label']=='compile' or c.get('descendant_process_peaks',[])==[]
            if not c.get('guard_termination'):
                assert c['terminal_memory']['peak_working_set_bytes']>=c['sampled_peak_bytes'] and c['wall_seconds']<=180
                assert unique[str(OUT/(c['label']+'.stdout'))]['sha256']==c['stdout_sha256'] and unique[str(OUT/(c['label']+'.stderr'))]['sha256']==c['stderr_sha256']
        r['retained_files']=len(unique);r['retained_bytes']=sum(v['bytes'] for v in unique.values());r['gates']['ALL_command_partials_and_output_files_retained']=True
        if failed:
            assert raw['failure']['stage'] in ('bindings','one_frozen_compile','first_CUDA_runtime_and_complete_controls','whole_ownstate_teacher_and_generation')
            r['main_failure']=raw['failure'];r['partial_control_bytes']=(OUT/'controls.bin').stat().st_size if (OUT/'controls.bin').exists() else 0
            if r['partial_control_bytes']:assert (OUT/'controls.bin').read_bytes()[:12]==b'M486CTL1'+struct.pack('<I',11)
            r['gates']['failure_preserved_without_candidate_replay']=True;r['decision']='Retain sole first failure. No backend arithmetic, whole quality or speed promotion beyond completed gates.'
        assert failed and raw['failure']['exception']=="AssertionError('main45min_host24GiB_parent1GiB')"
        assert raw['resource']['wall_seconds']>2700 and raw['resource']['parent_peak_bytes']<=(1<<30) and raw['resource']['parent_peak_bytes']+raw['resource']['native_peak_bytes']<=(24<<30)
        r['gates']['main_stop_is_temporal_within_memory_bounds']=True
        stage='independent ALL controls'
        shape=[(1,1,1,0),(17,15,1,1),(8,16,1,2),(9,17,1,3),(8,4096,1,4),(768,768,1,0),(768,768,29,1),(768,768,256,3),(3072,768,1,4),(768,3072,1,2),(32128,768,1,3)]
        with (OUT/'controls.bin').open('rb') as f:
            assert f.read(12)==b'M486CTL1'+struct.pack('<I',11);r['controls']=[]
            def read(dtype,dims):
                count=int(np.prod(dims));b=f.read(count*np.dtype(dtype).itemsize);assert len(b)==count*np.dtype(dtype).itemsize;return np.frombuffer(b,dtype=dtype).reshape(dims)
            for ci,(m,k,q,pattern) in enumerate(shape):
                mp=(m+7)//8*8;kp=(k+7)//8*8;assert struct.unpack('<7I',f.read(28))==(m,k,q,mp,kp,pattern,0)
                w=read('i1',(m,k));sc=read('<f4',(m,));x=read('<f4',(q,k));codes=read('<i2',(q,k));scale=read('<f4',(q,));parts=read('<i4',(q,4,mp));rec=read('<i8',(q,m));y=read('<f4',(q,m));oracle=read('<f4',(q,m))
                ew=(np.arange(m,dtype='i4')[:,None]*17+np.arange(k,dtype='i4')[None,:]*31+ci*13)%256-128
                if pattern==4:ew=np.full((m,k),-128)
                assert np.array_equal(w,ew) and np.isfinite(x).all() and sc.tobytes()==np.array([.001,1,1000,.000001],dtype='<f4')[np.arange(m)%4].tobytes()
                j=np.arange(k)[None,:];t=np.arange(q)[:,None]
                ex=np.zeros((q,k),dtype='<f4')
                if pattern==1:ex=np.where((j+t)%2,-32767,32767).astype('<f4')
                if pattern==2:ex=np.where(j==0,np.float32(32767),((j+t)%31-15).astype('<f4')+np.float32(.5)).astype('<f4')
                if pattern==3:ex=(((j*23+t*11)%1009-504).astype('<f4')*np.float32(.03125)).astype('<f4')
                if pattern==4:ex.fill(32767)
                assert x.tobytes()==ex.tobytes();maximum=np.abs(x).max(axis=1);qs=(maximum/np.float32(32767)).astype('<f4');qs[maximum==0]=1
                qc=np.minimum(32767,np.maximum(-32767,np.rint(np.divide(x,qs[:,None],dtype='f4')))).astype('<i2');assert qc.tobytes()==codes.tobytes() and qs.tobytes()==scale.tobytes()
                signed=codes.astype('i8');digits=[signed%128,(signed//128)%128,signed//16384];assert np.array_equal(digits[0]+128*digits[1]+16384*digits[2],signed)
                for a,digit in enumerate(digits):
                    ref=digit@w.astype('i8').T;assert np.array_equal(ref,parts[:,a,:m]) and np.abs(ref).max()<2**31
                assert not np.any(parts[:,:,m:]) and not np.any(parts[:,3])
                dot=signed@w.astype('i8').T;rebuild=parts[:,0,:m].astype('i8')+128*parts[:,1,:m].astype('i8')+16384*parts[:,2,:m].astype('i8')
                expected=((dot.astype('f8')*sc.astype('f8'))*qs.astype('f8')[:,None]).astype('<f4')
                assert np.array_equal(dot,rec) and np.array_equal(dot,rebuild) and expected.tobytes()==y.tobytes()==oracle.tobytes()
                r['controls'].append({'case':ci,'all_partial_reconstruction_and_output_bytes_exact':True});guard()
            assert f.read(1)==b''
        r['gates']['independent_ALL11_control_digits_partial_integer_and_float_bytes']=True
        del w,sc,x,codes,scale,parts,rec,y,oracle,ew,j,t,ex,maximum,qs,qc,signed,digits,ref,dot,rebuild,expected
        stage='complete retained source case prefix';prior=json.loads((DOC/'meth458_switch_matched_whole_cost_result.json').read_bytes());r['sources']={}
        total_outputs=0
        for s in binding['sources']:
            n=s['n'];got=raw['sources'][str(n)];qual=json.loads(Path(s['quality_path']).read_bytes());cohort=json.loads(Path(s['cohort_path']).read_bytes());cases=[]
            for c in got['cases']:
                full=all(str(b) in c['teacher'] for b in (0,1)) and all(len(c['generation'][str(b)].get(str(p),[]))==4 for b in (0,1) for p in (0,1))
                if not full:break
                cases.append(c)
            assert len(cases)==(96 if n==128 else 78) and all(qual['gates'].values())
            all_outputs=0
            for index,c in enumerate(cases):
                bi,ci=divmod(index,4);old=prior['sources'][str(n)]['cases'][index];qc=qual['books'][bi]['cases'][ci];case=cohort['items'][bi]['cases'][ci]
                assert (c['book'],c['case'],c['source_id'])==(bi,ci,old['source_id']) and c['backend_order']==[index%2,1-index%2]
                for b in (0,1):
                    tp=OUT/f'n{n}.book{bi}.case{ci}.backend{b}.teacher.0.bin';assert unique[str(tp)]['sha256']==qc['native_output_sha256'];logits,_=wire(tp,n);all_outputs+=1
                    vals=logits.astype('f8');target=np.array(case['target_ids']);nll=np.logaddexp.reduce(vals,axis=1)-vals[np.arange(14),target]
                    assert np.allclose(nll,qc['native']['per_target_nll'],rtol=0,atol=1e-10) and logits.argmax(-1).tolist()==qc['native']['greedy_teacher_forced_ids']
                    for profile in (0,1):
                        rows=c['generation'][str(b)][str(profile)];assert [a['repetition'] for a in rows]==[-1,0,1,2]
                        logpath=OUT/f'n{n}.book{bi}.case{ci}.backend{b}.profile{profile}.stdout';assert [json.loads(a) for a in logpath.read_text().splitlines()]==[{k:v for k,v in a.items() if k!='output_sha256'} for a in rows]
                        for a in rows:
                            fp=OUT/f"n{n}.book{bi}.case{ci}.backend{b}.profile{profile}.{a['repetition']}.bin";assert unique[str(fp)]['sha256']==qc['generation']['native_generation_sha256']==a['output_sha256'];head,h=wire(fp,n);ids=head.argmax(-1).tolist();all_outputs+=1
                            assert ids==a['generated_ids']==qc['generation']['native']['generated_ids'] and healthy(ids)==c['healthy_accepted']==old['healthy_accepted']
                            assert (len(ids),sum(1<i<32000 for i in ids))==(c['generated_tokens'],c['prose_tokens'])
                            assert [x['actual_mask'] for x in a['worker_affinity']]==[1<<v for v in s['affinity']] and a['threads']==s['threads'] and a['profile']==profile
                            for ph in (0,1):
                                for kd in range(4):
                                    for key in ('calls','code_bytes','f32_bytes','scale_bytes'):assert a['counters'][ph][kd][key]==old['modes'][str(profile)][a['repetition']+1]['counters'][ph][kd][key]
                            gpu=a['gpu'];assert gpu['enabled']==bool(b)
                            e={'calls':60+24*h['source']+85*h['positions'],'queries':84*h['source']+85*h['positions'],'h2d_bytes':313344*h['source']+316416*h['positions'],'d2h_bytes':1253376*h['source']+1767424*h['positions']}
                            for key,value in e.items():assert gpu[key]==(value if b else 0)
                            if b:assert gpu['matrix_count']==169 and gpu['weight_bytes']==166232064 and gpu['device_explicit_bytes']==178552832
                            assert a['full_generation_seconds']>0 and abs(a['full_generation_seconds']-a['encoder_seconds']-a['cross_kv_seconds']-a['decode_greedy_seconds'])<3e-8
            assert all_outputs==18*len(cases);total_outputs+=all_outputs
            r['sources'][str(n)]={'complete_cases':len(cases),'whole_output_files_byte_exact':all_outputs,'teacher_NLL_argmax_rederived':True,'ALL_state_route_head_bytes_and_ownstate_ID_acceptance_counters_verified':True,
                                 'donor_scope':'All96 source128 cases inherit the qualified ALL18 donor comparison through exact output identity. Source256 only the first78 case outputs; no new aggregate ALL18 subset claim.'}
        assert total_outputs==3132;r['gates']['ALL174_complete_cases_3132_outputs_exact_source_ownstate_and_counters']=True
        assert raw['sources']['128']['summaries']['0']['0']['seed']==486486
        assert 'seed485485' in (DOC/'METH_486_SHARED_INTEGER_BACKEND_PROTOCOL_20261006.md').read_text()
        r['gates']['unadmitted_bootstrap_seed_discrepancy_retained']=True
        r['performance_scope']='NO rate, bootstrap, decomposition or BOTH-source performance admission. Original ALL192 cohort incomplete; precomputed source128 summaries use486486 contrary to protocol485485. Later metadata correction must use485485 without model replay.'
        r['decision']='Independently admit ALL11 actual arithmetic controls and exact completed174-case prefix only. Preserve main time stop and original audit isolation failure. No complete192/3456 or >=50 qualification. Full goal active/incomplete.'
        r['end_utc']=stamp();guard();r['resource']={'wall_seconds':time.monotonic()-start,'peak_working_set_bytes':peak,'bytes_hashed':hashed};write(RET,r);print(json.dumps({'retention':str(RET),'main_completed':r['main_completed'],'gates':r['gates'],'resource':r['resource']}),flush=True)
    except BaseException as e:
        r['failure']={'stage':stage,'exception':repr(e),'traceback':traceback.format_exc()};r['end_utc']=stamp();r['resource']={'wall_seconds':time.monotonic()-start,'peak_working_set_bytes':peak,'bytes_hashed':hashed};write(RET.with_suffix('.failure.json'),r);print(json.dumps(r['failure']),flush=True);raise
if __name__=='__main__':audit()
