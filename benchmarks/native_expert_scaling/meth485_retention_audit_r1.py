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

ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';OUT=ROOT/'results/native_expert_scaling/meth485_shared_integer_backend'
RAW=DOC/'meth485_shared_integer_backend_result.json';RET=DOC/'RETENTION_485_R1_20261006.json'
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
    with p.open('xb') as f:f.write((json.dumps(v,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())

def audit():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();assert args.out.resolve()==RET.resolve() and not RET.exists() and not RET.with_suffix('.failure.json').exists() and sys.flags.optimize==0
    start=time.monotonic();peak=hashed=0;stage='binding'
    r={'experiment':'METH485-R1 independent retention: semantic ISO repair, original main unchanged','start_utc':stamp(),'process_instance':{'pid':os.getpid(),'create_time_unix':psutil.Process().create_time()},'commands':[],'gates':{}}
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
        own={os.getpid(),*(p.pid for p in psutil.Process().parents())};r['preserved_daemons']=[]
        for p in psutil.process_iter(['name','cmdline']):
            if p.pid in own:continue
            name=(p.info['name'] or '').lower();a=p.info['cmdline'] or []
            if name=='pythonw.exe' and len(a)==2 and Path(a[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():r['preserved_daemons'].append(p.pid);continue
            assert not(name.startswith(('python','clang')) or (name.startswith('meth') and name.endswith('.exe'))),('foreign',p.pid,name)
        failed=not RAW.exists();path=RAW.with_suffix('.failure.json') if failed else RAW;assert path.exists() and not (RAW.exists() and RAW.with_suffix('.failure.json').exists())
        raw=json.loads(path.read_bytes());binding=json.loads((DOC/'meth485_binding.json').read_bytes());assert digest(DOC/'meth485_binding.json')==raw['binding_sha256']
        r['raw']={'path':str(path),'sha256':digest(path),'bytes':path.stat().st_size};r['main_completed']=not failed
        extension=json.loads((DOC/'meth485_audit_r1_binding.json').read_bytes());assert extension['original_main_not_replayed'] is True
        for v in extension['files']:exact(v)
        r['audit_r1_binding_sha256']=digest(DOC/'meth485_audit_r1_binding.json');r['original_audit_failure_retained']=True
        assert sys.version==binding['runtime']['python'] and np.__version__==binding['runtime']['versions']['numpy'] and psutil.__version__==binding['runtime']['versions']['psutil']
        for v in binding['helpers']+binding['records']+binding['toolchain']+binding['cuda']+binding['runtime']['files']:exact(v)
        deriv=json.loads((DOC/'meth485_source_derivation.json').read_bytes());candidate=(ROOT/'benchmarks/native_expert_scaling/meth485_shared_integer_model.c').read_bytes().decode()
        for old,new in reversed(deriv['replacements']):assert candidate.count(new)==1;candidate=candidate.replace(new,old)
        assert hashlib.sha256(candidate.encode()).hexdigest()==deriv['source_sha256']
        engine=(ROOT/'benchmarks/phase60/engine.c').read_bytes();prefix=b'#ifdef SILICON_SWITCH_SHARED_INTEGER_GPU\r\n#include "../native_expert_scaling/meth485_shared_integer_entry.c"\r\n#elif defined(SILICON_SWITCH_FUNCTION_CAPTURE)'
        assert engine.startswith(prefix) and hashlib.sha256(b'#ifdef SILICON_SWITCH_FUNCTION_CAPTURE'+engine[len(prefix):]).hexdigest()==deriv['engine_original_sha256']
        for s in binding['sources']:
            exact(s['manifest']);exact(s['payload'])
            for v in s['references']:
                for kind in ('teacher','natural','donor_teacher','donor_generation'):exact(v[kind])
        r['gates']['frozen_sources_runtime_full_payload_and_reference_digests']=True
        winpath=ROOT/'results/native_expert_scaling/meth485_windows_terminal.json';win=json.loads(winpath.read_bytes())
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
            if r['partial_control_bytes']:assert (OUT/'controls.bin').read_bytes()[:12]==b'M485CTL1'+struct.pack('<I',11)
            r['gates']['failure_preserved_without_candidate_replay']=True;r['decision']='Retain sole first failure. No backend arithmetic, whole quality or speed promotion beyond completed gates.'
        else:
            stage='independent ALL controls'
            shape=[(1,1,1,0),(17,15,1,1),(8,16,1,2),(9,17,1,3),(8,4096,1,4),(768,768,1,0),(768,768,29,1),(768,768,256,3),(3072,768,1,4),(768,3072,1,2),(32128,768,1,3)]
            with (OUT/'controls.bin').open('rb') as f:
                assert f.read(12)==b'M485CTL1'+struct.pack('<I',11);r['controls']=[]
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
            stage='ALL fresh complete source parity and rate';prior=json.loads((DOC/'meth458_switch_matched_whole_cost_result.json').read_bytes());r['sources']={}
            for s in binding['sources']:
                n=s['n'];got=raw['sources'][str(n)];qual=json.loads(Path(s['quality_path']).read_bytes());cohort=json.loads(Path(s['cohort_path']).read_bytes());cases=got['cases'];assert len(cases)==96 and all(qual['gates'].values())
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
                summaries={};profile_admission={}
                for b in (0,1):
                    summaries[str(b)]={};times_byprofile=[]
                    for profile in (0,1):
                        timebooks=[];oc=[];pc=[];cold=[];loads=[]
                        for bi in range(24):
                            block=cases[bi*4:bi*4+4];times=0.;first=0.;load=0.;o=p=0
                            for c in block:
                                a=c['generation'][str(b)][str(profile)];times+=statistics.mean(x['full_generation_seconds'] for x in a if x['repetition']>=0);first+=a[0]['load_seconds']+a[0]['full_generation_seconds'];load+=a[0]['load_seconds']
                                if c['healthy_accepted']:o+=c['generated_tokens'];p+=c['prose_tokens']
                            timebooks.append(times);cold.append(first);loads.append(load);oc.append(o);pc.append(p)
                        rng=np.random.default_rng(485485);selection=rng.integers(0,24,size=(10000,24));den=np.array(timebooks)[selection].sum(axis=1)
                        expected=got['summaries'][str(b)][str(profile)];assert abs(expected['sum_case_mean_seconds']-sum(timebooks))<1e-10 and abs(expected['sum_first_request_with_load_seconds']-sum(cold))<1e-10 and abs(expected['sum_load_seconds']-sum(loads))<1e-10
                        for key,counts in [('ordinary',oc),('prose',pc)]:
                            draws=np.array(counts)[selection].sum(axis=1)/den;value=expected[key];assert value['accepted_ids']==sum(counts)
                            for label,number in [('warm_rate',sum(counts)/sum(timebooks)),('warm_lower95',float(np.quantile(draws,.05))),('warm_upper95',float(np.quantile(draws,.95))),('first_request_load_charged_rate',sum(counts)/sum(cold))]:assert abs(value[label]-number)<1e-10
                            for k in (1,10,100):assert abs(value['amortized_rates_requests_per_process'][str(k)]-sum(counts)/(sum(timebooks)+sum(loads)/k))<1e-10
                        summaries[str(b)][str(profile)]=expected;times_byprofile.append(timebooks)
                    ratio=sum(times_byprofile[1])/sum(times_byprofile[0]);ratios=[y/x for x,y in zip(*times_byprofile)];passed=.9<=ratio<=1.1 and all(.85<=v<=1.15 for v in ratios)
                    assert passed==got['profile_admission'][str(b)]['pass'];profile_admission[str(b)]=passed
                r['sources'][str(n)]={'all_fresh_outputs':all_outputs,'quality_transitivity_ALL18':True,'summaries':summaries,'profile_admission':profile_admission}
                g=summaries['1']['0'];expected={'warm_ordinary_lower95_ge50':g['ordinary']['warm_lower95']>=50,'warm_prose_lower95_ge50':g['prose']['warm_lower95']>=50,'fresh_matched_GPU_faster':g['sum_case_mean_seconds']<summaries['0']['0']['sum_case_mean_seconds'],'stretch_prose_lower95_ge100':g['prose']['warm_lower95']>=100};assert expected==raw['feasibility_gates'][str(n)]
            r['gates']['ALL3456_whole_outputs_source_donor_transitivity_and_cost_math']=True
            r['decision']='Independent arithmetic/parity/rate admission; speed result is confined to this fixed layout and declared workload. Full goal remains incomplete.'
        r['end_utc']=stamp();guard();r['resource']={'wall_seconds':time.monotonic()-start,'peak_working_set_bytes':peak,'bytes_hashed':hashed};write(RET,r);print(json.dumps({'retention':str(RET),'main_completed':r['main_completed'],'gates':r['gates'],'resource':r['resource']}),flush=True)
    except BaseException as e:
        r['failure']={'stage':stage,'exception':repr(e),'traceback':traceback.format_exc()};r['end_utc']=stamp();r['resource']={'wall_seconds':time.monotonic()-start,'peak_working_set_bytes':peak,'bytes_hashed':hashed};write(RET.with_suffix('.failure.json'),r);print(json.dumps(r['failure']),flush=True);raise
if __name__=='__main__':audit()
