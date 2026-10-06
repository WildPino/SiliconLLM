"""Independent numerical admission; imports neither inquiry nor fitting/math helpers."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
from decimal import Decimal,localcontext
import json
import math
from pathlib import Path
import struct
import traceback
from meth490_operations import ROOT,DOC,Context,write

RAW=DOC/'meth490_root_mass_result.json'
RET=DOC/'RETENTION_490_20261006.json'
OUT=ROOT/'results/native_expert_scaling/meth490_root_mass'
AUDIT_OUT=ROOT/'results/native_expert_scaling/meth490_root_mass_audit'

def audit():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);ap.add_argument('--binding-sha',required=True);a=ap.parse_args()
    assert a.out.resolve()==RET.resolve()
    ctx=Context(AUDIT_OUT,RET,300,512<<20)
    try:
        b=ctx.admit(a.binding_sha);ctx.binding=b
        sourcepath=RAW if RAW.exists() else RAW.with_suffix('.failure.json')
        raw=json.loads(sourcepath.read_bytes());ctx.head(sourcepath)
        if sourcepath!=RAW:
            eventpath=ROOT/'results/native_expert_scaling/meth490_windows_terminal.json';events=json.loads(eventpath.read_bytes())
            assert events['query_available'] and events['query_error'] is None and not events['matching_scientific_events']
            assert events['instances'][0]['pid']==raw['process_instance']['pid'] and events['instances'][0]['create_time_unix']==raw['process_instance']['create_time_unix']
            inventory=[]
            for p in sorted(OUT.iterdir()):
                if p.is_file():inventory.append({'path':str(p),'bytes':p.stat().st_size,'sha256':ctx.digest(p)})
            ctx.r['gates']['first_main_fault_partial_inventory_and_actual_windows_retained']=True
            result=ctx.finish({'main_completed':False,'raw':{'path':str(sourcepath),'bytes':sourcepath.stat().st_size,'sha256':ctx.digest(sourcepath)},
                              'main_windows_sha256':ctx.digest(eventpath),'inherited_partial_inventory':inventory,'decision':'FIRST_MAIN_FAULT_RETAINED_NO_ROOT_PROMOTION','scope':'No numerical imports or admission of incomplete fits.'})
            print(json.dumps({'retention':str(RET),'gates':result['gates'],'main_completed':False}),flush=True);return
        assert raw['binding_sha256']==a.binding_sha and all(raw['gates'].values())
        assert raw['optimizer_updates']==384 and raw['steps_per_root']==32 and raw['validation_training_examples']==0
        for item in raw['output_inventory']:ctx.exact(item)
        eventpath=ROOT/'results/native_expert_scaling/meth490_windows_terminal.json'
        events=json.loads(eventpath.read_bytes());assert events['query_available'] and events['query_error'] is None and not events['matching_scientific_events']
        instance=events['instances'][0];assert instance['pid']==raw['process_instance']['pid'] and instance['create_time_unix']==raw['process_instance']['create_time_unix']
        assert instance['start_utc']==raw['start_utc'] and instance['end_utc']==raw['end_utc']
        assert len(events['instances'])==1+len(raw['commands'])+sum(len(c['descendant_process_peaks']) for c in raw['commands'])
        assert [c['label'] for c in raw['commands']]==['compile','controls','negative_codec','predict']
        assert [c['returncode'] for c in raw['commands']]==[0,0,2,0]
        assert not (OUT/'negative_predictions.bin').exists() and (OUT/'fatal_native.log').stat().st_size==0
        ctx.r['gates']['sole_main_actual_terminal_windows_and_complete_output_inventory']=True
        ctx.r['raw']={'path':str(RAW),'bytes':RAW.stat().st_size,'sha256':ctx.digest(RAW)}
        ctx.r['main_windows_sha256']=ctx.digest(eventpath)
        controls=json.loads((OUT/'native_controls.json').read_bytes())
        assert controls==[json.loads(s) for s in (OUT/'controls.stdout').read_text().splitlines()] and len(controls)==12
        with localcontext() as c:
            c.prec=100
            for j,z in enumerate((-1000,-80,-20,-1,0,1,20,80,1000)):
                qr=1/(1+Decimal(-z).exp());ql=1/(1+Decimal(z).exp())
                expected=[struct.unpack('<f',struct.pack('<f',float(v)))[0] for v in (ql,qr)]
                assert controls[j]=={'control':j,'z':z,'left':expected[0],'right':expected[1]}
            toyx=((.2,-.4),(1.,.3),(-.7,.1));target=(0.,.3,1.);omega=(.2,.3,.5);theta=(.1,-.2,.4);oracle=Decimal(0)
            for x,y,w in zip(toyx,target,omega):
                z=Decimal(theta[2])+sum(Decimal(v)*Decimal(t) for v,t in zip(x,theta[:2]));p=1/(1+(-z).exp())
                oracle+=Decimal(w)*(-Decimal(y)*p.ln()-(1-Decimal(y))*(1-p).ln())
            m=json.loads((OUT/'math_controls.json').read_bytes())
            assert m['soft_target_BCE_gradient_max_error']<1e-9 and abs(m['loss']-float(oracle))<2e-15 and m['endpoint_targets_not_clipped'] and m['stable_pair_zero_one_and_extreme_controls']
        for k in range(2):
            lane=[0.]*8
            for i in range(768):
                av=(2**-100 if i%2 else 2**80) if k else (i%7-3)*2**-10
                xv=(2**-20 if i%2 else (-2**-80 if i%4 else 2**-80)) if k else (i%5-2)*2**-9
                lane[i%8]+=av*xv
            total=0.
            for j in range(4):total+=lane[j]+lane[j+4]
            cast=struct.unpack('<f',struct.pack('<f',total))[0]
            assert controls[9+k]=={'dot_control':k,'double_result':total,'f32_result':cast}
        assert controls[-1]['rounding']==0 and controls[-1]['CPU_affinity_mask']==1 and not controls[-1]['MXCSR']&0x8040
        assert (OUT/'negative_heads.bin').read_bytes()==b'BADMAGIC'+struct.pack('<IIQ',3076,768,12)
        assert (OUT/'negative_codec.stderr').read_bytes().replace(b'\r\n',b'\n')==b'root_mass_error:wire_magic\n'
        ctx.r['gates']['Decimal100_sigmoid_BCE_ordered_dot_FPU_and_negative_codec_controls']=True
        import numpy as np
        import threadpoolctl
        ctx.r['numerical_imports']=True
        threadpoolctl.threadpool_limits(limits=1)
        assert all(p['num_threads']==1 for p in threadpoolctl.threadpool_info())
        U,Q=238872,387036;limit=math.log1p(.01)/7
        assert raw['log_error_budget']==limit
        def stream(key,magic,size,reserved,count,dtype):
            p=Path(b['data'][key]['path']);assert p.stat().st_size==24+size*count
            with p.open('rb') as f:
                assert f.read(24)==struct.pack('<8sIIQ',magic,size,reserved,count)
                for begin in range(0,count,1024):
                    n=min(1024,count-begin);data=f.read(n*size);assert len(data)==n*size
                    yield begin,np.frombuffer(data,dtype=dtype,count=n)
                assert f.read(1)==b''
        ud=np.dtype({'names':['meta','value','x','score'],'formats':[('<u4',12),('<f8',3),('<f4',768),('<f4',128)],'offsets':[0,112,136,3208],'itemsize':3720})
        td=np.dtype({'names':['uid','maximum','winner','mass'],'formats':['<u4',('<f4',2),('<u2',2),('<f8',2)],'offsets':[0,4,12,16],'itemsize':3560})
        metadata=np.empty((U,12),'<u4')
        for begin,records in stream('unique_inputs.bin',b'M479UNI1',3720,768,U,ud):metadata[begin:begin+len(records)]=records['meta']
        def integers(key,magic,size,count,columns):
            dtype=np.dtype(('<u4',columns));pieces=[v.copy() for _,v in stream(key,magic,size,0,count,dtype)]
            return np.concatenate(pieces)
        ownership=integers('ownership.bin',b'M479OWN1',32,U,8);links=integers('query_links.bin',b'M479LNK1',52,Q,13)
        ids=np.arange(U,dtype='<u4');assert np.array_equal(metadata[:,0],ids) and np.array_equal(ownership[:,0],ids)
        assert np.array_equal(metadata[:,2],ownership[:,1]) and np.array_equal(links[:,6],metadata[links[:,12],2]) and np.array_equal(links[:,9],metadata[links[:,12],7])
        dev=(ownership[:,2]&5)!=0;val=(ownership[:,2]&2)!=0
        assert dev.sum()==159414 and val.sum()==79458 and not (dev&val).any()
        saved_mass=np.empty((U,2),'<f8');saved_max=np.empty((U,2),'<f4');saved_winner=np.empty((U,2),'<u2')
        for begin,records in stream('group_targets.bin',b'M479TGT1',3560,127,U,td):
            end=begin+len(records);assert np.array_equal(records['uid'],ids[begin:end])
            saved_mass[begin:end]=records['mass'];saved_max[begin:end]=records['maximum'];saved_winner[begin:end]=records['winner']
        trees=json.loads(Path(b['data']['membership.json']['path']).read_bytes())
        with (OUT/'heads.bin').open('rb') as f:
            assert f.read(24)==struct.pack('<8sIIQ',b'M490HED1',3076,768,12);heads=np.frombuffer(f.read(),'<f4').reshape(12,769)
        pd=np.dtype([('uid','<u4'),('z','<f4'),('left','<f4'),('right','<f4')])
        with (OUT/'predictions.bin').open('rb') as f:
            assert f.read(24)==struct.pack('<8sIIQ',b'M490PRD1',16,0,U);prediction=np.frombuffer(f.read(),pd)
        assert len(prediction)==U and np.array_equal(prediction['uid'],ids)
        source_pair=np.empty((U,2),'<f8');selected=np.empty(U,bool);ideal_z=np.empty(U,'<f8')
        maximum_gradient_difference=maximum_update_difference=maximum_normalization_difference=0.
        maximum_pair_ulp=0;updates=0;zero_mass=0
        for bank in range(12):
            ctx.quiet('audit_bank'+str(bank));ctx.phase='independent_root_targets_weights_optimizer_export_and_physical_predictions'
            bankids=np.flatnonzero(metadata[:,2]==bank);n=len(bankids);assert n==(22272 if bank<6 else 17540)
            x=np.empty((n,768),'<f4');s=np.empty((n,128),'<f4');v=np.empty((n,3),'<f8');offset=0
            for _,records in stream('unique_inputs.bin',b'M479UNI1',3720,768,U,ud):
                chosen=records[records['meta'][:,2]==bank];m=len(chosen)
                if m:
                    assert np.array_equal(chosen['meta'][:,0],bankids[offset:offset+m]);x[offset:offset+m]=chosen['x'];s[offset:offset+m]=chosen['score'];v[offset:offset+m]=chosen['value'];offset+=m
            assert offset==n and np.isfinite(x).all() and np.isfinite(s).all()
            nodes=trees[bank]['nodes'];assert trees[bank]['bank']==bank and len(nodes)==255
            groups=[nodes[nodes[0][key]]['ids'] for key in ('left','right')]
            assert groups[0]==sorted(set(groups[0])) and groups[1]==sorted(set(groups[1])) and len(groups[0])==len(groups[1])==64 and sorted(groups[0]+groups[1])==list(range(128))
            winner=np.argmax(s,axis=1);assert np.array_equal(winner,metadata[bankids,7])
            shifted=(s-s[np.arange(n),winner,None]).astype('<f4')
            terms=np.exp(shifted.astype('<f8')).astype('<f4').astype('<f8')
            total=np.zeros(n,'<f8')
            for expert in range(128):total+=terms[:,expert]
            assert total.tobytes()==v[:,2].tobytes() and (1/total).astype('<f4').tobytes()==v[:,1].astype('<f4').tobytes()
            mass=np.empty((n,2),'<f8')
            for side,g in enumerate(groups):
                mass[:,side]=0
                for expert in g:mass[:,side]+=terms[:,expert]
                at=np.argmax(s[:,g],axis=1);child=np.asarray(g,'<u2')[at]
                assert child.tobytes()==saved_winner[bankids,side].tobytes() and s[np.arange(n),child].tobytes()==saved_max[bankids,side].tobytes()
            assert mass.tobytes()==saved_mass[bankids].tobytes()
            source_pair[bankids]=mass/(mass[:,0]+mass[:,1])[:,None];zero_mass+=int((mass==0).sum())
            selected[bankids]=np.isin(winner,groups[1])
            local=np.flatnonzero(dev[bankids]);training_ids=bankids[local]
            omega_all=np.zeros(U);book_count=np.zeros(U,'<u4')
            for book in [*range(64),*range(128,192)]:
                occurrence=links[(links[:,2]==book)&(links[:,6]==bank),12]
                exposed=np.asarray(sorted(set(map(int,occurrence))),'<u4');assert len(exposed) and dev[exposed].all()
                omega_all[exposed]+=1/(128*len(exposed));book_count[exposed]+=1
            assert np.array_equal(book_count,ownership[:,6]*(ownership[:,1]==bank)) and np.all(omega_all[val]==0)
            omega=omega_all[training_ids];assert abs(float(omega.sum())-1)<1e-12
            with np.load(OUT/('bank%02d_history.npz'%bank),allow_pickle=False) as h:
                assert set(h.files)=={'before','gradient','first','second','after','loss','mu','sigma','constant','head','final_loss','training_uid','omega'}
                assert np.array_equal(h['training_uid'],training_ids) and np.array_equal(h['omega'],omega)
                data=x[local].astype('<f8');constant=np.all(data==data[0],axis=0)
                mean=np.zeros(768)
                for start in range(0,len(data),256):mean+=(data[start:start+256]*omega[start:start+256,None]).sum(axis=0,dtype='<f8')
                mean[constant]=data[0,constant];data-=h['mu']
                variance=np.zeros(768)
                for start in range(0,len(data),256):variance+=(data[start:start+256]**2*omega[start:start+256,None]).sum(axis=0,dtype='<f8')
                sigma=np.sqrt(variance);sigma[constant]=1
                normdiff=max(float(np.max(np.abs(mean-h['mu']))),float(np.max(np.abs(sigma-h['sigma']))))
                maximum_normalization_difference=max(maximum_normalization_difference,normdiff)
                assert np.allclose(mean,h['mu'],rtol=2e-12,atol=2e-12) and np.allclose(sigma,h['sigma'],rtol=2e-12,atol=2e-12)
                assert np.array_equal(constant,h['constant']) and np.all(h['sigma']>0)
                data/=h['sigma'];y=source_pair[training_ids,1]
                previous=np.zeros(769);first=np.zeros(769);second=np.zeros(769)
                for step in range(1,33):
                    before=h['before'][step-1];assert np.array_equal(before,previous)
                    z=data@before[:768]+before[768]
                    qr=np.exp(-np.logaddexp(0,-z));residual=omega*(qr-y)
                    g=np.zeros(769);loss=0.
                    for start in range(0,len(data),256):
                        end=start+256;g[:768]+=(data[start:end]*residual[start:end,None]).sum(axis=0,dtype='<f8');g[768]+=residual[start:end].sum(dtype='<f8')
                        loss+=float(np.sum(omega[start:end]*(y[start:end]*np.logaddexp(0,-z[start:end])+(1-y[start:end])*np.logaddexp(0,z[start:end]))))
                    saved=h['gradient'][step-1];gd=float(np.max(np.abs(g-saved)));maximum_gradient_difference=max(maximum_gradient_difference,gd)
                    assert np.allclose(g,saved,rtol=2e-9,atol=3e-11) and abs(loss-float(h['loss'][step-1]))<1e-10
                    first=.9*first+.1*saved;second=.999*second+.001*saved*saved
                    assert np.array_equal(first,h['first'][step-1]) and np.array_equal(second,h['second'][step-1])
                    expected=before-(.03/(1-.9**step))*first/(np.sqrt(second/(1-.999**step))+1e-8)
                    after=h['after'][step-1];udiff=float(np.max(np.abs(expected-after)));maximum_update_difference=max(maximum_update_difference,udiff)
                    assert np.allclose(expected,after,rtol=2e-13,atol=2e-14);previous=after.copy();updates+=1
                    ctx.guard()
                coef=previous[:768]/h['sigma'];correction=0.
                for j in range(768):correction+=float(coef[j]*h['mu'][j])
                exported=np.r_[coef,previous[768]-correction].astype('<f4')
                assert exported.tobytes()==h['head'].tobytes()==heads[bank].tobytes()
                z=data@previous[:768]+previous[768]
                final=float(np.sum(omega*(y*np.logaddexp(0,-z)+(1-y)*np.logaddexp(0,z))))
                assert abs(final-float(h['final_loss']))<1e-10 and abs(final-raw['models'][bank]['final_loss'])<1e-10
                del data
                full=x.astype('<f8');full-=h['mu'];full/=h['sigma'];ideal_z[bankids]=full@previous[:768]+previous[768];del full
            del terms,omega_all,book_count
            lane=[np.zeros(n,'<f8') for _ in range(8)]
            for j in range(8):
                for d in range(j,768,8):lane[j]+=x[:,d].astype('<f8')*float(heads[bank,d])
            result=np.zeros(n,'<f8')
            for j in range(4):result+=lane[j]+lane[j+4]
            expected_z=(result+float(heads[bank,768])).astype('<f4')
            assert expected_z.tobytes()==prediction['z'][bankids].tobytes()
            reference=np.column_stack((np.exp(-np.logaddexp(0,expected_z.astype('<f8'))),np.exp(-np.logaddexp(0,-expected_z.astype('<f8'))))).astype('<f4')
            actual=np.column_stack((prediction['left'][bankids],prediction['right'][bankids]))
            assert np.isfinite(actual).all() and np.all((actual>=0)&(actual<=1))
            ulp=np.abs(reference.view('<u4').astype('<i8')-actual.view('<u4').astype('<i8'));maximum_pair_ulp=max(maximum_pair_ulp,int(ulp.max()));assert np.all(ulp<=1)
            ctx.log(bank_audited=bank);del x,s,v,lane,result,mass
        assert updates==384 and zero_mass==raw['source_zero_root_child_mass_fields']
        ctx.r['gates']['ALL238872_source_roots_and_book_balanced_development_ownership']=True
        ctx.r['gates']['ALL384_gradients_moments_updates_and_raw_F32_exports']=True
        ctx.r['gates']['ALL238872_native_AVX_logits_BYTE_and_pair_within1ULP']=True
        # Rebuild every statistical view with independent aggregation fields.
        actual=np.column_stack((prediction['left'],prediction['right'])).astype('<f8')
        sums=np.abs(actual.sum(axis=1)-1);assert np.all(sums<=2**-24) and float(sums.max())==raw['maximum_pair_sum_error']
        source_selected=np.where(selected,source_pair[:,1],source_pair[:,0]);candidate=np.where(selected,actual[:,1],actual[:,0]);assert np.all(source_selected>0)
        error=np.full(U,np.inf);positive=candidate>0;error[positive]=np.abs(np.log(candidate[positive]/source_selected[positive]))
        relative=np.abs(candidate/source_selected-1)
        dd=np.dtype([('uid','<u4'),('source','<f8',(2,)),('right','<u4'),('error','<f8'),('relative','<f8'),('ideal','<f8')])
        with (OUT/'diagnostics.bin').open('rb') as f:
            assert f.read(24)==struct.pack('<8sIIQ',b'M490DIA1',48,0,U);diag=np.frombuffer(f.read(),dd)
        assert len(diag)==U and np.array_equal(diag['uid'],ids) and np.array_equal(diag['right'],selected)
        assert source_pair.tobytes()==diag['source'].tobytes() and np.allclose(ideal_z,diag['ideal'],rtol=1e-13,atol=1e-12)
        assert np.allclose(error,diag['error'],rtol=1e-11,atol=2e-14) and np.allclose(relative,diag['relative'],rtol=1e-10,atol=2e-14)
        assert np.array_equal(error>limit,diag['error']>limit)
        reports=json.loads((OUT/'complete_reports.json').read_bytes())
        assert {k:len(v) for k,v in reports.items()}=={'unique':39,'occurrence':72,'book':4608,'source_ID':9216}
        def inspect(row,at):
            assert row['count']==len(at) and row['over_log_budget']==int(np.count_nonzero(error[at]>limit)) and row['zero_candidate_selected']==int(np.count_nonzero(candidate[at]==0))
            if not len(at):assert all(row[k] is None for k in ('max_log_error','mean_log_error','max_relative_error'));return
            if np.isfinite(error[at]).all():
                assert abs(row['max_log_error']-float(error[at].max()))<1e-12 and abs(row['mean_log_error']-float(error[at].mean()))<1e-12
            else:assert row['max_log_error'] is row['mean_log_error'] is None
            assert abs(row['max_relative_error']-float(relative[at].max()))<1e-12
        for row in reports['unique']:
            m=np.ones(U,bool) if row['bank']==-1 else metadata[:,2]==row['bank'];m &= np.ones(U,bool) if row['role']=='ALL' else dev if row['role']=='development' else val;inspect(row,np.flatnonzero(m))
        groupkeys={'occurrence':(links[:,6]*3+links[:,5])*2+links[:,4],
                   'book':(links[:,2]*12+links[:,6])*2+links[:,4],
                   'source_ID':((links[:,6]*3+links[:,5])*2+links[:,4])*128+links[:,9]}
        for kind in ('occurrence','book','source_ID'):
            keys=groupkeys[kind];order=np.argsort(keys,kind='stable');sorted_keys=keys[order]
            bounds=np.r_[0,np.flatnonzero(sorted_keys[1:]!=sorted_keys[:-1])+1,Q]
            groups={int(sorted_keys[start]):links[order[start:end],12] for start,end in zip(bounds[:-1],bounds[1:])}
            total=0
            for row in reports[kind]:
                key=(row['book']*12+row['bank'])*2+row['mode'] if kind=='book' else (row['bank']*3+row['role'])*2+row['mode']
                if kind=='source_ID':key=key*128+row['source_ID']
                at=groups.get(key,np.empty(0,'<u4'));inspect(row,at);total+=len(at)
            assert total==Q
        feasibility={'EVERY_development_selected_root_log_error_within_eta_over7':bool(np.all(error[dev]<=limit)),
                     'EVERY_consumed_validation_selected_root_log_error_within_eta_over7':bool(np.all(error[val]<=limit))}
        assert feasibility==raw['feasibility'] and raw['overall']==reports['unique'][:3]
        ctx.r['gates']['ALL39_72_4608_9216_views_and_fixed_scientific_decision']=True
        ctx.r['gates']['10min_1GiB_main_5min_512MiB_audit_no_model_or_optimizer_replay']=raw['resource']['wall_seconds']<=600 and raw['resource']['parent_peak_bytes']+raw['resource']['native_peak_bytes']<=1<<30
        assert all(ctx.r['gates'].values())
        result=ctx.finish({'unique_inputs_audited':U,'occurrences_audited':Q,'optimizer_updates_audited':updates,
                          'maximum_independent_gradient_difference':maximum_gradient_difference,'maximum_update_difference':maximum_update_difference,
                          'maximum_normalization_difference':maximum_normalization_difference,'maximum_sigmoid_pair_ULP_difference':maximum_pair_ulp,
                          'feasibility':feasibility,'overall':raw['overall'],'decision':raw['decision'],
                          'scope':'Independent rederivation of saved trajectory, no optimizer refit or native/model command.'})
        print(json.dumps({'retention':str(RET),'gates':result['gates'],'feasibility':feasibility,'resource':result['resource']}),flush=True)
    except BaseException as e:ctx.fail(e);ctx.done.set();ctx.watchdog.cancel();traceback.print_exc();raise

if __name__=='__main__':audit()
