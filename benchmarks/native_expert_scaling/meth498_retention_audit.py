"""Independent all-case nearest-prior KKT, serialization, physical and report audit."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import traceback
from meth498_operations import Context,ROOT,DOC,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--binding-sha',required=True);ap.add_argument('--raw-sha',required=True);args=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth498_retention',Path(args.out).resolve())
    try:
        b=ctx.admit(args.binding_sha);ctx.binding=b;rp=DOC/'meth498_minimum_prior_fit_result.json';completed=rp.exists()
        if not completed:rp=rp.with_suffix('.failure.json')
        assert ctx.digest(rp)==args.raw_sha;raw=json.loads(rp.read_bytes())
        eventpath=ROOT/'results/native_expert_scaling/meth498_windows_terminal.json';event=json.loads(eventpath.read_bytes())
        assert event['query_available'] and event['query_error'] is None and not event['matching_scientific_events'] and event['instances'][0]['pid']==raw['process_instance']['pid']
        folder=ROOT/'results/native_expert_scaling/meth498_minimum_prior_fit'
        inventory=[{'path':str(p),'bytes':p.stat().st_size,'sha256':ctx.digest(p)} for p in sorted(folder.iterdir()) if p.is_file()]
        if not completed:
            assert raw['traceback'] and raw['partial_outputs'];ctx.r['gates']['sole_first_fault_and_ALL_partial_conversion_bytes_retained']=True
            result=ctx.finish({'main_completed':False,'raw':{'path':str(rp),'sha256':args.raw_sha},'main_windows_sha256':ctx.digest(eventpath),
                'retained_main_inventory':inventory,'decision':'FIRST_MINIMUM_PRIOR_FAULT_RETAINED_NO_CANDIDATE_ELIGIBILITY','source_FFN_calls':0,'model_calls':0,'native_calls':0,
                'scope':'First fault/partial bytes only. No complete candidate or function/quality/rate/goal promotion.'})
            print(json.dumps({'gates':result['gates'],'resource':ctx.terminal_resources['resource']}));return
        assert all(raw['gates'].values()) and raw['binding_sha256']==args.binding_sha
        lookup={Path(v['path']).name:v for v in inventory}
        for v in raw['output_inventory']:assert lookup[Path(v['path']).name]==v
        terminal=json.loads(Path(raw['terminal_resource_path']).read_bytes());assert terminal['raw_sha256']==args.raw_sha
        ctx.r['gates']['main_complete_and_ALL_current_candidate_output_and_terminal_SHA']=True
        import numpy as np
        import threadpoolctl
        threadpoolctl.threadpool_limits(1);assert all(v['num_threads']==1 for v in threadpoolctl.threadpool_info())
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.r['numerical_imports']=True
        control=json.loads((folder/'controls.json').read_bytes());exact=[[Fraction(7,6),Fraction(1,12),Fraction(5,3)],[Fraction(17,6),Fraction(11,12),Fraction(4,3)]]
        for row,target,prior in zip(exact,(Fraction(3),Fraction(6)),((Fraction(1,2),Fraction(-1,4),Fraction(1)),(Fraction(1),Fraction(0),Fraction(-1,2)))):
            assert row[0]+2*row[1]+row[2]==target
            delta=[v-p for v,p in zip(row,prior)];assert delta[0]==4*delta[1]/2==delta[2]
        assert np.allclose(control['nearest_prior_coefficients'],[[float(v) for v in row] for row in exact],rtol=0,atol=1e-14)
        assert control['I8_half_even_codes']==[[127,-127,0,2,2,0,-2,-2]] and control['I8_half_even_scale']==[1.]
        write(ctx.out/'independent_controls.json',{'nearest_prior_exact':[[str(v) for v in row] for row in exact],'I8_signed_half_even':[127,-127,0,2,2,0,-2,-2]})
        ctx.r['gates']['NEW498_independent_Fraction_equality_metric_stationarity_and_I8_control']=True
        def read(path,magic,width,reserved,count,dtype):
            with Path(path).open('rb') as f:
                assert f.read(24)==struct.pack('<8sIIQ',magic,width,reserved,count)
                a=np.fromfile(f,dtype,count=count);assert len(a)==count and not f.read(1);return a
        def data(key,*args):return read(b['data'][key]['path'],*args)
        ut=np.dtype([('m','<u4',(13,)),('hash','u1',(128,))]);it=np.dtype([('e','<u4'),('accept','<u4'),('alpha','<f4'),('x','<f4',(768,)),('q','<i2',(768,))])
        ft=np.dtype([('phi','<f4',(512,)),('q','<i2',(512,)),('alpha','<f4'),('l','<f4',(768,))]);pt=np.dtype([('id','<u4'),('p','<f4'),('logits','<f4',(24,)),('oracle','<f4',(768,)),('coupled','<f4',(768,))])
        uid=data('uid',b'M493U001',180,0,17540,ut);m=uid['m'];occ=data('occurrences',b'M493O001',68,0,19962,np.dtype(('<u4',(17,))))
        inputs=data('inputs',b'M495INP1',4620,768,17540,it);feat=data('features',b'M495FEA1',6148,512,17540,ft);target=data('targets',b'M493Y001',3072,768,17540,np.dtype(('<f4',(768,))))
        baseline=data('baseline_predictions',b'M495PRE1',6248,768,17540,pt);pred=read(folder/'predictions.bin',b'M495PRE1',6248,768,17540,pt)
        dev=(m[:,4]&5)!=0;val=(m[:,4]&2)!=0
        assert np.array_equal(m[:,0],np.arange(17540)) and np.all(m[:,2]==11) and np.all(m[:,3]<128) and np.all(m[:,6]==2) and (int(dev.sum()),int(val.sum()))==(11721,5819) and not np.any(dev&val) and np.all(dev|val)
        assert np.array_equal(inputs['e'],m[:,3]) and np.all(inputs['accept']==1)
        assert np.array_equal(occ[:,0],np.arange(19962)) and np.all(occ[:,1]<17540) and np.all(occ[:,3]==128) and np.all(occ[:,8]==11) and np.all(occ[:,12]==1)
        for oi,mi in ((14,1),(11,3),(15,12),(16,10)):assert np.array_equal(occ[:,oi],m[occ[:,1],mi])
        assert np.array_equal(np.bincount(occ[:,1],minlength=17540),m[:,7])
        for name in ('id','p','logits'):assert pred[name].tobytes()==baseline[name].tobytes()
        original=Path(b['data']['bank']['path']).read_bytes();bank=(folder/'bank.bin').read_bytes()
        assert len(bank)==len(original)==127232136 and bank[:223368]==original[:223368] and bank[223368:223368+992256]==original[223368:223368+992256]
        assert hashlib.sha256(bank[:223368]).hexdigest()==raw['fixed_prefix_sha256']
        calibration=json.loads(Path(b['data']['calibration']['path']).read_bytes());amps=np.array([v['a_bits'] for v in calibration['cells']],'<u4').view('<f4')
        with Path(b['data']['geometry']['path']).open('rb') as f:
            assert f.read(20)==struct.pack('<8sIII',b'M494GEO1',768,512,513);f.seek(20+3*768*768*8+513*513*8);k=np.fromfile(f,'<f8',513*513).reshape(513,513)
        kp=k[::-1,::-1].copy();t=np.linalg.cholesky(kp)
        def check(a,x,rhs,tol):
            difference=a@x-rhs;bound=tol*np.maximum(1,np.abs(a)@np.abs(x)+np.abs(rhs));ratio=float(np.max(np.abs(difference)/bound)) if difference.size else 0.
            assert np.isfinite(difference).all() and ratio<=1;return ratio
        factor=check(t,t.T,kp,5e-12);values=np.empty((17540,20),'<f8');seen=np.zeros(17540,bool);kkt=[]
        def parts(e):
            pos=223368+992256*e;lq=np.frombuffer(bank,'<i1',768*768,pos).reshape(768,768);ls=np.frombuffer(bank,'<f4',768,pos+589824);pos+=592896
            bq=np.frombuffer(bank,'<i1',768*512,pos).reshape(768,512);bs=np.frombuffer(bank,'<f4',768,pos+393216);bias=np.frombuffer(bank,'<f4',768,pos+396288);return lq,ls,bq,bs,bias
        def physical(q,w,s,a):
            dot=q.astype('<f8')@w.astype('<f8').T;integer=dot.astype('<i8');assert np.array_equal(integer,dot) and np.all(np.abs(integer)<=q.shape[1]*127*32767)
            return ((integer.astype('<f8')*s.astype('<f8')[None,:])*a.astype('<f8')[:,None]).astype('<f4')
        def encoder(c):
            maxima=np.max(np.abs(c),axis=1);s=(maxima.astype('<f8')/127).astype('<f4');s[maxima==0]=1
            ratio=(c.astype('<f8')/s[:,None].astype('<f8')).astype('<f4').astype('<f8');low=np.floor(ratio);rest=ratio-low
            code=np.clip(low+((rest>.5)|((rest==.5)&(low%2!=0))),-127,127).astype('<i1');return code,s
        ctx.phase='ALL127_independent_permuted_metric_KKT_128_serialization_codec_and17540_levels'
        with Path(b['data']['C0']['path']).open('rb') as sf,(folder/'coefficients_F64.bin').open('rb') as cf,(folder/'coefficients_F32.bin').open('rb') as ff:
            assert sf.read(24)==struct.pack('<8sIIQ',b'M494ABS1',1575936,513,128) and cf.read(24)==struct.pack('<8sIIQ',b'M498F641',3151872,513,128) and ff.read(24)==struct.pack('<8sIIQ',b'M498F321',1575936,513,128)
            for e in range(128):
                prior=(np.frombuffer(sf.read(1575936),'<f4').reshape(768,513).astype('<f8')*float(amps[e])).astype('<f4').astype('<f8')
                c=np.frombuffer(cf.read(3151872),'<f8').reshape(768,513);cc=np.frombuffer(ff.read(1575936),'<f4').reshape(768,513)
                assert np.isfinite(c).all() and c.astype('<f4').tobytes()==cc.tobytes();ids=np.flatnonzero(dev&(m[:,3]==e));assert calibration['cells'][e]['development']==len(ids)<=308
                assert raw['solve_cases'][e]['expert']==e and raw['solve_cases'][e]['development']==len(ids) and raw['solve_cases'][e]['retained_prior']==(not len(ids))
                if len(ids):
                    h=np.ones((len(ids),513),'<f8');h[:,:512]=feat['q'][ids].astype('<f8')*feat['alpha'][ids,None].astype('<f8');h=h[:,::-1].copy()
                    r=target[ids].astype('<f8')-feat['l'][ids].astype('<f8');feas=check(h,c[:,::-1].T,r,3e-10)
                    residual=h@c[:,::-1].T-r;rms=math.sqrt(float(np.sum(residual*residual))/float(np.sum(target[ids].astype('<f8')**2)));assert rms<=1e-7
                    zt=np.linalg.solve(t,h.T);white=check(t,zt,h.T,3e-12);q,qr=np.linalg.qr(zt,mode='reduced');qrcheck=check(q,qr,zt,5e-12)
                    orth=float(np.max(np.abs(q.T@q-np.eye(len(ids)))));assert orth<=5e-11
                    dt=t.T@(c-prior)[:,::-1].T;projection=q@(q.T@dt);bound=3e-10*np.maximum(1,np.abs(dt)+np.abs(q)@(np.abs(q.T)@np.abs(dt)))
                    stationary=float(np.max(np.abs(dt-projection)/bound));assert stationary<=1
                    objective=float(np.sum(dt*dt));assert np.isclose(objective,raw['solve_cases'][e]['prior_metric_displacement_energy'],rtol=1e-8,atol=1e-5)
                    kkt.append({'expert':e,'development':len(ids),'feasibility_ratio':feas,'development_fit_RMS':rms,'stationarity_ratio':stationary,'whitening_ratio':white,'QR_ratio':qrcheck,'orthogonality_maxabs':orth,'prior_metric_energy':objective})
                else:assert c.tobytes()==prior.tobytes();kkt.append({'expert':e,'development':0,'retained_prior':True})
                pos=223368+992256*e;assert bank[pos:pos+592896]==original[pos:pos+592896] and hashlib.sha256(bank[pos:pos+592896]).hexdigest()==raw['fixed_L_sha256'][e]
                lq,ls,bq,bs,bias=parts(e);code,scale=encoder(cc[:,:512]);assert code.tobytes()==bq.tobytes() and scale.tobytes()==bs.tobytes() and cc[:,512].tobytes()==bias.tobytes()
                assert np.all(lq!=-128) and np.all(bq!=-128) and np.all(bs>0)
                allids=np.flatnonzero(m[:,3]==e)
                for start in range(0,len(allids),127):
                    at=allids[start:start+127];left=physical(inputs['q'][at],lq,ls,inputs['alpha'][at]);assert left.tobytes()==feat['l'][at].tobytes()
                    right=physical(feat['q'][at],bq,bs,feat['alpha'][at]);output=((left.astype('<f8')+right.astype('<f8')).astype('<f4').astype('<f8')+bias.astype('<f8')[None,:]).astype('<f4')
                    assert output.tobytes()==pred['oracle'][at].tobytes()
                    alpha=feat['alpha'][at].astype('<f8');ph=feat['q'][at].astype('<f8');h=np.ones((len(at),513),'<f8');h[:,:512]=ph*alpha[:,None]
                    u64=left.astype('<f8')+h[:,::-1]@c[:,::-1].T;u32=left.astype('<f8')+h[:,::-1]@cc[:,::-1].astype('<f8').T
                    integer=(ph@bq.astype('<f8').T).astype('<i8');decoded=(left.astype('<f8')+((integer.astype('<f8')*bs.astype('<f8')[None,:])*alpha[:,None]))+bias.astype('<f8')[None,:]
                    y=target[at].astype('<f8');p=pred['oracle'][at].astype('<f8');pc=pred['coupled'][at].astype('<f8');errors=[u64-y,u32-u64,decoded-u32,p-decoded];total=p-y
                    diag=[np.einsum('ij,ij->i',v,v) for v in (y,errors[0],u32-y,total,pc-y,pc-p,errors[1],errors[2],errors[3])]
                    pairs=[np.einsum('ij,ij->i',errors[i],errors[j]) for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))]
                    residual=errors[0]+errors[1]+errors[2]+errors[3]-total;vb=3e-12*np.maximum(1,np.abs(y)+np.abs(u64)+np.abs(u32)+np.abs(decoded)+np.abs(p))
                    closure=diag[3]-(diag[1]+diag[6]+diag[7]+diag[8]+2*sum(pairs));eb=1e-10*np.maximum(1,diag[0]+diag[1]+diag[6]+diag[7]+diag[8]+2*sum(np.abs(v) for v in pairs))
                    pre=decoded-y;values[at]=np.column_stack((*diag,*pairs,np.max(np.abs(residual),axis=1),np.max(np.abs(residual)/vb,axis=1),np.abs(closure),np.abs(closure)/eb,np.einsum('ij,ij->i',pre,pre)))
                    assert np.isfinite(values[at]).all() and np.max(values[at][:,[16,18]])<=1;seen[at]=True;ctx.guard()
                if e%16==0:ctx.log(experts_audited=e+1);print(json.dumps({'experts_audited':e+1}),flush=True)
            assert not sf.read(1) and not cf.read(1) and not ff.read(1)
        assert seen.all();write(ctx.out/'independent_KKT_cases.json',{'prior_factor_ratio':factor,'cases':kkt})
        ctx.r['gates']['ALL127_permuted_metric_KKT_128_exact_serialization_codec_and_fixed_bytes']=True
        ctx.phase='ALL17540_independent_coupled_physical_output_BYTE'
        for e in range(128):
            ids=np.flatnonzero(pred['id']==e);lq,ls,bq,bs,bias=parts(e)
            for start in range(0,len(ids),127):
                at=ids[start:start+127];left=physical(inputs['q'][at],lq,ls,inputs['alpha'][at]);right=physical(feat['q'][at],bq,bs,feat['alpha'][at])
                output=((left.astype('<f8')+right.astype('<f8')).astype('<f4').astype('<f8')+bias.astype('<f8')[None,:]).astype('<f4');assert output.tobytes()==pred['coupled'][at].tobytes();ctx.guard()
        saved=read(folder/'energy_by_uid.bin',b'M498ENG1',160,20,17540,np.dtype(('<f8',(20,))))
        columns=list(range(15))+[19];assert np.allclose(values[:,columns],saved[:,columns],rtol=1e-9,atol=1e-5) and np.max(saved[:,[16,18]])<=1
        ctx.r['gates']['ALL17540_oracle_coupled_BYTE_20_energy_columns_cross_terms_and_closure']=True
        def metric(ix):
            n=len(ix);s=np.sum(values[ix,:15],axis=0) if n else np.zeros(15);correct=int(np.sum(pred['id'][ix]==m[ix,3]));v={'count':n,'ID_correct':correct,'ID_fidelity':correct/n if n else None,'source_energy':float(s[0])}
            for j,name in enumerate(('fit64','fit32','oracle','coupled','decision','serialization','parameters','arithmetic'),1):v[name+'_error_energy']=float(s[j]);v[name+'_RMS']=math.sqrt(float(s[j]/s[0])) if s[0] else None
            for j,name in enumerate(('fit_serial','fit_parameter','fit_arithmetic','serial_parameter','serial_arithmetic','parameter_arithmetic'),9):v[name+'_inner']=float(s[j])
            for j,name in enumerate(('max_identity_abs','max_identity_envelope_ratio','max_energy_closure_abs','max_energy_closure_envelope_ratio'),15):v[name]=float(np.max(values[ix,j])) if n else 0.
            v['pre_arithmetic_error_energy']=float(np.sum(values[ix,19])) if n else 0.;return v
        counts=np.bincount(m[dev,3],minlength=128);reports={v:[] for v in ('uid_roles','cells','rare','views','role_mode','exposures')}
        for split,mask in (('development',dev),('consumed_validation',val)):reports['uid_roles'].append({'split':split,**metric(np.flatnonzero(mask))})
        for e in range(128):
            for split,mask in (('development',dev),('consumed_validation',val)):reports['cells'].append({'expert':e,'development_exposure':int(counts[e]),'split':split,**metric(np.flatnonzero(mask&(m[:,3]==e)))})
        for name,lo,hi in (('0',0,0),('1..4',1,4),('5..15',5,15),('>=16',16,17540)):
            category=(counts>=lo)&(counts<=hi)
            for split,mask in (('development',dev),('consumed_validation',val)):reports['rare'].append({'development_class':name,'split':split,**metric(np.flatnonzero(mask&category[m[:,3]]))})
        for book in range(192):
            for mode in range(2):
                for accepted in range(2):reports['views'].append({'book':book,'role':book//64,'mode':mode,'accepted':accepted,**metric(occ[(occ[:,4]==book)&(occ[:,6]==mode)&(occ[:,12]==accepted),1])})
        for role in range(3):
            for mode in range(2):reports['role_mode'].append({'role':role,'mode':mode,**metric(occ[(occ[:,7]==role)&(occ[:,6]==mode),1])})
            ix=occ[occ[:,7]==role,1];source=np.bincount(m[ix,3],minlength=128);candidate=np.bincount(pred['id'][ix],minlength=128)
            reports['exposures'].extend({'role':role,'expert':e,'source':int(source[e]),'candidate':int(candidate[e])} for e in range(128))
        residualnames={'max_identity_abs','max_identity_envelope_ratio','max_energy_closure_abs','max_energy_closure_envelope_ratio'}
        for label in ('uid_roles','cells','rare','views','role_mode'):
            assert len(reports[label])==len(raw['reports'][label])
            for current,old in zip(reports[label],raw['reports'][label]):
                assert current.keys()==old.keys()
                for name,v in current.items():
                    if name in residualnames:
                        assert v>=0 and old[name]>=0
                        if name.endswith('ratio'):assert v<=1 and old[name]<=1
                    elif isinstance(v,float):assert np.isclose(v,old[name],rtol=1e-9,atol=1e-5),(label,name,v,old[name])
                    else:assert v==old[name]
        assert reports['exposures']==raw['reports']['exposures'] and sum(len(reports[k]) for k in ('uid_roles','cells','rare','views','role_mode'))==1040
        rm=reports['role_mode'];outcomes={name+'_ALL_six_RMS_1pct':all(v['count'] and v[name+'_RMS'] is not None and v[name+'_RMS']<=.01 for v in rm) for name in ('fit64','fit32','oracle','coupled')}
        outcomes['ID_ALL_six_999permille']=all(v['count'] and v['ID_fidelity']>=.999 for v in rm);assert outcomes==raw['reports']['outcomes']
        decision='NEXT_SOURCE_PRIOR_TRANSPORT' if not outcomes['fit64_ALL_six_RMS_1pct'] else 'NEXT_COEFFICIENT_SERIALIZATION' if not outcomes['fit32_ALL_six_RMS_1pct'] else 'NEXT_PHYSICAL_COEFFICIENT_ENCODING' if not outcomes['oracle_ALL_six_RMS_1pct'] else 'NEXT_UNRESOLVED_ROUTING_AND_MASS'
        assert decision==raw['decision'] and len(raw['commands'])==1 and raw['commands'][0]['returncode']==0
        assert raw['readout_solves']==127 and raw['coefficient_cases']==128 and raw['optimizer_updates']==raw['source_FFN_calls']==raw['model_calls']==0 and raw['native_calls']==1
        assert raw['physical_bank_bytes']==127232136 and raw['same_logical_weight_bytes_per_12_bank_token']==14587008
        ctx.r['gates']['ALL1040_reports_384_exposures_frozen_outcomes_decision_and_cost']=True
        result=ctx.finish({'main_completed':True,'raw':{'path':str(rp),'sha256':args.raw_sha},'main_windows_sha256':ctx.digest(eventpath),'retained_main_inventory':inventory,
            'UIDs_audited':17540,'occurrences_audited':19962,'experts_audited':128,'readout_solves_audited':127,'metric_groups_audited':1040,'exposure_groups_audited':384,
            'outcomes':outcomes,'decision':decision,'max_identity_envelope_ratio':float(values[:,16].max()),'max_energy_closure_envelope_ratio':float(values[:,18].max()),
            'source_FFN_calls':0,'model_calls':0,'native_calls':0,'scope':'Independent whole one-bank changed-readout candidate audit only; no fresh quality, SAME rate/DRAM/useful n/family or goal promotion.'})
        print(json.dumps({'gates':result['gates'],'outcomes':outcomes,'decision':decision,'resource':ctx.terminal_resources['resource']}),flush=True)
    except BaseException as exc:
        ctx.fail(exc);print(json.dumps({'error':str(exc),'traceback':traceback.format_exc()}),flush=True);raise

if __name__=='__main__':main()
