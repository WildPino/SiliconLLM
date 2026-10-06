"""Independent whole-domain physical, fit, optimizer and report audit."""
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
from meth495_operations import Context,ROOT,DOC,write

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--binding-sha',required=True); ap.add_argument('--raw-sha',required=True); args=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth495_retention',Path(args.out).resolve())
    try:
        b=ctx.admit(args.binding_sha); ctx.binding=b; rawpath=DOC/'meth495_weighted_hybrid_fit_result.json'; completed=rawpath.exists()
        if not completed: rawpath=rawpath.with_suffix('.failure.json')
        assert ctx.digest(rawpath)==args.raw_sha; raw=json.loads(rawpath.read_bytes())
        eventpath=ROOT/'results/native_expert_scaling/meth495_windows_terminal.json'; event=json.loads(eventpath.read_bytes())
        assert event['query_available'] and event['query_error'] is None and not event['matching_scientific_events']
        assert event['instances'][0]['pid']==raw['process_instance']['pid']
        folder=ROOT/'results/native_expert_scaling/meth495_weighted_hybrid_fit'
        inventory=[{'path':str(p),'bytes':p.stat().st_size,'sha256':ctx.digest(p)} for p in sorted(folder.iterdir()) if p.is_file()]
        if not completed:
            assert raw['traceback'] and raw['partial_outputs']; ctx.r['gates']['sole_first_main_fault_and_ALL_partial_bytes_retained']=True
            result=ctx.finish({'main_completed':False,'raw':{'path':str(rawpath),'sha256':args.raw_sha},'main_windows_sha256':ctx.digest(eventpath),
                'retained_main_inventory':inventory,'decision':'FIRST_HYBRID_FIT_FAULT_RETAINED_NO_RECIPE_ELIGIBILITY',
                'source_FFN_calls':0,'model_calls':0,'scope':'First fault only, no complete candidate or quality/rate/goal promotion.'})
            print(json.dumps({'gates':result['gates'],'resource':ctx.terminal_resources['resource']})); return
        assert all(raw['gates'].values()) and raw['binding_sha256']==args.binding_sha
        lookup={Path(v['path']).name:v for v in inventory}
        for v in raw['output_inventory']: assert lookup[Path(v['path']).name]==v
        terminal=json.loads(Path(raw['terminal_resource_path']).read_bytes()); assert terminal['raw_sha256']==args.raw_sha
        ctx.r['gates']['main_complete_and_ALL_current_output_and_terminal_SHA']=True
        import numpy as np
        import threadpoolctl
        threadpoolctl.threadpool_limits(1); assert all(v['num_threads']==1 for v in threadpoolctl.threadpool_info())
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore'); ctx.r['numerical_imports']=True
        def read(path,magic,width,reserved,count,dtype):
            with Path(path).open('rb') as f:
                assert f.read(24)==struct.pack('<8sIIQ',magic,width,reserved,count)
                a=np.fromfile(f,dtype,count=count); assert len(a)==count and not f.read(1); return a
        def encoder(x,limit):
            maxima=np.max(np.abs(x),axis=1); scale=(maxima.astype('<f8')/limit).astype('<f4'); scale[maxima==0]=1
            ratio=(x.astype('<f8')/scale[:,None].astype('<f8')).astype('<f4').astype('<f8')
            low=np.floor(ratio); rest=ratio-low; integer=low+((rest>.5)|((rest==.5)&(low%2!=0)))
            return np.clip(integer,-limit,limit).astype('<i1' if limit==127 else '<i2'),scale
        native=json.loads((folder/'native_controls.json').read_bytes()); expected=[]
        for code in range(81):
            trits=[(code//(3**j))%3-1 for j in range(4)]; expected.append(sum(a*x for a,x in zip(trits,(32767,-17213,43,-8191))))
        assert native[0]=={'table81':expected} and native[1]=={'alpha_bits':0x40000000,'codes':[32767,-32767,0,2,2,0,-2,-2]}
        assert native[2]['L_I64']==768*126*32766>2**31-1 and native[2]['B_I32']==512*126*32766<2**31
        assert native[2]['RNE']==0 and native[2]['CPU0']==1 and not native[2]['MXCSR']&0x8040
        control=json.loads((folder/'math_controls.json').read_bytes()); exact=[Fraction(110,57),Fraction(34,57)]
        assert np.allclose(control['ridge_coefficient'][0],[float(v) for v in exact],rtol=0,atol=1e-14)
        expected_step=[.01*.5/(.5+1e-8),-.01*.25/(.25+1e-8)]
        assert np.allclose(control['dyadic_Adam_first_step'][0],expected_step,rtol=0,atol=1e-14)
        assert not (folder/'negative_features.bin').exists()
        assert (folder/'negative_magic.stderr').read_bytes().replace(b'\r\n',b'\n')==b'hybrid495_error:wire_magic\n'
        assert [v['returncode'] for v in raw['commands']]==[0,0,2,0,0]
        write(ctx.out/'independent_controls.json',{'ridge_exact':[str(v) for v in exact],'Adam_dyadic_expected':expected_step,'table81_exact':expected})
        ctx.r['gates']['NEW495_exact_control_expectations_and_actual_native_commands']=True
        it=np.dtype([('e','<u4'),('accept','<u4'),('alpha','<f4'),('x','<f4',(768,)),('q','<i2',(768,))])
        ft=np.dtype([('phi','<f4',(512,)),('q','<i2',(512,)),('alpha','<f4'),('l','<f4',(768,))])
        pt=np.dtype([('id','<u4'),('p','<f4'),('logits','<f4',(24,)),('oracle','<f4',(768,)),('coupled','<f4',(768,))])
        ut=np.dtype([('m','<u4',(13,)),('hash','u1',(128,))]); pi=b['fit_inputs']
        uid=read(pi['uid']['path'],b'M493U001',180,0,17540,ut); m=uid['m']
        occ=read(pi['occurrences']['path'],b'M493O001',68,0,19962,np.dtype(('<u4',(17,))))
        inputs=read(folder/'inputs.bin',b'M495INP1',4620,768,17540,it)
        feat=read(folder/'features.bin',b'M495FEA1',6148,512,17540,ft)
        pred=read(folder/'predictions.bin',b'M495PRE1',6248,768,17540,pt)
        y=read(pi['targets']['path'],b'M493Y001',3072,768,17540,np.dtype(('<f4',(768,))))
        dev=(m[:,4]&5)!=0; val=(m[:,4]&2)!=0
        assert np.array_equal(m[:,0],np.arange(17540)) and np.all(m[:,2]==11) and (int(dev.sum()),int(val.sum()))==(11721,5819) and not np.any(dev&val)
        assert np.array_equal(inputs['e'],m[:,3]) and np.all(inputs['accept']==1) and np.array_equal(occ[:,0],np.arange(19962))
        assert np.all(m[:,6]==2) and np.all(occ[:,3]==128) and np.all(occ[:,8]==11) and np.all(occ[:,12]==1)
        assert np.array_equal(occ[:,14],m[occ[:,1],1]) and np.array_equal(occ[:,11],m[occ[:,1],3])
        assert np.array_equal(occ[:,15],m[occ[:,1],12]) and np.array_equal(occ[:,16],m[occ[:,1],10])
        assert np.array_equal(np.bincount(occ[:,1],minlength=17540),m[:,7])
        ctx.phase='ALL17540_independent_original_source_input_and_A16_BYTE_joins'
        with Path(pi['query']['path']).open('rb') as f:
            for j in range(17540):
                f.seek(448+4732*int(m[j,11])); q=f.read(4732); assert len(q)==4732
                assert struct.unpack_from('<H',q,14)[0]==int(m[j,3]) and q[13]==1 and q[24:28]==m[j,10].tobytes()
                assert inputs['x'][j].tobytes()==q[28:3100] and inputs['q'][j].tobytes()==q[3100:4636] and inputs['alpha'][j].tobytes()==q[20:24]==m[j,12].tobytes()
                assert hashlib.sha256(q[28:3100]).digest()==uid[j]['hash'][:32].tobytes()==q[4636:4668]
                assert hashlib.sha256(q[3100:4636]).digest()==uid[j]['hash'][32:64].tobytes()==q[4668:4700]
                assert hashlib.sha256(q[3100:4636]+q[20:24]).digest()==uid[j]['hash'][64:96].tobytes()==q[4700:4732]
                if j%512==0:ctx.guard()
        codes,scale=encoder(inputs['x'],32767); assert codes.tobytes()==inputs['q'].tobytes() and scale.tobytes()==inputs['alpha'].tobytes(); del codes,scale
        prior_bank=Path(pi['bank']['path']).read_bytes(); bank=(folder/'bank.bin').read_bytes()
        assert len(bank)==len(prior_bank)==127232136 and prior_bank[:8]==b'M494BNK1' and bank[:8]==b'M495BNK1'
        assert bank[8:100392]==prior_bank[8:100392]
        packed=np.frombuffer(bank,offset=40,count=98304,dtype='u1').reshape(512,192)
        t=np.stack([((packed.astype('<i2')//(3**k))%3-1) for k in range(4)],axis=2).reshape(512,768)
        sigma=np.frombuffer(bank,offset=98344,count=512,dtype='<f4')
        for start in range(0,17540,127):
            stop=min(start+127,17540); dot=inputs['q'][start:stop].astype('<f8')@t.astype('<f8').T
            phi=np.abs((dot*sigma.astype('<f8')[None,:]*inputs['alpha'][start:stop,None].astype('<f8')).astype('<f4'))
            q,a=encoder(phi,32767)
            assert phi.tobytes()==feat['phi'][start:stop].tobytes() and q.tobytes()==feat['q'][start:stop].tobytes() and a.tobytes()==feat['alpha'][start:stop].tobytes(); ctx.guard()
        def parts(blob,e):
            at=223368+992256*e
            lq=np.frombuffer(blob,dtype='<i1',offset=at,count=768*768).reshape(768,768); at+=768*768
            ls=np.frombuffer(blob,dtype='<f4',offset=at,count=768); at+=3072
            bq=np.frombuffer(blob,dtype='<i1',offset=at,count=768*512).reshape(768,512); at+=768*512
            bs=np.frombuffer(blob,dtype='<f4',offset=at,count=768); at+=3072
            bias=np.frombuffer(blob,dtype='<f4',offset=at,count=768); return lq,ls,bq,bs,bias
        def projection(q,w,sc,a):
            integer=np.matmul(q.astype('<f8'),w.astype('<f8').T)
            return np.multiply(np.multiply(integer,sc.astype('<f8')[None,:]),a.astype('<f8')[:,None]).astype('<f4')
        for e in range(128):
            at=223368+992256*e; assert bank[at:at+592896]==prior_bank[at:at+592896]
            assert hashlib.sha256(bank[at:at+592896]).hexdigest()==raw['fixed_L_block_sha256'][e]
            ids=np.flatnonzero(inputs['e']==e)
            if len(ids):
                lq,ls,*_=parts(bank,e); expected=projection(inputs['q'][ids],lq,ls,inputs['alpha'][ids])
                assert expected.tobytes()==feat['l'][ids].tobytes()
        ctx.r['gates']['ALL_original_source_UID_occurrence_LUT_feature_and_fixed_L_BYTES']=True
        with Path(pi['geometry']['path']).open('rb') as f:
            assert f.read(20)==struct.pack('<8sIII',b'M494GEO1',768,512,513)
            f.seek(20+3*768*768*8+513*513*8); kr=np.fromfile(f,'<f8',513*513).reshape(513,513)
        calibration=json.loads(Path(pi['calibration']['path']).read_bytes()); amps=np.array([v['a_bits'] for v in calibration['cells']],'<u4').view('<f4')
        ctx.phase='ALL128_independent_fit_equations_F32_envelopes_and_B_bias_BYTES'
        fits=[]; counts=np.bincount(m[dev,3],minlength=128)
        with (folder/'fitted_coefficients.bin').open('rb') as cf,Path(pi['C0']['path']).open('rb') as pf:
            assert cf.read(24)==struct.pack('<8sIIQ',b'M495FIT1',1575936,513,128)
            assert pf.read(24)==struct.pack('<8sIIQ',b'M494ABS1',1575936,513,128)
            for e in range(128):
                c=np.frombuffer(cf.read(1575936),'<f4').reshape(768,513)
                original=np.frombuffer(pf.read(1575936),'<f4').reshape(768,513)
                prior=(original.astype('<f8')*float(amps[e])).astype('<f4'); ids=np.flatnonzero(dev&(m[:,3]==e)); n=len(ids)
                assert n==calibration['cells'][e]['development']==raw['function_fits'][e]['development']
                if n:
                    hh=np.zeros((513,513),'<f8'); rh=np.zeros((768,513),'<f8')
                    for start in range(0,n,127):
                        at=ids[start:start+127]; h=np.ones((len(at),513),'<f8'); h[:,:512]=feat['q'][at].astype('<f8')*feat['alpha'][at,None].astype('<f8')
                        r=y[at].astype('<f8')-feat['l'][at].astype('<f8'); hh+=h.T@h; rh+=r.T@h
                    gamma=hh/n+.01*kr; rhs=rh/n+.01*prior.astype('<f8')@kr
                    residual=rhs-c.astype('<f8')@gamma
                    cast=(np.abs(c.astype('<f8'))*(2**-24/(1-2**-24))+2**-150)@np.abs(gamma)
                    arithmetic=1e-7*np.maximum(1,np.abs(rhs)+np.abs(c.astype('<f8'))@np.abs(gamma))
                    ratio=float(np.max(np.abs(residual)/(cast+arithmetic))); assert ratio<=1
                    q,sc=encoder(c[:,:512],127); _,_,bq,bs,bias=parts(bank,e)
                    assert q.tobytes()==bq.tobytes() and sc.tobytes()==bs.tobytes() and c[:,512].tobytes()==bias.tobytes()
                    fits.append({'expert':e,'development':n,'equation_envelope_ratio':ratio})
                else:
                    assert c.tobytes()==prior.tobytes(); pos=223368+e*992256; assert bank[pos:pos+992256]==prior_bank[pos:pos+992256]
                    fits.append({'expert':e,'development':0,'equation_envelope_ratio':None,'complete_prior_BYTES_retained':True})
                ctx.guard()
            assert not cf.read(1) and not pf.read(1)
        ctx.r['gates']['ALL128_prior_regularized_fit_equations_and_physical_B_bias_BYTES']=True
        rms=read(folder/'key_geometry.bin',b'M495KEY1',10248,1281,1,np.dtype(('<f8',(1281,))))[0]
        x=np.column_stack((inputs['x'],feat['phi'])).astype('<f4'); rms_sum=np.zeros(1280,'<f8'); ids=np.flatnonzero(dev)
        for start in range(0,len(ids),137):
            z=x[ids[start:start+137]].astype('<f8'); rms_sum+=np.sum(z*z,axis=0)
        expected_rms=np.sqrt(rms_sum/11721); expected_rms[expected_rms==0]=1
        assert rms[-1]==1 and np.allclose(rms[:1280],expected_rms,rtol=3e-12,atol=3e-12)
        z=np.ones((11721,1281),'<f8'); z[:,:1280]=x[dev].astype('<f8')/rms[:1280]
        sourcekeys=np.frombuffer(prior_bank,offset=100392,count=24*1281,dtype='<f4').reshape(24,1281)
        theta0=sourcekeys.astype('<f8')*rms; labels=m[dev,3]; keyfit=json.loads((folder/'key_fit.json').read_bytes())
        def gradient(theta):
            grad=np.zeros((24,1281),'<f8'); total=0.
            for start in range(0,11721,137):
                xx=z[start:start+137]; target=labels[start:start+len(xx)]; scores=xx@theta.T
                for lo,hi,label in ((0,8,target//16),(8,24,target%16)):
                    s=scores[:,lo:hi]; maximum=np.max(s,axis=1); p=np.exp(s-maximum[:,None]); den=np.sum(p,axis=1)
                    total+=float(np.sum(np.log(den)+maximum-s[np.arange(len(xx)),label]))
                    p/=den[:,None]; p[np.arange(len(xx)),label]-=1; grad[lo:hi]+=p.T@xx
            difference=theta-theta0; return total/11721+.00005*float(np.sum(difference*difference)),grad/11721+.0001*difference
        def near(a,b): assert np.all(np.abs(a-b)<=2e-10+2e-10*np.abs(b))
        ctx.phase='ALL128_independent_block137_Adam_transition_audits'
        transitions=[]
        with (folder/'optimizer_history.bin').open('rb') as f:
            assert f.read(24)==struct.pack('<8sIIQ',b'M495OPT1',737856,1281,129)
            def state():
                data=f.read(737856); assert len(data)==737856; return np.frombuffer(data,'<f8').reshape(3,24,1281).copy()
            previous=state(); assert previous[0].tobytes()==theta0.tobytes() and np.count_nonzero(previous[1:])==0
            for step in range(1,129):
                loss,g=gradient(previous[0]); actual=state(); mom=.9*previous[1]+.1*g; var=.999*previous[2]+.001*g*g
                theta=previous[0]-.01*(mom/(1-.9**step))/(np.sqrt(var/(1-.999**step))+1e-8)
                assert np.isfinite(actual).all() and np.all(actual[2]>=0)
                near(actual[0],theta); near(actual[1],mom); near(actual[2],var)
                record=keyfit['trace'][step-1]; assert record['step']==step
                assert math.isclose(record['pre_update_loss'],loss,rel_tol=1e-10,abs_tol=1e-8)
                assert math.isclose(record['pre_update_gradient_norm'],float(np.linalg.norm(g)),rel_tol=1e-10,abs_tol=1e-8)
                transitions.append({'step':step,'max_theta_transition_abs':float(np.max(np.abs(actual[0]-theta)))})
                previous=actual
                if step%16==0: ctx.log(Adam_transitions_audited=step); print(json.dumps({'Adam_transitions_audited':step}),flush=True)
                ctx.guard()
            assert not f.read(1)
        loss,g=gradient(previous[0]); assert math.isclose(loss,keyfit['final_loss'],rel_tol=1e-10,abs_tol=1e-8)
        assert math.isclose(float(np.linalg.norm(g)),keyfit['final_gradient_norm'],rel_tol=1e-10,abs_tol=1e-8)
        gap=float(np.sum(g*g)/.0002); assert math.isclose(gap,keyfit['computed_real_formula_gradient_gap'],rel_tol=1e-10,abs_tol=1e-8)
        keys=np.frombuffer(bank,offset=100392,count=24*1281,dtype='<f4').reshape(24,1281)
        assert (previous[0]/rms).astype('<f4').tobytes()==keys.tobytes(); del z,previous,theta0,g,sourcekeys
        ctx.r['gates']['ALL128_Adam_transitions_loss_RMS_normalization_and_final_key_BYTES']=True
        ctx.phase='ALL17540_independent_oracle_coupled_ID_logit_probability_BYTES'
        logits=np.empty((17540,24),'<f4')
        for start in range(0,17540,127):
            stop=min(start+127,17540); lanes=np.zeros((stop-start,24,8),'<f8')
            for j in range(1280): lanes[:,:,j%8]=lanes[:,:,j%8]+x[start:stop,j,None].astype('<f8')*keys[None,:,j].astype('<f8')
            total=np.zeros((stop-start,24),'<f8')
            for j in range(4): total=total+(lanes[:,:,j]+lanes[:,:,j+4])
            logits[start:stop]=(total+keys[None,:,1280].astype('<f8')).astype('<f4'); ctx.guard()
        picked=np.argmax(logits[:,:8],axis=1)*16+np.argmax(logits[:,8:],axis=1)
        assert logits.tobytes()==pred['logits'].tobytes() and picked.astype('<u4').tobytes()==pred['id'].tobytes()
        for k,row in enumerate(logits):
            probabilities=[]
            for lo,hi in ((0,8),(8,24)):
                maximum=float(max(row[lo:hi])); values=[np.float32(math.exp(float(v)-maximum)) for v in row[lo:hi]]
                probabilities.append(np.float32(1./sum(float(v) for v in values)))
            expected=np.float32(float(probabilities[0])*float(probabilities[1])); assert expected.tobytes()==pred['p'][k].tobytes()
        for field,chosen in (('oracle',m[:,3]),('coupled',picked)):
            for e in range(128):
                at=np.flatnonzero(chosen==e)
                if not len(at):continue
                lq,ls,bq,bs,bias=parts(bank,e)
                for start in range(0,len(at),127):
                    ids=at[start:start+127]; a=projection(inputs['q'][ids],lq,ls,inputs['alpha'][ids]); bb=projection(feat['q'][ids],bq,bs,feat['alpha'][ids])
                    expected=np.add(np.add(a,bb,dtype=np.float32),bias,dtype=np.float32)
                    assert expected.tobytes()==pred[field][ids].tobytes(); ctx.guard()
        ctx.r['gates']['ALL17540_independent_native_prediction_BYTES']=True
        energy=np.empty((17540,4),'<f8')
        for start in range(0,17540,127):
            stop=min(start+127,17540); yy=y[start:stop].astype('<f8'); a=pred['oracle'][start:stop].astype('<f8'); c=pred['coupled'][start:stop].astype('<f8')
            for col,values in enumerate((yy,a-yy,c-yy,c-a)):energy[start:stop,col]=np.einsum('ij,ij->i',values,values)
        def metric(indices):
            n=len(indices); sums=[math.fsum(float(v) for v in energy[indices,j]) for j in range(4)]; correct=sum(int(picked[k]==m[k,3]) for k in indices)
            result={'count':n,'ID_correct':correct,'ID_fidelity':correct/n if n else None,'source_energy':sums[0],
                'oracle_error_energy':sums[1],'coupled_error_energy':sums[2],'decision_error_energy':sums[3]}
            for j,name in enumerate(('oracle','coupled','decision'),1):result[name+'_RMS']=math.sqrt(sums[j]/sums[0]) if sums[0] else None
            return result
        report=raw['reports']; group_count=0
        def compare(record,indices):
            nonlocal group_count
            for key,value in metric(indices).items():
                actual=record[key]
                if value is None or isinstance(value,int):assert actual==value
                else:assert math.isclose(value,actual,rel_tol=1e-10,abs_tol=1e-8),(key,value,actual)
            group_count+=1
        assert [v['split'] for v in report['uid_roles']]==['development','consumed_validation']
        for record in report['uid_roles']:compare(record,np.flatnonzero(dev if record['split']=='development' else val))
        assert {(v['expert'],v['split']) for v in report['cells']}=={(e,s) for e in range(128) for s in ('development','consumed_validation')} and len(report['cells'])==256
        for record in report['cells']:
            e=record['expert']; assert record['development_exposure']==counts[e]
            compare(record,np.flatnonzero((dev if record['split']=='development' else val)&(m[:,3]==e)))
        categories={'0':counts==0,'1..4':(counts>=1)&(counts<=4),'5..15':(counts>=5)&(counts<=15),'>=16':counts>=16}
        assert len(report['rare'])==8 and {(v['split'],v['development_class']) for v in report['rare']}=={(s,c) for s in ('development','consumed_validation') for c in categories}
        for record in report['rare']:compare(record,np.flatnonzero((dev if record['split']=='development' else val)&categories[record['development_class']][m[:,3]]))
        assert len(report['views'])==768 and {(v['book'],v['mode'],v['accepted']) for v in report['views']}=={(bk,mo,a) for bk in range(192) for mo in range(2) for a in range(2)}
        for record in report['views']:
            bk=record['book']; assert record['role']==(0 if bk<64 else 1 if bk<128 else 2)
            compare(record,occ[(occ[:,4]==bk)&(occ[:,6]==record['mode'])&(occ[:,12]==record['accepted']),1])
        assert len(report['role_mode'])==6 and {(v['role'],v['mode']) for v in report['role_mode']}=={(r,mo) for r in range(3) for mo in range(2)}
        for record in report['role_mode']:compare(record,occ[(occ[:,7]==record['role'])&(occ[:,6]==record['mode']),1])
        assert len(report['exposures'])==384 and {(v['role'],v['expert']) for v in report['exposures']}=={(r,e) for r in range(3) for e in range(128)}
        for record in report['exposures']:
            rows=occ[occ[:,7]==record['role'],1]; e=record['expert']
            assert record['source']==np.count_nonzero(m[rows,3]==e) and record['candidate']==np.count_nonzero(picked[rows]==e)
        recipe={'coupled_weighted_RMS_all_six_role_modes':all(v['count'] and v['coupled_RMS'] is not None and v['coupled_RMS']<=.01 for v in report['role_mode']),
            'ID_fidelity_all_six_role_modes':all(v['count'] and v['ID_fidelity']>=.999 for v in report['role_mode'])}
        assert recipe==report['recipe_gates'] and group_count==1040 and sum(v['count'] for v in report['views'])==19962
        hashes=[hashlib.sha256(bank[223368+e*992256:223368+(e+1)*992256]).hexdigest() for e in range(128)]
        assert hashes==raw['private_block_sha256'] and len(set(hashes))==raw['unique_physical_private_blocks']
        ctx.r['gates']['ALL1040_metrics_384_exposures_recipe_gates_and_private_hashes_recounted']=True
        passed=all(recipe.values()); result=ctx.finish({'main_completed':True,'raw':{'path':str(rawpath),'sha256':args.raw_sha},
            'main_windows_sha256':ctx.digest(eventpath),'retained_main_inventory':inventory,'function_fit_audits':fits,'Adam_transitions':transitions,
            'UIDs_audited':17540,'occurrences_audited':19962,'experts_audited':128,'metric_groups_audited':1040,'exposure_groups_audited':384,
            'recipe_gates':recipe,'decision':'FIXED_HYBRID_LOCAL_RECIPE_PASS_INDEPENDENTLY_ADMITTED' if passed else 'FIXED_HYBRID_LOCAL_RECIPE_FAIL_INDEPENDENTLY_ADMITTED',
            'source_FFN_calls':0,'model_calls':0,'optimizer_updates':0,'scope':'Complete consumed-domain local candidate only; no composed/model/fresh quality, SAME rate, useful n/DRAM or goal promotion.'})
        print(json.dumps({'gates':result['gates'],'recipe_gates':recipe,'resource':ctx.terminal_resources['resource']}),flush=True)
    except BaseException as exc:
        ctx.fail(exc); print(json.dumps({'error':str(exc),'traceback':traceback.format_exc()}),flush=True); raise

if __name__=='__main__':main()
