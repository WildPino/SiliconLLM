"""Independent complete fixed-level decomposition, controls and report audit."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import struct
import traceback
from meth496_operations import Context,ROOT,DOC,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--binding-sha',required=True);ap.add_argument('--raw-sha',required=True);args=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth496_retention',Path(args.out).resolve())
    try:
        b=ctx.admit(args.binding_sha);ctx.binding=b;rp=DOC/'meth496_readout_error_result.json';completed=rp.exists()
        if not completed:rp=rp.with_suffix('.failure.json')
        assert ctx.digest(rp)==args.raw_sha;raw=json.loads(rp.read_bytes())
        eventpath=ROOT/'results/native_expert_scaling/meth496_windows_terminal.json';event=json.loads(eventpath.read_bytes())
        assert event['query_available'] and event['query_error'] is None and not event['matching_scientific_events']
        assert event['instances'][0]['pid']==raw['process_instance']['pid']
        folder=ROOT/'results/native_expert_scaling/meth496_readout_error'
        inventory=[{'path':str(p),'bytes':p.stat().st_size,'sha256':ctx.digest(p)} for p in sorted(folder.iterdir()) if p.is_file()]
        if not completed:
            assert raw['traceback'] and raw['partial_outputs'];ctx.r['gates']['sole_first_fault_and_ALL_partial_bytes_retained']=True
            result=ctx.finish({'main_completed':False,'raw':{'path':str(rp),'sha256':args.raw_sha},'main_windows_sha256':ctx.digest(eventpath),
                'retained_main_inventory':inventory,'decision':'FIRST_DECOMPOSITION_FAULT_RETAINED_NO_DIAGNOSTIC_ELIGIBILITY',
                'source_FFN_calls':0,'model_calls':0,'native_calls':0,'scope':'First fault retained; no numerical conclusion or goal promotion.'})
            print(json.dumps({'gates':result['gates'],'resource':ctx.terminal_resources['resource']}));return
        assert all(raw['gates'].values()) and raw['binding_sha256']==args.binding_sha
        lookup={Path(v['path']).name:v for v in inventory}
        for v in raw['output_inventory']:assert lookup[Path(v['path']).name]==v
        terminal=json.loads(Path(raw['terminal_resource_path']).read_bytes());assert terminal['raw_sha256']==args.raw_sha
        ctx.r['gates']['main_complete_and_ALL_current_output_and_terminal_SHA']=True
        import numpy as np
        import threadpoolctl
        threadpoolctl.threadpool_limits(1);assert all(v['num_threads']==1 for v in threadpoolctl.threadpool_info())
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.r['numerical_imports']=True
        controls=json.loads((folder/'controls.json').read_bytes())
        vectors=[[Fraction.from_float(float(v)) for v in controls[k][0]] for k in ('y','u','q','p')]
        yy,uu,qq,pp=vectors
        assert yy==[Fraction(1),Fraction(-2),Fraction(4),Fraction(0)]
        assert uu==[v+d for v,d in zip(yy,(Fraction(1,8),Fraction(1,16),Fraction(-1,4),Fraction(1,2)))]
        assert qq==[v+d+r for v,d,r in zip(uu,(Fraction(1,4),Fraction(-1,8),Fraction(1,8),Fraction(-1,4)),(Fraction(1,2**25),Fraction(1,2**24),Fraction(1,2**24),Fraction(1,2**27)))]
        assert pp==[Fraction.from_float(float(np.float32(float(v)))) for v in qq]
        errors=[[a-b for a,b in zip(uu,yy)],[a-b for a,b in zip(qq,uu)],[a-b for a,b in zip(pp,qq)]]
        total=[a-b for a,b in zip(pp,yy)]
        assert all(sum(parts)==v for parts,v in zip(zip(*errors),total))
        exact=[sum(v*v for v in row) for row in [yy,*errors,total]]
        exact.extend(sum(a*c for a,c in zip(errors[i],errors[j])) for i,j in ((0,1),(0,2),(1,2)))
        exact.append(sum((a-c)**2 for a,c in zip(qq,yy)))
        assert exact[4]==sum(exact[1:4])+2*sum(exact[5:8])
        assert np.allclose(controls['energies'][0][:9],[float(v) for v in exact],rtol=0,atol=1e-14)
        assert max(controls['energies'][0][10],controls['energies'][0][12])<=1
        integer=sum(((17*i)%32768)*((3*i)%255-127) for i in range(512));assert controls['signed512_exact_integer_dot']==integer
        write(ctx.out/'independent_controls.json',{'exact_dyadic_energies':[str(v) for v in exact],'signed512_literal_integer_sum':integer})
        ctx.r['gates']['NEW496_independent_Fraction_identity_cross_terms_and_literal_integer_control']=True
        def read(key,magic,width,reserved,count,dtype):
            with Path(b['data'][key]['path']).open('rb') as f:
                assert f.read(24)==struct.pack('<8sIIQ',magic,width,reserved,count)
                x=np.fromfile(f,dtype,count=count);assert len(x)==count and not f.read(1);return x
        uid=read('uid',b'M493U001',180,0,17540,np.dtype([('m','<u4',(13,)),('hash','u1',(128,))]));m=uid['m']
        occ=read('occurrences',b'M493O001',68,0,19962,np.dtype(('<u4',(17,))))
        feat=read('features',b'M495FEA1',6148,512,17540,np.dtype([('phi','<f4',(512,)),('q','<i2',(512,)),('alpha','<f4'),('l','<f4',(768,))]))
        pred=read('predictions',b'M495PRE1',6248,768,17540,np.dtype([('id','<u4'),('p','<f4'),('logits','<f4',(24,)),('oracle','<f4',(768,)),('coupled','<f4',(768,))]))
        target=read('targets',b'M493Y001',3072,768,17540,np.dtype(('<f4',(768,))))
        dev=(m[:,4]&5)!=0;val=(m[:,4]&2)!=0
        assert np.array_equal(m[:,0],np.arange(17540)) and np.all(m[:,2]==11) and np.all(m[:,3]<128) and np.all(m[:,6]==2)
        assert (int(dev.sum()),int(val.sum()))==(11721,5819) and not np.any(dev&val) and np.all(dev|val)
        assert np.array_equal(occ[:,0],np.arange(19962)) and np.all(occ[:,1]<17540) and np.all(occ[:,3]==128) and np.all(occ[:,8]==11) and np.all(occ[:,12]==1)
        for oi,mi in ((14,1),(11,3),(15,12),(16,10)):assert np.array_equal(occ[:,oi],m[occ[:,1],mi])
        assert np.array_equal(np.bincount(occ[:,1],minlength=17540),m[:,7])
        bank=Path(b['data']['bank']['path']).read_bytes();assert len(bank)==127232136 and bank[:40]==struct.pack('<8s8I',b'M495BNK1',768,3072,512,128,11,4,8,16)
        with (folder/'energy_by_uid.bin').open('rb') as f:
            assert f.read(24)==struct.pack('<8sIIQ',b'M496ENG1',104,13,17540)
            saved=np.fromfile(f,'<f8',17540*13).reshape(17540,13);assert not f.read(1)
        values=np.empty_like(saved);seen=np.zeros(17540,bool);cells=[];maxdiff=0.
        ctx.phase='independent_ALL128_and17540_fixed_levels_with_alternate_U_factorization'
        with Path(b['data']['coefficients']['path']).open('rb') as f:
            assert f.read(24)==struct.pack('<8sIIQ',b'M495FIT1',1575936,513,128)
            for e in range(128):
                cb=f.read(1575936);assert len(cb)==1575936;coef=np.frombuffer(cb,'<f4').reshape(768,513).astype('<f8')
                at=223368+992256*e+592896;codes=np.frombuffer(bank,'<i1',768*512,at).reshape(768,512)
                scales=np.frombuffer(bank,'<f4',768,at+393216).astype('<f8');bias=np.frombuffer(bank,'<f4',768,at+396288).astype('<f8')
                assert np.isfinite(coef).all() and np.all(codes!=-128) and np.isfinite(scales).all() and np.all(scales>0) and np.array_equal(coef[:,512],bias)
                indices=np.flatnonzero(m[:,3]==e);cells.append({'expert':e,'UIDs':len(indices),'development':int(dev[indices].sum()),'consumed_validation':int(val[indices].sum())})
                for start in range(0,len(indices),127):
                    ids=indices[start:start+127];q=feat['q'][ids].astype('<f8');a=feat['alpha'][ids].astype('<f8')
                    assert np.isfinite(a).all() and np.all(a>0) and np.all(np.abs(q)<=32767)
                    left=feat['l'][ids].astype('<f8');u=(left+((q@coef[:,:512].T)*a[:,None]))+coef[:,512][None,:]
                    dot=q@codes.astype('<f8').T;assert np.all(np.abs(dot)<=512*127*32767)
                    integer=dot.astype('<i8');assert np.array_equal(dot,integer)
                    decoded=(left+((integer.astype('<f8')*scales[None,:])*a[:,None]))+bias[None,:]
                    p=pred['oracle'][ids].astype('<f8');y=target[ids].astype('<f8')
                    ef=u-y;eq=decoded-u;ea=p-decoded;et=p-y
                    diag=[np.einsum('ij,ij->i',v,v) for v in (y,ef,eq,ea,et)]
                    cross=[np.einsum('ij,ij->i',v,w) for v,w in ((ef,eq),(ef,ea),(eq,ea))]
                    residual=(ef+eq+ea)-et;bound=3e-12*np.maximum(1,np.abs(y)+np.abs(u)+np.abs(decoded)+np.abs(p))
                    closure=diag[4]-(diag[1]+diag[2]+diag[3]+2*(cross[0]+cross[1]+cross[2]))
                    energybound=1e-10*np.maximum(1,diag[0]+diag[1]+diag[2]+diag[3]+2*(np.abs(cross[0])+np.abs(cross[1])+np.abs(cross[2])))
                    pre=decoded-y
                    row=np.column_stack((*diag,*cross,np.einsum('ij,ij->i',pre,pre),np.max(np.abs(residual),axis=1),np.max(np.abs(residual)/bound,axis=1),np.abs(closure),np.abs(closure)/energybound))
                    assert np.isfinite(row).all() and np.max(row[:,[10,12]])<=1 and np.max(saved[ids][:,[10,12]])<=1
                    assert np.allclose(row[:,:9],saved[ids,:9],rtol=1e-9,atol=1e-5)
                    maxdiff=max(maxdiff,float(np.max(np.abs(row[:,:9]-saved[ids,:9]))));values[ids]=row;seen[ids]=True;ctx.guard()
                if e%16==0:ctx.log(experts_audited=e+1);print(json.dumps({'experts_audited':e+1}),flush=True)
            assert not f.read(1)
        assert seen.all() and cells==raw['cells']
        ctx.r['gates']['ALL17540_independent_levels_13_columns_cross_terms_and_both_closure_envelopes']=True
        def metric(indices):
            n=len(indices);s=np.sum(values[indices,:9],axis=0) if n else np.zeros(9);correct=int(np.sum(pred['id'][indices]==m[indices,3]))
            v={'count':n,'ID_correct':correct,'ID_fidelity':correct/n if n else None,'source_energy':float(s[0])}
            for j,name in enumerate(('fit','parameters','arithmetic','total'),1):v[name+'_error_energy']=float(s[j]);v[name+'_RMS']=math.sqrt(float(s[j]/s[0])) if s[0] else None
            for j,name in enumerate(('fit_parameters','fit_arithmetic','parameters_arithmetic'),5):v[name+'_inner']=float(s[j])
            v['pre_arithmetic_error_energy']=float(s[8]);v['pre_arithmetic_RMS']=math.sqrt(float(s[8]/s[0])) if s[0] else None
            for j,name in enumerate(('max_identity_abs','max_identity_envelope_ratio','max_energy_closure_abs','max_energy_closure_envelope_ratio'),9):v[name]=float(np.max(values[indices,j])) if n else 0.
            return v
        reports={k:[] for k in ('uid_roles','cells','rare','views','role_mode','exposures')};counts=np.bincount(m[dev,3],minlength=128)
        for name,mask in (('development',dev),('consumed_validation',val)):reports['uid_roles'].append({'split':name,**metric(np.flatnonzero(mask))})
        for e in range(128):
            for name,mask in (('development',dev),('consumed_validation',val)):reports['cells'].append({'expert':e,'development_exposure':int(counts[e]),'split':name,**metric(np.flatnonzero(mask&(m[:,3]==e)))})
        for name,lo,hi in (('0',0,0),('1..4',1,4),('5..15',5,15),('>=16',16,17540)):
            category=(counts>=lo)&(counts<=hi)
            for split,mask in (('development',dev),('consumed_validation',val)):reports['rare'].append({'development_class':name,'split':split,**metric(np.flatnonzero(mask&category[m[:,3]]))})
        for book in range(192):
            for mode in range(2):
                for accepted in range(2):reports['views'].append({'book':book,'role':book//64,'mode':mode,'accepted':accepted,**metric(occ[(occ[:,4]==book)&(occ[:,6]==mode)&(occ[:,12]==accepted),1])})
        for role in range(3):
            for mode in range(2):reports['role_mode'].append({'role':role,'mode':mode,**metric(occ[(occ[:,7]==role)&(occ[:,6]==mode),1])})
            ix=occ[occ[:,7]==role,1];sc=np.bincount(m[ix,3],minlength=128);pc=np.bincount(pred['id'][ix],minlength=128)
            for e in range(128):reports['exposures'].append({'role':role,'expert':e,'source':int(sc[e]),'candidate':int(pc[e])})
        residualnames={'max_identity_abs','max_identity_envelope_ratio','max_energy_closure_abs','max_energy_closure_envelope_ratio'}
        for label in ('uid_roles','cells','rare','views','role_mode'):
            assert len(reports[label])==len(raw['reports'][label])
            for current,original in zip(reports[label],raw['reports'][label]):
                assert current.keys()==original.keys()
                for key,v in current.items():
                    if key in residualnames:
                        assert v>=0 and original[key]>=0
                        if key.endswith('ratio'):assert v<=1 and original[key]<=1
                    elif isinstance(v,float):assert np.isclose(v,original[key],rtol=1e-9,atol=1e-5),(label,key,v,original[key])
                    else:assert v==original[key],(label,key)
        assert reports['exposures']==raw['reports']['exposures'] and sum(len(reports[k]) for k in ('uid_roles','cells','rare','views','role_mode'))==1040
        assert sum(v['count'] for v in reports['views'])==sum(v['count'] for v in reports['role_mode'])==19962
        rm=reports['role_mode'];outcomes={'ALL_six_unquantized_U_RMS_1pct':all(v['count'] and v['fit_RMS'] is not None and v['fit_RMS']<=.01 for v in rm),
            'ALL_six_decoded_Q_RMS_1pct':all(v['count'] and v['pre_arithmetic_RMS'] is not None and v['pre_arithmetic_RMS']<=.01 for v in rm),
            'ALL_six_saved_P_RMS_1pct':all(v['count'] and v['total_RMS'] is not None and v['total_RMS']<=.01 for v in rm),
            'ALL_six_parameters_RMS_at_least_10x_arithmetic':all(v['count'] and v['parameters_RMS']>=10*v['arithmetic_RMS'] for v in rm)}
        assert outcomes==raw['reports']['diagnostic_outcomes']
        decision='UNQUANTIZED_FITTED_FUNCTION_REQUIRES_ANALYSIS_BEFORE_PRECISION_CHANGE' if not outcomes['ALL_six_unquantized_U_RMS_1pct'] else ('PHYSICAL_READOUT_COEFFICIENT_ERROR_SELECTED_FOR_NEXT_CHANGE' if not outcomes['ALL_six_saved_P_RMS_1pct'] and outcomes['ALL_six_parameters_RMS_at_least_10x_arithmetic'] else 'PHYSICAL_ARITHMETIC_ERROR_REQUIRES_ANALYSIS')
        assert decision==raw['decision']
        for key in ('source_FFN_calls','model_calls','native_calls','optimizer_updates','projection_solves'):assert raw[key]==0
        ctx.r['gates']['ALL1040_reports_384_exposures_original_domains_and_frozen_decision']=True
        result=ctx.finish({'main_completed':True,'raw':{'path':str(rp),'sha256':args.raw_sha},'main_windows_sha256':ctx.digest(eventpath),
            'retained_main_inventory':inventory,'UIDs_audited':17540,'occurrences_audited':19962,'experts_audited':128,'metric_groups_audited':1040,'exposure_groups_audited':384,
            'max_energy_inner_difference':maxdiff,'max_identity_envelope_ratio':float(values[:,10].max()),'max_energy_closure_envelope_ratio':float(values[:,12].max()),
            'reports':reports,'diagnostic_outcomes':outcomes,'decision':decision,'source_FFN_calls':0,'model_calls':0,'native_calls':0,
            'scope':'Independently audited fixed-artifact numerical decomposition only. No quality, speed, useful n or goal promotion.'})
        print(json.dumps({'gates':result['gates'],'decision':decision,'resource':ctx.terminal_resources['resource']}),flush=True)
    except BaseException as exc:
        ctx.fail(exc);print(json.dumps({'error':str(exc),'traceback':traceback.format_exc()}),flush=True);raise

if __name__=='__main__':main()
