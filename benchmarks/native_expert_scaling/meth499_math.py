"""Fixed unweighted-prior QR learner, factorized accounting and complete reports."""
import math
import numpy as np
from meth498_math import UID,INPUT,FEATURE,PRED as BASELINE,minimum,envelope,quant,parts,block,emit
NATIVE_INPUT=np.dtype([('core',INPUT),('source_p','<f4')])
PRED=np.dtype([('id','<u4'),('p','<f4'),('logits','<f4',(24,)),('F','<f4',(768,)),('oracle','<f4',(768,)),
    ('sameID_mass','<f4',(768,)),('choice_source_mass','<f4',(768,)),('coupled','<f4',(768,))])
QUERY=np.dtype([('ledger_record','<u8'),('book','<u2'),('case','u1'),('mode','u1'),('role','u1'),('accepted','u1'),
    ('expert','<u2'),('index','<u4'),('alpha','<f4'),('probability','<f4'),('input','<f4',(768,)),('codes','<i2',(768,)),
    ('input_sha','u1',(32,)),('code_sha','u1',(32,)),('pair_sha','u1',(32,))])
PRODUCT_INPUT=[0x80000000,0x3f800000,0x3f800003,0x7f7fffff,3,5,6,0x80000006]
PRODUCT_EXPECTED=[0x80000000,0x3e800000,0x3e800003,0x7e7fffff,1,1,2,0x80000002]

def controls(native):
    h=np.array([[2.,-1.,1.]],'<f8');r=np.array([[4.,-2.]],'<f8');prior=np.array([[1.,.5,-1.],[-.5,1.,2.]],'<f8')
    c,record=minimum(h,r,prior,np.diag([2.,1.,1.]),20.,lambda:None)
    expected=np.array([[19/12,-2/3,1/6],[-5/6,5/3,4/3]],'<f8');assert np.max(np.abs(c-expected))<=1e-14
    code,scale=quant(np.array([[127,-127,3.5,4.5,-3.5,-4.5,0,1]],'<f4'))
    assert code.tolist()==[[127,-127,4,4,-4,-4,0,1]] and scale.tolist()==[1.]
    x=[32766,-30001,85,-6179];table=[]
    for i in range(81):
        z=i;s=0
        for v in x:s+=(z%3-1)*v;z//=3
        table.append(s)
    assert native[0]=={'table81':table}
    assert native[1]=={'alpha_bits':0x3f800000,'codes':[32767,-32767,0,2,2,0,-2,-2]}
    assert native[2]['L_I64']==768*127*32767>2**31 and native[2]['B_I32']==512*127*32767<2**31
    assert native[2]['RNE']==0 and not native[2]['MXCSR']&0x8040 and native[2]['CPU0']==1
    actual=np.multiply(np.array(PRODUCT_INPUT,'<u4').view('<f4'),np.float32(.25),dtype=np.float32).view('<u4').tolist()
    assert actual==PRODUCT_EXPECTED and native[3]=={'quarter_product_bits':PRODUCT_EXPECTED,'rejected_bits':0}
    return {'nearest_prior_coefficients':c.tolist(),'nearest_prior_record':record,'I8_codes':code.tolist(),'I8_scale':scale.tolist(),
        'product_input_bits':PRODUCT_INPUT,'product_expected_bits':PRODUCT_EXPECTED,'native':native}

def decomposition(y,u64,u32,decoded,physical):
    errors=[u64-y,u32-u64,decoded-u32,physical-decoded];total=physical-y
    diag=[np.sum(v*v,axis=1) for v in (y,errors[0],u32-y,total,errors[1],errors[2],errors[3])]
    pairs=[np.sum(errors[i]*errors[j],axis=1) for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))]
    identity=sum(errors)-total;bound=3e-12*np.maximum(1,np.abs(y)+np.abs(u64)+np.abs(u32)+np.abs(decoded)+np.abs(physical))
    closure=diag[3]-(diag[1]+diag[4]+diag[5]+diag[6]+2*sum(pairs))
    cb=1e-10*np.maximum(1,diag[0]+diag[1]+diag[4]+diag[5]+diag[6]+2*sum(np.abs(v) for v in pairs))
    a=np.column_stack((*diag,*pairs,np.max(np.abs(identity),axis=1),np.max(np.abs(identity)/bound,axis=1),
        np.abs(closure),np.abs(closure)/cb,np.sum((decoded-y)**2,axis=1)))
    assert np.isfinite(a).all() and np.max(a[:,[14,16]])<=1;return a

def energies(f,y,u64,u32,q,pred,p_source):
    raw=decomposition(f,u64,u32,q,pred['F'].astype('<f8'))
    weighted=decomposition(y,p_source[:,None]*u64,p_source[:,None]*u32,p_source[:,None]*q,pred['oracle'].astype('<f8'))
    ef=pred['oracle'].astype('<f8')-y;ec=pred['choice_source_mass'].astype('<f8')-pred['oracle'].astype('<f8')
    em=pred['coupled'].astype('<f8')-pred['choice_source_mass'].astype('<f8');total=pred['coupled'].astype('<f8')-y
    z=pred['sameID_mass'].astype('<f8')-pred['oracle'].astype('<f8')
    d=[np.sum(v*v,axis=1) for v in (total,ec,em,z)]
    crosses=[np.sum(v*w,axis=1) for v,w in ((ef,ec),(ef,em),(ec,em))]
    identity=ef+ec+em-total;ib=3e-12*np.maximum(1,np.abs(y)+np.abs(pred['oracle'].astype('<f8'))+
        np.abs(pred['choice_source_mass'].astype('<f8'))+np.abs(pred['coupled'].astype('<f8')))
    closure=d[0]-(weighted[:,3]+d[1]+d[2]+2*sum(crosses));eb=1e-10*np.maximum(1,weighted[:,0]+weighted[:,3]+d[1]+d[2]+2*sum(np.abs(v) for v in crosses))
    a=np.column_stack((raw,weighted,*d,(pred['p'].astype('<f8')-p_source)**2,p_source**2,*crosses,
        np.max(np.abs(identity)/ib,axis=1),np.abs(closure)/eb))
    assert a.shape==(len(y),47) and np.isfinite(a).all() and np.max(a[:,[45,46]])<=1;return a

def reports(meta,occ,data,picked,psrc,phat):
    def metric(ix):
        n=len(ix);s=np.sum(data[ix],axis=0) if n else np.zeros(47);correct=int(np.sum(picked[ix]==meta[ix,3]))
        v={'count':n,'ID_correct':correct,'ID_fidelity':correct/n if n else None,'source_energy':float(s[18]),'unweighted_source_energy':float(s[0])}
        for prefix,offset in (('unweighted',0),('weighted',18)):
            for j,name in enumerate(('fit64','fit32','oracle','serialization','parameters','arithmetic'),1):
                energy=float(s[offset+j]);v[prefix+'_'+name+'_error_energy']=energy;v[prefix+'_'+name+'_RMS']=math.sqrt(energy/s[offset]) if s[offset] else None
            for j,name in enumerate(('fit_serial','fit_parameter','fit_arithmetic','serial_parameter','serial_arithmetic','parameter_arithmetic'),7):v[prefix+'_'+name+'_inner']=float(s[offset+j])
            for j,name in ((13,'identity_maxabs'),(14,'identity_ratio'),(15,'closure_maxabs'),(16,'closure_ratio')):
                v[prefix+'_'+name]=float(np.max(data[ix,offset+j])) if n else 0.
            v[prefix+'_pre_arithmetic_error_energy']=float(s[offset+17])
        for j,name in enumerate(('coupled','choice','mass','sameID_mass'),36):
            energy=float(s[j]);v[name+'_error_energy']=energy;v[name+'_RMS']=math.sqrt(energy/s[18]) if s[18] else None
        for j,name in enumerate(('fit_choice','fit_mass','choice_mass'),42):v[name+'_inner']=float(s[j])
        v['probability_relative_RMS']=math.sqrt(float(s[40]/s[41])) if s[41] else None
        v['probability_within1pct']=int(np.sum(np.abs(phat[ix].astype('<f8')/psrc[ix].astype('<f8')-1)<=.01))
        matched=ix[picked[ix]==meta[ix,3]];v['same_ID_probability_count']=len(matched)
        sp=float(np.sum(data[matched,41]));v['same_ID_probability_relative_RMS']=math.sqrt(float(np.sum(data[matched,40]))/sp) if sp else None
        v['coupled_identity_ratio']=float(np.max(data[ix,45])) if n else 0.;v['coupled_closure_ratio']=float(np.max(data[ix,46])) if n else 0.
        return v
    dev=(meta[:,4]&5)!=0;counts=np.bincount(meta[dev,3],minlength=128)
    roles=[{'split':s,**metric(np.flatnonzero(mask))} for s,mask in (('development',dev),('consumed_validation',~dev))]
    cells=[{'expert':e,'development_exposure':int(counts[e]),'split':s,**metric(np.flatnonzero(mask&(meta[:,3]==e)))} for e in range(128) for s,mask in (('development',dev),('consumed_validation',~dev))]
    rare=[{'development_class':name,'split':s,**metric(np.flatnonzero(mask&category[meta[:,3]]))}
        for name,category in (('0',counts==0),('1..4',(counts>=1)&(counts<=4)),('5..15',(counts>=5)&(counts<=15)),('>=16',counts>=16)) for s,mask in (('development',dev),('consumed_validation',~dev))]
    views=[{'book':book,'role':book//64,'mode':mo,'accepted':accepted,**metric(occ[(occ[:,4]==book)&(occ[:,6]==mo)&(occ[:,12]==accepted),1])} for book in range(192) for mo in range(2) for accepted in range(2)]
    rm=[{'role':r,'mode':mo,**metric(occ[(occ[:,7]==r)&(occ[:,6]==mo),1])} for r in range(3) for mo in range(2)]
    exposures=[]
    for role in range(3):
        ids=occ[occ[:,7]==role,1];source=np.bincount(meta[ids,3],minlength=128);candidate=np.bincount(picked[ids],minlength=128)
        exposures.extend({'role':role,'expert':e,'source':int(source[e]),'candidate':int(candidate[e])} for e in range(128))
    outcomes={name+'_ALL_six_RMS_1pct':all(v['count'] and v[name+'_RMS'] is not None and v[name+'_RMS']<=.01 for v in rm)
        for name in ('unweighted_fit64','unweighted_fit32','unweighted_oracle','weighted_fit64','weighted_fit32','weighted_oracle','coupled')}
    outcomes['ID_ALL_six_999permille']=all(v['count'] and v['ID_fidelity']>=.999 for v in rm)
    outcomes['mass_ALL_six_relative_RMS_1pct']=all(v['count'] and v['probability_relative_RMS'] is not None and v['probability_relative_RMS']<=.01 for v in rm)
    return {'uid_roles':roles,'cells':cells,'rare':rare,'views':views,'role_mode':rm,'exposures':exposures,'outcomes':outcomes}

def decision(o):
    if not (o['unweighted_fit64_ALL_six_RMS_1pct'] and o['weighted_fit64_ALL_six_RMS_1pct']):return 'NEXT_SOURCE_FEATURE_INFORMATION'
    if not (o['unweighted_fit32_ALL_six_RMS_1pct'] and o['weighted_fit32_ALL_six_RMS_1pct']):return 'NEXT_COEFFICIENT_SERIALIZATION'
    if not (o['unweighted_oracle_ALL_six_RMS_1pct'] and o['weighted_oracle_ALL_six_RMS_1pct']):return 'NEXT_PHYSICAL_COEFFICIENT_ENCODING'
    if not (o['coupled_ALL_six_RMS_1pct'] and o['ID_ALL_six_999permille'] and o['mass_ALL_six_relative_RMS_1pct']):return 'NEXT_ROUTING_AND_MASS'
    return 'LOCAL_FACTORIZED_TRANSFER_ELIGIBLE_FOR_COMPOSED_CONTEXT'
