"""One fixed nearest-prior QR learner, physical reference and complete reports."""
import math
import numpy as np
UID=np.dtype([('m','<u4',(13,)),('hash','u1',(128,))])
INPUT=np.dtype([('e','<u4'),('accept','<u4'),('alpha','<f4'),('x','<f4',(768,)),('q','<i2',(768,))])
FEATURE=np.dtype([('phi','<f4',(512,)),('q','<i2',(512,)),('alpha','<f4'),('l','<f4',(768,))])
PRED=np.dtype([('id','<u4'),('p','<f4'),('logits','<f4',(24,)),('oracle','<f4',(768,)),('coupled','<f4',(768,))])

def envelope(a,x,rhs,tolerance):
    error=a@x-rhs;bound=tolerance*np.maximum(1,np.abs(a)@np.abs(x)+np.abs(rhs))
    ratio=float(np.max(np.abs(error)/bound)) if error.size else 0.
    assert np.isfinite(error).all() and np.isfinite(bound).all() and ratio<=1,('equation_envelope',ratio)
    return ratio

def triangular(a,b,lower,guard):
    assert a.shape==(len(a),len(a)) and b.shape[0]==len(a) and np.all(np.diag(a)!=0)
    answer=np.empty_like(b)
    for i in (range(len(a)) if lower else range(len(a)-1,-1,-1)):
        at=slice(0,i) if lower else slice(i+1,len(a))
        answer[i]=(b[i]-a[i,at]@answer[at])/a[i,i]
        if i%32==0:guard()
    assert np.isfinite(answer).all();return answer

def minimum(h,r,prior,t,source_energy,guard):
    if not len(h):return prior.copy(),{'development':0,'retained_prior':True}
    zt=triangular(t,h.T,True,guard);white_ratio=envelope(t,zt,h.T,3e-12)
    q,qr=np.linalg.qr(zt,mode='reduced');assert q.shape==zt.shape and np.isfinite(qr).all()
    factor_ratio=envelope(q,qr,zt,5e-12);orth=float(np.max(np.abs(q.T@q-np.eye(len(h)))));assert orth<=5e-11
    e=r-h@prior.T;x=triangular(qr.T,e,True,guard);solve_ratio=envelope(qr.T,x,e,3e-12)
    dt=q@x;delta_t=triangular(t.T,dt,False,guard);back_ratio=envelope(t.T,delta_t,dt,3e-12)
    c=prior+delta_t.T;feasible_ratio=envelope(h,c.T,r,3e-10)
    difference=h@c.T-r;relative=math.sqrt(float(np.sum(difference*difference))/source_energy);assert relative<=1e-7
    whitened=t.T@(c-prior).T;project=q@(q.T@whitened)
    bound=3e-10*np.maximum(1,np.abs(whitened)+np.abs(q)@(np.abs(q.T)@np.abs(whitened)))
    stationary=float(np.max(np.abs(whitened-project)/bound));assert np.isfinite(c).all() and stationary<=1
    diag=np.abs(np.diag(qr))
    return c,{'development':len(h),'retained_prior':False,'whitening_envelope_ratio':white_ratio,'QR_envelope_ratio':factor_ratio,
        'QR_orthogonality_maxabs':orth,'constraint_solve_envelope_ratio':solve_ratio,'backtransform_envelope_ratio':back_ratio,
        'feasibility_envelope_ratio':feasible_ratio,'development_F64_fit_RMS':relative,'stationarity_envelope_ratio':stationary,
        'prior_metric_displacement_energy':float(np.sum(whitened*whitened)),'QR_min_abs_diagonal':float(diag.min()),'QR_max_abs_diagonal':float(diag.max())}

def quant(c):
    mx=np.max(np.abs(c),axis=1);scale=np.divide(mx,np.float32(127),dtype=np.float32);scale[mx==0]=1
    assert np.isfinite(scale).all() and np.all(scale>0)
    code=np.clip(np.rint(np.divide(c,scale[:,None],dtype=np.float32)),-127,127).astype('<i1');return code,scale

def controls():
    h=np.array([[1.,2.,1.]],'<f8');r=np.array([[3.,6.]],'<f8');prior=np.array([[.5,-.25,1.],[1.,0.,-.5]],'<f8')
    c,record=minimum(h,r,prior,np.diag([1.,2.,1.]),45.,lambda:None)
    expected=np.array([[7/6,1/12,5/3],[17/6,11/12,4/3]],'<f8');assert np.max(np.abs(c-expected))<=1e-14
    code,scale=quant(np.array([[127,-127,.5,1.5,2.5,-.5,-1.5,-2.5]],'<f4'));assert code.tolist()==[[127,-127,0,2,2,0,-2,-2]] and scale.tolist()==[1.]
    return {'nearest_prior_coefficients':c.tolist(),'nearest_prior_record':record,'I8_half_even_codes':code.tolist(),'I8_half_even_scale':scale.tolist()}

def parts(bank,e):
    at=223368+992256*e
    lq=np.frombuffer(bank,'<i1',768*768,at).reshape(768,768);ls=np.frombuffer(bank,'<f4',768,at+589824)
    at+=592896;bq=np.frombuffer(bank,'<i1',768*512,at).reshape(768,512)
    bs=np.frombuffer(bank,'<f4',768,at+393216);bias=np.frombuffer(bank,'<f4',768,at+396288);return lq,ls,bq,bs,bias

def block(q,w,s,a):
    integer=q.astype('<f8')@w.astype('<f8').T
    assert np.array_equal(integer,np.rint(integer)) and np.all(np.abs(integer)<=q.shape[1]*127*32767)
    return ((integer*s.astype('<f8')[None,:])*a.astype('<f8')[:,None]).astype('<f4')

def emit(bank,inputs,features,ids,guard):
    output=np.empty((len(inputs),768),'<f4')
    for e in range(128):
        positions=np.flatnonzero(ids==e);lq,ls,bq,bs,bias=parts(bank,e)
        for start in range(0,len(positions),256):
            at=positions[start:start+256];left=block(inputs['q'][at],lq,ls,inputs['alpha'][at]);right=block(features['q'][at],bq,bs,features['alpha'][at])
            output[at]=np.add(np.add(left,right,dtype=np.float32),bias,dtype=np.float32);guard()
    assert np.isfinite(output).all();return output

def energies(y,u64,u32,decoded,p,coupled):
    errors=[u64-y,u32-u64,decoded-u32,p-decoded];total=p-y
    pairs=[np.sum(errors[i]*errors[j],axis=1) for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))]
    diag=[np.sum(v*v,axis=1) for v in (y,errors[0],u32-y,total,coupled-y,coupled-p,errors[1],errors[2],errors[3])]
    identity=sum(errors)-total;bound=3e-12*np.maximum(1,np.abs(y)+np.abs(u64)+np.abs(u32)+np.abs(decoded)+np.abs(p))
    closure=diag[3]-(diag[1]+diag[6]+diag[7]+diag[8]+2*sum(pairs))
    cb=1e-10*np.maximum(1,diag[0]+diag[1]+diag[6]+diag[7]+diag[8]+2*sum(np.abs(v) for v in pairs))
    answer=np.column_stack((*diag,*pairs,np.max(np.abs(identity),axis=1),np.max(np.abs(identity)/bound,axis=1),np.abs(closure),np.abs(closure)/cb,np.sum((decoded-y)**2,axis=1)))
    assert answer.shape==(len(y),20) and np.isfinite(answer).all() and np.max(answer[:,[16,18]])<=1;return answer

def reports(meta,occ,data,picked):
    def metric(indices):
        n=len(indices);correct=int(np.sum(picked[indices]==meta[indices,3]));s=np.sum(data[indices,:15],axis=0) if n else np.zeros(15)
        result={'count':n,'ID_correct':correct,'ID_fidelity':correct/n if n else None,'source_energy':float(s[0])}
        for j,name in enumerate(('fit64','fit32','oracle','coupled','decision','serialization','parameters','arithmetic'),1):
            result[name+'_error_energy']=float(s[j]);result[name+'_RMS']=math.sqrt(float(s[j]/s[0])) if s[0] else None
        for j,name in enumerate(('fit_serial','fit_parameter','fit_arithmetic','serial_parameter','serial_arithmetic','parameter_arithmetic'),9):result[name+'_inner']=float(s[j])
        for j,name in enumerate(('max_identity_abs','max_identity_envelope_ratio','max_energy_closure_abs','max_energy_closure_envelope_ratio'),15):result[name]=float(np.max(data[indices,j])) if n else 0.
        result['pre_arithmetic_error_energy']=float(np.sum(data[indices,19])) if n else 0.;return result
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
    gates={name+'_ALL_six_RMS_1pct':all(v['count'] and v[name+'_RMS'] is not None and v[name+'_RMS']<=.01 for v in rm) for name in ('fit64','fit32','oracle','coupled')}
    gates['ID_ALL_six_999permille']=all(v['count'] and v['ID_fidelity']>=.999 for v in rm)
    return {'uid_roles':roles,'cells':cells,'rare':rare,'views':views,'role_mode':rm,'exposures':exposures,'outcomes':gates}

def decision(outcome):
    return 'NEXT_SOURCE_PRIOR_TRANSPORT' if not outcome['fit64_ALL_six_RMS_1pct'] else 'NEXT_COEFFICIENT_SERIALIZATION' if not outcome['fit32_ALL_six_RMS_1pct'] else 'NEXT_PHYSICAL_COEFFICIENT_ENCODING' if not outcome['oracle_ALL_six_RMS_1pct'] else 'NEXT_UNRESOLVED_ROUTING_AND_MASS'
