"""Fixed three-level arithmetic and complete reporting; no fitting."""
import math
import numpy as np
UID=np.dtype([('m','<u4',(13,)),('hash','u1',(128,))])
FEATURE=np.dtype([('phi','<f4',(512,)),('q','<i2',(512,)),('alpha','<f4'),('l','<f4',(768,))])
PRED=np.dtype([('id','<u4'),('p','<f4'),('logits','<f4',(24,)),('oracle','<f4',(768,)),('coupled','<f4',(768,))])

def energy(y,u,q,p):
    fit=u-y; param=q-u; arithmetic=p-q; total=p-y
    inner=[np.sum(a*b,axis=1) for a,b in ((fit,param),(fit,arithmetic),(param,arithmetic))]
    diagonal=[np.sum(a*a,axis=1) for a in (y,fit,param,arithmetic,total)]
    identity=(fit+param+arithmetic)-total
    vector_bound=3e-12*np.maximum(1,np.abs(y)+np.abs(u)+np.abs(q)+np.abs(p))
    vector_ratio=np.max(np.abs(identity)/vector_bound,axis=1)
    closure=diagonal[4]-(diagonal[1]+diagonal[2]+diagonal[3]+2*(inner[0]+inner[1]+inner[2]))
    closure_bound=1e-10*np.maximum(1,diagonal[0]+diagonal[1]+diagonal[2]+diagonal[3]+2*(np.abs(inner[0])+np.abs(inner[1])+np.abs(inner[2])))
    answer=np.column_stack((*diagonal,*inner,np.sum((q-y)**2,axis=1),np.max(np.abs(identity),axis=1),vector_ratio,np.abs(closure),np.abs(closure)/closure_bound))
    assert answer.shape==(len(y),13) and np.isfinite(answer).all() and np.max(answer[:,[10,12]])<=1
    return answer

def controls():
    y=np.array([[1,-2,4,0]],'<f8'); u=y+np.array([[.125,.0625,-.25,.5]],'<f8')
    q=u+np.array([[.25,-.125,.125,-.25]],'<f8')+np.array([[2**-25,2**-24,2**-24,2**-27]],'<f8')
    p=q.astype('<f4').astype('<f8'); values=energy(y,u,q,p)
    code=np.array([(17*i)%32768 for i in range(512)],'<i2'); weight=np.array([(3*i)%255-127 for i in range(512)],'<i1')
    integer=sum(int(a)*int(b) for a,b in zip(code,weight)); dot=float(code.astype('<f8')@weight.astype('<f8'))
    assert dot==integer and abs(integer)<2**31
    return {'y':y.tolist(),'u':u.tolist(),'q':q.tolist(),'p':p.tolist(),'energies':values.tolist(),'signed512_exact_integer_dot':integer}

def reports(meta,occ,data,picked):
    def metric(indices):
        n=len(indices); correct=int(np.count_nonzero(picked[indices]==meta[indices,3])); sums=np.sum(data[indices,:9],axis=0) if n else np.zeros(9)
        result={'count':n,'ID_correct':correct,'ID_fidelity':correct/n if n else None,'source_energy':float(sums[0])}
        for j,name in enumerate(('fit','parameters','arithmetic','total'),1):
            result[name+'_error_energy']=float(sums[j]); result[name+'_RMS']=math.sqrt(float(sums[j])/float(sums[0])) if sums[0] else None
        for j,name in enumerate(('fit_parameters','fit_arithmetic','parameters_arithmetic'),5):result[name+'_inner']=float(sums[j])
        result['pre_arithmetic_error_energy']=float(sums[8]); result['pre_arithmetic_RMS']=math.sqrt(float(sums[8])/float(sums[0])) if sums[0] else None
        result['max_identity_abs']=float(np.max(data[indices,9])) if n else 0.
        result['max_identity_envelope_ratio']=float(np.max(data[indices,10])) if n else 0.
        result['max_energy_closure_abs']=float(np.max(data[indices,11])) if n else 0.
        result['max_energy_closure_envelope_ratio']=float(np.max(data[indices,12])) if n else 0.
        return result
    dev=(meta[:,4]&5)!=0; counts=np.bincount(meta[dev,3],minlength=128)
    roles=[{'split':s,**metric(np.flatnonzero(mask))} for s,mask in (('development',dev),('consumed_validation',~dev))]
    cells=[{'expert':e,'development_exposure':int(counts[e]),'split':s,**metric(np.flatnonzero(mask&(meta[:,3]==e)))}
        for e in range(128) for s,mask in (('development',dev),('consumed_validation',~dev))]
    rare=[{'development_class':name,'split':s,**metric(np.flatnonzero(mask&category[meta[:,3]]))}
        for name,category in (('0',counts==0),('1..4',(counts>=1)&(counts<=4)),('5..15',(counts>=5)&(counts<=15)),('>=16',counts>=16))
        for s,mask in (('development',dev),('consumed_validation',~dev))]
    views=[{'book':book,'role':0 if book<64 else 1 if book<128 else 2,'mode':mode,'accepted':accepted,
        **metric(occ[(occ[:,4]==book)&(occ[:,6]==mode)&(occ[:,12]==accepted),1])}
        for book in range(192) for mode in range(2) for accepted in range(2)]
    rm=[{'role':r,'mode':mo,**metric(occ[(occ[:,7]==r)&(occ[:,6]==mo),1])} for r in range(3) for mo in range(2)]
    exposures=[]
    for role in range(3):
        indices=occ[occ[:,7]==role,1]; source=np.bincount(meta[indices,3],minlength=128); candidate=np.bincount(picked[indices],minlength=128)
        exposures.extend({'role':role,'expert':e,'source':int(source[e]),'candidate':int(candidate[e])} for e in range(128))
    diagnostic={'ALL_six_unquantized_U_RMS_1pct':all(v['count'] and v['fit_RMS'] is not None and v['fit_RMS']<=.01 for v in rm),
        'ALL_six_decoded_Q_RMS_1pct':all(v['count'] and v['pre_arithmetic_RMS'] is not None and v['pre_arithmetic_RMS']<=.01 for v in rm),
        'ALL_six_saved_P_RMS_1pct':all(v['count'] and v['total_RMS'] is not None and v['total_RMS']<=.01 for v in rm),
        'ALL_six_parameters_RMS_at_least_10x_arithmetic':all(v['count'] and v['parameters_RMS']>=10*v['arithmetic_RMS'] for v in rm)}
    return {'uid_roles':roles,'cells':cells,'rare':rare,'views':views,'role_mode':rm,'exposures':exposures,'diagnostic_outcomes':diagnostic}
