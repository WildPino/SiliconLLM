"""Fixed learner/reference operators and exhaustive report definitions."""
import math
import numpy as np
N,D,R=17540,768,512
EB,OFFSET=992256,223368
INPUT=np.dtype([('expert','<u4'),('accepted','<u4'),('alpha','<f4'),('x','<f4',(D,)),('q','<i2',(D,))])
FEATURE=np.dtype([('phi','<f4',(R,)),('q','<i2',(R,)),('alpha','<f4'),('linear','<f4',(D,))])
PRED=np.dtype([('expert','<u4'),('p','<f4'),('logits','<f4',(24,)),('oracle','<f4',(D,)),('coupled','<f4',(D,))])
UID=np.dtype([('meta','<u4',(13,)),('hashes','u1',(128,))])

def quant(x,maximum):
    mx=np.max(np.abs(x),axis=1); a=(mx/np.float32(maximum)).astype('<f4'); a[mx==0]=1
    assert np.isfinite(a).all() and np.all(a>0)
    codes=np.clip(np.rint(np.divide(x,a[:,None],dtype=np.float32)),-maximum,maximum)
    return codes.astype('<i1' if maximum==127 else '<i2'),a

def dictionary(bank):
    packed=np.frombuffer(bank,offset=40,count=98304,dtype='u1').reshape(R,192)
    assert np.all(packed<=80); v=packed.astype('<i2'); t=np.empty((R,192,4),'<i1')
    for j in range(4): t[:,:,j]=v%3-1; v//=3
    scales=np.frombuffer(bank,offset=98344,count=512,dtype='<f4')
    return t.reshape(R,D),scales

def expert(bank,e):
    at=OFFSET+e*EB
    lq=np.frombuffer(bank,offset=at,count=D*D,dtype='<i1').reshape(D,D); at+=D*D
    ls=np.frombuffer(bank,offset=at,count=D,dtype='<f4'); at+=D*4
    bq=np.frombuffer(bank,offset=at,count=D*R,dtype='<i1').reshape(D,R); at+=D*R
    bs=np.frombuffer(bank,offset=at,count=D,dtype='<f4'); at+=D*4
    bias=np.frombuffer(bank,offset=at,count=D,dtype='<f4')
    return lq,ls,bq,bs,bias

def block(q,w,scale,alpha):
    # Integer products, every intermediate and exact sum below 2^53.
    dot=q.astype('<f8')@w.astype('<f8').T
    return ((dot*scale.astype('<f8')[None,:])*alpha.astype('<f8')[:,None]).astype('<f4')

def emit(bank,inputs,features,ids):
    y=np.empty((len(inputs),D),'<f4')
    for e in range(128):
        at=np.flatnonzero(ids==e)
        if not len(at): continue
        lq,ls,bq,bs,bias=expert(bank,e)
        left=block(inputs['q'][at],lq,ls,inputs['alpha'][at])
        right=block(features['q'][at],bq,bs,features['alpha'][at])
        y[at]=np.add(np.add(left,right,dtype=np.float32),bias,dtype=np.float32)
    y[inputs['accepted']==0]=0; assert np.isfinite(y).all(); return y

def ordered_keys(features,keys,guard):
    logits=np.empty((len(features),24),'<f4')
    for start in range(0,len(features),256):
        x=features[start:start+256].astype('<f8'); lanes=np.zeros((len(x),24,8),'<f8')
        for i in range(1280): lanes[:,:,i%8]+=x[:,i,None]*keys[None,:,i].astype('<f8')
        total=np.zeros((len(x),24),'<f8')
        for j in range(4): total+=lanes[:,:,j]+lanes[:,:,j+4]
        total+=keys[None,:,1280].astype('<f8'); logits[start:start+len(x)]=total.astype('<f4'); guard()
    ids=np.argmax(logits[:,:8],axis=1)*16+np.argmax(logits[:,8:],axis=1)
    p=np.empty(len(features),'<f4')
    for k,row in enumerate(logits):
        probabilities=[]
        for axis in (row[:8],row[8:]):
            winner=int(np.argmax(axis)); maximum=float(axis[winner])
            values=[np.float32(math.exp(float(v)-maximum)) for v in axis]
            denominator=sum(float(v) for v in values)
            probabilities.append(np.float32(float(values[winner])/denominator))
        p[k]=np.multiply(*probabilities,dtype=np.float32)
    return logits,ids.astype('<u4'),p

def loss_gradient(x,theta,prior,ids):
    scores=x@theta.T; grad=np.empty_like(theta); loss=0.
    for start,width,label in ((0,8,ids//16),(8,16,ids%16)):
        z=scores[:,start:start+width]; maximum=np.max(z,axis=1); exp=np.exp(z-maximum[:,None]); den=exp.sum(axis=1)
        loss+=float(np.mean(np.log(den)+maximum-z[np.arange(len(x)),label]))
        probability=exp/den[:,None]; probability[np.arange(len(x)),label]-=1
        grad[start:start+width]=probability.T@x/len(x)
    difference=theta-prior; loss+=.00005*float(np.sum(difference*difference)); grad+=.0001*difference
    return loss,grad

def controls():
    # New scalar ridge fixture and one exact dyadic Adam transition.
    h=np.array([[1.,2.],[2.,1.]],'<f8'); r=np.array([[3.,5.]],'<f8'); gram=h@h.T/2+np.eye(2)/4
    target=r@h.T/2; c=np.linalg.solve(gram,target.T).T
    assert np.max(np.abs(c@gram-target))<1e-14
    g=np.array([[.5,-.25]],'<f8'); m=.1*g; v=.001*g*g
    step=.01*(m/(1-.9))/(np.sqrt(v/(1-.999))+1e-8)
    assert np.max(np.abs(step-np.array([[.01*.5/(.5+1e-8),-.01*.25/(.25+1e-8)]])))<1e-14
    return {'ridge_coefficient':c.tolist(),'ridge_equation_residual_max':float(np.max(np.abs(c@gram-target))),
        'dyadic_Adam_first_step':step.tolist(),'integer_exactness_bound':768*127*32767,'B_I32_bound':512*127*32767}

def energies(y,pred):
    answer=np.empty((len(y),4),'<f8')
    for start in range(0,len(y),256):
        real=y[start:start+256].astype('<f8'); oracle=pred['oracle'][start:start+256].astype('<f8'); coupled=pred['coupled'][start:start+256].astype('<f8')
        answer[start:start+len(real)]=np.column_stack((np.sum(real*real,axis=1),np.sum((oracle-real)**2,axis=1),
            np.sum((coupled-real)**2,axis=1),np.sum((coupled-oracle)**2,axis=1)))
    return answer

def reports(meta,occ,energy,picked):
    def metric(indices):
        count=len(indices)
        if not count: return {'count':0,'ID_correct':0,'ID_fidelity':None,'source_energy':0.,'oracle_error_energy':0.,
            'coupled_error_energy':0.,'decision_error_energy':0.,'oracle_RMS':None,'coupled_RMS':None,'decision_RMS':None}
        sums=np.sum(energy[indices],axis=0); correct=int(np.count_nonzero(picked[indices]==meta[indices,3])); source=float(sums[0])
        out={'count':count,'ID_correct':correct,'ID_fidelity':correct/count,'source_energy':source,
            'oracle_error_energy':float(sums[1]),'coupled_error_energy':float(sums[2]),'decision_error_energy':float(sums[3])}
        for j,label in enumerate(('oracle','coupled','decision'),1): out[label+'_RMS']=math.sqrt(float(sums[j])/source) if source else None
        return out
    dev=(meta[:,4]&5)!=0; counts=np.bincount(meta[dev,3],minlength=128)
    uid_roles=[{'split':label,**metric(np.flatnonzero(mask))} for label,mask in (('development',dev),('consumed_validation',~dev))]
    cells=[{'expert':e,'development_exposure':int(counts[e]),'split':label,**metric(np.flatnonzero(mask&(meta[:,3]==e)))}
        for e in range(128) for label,mask in (('development',dev),('consumed_validation',~dev))]
    rare=[{'split':label,'development_class':name,**metric(np.flatnonzero(mask&category[meta[:,3]]))}
        for name,category in (('0',counts==0),('1..4',(counts>=1)&(counts<=4)),('5..15',(counts>=5)&(counts<=15)),('>=16',counts>=16))
        for label,mask in (('development',dev),('consumed_validation',~dev))]
    def occurrences(mask): return metric(occ[mask,1])
    views=[{'book':book,'role':0 if book<64 else 1 if book<128 else 2,'mode':mode,'accepted':accepted,
        **occurrences((occ[:,4]==book)&(occ[:,6]==mode)&(occ[:,12]==accepted))}
        for book in range(192) for mode in range(2) for accepted in range(2)]
    role_mode=[{'role':role,'mode':mode,**occurrences((occ[:,7]==role)&(occ[:,6]==mode))} for role in range(3) for mode in range(2)]
    exposures=[]
    for role in range(3):
        rows=occ[occ[:,7]==role,1]; source=np.bincount(meta[rows,3],minlength=128); candidate=np.bincount(picked[rows],minlength=128)
        exposures.extend({'role':role,'expert':e,'source':int(source[e]),'candidate':int(candidate[e])} for e in range(128))
    gates={'coupled_weighted_RMS_all_six_role_modes':all(v['count'] and v['coupled_RMS'] is not None and v['coupled_RMS']<=.01 for v in role_mode),
        'ID_fidelity_all_six_role_modes':all(v['count'] and v['ID_fidelity']>=.999 for v in role_mode)}
    return {'uid_roles':uid_roles,'cells':cells,'rare':rare,'views':views,'role_mode':role_mode,'exposures':exposures,'recipe_gates':gates}
