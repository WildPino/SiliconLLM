"""New inquiry checks. Independent retention auditor does not import this module."""
import struct
from pathlib import Path
import numpy as np

SHAPES = [(1,1,1,0),(17,15,1,1),(8,16,1,2),(9,17,1,3),(8,4096,1,4),
          (768,768,1,0),(768,768,29,1),(768,768,256,3),(3072,768,1,4),(768,3072,1,2),(32128,768,1,3)]
MASK = np.array([1,2,4,5,7,8,10,11])

def control_check(path):
    results=[]
    with Path(path).open('rb') as f:
        assert f.read(8)==b'M485CTL1' and struct.unpack('<I',f.read(4))==(11,)
        def array(dtype,shape):
            count=int(np.prod(shape)); data=f.read(count*np.dtype(dtype).itemsize)
            assert len(data)==count*np.dtype(dtype).itemsize
            return np.frombuffer(data,dtype=dtype).reshape(shape)
        for ci,(m,d,q,pattern) in enumerate(SHAPES):
            mp,dp=(m+7)&~7,(d+7)&~7
            assert struct.unpack('<7I',f.read(28))==(m,d,q,mp,dp,pattern,0)
            w=array('i1',(m,d)); scales=array('<f4',(m,));x=array('<f4',(q,d));codes=array('<i2',(q,d));alpha=array('<f4',(q,))
            partial=array('<i4',(q,4,mp)); reconstructed=array('<i8',(q,m));y=array('<f4',(q,m));ref=array('<f4',(q,m))
            ri=np.arange(m,dtype='i4')[:,None];di=np.arange(d,dtype='i4')[None,:];ti=np.arange(q,dtype='i4')[:,None]
            ew=np.full((m,d),-128,dtype='i1') if pattern==4 else ((ri*17+di*31+ci*13)%256-128).astype('i1')
            ex=np.zeros((q,d),dtype='f4')
            if pattern==1:ex=np.where((di+ti)%2,-32767,32767).astype('f4')
            if pattern==2:ex=np.where(di==0,np.float32(32767),((di+ti)%31-15).astype('f4')+np.float32(.5)).astype('f4')
            if pattern==3:ex=(((di*23+ti*11)%1009-504).astype('f4')*np.float32(.03125)).astype('f4')
            if pattern==4:ex=np.full((q,d),32767,dtype='f4')
            assert w.tobytes()==ew.tobytes() and x.tobytes()==ex.tobytes()
            es=np.array([.001,1,1000,.000001],dtype='f4')[np.arange(m)%4];assert scales.tobytes()==es.tobytes()
            maximum=np.max(np.abs(x),axis=1);ea=np.where(maximum==0,np.float32(1),maximum/np.float32(32767)).astype('f4')
            eq=np.clip(np.rint((x/ea[:,None]).astype('f4')),-32767,32767).astype('<i2')
            assert codes.tobytes()==eq.tobytes() and alpha.tobytes()==ea.tobytes()
            u=codes.view('<u2');digits=[(u&127).astype('i8'),((u>>7)&127).astype('i8'),((u>>14).astype('i8')-4*(u>=32768))]
            assert np.array_equal(digits[0]+128*digits[1]+16384*digits[2],codes.astype('i8'))
            for j,digit in enumerate(digits):
                expected=digit@w.astype('i8').T
                assert np.max(np.abs(expected))<= (16256 if j<2 else 256)*dp<2**31
                assert np.array_equal(partial[:,j,:m].astype('i8'),expected)
            assert not np.any(partial[:,:,m:]) and not np.any(partial[:,3,:])
            direct=codes.astype('i8')@w.astype('i8').T
            combined=partial[:,0,:m].astype('i8')+128*partial[:,1,:m].astype('i8')+16384*partial[:,2,:m].astype('i8')
            ey=((direct.astype('f8')*scales.astype('f8')[None,:])*alpha.astype('f8')[:,None]).astype('<f4')
            assert np.array_equal(reconstructed,direct) and np.array_equal(combined,direct) and y.tobytes()==ey.tobytes()==ref.tobytes()
            results.append({'case':ci,'rows':m,'cols':d,'queries':q,'all_partials_integer_and_output_bytes_exact':True})
        assert f.read(1)==b''
    return results

def wire(path,n):
    data=Path(path).read_bytes();assert data[:8]==b'SWR32O01'
    s,t,d,le,ld,v,nr=struct.unpack_from('<7I',data,8);assert (d,le,ld,v)==(768,12,12,32128) and 0<s<=256 and 0<t<=64 and nr==6*(s+t)
    a=(le+2)*s*d;b=t*(ld+2)*d;c=t*v;end=36+4*(a+b+c)
    assert len(data)==end+nr*12
    values=np.frombuffer(data,dtype='<f4',count=a+b+c,offset=36);assert np.isfinite(values).all()
    logits=values[a+b:].reshape(t,v)
    routes=np.frombuffer(data,dtype=[('e','<i4'),('a','<i4'),('p','<f4')],count=nr,offset=end)
    assert ((routes['e']>=0)&(routes['e']<n)).all() and np.isin(routes['a'],[0,1]).all() and ((routes['p']>0)&(routes['p']<=1)).all() and np.isfinite(routes['p']).all()
    return logits.argmax(-1).tolist(),logits

def prediction(logits,case):
    values=logits.astype('f8');shift=values.max(-1,keepdims=True);target=case['target_ids'];truth=np.array(case['masked_spans_ids']).reshape(8)
    nll=np.log(np.exp(values-shift).sum(-1))+shift[:,0]-values[np.arange(14),target];choices=logits.argmax(-1)
    return {'mean_nll':float(nll.mean()),'mean_span_nll':float(nll[MASK].mean()),'correct_span_tokens':int(np.sum(choices[MASK]==truth)),
            'correct_fields':int(np.all((choices[MASK]==truth).reshape(4,2),axis=1).sum()),'per_target_nll':nll.tolist(),'greedy_teacher_forced_ids':choices.tolist()}

def accepted(ids):
    fields=[[] for _ in range(4)];expected=0;current=-1;bad=False;closed=False
    for token in ids:
        if token==1:break
        if token==0:bad=True;continue
        if token>=32000:
            if expected<=4 and token==32099-expected:
                if expected==4:closed=True;current=-1
                else:current=expected
                expected+=1
            else:bad=True
        elif 0<=current<4:fields[current].append(token)
        else:bad=True
    if bad:fields=[[] for _ in range(4)]
    triples=[[(f[i],f[i+1],f[i+2]) for i in range(max(len(f)-2,0))] for f in fields]
    total=sum(map(len,triples));repeated=(total-sum(len(set(v)) for v in triples))/max(total,1)
    healthy=closed and expected==5 and not bad and all(fields) and ids[-1]==32095 and repeated<=.5
    return bool(healthy),len(ids),sum(1<token<32000 for token in ids)

def expected_gpu(s,t):
    return {'calls':60+24*s+85*t,'queries':84*s+85*t,'h2d_bytes':313344*s+316416*t,'d2h_bytes':1253376*s+1767424*t}

def rate_summary(cases,backend,profile):
    time=[];ordinary=[];prose=[];cold=[];loads=[]
    for bi in range(24):
        group=cases[4*bi:4*bi+4];assert len(group)==4
        sec=full=oc=pc=lc=0.
        for c in group:
            rows=c['generation'][str(backend)][str(profile)];warm=np.mean([r['full_generation_seconds'] for r in rows if r['repetition']>=0])
            sec+=float(warm);full+=rows[0]['load_seconds']+rows[0]['full_generation_seconds'];lc+=rows[0]['load_seconds']
            oc+=c['generated_tokens'] if c['healthy_accepted'] else 0;pc+=c['prose_tokens'] if c['healthy_accepted'] else 0
        time.append(sec);ordinary.append(oc);prose.append(pc);cold.append(full);loads.append(lc)
    rng=np.random.default_rng(485485);indices=rng.integers(0,24,size=(10000,24));den=np.array(time)[indices].sum(-1)
    out={'sum_case_mean_seconds':sum(time),'sum_first_request_with_load_seconds':sum(cold),'sum_load_seconds':sum(loads),'bootstrap_unit':'book','draws':10000,'seed':485485}
    for name,num in [('ordinary',ordinary),('prose',prose)]:
        draws=np.array(num)[indices].sum(-1)/den;total=sum(num)
        out[name]={'accepted_ids':total,'warm_rate':total/sum(time),'warm_lower95':float(np.quantile(draws,.05)),
                   'warm_upper95':float(np.quantile(draws,.95)),'first_request_load_charged_rate':total/sum(cold),
                   'amortized_rates_requests_per_process':{str(k):total/(sum(time)+sum(loads)/k) for k in (1,10,100)}}
    return out
