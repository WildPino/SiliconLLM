"""Full stored kept-site audit: independent math plus qualified native replay.

Numerical equations from the qualified full-width audit, explicit shared source.
GPU attention/FFN/head are witnessed by unchanged source/producer, not replayed.
"""
from pathlib import Path
import time
from falcon_auxiliary_s6_transport import tensor_reader


def audit_native(b,sequences,np,guard):
    from bf_activation_backend import load_qualified_backend
    from token_state import arrays,arguments,ptrs
    from fixed_depth_bridge import kept_sites
    import ctypes
    dll,create,destroy,tokens,rows,snapshot=load_qualified_backend(b['activation_closure']['path'])
    last=dll.ts_last;last.argtypes=[ctypes.c_void_p];last.restype=ctypes.c_uint32
    table=np.memmap(b['table']['path'],'u2','r',shape=(65537,6704))
    get,stream=tensor_reader(Path(b['source'])/'model.safetensors',np)
    heads=np.repeat(np.arange(48),64);maxima={};counts=dict(sequence_sites=0,native_positions=0,output_words=0,integer_words=0)
    def decode(x):return (x.astype('u4')<<16).view('f4').astype('f8')
    def bf(x):
        u=x.astype('f4').view('u4');hi=u>>16;lo=u&65535
        return (hi+((lo>32768)|((lo==32768)&((hi&1)!=0)))).astype('u2')
    def test(key,x,y,tol):
        error=float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30));assert np.isfinite(error) and error<=tol,(key,error,tol)
        maxima[key]=max(maxima.get(key,0),error)
    def equal(x,y):assert x.shape==y.shape and x.dtype==y.dtype and np.array_equal(x.view('u1'),y.view('u1'))
    context=None
    try:
        for export in b['exports']:
            site=export['site'];applicable=[s for s in sequences if site in s['active_sites']]
            if not applicable:continue
            with np.load(export['file']['path'],allow_pickle=False) as z:p={k:z[k].copy() for k in z.files}
            w=ptrs([p[k] for k in ('Ws','Vs','Alog','D','gamma')]);convp=ptrs([p[k] for k in ('cw','cb','db')])
            G=get(f'model.layers.{site}.mamba.in_proj.weight') if site else None
            q64=p['q'].astype('f8')
            for seq in applicable:
                guard();item=next(q for q in seq['sites'] if q['site']==site);n=seq['seen'];width=2048 if site==0 else 10800
                assert item['trace']['shape']==[n,width] and item['trace']['bytes']==n*width*2
                trace=np.memmap(item['trace']['path'],'u2','r',shape=(n,width));tags=np.array(seq['consumed_ids'],'u4')
                if site==0:input_rows=np.ascontiguousarray(table[tags]);recorded=trace
                else:
                    H=decode(trace[:,:2048]);expected=bf(decode(bf(H@G.T))*decode(np.array(b['MuP_bits'],'u2')))
                    input_rows=np.ascontiguousarray(trace[:,2048:8752]);recorded=trace[:,8752:]
                    test('contextual_generator',decode(input_rows),decode(expected),b['contextual_row_relative']);del H,expected
                context=create();assert context;out=arrays(n,np)
                for lo in range(0,n,128):
                    guard();hi=min(n,lo+128)
                    assert rows(context,hi-lo,tags[lo:hi].ctypes.data,input_rows[lo:hi].ctypes.data,w,convp,
                        p['q'].ctypes.data,p['scale'].ctypes.data,*arguments(out,lo,hi,np))==0
                state=np.empty((3072,96),'f4');ring=np.empty((3,3584),'u2');seen=np.zeros(1,'u8')
                assert snapshot(context,state.ctypes.data,ring.ctypes.data,seen.ctypes.data)==0 and seen[0]==n and last(context)==tags[-1]
                destroy(context);context=None;equal(out['output'],np.ascontiguousarray(recorded))
                with np.load(item['final']['path']) as z:
                    equal(state,z['state']);equal(ring,z['ring']);equal(seen,z['seen']);equal(np.array([tags[-1]],'u4'),z['last'])
                raw=decode(input_rows);bc=raw[:,3072:6656];pad=np.vstack([np.zeros((3,3584)),bc])
                summed=np.broadcast_to(p['cb'],(n,3584)).copy()
                for j in range(4):summed+=pad[j:j+n]*p['cw'][:,j][None,:]
                cv=decode(bf(summed));activated=decode(bf(cv*np.exp(-np.logaddexp(0,-cv))))
                delta=decode(bf(raw[:,6656:]+p['db']));soft=np.logaddexp(0,delta);soft[delta>20]=delta[delta>20]
                fields=np.concatenate([bf(raw[:,:3072]),bf(activated),bf(soft)],axis=1)
                test('fields',decode(out['fields']),decode(fields),b['numeric']['field_relative'])
                fd=decode(out['fields']);x=fd[:,3072:6144];bb=fd[:,6144:6400];cc=fd[:,6400:6656];dt=fd[:,6656:];gate=fd[:,:3072]
                test('B_projection',out['Bp'].astype('f8'),bb@p['Ws'].astype('f8').T,b['numeric']['projected_relative'])
                test('C_projection',out['Cp'].astype('f8'),cc@p['Vs'].astype('f8').T,b['numeric']['projected_relative'])
                fs=np.zeros((3072,96));latent=np.empty((n,3072));A=-np.exp(p['Alog'].astype('f8'))
                for pos in range(n):
                    if pos%128==0:guard()
                    dd=dt[pos,heads];fs*=np.exp(dd*A[heads])[:,None]
                    fs+=(dd*x[pos])[:,None]*out['Bp'][pos].astype('f8')[None,:]
                    latent[pos]=np.sum(fs*out['Cp'][pos].astype('f8')[None,:],axis=1)+p['D'][heads]*x[pos]
                test('scan',out['latent'].astype('f8'),latent,b['numeric']['scan_relative'])
                test('state',state.astype('f8'),fs,b['numeric']['state_relative'])
                ug=out['latent'].astype('f8')*gate*np.exp(-np.logaddexp(0,-gate))
                norm=bf(ug/np.sqrt(np.mean(ug**2,axis=1,keepdims=True)+1e-5)*p['gamma'])
                test('norm',decode(out['norm']),decode(norm),b['numeric']['controlled_relative'])
                Z=decode(out['norm']);scales=np.max(np.abs(Z),axis=1)/32767
                ratio=np.divide(Z,scales[:,None],out=np.zeros_like(Z),where=scales[:,None]!=0)
                floor=np.floor(ratio);frac=ratio-floor
                ints=np.clip(floor+((frac>.5)|((frac==.5)&(floor.astype('i8')%2!=0))),-32767,32767).astype('i2')
                equal(ints,out['r']);equal(scales,out['a'])
                dots=ints.astype('f8')@q64.T
                assert np.array_equal(dots,np.rint(dots)) and np.max(np.abs(dots))<=36640321536
                equal(dots.astype('i8'),out['dots']);gold=(dots*scales[:,None])*p['scale']
                equal(gold,out['raw']);equal(bf(gold),np.ascontiguousarray(recorded))
                counts['sequence_sites']+=1;counts['native_positions']+=n;counts['output_words']+=recorded.size;counts['integer_words']+=dots.size
                print(dict(site=site,sequence=seq['label'],native_positions=counts['native_positions']),flush=True)
                del out,trace,input_rows,recorded,raw,bc,pad,summed,cv,activated,delta,soft,fields,fd,x,bb,cc,dt,gate,fs,latent,ug,norm,Z,scales,ints,dots,gold
    finally:
        if context is not None:destroy(context)
        stream.close()
    assert counts['sequence_sites']==sum(len(s['active_sites']) for s in sequences)
    assert counts['native_positions']==sum(s['seen']*len(s['active_sites']) for s in sequences)
    return dict(counts=counts,maxima=maxima,all_stored_native_words_and_state_exact=True,independent_F64_and_integer_controls=True)
