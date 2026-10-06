"""Independent source joins, unweighted-prior KKT, native products and full accounting."""
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
from meth499_operations import Context,ROOT,DOC,write
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--binding-sha',required=True);ap.add_argument('--raw-sha',required=True);args=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth499_retention',Path(args.out).resolve())
    try:
        b=ctx.admit(args.binding_sha);ctx.binding=b;rp=DOC/'meth499_factor_fit_result.json';completed=rp.exists()
        if not completed:rp=rp.with_suffix('.failure.json')
        assert ctx.digest(rp)==args.raw_sha;raw=json.loads(rp.read_bytes());folder=ROOT/'results/native_expert_scaling/meth499_factor_fit'
        eventpath=ROOT/'results/native_expert_scaling/meth499_windows_terminal.json';event=json.loads(eventpath.read_bytes())
        assert event['query_available'] and event['query_error'] is None and not event['matching_scientific_events'] and event['instances'][0]['pid']==raw['process_instance']['pid']
        inventory=[{'path':str(p),'bytes':p.stat().st_size,'sha256':ctx.digest(p)} for p in sorted(folder.iterdir()) if p.is_file()]
        if not completed:
            assert raw['traceback'] and raw['partial_outputs'];ctx.r['gates']['sole_first_fault_and_ALL_partial_bytes_retained']=True
            result=ctx.finish({'main_completed':False,'raw':{'path':str(rp),'sha256':args.raw_sha},'main_windows_sha256':ctx.digest(eventpath),'retained_main_inventory':inventory,
                'decision':'FIRST_FACTORIZED_PIPELINE_FAULT_RETAINED_NO_ELIGIBILITY','source_FFN_calls':0,'model_calls':0,'native_calls':0,'scope':'First fault only; no complete local/whole quality/rate/goal promotion.'})
            print(json.dumps({'gates':result['gates'],'resource':ctx.terminal_resources['resource']}));return
        assert all(raw['gates'].values()) and raw['binding_sha256']==args.binding_sha
        lookup={Path(v['path']).name:v for v in inventory}
        for v in raw['output_inventory']:assert lookup[Path(v['path']).name]==v
        terminal=json.loads(Path(raw['terminal_resource_path']).read_bytes());assert terminal['raw_sha256']==args.raw_sha
        ctx.r['gates']['complete_main_and_ALL_current_output_terminal_SHA']=True
        import numpy as np
        import threadpoolctl
        threadpoolctl.threadpool_limits(1);assert all(v['num_threads']==1 for v in threadpoolctl.threadpool_info())
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.r['numerical_imports']=True
        control=json.loads((folder/'controls.json').read_bytes());exact=[[Fraction(19,12),Fraction(-2,3),Fraction(1,6)],[Fraction(-5,6),Fraction(5,3),Fraction(4,3)]]
        for row,target,prior in zip(exact,(Fraction(4),Fraction(-2)),((Fraction(1),Fraction(1,2),Fraction(-1)),(Fraction(-1,2),Fraction(1),Fraction(2)))):
            assert 2*row[0]-row[1]+row[2]==target
            d=[v-p for v,p in zip(row,prior)];assert 2*d[0]==-d[1]==d[2]
        assert np.allclose(control['nearest_prior_coefficients'],[[float(v) for v in row] for row in exact],rtol=0,atol=1e-14)
        assert control['I8_codes']==[[127,-127,4,4,-4,-4,0,1]] and control['I8_scale']==[1.]
        native=[json.loads(v) for v in (folder/'new_controls.stdout').read_text().splitlines()]
        x=[32766,-30001,85,-6179];table=[sum(((i//(3**j))%3-1)*x[j] for j in range(4)) for i in range(81)]
        assert native[0]=={'table81':table} and native[1]=={'alpha_bits':0x3f800000,'codes':[32767,-32767,0,2,2,0,-2,-2]}
        assert native[2]['L_I64']==768*127*32767>2**31 and native[2]['B_I32']==512*127*32767<2**31
        assert native[2]['RNE']==0 and not native[2]['MXCSR']&0x8040 and native[2]['CPU0']==1
        inputs_bits=[0x80000000,0x3f800000,0x3f800003,0x7f7fffff,3,5,6,0x80000006];expected_bits=[0x80000000,0x3e800000,0x3e800003,0x7e7fffff,1,1,2,0x80000002]
        def dyadic(bits):
            e=(bits>>23)&255;m=bits&0x7fffff;assert e!=255
            if e:m+=1<<23;power=e-150
            else:power=-149
            v=Fraction(m)*(Fraction(2)**power);return -v if bits>>31 else v
        proof=[]
        for fb,yb in zip(inputs_bits,expected_bits):
            exact_abs=abs(dyadic(fb))/4;mag=yb&0x7fffffff;value=dyadic(mag)
            lo=dyadic(mag-1) if mag else -dyadic(1);hi=dyadic(mag+1)
            assert (fb>>31)==(yb>>31) and (lo+value)/2<=exact_abs<=(value+hi)/2
            if exact_abs in ((lo+value)/2,(value+hi)/2):assert mag%2==0
            proof.append({'input_bits':fb,'expected_bits':yb,'exact_product':str(dyadic(fb)/4)})
        assert control['product_input_bits']==inputs_bits and control['product_expected_bits']==expected_bits
        assert native[3]=={'quarter_product_bits':expected_bits,'rejected_bits':0} and control['native']==native
        assert (folder/'negative_magic.stderr').read_bytes().replace(b'\r\n',b'\n')==b'factor499_error:wire_magic\n' and not (folder/'negative_predictions.bin').exists()
        write(ctx.out/'independent_controls.json',{'exact_nearest_prior':[[str(v) for v in row] for row in exact],'exact_product_rounding':proof,'native':native})
        ctx.r['gates']['NEW499_independent_Fraction_KKT_RNE_products_integer_LUT_and_negative_control']=True
        ut=np.dtype([('m','<u4',(13,)),('hash','u1',(128,))]);it=np.dtype([('e','<u4'),('accept','<u4'),('alpha','<f4'),('x','<f4',(768,)),('q','<i2',(768,))])
        ft=np.dtype([('phi','<f4',(512,)),('q','<i2',(512,)),('alpha','<f4'),('l','<f4',(768,))])
        pt=np.dtype([('id','<u4'),('p','<f4'),('logits','<f4',(24,)),('F','<f4',(768,)),('oracle','<f4',(768,)),('sameID_mass','<f4',(768,)),('choice_source_mass','<f4',(768,)),('coupled','<f4',(768,))])
        bt=np.dtype([('id','<u4'),('p','<f4'),('logits','<f4',(24,)),('oracle','<f4',(768,)),('coupled','<f4',(768,))])
        def read(path,magic,width,reserved,count,dtype):
            with Path(path).open('rb') as f:
                assert f.read(24)==struct.pack('<8sIIQ',magic,width,reserved,count);a=np.fromfile(f,dtype,count=count);assert len(a)==count and not f.read(1);return a
        def data(key,*rest):return read(b['data'][key]['path'],*rest)
        uid=data('uid',b'M493U001',180,0,17540,ut);m=uid['m'];occ=data('occurrences',b'M493O001',68,0,19962,np.dtype(('<u4',(17,))))
        ni=read(folder/'inputs.bin',b'M499INP1',4624,768,17540,np.dtype([('core',it),('source_p','<f4')]));inputs=ni['core'];psrc=ni['source_p']
        feat=data('features',b'M495FEA1',6148,512,17540,ft);y=data('targets',b'M493Y001',3072,768,17540,np.dtype(('<f4',(768,))))
        target=read(folder/'unweighted_targets.bin',b'M499F001',3072,768,17540,np.dtype(('<f4',(768,))))
        pred=read(folder/'predictions.bin',b'M499PRE1',15464,768,17540,pt);baseline=data('baseline_predictions',b'M495PRE1',6248,768,17540,bt)
        for name in ('id','p','logits'):assert pred[name].tobytes()==baseline[name].tobytes()
        del baseline
        with Path(b['data']['inputs']['path']).open('rb') as f:
            assert f.read(24)==struct.pack('<8sIIQ',b'M495INP1',4620,768,17540)
            for start in range(0,17540,193):
                part=np.fromfile(f,it,count=min(193,17540-start));assert inputs[start:start+len(part)].tobytes()==part.tobytes();ctx.guard()
            assert not f.read(1)
        dev=(m[:,4]&5)!=0;val=(m[:,4]&2)!=0
        assert np.array_equal(m[:,0],np.arange(17540)) and np.all(m[:,2]==11) and np.all(m[:,3]<128) and np.all(m[:,6]==2)
        assert (int(dev.sum()),int(val.sum()))==(11721,5819) and not np.any(dev&val) and np.all(dev|val)
        assert np.array_equal(inputs['e'],m[:,3]) and np.all(inputs['accept']==1) and psrc.tobytes()==m[:,10].tobytes()
        assert np.all(np.isfinite(psrc)&(psrc>0)&(psrc<=1)) and np.array_equal(occ[:,0],np.arange(19962)) and np.all(occ[:,1]<17540)
        assert np.all(occ[:,3]==128) and np.all(occ[:,8]==11) and np.all(occ[:,12]==1)
        for oi,mi in ((14,1),(11,3),(15,12),(16,10)):assert np.array_equal(occ[:,oi],m[occ[:,1],mi])
        assert np.array_equal(np.bincount(occ[:,1],minlength=17540),m[:,7])
        qmap=np.load(b['data']['queries']['path'],mmap_mode='r',allow_pickle=False);fmap=np.load(b['data']['references']['path'],mmap_mode='r',allow_pickle=False)
        qdtype=np.dtype([('ledger_record','<u8'),('book','<u2'),('case','u1'),('mode','u1'),('role','u1'),('accepted','u1'),('expert','<u2'),('index','<u4'),
            ('alpha','<f4'),('probability','<f4'),('input','<f4',(768,)),('codes','<i2',(768,)),('input_sha','S32'),('code_sha','S32'),('pair_sha','S32')])
        assert qmap.shape==(19962,) and qmap.dtype==qdtype and fmap.shape==(19962,768) and fmap.dtype==np.dtype('<f4')
        seen=np.zeros(17540,bool);ctx.phase='ALL19962_independent_saved_reference_UID_input_mass_hash_and_role_joins'
        for qid in range(19962):
            q=qmap[qid];qb=q.tobytes();fb=fmap[qid].tobytes();o=occ[qid];k=int(o[1])
            assert int(q['ledger_record'])==int(o[2]) and (int(q['book']),int(q['case']),int(q['mode']),int(q['accepted']),int(q['expert']),int(q['index']))==tuple(map(int,o[[4,5,6,12,11,10]]))
            assert int(q['role'])==(1 if 64<=int(q['book'])<128 else 0) and int(o[7])==int(q['book'])//64
            assert qb[20:24]==inputs['alpha'][k].tobytes() and qb[24:28]==psrc[k].tobytes() and qb[28:3100]==inputs['x'][k].tobytes() and qb[3100:4636]==inputs['q'][k].tobytes()
            hi=hashlib.sha256(qb[28:3100]).digest();hc=hashlib.sha256(qb[3100:4636]).digest();hp=hashlib.sha256(qb[3100:4636]+qb[20:24]).digest();hf=hashlib.sha256(fb).digest()
            assert hi+hc+hp+hf==uid['hash'][k].tobytes() and hi+hc+hp==qb[4636:4732] and fb==target[k].tobytes()
            assert np.isfinite(fmap[qid]).all() and (fmap[qid].astype('<f8')*float(psrc[k])).astype('<f4').tobytes()==y[k].tobytes()
            if not seen[k]:assert qid==int(m[k,11]);seen[k]=True
            if qid%193==0:ctx.guard()
        assert seen.all();del qmap,fmap
        ctx.r['gates']['ALL17540_F_source_and19962_complete_independent_reference_mass_control_joins']=True
        bank=(folder/'bank.bin').read_bytes();assert len(bank)==127232136 and bank[:40]==struct.pack('<8s8I',b'M499BNK1',768,3072,512,128,11,4,8,16)
        with Path(b['data']['bank']['path']).open('rb') as f:original=f.read(223368)
        assert bank[8:223368]==original[8:] and hashlib.sha256(bank[8:223368]).hexdigest()==raw['fixed_dictionary_keys_sha256']
        with Path(b['data']['geometry']['path']).open('rb') as f:
            assert f.read(20)==struct.pack('<8sIII',b'M494GEO1',768,512,513);f.seek(20+3*768*768*8+513*513*8);kreg=np.fromfile(f,'<f8',513*513).reshape(513,513)
        kp=kreg[::-1,::-1].copy();t=np.linalg.cholesky(kp)
        def check(a,x,rhs,tol):
            d=a@x-rhs;bound=tol*np.maximum(1,np.abs(a)@np.abs(x)+np.abs(rhs));ratio=float(np.max(np.abs(d)/bound)) if d.size else 0.
            assert np.isfinite(d).all() and np.isfinite(bound).all() and ratio<=1;return ratio
        factor=check(t,t.T,kp,5e-12);values=np.empty((17540,47),'<f8');seen[:]=False;kkt=[]
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
        def decomposed(f,u64,u32,q,p):
            errors=[u64-f,u32-u64,q-u32,p-q];total=p-f
            diag=[np.einsum('ij,ij->i',v,v) for v in (f,errors[0],u32-f,total,errors[1],errors[2],errors[3])]
            crosses=[np.einsum('ij,ij->i',errors[i],errors[j]) for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))]
            residual=errors[0]+errors[1]+errors[2]+errors[3]-total;vb=3e-12*np.maximum(1,np.abs(f)+np.abs(u64)+np.abs(u32)+np.abs(q)+np.abs(p))
            closure=diag[3]-(diag[1]+diag[4]+diag[5]+diag[6]+2*sum(crosses));eb=1e-10*np.maximum(1,diag[0]+diag[1]+diag[4]+diag[5]+diag[6]+2*sum(np.abs(v) for v in crosses))
            pre=q-f;a=np.column_stack((*diag,*crosses,np.max(np.abs(residual),axis=1),np.max(np.abs(residual)/vb,axis=1),np.abs(closure),np.abs(closure)/eb,np.einsum('ij,ij->i',pre,pre)))
            assert np.isfinite(a).all() and np.max(a[:,[14,16]])<=1;return a
        ctx.phase='ALL128_unweighted_L0_C0_prior_codec_and127_independent_permuted_KKT_cases'
        with Path(b['data']['L0']['path']).open('rb') as lf,Path(b['data']['C0']['path']).open('rb') as sf,(folder/'coefficients_F64.bin').open('rb') as cf,(folder/'coefficients_F32.bin').open('rb') as ff:
            assert lf.read(24)==struct.pack('<8sIIQ',b'M494LIN1',2359296,768,128) and sf.read(24)==struct.pack('<8sIIQ',b'M494ABS1',1575936,513,128)
            assert cf.read(24)==struct.pack('<8sIIQ',b'M499F641',3151872,513,128) and ff.read(24)==struct.pack('<8sIIQ',b'M499F321',1575936,513,128)
            for e in range(128):
                l0=np.frombuffer(lf.read(2359296),'<f4').reshape(768,768);prior=np.frombuffer(sf.read(1575936),'<f4').reshape(768,513).astype('<f8')
                c=np.frombuffer(cf.read(3151872),'<f8').reshape(768,513);cc=np.frombuffer(ff.read(1575936),'<f4').reshape(768,513)
                assert np.isfinite(c).all() and c.astype('<f4').tobytes()==cc.tobytes();lq,ls,bq,bs,bias=parts(e);code,scale=encoder(l0)
                assert code.tobytes()==lq.tobytes() and scale.tobytes()==ls.tobytes()
                code,scale=encoder(cc[:,:512]);assert code.tobytes()==bq.tobytes() and scale.tobytes()==bs.tobytes() and cc[:,512].tobytes()==bias.tobytes()
                assert np.all(lq!=-128) and np.all(bq!=-128) and np.all(ls>0) and np.all(bs>0)
                ids=np.flatnonzero(dev&(m[:,3]==e));assert len(ids)<=308 and raw['solve_cases'][e]['expert']==e and raw['solve_cases'][e]['development']==len(ids) and raw['solve_cases'][e]['retained_prior']==(not len(ids))
                if len(ids):
                    left=physical(inputs['q'][ids],lq,ls,inputs['alpha'][ids]);h=np.ones((len(ids),513),'<f8');h[:,:512]=feat['q'][ids].astype('<f8')*feat['alpha'][ids,None].astype('<f8');h=h[:,::-1].copy()
                    r=target[ids].astype('<f8')-left.astype('<f8');feas=check(h,c[:,::-1].T,r,3e-10)
                    residual=h@c[:,::-1].T-r;rms=math.sqrt(float(np.sum(residual*residual))/float(np.sum(target[ids].astype('<f8')**2)));assert rms<=1e-7
                    zt=np.linalg.solve(t,h.T);white=check(t,zt,h.T,3e-12);q,qr=np.linalg.qr(zt,mode='reduced');qrcheck=check(q,qr,zt,5e-12)
                    orth=float(np.max(np.abs(q.T@q-np.eye(len(ids)))));assert orth<=5e-11
                    dt=t.T@(c-prior)[:,::-1].T;projection=q@(q.T@dt);bound=3e-10*np.maximum(1,np.abs(dt)+np.abs(q)@(np.abs(q.T)@np.abs(dt)))
                    stationary=float(np.max(np.abs(dt-projection)/bound));assert stationary<=1;objective=float(np.sum(dt*dt))
                    assert np.isclose(objective,raw['solve_cases'][e]['prior_metric_displacement_energy'],rtol=1e-8,atol=1e-5)
                    kkt.append({'expert':e,'development':len(ids),'feasibility_ratio':feas,'development_fit_RMS':rms,'stationarity_ratio':stationary,'whitening_ratio':white,'QR_ratio':qrcheck,'orthogonality_maxabs':orth,'prior_metric_energy':objective})
                else:assert c.tobytes()==prior.tobytes();kkt.append({'expert':e,'development':0,'retained_prior':True})
                allids=np.flatnonzero(m[:,3]==e)
                for start in range(0,len(allids),127):
                    at=allids[start:start+127];left=physical(inputs['q'][at],lq,ls,inputs['alpha'][at]);right=physical(feat['q'][at],bq,bs,feat['alpha'][at])
                    output=((left.astype('<f8')+right.astype('<f8')).astype('<f4').astype('<f8')+bias.astype('<f8')[None,:]).astype('<f4');assert output.tobytes()==pred['F'][at].tobytes()
                    assert (output.astype('<f8')*psrc[at].astype('<f8')[:,None]).astype('<f4').tobytes()==pred['oracle'][at].tobytes()
                    assert (output.astype('<f8')*pred['p'][at].astype('<f8')[:,None]).astype('<f4').tobytes()==pred['sameID_mass'][at].tobytes()
                    alpha=feat['alpha'][at].astype('<f8');ph=feat['q'][at].astype('<f8');h=np.ones((len(at),513),'<f8');h[:,:512]=ph*alpha[:,None]
                    u64=left.astype('<f8')+h[:,::-1]@c[:,::-1].T;u32=left.astype('<f8')+h[:,::-1]@cc[:,::-1].astype('<f8').T
                    integer=(ph@bq.astype('<f8').T).astype('<i8');decoded=(left.astype('<f8')+((integer.astype('<f8')*bs.astype('<f8')[None,:])*alpha[:,None]))+bias.astype('<f8')[None,:]
                    fs=target[at].astype('<f8');ys=y[at].astype('<f8');p=psrc[at].astype('<f8')[:,None]
                    decomp=decomposed(fs,u64,u32,decoded,pred['F'][at].astype('<f8'));weighted=decomposed(ys,p*u64,p*u32,p*decoded,pred['oracle'][at].astype('<f8'))
                    ef=pred['oracle'][at].astype('<f8')-ys;ec=pred['choice_source_mass'][at].astype('<f8')-pred['oracle'][at].astype('<f8');em=pred['coupled'][at].astype('<f8')-pred['choice_source_mass'][at].astype('<f8')
                    total=pred['coupled'][at].astype('<f8')-ys;sm=pred['sameID_mass'][at].astype('<f8')-pred['oracle'][at].astype('<f8')
                    diag=[np.einsum('ij,ij->i',v,v) for v in (total,ec,em,sm)];cross=[np.einsum('ij,ij->i',v,w) for v,w in ((ef,ec),(ef,em),(ec,em))]
                    residual=ef+ec+em-total;vb=3e-12*np.maximum(1,np.abs(ys)+np.abs(pred['oracle'][at].astype('<f8'))+np.abs(pred['choice_source_mass'][at].astype('<f8'))+np.abs(pred['coupled'][at].astype('<f8')))
                    closure=diag[0]-(weighted[:,3]+diag[1]+diag[2]+2*sum(cross));eb=1e-10*np.maximum(1,weighted[:,0]+weighted[:,3]+diag[1]+diag[2]+2*sum(np.abs(v) for v in cross))
                    values[at]=np.column_stack((decomp,weighted,*diag,(pred['p'][at].astype('<f8')-psrc[at].astype('<f8'))**2,psrc[at].astype('<f8')**2,*cross,np.max(np.abs(residual)/vb,axis=1),np.abs(closure)/eb))
                    assert np.isfinite(values[at]).all() and np.max(values[at][:,[45,46]])<=1;seen[at]=True;ctx.guard()
                if e%16==0:ctx.log(experts_audited=e+1);print(json.dumps({'experts_audited':e+1}),flush=True)
            assert not lf.read(1) and not sf.read(1) and not cf.read(1) and not ff.read(1)
        assert seen.all();write(ctx.out/'independent_KKT_cases.json',{'prior_factor_ratio':factor,'cases':kkt})
        ctx.r['gates']['ALL128_unweighted_source_prior_encodings_serialization_and127_permuted_metric_KKT']=True
        ctx.phase='ALL17540_independent_choice_source_mass_and_actual_coupled_native_BYTE'
        for e in range(128):
            ids=np.flatnonzero(pred['id']==e);lq,ls,bq,bs,bias=parts(e)
            for start in range(0,len(ids),127):
                at=ids[start:start+127];left=physical(inputs['q'][at],lq,ls,inputs['alpha'][at]);right=physical(feat['q'][at],bq,bs,feat['alpha'][at])
                output=((left.astype('<f8')+right.astype('<f8')).astype('<f4').astype('<f8')+bias.astype('<f8')[None,:]).astype('<f4')
                assert (output.astype('<f8')*psrc[at].astype('<f8')[:,None]).astype('<f4').tobytes()==pred['choice_source_mass'][at].tobytes()
                assert (output.astype('<f8')*pred['p'][at].astype('<f8')[:,None]).astype('<f4').tobytes()==pred['coupled'][at].tobytes();ctx.guard()
        saved=read(folder/'energy_by_uid.bin',b'M499ENG1',376,47,17540,np.dtype(('<f8',(47,))))
        columns=list(range(13))+[17]+list(range(18,31))+list(range(35,45))
        assert np.allclose(values[:,columns],saved[:,columns],rtol=1e-9,atol=1e-5) and np.max(saved[:,[14,16,32,34,45,46]])<=1
        ctx.r['gates']['ALL17540_native_five_vectors_dual_error_levels_choice_mass_47_columns_and_closure']=True
        def metric(ix):
            n=len(ix);s=np.sum(values[ix],axis=0) if n else np.zeros(47);correct=int(np.sum(pred['id'][ix]==m[ix,3]))
            v={'count':n,'ID_correct':correct,'ID_fidelity':correct/n if n else None,'source_energy':float(s[18]),'unweighted_source_energy':float(s[0])}
            for prefix,offset in (('unweighted',0),('weighted',18)):
                for j,name in enumerate(('fit64','fit32','oracle','serialization','parameters','arithmetic'),1):
                    energy=float(s[offset+j]);v[prefix+'_'+name+'_error_energy']=energy;v[prefix+'_'+name+'_RMS']=math.sqrt(energy/s[offset]) if s[offset] else None
                for j,name in enumerate(('fit_serial','fit_parameter','fit_arithmetic','serial_parameter','serial_arithmetic','parameter_arithmetic'),7):v[prefix+'_'+name+'_inner']=float(s[offset+j])
                for j,name in ((13,'identity_maxabs'),(14,'identity_ratio'),(15,'closure_maxabs'),(16,'closure_ratio')):v[prefix+'_'+name]=float(np.max(values[ix,offset+j])) if n else 0.
                v[prefix+'_pre_arithmetic_error_energy']=float(s[offset+17])
            for j,name in enumerate(('coupled','choice','mass','sameID_mass'),36):
                energy=float(s[j]);v[name+'_error_energy']=energy;v[name+'_RMS']=math.sqrt(energy/s[18]) if s[18] else None
            for j,name in enumerate(('fit_choice','fit_mass','choice_mass'),42):v[name+'_inner']=float(s[j])
            v['probability_relative_RMS']=math.sqrt(float(s[40]/s[41])) if s[41] else None
            v['probability_within1pct']=int(np.count_nonzero(np.abs(pred['p'][ix].astype('<f8')/psrc[ix].astype('<f8')-1)<=.01))
            matched=ix[pred['id'][ix]==m[ix,3]];v['same_ID_probability_count']=len(matched);sp=float(np.sum(values[matched,41]))
            v['same_ID_probability_relative_RMS']=math.sqrt(float(np.sum(values[matched,40]))/sp) if sp else None
            v['coupled_identity_ratio']=float(np.max(values[ix,45])) if n else 0.;v['coupled_closure_ratio']=float(np.max(values[ix,46])) if n else 0.;return v
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
        for label in ('uid_roles','cells','rare','views','role_mode'):
            assert len(reports[label])==len(raw['reports'][label])
            for current,old in zip(reports[label],raw['reports'][label]):
                assert current.keys()==old.keys()
                for name,v in current.items():
                    if name.endswith(('identity_maxabs','identity_ratio','closure_maxabs','closure_ratio')):
                        assert v>=0 and old[name]>=0
                        if name.endswith('ratio'):assert v<=1 and old[name]<=1
                    elif isinstance(v,float):assert np.isclose(v,old[name],rtol=1e-9,atol=1e-5),(label,name,v,old[name])
                    else:assert v==old[name]
        assert reports['exposures']==raw['reports']['exposures'] and sum(len(reports[k]) for k in ('uid_roles','cells','rare','views','role_mode'))==1040
        rm=reports['role_mode'];outcomes={name+'_ALL_six_RMS_1pct':all(v['count'] and v[name+'_RMS'] is not None and v[name+'_RMS']<=.01 for v in rm)
            for name in ('unweighted_fit64','unweighted_fit32','unweighted_oracle','weighted_fit64','weighted_fit32','weighted_oracle','coupled')}
        outcomes['ID_ALL_six_999permille']=all(v['count'] and v['ID_fidelity']>=.999 for v in rm)
        outcomes['mass_ALL_six_relative_RMS_1pct']=all(v['count'] and v['probability_relative_RMS'] is not None and v['probability_relative_RMS']<=.01 for v in rm);assert outcomes==raw['reports']['outcomes']
        if not(outcomes['unweighted_fit64_ALL_six_RMS_1pct'] and outcomes['weighted_fit64_ALL_six_RMS_1pct']):decision='NEXT_SOURCE_FEATURE_INFORMATION'
        elif not(outcomes['unweighted_fit32_ALL_six_RMS_1pct'] and outcomes['weighted_fit32_ALL_six_RMS_1pct']):decision='NEXT_COEFFICIENT_SERIALIZATION'
        elif not(outcomes['unweighted_oracle_ALL_six_RMS_1pct'] and outcomes['weighted_oracle_ALL_six_RMS_1pct']):decision='NEXT_PHYSICAL_COEFFICIENT_ENCODING'
        elif not(outcomes['coupled_ALL_six_RMS_1pct'] and outcomes['ID_ALL_six_999permille'] and outcomes['mass_ALL_six_relative_RMS_1pct']):decision='NEXT_ROUTING_AND_MASS'
        else:decision='LOCAL_FACTORIZED_TRANSFER_ELIGIBLE_FOR_COMPOSED_CONTEXT'
        assert decision==raw['decision'] and [v['label'] for v in raw['commands']]==['compile','new_controls','negative_magic','new_candidate_predict']
        assert [v['returncode'] for v in raw['commands']]==[0,0,2,0]
        assert raw['readout_solves']==127 and raw['coefficient_cases']==128 and raw['optimizer_updates']==raw['source_FFN_calls']==raw['model_calls']==0 and raw['native_calls']==3 and raw['compiler_calls']==1
        ctx.exact(raw['native_binary']);assert raw['physical_bank_bytes']==127232136 and raw['same_logical_weight_bytes_per_12_bank_token']==14587008 and raw['additional_mass_multiplications_per_selected_bank']==768
        ctx.r['gates']['ALL1040_reports384_exposures_frozen_outcomes_decision_actual_commands_and_cost']=True
        result=ctx.finish({'main_completed':True,'raw':{'path':str(rp),'sha256':args.raw_sha},'main_windows_sha256':ctx.digest(eventpath),'retained_main_inventory':inventory,
            'UIDs_audited':17540,'occurrences_audited':19962,'experts_audited':128,'readout_solves_audited':127,'metric_groups_audited':1040,'exposure_groups_audited':384,
            'outcomes':outcomes,'decision':decision,'source_FFN_calls':0,'model_calls':0,'native_calls':0,
            'scope':'Independent complete local factorized candidate audit. No fresh/model quality, SAME rate/DRAM/useful n/family or goal promotion.'})
        print(json.dumps({'gates':result['gates'],'outcomes':outcomes,'decision':decision,'resource':ctx.terminal_resources['resource']}),flush=True)
    except BaseException as exc:
        ctx.fail(exc);print(json.dumps({'error':str(exc),'traceback':traceback.format_exc()}),flush=True);raise
if __name__=='__main__':main()
