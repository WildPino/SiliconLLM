"""Independent full prior audit: no import of prior compiler or math module."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import traceback
from meth494_operations import Context,ROOT,DOC,write

def positive_rational_f32(value):
    n,d=value.numerator,value.denominator; assert n>0
    e=n.bit_length()-d.bit_length()
    if (n < d*(1<<e)) if e>=0 else (n*(1<<-e)<d): e-=1
    assert -126<=e<=127
    if e<=23: num,den=n*(1<<(23-e)),d
    else: num,den=n,d*(1<<(e-23))
    q,r=divmod(num,den); q+=int(2*r>den or (2*r==den and q%2==1))
    if q==1<<24: q>>=1; e+=1
    return ((e+127)<<23)+(q-(1<<23))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--binding-sha',required=True)
    ap.add_argument('--raw-sha',required=True); args=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth494_retention',Path(args.out).resolve(),900,512<<20)
    try:
        b=ctx.admit(args.binding_sha); ctx.binding=b; rawpath=DOC/'meth494_hybrid_prior_result.json'; completed=rawpath.exists()
        if not completed: rawpath=rawpath.with_suffix('.failure.json')
        assert ctx.digest(rawpath)==args.raw_sha; raw=json.loads(rawpath.read_bytes())
        eventpath=ROOT/'results/native_expert_scaling/meth494_windows_terminal.json'; event=json.loads(eventpath.read_bytes())
        assert event['query_available'] and event['query_error'] is None and not event['matching_scientific_events']
        assert event['instances'][0]['pid']==raw['process_instance']['pid']
        inventory=[]; folder=ROOT/'results/native_expert_scaling/meth494_hybrid_prior'
        for p in sorted(folder.iterdir()):
            if p.is_file(): inventory.append({'path':str(p),'bytes':p.stat().st_size,'sha256':ctx.digest(p)})
        if not completed:
            assert raw['traceback'] and raw['partial_outputs']; ctx.r['gates']['sole_first_main_fault_and_all_partial_bytes_retained']=True
            result=ctx.finish({'main_completed':False,'raw':{'path':str(rawpath),'sha256':args.raw_sha},
                'main_windows_sha256':ctx.digest(eventpath),'retained_main_inventory':inventory,
                'decision':'FIRST_PRIOR_COMPILER_FAULT_RETAINED_NO_PRIOR_ELIGIBILITY','source_FFN_calls':0,'model_calls':0,
                'scope':'Original fault only; no complete numerical prior or goal promotion.'})
            print(json.dumps({'gates':result['gates'],'resource':ctx.terminal_resources['resource']})); return
        assert all(raw['gates'].values()) and raw['binding_sha256']==args.binding_sha
        lookup={Path(v['path']).name:v for v in inventory}
        for v in raw['output_inventory']:
            assert lookup[Path(v['path']).name]==v
        terminal=json.loads(Path(raw['terminal_resource_path']).read_bytes()); assert terminal['raw_sha256']==args.raw_sha
        ctx.r['gates']['main_complete_terminal_and_ALL_current_output_SHA']=True
        import numpy as np
        import threadpoolctl
        threadpoolctl.threadpool_limits(1); assert all(v['num_threads']==1 for v in threadpoolctl.threadpool_info())
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore'); ctx.r['numerical_imports']=True
        def kernel(x,y):
            u=np.linalg.norm(x,axis=1); v=np.linalg.norm(y,axis=1); den=np.outer(u,v)
            cos=np.zeros((len(x),len(y)),'<f8'); np.divide(x@y.T,den,out=cos,where=den>0)
            assert np.isfinite(cos).all() and np.max(np.abs(cos))<=1+2e-12
            angle=np.arccos(np.clip(cos,-1,1))
            return (2/math.pi)*den*(np.sin(angle)+(math.pi/2-angle)*np.cos(angle))
        def quantize(value):
            maximum=np.max(np.abs(value),axis=1)
            scale=(maximum.astype('<f8')/127).astype('<f4'); scale[maximum==0]=1
            assert np.all(scale>0)
            ratio=(value.astype('<f8')/scale[:,None].astype('<f8')).astype('<f4').astype('<f8')
            low=np.floor(ratio); rem=ratio-low
            rounded=low+((rem>.5)|((rem==.5)&(low%2!=0)))
            return np.maximum(-127,np.minimum(127,rounded)).astype('<i1'),scale
        # Fresh independently evaluated analytical corners, not old controls.
        corner=np.array([[1,0],[-1,0],[0,1],[0,0],[3,0]],'<f8'); kc=kernel(corner,corner)
        assert max(abs(kc[0,0]-1),abs(kc[0,1]-1),abs(kc[0,2]-2/math.pi),abs(kc[4,0]-3))<3e-15
        assert np.count_nonzero(kc[3])==0
        q,sq=quantize(np.array([[127,1.5,2.5,-1.5,-2.5,0],[0]*6],'<f4'))
        assert q.tolist()==[[127,2,2,-2,-2,0],[0]*6] and sq.tolist()==[1,1]
        assert positive_rational_f32(Fraction(1,2))==0x3f000000
        assert positive_rational_f32(Fraction((1<<24)+1,1<<24))==0x3f800000
        assert positive_rational_f32(Fraction((1<<24)+3,1<<24))==0x3f800002
        wi=[[1,-2,0,1],[0,1,2,-1],[-1,0,1,2]]; wo=[[Fraction(1,4),Fraction(-1,4),Fraction(1,2)],
            [Fraction(-1,2),Fraction(1,4),Fraction(1,4)]]; xs=[[0,1,-1,2],[2,-1,1,0],[-2,1,-1,0]]
        exact=[]
        for x in xs:
            z=[sum(a*b for a,b in zip(row,x)) for row in wi]
            out=[sum(v*max(t,0) for v,t in zip(row,z)) for row in wo]
            split=[sum(v*(t+abs(t))/2 for v,t in zip(row,z)) for row in wo]; assert out==split
            exact.append([float(v) for v in out])
        controls=json.loads((folder/'controls.json').read_bytes())
        assert np.asarray(exact).T.tolist()==controls['dyadic_identity_values']
        assert np.allclose(kc,controls['absolute_kernel_values'],rtol=0,atol=3e-15)
        assert controls['packed_group_codes']==[40,0,80] and controls['quant_codes']==q.tolist()
        write(ctx.out/'independent_controls.json',{'exact_rational_identity':exact,'angle_kernel':kc.tolist(),'quant_codes':q.tolist(),
            'exact_rational_mean_half_even_controls':True})
        ctx.r['gates']['NEW494_independent_rational_angle_and_encoding_controls']=True
        ub=Path(b['prior_inputs']['uid']['path']).read_bytes(); assert ub[:24]==struct.pack('<8sIIQ',b'M493U001',180,0,17540)
        info=np.frombuffer(ub[24:],np.dtype([('m','<u4',(13,)),('hash','u1',(128,))])); m=info['m']
        assert np.array_equal(m[:,0],np.arange(17540)) and np.all(m[:,2]==11)
        development=np.flatnonzero(m[:,4]&5!=0); assert len(development)==11721 and np.count_nonzero(m[:,4]&2)==5819
        sums=[Fraction(0) for _ in range(128)]; counts=[0]*128; cov=np.zeros((768,768),'<f8'); block=[]
        ctx.phase='independent11721_development_covariance_and_exact_rational128_means'
        with Path(b['prior_inputs']['query']['path']).open('rb') as f:
            for number,k in enumerate(development):
                f.seek(448+4732*int(m[k,11])); qb=f.read(4732); assert len(qb)==4732
                assert struct.unpack_from('<H',qb,14)[0]==int(m[k,3]) and qb[24:28]==m[k,10].tobytes() and qb[20:24]==m[k,12].tobytes()
                xb=qb[28:3100]; assert hashlib.sha256(xb).digest()==info[k]['hash'][:32].tobytes()
                p=struct.unpack_from('<f',qb,24)[0]; e=int(m[k,3]); assert 1/128<=p<=1
                sums[e]+=Fraction.from_float(p); counts[e]+=1; x=np.frombuffer(xb,'<f4').astype('<f8'); assert np.isfinite(x).all(); block.append(x)
                if len(block)==127 or number+1==11721:
                    mat=np.stack(block); cov+=mat.T@mat; block.clear(); ctx.guard()
        expected_s=cov/11721; expected_s=.5*(expected_s+expected_s.T)
        global_bits=positive_rational_f32(sum(sums,Fraction(0))/11721)
        calibration=json.loads((folder/'calibration.json').read_bytes()); assert calibration['global_p_bits']==global_bits
        amps=np.empty(128,'<f4')
        for e,cell in enumerate(calibration['cells']):
            expected=positive_rational_f32(sums[e]/counts[e]) if counts[e] else global_bits
            assert cell=={'expert':e,'development':counts[e],'a_bits':expected,'unsupported_global_amplitude_fallback':counts[e]==0}
            amps[e]=np.array([expected],'<u4').view('<f4')[0]
        assert len(calibration['cells'])==128 and counts.count(0)==1
        with (folder/'geometry.bin').open('rb') as f:
            assert f.read(20)==struct.pack('<8sIII',b'M494GEO1',768,512,513)
            def array(shape):
                count=math.prod(shape); data=f.read(count*8); assert len(data)==count*8; return np.frombuffer(data,'<f8').reshape(shape).copy()
            s=array((768,768)); c=array((768,768)); chol=array((768,768)); k=array((513,513)); kr=array((513,513)); ev=array((513,513)); eig=array((513,))
            assert not f.read(1)
        assert np.allclose(s,expected_s,rtol=3e-12,atol=3e-12)
        ridge=.001*np.trace(s)/768; assert np.allclose(c,s+ridge*np.eye(768),rtol=3e-12,atol=3e-12)
        assert np.count_nonzero(np.triu(chol,1))==0 and np.all(np.diag(chol)>0)
        assert np.linalg.norm(c-chol@chol.T)<=3e-12*max(1,np.linalg.norm(c))
        ctx.r['gates']['development_only_covariance_and_ALL128_exact_F32_mean_calibrations']=True
        ext={(v['expert'],v['kind'],v['component']):v for v in b['source_extents']}
        def source(e):
            qi=np.frombuffer(ctx.extent(ext[e,'wi','code']),'<i1').reshape(3072,768)
            si=np.frombuffer(ctx.extent(ext[e,'wi','scale']),'<f4')
            qo=np.frombuffer(ctx.extent(ext[e,'wo','code']),'<i1').reshape(768,3072)
            so=np.frombuffer(ctx.extent(ext[e,'wo','scale']),'<f4')
            assert np.isfinite(si).all() and np.isfinite(so).all() and np.all(si>=0) and np.all(so>=0) and np.all(qi!=-128) and np.all(qo!=-128)
            return qi,si,qo,so
        selection=json.loads((folder/'selection.json').read_bytes()); assert len(selection)==512
        bp=(folder/'bank_prior.bin').open('rb'); assert bp.read(40)==struct.pack('<8s8I',b'M494BNK1',768,3072,512,128,11,4,8,16)
        packed=np.frombuffer(bp.read(98304),'u1').reshape(512,192); assert np.all(packed<=80)
        trits=np.empty((512,192,4),'<i1'); remainder=packed.astype('<u2')
        for digit in range(4): trits[:,:,digit]=remainder%3-1; remainder//=3
        assert np.count_nonzero(remainder)==0; t=trits.reshape(512,768)
        sigma=np.frombuffer(bp.read(2048),'<f4'); assert np.isfinite(sigma).all() and np.all(sigma>0)
        keys=np.frombuffer(bp.read(122976),'<f4').reshape(24,1281)
        router=np.frombuffer(ctx.extent(ext[None,'router','code']),'<f4').astype('<f8').reshape(8,16,768)
        expected_keys=np.zeros((24,1281),'<f4'); grand=np.sum(router,axis=(0,1))/128
        expected_keys[:8,:768]=(np.sum(router,axis=1)/16-.5*grand).astype('<f4')
        expected_keys[8:,:768]=(np.sum(router,axis=0)/8-.5*grand).astype('<f4')
        assert expected_keys.tobytes()==keys.tobytes()
        a=t.astype('<f8')*sigma[:,None]; wa=a@chol; expected_k=np.ones((513,513),'<f8'); expected_k[:512,:512]=kernel(wa,wa)
        means=np.linalg.norm(wa,axis=1)*math.sqrt(2/math.pi); expected_k[512,:512]=means; expected_k[:512,512]=means
        assert np.allclose(k,expected_k,rtol=3e-12,atol=3e-12)
        lam=.001*np.trace(k)/513; assert np.allclose(kr,k+lam*np.eye(513),rtol=3e-12,atol=3e-12)
        orth=float(np.linalg.norm(ev.T@ev-np.eye(513))); eigenres=float(np.linalg.norm(kr-(ev*eig)@ev.T))
        assert orth<=1e-10 and np.all(eig[1:]>=eig[:-1]) and eig[0]>=.99*lam
        lower=float(eig[0]*(1-orth)-eigenres-1e-10*max(1,np.linalg.norm(kr))); assert lower>=.98*lam
        ctx.r['gates']['independent_angle_Gram_Cholesky_eigen_residual_and_positive_lower_bound']=True
        lp=(folder/'L0_prior.bin').open('rb'); cp=(folder/'C0_prior.bin').open('rb')
        assert lp.read(24)==struct.pack('<8sIIQ',b'M494LIN1',2359296,768,128)
        assert cp.read(24)==struct.pack('<8sIIQ',b'M494ABS1',1575936,513,128)
        ctx.phase='ALL128_independent_source_selection_linear_normal_equations_and_physical_BYTE_quantization'
        results=[]
        for e in range(128):
            qi,si,qo,so=source(e); w=qi.astype('<f8')*si[:,None]; v=qo.astype('<f8')*so[:,None]
            score=.5*np.einsum('ij,ij->i',w,w)*np.einsum('ij,ij->j',v,v)
            # Dominance uses tolerant score comparison; exact ties use hidden index.
            ids=[selection[e*4+j]['hidden_row'] for j in range(4)]
            assert len(set(ids))==4 and all(0<=h<3072 for h in ids)
            exact_order_score=.5*np.sum(w*w,axis=1)*np.sum(v*v,axis=0)
            assert ids==np.lexsort((np.arange(3072),-exact_order_score))[:4].tolist()
            for j,h in enumerate(ids):
                item=selection[e*4+j]; assert item['dictionary_row']==e*4+j and item['source_expert']==e
                assert abs(item['importance_f64']-score[h])<=2e-12*max(1,abs(score[h]))
                excluded=np.ones(3072,bool); excluded[ids[:j+1]]=False
                assert np.all(score[excluded]<=score[h]+2e-12*max(1,abs(score[h])))
                integer=qi[h].astype('<i8'); norm2=int(integer@integer); target_t=np.zeros(768,'<i1')
                if norm2 and si[h]!=0:
                    at=sorted(range(768),key=lambda z:(-abs(int(integer[z])),z))[:256]
                    target_t[at]=np.sign(integer[at]); nz=int(np.count_nonzero(target_t))
                    target_sigma=np.float32(sum(abs(int(integer[z])) for z in at)/(nz*math.sqrt(norm2)))
                else: nz=0; target_sigma=np.float32(1)
                assert target_t.tobytes()==t[e*4+j].tobytes() and target_sigma.tobytes()==sigma[e*4+j].tobytes()
                assert item['nonzero']==nz and item['sigma_bits']==int(target_sigma.view('<u4'))
            linear=np.frombuffer(lp.read(2359296),'<f4').reshape(768,768)
            coef=np.frombuffer(cp.read(1575936),'<f4').reshape(768,513)
            assert np.isfinite(linear).all() and np.isfinite(coef).all()
            recomputed_l=(.5*so[:,None])*(qo.astype('<f8')@w)
            linear_error=float(np.max(np.abs(linear.astype('<f8')-recomputed_l)/(4*2**-24*np.maximum(1,np.abs(recomputed_l)))))
            assert linear_error<=1
            ww=(qi.astype('<f8')@chol)*si[:,None]
            cross=np.empty((3072,513),'<f8'); cross[:,:512]=kernel(ww,wa); cross[:,512]=np.linalg.norm(ww,axis=1)*math.sqrt(2/math.pi)
            target=(.5*so[:,None])*(qo.astype('<f8')@cross); f64=coef.astype('<f8')
            residual=target-f64@kr
            casting=(np.abs(f64)*(2**-24/(1-2**-24))+2**-150)@np.abs(kr)
            arithmetic=1e-7*np.maximum(1,np.abs(target)+np.abs(f64)@np.abs(kr))
            residual_ratio=float(np.max(np.abs(residual)/(casting+arithmetic))); assert residual_ratio<=1
            objective_gap=float(np.sum(residual*residual)/lower)
            wl=(linear.astype('<f8')*float(amps[e])).astype('<f4'); wc=(f64*float(amps[e])).astype('<f4')
            lq,ls=quantize(wl); bq,bs=quantize(wc[:,:512])
            for z in (lq,ls,bq,bs,wc[:,512]): assert bp.read(z.nbytes)==z.tobytes(),('physical_BYTE',e,z.shape)
            results.append({'expert':e,'linear_rounding_envelope_ratio':linear_error,'equation_casting_arithmetic_envelope_ratio':residual_ratio,
                'computed_Gram_objective_gap_upper':objective_gap})
            del qi,si,qo,so,w,v,score,exact_order_score,linear,coef,recomputed_l,ww,cross,target,f64,residual,casting,arithmetic,wl,wc,lq,ls,bq,bs
            if e%8==0: ctx.log(experts_audited=e+1); print(json.dumps({'experts_audited':e+1}),flush=True)
            ctx.guard()
        assert not lp.read(1) and not cp.read(1) and not bp.read(1)
        for f in (lp,cp,bp): f.close()
        assert [lookup[n]['bytes'] for n in ('bank_prior.bin','L0_prior.bin','C0_prior.bin','geometry.bin')]==[127232136,301989912,201719832,20475956]
        ctx.r['gates']['ALL128_source_dictionary_full_linear_and_Gaussian_normal_equation_envelopes']=True
        ctx.r['gates']['ALL_physical_weights_scales_bias_packed_trits_and_centroid_keys_BYTE']=True
        cost=json.loads((folder/'cost_contract.json').read_bytes()); private=768*768+768*4+768*512+768*4+768*4
        assert cost['private_bytes_per_expert_per_bank']==private and cost['private_bytes_slope_all12']==12*private
        shared=512*768//4+512*4; keys_bytes=24*(768+512+1)*4
        assert cost['shared_A_bytes_all12']==12*shared and cost['initial_keys_bytes_all12']==12*keys_bytes
        assert cost['logical_weight_read_bytes_per_token_all12']==12*(private+shared+keys_bytes)
        assert cost['logical_LUT_read_bytes_all12']==12*512*192*4
        assert cost['logical_LUT_write_bytes_all12']==12*192*81*4
        assert cost['per_bank_integer_MAC']==768*(768+512) and cost['per_bank_LUT_lookups']==512*192
        assert cost['per_bank_key_F64_MAC']==24*(768+512) and cost['original_FFN_MAC']==2*768*3072
        assert not cost['DRAM_or_rate_measurement'] and not cost['core_head_state_quantizer_and_other_traffic_included']
        ctx.r['gates']['physical_wire_sizes_and_explicit_logical_cost_independently_recounted']=True
        result=ctx.finish({'main_completed':True,'raw':{'path':str(rawpath),'sha256':args.raw_sha},
            'main_windows_sha256':ctx.digest(eventpath),'retained_main_inventory':inventory,'experts_audited':128,
            'dictionary_rows_audited':512,'development_inputs_audited':11721,'private_physical_bytes_audited':128*private,
            'eigen_orthogonality_Frobenius':orth,'eigen_reconstruction_Frobenius':eigenres,'computed_Gram_eigen_lower':lower,
            'prior_audits':results,'source_FFN_calls':0,'model_calls':0,'gradient_updates':0,'Gaussian_projection_solves':0,
            'decision':'HYBRID_GAUSSIAN_PRIOR_AND_LOGICAL_COST_INDEPENDENTLY_ADMITTED',
            'scope':'Numerical projection envelopes for chosen computed Gaussian Gram, not formal real interval certificates or model quality/speed/goal completion.'})
        print(json.dumps({'gates':result['gates'],'resource':ctx.terminal_resources['resource']}),flush=True)
    except BaseException as exc:
        ctx.fail(exc); print(json.dumps({'error':str(exc),'traceback':traceback.format_exc()}),flush=True); raise

if __name__=='__main__': main()
