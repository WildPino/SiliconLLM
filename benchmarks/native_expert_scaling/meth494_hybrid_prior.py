"""One weight-informed Gaussian projection; no source FFN/model or target fit."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import traceback
from meth494_operations import Context,ROOT,DOC,write

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--binding-sha',required=True)
    args=ap.parse_args(); ctx=Context(ROOT/'results/native_expert_scaling/meth494_hybrid_prior',Path(args.out).resolve(),600,512<<20)
    try:
        b=ctx.admit(args.binding_sha); ctx.binding=b
        import numpy as np
        import threadpoolctl
        threadpoolctl.threadpool_limits(1); assert all(v['num_threads']==1 for v in threadpoolctl.threadpool_info())
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore'); ctx.r['numerical_imports']=True
        from meth494_math import absolute_kernel,ternary_row,packed_ternary,quant_rows,controls
        write(ctx.out/'controls.json',controls()); ctx.r['gates']['new494_analytic_and_encoding_controls']=True
        ext={(v['expert'],v['kind'],v['component']):v for v in b['source_extents']}
        def weights(e):
            answer=[]
            for k,shape in (('wi',(3072,768)),('wo',(768,3072))):
                q=np.frombuffer(ctx.extent(ext[e,k,'code']),'<i1').reshape(shape)
                s=np.frombuffer(ctx.extent(ext[e,k,'scale']),'<f4')
                assert np.isfinite(s).all() and np.all(s>=0) and np.all(q!=-128)
                answer.extend((q,s))
            return answer
        ctx.phase='all128_weight_only_four_rows_and_shared_ternary_dictionary'
        t=np.zeros((512,768),'<i1'); sigma=np.empty(512,'<f4'); selection=[]
        for e in range(128):
            qi,si,qo,so=weights(e); w=qi.astype('<f8')*si[:,None]; v=qo.astype('<f8')*so[:,None]
            importance=.5*np.sum(w*w,axis=1)*np.sum(v*v,axis=0)
            assert np.isfinite(importance).all()
            chosen=np.lexsort((np.arange(3072),-importance))[:4]
            for j,h in enumerate(chosen):
                at=e*4+j; t[at],sigma[at],nz=ternary_row(qi[h],si[h])
                selection.append({'dictionary_row':at,'source_expert':e,'hidden_row':int(h),
                    'importance_f64':float(importance[h]),'nonzero':nz,'sigma_bits':int(sigma[at].view('<u4'))})
            del qi,si,qo,so,w,v,importance; ctx.guard()
        write(ctx.out/'selection.json',selection); packed=packed_ternary(t); a=t.astype('<f8')*sigma[:,None]
        ctx.r['gates']['ALL128_weight_derived_512_dictionary_rows_no_response_selection']=True
        ctx.phase='only11721_canonical_development_covariance_and_p_calibration'
        ip=Path(b['prior_inputs']['uid']['path']); ub=ip.read_bytes()
        assert ub[:24]==struct.pack('<8sIIQ',b'M493U001',180,0,17540)
        info=np.frombuffer(ub[24:],np.dtype([('meta','<u4',(13,)),('hash','u1',(128,))]))
        meta=info['meta']; assert np.array_equal(meta[:,0],np.arange(17540)) and np.all(meta[:,2]==11)
        development=np.flatnonzero(meta[:,4]&5!=0); assert len(development)==11721 and np.count_nonzero(meta[:,4]&2)==5819
        covariance=np.zeros((768,768),'<f8'); sums=np.zeros(128,'<f8'); counts=np.zeros(128,'<u4')
        p_all=0.; xblock=[]; dev_count=0
        with Path(b['prior_inputs']['query']['path']).open('rb') as qf:
            assert qf.read(8)==b'\x93NUMPY\x01\x00'
            for k in development:
                qf.seek(448+4732*int(meta[k,11])); raw=qf.read(4732); assert len(raw)==4732
                assert struct.unpack_from('<H',raw,14)[0]==int(meta[k,3])
                assert raw[24:28]==meta[k,10].tobytes() and raw[20:24]==meta[k,12].tobytes()
                xb=raw[28:3100]; assert hashlib.sha256(xb).digest()==info[k]['hash'][:32].tobytes()
                x=np.frombuffer(xb,'<f4'); p=struct.unpack_from('<f',raw,24)[0]
                assert np.isfinite(x).all() and 1/128<=p<=1
                xblock.append(x.astype('<f8')); e=int(meta[k,3]); sums[e]+=p; counts[e]+=1; p_all+=p; dev_count+=1
                if len(xblock)==256 or dev_count==11721:
                    block=np.stack(xblock); covariance+=block.T@block; xblock.clear(); ctx.guard()
        s=covariance/11721; s=(s+s.T)*.5; ridge=1e-3*float(np.trace(s))/768
        assert ridge>0 and np.isfinite(s).all(); c=s+ridge*np.eye(768); chol=np.linalg.cholesky(c)
        global_p=np.float32(p_all/11721); calibration=[]; amplitudes=np.empty(128,'<f4')
        for e in range(128):
            amplitudes[e]=np.float32(sums[e]/int(counts[e])) if counts[e] else global_p
            calibration.append({'expert':e,'development':int(counts[e]),'a_bits':int(amplitudes[e].view('<u4')),
                'unsupported_global_amplitude_fallback':not bool(counts[e])})
        write(ctx.out/'calibration.json',{'global_p_bits':int(global_p.view('<u4')),'cells':calibration})
        assert int(counts.sum())==11721 and np.count_nonzero(counts==0)==1
        ctx.r['gates']['ONLY11721_UID_development_geometry_and_all128_calibration']=True
        ctx.phase='Gaussian_shared_even_Gram_and_eigenvalues'
        white_a=a@chol; k=np.ones((513,513),'<f8'); k[:512,:512],clipped=absolute_kernel(white_a,white_a)
        means=np.sqrt(np.sum(white_a*white_a,axis=1))*np.sqrt(2/np.pi); k[:512,512]=means; k[512,:512]=means
        k=(k+k.T)*.5; lam=1e-3*float(np.trace(k))/513; kr=k+lam*np.eye(513)
        eig,ev=np.linalg.eigh(kr); assert lam>0 and eig[0]>=.99*lam
        ck=np.linalg.cholesky(kr)
        gp=ctx.out/'geometry.bin'
        with gp.open('xb') as f:
            f.write(struct.pack('<8sIII',b'M494GEO1',768,512,513))
            for z in (s,c,chol,k,kr,ev,eig): f.write(z.astype('<f8').tobytes())
        router=np.frombuffer(ctx.extent(ext[None,'router','code']),'<f4').astype('<f8').reshape(8,16,768)
        assert np.isfinite(router).all(); grand=router.mean(axis=(0,1)); keys=np.zeros((24,1281),'<f4')
        keys[:8,:768]=(router.mean(axis=1)-.5*grand).astype('<f4')
        keys[8:,:768]=(router.mean(axis=0)-.5*grand).astype('<f4')
        ctx.phase='ALL128_full_private_linear_and_Gaussian_even_solves_and_physical_export'
        lp=ctx.out/'L0_prior.bin'; cp=ctx.out/'C0_prior.bin'; bp=ctx.out/'bank_prior.bin'
        summaries=[]
        with lp.open('xb') as lf,cp.open('xb') as cf,bp.open('xb') as bf:
            lf.write(struct.pack('<8sIIQ',b'M494LIN1',2359296,768,128))
            cf.write(struct.pack('<8sIIQ',b'M494ABS1',1575936,513,128))
            bf.write(struct.pack('<8s8I',b'M494BNK1',768,3072,512,128,11,4,8,16))
            bf.write(packed.tobytes()); bf.write(sigma.tobytes()); bf.write(keys.tobytes())
            for e in range(128):
                qi,si,qo,so=weights(e); w=qi.astype('<f8')*si[:,None]; v=qo.astype('<f8')*so[:,None]
                linear=(.5*(v@w)).astype('<f4'); white_w=w@chol
                cross=np.empty((3072,513),'<f8'); cross[:,:512],extra=absolute_kernel(white_w,white_a); clipped+=extra
                cross[:,512]=np.sqrt(np.sum(white_w*white_w,axis=1))*np.sqrt(2/np.pi)
                target=.5*(v@cross); coef=np.linalg.solve(ck.T,np.linalg.solve(ck,target.T)).T.astype('<f4')
                assert np.isfinite(linear).all() and np.isfinite(coef).all()
                lf.write(linear.tobytes()); cf.write(coef.tobytes())
                weighted_l=np.multiply(linear,amplitudes[e],dtype=np.float32)
                weighted_c=np.multiply(coef,amplitudes[e],dtype=np.float32)
                lq,ls=quant_rows(weighted_l); bq,bs=quant_rows(weighted_c[:,:512])
                for z in (lq,ls,bq,bs,weighted_c[:,512]): bf.write(z.tobytes())
                residual=target-coef.astype('<f8')@kr
                summaries.append({'expert':e,'prior_equation_residual_Frobenius':float(np.linalg.norm(residual)),
                    'L0_Frobenius':float(np.linalg.norm(linear.astype('<f8'))),'C0_Frobenius':float(np.linalg.norm(coef.astype('<f8')))})
                del qi,si,qo,so,w,v,white_w,cross,target,linear,coef,weighted_l,weighted_c,lq,ls,bq,bs,residual
                if e%8==0: ctx.log(experts_completed=e+1); print(json.dumps({'experts_completed':e+1}),flush=True)
                ctx.guard()
        assert (bp.stat().st_size,lp.stat().st_size,cp.stat().st_size,gp.stat().st_size)==(127232136,301989912,201719832,20475956)
        cost={'banks':12,'private_bytes_per_expert_per_bank':992256,'private_bytes_slope_all12':11907072,
            'shared_A_bytes_all12':1204224,'initial_keys_bytes_all12':1475712,'logical_weight_read_bytes_per_token_all12':14587008,
            'logical_LUT_read_bytes_all12':4718592,'logical_LUT_write_bytes_all12':746496,
            'per_bank_integer_MAC':983040,'per_bank_LUT_lookups':98304,'per_bank_key_F64_MAC':30720,
            'per_bank_exponentials':24,'per_bank_normalizers':2,'original_FFN_MAC':4718592,
            'DRAM_or_rate_measurement':False,'core_head_state_quantizer_and_other_traffic_included':False}
        write(ctx.out/'cost_contract.json',cost)
        ctx.r['gates']['ALL128_Gaussian_priors_serialized_full_rank_linear_and_physical_byte_counts']=True
        ctx.r['gates']['complete_explicit_logical_cost_not_DRAM_or_speed']=True
        result=ctx.finish({'decision':'HYBRID_GAUSSIAN_PRIOR_COMPILED_PENDING_INDEPENDENT_AUDIT',
            'experts':128,'dictionary_rows':512,'development_inputs':11721,'covariance_ridge':ridge,'kernel_ridge':lam,
            'kernel_eigmin':float(eig[0]),'kernel_eigmax':float(eig[-1]),'correlation_clips':clipped,
            'priors':summaries,'Gaussian_projection_solves':128,'gradient_updates':0,'source_FFN_calls':0,'model_calls':0,
            'scope':'Untrained one-bank Gaussian prior; no local or model quality, DRAM, speed, n usefulness or goal promotion.'})
        print(json.dumps({'gates':result['gates'],'resource':ctx.terminal_resources['resource']}),flush=True)
    except BaseException as exc:
        ctx.fail(exc); print(json.dumps({'error':str(exc),'traceback':traceback.format_exc()}),flush=True); raise

if __name__=='__main__': main()
