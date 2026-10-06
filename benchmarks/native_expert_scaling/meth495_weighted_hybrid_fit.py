"""One fixed weighted-function/control learner and native full-domain evaluation."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import hashlib
import json
from pathlib import Path
import struct
import traceback
from meth495_operations import Context,ROOT,DOC,write

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--binding-sha',required=True); args=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth495_weighted_hybrid_fit',Path(args.out).resolve())
    try:
        b=ctx.admit(args.binding_sha); ctx.binding=b
        import numpy as np
        import threadpoolctl
        threadpoolctl.threadpool_limits(1); assert all(v['num_threads']==1 for v in threadpoolctl.threadpool_info())
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore'); ctx.r['numerical_imports']=True
        import meth495_math as M
        write(ctx.out/'math_controls.json',M.controls()); ctx.phase='compile_candidate_and_new_native_controls'
        binary=ctx.out/'meth495_hybrid_physical.exe'
        ctx.run([b['compiler']['path'],'-O3','-std=c11','-march=x86-64-v3','-fno-fast-math','-ffp-contract=off',
            ROOT/'benchmarks/native_expert_scaling/meth495_hybrid_physical.c','-o',binary],'compile')
        native=[json.loads(s) for s in ctx.run([binary,'controls'],'new_controls')]
        expected=[]
        for p in range(81):
            code=p; total=0
            for x in (32767,-17213,43,-8191): total+=(code%3-1)*x; code//=3
            expected.append(total)
        assert native[0]=={'table81':expected}
        assert native[1]=={'alpha_bits':0x40000000,'codes':[32767,-32767,0,2,2,0,-2,-2]}
        assert native[2]['L_I64']==768*126*32766 and native[2]['B_I32']==512*126*32766
        assert native[2]['RNE']==0 and native[2]['CPU0']==1 and not native[2]['MXCSR']&0x8040
        write(ctx.out/'native_controls.json',native)
        bad=ctx.out/'negative_bank.bin'
        with bad.open('xb') as f:f.write(b'BADMAGIC'+struct.pack('<8I',768,3072,512,128,11,4,8,16))
        negative=ctx.out/'negative_features.bin'
        ctx.run([binary,'features',bad,'unused_source_path',negative],'negative_magic',2)
        assert not negative.exists() and (ctx.out/'negative_magic.stderr').read_bytes().replace(b'\r\n',b'\n')==b'hybrid495_error:wire_magic\n'
        ctx.r['gates']['NEW495_ridge_Adam_LUT_A16_I64_I32_FPU_and_negative_codec_controls']=True
        def wire(desc,magic,width,reserved,count,dtype):
            with Path(desc['path']).open('rb') as f:
                assert f.read(24)==struct.pack('<8sIIQ',magic,width,reserved,count)
                values=np.fromfile(f,dtype,count=count); assert len(values)==count and not f.read(1); return values
        ctx.phase='ALL17540_UID_and19962_occurrence_source_input_control_joins'
        pi=b['fit_inputs']; info=wire(pi['uid'],b'M493U001',180,0,17540,M.UID); meta=info['meta']
        occ=wire(pi['occurrences'],b'M493O001',68,0,19962,np.dtype(('<u4',(17,))))
        assert np.array_equal(meta[:,0],np.arange(17540)) and np.all(meta[:,2]==11) and np.all(meta[:,3]<128) and np.all(meta[:,6]==2)
        dev=(meta[:,4]&5)!=0; val=(meta[:,4]&2)!=0
        assert (int(dev.sum()),int(val.sum()))==(11721,5819) and not np.any(dev&val)
        assert np.array_equal(occ[:,0],np.arange(19962)) and np.all(occ[:,1]<17540)
        assert np.array_equal(occ[:,14],meta[occ[:,1],1]) and np.array_equal(occ[:,11],meta[occ[:,1],3])
        assert np.all(occ[:,3]==128) and np.all(occ[:,8]==11) and np.all(occ[:,12]==1)
        assert np.array_equal(occ[:,15],meta[occ[:,1],12]) and np.array_equal(occ[:,16],meta[occ[:,1],10])
        assert np.array_equal(np.bincount(occ[:,1],minlength=17540),meta[:,7])
        inputs=np.empty(17540,M.INPUT); inputs['expert']=meta[:,3]; inputs['accepted']=1
        with Path(pi['query']['path']).open('rb') as f:
            assert f.read(8)==b'\x93NUMPY\x01\x00'
            for k in range(17540):
                f.seek(448+4732*int(meta[k,11])); q=f.read(4732); assert len(q)==4732
                assert struct.unpack_from('<H',q,14)[0]==int(meta[k,3]) and q[13]==1
                assert q[24:28]==meta[k,10].tobytes() and q[20:24]==meta[k,12].tobytes()
                xb=q[28:3100]; cb=q[3100:4636]
                assert hashlib.sha256(xb).digest()==q[4636:4668]==info[k]['hashes'][:32].tobytes()
                assert hashlib.sha256(cb).digest()==q[4668:4700]==info[k]['hashes'][32:64].tobytes()
                assert hashlib.sha256(cb+q[20:24]).digest()==q[4700:4732]==info[k]['hashes'][64:96].tobytes()
                inputs['x'][k]=np.frombuffer(xb,'<f4'); inputs['q'][k]=np.frombuffer(cb,'<i2'); inputs['alpha'][k]=struct.unpack_from('<f',q,20)[0]
                if k%512==0: ctx.guard()
        q,a=M.quant(inputs['x'],32767); assert q.tobytes()==inputs['q'].tobytes() and a.tobytes()==inputs['alpha'].tobytes(); del q,a
        sourcefile=ctx.out/'inputs.bin'
        with sourcefile.open('xb') as f:f.write(struct.pack('<8sIIQ',b'M495INP1',4620,768,17540)); f.write(inputs.tobytes())
        ctx.r['gates']['ALL_original_UID_occurrence_input_A16_and_control_BYTE_joins']=True
        ctx.phase='native_shared_LUT_features_and_fixed_private_L_for_ALL_UIDs'
        featurefile=ctx.out/'features.bin'; ctx.run([binary,'features',pi['bank']['path'],sourcefile,featurefile],'features')
        features=wire({'path':str(featurefile)},b'M495FEA1',6148,512,17540,M.FEATURE)
        bank=bytearray(Path(pi['bank']['path']).read_bytes()); assert len(bank)==127232136 and bank[:8]==b'M494BNK1'
        t,sigma=M.dictionary(bank)
        for start in range(0,17540,256):
            stop=min(start+256,17540); dot=inputs['q'][start:stop].astype('<f8')@t.astype('<f8').T
            phi=np.abs(((dot*sigma.astype('<f8'))*inputs['alpha'][start:stop,None].astype('<f8')).astype('<f4'))
            qp,ap=M.quant(phi,32767)
            assert phi.tobytes()==features['phi'][start:stop].tobytes() and qp.tobytes()==features['q'][start:stop].tobytes() and ap.tobytes()==features['alpha'][start:stop].tobytes()
            ctx.guard()
        l_sha=[]
        for e in range(128):
            at=np.flatnonzero(meta[:,3]==e); lq,ls,*_=M.expert(bank,e)
            if len(at): assert M.block(inputs['q'][at],lq,ls,inputs['alpha'][at]).tobytes()==features['linear'][at].tobytes()
            pos=M.OFFSET+e*M.EB; l_sha.append(hashlib.sha256(bank[pos:pos+768*768+768*4]).hexdigest())
        ctx.r['gates']['ALL_native_shared_ternary_LUT_A16_features_and_fixed_L_BYTE']=True
        target=wire(pi['targets'],b'M493Y001',3072,768,17540,np.dtype(('<f4',(768,))))
        calibration=json.loads(Path(pi['calibration']['path']).read_bytes()); amps=np.array([v['a_bits'] for v in calibration['cells']],'<u4').view('<f4')
        gpath=Path(pi['geometry']['path'])
        with gpath.open('rb') as f:
            assert f.read(20)==struct.pack('<8sIII',b'M494GEO1',768,512,513)
            f.seek(20+3*768*768*8+513*513*8); kreg=np.fromfile(f,'<f8',513*513).reshape(513,513)
        fitfile=ctx.out/'fitted_coefficients.bin'; fit_summaries=[]; ctx.phase='ONE127_development_only_regularized_B_bias_solves_all128_IDs_retained'
        with Path(pi['C0']['path']).open('rb') as cf,fitfile.open('xb') as fitted:
            assert cf.read(24)==struct.pack('<8sIIQ',b'M494ABS1',1575936,513,128)
            fitted.write(struct.pack('<8sIIQ',b'M495FIT1',1575936,513,128))
            for e in range(128):
                original=np.frombuffer(cf.read(1575936),'<f4').reshape(768,513)
                prior=np.multiply(original,amps[e],dtype=np.float32); at=np.flatnonzero(dev&(meta[:,3]==e)); count=len(at)
                assert calibration['cells'][e]['development']==count
                if count:
                    h=np.ones((count,513),'<f8'); h[:,:512]=features['q'][at].astype('<f8')*features['alpha'][at,None].astype('<f8')
                    residual=target[at].astype('<f8')-features['linear'][at].astype('<f8')
                    gram=h.T@h/count+.01*kreg; rhs=residual.T@h/count+.01*prior.astype('<f8')@kreg
                    ch=np.linalg.cholesky(gram); coef=np.linalg.solve(ch.T,np.linalg.solve(ch,rhs.T)).T.astype('<f4')
                    assert np.isfinite(coef).all(); bq,bs=M.quant(coef[:,:512],127)
                    pos=M.OFFSET+e*M.EB+768*768+768*4
                    bank[pos:pos+768*512]=bq.tobytes(); bank[pos+768*512:pos+768*512+768*4]=bs.tobytes(); bank[pos+768*512+768*4:pos+768*512+768*8]=coef[:,512].tobytes()
                    equation=rhs-coef.astype('<f8')@gram
                    fit_summaries.append({'expert':e,'development':count,'normal_equation_residual_Frobenius':float(np.linalg.norm(equation)),'retained_prior':False})
                    del h,residual,gram,rhs,ch,equation,bq,bs
                else:
                    coef=prior; fit_summaries.append({'expert':e,'development':0,'normal_equation_residual_Frobenius':None,'retained_prior':True})
                fitted.write(coef.tobytes()); ctx.guard()
            assert not cf.read(1)
        assert sum(v['development']>0 for v in fit_summaries)==127
        ctx.r['gates']['ONE127_regularized_function_solves_with_prior_retained_empty_ID_and_fixed_L']=True
        ctx.phase='ONE_fixed128_full_development_Adam_key_updates'
        keyinputs=np.column_stack((inputs['x'],features['phi'])).astype('<f4')
        rms=np.sqrt(np.mean(keyinputs[dev].astype('<f8')**2,axis=0)); rms[rms==0]=1; rms=np.append(rms,1.)
        z=np.ones((11721,1281),'<f8'); z[:,:1280]=keyinputs[dev].astype('<f8')/rms[:1280]
        sourcekeys=np.frombuffer(bank,offset=100392,count=24*1281,dtype='<f4').reshape(24,1281).copy()
        prior_theta=sourcekeys.astype('<f8')*rms; theta=prior_theta.copy(); mom=np.zeros_like(theta); var=np.zeros_like(theta)
        history=ctx.out/'optimizer_history.bin'; trace=[]; labels=meta[dev,3]
        with history.open('xb') as f:
            f.write(struct.pack('<8sIIQ',b'M495OPT1',737856,1281,129))
            for value in (theta,mom,var): f.write(value.tobytes())
            for step in range(1,129):
                loss,grad=M.loss_gradient(z,theta,prior_theta,labels); mom=.9*mom+.1*grad; var=.999*var+.001*grad*grad
                theta-=.01*(mom/(1-.9**step))/(np.sqrt(var/(1-.999**step))+1e-8)
                for value in (theta,mom,var): f.write(value.tobytes())
                assert np.isfinite(theta).all(); trace.append({'step':step,'pre_update_loss':loss,'pre_update_gradient_norm':float(np.linalg.norm(grad))})
                if step%16==0: ctx.log(Adam_updates=step); print(json.dumps({'Adam_updates':step}),flush=True)
                ctx.guard()
        loss,grad=M.loss_gradient(z,theta,prior_theta,labels)
        key_record={'trace':trace,'final_loss':loss,'final_gradient_norm':float(np.linalg.norm(grad)),
            'computed_real_formula_gradient_gap':float(np.sum(grad*grad)/.0002),'optimizer_updates':128,'optimum_certified':False}
        write(ctx.out/'key_fit.json',key_record)
        with (ctx.out/'key_geometry.bin').open('xb') as f:f.write(struct.pack('<8sIIQ',b'M495KEY1',10248,1281,1)); f.write(rms.astype('<f8').tobytes())
        learned_keys=(theta/rms).astype('<f4'); bank[100392:223368]=learned_keys.tobytes(); bank[:8]=b'M495BNK1'
        del z,prior_theta,theta,mom,var,grad,rms,sourcekeys
        for e,sha in enumerate(l_sha):
            pos=M.OFFSET+e*M.EB; assert hashlib.sha256(bank[pos:pos+768*768+768*4]).hexdigest()==sha
        bankfile=ctx.out/'bank.bin'
        with bankfile.open('xb') as f:f.write(bank)
        ctx.r['gates']['ONE128_key_updates_final_only_and_no_dictionary_or_L_change']=True
        ctx.phase='native_ALL17540_coupled_and_oracle_outputs_and_full_Python_BYTE_verification'
        predfile=ctx.out/'predictions.bin'; ctx.run([binary,'predict',bankfile,sourcefile,predfile],'predict')
        pred=wire({'path':str(predfile)},b'M495PRE1',6248,768,17540,M.PRED)
        oracle=M.emit(bank,inputs,features,meta[:,3]); assert oracle.tobytes()==pred['oracle'].tobytes(); del oracle
        logits,ids,p=M.ordered_keys(keyinputs,learned_keys,ctx.guard); del keyinputs
        assert ids.tobytes()==pred['expert'].tobytes() and logits.tobytes()==pred['logits'].tobytes() and p.tobytes()==pred['p'].tobytes()
        coupled=M.emit(bank,inputs,features,ids); assert coupled.tobytes()==pred['coupled'].tobytes(); del coupled,logits,p
        ctx.r['gates']['ALL17540_native_oracle_coupled_ID_logits_and_candidate_mass_BYTE']=True
        energy=M.energies(target,pred); reports=M.reports(meta,occ,energy,ids)
        assert sum(v['count'] for v in reports['views'])==sum(v['count'] for v in reports['role_mode'])==19962
        ctx.r['gates']['ALL_UID_occurrence_128_ID_768_view_rare_and_six_role_mode_denominators']=True
        private_hashes=[hashlib.sha256(bank[M.OFFSET+e*M.EB:M.OFFSET+(e+1)*M.EB]).hexdigest() for e in range(128)]
        passed=all(reports['recipe_gates'].values())
        result=ctx.finish({'decision':'FIXED_HYBRID_LOCAL_RECIPE_PASS_PENDING_INDEPENDENT_AUDIT' if passed else 'FIXED_HYBRID_LOCAL_RECIPE_FAIL_PENDING_INDEPENDENT_AUDIT',
            'reports':reports,'function_fits':fit_summaries,'key_fit':key_record,'unique_physical_private_blocks':len(set(private_hashes)),
            'private_block_sha256':private_hashes,'fixed_L_block_sha256':l_sha,'experts':128,'function_solves':127,'Adam_updates':128,
            'source_FFN_calls':0,'model_calls':0,'scope':'One-bank consumed-domain local candidate; no model quality, SAME rate, DRAM, useful n or goal promotion.'})
        print(json.dumps({'gates':result['gates'],'recipe_gates':reports['recipe_gates'],'resource':ctx.terminal_resources['resource']}),flush=True)
    except BaseException as exc:
        ctx.fail(exc); print(json.dumps({'error':str(exc),'traceback':traceback.format_exc()}),flush=True); raise

if __name__=='__main__': main()
