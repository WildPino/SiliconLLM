"""One complete unweighted-prior learner and new factorized native candidate."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct
import traceback
from meth499_operations import Context,ROOT,DOC,write
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--binding-sha',required=True);args=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth499_factor_fit',Path(args.out).resolve())
    try:
        b=ctx.admit(args.binding_sha);ctx.binding=b
        import numpy as np
        import threadpoolctl
        threadpoolctl.threadpool_limits(1);assert all(v['num_threads']==1 for v in threadpoolctl.threadpool_info())
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.r['numerical_imports']=True
        import meth499_math as M
        binary=ctx.out/'meth499_factor_physical.exe';ctx.phase='compile_NEW499_variable_mass_native_contract'
        ctx.run([b['compiler']['path'],'-O3','-std=c11','-march=x86-64-v3','-fno-fast-math','-ffp-contract=off',
            ROOT/'benchmarks/native_expert_scaling/meth499_factor_physical.c','-o',binary],'compile')
        binary_desc={'path':str(binary),'bytes':binary.stat().st_size,'sha256':ctx.digest(binary)}
        allowed={Path(v['path']).resolve() for v in b['catalog']}|{binary.resolve()}
        assert {Path(v).resolve() for v in ctx.r['commands'][0]['observed_modules']}<=allowed
        rows=[json.loads(v) for v in ctx.run([binary,'controls'],'new_controls')]
        assert len(rows)==4;write(ctx.out/'controls.json',M.controls(rows))
        bad=ctx.out/'negative_bank.bin'
        with bad.open('xb') as f:f.write(b'BADMAGIC'+struct.pack('<8I',768,3072,512,128,11,4,8,16))
        negative=ctx.out/'negative_predictions.bin';ctx.run([binary,'predict',bad,'unused_source_path',negative],'negative_magic',2)
        assert not negative.exists() and (ctx.out/'negative_magic.stderr').read_bytes().replace(b'\r\n',b'\n')==b'factor499_error:wire_magic\n'
        ctx.r['gates']['NEW499_QR_I8_LUT_A16_full_integer_RNE_product_and_negative_wire_controls']=True
        def read(key,magic,width,reserved,count,dtype):
            with Path(b['data'][key]['path']).open('rb') as f:
                assert f.read(24)==struct.pack('<8sIIQ',magic,width,reserved,count)
                a=np.fromfile(f,dtype,count=count);assert len(a)==count and not f.read(1);return a
        uid=read('uid',b'M493U001',180,0,17540,M.UID);m=uid['m'];occ=read('occurrences',b'M493O001',68,0,19962,np.dtype(('<u4',(17,))))
        inputs=read('inputs',b'M495INP1',4620,768,17540,M.INPUT);feat=read('features',b'M495FEA1',6148,512,17540,M.FEATURE)
        y=read('targets',b'M493Y001',3072,768,17540,np.dtype(('<f4',(768,))))
        baseline=read('baseline_predictions',b'M495PRE1',6248,768,17540,M.BASELINE)
        dev=(m[:,4]&5)!=0;val=(m[:,4]&2)!=0;psrc=m[:,10].copy().view('<f4')
        assert np.array_equal(m[:,0],np.arange(17540)) and np.all(m[:,2]==11) and np.all(m[:,3]<128) and np.all(m[:,6]==2)
        assert (int(dev.sum()),int(val.sum()))==(11721,5819) and not np.any(dev&val) and np.all(dev|val)
        assert np.array_equal(inputs['e'],m[:,3]) and np.all(inputs['accept']==1) and np.all(np.isfinite(psrc)&(psrc>0)&(psrc<=1))
        assert np.array_equal(occ[:,0],np.arange(19962)) and np.all(occ[:,1]<17540) and np.all(occ[:,3]==128) and np.all(occ[:,8]==11) and np.all(occ[:,12]==1)
        for oi,mi in ((14,1),(11,3),(15,12),(16,10)):assert np.array_equal(occ[:,oi],m[occ[:,1],mi])
        assert np.array_equal(np.bincount(occ[:,1],minlength=17540),m[:,7])
        ctx.phase='ALL19962_saved_unweighted_reference_and_original_UID_occurrence_joins'
        target=np.empty((17540,768),'<f4');seen=np.zeros(17540,bool)
        with Path(b['data']['queries']['path']).open('rb') as qf,Path(b['data']['references']['path']).open('rb') as ff:
            assert qf.read(8)==ff.read(8)==b'\x93NUMPY\x01\x00'
            qh=ast.literal_eval(qf.read(struct.unpack('<H',qf.read(2))[0]).decode('ascii'));fh=ast.literal_eval(ff.read(struct.unpack('<H',ff.read(2))[0]).decode('ascii'))
            expected_descr=np.lib.format.dtype_to_descr(M.QUERY)[:12]+[(name,'|S32') for name in ('input_sha','code_sha','pair_sha')]
            assert qh=={'descr':expected_descr,'fortran_order':False,'shape':(19962,)} and fh=={'descr':'<f4','fortran_order':False,'shape':(19962,768)}
            assert qf.tell()==448 and ff.tell()==128
            for qid in range(19962):
                qb=qf.read(4732);fb=ff.read(3072);assert len(qb)==4732 and len(fb)==3072;q=np.frombuffer(qb,M.QUERY)[0];o=occ[qid];k=int(o[1])
                assert int(q['ledger_record'])==int(o[2]) and (int(q['book']),int(q['case']),int(q['mode']),int(q['accepted']),int(q['expert']),int(q['index']))==tuple(map(int,o[[4,5,6,12,11,10]]))
                assert int(q['role'])==(1 if 64<=int(q['book'])<128 else 0) and int(o[7])==int(q['book'])//64
                assert q['probability'].tobytes()==psrc[k].tobytes() and q['alpha'].tobytes()==inputs['alpha'][k].tobytes()
                xb=q['input'].tobytes();cb=q['codes'].tobytes();ab=q['alpha'].tobytes()
                assert xb==inputs['x'][k].tobytes() and cb==inputs['q'][k].tobytes()
                hashes=hashlib.sha256(xb).digest()+hashlib.sha256(cb).digest()+hashlib.sha256(cb+ab).digest()+hashlib.sha256(fb).digest()
                assert hashes==uid['hash'][k].tobytes() and hashes[:96]==q['input_sha'].tobytes()+q['code_sha'].tobytes()+q['pair_sha'].tobytes()
                f=np.frombuffer(fb,'<f4');assert np.isfinite(f).all() and np.multiply(f,psrc[k],dtype=np.float32).tobytes()==y[k].tobytes()
                if seen[k]:assert target[k].tobytes()==fb
                else:assert qid==int(m[k,11]);target[k]=f;seen[k]=True
                if qid%256==0:ctx.guard()
            assert not qf.read(1) and not ff.read(1)
        assert seen.all()
        with (ctx.out/'unweighted_targets.bin').open('xb') as f:f.write(struct.pack('<8sIIQ',b'M499F001',3072,768,17540));f.write(target.tobytes())
        inputpath=ctx.out/'inputs.bin'
        with inputpath.open('xb') as f:
            f.write(struct.pack('<8sIIQ',b'M499INP1',4624,768,17540))
            for start in range(0,17540,256):
                part=np.empty(min(256,17540-start),M.NATIVE_INPUT);part['core']=inputs[start:start+len(part)];part['source_p']=psrc[start:start+len(part)];f.write(part.tobytes());ctx.guard()
        ctx.r['gates']['ALL17540_unweighted_F_19962_reference_hash_input_mass_control_role_joins_no_rounded_division']=True
        bank=bytearray(Path(b['data']['bank']['path']).read_bytes());assert len(bank)==127232136 and bank[:40]==struct.pack('<8s8I',b'M495BNK1',768,3072,512,128,11,4,8,16)
        prefix=hashlib.sha256(bank[8:223368]).hexdigest();bank[:8]=b'M499BNK1'
        with Path(b['data']['geometry']['path']).open('rb') as f:
            assert f.read(20)==struct.pack('<8sIII',b'M494GEO1',768,512,513);f.seek(20+3*768*768*8+513*513*8);kreg=np.fromfile(f,'<f8',513*513).reshape(513,513)
        t=np.linalg.cholesky(kreg);factor=M.envelope(t,t.T,kreg,5e-12);summaries=[];ctx.phase='ALL128_unweighted_source_L0_encodings_and127_development_only_prior_QR_solves'
        with Path(b['data']['L0']['path']).open('rb') as lf,Path(b['data']['C0']['path']).open('rb') as cf,(ctx.out/'coefficients_F64.bin').open('xb') as f64,(ctx.out/'coefficients_F32.bin').open('xb') as f32:
            assert lf.read(24)==struct.pack('<8sIIQ',b'M494LIN1',2359296,768,128) and cf.read(24)==struct.pack('<8sIIQ',b'M494ABS1',1575936,513,128)
            f64.write(struct.pack('<8sIIQ',b'M499F641',3151872,513,128));f32.write(struct.pack('<8sIIQ',b'M499F321',1575936,513,128))
            for e in range(128):
                l0=np.frombuffer(lf.read(2359296),'<f4').reshape(768,768);prior=np.frombuffer(cf.read(1575936),'<f4').reshape(768,513).astype('<f8')
                code,scale=M.quant(l0);pos=223368+992256*e;bank[pos:pos+589824]=code.tobytes();bank[pos+589824:pos+592896]=scale.tobytes()
                ids=np.flatnonzero(dev&(m[:,3]==e));assert len(ids)<=308;lq,ls,*_=M.parts(bank,e);left=M.block(inputs['q'][ids],lq,ls,inputs['alpha'][ids])
                h=np.ones((len(ids),513),'<f8');h[:,:512]=feat['q'][ids].astype('<f8')*feat['alpha'][ids,None].astype('<f8')
                ft=target[ids].astype('<f8');c,record=M.minimum(h,ft-left.astype('<f8'),prior,t,float(np.sum(ft*ft)),ctx.guard);c32=c.astype('<f4');assert np.isfinite(c32).all()
                bcode,bscale=M.quant(c32[:,:512]);at=pos+592896;bank[at:at+393216]=bcode.tobytes();bank[at+393216:at+396288]=bscale.tobytes();bank[at+396288:at+399360]=c32[:,512].tobytes()
                f64.write(c.tobytes());f32.write(c32.tobytes());summaries.append({'expert':e,**record});ctx.guard()
                if e%16==0:ctx.log(experts_fitted=e+1);print(json.dumps({'experts_fitted':e+1}),flush=True)
            assert not lf.read(1) and not cf.read(1)
        assert sum(not v['retained_prior'] for v in summaries)==127 and hashlib.sha256(bank[8:223368]).hexdigest()==prefix
        bankpath=ctx.out/'bank.bin'
        with bankpath.open('xb') as f:f.write(bank)
        ctx.r['gates']['ALL128_unweighted_L0_and_C0_prior_cases127_KKT_solves_C64_C32_same_A_keys']=True
        ctx.phase='ONE_NEW499_factorized_native_prediction_ALL17540_UIDs';ctx.exact(binary_desc)
        pp=ctx.out/'predictions.bin';ctx.run([binary,'predict',bankpath,inputpath,pp],'new_candidate_predict')
        observed={Path(v).resolve() for v in ctx.r['commands'][-1]['observed_modules']};assert observed and observed<=allowed
        for v in b['native_modules']:ctx.exact(v)
        ctx.exact(binary_desc)
        for command in ctx.r['commands']:assert {Path(v).resolve() for v in command['observed_modules']}<=allowed
        with pp.open('rb') as f:
            assert f.read(24)==struct.pack('<8sIIQ',b'M499PRE1',15464,768,17540);pred=np.fromfile(f,M.PRED,count=17540);assert len(pred)==17540 and not f.read(1)
        for field in ('id','p','logits'):assert pred[field].tobytes()==baseline[field].tobytes()
        del baseline
        ctx.r['gates']['NEW499_actual_compiler_native_modules_and_unchanged_ID_logit_probability_BYTE']=True
        assert M.emit(bank,inputs,feat,m[:,3],ctx.guard).tobytes()==pred['F'].tobytes()
        assert np.multiply(pred['F'],psrc[:,None],dtype=np.float32).tobytes()==pred['oracle'].tobytes()
        assert np.multiply(pred['F'],pred['p'][:,None],dtype=np.float32).tobytes()==pred['sameID_mass'].tobytes()
        fc=M.emit(bank,inputs,feat,pred['id'],ctx.guard)
        assert np.multiply(fc,psrc[:,None],dtype=np.float32).tobytes()==pred['choice_source_mass'].tobytes()
        assert np.multiply(fc,pred['p'][:,None],dtype=np.float32).tobytes()==pred['coupled'].tobytes();del fc
        ctx.r['gates']['ALL17540_unweighted_F_source_mass_sameID_mass_choice_and_coupled_native_BYTE']=True
        data=np.empty((17540,47),'<f8');seen[:]=False;ctx.phase='ALL17540_unweighted_weighted_four_levels_and_choice_mass_cross_terms'
        with (ctx.out/'coefficients_F64.bin').open('rb') as f64,(ctx.out/'coefficients_F32.bin').open('rb') as f32:
            assert f64.read(24)==struct.pack('<8sIIQ',b'M499F641',3151872,513,128) and f32.read(24)==struct.pack('<8sIIQ',b'M499F321',1575936,513,128)
            for e in range(128):
                c=np.frombuffer(f64.read(3151872),'<f8').reshape(768,513);cc=np.frombuffer(f32.read(1575936),'<f4').reshape(768,513);assert c.astype('<f4').tobytes()==cc.tobytes()
                lq,ls,bq,bs,bias=M.parts(bank,e);ids=np.flatnonzero(m[:,3]==e)
                for start in range(0,len(ids),256):
                    at=ids[start:start+256];l=M.block(inputs['q'][at],lq,ls,inputs['alpha'][at]).astype('<f8')
                    alpha=feat['alpha'][at].astype('<f8');q=feat['q'][at].astype('<f8');h=np.ones((len(at),513),'<f8');h[:,:512]=q*alpha[:,None]
                    u64=l+h@c.T;u32=l+h@cc.astype('<f8').T;integer=q@bq.astype('<f8').T
                    decoded=(l+((integer*bs.astype('<f8')[None,:])*alpha[:,None]))+bias.astype('<f8')[None,:]
                    data[at]=M.energies(target[at].astype('<f8'),y[at].astype('<f8'),u64,u32,decoded,pred[at],psrc[at].astype('<f8'));seen[at]=True;ctx.guard()
            assert not f64.read(1) and not f32.read(1)
        assert seen.all()
        with (ctx.out/'energy_by_uid.bin').open('xb') as f:f.write(struct.pack('<8sIIQ',b'M499ENG1',376,47,17540));f.write(data.tobytes())
        ctx.r['gates']['ALL17540_dual_function_error_levels_and_nonorthogonal_choice_mass_closure']=True
        reports=M.reports(m,occ,data,pred['id'],psrc,pred['p']);old=json.loads((DOC/'meth495_weighted_hybrid_fit_result.json').read_bytes())['reports']
        for label in ('uid_roles','cells','rare','views','role_mode'):
            assert len(reports[label])==len(old[label])
            for current,previous in zip(reports[label],old[label]):
                for name in ('count','ID_correct','ID_fidelity'):assert current[name]==previous[name]
                assert np.isclose(current['source_energy'],previous['source_energy'],rtol=1e-10,atol=1e-8)
        assert reports['exposures']==old['exposures'] and sum(len(reports[v]) for v in ('uid_roles','cells','rare','views','role_mode'))==1040
        assert sum(v['count'] for v in reports['views'])==sum(v['count'] for v in reports['role_mode'])==19962
        ctx.r['gates']['ALL1040_reports384_exposures_and_original_source_controls_ID_denominators']=True
        result=ctx.finish({'reports':reports,'solve_cases':summaries,'prior_factor_envelope_ratio':factor,'fixed_dictionary_keys_sha256':prefix,'native_binary':binary_desc,
            'UIDs':17540,'occurrences':19962,'readout_solves':127,'coefficient_cases':128,'optimizer_updates':0,'source_FFN_calls':0,'model_calls':0,'native_calls':3,'compiler_calls':1,
            'physical_bank_bytes':127232136,'same_logical_weight_bytes_per_12_bank_token':14587008,'additional_mass_multiplications_per_selected_bank':768,
            'decision':M.decision(reports['outcomes']),'scope':'Complete original/consumed one-bank factorized candidate; pending independent audit. No fresh/model quality, SAME rate, DRAM/useful n/family or goal promotion.'})
        print(json.dumps({'gates':result['gates'],'outcomes':reports['outcomes'],'decision':result['decision'],'resource':ctx.terminal_resources['resource']}),flush=True)
    except BaseException as exc:
        ctx.fail(exc);print(json.dumps({'error':str(exc),'traceback':traceback.format_exc()}),flush=True);raise
if __name__=='__main__':main()
