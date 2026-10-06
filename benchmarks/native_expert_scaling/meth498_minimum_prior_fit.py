"""One full nearest-prior development learner/export/new native prediction."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import hashlib
import json
from pathlib import Path
import struct
import traceback
from meth498_operations import Context,ROOT,DOC,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--binding-sha',required=True);args=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth498_minimum_prior_fit',Path(args.out).resolve())
    try:
        b=ctx.admit(args.binding_sha);ctx.binding=b
        import numpy as np
        import threadpoolctl
        threadpoolctl.threadpool_limits(1);assert all(v['num_threads']==1 for v in threadpoolctl.threadpool_info())
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.r['numerical_imports']=True
        import meth498_math as M
        write(ctx.out/'controls.json',M.controls());ctx.r['gates']['NEW498_minimum_prior_exact_fixture_and_I8_half_even_controls']=True
        def read(key,magic,width,reserved,count,dtype):
            with Path(b['data'][key]['path']).open('rb') as f:
                assert f.read(24)==struct.pack('<8sIIQ',magic,width,reserved,count)
                a=np.fromfile(f,dtype,count=count);assert len(a)==count and not f.read(1);return a
        uid=read('uid',b'M493U001',180,0,17540,M.UID);m=uid['m'];occ=read('occurrences',b'M493O001',68,0,19962,np.dtype(('<u4',(17,))))
        inputs=read('inputs',b'M495INP1',4620,768,17540,M.INPUT);feat=read('features',b'M495FEA1',6148,512,17540,M.FEATURE)
        y=read('targets',b'M493Y001',3072,768,17540,np.dtype(('<f4',(768,))))
        baseline=read('baseline_predictions',b'M495PRE1',6248,768,17540,M.PRED)
        dev=(m[:,4]&5)!=0;val=(m[:,4]&2)!=0
        assert np.array_equal(m[:,0],np.arange(17540)) and np.all(m[:,2]==11) and np.all(m[:,3]<128) and np.all(m[:,6]==2)
        assert (int(dev.sum()),int(val.sum()))==(11721,5819) and not np.any(dev&val) and np.all(dev|val)
        assert np.array_equal(inputs['e'],m[:,3]) and np.all(inputs['accept']==1)
        assert np.array_equal(occ[:,0],np.arange(19962)) and np.all(occ[:,1]<17540) and np.all(occ[:,3]==128) and np.all(occ[:,8]==11) and np.all(occ[:,12]==1)
        for oi,mi in ((14,1),(11,3),(15,12),(16,10)):assert np.array_equal(occ[:,oi],m[occ[:,1],mi])
        assert np.array_equal(np.bincount(occ[:,1],minlength=17540),m[:,7])
        ctx.r['gates']['ALL17540_UID_19962_original_occurrences_and_fixed_input_feature_domains']=True
        bank=bytearray(Path(b['data']['bank']['path']).read_bytes());assert len(bank)==127232136 and bank[:40]==struct.pack('<8s8I',b'M495BNK1',768,3072,512,128,11,4,8,16)
        prefix=hashlib.sha256(bank[:223368]).hexdigest();lhash=[hashlib.sha256(bank[223368+992256*e:223368+992256*e+592896]).hexdigest() for e in range(128)]
        empty_bytes=bytes(bank[223368:223368+992256])
        calibration=json.loads(Path(b['data']['calibration']['path']).read_bytes());amps=np.array([v['a_bits'] for v in calibration['cells']],'<u4').view('<f4')
        with Path(b['data']['geometry']['path']).open('rb') as f:
            assert f.read(20)==struct.pack('<8sIII',b'M494GEO1',768,512,513)
            f.seek(20+3*768*768*8+513*513*8);k=np.fromfile(f,'<f8',513*513).reshape(513,513)
        t=np.linalg.cholesky(k);factor=M.envelope(t,t.T,k,5e-12);ctx.r['gates']['SAME_positive_prior_metric_factor_and_fixed_calibration']=True
        summaries=[];ctx.phase='ONE127_development_only_QR_minimum_prior_solves_ALL128_cases'
        with Path(b['data']['C0']['path']).open('rb') as source,(ctx.out/'coefficients_F64.bin').open('xb') as f64,(ctx.out/'coefficients_F32.bin').open('xb') as f32:
            assert source.read(24)==struct.pack('<8sIIQ',b'M494ABS1',1575936,513,128)
            f64.write(struct.pack('<8sIIQ',b'M498F641',3151872,513,128));f32.write(struct.pack('<8sIIQ',b'M498F321',1575936,513,128))
            for e in range(128):
                cb=source.read(1575936);assert len(cb)==1575936;prior=np.multiply(np.frombuffer(cb,'<f4').reshape(768,513),amps[e],dtype=np.float32).astype('<f8')
                ids=np.flatnonzero(dev&(m[:,3]==e));assert calibration['cells'][e]['development']==len(ids)<=308
                h=np.ones((len(ids),513),'<f8');h[:,:512]=feat['q'][ids].astype('<f8')*feat['alpha'][ids,None].astype('<f8')
                target=y[ids].astype('<f8');r=target-feat['l'][ids].astype('<f8')
                c,record=M.minimum(h,r,prior,t,float(np.sum(target*target)),ctx.guard);c32=c.astype('<f4');assert np.isfinite(c32).all()
                code,scale=M.quant(c32[:,:512]);pos=223368+992256*e+592896
                bank[pos:pos+393216]=code.tobytes();bank[pos+393216:pos+396288]=scale.tobytes();bank[pos+396288:pos+399360]=c32[:,512].tobytes()
                f64.write(c.tobytes());f32.write(c32.tobytes());summaries.append({'expert':e,**record});ctx.guard()
                if e%16==0:ctx.log(experts_fitted=e+1);print(json.dumps({'experts_fitted':e+1}),flush=True)
            assert not source.read(1)
        assert sum(not v['retained_prior'] for v in summaries)==127 and bytes(bank[223368:223368+992256])==empty_bytes
        assert hashlib.sha256(bank[:223368]).hexdigest()==prefix and lhash==[hashlib.sha256(bank[223368+992256*e:223368+992256*e+592896]).hexdigest() for e in range(128)]
        bankpath=ctx.out/'bank.bin'
        with bankpath.open('xb') as f:f.write(bank)
        ctx.r['gates']['ALL127_equality_stationarity_solves_C64_C32_and128_exports_same_A_L_keys_empty_prior']=True
        ctx.phase='ONE_changed_artifact_native_prediction_ALL17540_original_UIDs'
        pp=ctx.out/'predictions.bin';ctx.run([b['data']['native']['path'],'predict',bankpath,b['data']['inputs']['path'],pp],'new_candidate_predict')
        observed={Path(v).resolve() for v in ctx.r['commands'][0]['observed_modules']};allowed={Path(v['path']).resolve() for v in b['native_modules']}
        assert observed and observed<=allowed
        for v in b['native_modules']:ctx.exact(v)
        with pp.open('rb') as f:
            assert f.read(24)==struct.pack('<8sIIQ',b'M495PRE1',6248,768,17540)
            pred=np.fromfile(f,M.PRED,count=17540);assert len(pred)==17540 and not f.read(1)
        for field in ('id','p','logits'):assert pred[field].tobytes()==baseline[field].tobytes()
        ctx.r['gates']['ONE_native_changed_bank_actual_modules_and_unchanged_ID_logit_probability_BYTE']=True
        assert M.emit(bank,inputs,feat,m[:,3],ctx.guard).tobytes()==pred['oracle'].tobytes()
        assert M.emit(bank,inputs,feat,pred['id'],ctx.guard).tobytes()==pred['coupled'].tobytes()
        ctx.r['gates']['ALL17540_native_oracle_and_coupled_physical_output_BYTE']=True
        data=np.empty((17540,20),'<f8');seen=np.zeros(17540,bool);ctx.phase='ALL17540_four_error_levels_and_nonorthogonal_cross_terms'
        with (ctx.out/'coefficients_F64.bin').open('rb') as f64,(ctx.out/'coefficients_F32.bin').open('rb') as f32:
            assert f64.read(24)==struct.pack('<8sIIQ',b'M498F641',3151872,513,128) and f32.read(24)==struct.pack('<8sIIQ',b'M498F321',1575936,513,128)
            for e in range(128):
                c=np.frombuffer(f64.read(3151872),'<f8').reshape(768,513);cc=np.frombuffer(f32.read(1575936),'<f4').reshape(768,513);assert c.astype('<f4').tobytes()==cc.tobytes()
                lq,ls,bq,bs,bias=M.parts(bank,e);ids=np.flatnonzero(m[:,3]==e)
                for start in range(0,len(ids),256):
                    at=ids[start:start+256];left=M.block(inputs['q'][at],lq,ls,inputs['alpha'][at]);assert left.tobytes()==feat['l'][at].tobytes()
                    a=feat['alpha'][at].astype('<f8');q=feat['q'][at].astype('<f8');h=np.ones((len(at),513),'<f8');h[:,:512]=q*a[:,None]
                    l=left.astype('<f8');u64=l+h@c.T;u32=l+h@cc.astype('<f8').T;integer=q@bq.astype('<f8').T
                    decoded=(l+((integer*bs.astype('<f8')[None,:])*a[:,None]))+bias.astype('<f8')[None,:]
                    data[at]=M.energies(y[at].astype('<f8'),u64,u32,decoded,pred['oracle'][at].astype('<f8'),pred['coupled'][at].astype('<f8'));seen[at]=True;ctx.guard()
            assert not f64.read(1) and not f32.read(1)
        assert seen.all()
        with (ctx.out/'energy_by_uid.bin').open('xb') as f:f.write(struct.pack('<8sIIQ',b'M498ENG1',160,20,17540));f.write(data.tobytes())
        ctx.r['gates']['ALL17540_fitted_serialized_parameter_arithmetic_levels_and_closure']=True
        reports=M.reports(m,occ,data,pred['id']);oldsource=json.loads((DOC/'meth495_weighted_hybrid_fit_result.json').read_bytes())['reports']
        for label in ('uid_roles','cells','rare','views','role_mode'):
            assert len(reports[label])==len(oldsource[label])
            for current,old in zip(reports[label],oldsource[label]):assert current['count']==old['count'] and current['ID_correct']==old['ID_correct'] and current['ID_fidelity']==old['ID_fidelity'] and np.isclose(current['source_energy'],old['source_energy'],rtol=1e-10,atol=1e-8)
        assert reports['exposures']==oldsource['exposures'] and sum(len(reports[v]) for v in ('uid_roles','cells','rare','views','role_mode'))==1040
        assert sum(v['count'] for v in reports['views'])==sum(v['count'] for v in reports['role_mode'])==19962
        ctx.r['gates']['ALL1040_reports_384_exposures_and_original_source_ID_denominators']=True
        result=ctx.finish({'reports':reports,'solve_cases':summaries,'prior_factor_envelope_ratio':factor,'fixed_prefix_sha256':prefix,'fixed_L_sha256':lhash,
            'UIDs':17540,'occurrences':19962,'readout_solves':127,'coefficient_cases':128,'optimizer_updates':0,'source_FFN_calls':0,'model_calls':0,'native_calls':1,
            'max_identity_envelope_ratio':float(data[:,16].max()),'max_energy_closure_envelope_ratio':float(data[:,18].max()),'decision':M.decision(reports['outcomes']),
            'physical_bank_bytes':127232136,'same_logical_weight_bytes_per_12_bank_token':14587008,
            'scope':'One-bank original/consumed-domain changed readout candidate. Pending independent audit; no fresh/model quality, SAME rate, physical DRAM, useful n/family or goal promotion.'})
        print(json.dumps({'gates':result['gates'],'outcomes':reports['outcomes'],'decision':result['decision'],'resource':ctx.terminal_resources['resource']}),flush=True)
    except BaseException as exc:
        ctx.fail(exc);print(json.dumps({'error':str(exc),'traceback':traceback.format_exc()}),flush=True);raise

if __name__=='__main__':main()
