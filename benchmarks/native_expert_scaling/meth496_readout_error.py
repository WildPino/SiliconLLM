"""One unchanged-artifact fitted/coefficient/arithmetic decomposition."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import json
from pathlib import Path
import struct
import traceback
from meth496_operations import Context,ROOT,DOC,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--binding-sha',required=True);args=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth496_readout_error',Path(args.out).resolve())
    try:
        b=ctx.admit(args.binding_sha);ctx.binding=b
        import numpy as np
        import threadpoolctl
        threadpoolctl.threadpool_limits(1);assert all(v['num_threads']==1 for v in threadpoolctl.threadpool_info())
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.r['numerical_imports']=True
        import meth496_math as M
        write(ctx.out/'controls.json',M.controls());ctx.r['gates']['NEW496_dyadic_signed_cross_terms_and_exact_integer_dot_controls']=True
        def wire(key,magic,width,reserved,count,dtype):
            with Path(b['data'][key]['path']).open('rb') as f:
                assert f.read(24)==struct.pack('<8sIIQ',magic,width,reserved,count)
                x=np.fromfile(f,dtype,count=count);assert len(x)==count and not f.read(1);return x
        info=wire('uid',b'M493U001',180,0,17540,M.UID);m=info['m']
        occ=wire('occurrences',b'M493O001',68,0,19962,np.dtype(('<u4',(17,))))
        features=wire('features',b'M495FEA1',6148,512,17540,M.FEATURE)
        pred=wire('predictions',b'M495PRE1',6248,768,17540,M.PRED)
        y=wire('targets',b'M493Y001',3072,768,17540,np.dtype(('<f4',(768,))))
        assert np.array_equal(m[:,0],np.arange(17540)) and np.all(m[:,2]==11) and np.all(m[:,3]<128) and np.all(m[:,6]==2)
        dev=(m[:,4]&5)!=0;val=(m[:,4]&2)!=0;assert (int(dev.sum()),int(val.sum()))==(11721,5819) and not np.any(dev&val)
        assert np.array_equal(occ[:,0],np.arange(19962)) and np.all(occ[:,1]<17540)
        assert np.array_equal(occ[:,14],m[occ[:,1],1]) and np.array_equal(occ[:,11],m[occ[:,1],3]) and np.all(occ[:,12]==1)
        assert np.array_equal(occ[:,15],m[occ[:,1],12]) and np.array_equal(occ[:,16],m[occ[:,1],10])
        assert np.array_equal(np.bincount(occ[:,1],minlength=17540),m[:,7])
        bank=Path(b['data']['bank']['path']).read_bytes();assert len(bank)==127232136 and bank[:40]==struct.pack('<8s8I',b'M495BNK1',768,3072,512,128,11,4,8,16)
        ctx.r['gates']['ALL17540_UID_19962_occurrence_original_domains_and_input_wire_joins']=True
        values=np.empty((17540,13),'<f8');seen=np.zeros(17540,bool);cells=[]
        ctx.phase='ALL128_coefficient_cases_and17540_fixed_U_Q_P_energy_levels'
        with Path(b['data']['coefficients']['path']).open('rb') as f:
            assert f.read(24)==struct.pack('<8sIIQ',b'M495FIT1',1575936,513,128)
            for e in range(128):
                cb=f.read(1575936);assert len(cb)==1575936;coef=np.frombuffer(cb,'<f4').reshape(768,513).astype('<f8');assert np.isfinite(coef).all()
                pos=223368+e*992256+592896
                bq=np.frombuffer(bank,offset=pos,count=768*512,dtype='<i1').reshape(768,512);pos+=768*512
                bs=np.frombuffer(bank,offset=pos,count=768,dtype='<f4').astype('<f8');bias=np.frombuffer(bank,offset=pos+3072,count=768,dtype='<f4').astype('<f8')
                assert np.all(bq!=-128) and np.isfinite(bs).all() and np.all(bs>0) and coef[:,512].tobytes()==bias.tobytes()
                at=np.flatnonzero(m[:,3]==e);cells.append({'expert':e,'UIDs':len(at),'development':int(dev[at].sum()),'consumed_validation':int(val[at].sum())})
                for start in range(0,len(at),256):
                    ids=at[start:start+256];q=features['q'][ids].astype('<f8');a=features['alpha'][ids].astype('<f8')
                    assert np.isfinite(a).all() and np.all(a>0) and np.all(np.abs(q)<=32767)
                    h=np.ones((len(ids),513),'<f8');h[:,:512]=q*a[:,None]
                    left=features['l'][ids].astype('<f8');u=left+h@coef.T
                    integer=q@bq.astype('<f8').T;assert np.all(np.abs(integer)<=512*127*32767) and np.array_equal(integer,np.rint(integer))
                    decoded=(left+((integer*bs[None,:])*a[:,None]))+bias[None,:]
                    p=pred['oracle'][ids].astype('<f8');yy=y[ids].astype('<f8')
                    values[ids]=M.energy(yy,u,decoded,p);seen[ids]=True;ctx.guard()
                if e%16==0:ctx.log(experts_decomposed=e+1);print(json.dumps({'experts_decomposed':e+1}),flush=True)
            assert not f.read(1)
        assert seen.all() and np.isfinite(values).all()
        ep=ctx.out/'energy_by_uid.bin'
        with ep.open('xb') as f:f.write(struct.pack('<8sIIQ',b'M496ENG1',104,13,17540));f.write(values.tobytes())
        assert ep.stat().st_size==1824184
        ctx.r['gates']['ALL17540_three_fixed_levels_pairwise_cross_terms_and_vector_energy_closure']=True
        reports=M.reports(m,occ,values,pred['id'])
        assert sum(v['count'] for v in reports['views'])==sum(v['count'] for v in reports['role_mode'])==19962
        source=json.loads((DOC/'meth495_weighted_hybrid_fit_result.json').read_bytes())['reports']
        for label in ('uid_roles','cells','rare','views','role_mode'):
            assert len(reports[label])==len(source[label])
            for current,old in zip(reports[label],source[label]):
                assert current['count']==old['count'] and current['ID_correct']==old['ID_correct'] and current['ID_fidelity']==old['ID_fidelity']
                assert np.isclose(current['source_energy'],old['source_energy'],rtol=1e-10,atol=1e-8)
                assert np.isclose(current['total_error_energy'],old['oracle_error_energy'],rtol=1e-10,atol=1e-8)
        assert reports['exposures']==source['exposures']
        ctx.r['gates']['ALL1040_reports_384_exposures_and_saved495_oracle_ID_denominators_preserved']=True
        outcome=reports['diagnostic_outcomes']
        decision='UNQUANTIZED_FITTED_FUNCTION_REQUIRES_ANALYSIS_BEFORE_PRECISION_CHANGE' if not outcome['ALL_six_unquantized_U_RMS_1pct'] else (
            'PHYSICAL_READOUT_COEFFICIENT_ERROR_SELECTED_FOR_NEXT_CHANGE' if not outcome['ALL_six_saved_P_RMS_1pct'] and outcome['ALL_six_parameters_RMS_at_least_10x_arithmetic'] else 'PHYSICAL_ARITHMETIC_ERROR_REQUIRES_ANALYSIS')
        result=ctx.finish({'reports':reports,'cells':cells,'UIDs':17540,'occurrences':19962,'coefficient_cases':128,
            'max_identity_abs':float(values[:,9].max()),'max_identity_envelope_ratio':float(values[:,10].max()),
            'max_energy_closure_abs':float(values[:,11].max()),'max_energy_closure_envelope_ratio':float(values[:,12].max()),
            'decision':decision,'source_FFN_calls':0,'model_calls':0,'native_calls':0,'optimizer_updates':0,'projection_solves':0,
            'scope':'Fixed-artifact numerical error accounting, pending independent audit. No new candidate, quality, speed, useful n or goal promotion.'})
        print(json.dumps({'gates':result['gates'],'diagnostic_outcomes':outcome,'resource':ctx.terminal_resources['resource']}),flush=True)
    except BaseException as exc:
        ctx.fail(exc);print(json.dumps({'error':str(exc),'traceback':traceback.format_exc()}),flush=True);raise

if __name__=='__main__':main()
