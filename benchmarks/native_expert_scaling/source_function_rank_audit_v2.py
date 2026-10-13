"""Independent full row-block moments, square-root/eigen checks and two rectangular SVDs."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic()


def main(a):
    import numpy as np
    import psutil
    re,te=extent(a.report),extent(a.terminal)
    assert re['sha256']==a.report_sha and te['sha256']==a.terminal_sha
    r=json.loads(a.report.read_bytes())
    t=json.loads(a.terminal.read_bytes())
    assert t['exit_code']==0 and t['result']==re
    assert extent(r['binding']['path'])==r['binding']
    b=json.loads(Path(r['binding']['path']).read_bytes())
    assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    assert np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil']
    assert not a.out.exists() and b['ranks']==[128,256,512,1024,2048]
    assert b['sites']==[0,12,23] and len(r['records'])==3
    assert [q['site'] for q in r['records']]==b['sites']
    assert b['sections']==[['gate',0,3072],['x',3072,6144],['B',6144,6400],
                           ['C',6400,6656],['delta',6656,6704]]
    proc=psutil.Process()
    proc.cpu_affinity(list(range(6)))
    receipt=dict(schema='SOURCE_FUNCTION_RANK_AUDIT_TERMINAL_V2',command=sys.orig_argv,
                 pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    stage='startup'
    stream=None
    maximum={}
    verification={}
    rows=[]
    words=0
    data_words=0
    num=b['numeric']

    def guard():
        assert time.monotonic()-START<=b['limits']['audit_seconds']
        assert proc.memory_info().peak_wset<=b['limits']['OS_bytes']
        assert not proc.children(recursive=True)

    extra=[extent(Path(__file__)),extent(DOC/'SOURCE_FUNCTION_RANK_AUDIT_V2_PROTOCOL_20261013.md')]

    def check():
        for e in b['inputs']+extra+[r['binding'],re,te]+[q['file'] for q in r['records']]:
            guard()
            assert extent(e['path'])==e,e['path']

    def metric(key,error,tolerance):
        assert np.isfinite(error)
        maximum[key]=max(maximum.get(key,0.),float(error))
        verification[key]=bool(verification.get(key,True) and error<=tolerance)
        return float(error)

    def matrix(key,x,y,tol):
        assert x.shape==y.shape
        return metric(key,np.linalg.norm(x-y)/np.linalg.norm(y),tol)

    def meta(x,y):
        if isinstance(y,dict):
            assert isinstance(x,dict) and x.keys()==y.keys()
            for k in y:
                meta(x[k],y[k])
        elif isinstance(y,bool) or isinstance(y,int):
            assert x==y,(stage,x,y)
        else:
            metric('metadata_absolute_or_relative',abs(x-y)/max(abs(y),1.),
                   num['metadata_absolute_or_relative'])

    def eigen(M,ev,U):
        assert M.shape==U.shape==(2048,2048) and ev.shape==(2048,)
        assert np.all(np.diff(ev)>=0) and ev[-1]>0
        asym=float(np.linalg.norm(M-M.T)/np.linalg.norm(M))
        orth=float(np.max(np.abs(U.T@U-np.eye(2048))))
        res=float(np.linalg.norm(M@U-U*ev)/np.linalg.norm(M))
        neg=float(max(0.,-ev[0])/ev[-1])
        pos=np.maximum(ev,0.)
        total=float(pos.sum())
        m=dict(asymmetry_relative=asym,eigen_residual_relative=res,orthogonal_max=orth,
               raw_minimum_eigenvalue=float(ev[0]),raw_maximum_eigenvalue=float(ev[-1]),
               negative_eigenvalue_relative=neg,negative_modes=int((ev<0).sum()),
               exact_zero_modes=int((ev==0).sum()),negative_mass_removed=float(-ev[ev<0].sum()),
               negative_mass_relative=float(-ev[ev<0].sum()/total),
               trace=float(np.trace(M)),nonnegative_spectrum_sum=total)
        g=dict(symmetry=asym<=num['symmetry_relative'],residual=res<=num['eigen_residual_relative'],
               orthogonality=orth<=num['orthogonal_max'],PSD=neg<=num['negative_eigenvalue_relative'])
        return m,g

    def tail(ev):
        pos=np.maximum(ev,0.)
        total=float(pos.sum())
        return {str(k):dict(discarded_squared_energy=float(sum(pos[:2048-k])),
                           relative_squared_energy=float(sum(pos[:2048-k])/total),
                           relative_RMS=float(np.sqrt(sum(pos[:2048-k])/total))) for k in b['ranks']}

    try:
        check()
        cad=json.loads(Path(b['capture_adjudication']['path']).read_bytes())
        tad=json.loads(Path(b['temporal_adjudication']['path']).read_bytes())
        assert cad['decision']=='INDEPENDENT_DATA_AND_SOURCE_CONTINUITY_PASS' and cad['report']==b['capture']
        assert tad['decision']=='INDEPENDENT_STORED_PASS' and tad['report']==b['temporal']
        c=json.loads(Path(b['capture']['path']).read_bytes())
        assert c['binding_sha256']==b['capture_binding']['sha256']
        native=json.loads(Path(b['temporal']['path']).read_bytes())
        source={q['id']:q for q in c['records'] if q['split']=='FIT'}
        expected=[]
        for s in b['sites']:
            for q in native['records']:
                if q['site']==s and q['split']=='FIT':
                    h=source[q['id']]
                    assert q['history']==h['history'] and q['domain']==h['domain']
                    expected.append(dict(site=s,id=q['id'],split='FIT',domain=q['domain'],
                                         history=q['history'],H=h['sites'][s]['fields']['H'],native=q['file']))
        assert b['records']==expected
        get,stream=tensor_reader(b['source_weights'],np)
        for q in r['records']:
            site=q['site']
            stage=f'{site}/independent_moments'
            rr=[e for e in b['records'] if e['site']==site]
            assert len(rr)==24 and sum(e['history'] for e in rr)==11148
            CH=np.zeros((2048,2048))
            CY=np.zeros_like(CH)
            stack=[]
            for e in rr:
                guard()
                bits=np.fromfile(e['H']['path'],'<u2').reshape(e['H']['shape'])
                H=(bits.astype('<u4')*65536).view('<f4').astype('f8')
                with np.load(e['native']['path'],allow_pickle=False) as z:
                    Y=z['reconstructed_denominator__raw_scores'].copy()
                assert Y.dtype==np.dtype('<f4') and Y.shape==H.shape==(e['history'],2048)
                Y=Y.astype('f8')
                assert np.isfinite(H).all() and np.isfinite(Y).all()
                hc=np.zeros_like(CH)
                yc=np.zeros_like(CY)
                for lo in range(0,e['history'],32):
                    guard()
                    hb=H[lo:lo+32]
                    yb=Y[lo:lo+32]
                    hc+=np.einsum('ti,tj->ij',hb,hb,optimize=True)
                    yc+=np.einsum('ti,tj->ij',yb,yb,optimize=True)
                CH+=hc/e['history']/24
                CY+=yc/e['history']/24
                stack.append(Y/np.sqrt(24*e['history']))
                data_words+=H.size+Y.size
            with np.load(q['file']['path'],allow_pickle=False) as z:
                arrays={k:z[k].copy() for k in z.files}
            assert set(arrays)=={'C_H','H_eigenvalues','H_eigenvectors','root','section_energies',
                                 'row_scales','K','input_eigenvalues','input_eigenvectors',
                                 'C_Y','output_eigenvalues','output_eigenvectors'}
            assert all(v.dtype==np.dtype('<f8') and np.isfinite(v).all() for v in arrays.values())
            words+=sum(v.size for v in arrays.values())
            matrix('H_block_Gram_relative',arrays['C_H'],CH,num['matrix_relative'])
            matrix('Y_block_Gram_relative',arrays['C_Y'],CY,num['matrix_relative'])
            hm,hg=eigen(arrays['C_H'],arrays['H_eigenvalues'],arrays['H_eigenvectors'])
            L=arrays['root']
            assert L.shape==(2048,2048)
            root=(arrays['H_eigenvectors']*np.sqrt(np.maximum(arrays['H_eigenvalues'],0.)))@arrays['H_eigenvectors'].T
            matrix('root_reconstruction_relative',L,root,num['matrix_relative'])
            rooterr=matrix('root_square_relative',L@L,CH,num['root_square_relative'])
            hm['root_square_relative']=float(np.linalg.norm(L@L-arrays['C_H'])/np.linalg.norm(arrays['C_H']))
            hg['root_square']=hm['root_square_relative']<=num['root_square_relative']
            W=get(f'model.layers.{site}.mamba.in_proj.weight')
            assert W.shape==(6704,2048)
            WL=W@L
            energies=np.array([np.linalg.norm(WL[lo:hi])**2 for _,lo,hi in b['sections']])
            assert (energies>0).all()
            matrix('section_energy_relative',arrays['section_energies'],energies,num['root_square_relative'])
            scales=np.concatenate([np.repeat(en**(-.5),hi-lo)
                                   for (_,lo,hi),en in zip(b['sections'],energies)])
            matrix('row_scale_relative',arrays['row_scales'],scales,num['root_square_relative'])
            J=scales[:,None]*WL
            gram=np.zeros_like(CH)
            for lo in range(0,6704,128):
                guard()
                block=J[lo:lo+128]
                gram+=block.T@block
            matrix('whitened_block_K_relative',arrays['K'],gram,num['root_square_relative'])
            km,kg=eigen(arrays['K'],arrays['input_eigenvalues'],arrays['input_eigenvectors'])
            km['balanced_trace_relative_to_five']=float(abs(np.trace(arrays['K'])-5)/5)
            kg['balanced_trace']=km['balanced_trace_relative_to_five']<=num['root_square_relative']
            ym,yg=eigen(arrays['C_Y'],arrays['output_eigenvalues'],arrays['output_eigenvectors'])
            meta(q['input_covariance_metrics'],hm)
            meta(q['input_metrics'],km)
            meta(q['output_metrics'],ym)
            assert q['numeric_gates']==dict(covariance=hg,input=kg,output=yg)
            meta(q['section_energies'],{s:float(en) for (s,_,_),en in zip(b['sections'],arrays['section_energies'])})
            meta(q['input_tails'],tail(arrays['input_eigenvalues']))
            meta(q['output_tails'],tail(arrays['output_eigenvalues']))
            assert q['cases']==24 and q['positions']==11148
            stage=f'{site}/direct_rectangular_SVD'
            guard()
            sigma=np.linalg.svd(J,compute_uv=False)[::-1]
            iserr=metric('input_direct_squared_spectrum_relative',
                        np.max(np.abs(sigma**2-arrays['input_eigenvalues']))/sigma[-1]**2,
                        num['direct_squared_spectrum_relative'])
            weighted=np.concatenate(stack)
            assert weighted.shape==(11148,2048)
            guard()
            osigma=np.linalg.svd(weighted,compute_uv=False)[::-1]
            oserr=metric('output_direct_squared_spectrum_relative',
                        np.max(np.abs(osigma**2-arrays['output_eigenvalues']))/osigma[-1]**2,
                        num['direct_squared_spectrum_relative'])
            rows.append(dict(site=site,numeric_gates=dict(covariance=hg,input=kg,output=yg),
                             root_square_relative=rooterr,input_direct_squared_spectrum_relative=iserr,
                             output_direct_squared_spectrum_relative=oserr,
                             input_tails_from_direct_SVD=tail(sigma**2),
                             output_tails_from_direct_SVD=tail(osigma**2)))
            print(json.dumps(dict(site=site,complete=len(rows),seconds=time.monotonic()-START,
                                  input_SVD_relative=iserr,output_SVD_relative=oserr)),flush=True)
            del arrays,CH,CY,W,WL,J,L,root,weighted,stack,gram,H,Y,hc,yc,bits
        stream.close()
        stream=None
        check()
        guard()
        ok=all(all(g.values()) for q in rows for g in q['numeric_gates'].values())
        assert r['numeric_all_pass']==ok
        assert all(r[k]==0 for k in ['DEV_arrays_read','source_forwards','native_calls',
                                     'optimizer_updates','factor_kernels_built'])
        ad=dict(schema='SOURCE_FUNCTION_RANK_AUDIT_V2',source=extent(Path(__file__)),
                freeze=a.freeze,report=re,terminal=te,binding=r['binding'],records=rows,
                all_stored_array_words_verified=words,all_FIT_H_and_score_words=data_words,
                maximum=maximum,verification_gates=verification,numeric_all_pass=ok,
                before_after_hashes_exact=True,
                decision='INDEPENDENT_STORED_PASS' if ok and all(verification.values())
                         else 'INDEPENDENT_STORED_NUMERIC_FAILURE',
                seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
                scope='All72 FIT histories/full H and stored score moments via row-block sums;'
                      'every matrix/eigenvector/value/root/scale/tail and all2048 singular values '
                      'of rectangular whitened input and equal-history weighted output stack/site. '
                      'Finite globally-linear proxies only;no new factor kernel/field/state/causal '
                      'quality or speed admission.')
        write(a.out,ad)
        receipt.update(exit_code=0,error=None,result=extent(a.out))
        print(json.dumps(dict(adjudication=extent(a.out),decision=ad['decision'],maximum=maximum)),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),stage=stage,traceback=traceback.format_exc())
        raise
    finally:
        if stream is not None:
            stream.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['report','terminal','out']:
        p.add_argument('--'+k,type=Path,required=True)
    for k in ['report-sha','terminal-sha','freeze']:
        p.add_argument('--'+k,required=True)
    main(p.parse_args())
