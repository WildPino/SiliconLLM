"""ALL packed source bytes, NEW scalar C I64/F32 outputs and separate metrics."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import ctypes
from fractions import Fraction
import json
import math
import shlex
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from meth528_operations import Context,ROOT,DOC,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding-sha',required=True);ap.add_argument('--main-sha',required=True);args=ap.parse_args();ctx=None
    try:
        ctx=Context('audit',args.binding_sha);rp=DOC/'meth528_main_result.json';assert ctx.digest(rp)==args.main_sha
        raw=json.loads(rp.read_bytes());assert raw['binding_sha256']==args.binding_sha and len(raw['gates'])==7 and all(raw['gates'].values())
        folder=ROOT/'results/native_expert_scaling/meth528_atoms'
        for v in raw['output_inventory']:assert ctx.digest(v['path'])==v['sha256'] and Path(v['path']).stat().st_size==v['bytes']
        assert json.loads((folder/'terminal_resource.json').read_bytes())['result_sha256']==args.main_sha
        ctx.r['gates']['ALL_original_main_packed_banks_vectors_codes_metrics_and_terminal_SHA']=True
        np=ctx.numpy();import meth528_contract as S;import meth528_audit_math as A
        dll=ctx.out/'meth528_scalar.dll';source=ROOT/'benchmarks/native_expert_scaling/meth528_verifier.c'
        compiler=ctx.b['compiler'];obj=ctx.out/'meth528_scalar.o'
        base=[compiler['path'],'--sysroot='+compiler['sysroot'],'-target',compiler['target'],'-fintegrated-cc1']
        ctx.compile([*base,'-std=c11','-O2','-ffp-contract=off','-fno-fast-math','-c',str(source),'-o',str(obj)],'frontend')
        linkmeta=ctx.compile([*base,'-###','-shared',str(obj),'-o',str(dll),'-lm'],'link_plan')
        commands=[shlex.split(line.strip()) for line in linkmeta.read_text(encoding='utf8').splitlines() if line.strip().startswith('"')]
        assert len(commands)==1 and Path(commands[0][0]).resolve()==Path(compiler['path']).parent/'ld.lld.exe'
        command=commands[0];assert str(obj) in command and str(dll) in command
        ctx.compile(command,'linker')
        dll_sha=ctx.digest(dll);lib=ctypes.CDLL(str(dll));lib.m528_quant.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*3;lib.m528_quant.restype=ctypes.c_int
        lib.m528_eval.argtypes=[ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*11;lib.m528_eval.restype=ctypes.c_int;calls=0
        def pointer(a):return None if a is None else ctypes.c_void_p(a.ctypes.data)
        def quant(x):
            nonlocal calls
            x=np.ascontiguousarray(x,'<f4');q=np.empty(len(x),'<i2');a=np.empty(1,'<f4');calls+=1
            assert lib.m528_quant(len(x),pointer(x),pointer(q),pointer(a))==0;return q,a
        def evaluate(dots,alpha,si,wo,so,physical=True,continuous=True):
            nonlocal calls
            inputs=[np.ascontiguousarray(a,dtype) for a,dtype in zip((dots,alpha,si,wo,so),('<i8','<f4','<f4','i1','<f4'))]
            n,b=inputs[0].shape
            q=np.empty((n,b),'<i2') if physical else None;a=np.empty(n,'<f4') if physical else None
            integer=np.empty((n,768),'<i8') if physical else None;out=np.empty((n,768),'<f4') if physical else None
            c=np.empty((n,768),'<f8') if continuous else None;absolute=np.empty_like(c) if continuous else None;calls+=1
            assert lib.m528_eval(n,b,*[pointer(v) for v in [*inputs,q,a,integer,out,c,absolute]])==0
            return out,integer,q,a,c,absolute
        ctl=json.loads((folder/'controls.json').read_bytes());ii=[[1,0],[0,1],[1,1],[1,-1]];oo=[[1,2,-3,4],[-1,3,1,2]]
        for row in ctl['piecewise']:
            x=[Fraction(v) for v in row['x']];z=[sum(a*b for a,b in zip(w,x)) for w in ii]
            f=[sum(w[j]*max(z[j],0) for j in range(4)) for w in oo]
            g=[sum(w[j]*max(z[j],0) for j in (0,3)) for w in oo]
            fold=[w[0]*max(z[0],0)+w[3]*z[3] for w in oo];p=[sum(w[j]*max(z[j],0) for j in (1,2)) for w in oo];m=[w[3]*max(-z[3],0) for w in oo]
            for key,value in (('source',f),('atoms',g),('fold',fold),('P',p),('M',m)):assert value==[Fraction(v) for v in row[key]]
            assert [f[j]-g[j] for j in range(2)]==p and [f[j]-fold[j] for j in range(2)]==[p[j]+m[j] for j in range(2)]
        q,a=quant([32767,-32767,1.5,2.5,4.5,5.5,-1.5,-4.5]);assert [q.tolist()]==ctl['tie_codes'] and a.tolist()==[1.]
        q,a=quant([0,0,0,0]);assert not q.any() and a.tolist()==[ctl['zero_alpha']]==[1.]
        wide=evaluate(np.full((1,3072),32767,'<i8'),np.ones(1,'<f4'),np.ones(3072,'<f4'),np.full((768,3072),127,'i1'),np.ones(768,'<f4'))
        assert np.all(wide[1]==ctl['wide_WO_integer']==sum(32767*127 for _ in range(3072))) and ctl['wide_WO_integer']>2**31
        assert np.all(wide[0]==np.float32(ctl['wide_WO_physical']))
        ctx.r['gates']['NEW_independent_Fraction_atom_identity_C_A16_tie_zero_and_wide_I64_controls']=True
        m,occ,dev,counts=S.metadata(ctx);prices=np.zeros(17540,'<i8');freeze=json.loads((folder/'development_mask_freeze.json').read_bytes());cases=[]
        stats=np.empty((17540,24),'<f8');tolerances=np.empty_like(stats);exact=0;maxes=0;terms=0;maximum_cont_ratio=0.;maximum_identity=0.;seen=np.zeros(17540,bool)
        for e in range(1,128):
            ix=np.where(m[:,3]==e)[0];inp=S.records(ctx,'inputs',ix);assert np.all(inp['accept']==1) and np.all(inp['e']==e)
            assert inp['p'].tobytes()==m[ix,10].tobytes() and inp['alpha'].tobytes()==m[ix,12].tobytes()
            wi,si,wo,so=S.weights(ctx,e);u=S.old(ctx,'old_hinges',e).astype(int);signs=S.old(ctx,'old_signs',e);anchor=int(S.old(ctx,'old_anchors',e))
            assert dev[anchor] and m[anchor,3]==e and len(set(int(v) for v in u))==512
            v=np.array(sorted(set(int(j) for j in u)|{j for j in range(3072) if signs[j]}),'<u2')
            rotation=np.array(sorted((int(j)+1536)%3072 for j in v),'<u2');us=set(int(a) for a in u);vs=set(int(a) for a in v)
            f=np.array([j for j in range(3072) if signs[j] and j not in us],int)
            case=dict(expert=e,anchor_UID=anchor,development=int(counts[e]),width=len(v),extra_anchor_positive_rows=len(v)-512,candidate_bank_bytes=32+6*len(v)+3072+1536*len(v),active_MACs=1536*len(v),rotation=1536)
            assert case==freeze['cases'][e-1];prices[ix]=case['candidate_bank_bytes']
            dots=S.old(ctx,'old_signed_dots',ix);assert dots.dtype==np.dtype('<i8') and np.max(np.abs(dots))<=768*128*32767
            old=S.old(ctx,'old_predictions',ix);h32=S.old(ctx,'hidden',ix);ref=S.records(ctx,'unweighted',ix);target=S.records(ctx,'targets',ix)
            assert np.multiply(ref,inp['p'][:,None],dtype=np.float32).tobytes()==target.tobytes()
            physical=S.get(folder,'physical',ix);integer=S.get(folder,'integer_WO',ix);aa=S.get(folder,'hidden_alpha',ix);g=S.get(folder,'continuous',ix);negative=S.get(folder,'negative_fold',ix)
            candidate=None
            for arm,ids in enumerate((v,rotation)):
                saved=S.unpack(folder/f'e{e:03d}_a{arm}.bin',e,arm);expected=(ids,si[ids],so,wi[ids],wo[:,ids])
                assert all(a.tobytes(order='C')==b.tobytes(order='C') for a,b in zip(saved,expected))
                answer=evaluate(dots[:,ids],inp['alpha'],saved[1],saved[4],saved[2],True,arm==0)
                out,dd,qh,ah,c,absolute=answer
                assert dd.tobytes()==integer[:,arm].tobytes() and out.tobytes()==physical[:,arm].tobytes() and ah.tobytes()==aa[:,arm].tobytes()
                oldqh=np.load(folder/f'e{e:03d}_a{arm}_hidden_codes.npy',allow_pickle=False);assert oldqh.shape==qh.shape and oldqh.tobytes()==qh.tobytes()
                terms+=len(ix)*768*len(ids)
                if arm==0:
                    bnd=2e-11*np.maximum(1,absolute)*(1+8*65536/((1<<53)-65536));rat=float(np.max(np.abs(c-g)/bnd));assert rat<=1
                    maximum_cont_ratio=max(maximum_cont_ratio,rat);candidate=(qh,ah,absolute)
            qh,ah,abs_atoms=candidate;outside=[j for j in range(3072) if j not in vs]
            zero=np.all(h32[:,outside]==0,axis=1);sourcea=S.old(ctx,'scales',ix);sourceq=S.old(ctx,'codes',ix)
            assert ah[zero].tobytes()==sourcea[zero].tobytes() and qh[zero].tobytes()==sourceq[zero][:,v].tobytes()
            assert physical[zero,0].tobytes()==ref[zero].tobytes();exact+=int(zero.sum());maxes+=int(np.sum(ah.view('<u4')==sourcea.view('<u4')))
            neg=evaluate(-dots[:,f],inp['alpha'],si[f],wo[:,f],so,False,True)
            bnd=2e-11*np.maximum(1,neg[5])*(1+8*65536/((1<<53)-65536));rat=float(np.max(np.abs(neg[4]-negative)/bnd));assert rat<=1
            maximum_cont_ratio=max(maximum_cont_ratio,rat)
            bound=A.identity_bound(inp,h32,wi,si,wo,so,u,f,S.old(ctx,'old_fold64',e),abs_atoms,neg[5])
            stat,tol=A.metrics(g,negative,physical,old,ref,target,inp['p'],bound);stats[ix]=stat;tolerances[ix]=tol;maximum_identity=max(maximum_identity,float(np.max(stat[:,11])))
            case.update(omitted_zero_BYTE_equal_UIDs=int(zero.sum()),global_hidden_max_preserved_UIDs=int(np.sum(ah.view('<u4')==sourcea.view('<u4'))));assert case==raw['cases'][e-1];cases.append(case);seen[ix]=True;ctx.guard()
            if e%16==0:print('scalar_C_atom_audit_parent='+str(e),flush=True)
        assert seen.all() and calls==384
        ctx.r['gates']['ALL127_original_dev_mask_and_exact_original_packed_I8_scale_ID_contracts']=True
        ctx.r['gates']['ALL35080_new_physical_I64_F32_A16_BYTE_vectors_and_omitted_zero_source_domain_verified']=True
        ctx.r['gates']['ALL35080_continuous_atom_negative_fold_vectors_and_structural_bound_identities']=True
        recorded=S.get(folder,'metrics',slice(None));cols=list(range(11))+list(range(12,23))
        eratio=float(np.max(np.abs(recorded[:,cols]-stats[:,cols])/tolerances[:,cols]));assert eratio<=1 and np.max(recorded[:,[11,23]])<=1
        domains=[]
        for split,mask in (('development',dev),('consumed_validation',~dev)):domains.append((dict(kind='uid',split=split),np.where(mask)[0]))
        for label,lo,hi in (('0',0,0),('1..4',1,4),('5..15',5,15),('>=16',16,17540)):
            for split,mask in (('development',dev),('consumed_validation',~dev)):domains.append((dict(kind='rare',development_class=label,split=split),np.where(mask&(counts[m[:,3]]>=lo)&(counts[m[:,3]]<=hi))[0]))
        for e in range(128):
            for split,mask in (('development',dev),('consumed_validation',~dev)):domains.append((dict(kind='parent',expert=e,split=split),np.where(mask&(m[:,3]==e))[0]))
        for book in range(192):
            for mode in range(2):domains.append((dict(kind='book',book=book,role=book//64,mode=mode),occ[np.where((occ[:,4]==book)&(occ[:,6]==mode))[0],1]))
        for role in range(3):
            for mode in range(2):domains.append((dict(kind='role',role=role,mode=mode),occ[np.where((occ[:,7]==role)&(occ[:,6]==mode))[0],1]))
        assert len(domains)==len(raw['views'])==656;audited=[]
        for (label,ids),view in zip(domains,raw['views']):
            assert view['count']==len(ids) and all(view[k]==v for k,v in label.items())
            sums=[math.fsum(float(v) for v in stats[ids,j]) for j in cols]
            bs=[math.fsum(float(v) for v in tolerances[ids,j])+3e-11*max(1,math.fsum(abs(float(v)) for v in stats[ids,j])) for j in cols]
            assert all(abs(a-b)<=bound for a,b,bound in zip(sums,view['sums'],bs));v={**label,'count':len(ids)}
            for prefix,off in (('unweighted',0),('weighted',11)):
                for j,arm in enumerate(('continuous_atoms','physical_atoms','physical_rotation','physical_old_fold'),1):
                    rms=math.sqrt(sums[off+j]/sums[off]) if sums[off] else None;previous=view[prefix+'_'+arm+'_RMS']
                    assert (rms is None and previous is None) or (rms is not None and previous is not None and math.isclose(rms,previous,rel_tol=3e-10,abs_tol=3e-10));v[prefix+'_'+arm+'_RMS']=rms
            price=math.fsum(int(a) for a in prices[ids])/(len(ids)*4733952) if len(ids) else None
            assert price==view['selected_FFN_byte_ratio'] or (price is not None and math.isclose(price,view['selected_FFN_byte_ratio'],rel_tol=1e-14));v['selected_FFN_byte_ratio']=price;audited.append(v)
        eligible=A.eligibility(audited);decision='VARIABLE_SOURCE_ATOMS_ELIGIBLE_FOR_NEW_MULTIREGION_TEST_NOT_WHOLE_PROMOTION' if all(eligible.values()) else 'VARIABLE_SOURCE_ATOM_SINGLE_REFERENCE_RECIPE_CLOSED'
        assert eligible==raw['eligibility'] and decision==raw['decision'];ctx.r['gates']['ALL656_independent_domain_metrics_prices_and_FIXED_six_rare_rotation_eligibility']=True
        ctx.r['gates']['unchanged_decision_original523_resource_fault_and_zero_source_native_prefix_replay']=True
        write(ctx.out/'verified_atom_outputs.json',dict(C_verifier_sha256=dll_sha,C_calls=calls,physical_BYTE_vectors=35080,continuous_vectors=35080,exact_selected_WO_product_terms=terms,
            maximum_continuous_bound_ratio=maximum_cont_ratio,maximum_metric_bound_ratio=eratio,maximum_structural_identity_bound_ratio=maximum_identity))
        ctx.finish(dict(main_sha256=args.main_sha,cases=cases,views=raw['views'],eligibility=eligible,decision=decision,
            candidate_functions=127,control_functions=127,candidate_bank_bytes=sum(c['candidate_bank_bytes'] for c in cases),
            new_candidate_physical_vectors=35080,new_continuous_atom_vectors=17540,new_negative_fold_vectors=17540,new_exact_selected_WO_product_terms=terms,
            omitted_zero_BYTE_equal_UIDs=exact,global_hidden_max_preserved_UIDs=maxes,new_C_verifier_calls=calls,C_verifier_sha256=dll_sha,
            maximum_continuous_bound_ratio=maximum_cont_ratio,maximum_metric_bound_ratio=eratio,maximum_structural_identity_bound_ratio=maximum_identity,
            source_response_function_calls=0,full_source_WI_projection_rows=0,old_native_source_WO_replays=0,model_calls=0,native_calls=calls,
            readout_fits=0,consumed_rows_for_selection=0,upstream_original523_main_completed=False,physical_DRAM_verified=False,
            scope='Independent NEW scalar C selected-WO/quantizer verifier and masked continuous residual arithmetic; no original source WI/FFN/model/native prefix replay, fresh quality, useful-n, C inference speed/DRAM or whole promotion.'))
    except BaseException:
        if ctx:ctx.fail()
        raise


if __name__=='__main__':main()
