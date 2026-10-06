"""ONE original-WI angular-region construction and complete finite-domain economy."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
import argparse,hashlib,json,struct
from pathlib import Path
from meth504_operations import Context,ROOT,DOC,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--binding-sha',required=True);a=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth504_regions',Path(a.out).resolve())
    try:
        b=ctx.admit(a.binding_sha)
        import numpy as np
        import threadpoolctl as T
        import meth504_region_math as M
        T.threadpool_limits(1);pools=T.threadpool_info();assert pools and all(v['num_threads']==1 for v in pools)
        for v in pools:assert any(d['path']==str(Path(v['filepath']).resolve()) for d in b['catalog'])
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.r['numerical_imports']=True
        write(ctx.out/'controls.json',M.controls());ctx.r['gates']['NEW_strict_integer_cone_tiny_hidden_output_identity_and_104bit_controls']=True
        N,D,H,B=17540,768,3072,1536
        def wire(key,magic,width,reserved,dtype,count):
            p=Path(b['data'][key]['path']);dtype=np.dtype(dtype);assert p.stat().st_size==24+count*dtype.itemsize
            with p.open('rb') as f:assert f.read(24)==struct.pack('<8sIIQ',magic,width,reserved,count)
            return np.memmap(p,mode='r',offset=24,dtype=dtype,shape=(count,))
        m=wire('uid',b'M493U001',180,0,[('m','<u4',(13,)),('hash','u1',(128,))],N)['m']
        it=np.dtype([('e','<u4'),('accept','<u4'),('alpha','<f4'),('x','<f4',(D,)),('q','<i2',(D,))])
        ni=wire('inputs',b'M499INP1',4624,D,[('core',it),('p','<f4')],N);q=ni['core']['q']
        assert np.all(q>=-32767) and np.all(q<=32767)
        occ=wire('occurrences',b'M493O001',68,0,np.dtype(('<u4',(17,))),19962)
        dev=(m[:,4]&5)!=0;val=(m[:,4]&2)!=0
        assert np.array_equal(m[:,0],np.arange(N)) and np.array_equal(ni['core']['e'],m[:,3]) and np.all(ni['core']['accept']==1) and np.all(m[:,2]==11)
        assert ni['p'].tobytes()==m[:,10].tobytes() and (int(dev.sum()),int(val.sum()))==(11721,5819) and np.all(dev^val)
        assert np.array_equal(occ[:,0],np.arange(19962)) and np.all(occ[:,12]==1) and np.all(occ[:,8]==11) and np.array_equal(np.bincount(occ[:,1],minlength=N),m[:,7])
        for oi,mi in [(14,1),(11,3),(15,12),(16,10)]:assert np.array_equal(occ[:,oi],m[occ[:,1],mi])
        def arr(k):return np.load(b['data'][k]['path'],mmap_mode='r',allow_pickle=False)
        assignment,hidden,codes,scales=[arr(k) for k in ['assignment','hidden','codes','scales']]
        assert hidden.shape==codes.shape==(N,H) and np.all(assignment[dev]>=0) and np.all(assignment[val]==-2)
        ctx.r['gates']['ALL_original_UID_roles_parent_mass_and_development_groups_contracts']=True
        entries=json.loads((DOC/'meth380_switch_base128_export_result.json').read_bytes())['tensors'];payload=np.memmap(b['data']['payload']['path'],mode='r',dtype='u1')
        raw503=json.loads(Path(b['data']['raw503']['path']).read_bytes());rows=[];centres=unsupported=valid=copies=pair_count=0
        def output(name,shape,dtype):return np.lib.format.open_memmap(ctx.out/name,mode='w+',shape=shape,dtype=dtype)
        norm=output('query_norm2.npy',(N,),'<i8');norm[:]=np.sum(q.astype('<i8')**2,axis=1)
        chosen=output('selected_region.npy',(N,),'<i4');chosen[:]=-1;chosen[norm==0]=-2
        widths=output('selected_width.npy',(N,),'<u2');widths[:]=H;widths[norm==0]=0
        ctx.phase='ALL128_development_code_centres_NEW_negative_margin_and_exact_rational_masks'
        # Construct EVERY development region before any consumed predicate is evaluated.
        for e in range(128):
            at=np.flatnonzero(m[:,3]==e);di=at[dev[at]]
            if not len(at):assert e==0;rows.append({'expert':e,'status':'EMPTY_UNCOMPILED_UNPROMOTED','regions':[]});continue
            C=raw503['experts'][e]['children'];groups=[c for c in range(C) if np.any(assignment[di]==c)];seeds=[]
            assert len(groups)==C-raw503['experts'][e]['unsupported_selector_children']
            for c in groups:
                group=di[assignment[di]==c];seeds.append(M.choose_seed(q[group],group))
            v=entries[f'decoder.block.11.layer.2.mlp.experts.expert_{e}.wi.weight'];assert v['shape']==[H,D] and v['encoding']==1
            wi=np.frombuffer(payload,'i1',v['elements'],v['offset']).reshape(H,D);si=np.frombuffer(payload,'<f4',H,v['scale_offset'])
            assert hashlib.sha256(wi.tobytes()).hexdigest()==v['sha256'] and hashlib.sha256(si.tobytes()).hexdigest()==v['scale_sha256'] and np.all(si>0) and np.isfinite(si).all()
            R=np.sum(wi.astype('<i8')**2,axis=1).astype('<u4');d=q[seeds].astype('<f8')@wi.astype('<f8').T
            assert np.array_equal(d,np.rint(d));d=d.astype('<i8');mk=np.zeros((len(seeds),H),'u1');rr=[]
            for c,k in enumerate(seeds):
                mk[c],cert=M.mask(d[c],R,B,int(norm[k]));good=cert['status'] in ['ANGULAR_CONE','ALL_NONZERO_ROWS_RETAINED']
                rr.append({'region':c,'source_group':groups[c],'seed_UID':k,'width':int(mk[c].sum()),**cert});valid+=good;unsupported+=not good
            np.save(ctx.out/f'seed_codes_e{e:03}.npy',q[seeds]);np.save(ctx.out/f'centre_dots_e{e:03}.npy',d);np.save(ctx.out/f'row_norm2_e{e:03}.npy',R);np.save(ctx.out/f'masks_e{e:03}.npy',mk)
            rows.append({'expert':e,'development':len(di),'consumed':int(val[at].sum()),'regions':rr,'selector_dot_products_per_query':len(seeds)*D})
            centres+=len(seeds);copies+=int(mk.sum());pair_count+=len(at)*len(seeds);ctx.guard()
        assert centres==637 and valid+unsupported==637
        ctx.r['gates']['ALL637_development_only_seeds_NEW_source_negative_dots_and_maximal_fixed_centre_rational_masks']=True
        ctx.phase='fixed_regions_ALL_original_exact_predicates_and_support_quantizer_certificates'
        for e,row in enumerate(rows):
            at=np.flatnonzero(m[:,3]==e)
            if not len(at):continue
            seeds=np.load(ctx.out/f'seed_codes_e{e:03}.npy');mk=np.load(ctx.out/f'masks_e{e:03}.npy')
            dots=q[at].astype('<f8')@seeds.astype('<f8').T;assert np.array_equal(dots,np.rint(dots));dots=dots.astype('<i8')
            pred=np.zeros(dots.shape,'u1')
            for i,k in enumerate(at):
                for c,cert in enumerate(row['regions']):pred[i,c]=M.accepts(int(dots[i,c]),int(norm[k]),cert)
                matches=np.flatnonzero(pred[i])
                if norm[k]==0:assert not codes[k].any();continue
                if len(matches):
                    c=int(matches[0]);chosen[k]=c;widths[k]=row['regions'][c]['width'];j=np.flatnonzero(mk[c])
                    assert not codes[k,mk[c]==0].any() and np.all(hidden[k,mk[c]==0]==0)
                    qc,ac=M.quant(hidden[k,j][None,:]);assert qc[0].tobytes()==codes[k,j].tobytes() and ac.tobytes()==scales[k:k+1].tobytes()
            np.save(ctx.out/f'query_dots_e{e:03}.npy',dots);np.save(ctx.out/f'predicates_e{e:03}.npy',pred);ctx.guard()
        norm.flush();chosen.flush();widths.flush();covered=(chosen!=-1)
        ctx.r['gates']['ALL_exact_integer_query_predicates_lowest_region_and_saved_hidden_zero_scale_identity']=True
        def stats(take,w=None):
            if w is None:w=np.ones(N,dtype='<i8')
            total=int(w[take].sum());hit=int(w[take][covered[take]].sum())
            return {'UIDs':len(take),'occurrences':total,'certified':hit,'certified_fraction':hit/total if total else None,'fallback':total-hit,
                'mean_selected_width':float(np.sum(widths[take].astype('<i8')*w[take])/total) if total else None,
                'max_selected_width':int(widths[take].max()) if len(take) else None}
        views={'development':stats(np.flatnonzero(dev)),'consumed':stats(np.flatnonzero(val))}
        for name,mode in [('natural_consumed',1),('teacher_consumed',0)]:
            flag=(occ[:,7]==1)&(occ[:,6]==mode);w=np.bincount(occ[flag,1],minlength=N);views[name]=stats(np.flatnonzero(w),w)
        counts=np.bincount(m[dev,3],minlength=128)[m[:,3]];rare={}
        for name,flag in [('dev_count1_4',(counts>=1)&(counts<=4)),('dev_count5_15',(counts>=5)&(counts<=15))]:rare[name]=stats(np.flatnonzero(flag&val))
        expert_coverage=[]
        for row in rows[1:]:
            at=np.flatnonzero((m[:,3]==row['expert'])&val);row['consumed_coverage']=stats(at)
            if len(at):expert_coverage.append(float(np.mean(covered[at])))
        eligibility={name+'_certified_ge_90pct':10*views[name]['certified']>=9*views[name]['occurrences'] for name in ['consumed','natural_consumed']}
        fallbackcopies=128*H;logical={'new_region_atom_copies':copies,'original_fallback_atom_copies_ALL128':fallbackcopies,'total_atom_copies':copies+fallbackcopies,
            'I8_coefficients':2*D*(copies+fallbackcopies),'WI_F32_scales':4*(copies+fallbackcopies),'WO_F32_scales':4*D*(valid+128),'u16_region_indices':2*copies,
            'centre_I16_codes':2*D*centres,'certificate_A_n_r_bytes':24*centres,'query_dot_products':D*pair_count,'large_integer_predicates':pair_count,
            'source_WI_centre_rows':centres*H,'source_WI_centre_products':centres*H*D}
        decision='NEXT_PHYSICAL_CERTIFIED_REGION_AND_FRESH_ELIGIBILITY' if all(eligibility.values()) else 'CLOSE_THIS_FINITE_ISOTROPIC_INPUT_CONE_RECIPE'
        ctx.r['gates']['ALL_original_views_rare_fallback_logical_cost_and_frozen_economic_decision']=True
        result=ctx.finish({'experts':rows,'views':views,'rare_consumed':rare,'eligibility':eligibility,'decision':decision,'logical_cost':logical,
            'centres':centres,'valid_regions':valid,'unsupported_regions':unsupported,'equal_expert_consumed_coverage':float(np.mean(expert_coverage)),
            'worst_expert_consumed_coverage':min(expert_coverage),'new_negative_margin_centre_inputs':centres,'new_full_F_or_WO_rows':0,'model_native_GPU_calls':0,'gradient_updates':0,
            'scope':'Source-input cone geometry/economy only. Fallback3072 and complete source storage charged; no native/fresh/rate/DRAM/LUT/family promotion.'})
        print(json.dumps({'gates':result['gates'],'views':views,'centres':centres,'unsupported':unsupported,'eligibility':eligibility,'decision':decision,'logical_cost':logical,'resource':ctx.terminal_resources['resource']}),flush=True)
    except BaseException as e:ctx.done.set();ctx.watchdog.cancel();ctx.fail(e);raise
if __name__=='__main__':main()
