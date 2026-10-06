"""Independent Fraction ranking, I64 source-centre dots and exact cone comparisons."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
import argparse,hashlib,json,struct
from fractions import Fraction
from pathlib import Path
from meth504_operations import Context,ROOT,DOC,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--binding-sha',required=True);ap.add_argument('--raw-sha',required=True);a=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth504_retention',Path(a.out).resolve())
    try:
        b=ctx.admit(a.binding_sha);rp=DOC/'meth504_region_result.json';assert ctx.digest(rp)==a.raw_sha
        raw=json.loads(rp.read_bytes());assert all(raw['gates'].values())
        for v in raw['output_inventory']:ctx.exact(v)
        folder=ROOT/'results/native_expert_scaling/meth504_regions';assert json.loads((folder/'terminal_resources.json').read_bytes())['raw_sha256']==a.raw_sha
        ctx.r['gates']['sole_main_ALL_outputs_and_terminal_SHA']=True
        import numpy as np
        import threadpoolctl as T
        T.threadpool_limits(1);assert all(v['num_threads']==1 for v in T.threadpool_info());ctx.r['numerical_imports']=True
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');N,D,H,B=17540,768,3072,1536
        def wire(key,magic,width,reserved,dtype,count):
            p=Path(b['data'][key]['path']);dtype=np.dtype(dtype);assert p.stat().st_size==24+count*dtype.itemsize
            with p.open('rb') as f:assert f.read(24)==struct.pack('<8sIIQ',magic,width,reserved,count)
            return np.memmap(p,mode='r',offset=24,dtype=dtype,shape=(count,))
        m=wire('uid',b'M493U001',180,0,[('m','<u4',(13,)),('hash','u1',(128,))],N)['m']
        it=np.dtype([('e','<u4'),('accept','<u4'),('alpha','<f4'),('x','<f4',(D,)),('q','<i2',(D,))])
        ni=wire('inputs',b'M499INP1',4624,D,[('core',it),('p','<f4')],N);q=ni['core']['q']
        occ=wire('occurrences',b'M493O001',68,0,np.dtype(('<u4',(17,))),19962)
        dev=(m[:,4]&5)!=0;val=(m[:,4]&2)!=0
        assert np.array_equal(m[:,0],np.arange(N)) and np.array_equal(ni['core']['e'],m[:,3]) and np.all(ni['core']['accept']==1) and np.all(m[:,2]==11)
        assert ni['p'].tobytes()==m[:,10].tobytes() and (int(dev.sum()),int(val.sum()))==(11721,5819) and np.all(dev^val) and np.all(q>=-32767)
        assert np.array_equal(occ[:,0],np.arange(19962)) and np.all(occ[:,12]==1) and np.all(occ[:,8]==11) and np.array_equal(np.bincount(occ[:,1],minlength=N),m[:,7])
        for oi,mi in [(14,1),(11,3),(15,12),(16,10)]:assert np.array_equal(occ[:,oi],m[occ[:,1],mi])
        def arr(k):return np.load(b['data'][k]['path'],mmap_mode='r',allow_pickle=False)
        assignment,hidden,codes,scales=[arr(k) for k in ['assignment','hidden','codes','scales']]
        chosen=np.load(folder/'selected_region.npy',mmap_mode='r');widths=np.load(folder/'selected_width.npy',mmap_mode='r');norm=np.load(folder/'query_norm2.npy',mmap_mode='r')
        assert np.array_equal(norm,np.einsum('ij,ij->i',q.astype('<i8'),q.astype('<i8')))
        # Independent rational angle tests, with boundary equality strictly rejected.
        def inside(t,Q,cert):
            t,Q=int(t),int(Q)
            if Q==0:return True
            if cert['status']=='ALL_NONZERO_ROWS_RETAINED':return True
            if cert['status']!='ANGULAR_CONE' or t<=0:return False
            A,n,r=int(cert['A']),int(cert['n']),int(cert['r'])
            assert t*t<=A*Q
            return Fraction(A*Q-t*t,A*Q)<Fraction(n,r*A)
        cc={'status':'ANGULAR_CONE','A':9,'n':144,'r':25}
        assert inside(9,9,cc) and not inside(9,25,cc) and not inside(-9,9,cc) and inside(0,0,cc)
        for u in range(-6,7):
            for v in range(-6,7):
                if inside(3*u,u*u+v*v,cc) and (u or v):assert -u<0 and -4*u+3*v<0
        ctx.r['gates']['independent_Fraction_strict_boundary_orientation_and_tiny_sign_controls']=True
        ctx.r['gates']['ALL_original_roles_inputs_groups_and_I64_query_norms_independent']=True
        entries=json.loads((DOC/'meth380_switch_base128_export_result.json').read_bytes())['tensors'];payload=np.memmap(b['data']['payload']['path'],mode='r',dtype='u1')
        old=json.loads(Path(b['data']['raw503']['path']).read_bytes());centres=valid=unsupported=copies=pair_count=certified=0
        expected_chosen=np.full(N,-1,'<i4');expected_chosen[norm==0]=-2;expected_width=np.full(N,H,'<u2');expected_width[norm==0]=0
        ctx.phase='ALL128_independent_source_integer_negative_margins_rational_masks_and_predicates'
        for e,row in enumerate(raw['experts']):
            assert row['expert']==e;at=np.flatnonzero(m[:,3]==e);di=at[dev[at]]
            if not len(at):assert e==0 and row['status']=='EMPTY_UNCOMPILED_UNPROMOTED';continue
            C=old['experts'][e]['children'];groups=[c for c in range(C) if np.any(assignment[di]==c)];seeds=[]
            for c in groups:
                group=di[assignment[di]==c];z=q[group].astype('<f8');z=np.divide(z,np.linalg.norm(z,axis=1)[:,None],out=np.zeros_like(z),where=np.linalg.norm(z,axis=1)[:,None]>0)
                seeds.append(int(group[np.argmax(np.matmul(z,np.mean(z,axis=0)))]))
            seedcodes=np.load(folder/f'seed_codes_e{e:03}.npy');mask=np.load(folder/f'masks_e{e:03}.npy');d_saved=np.load(folder/f'centre_dots_e{e:03}.npy');R_saved=np.load(folder/f'row_norm2_e{e:03}.npy')
            assert np.array_equal(seedcodes,q[seeds]) and len(row['regions'])==len(seeds) and np.all(mask<=1)
            v=entries[f'decoder.block.11.layer.2.mlp.experts.expert_{e}.wi.weight'];assert v['shape']==[H,D] and v['encoding']==1
            wi=np.frombuffer(payload,'i1',v['elements'],v['offset']).reshape(H,D);si=np.frombuffer(payload,'<f4',H,v['scale_offset'])
            assert hashlib.sha256(wi.tobytes()).hexdigest()==v['sha256'] and hashlib.sha256(si.tobytes()).hexdigest()==v['scale_sha256'] and np.all(si>0) and np.isfinite(si).all()
            w64=wi.astype('<i8');R=np.einsum('ij,ij->i',w64,w64);d=(w64@seedcodes.astype('<i8').T).T
            assert np.array_equal(R,R_saved) and np.array_equal(d,d_saved)
            for c,k in enumerate(seeds):
                required=[j for j in range(H) if R[j] and d[c,j]>=0];negative=[j for j in range(H) if R[j] and d[c,j]<0]
                ordered=sorted(negative,key=lambda j:(Fraction(int(d[c,j])**2,int(R[j])),j));mk=np.zeros(H,'u1');A=int(norm[k])
                cert={'region':c,'source_group':groups[c],'seed_UID':k}
                if A==0:cert.update(status='ZERO_CENTRE_UNSUPPORTED',A=0)
                elif len(required)>B:cert.update(status='CENTRE_NONNEGATIVE_WIDTH_ABOVE_B',A=A,required=len(required))
                else:
                    take=min(B-len(required),len(ordered));mk[required+ordered[:take]]=1;omitted=ordered[take:]
                    if omitted:
                        j=omitted[0];cert.update(status='ANGULAR_CONE',A=A,required=len(required),limiting_neuron=j,n=int(d[c,j])**2,r=int(R[j]))
                    else:cert.update(status='ALL_NONZERO_ROWS_RETAINED',A=A,required=len(required))
                cert['width']=int(mk.sum());assert cert==row['regions'][c] and np.array_equal(mk,mask[c]) and cert['width']<=B
                good=cert['status'] in ['ANGULAR_CONE','ALL_NONZERO_ROWS_RETAINED'];valid+=good;unsupported+=not good
            dots=q[at].astype('<i8')@seedcodes.astype('<i8').T;assert np.array_equal(dots,np.load(folder/f'query_dots_e{e:03}.npy'))
            pred=np.array([[inside(dots[i,c],norm[k],cert) for c,cert in enumerate(row['regions'])] for i,k in enumerate(at)],'u1')
            assert np.array_equal(pred,np.load(folder/f'predicates_e{e:03}.npy'))
            for i,k in enumerate(at):
                if norm[k]==0:assert not codes[k].any();continue
                matches=np.flatnonzero(pred[i])
                if len(matches):c=int(matches[0]);expected_chosen[k]=c;expected_width[k]=row['regions'][c]['width']
            # Every selected certified input: rederive omitted integer signs from
            # original coefficients, independent of any saved support or main bound.
            for c,cert in enumerate(row['regions']):
                take=at[expected_chosen[at]==c]
                if not len(take):continue
                omitted=(mask[c]==0)&(R>0);j=np.flatnonzero(mask[c]);wc=wi[omitted].astype('<f8')
                for k in range(0,len(take),128):
                    ix=take[k:k+128];actual=q[ix].astype('<f8')@wc.T
                    assert np.array_equal(actual,np.rint(actual)) and np.all(actual<0)
                    assert not codes[ix][:,mask[c]==0].any() and np.all(hidden[ix][:,mask[c]==0]==0)
                    h=hidden[ix][:,j];maxv=np.amax(np.abs(h),axis=1) if len(j) else np.zeros(len(ix),'<f4');s=np.divide(maxv,np.float32(32767),dtype=np.float32);s[maxv==0]=1
                    qc=np.clip(np.rint(h/s[:,None]),-32767,32767).astype('<i2')
                    assert qc.tobytes()==codes[ix][:,j].tobytes() and s.tobytes()==scales[ix].tobytes();certified+=len(ix);ctx.guard()
            centres+=len(seeds);copies+=int(mask.sum());pair_count+=len(at)*len(seeds);ctx.guard()
            if e%32==31:print(json.dumps({'expert':e,'seconds':ctx.resources()['wall_seconds']}),flush=True)
        assert np.array_equal(chosen,expected_chosen) and np.array_equal(widths,expected_width) and centres==raw['centres']==637 and valid==raw['valid_regions'] and unsupported==raw['unsupported_regions']
        assert certified==int(np.sum(chosen>=0))
        ctx.r['gates']['ALL637_NEW_negative_margins_Fraction_masks_and_input_only_seeds_independent']=True
        ctx.r['gates']['ALL_predicates_chosen_query_actual_omitted_source_integer_signs_hidden_zero_and_scale_BYTE']=True
        covered=chosen!=-1
        def verify(view,take,w=None):
            if w is None:w=np.ones(N,dtype='<i8')
            total=int(w[take].sum());hit=int(w[take][covered[take]].sum())
            assert view=={'UIDs':len(take),'occurrences':total,'certified':hit,'certified_fraction':hit/total if total else None,'fallback':total-hit,
                'mean_selected_width':float(np.sum(widths[take].astype('<i8')*w[take])/total) if total else None,'max_selected_width':int(widths[take].max()) if len(take) else None}
        for name,flag in [('development',dev),('consumed',val)]:verify(raw['views'][name],np.flatnonzero(flag))
        for name,mode in [('natural_consumed',1),('teacher_consumed',0)]:
            w=np.bincount(occ[(occ[:,7]==1)&(occ[:,6]==mode),1],minlength=N);verify(raw['views'][name],np.flatnonzero(w),w)
        counts=np.bincount(m[dev,3],minlength=128)[m[:,3]]
        for name,flag in [('dev_count1_4',(counts>=1)&(counts<=4)),('dev_count5_15',(counts>=5)&(counts<=15))]:verify(raw['rare_consumed'][name],np.flatnonzero(flag&val))
        ratios=[]
        for row in raw['experts'][1:]:
            at=np.flatnonzero((m[:,3]==row['expert'])&val);verify(row['consumed_coverage'],at)
            if len(at):ratios.append(float(np.mean(covered[at])))
        assert raw['equal_expert_consumed_coverage']==float(np.mean(ratios)) and raw['worst_expert_consumed_coverage']==min(ratios)
        fc=128*H;expected={'new_region_atom_copies':copies,'original_fallback_atom_copies_ALL128':fc,'total_atom_copies':copies+fc,'I8_coefficients':2*D*(copies+fc),
            'WI_F32_scales':4*(copies+fc),'WO_F32_scales':4*D*(valid+128),'u16_region_indices':2*copies,'centre_I16_codes':2*D*centres,'certificate_A_n_r_bytes':24*centres,
            'query_dot_products':D*pair_count,'large_integer_predicates':pair_count,'source_WI_centre_rows':centres*H,'source_WI_centre_products':centres*H*D}
        assert raw['logical_cost']==expected
        eligibility={name+'_certified_ge_90pct':10*raw['views'][name]['certified']>=9*raw['views'][name]['occurrences'] for name in ['consumed','natural_consumed']}
        assert eligibility==raw['eligibility'];decision='NEXT_PHYSICAL_CERTIFIED_REGION_AND_FRESH_ELIGIBILITY' if all(eligibility.values()) else 'CLOSE_THIS_FINITE_ISOTROPIC_INPUT_CONE_RECIPE';assert decision==raw['decision']
        ctx.r['gates']['ALL_domain_views_rare_fallback_cost_104bit_budget_and_frozen_economy_independent']=True
        result=ctx.finish({'main_completed':True,'raw':{'path':str(rp),'sha256':a.raw_sha},'views':raw['views'],'rare_consumed':raw['rare_consumed'],'logical_cost':expected,
            'eligibility':eligibility,'decision':decision,'unique_inputs_audited':N,'certified_inputs_actual_omitted_signs_audited':certified,'new_full_F_or_WO_rows':0,
            'scope':'Independent angular-region geometry/cost only; failed economy cannot justify physical implementation.'})
        print(json.dumps({'gates':result['gates'],'views':result['views'],'decision':decision,'resource':ctx.terminal_resources['resource']}),flush=True)
    except BaseException as e:ctx.done.set();ctx.watchdog.cancel();ctx.fail(e);raise
if __name__=='__main__':main()
