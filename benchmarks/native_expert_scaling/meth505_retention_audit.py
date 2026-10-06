"""Independent file inversion, original gold joins and native cost admission; no main import."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
import argparse,json,struct
from pathlib import Path
from meth505_operations import Context,ROOT,DOC,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--binding-sha',required=True);a=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth505_retention',Path(a.out).resolve())
    try:
        b=ctx.admit(a.binding_sha);rp=DOC/'meth505_exact_sparse_result.json';ctx.head(rp);raw=json.loads(rp.read_bytes());rdesc={'path':str(rp),'bytes':rp.stat().st_size,'sha256':ctx.digest(rp)}
        regpath=DOC/'meth505_main_tool_registration.json';ctx.head(regpath);reg=json.loads(regpath.read_bytes())
        assert reg['actual_exit_code']==0 and reg['wait_tool_results'][-1]['exit_code']==0 and all(raw['gates'].values()) and len(raw['commands'])==3
        assert raw['binding_sha256']==a.binding_sha and raw['commands'][0]['label']=='compile' and raw['commands'][1]['label']=='primal' and raw['commands'][2]['label']=='bench'
        inv={Path(v['path']).name:v for v in raw['output_inventory']}
        for v in inv.values():ctx.exact(v)
        terminal=ROOT/'results/native_expert_scaling/meth505_exact_sparse/terminal_resources.json';t=json.loads(terminal.read_bytes());assert t['raw_sha256']==rdesc['sha256'] and t['resource']['wall_seconds']<=900 and t['resource']['parent_peak_bytes']+t['resource']['native_peak_bytes']<=4<<30
        ctx.r['gates']['actual_main_exit0_full_inventory_and_terminal_resource_SHA']=True
        import numpy as np
        import threadpoolctl as T
        T.threadpool_limits(1);pools=T.threadpool_info();assert pools and all(v['num_threads']==1 for v in pools)
        for v in pools:assert any(d['path']==str(Path(v['filepath']).resolve()) for d in b['catalog'])
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.r['numerical_imports']=True
        N,D,H,E=17540,768,3072,128;root=ROOT/'results/native_expert_scaling/meth505_exact_sparse'
        def read(key,magic,width,res,n,dt):
            p=Path(b['data'][key]['path']);assert p.stat().st_size==24+n*width
            with p.open('rb') as f:assert struct.unpack('<8sIIQ',f.read(24))==(magic,width,res,n)
            return np.memmap(p,offset=24,mode='r',dtype=dt,shape=(n,))
        u=read('uid',b'M493U001',180,0,N,np.dtype([('m','<u4',(13,)),('sha','u1',(128,))]));m=u['m']
        oc=read('occurrences',b'M493O001',68,0,19962,np.dtype(('<u4',(17,))))
        it=np.dtype([('e','<u4'),('a','<u4'),('alpha','<f4'),('x','<f4',(D,)),('q','<i2',(D,)),('p','<f4')]);inp=read('inputs',b'M499INP1',4624,D,N,it)
        gold=read('targets',b'M499F001',3072,D,N,np.dtype(('<f4',(D,))))
        development=(m[:,4]&5)>0;consumed=(m[:,4]&2)>0
        assert np.array_equal(m[:,0],np.arange(N)) and np.all(development^consumed) and (int(development.sum()),int(consumed.sum()))==(11721,5819)
        assert np.array_equal(inp['e'],m[:,3]) and inp['p'].tobytes()==m[:,10].tobytes() and inp['alpha'].tobytes()==m[:,12].tobytes() and np.all(inp['a']==1)
        assert np.array_equal(oc[:,0],np.arange(19962)) and np.all(oc[:,8]==11) and np.all(oc[:,12]==1) and np.array_equal(np.bincount(oc[:,1],minlength=N),m[:,7])
        for oi,mi in [(14,1),(11,3),(15,12),(16,10)]:assert np.array_equal(oc[:,oi],m[oc[:,1],mi])
        assert np.all(oc[:,4]<192) and np.all(oc[:,6]<=1) and np.array_equal(oc[:,7],np.where(oc[:,4]>=128,2,np.where(oc[:,4]>=64,1,0)))
        ctx.r['gates']['ALL_original_UID_occurrence_mass_roles_book_domains_preserved']=True
        ctx.phase='independent-bank-inverse';bank=np.memmap(root/'bank.bin',mode='r',dtype='u1');payload=np.memmap(b['data']['payload']['path'],mode='r',dtype='u1')
        assert len(bank)==605949952 and bank[:32].tobytes()==struct.pack('<8sIIIIQ',b'M505BNK1',D,H,E,4096,4733952) and not bank[32:4096].any()
        entries=json.loads((DOC/'meth380_switch_base128_export_result.json').read_bytes())['tensors']
        for e in range(E):
            base=4096+e*4733952
            for name,offset,so,shape in [('wi',0,2359296,(H,D)),('wo',2371584,4730880,(D,H))]:
                v=entries[f'decoder.block.11.layer.2.mlp.experts.expert_{e}.{name}.weight'];assert v['shape']==list(shape) and v['encoding']==1
                source=payload[v['offset']:v['offset']+D*H].reshape(shape);physical=bank[base+offset:base+offset+D*H].reshape((H,D))
                if name=='wo':physical=physical.T
                assert source.tobytes()==physical.tobytes() and payload[v['scale_offset']:v['scale_offset']+v['scale_bytes']].tobytes()==bank[base+so:base+so+v['scale_bytes']].tobytes()
                assert (base+offset)%64==(base+so)%64==0;ctx.guard()
        ctx.r['gates']['ALL128_independent_source_I8_inverse_positive_scales_header_padding_extents']=True
        ctx.phase='independent-primal'
        pt=np.dtype([('id','<u4'),('e','<u4'),('nz','<u4'),('h','<f4',(H,)),('q','<i2',(H,)),('alpha','<f4'),('F','<f4',(D,))]);assert pt.itemsize==21520
        pp=root/'primal.bin'
        with pp.open('rb') as f:assert f.read(24)==struct.pack('<8sIIQ',b'M505OUT1',21520,D,N)
        assert pp.stat().st_size==377460824;pr=np.memmap(pp,mode='r',offset=24,dtype=pt,shape=(N,))
        hs,qs,al=[np.load(b['data'][k]['path'],mmap_mode='r',allow_pickle=False) for k in ['hidden','codes','scales']];assert hs.shape==qs.shape==(N,H) and al.shape==(N,)
        nz=np.zeros(N,dtype='<u4')
        for start in range(0,N,64):
            end=min(N,start+64);v=pr[start:end]
            assert np.array_equal(v['id'],np.arange(start,end)) and v['e'].tobytes()==inp['e'][start:end].tobytes()
            assert v['h'].tobytes()==hs[start:end].tobytes() and v['q'].tobytes()==qs[start:end].tobytes() and v['alpha'].tobytes()==al[start:end].tobytes() and v['F'].tobytes()==gold[start:end].tobytes()
            maximum=np.max(np.abs(v['h']),axis=1);scale=np.where(maximum==0,np.float32(1),maximum/np.float32(32767)).astype('<f4')
            reconstructed=np.clip(np.rint(v['h']/scale[:,None]),-32767,32767).astype('<i2')
            assert reconstructed.tobytes()==v['q'].tobytes() and scale.tobytes()==v['alpha'].tobytes() and np.all(v['h']>=0)
            nz[start:end]=np.sum(v['q']!=0,axis=1);assert np.array_equal(nz[start:end],v['nz']);ctx.guard()
        assert nz.tobytes()==np.load(root/'active.npy',allow_pickle=False).tobytes()
        assert raw['native_primal']['primal_controls'] and raw['native_primal']['bank_bytes']==605949952
        ctx.r['gates']['ALL17540_independent_native_hidden_A16_scale_F_BYTE_and_exact_zero_count']=True
        ctx.phase='independent-cost'
        tp=root/'timing.bin'
        with tp.open('rb') as f:assert f.read(24)==struct.pack('<8sIIQ',b'M505TIM1',8,3,6*N)
        assert tp.stat().st_size==841944;times=np.memmap(tp,mode='r',offset=24,dtype='<f8',shape=(3,2,N));assert np.isfinite(times).all() and np.all(times>0)
        # Independent sorting/interpolated quantile rather than main's np.quantile.
        def stats(ids):
            ids=np.asarray(ids,np.int64);count=len(ids)
            if not count:return {'UIDs':0,'occurrences':0}
            a0=times[:,0,ids].reshape(-1);a1=times[:,1,ids].reshape(-1)
            means=[float(a0.mean()),float(a1.mean())];p95=[]
            for vector in [a0,a1]:
                order=np.sort(vector);pos=(len(order)-1)*.95;lo=int(pos);delta=pos-lo
                # Match NumPy linear interpolation's stable half interval evaluation.
                left,right=order[lo],order[min(lo+1,len(order)-1)];diff=right-left
                p95.append(float(right-diff*(1-delta) if delta>=.5 else left+diff*delta))
            active=nz[ids].astype(np.int64);weightbytes=D*H+D*active+4*(H+D);lines=(D*H)//64+(D//64)*active+4*(H+D)//64
            return {'UIDs':len(np.unique(ids)),'occurrences':count,'baseline_mean_seconds':means[0],'candidate_mean_seconds':means[1],'mean_ratio':means[1]/means[0],
                    'baseline_p95_seconds':p95[0],'candidate_p95_seconds':p95[1],'p95_ratio':p95[1]/p95[0],'mean_active':float(active.mean()),'max_active':int(active.max()),
                    'candidate_mean_addressed_weight_bytes':float(weightbytes.mean()),'candidate_mean_addressed_weight_lines64':float(lines.mean()),'baseline_addressed_weight_bytes':2*D*H+4*(H+D),'baseline_addressed_weight_lines64':(2*D*H+4*(H+D))//64}
        views={'all':stats(np.arange(N)),'development':stats(np.where(development)[0]),'consumed':stats(np.where(consumed)[0])}
        for name,mode in [('natural_consumed',1),('teacher_consumed',0)]:views[name]=stats(oc[(oc[:,7]==1)&(oc[:,6]==mode),1])
        books=[{'book':i,**stats(oc[oc[:,4]==i,1])} for i in range(192)]
        devcounts=np.bincount(m[development,3],minlength=E)[m[:,3]];rare={name:stats(np.where(consumed&flag)[0]) for name,flag in [('dev_count1_4',(devcounts>=1)&(devcounts<=4)),('dev_count5_15',(devcounts>=5)&(devcounts<=15))]}
        experts=[{'expert':e,'development':int(np.sum(development&(m[:,3]==e))),**stats(np.where(consumed&(m[:,3]==e))[0])} for e in range(E)]
        def same(actual,expected):
            assert actual.keys()==expected.keys()
            for k in actual:
                if isinstance(actual[k],dict):same(actual[k],expected[k])
                elif isinstance(actual[k],float):assert abs(actual[k]-expected[k])<=max(abs(actual[k])*2e-14,1e-15),(k,actual[k],expected[k])
                else:assert actual[k]==expected[k],k
        same(views,raw['views']);same(rare,raw['rare_consumed'])
        for av,rv in zip(books,raw['books'],strict=True):same(av,rv)
        for av,rv in zip(experts,raw['experts'],strict=True):same(av,rv)
        gate={name+'_mean_le_0_80':views[name]['mean_ratio']<=.80 for name in ['consumed','natural_consumed']}
        gate.update({name+'_p95_le_1_00':views[name]['p95_ratio']<=1 for name in ['consumed','natural_consumed']});gate['every_source_book_mean_le_1_00']=all(v.get('mean_ratio',0)<=1 for v in books)
        sourcebytes=E*(2*D*H+4*(H+D));gate['storage_ratio_le_1_01']=605949952/sourcebytes<=1.01;assert gate==raw['eligibility']
        ctx.r['gates']['independent_ALL_times_book_rare_domains_address_sets_and_frozen_gates']=True
        assert raw['native_bench']['timed_output_comparisons']==140320 and np.array_equal(times.sum(axis=2),raw['per_sweep_arm_seconds'])
        for v in raw['observed_bound_modules']:ctx.exact(v)
        for stage in ['native_primal','native_bench']:
            n=raw[stage];assert len(n['worker_affinity'])==1 and n['worker_affinity'][0]['actual_mask']==1 and n['native_peak_bytes']<=2<<30 and n['native_seconds']<=120
        ctx.r['gates']['actual_native_module_SHA_scalar_controls_inline_checks_CPU0_and_process_bounds']=True
        decision='INTEGRATE_ALL_BANKS_HEAD_FRESH_SAME_ARTIFACT_RATE' if all(gate.values()) else 'CLOSE_THIS_ORIGINAL_WI_SPARSE_WO_LOCAL_COST_RECIPE';assert raw['decision']==decision
        result=ctx.finish({'main_completed':True,'raw':rdesc,'unique_inputs_audited':N,'views':raw['views'],'books':raw['books'],'rare_consumed':raw['rare_consumed'],'experts':raw['experts'],'storage':raw['storage'],'eligibility':gate,'decision':decision,
                           'independent_metric_tolerance':2e-14,'scope':'Physical bank and complete local native function/cost. No fresh whole-model/useful n/normalized LUT/DRAM/family promotion.'})
        print(json.dumps({'gates':result['gates'],'eligibility':gate,'decision':decision,'resource':result['resource']}))
    except BaseException as e:ctx.fail(e);raise
if __name__=='__main__':main()
