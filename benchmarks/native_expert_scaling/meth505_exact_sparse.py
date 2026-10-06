"""ONE new physical original-WI/column-WO primal and matched full-trace cost."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
import argparse,gc,hashlib,json,shutil,struct
from pathlib import Path
from meth505_operations import Context,ROOT,DOC,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--binding-sha',required=True);a=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth505_exact_sparse',Path(a.out).resolve())
    try:
        b=ctx.admit(a.binding_sha)
        import numpy as np
        import threadpoolctl as T
        T.threadpool_limits(1);pools=T.threadpool_info();assert pools and all(v['num_threads']==1 for v in pools)
        for v in pools:assert any(d['path']==str(Path(v['filepath']).resolve()) for d in b['catalog'])
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.r['numerical_imports']=True;ctx.r['runtime']={'BLAS':pools,'CPU_affinity':[0],'environment':b['runtime_environment']}
        N,D,H,E,STRIDE=17540,768,3072,128,4733952
        def wire(key,magic,stride,res,dtype,n):
            p=Path(b['data'][key]['path']);assert p.stat().st_size==24+stride*n and p.open('rb').read(24)==struct.pack('<8sIIQ',magic,stride,res,n)
            return np.memmap(p,mode='r',offset=24,dtype=dtype,shape=(n,))
        it=np.dtype([('e','<u4'),('accept','<u4'),('alpha','<f4'),('x','<f4',(D,)),('q','<i2',(D,)),('p','<f4')])
        inputs=wire('inputs',b'M499INP1',4624,D,it,N);target=wire('targets',b'M499F001',3072,D,np.dtype(('<f4',(D,))),N)
        uid=wire('uid',b'M493U001',180,0,np.dtype([('m','<u4',(13,)),('hash','u1',(128,))]),N);m=uid['m']
        occ=wire('occurrences',b'M493O001',68,0,np.dtype(('<u4',(17,))),19962)
        assert np.array_equal(m[:,0],np.arange(N)) and np.all(m[:,2]==11) and np.array_equal(m[:,3],inputs['e']) and np.all(inputs['accept']==1)
        assert np.array_equal(m[:,10],inputs['p'].view('<u4')) and np.array_equal(m[:,12],inputs['alpha'].view('<u4'))
        assert np.array_equal(occ[:,0],np.arange(19962)) and np.all(occ[:,8]==11) and np.all(occ[:,12]==1) and np.array_equal(np.bincount(occ[:,1],minlength=N),m[:,7])
        for oi,mi in [(14,1),(11,3),(15,12),(16,10)]:assert np.array_equal(occ[:,oi],m[occ[:,1],mi])
        dev=(m[:,4]&5)!=0;cons=(m[:,4]&2)!=0;assert np.all(dev^cons) and (int(dev.sum()),int(cons.sum()))==(11721,5819)
        ctx.r['gates']['ALL_original_UID_occurrence_parent_mass_input_quant_roles']=True
        ctx.phase='physical-bank'
        entries=json.loads((DOC/'meth380_switch_base128_export_result.json').read_bytes())['tensors'];payload=np.memmap(b['data']['payload']['path'],mode='r',dtype='u1')
        bank=ctx.out/'bank.bin';extents=[]
        with bank.open('xb') as f:
            header=struct.pack('<8sIIIIQ',b'M505BNK1',D,H,E,4096,STRIDE);f.write(header+bytes(4096-len(header)))
            for e in range(E):
                for op in ['wi','wo']:
                    v=entries[f'decoder.block.11.layer.2.mlp.experts.expert_{e}.{op}.weight'];shape=(H,D) if op=='wi' else (D,H)
                    assert v['shape']==list(shape) and v['encoding']==1 and v['bytes']==D*H
                    raw=payload[v['offset']:v['offset']+v['bytes']].tobytes();sc=payload[v['scale_offset']:v['scale_offset']+v['scale_bytes']].tobytes()
                    assert hashlib.sha256(raw).hexdigest()==v['sha256'] and hashlib.sha256(sc).hexdigest()==v['scale_sha256']
                    scales=np.frombuffer(sc,'<f4');assert np.isfinite(scales).all() and np.all(scales>0)
                    coded=np.frombuffer(raw,'i1').reshape(shape);data=raw if op=='wi' else coded.T.copy(order='C').tobytes()
                    assert f.tell()%64==0;offset=f.tell();f.write(data);scaleoffset=f.tell();f.write(sc)
                    extents.append({'expert':e,'op':op,'offset':offset,'scale_offset':scaleoffset,'source':v})
                assert f.tell()==4096+(e+1)*STRIDE;ctx.guard()
        assert bank.stat().st_size==605949952
        physical=np.memmap(bank,mode='r',dtype='u1')
        for v in extents:
            s=v['source'];raw=physical[v['offset']:v['offset']+D*H].tobytes()
            if v['op']=='wo':raw=np.frombuffer(raw,'i1').reshape(H,D).T.copy(order='C').tobytes()
            assert raw==payload[s['offset']:s['offset']+D*H].tobytes()
            assert physical[v['scale_offset']:v['scale_offset']+s['scale_bytes']].tobytes()==payload[s['scale_offset']:s['scale_offset']+s['scale_bytes']].tobytes()
            ctx.guard()
        write(ctx.out/'extents.json',extents);del physical,payload,raw,sc,coded,data;gc.collect()
        ctx.r['gates']['ALL128_actual_I8_bank_inverse_scales_alignment_no_information_loss']=True
        ctx.phase='compile';os.environ.update(b['runtime_environment']);os.environ['PATH']=str(Path(b['compiler']['path']).parent)+os.pathsep+os.environ.get('PATH','')
        dll=ctx.out/'libomp.dll';shutil.copyfile(b['libomp']['path'],dll);assert ctx.digest(dll)==b['libomp']['sha256']
        binary=ctx.out/'meth505_exact_sparse_cpu.exe';source=ROOT/'benchmarks/native_expert_scaling/meth505_exact_sparse_cpu.c'
        ctx.run([b['compiler']['path'],'-O3','-std=c11','-march=x86-64-v3','-fno-fast-math','-ffp-contract=off','-fopenmp',str(source),'-lpsapi','-o',str(binary)],'compile')
        binarysha=ctx.digest(binary);ctx.r['binary']={'path':str(binary),'bytes':binary.stat().st_size,'sha256':binarysha}
        argv=[str(binary),'--primal',b['data']['manifest']['path'],str(bank),b['data']['inputs']['path'],b['data']['targets']['path'],str(ctx.out/'primal.bin')]
        ctx.phase='native-primal';primal=json.loads(ctx.run(argv,'primal')[-1]);assert primal['primal_controls'] and primal['worker_affinity'][0]['actual_mask']==1
        pt=np.dtype([('uid','<u4'),('e','<u4'),('nonzero','<u4'),('hidden','<f4',(H,)),('codes','<i2',(H,)),('alpha','<f4'),('down','<f4',(D,))]);assert pt.itemsize==21520
        p=ctx.out/'primal.bin';assert p.stat().st_size==377460824 and p.open('rb').read(24)==struct.pack('<8sIIQ',b'M505OUT1',21520,D,N)
        wireout=np.memmap(p,mode='r',offset=24,dtype=pt,shape=(N,));hidden,codes,scales=[np.load(b['data'][k]['path'],mmap_mode='r',allow_pickle=False) for k in ['hidden','codes','scales']]
        for lo in range(0,N,128):
            hi=min(N,lo+128);q=wireout[lo:hi]
            assert np.array_equal(q['uid'],np.arange(lo,hi)) and q['e'].tobytes()==inputs['e'][lo:hi].tobytes()
            assert q['hidden'].tobytes()==hidden[lo:hi].tobytes() and q['codes'].tobytes()==codes[lo:hi].tobytes() and q['alpha'].tobytes()==scales[lo:hi].tobytes() and q['down'].tobytes()==target[lo:hi].tobytes()
            assert np.array_equal(q['nonzero'],np.count_nonzero(codes[lo:hi],axis=1));ctx.guard()
        active=wireout['nonzero'].copy();np.save(ctx.out/'active.npy',active);del wireout,hidden,codes,scales,q;gc.collect()
        ctx.r['gates']['NEW_native_scalar_extrema_controls_and_ALL17540_hidden_codes_alpha_F_BYTE']=True
        ctx.phase='matched-cost';argv[1]='--bench';argv[-1]=str(ctx.out/'timing.bin');bench=json.loads(ctx.run(argv,'bench')[-1]);assert bench['timed_output_comparisons']==140320 and bench['worker_affinity'][0]['actual_mask']==1
        tp=ctx.out/'timing.bin';assert tp.stat().st_size==841944 and tp.open('rb').read(24)==struct.pack('<8sIIQ',b'M505TIM1',8,3,6*N)
        times=np.memmap(tp,mode='r',offset=24,dtype='<f8',shape=(3,2,N));assert np.isfinite(times).all() and np.all(times>0)
        allowed={str(Path(v['path']).resolve()).lower():v for v in b['catalog']};allowed[str(binary).lower()]=ctx.r['binary'];allowed[str(dll).lower()]={'path':str(dll),'bytes':dll.stat().st_size,'sha256':b['libomp']['sha256']}
        observed=[]
        for cmd in ctx.r['commands']:
            assert cmd['returncode']==0 and cmd['observed_modules']
            for path in cmd['observed_modules']:
                v=allowed.get(str(Path(path).resolve()).lower());assert v is not None,('unbound_actual_module',path);ctx.exact(v);observed.append(v)
        ctx.r['gates']['actual_compiler_native_modules_affinity_OS_peaks_and_every_timed_F_BYTE']=True
        ctx.phase='statistics'
        def stats(ids):
            ids=np.asarray(ids,dtype=np.int64);k=len(ids)
            if not k:return {'UIDs':0,'occurrences':0}
            t=times[:,:,ids];means=t.mean(axis=(0,2));p95=np.quantile(t.transpose(1,0,2).reshape(2,-1),.95,axis=1,method='linear')
            ac=active[ids].astype(np.int64);cb=D*H+D*ac+4*(H+D);cl=(D*H)//64+(D//64)*ac+4*(H+D)//64
            return {'UIDs':len(set(ids.tolist())),'occurrences':k,'baseline_mean_seconds':float(means[0]),'candidate_mean_seconds':float(means[1]),'mean_ratio':float(means[1]/means[0]),
                    'baseline_p95_seconds':float(p95[0]),'candidate_p95_seconds':float(p95[1]),'p95_ratio':float(p95[1]/p95[0]),'mean_active':float(ac.mean()),'max_active':int(ac.max()),
                    'candidate_mean_addressed_weight_bytes':float(cb.mean()),'candidate_mean_addressed_weight_lines64':float(cl.mean()),'baseline_addressed_weight_bytes':2*D*H+4*(H+D),'baseline_addressed_weight_lines64':(2*D*H+4*(H+D))//64}
        views={'all':stats(np.arange(N)),'development':stats(np.flatnonzero(dev)),'consumed':stats(np.flatnonzero(cons))}
        for name,mode in [('natural_consumed',1),('teacher_consumed',0)]:views[name]=stats(occ[(occ[:,7]==1)&(occ[:,6]==mode),1])
        books=[{'book':book,**stats(occ[occ[:,4]==book,1])} for book in range(192)]
        counts=np.bincount(m[dev,3],minlength=E)[m[:,3]];rare={name:stats(np.flatnonzero(cons&flag)) for name,flag in [('dev_count1_4',(counts>=1)&(counts<=4)),('dev_count5_15',(counts>=5)&(counts<=15))]}
        experts=[{'expert':e,'development':int(np.sum(dev&(m[:,3]==e))),**stats(np.flatnonzero(cons&(m[:,3]==e)))} for e in range(E)]
        gates={name+'_mean_le_0_80':views[name]['mean_ratio']<=.80 for name in ['consumed','natural_consumed']}
        gates.update({name+'_p95_le_1_00':views[name]['p95_ratio']<=1 for name in ['consumed','natural_consumed']});gates['every_source_book_mean_le_1_00']=all(v.get('mean_ratio',0)<=1 for v in books)
        sourcebytes=E*(2*D*H+4*(H+D));gates['storage_ratio_le_1_01']=bank.stat().st_size/sourcebytes<=1.01
        ctx.r['gates']['all_frozen_domains_times_denominators_no_query_filter_or_extra_sweep']=True
        value={'native_primal':primal,'native_bench':bench,'observed_bound_modules':observed,'unique_inputs':N,'occurrences':19962,'exposed_IDs':len(set(m[:,3].tolist())),
               'views':views,'books':books,'rare_consumed':rare,'experts':experts,'eligibility':gates,'storage':{'candidate_bytes':bank.stat().st_size,'source_bank_bytes':sourcebytes,'ratio':bank.stat().st_size/sourcebytes,'simultaneously_retained_source_and_candidate_bytes':sourcebytes+bank.stat().st_size},
               'per_sweep_arm_seconds':times.sum(axis=2).tolist(),'decision':'INTEGRATE_ALL_BANKS_HEAD_FRESH_SAME_ARTIFACT_RATE' if all(gates.values()) else 'CLOSE_THIS_ORIGINAL_WI_SPARSE_WO_LOCAL_COST_RECIPE',
               'scope':'Exact one-bank operator and local CPU0 trace cost only. Addressed weight lines are actual layout access sets, not hardware DRAM counters. Whole model/fresh/routing mass/useful n/family gates remain open.'}
        r=ctx.finish(value);print(json.dumps({'gates':r['gates'],'eligibility':gates,'views':views,'decision':r['decision'],'resource':r['resource']}))
    except BaseException as e:ctx.fail(e);raise
if __name__=='__main__':main()
