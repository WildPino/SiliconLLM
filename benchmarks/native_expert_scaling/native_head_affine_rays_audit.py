"""Every ray constraint and independent scalar FMA native-bit audit."""
import argparse,ctypes,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; B=ROOT/'benchmarks/native_expert_scaling'; DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'; SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write
START=time.monotonic()


def main(a):
    import numpy as np,psutil
    assert extent(a.report)['sha256']==a.report_sha and extent(a.terminal)['sha256']==a.terminal_sha
    r=json.loads(a.report.read_bytes()); terminal=json.loads(a.terminal.read_bytes()); assert terminal['exit_code']==0 and terminal['result']==extent(a.report)
    assert extent(r['binding']['path'])==r['binding']; b=json.loads(Path(r['binding']['path']).read_bytes())
    proc=psutil.Process(); proc.cpu_affinity(list(range(6))); mapping=None; maxrel=maxscalar=0.
    extra=[extent(__file__),extent(DOC/'NATIVE_HEAD_AFFINE_RAYS_AUDIT_PROTOCOL_20261012.md'),extent(a.report),extent(a.terminal),r['binding']]
    receipt=dict(schema='NATIVE_HEAD_AFFINE_RAYS_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():assert time.monotonic()-START<=900 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
    def check():
        for e in b['inputs']+r['outputs']+extra:guard(); assert extent(e['path'])==e,e['path']
    def load(e):
        with np.load(e['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def compare(x,y,tolerance=1e-9):
        nonlocal maxrel
        assert x.shape==y.shape and np.array_equal(np.isnan(x),np.isnan(y)) and np.array_equal(np.isinf(x),np.isinf(y))
        special=~np.isfinite(x); assert np.array_equal(x[special],y[special],equal_nan=True)
        finite=~special; error=float(np.linalg.norm(x[finite]-y[finite])/max(1.,float(np.linalg.norm(y[finite])))); maxrel=max(maxrel,error); assert error<=tolerance,error
    def scalar(x,y,tolerance=1e-8):
        nonlocal maxscalar
        error=abs(float(x)-float(y)); maxscalar=max(maxscalar,error); assert error<=tolerance*max(1.,abs(float(y))),(x,y)
    def interval(aa,bb,j):
        lower=0.; upper=math.inf; zero=False
        # Independent extrema with masked array reductions,not producer helper.
        use=np.ones(65537,bool); use[j]=False
        pp=(aa>0)&use; nn=(aa<0)&use
        if pp.any():lower=max(0.,float(np.max(bb[pp]/aa[pp])))
        if nn.any():upper=float(np.min(bb[nn]/aa[nn]))
        zero=bool(np.any((aa==0)&use&(bb>0)))
        return lower,upper,not zero and math.isfinite(lower) and lower<=upper
    def partial(t,C):
        score=np.zeros((len(t),65537),'f8')
        for col in range(0,256,32):score+=t[:,col:col+32]@C[:,col:col+32].T
        return score
    try:
        check(); p=load(r['codes']); native=b['prior']['native_report']; C32=np.fromfile(native['head_F32']['path'],'<f4').reshape(65537,256)
        beta32=np.fromfile(native['bias_F32']['path'],'<f4'); epsilon=float(np.fromfile(native['epsilon_F32']['path'],'<f4')[0]); C=C32.astype('f8'); beta=beta32.astype('f8')
        targets=np.array(b['targets'],'i8'); assert np.array_equal(p['targets'],targets) and len(targets)==2361
        center=np.zeros(256,'f8')
        for lo in range(0,65537,2048):center+=C[lo:lo+2048].sum(axis=0)
        center/=65537; compare(p['center'],center,1e-12)
        directions=C[targets]-center; lengths=np.sqrt(np.einsum('ij,ij->i',directions,directions)); directions/=lengths[:,None]
        compare(p['directions'],directions,1e-12); compare(p['direction_lengths'],lengths,1e-12)
        reconstructed_rows=np.full(len(targets),-1,'i8'); scalars=np.full(len(targets),np.nan,'f8'); states=[]; counts=[0,0]
        for row,j in enumerate(targets):
            guard()
            # Subtract actual row coefficients FIRST,then sum 256 products.
            # Exact identical rows therefore produce zero without cancellation.
            aa=np.einsum('ij,j->i',C[j]-C,directions[row],optimize=False); rhs=beta-beta[j]
            for mi,margin in enumerate([0.,1.]):
                lower,upper,feasible=interval(aa,rhs+margin,j); stored=p['zero_intervals' if mi==0 else 'margin_intervals'][row]
                scalar(lower,stored[0]); assert math.isinf(upper)==math.isinf(stored[1])
                if math.isfinite(upper):scalar(upper,stored[1])
                assert feasible==bool(stored[5]); counts[mi]+=int(feasible)
                # Stored active constraints independently witness endpoint values.
                lk,uk,zk=(int(v) for v in stored[2:5])
                if lk>=0:assert aa[lk]>0; scalar((rhs[lk]+margin)/aa[lk],stored[0])
                if uk>=0:assert aa[uk]<0; scalar((rhs[uk]+margin)/aa[uk],stored[1])
                if zk>=0:assert aa[zk]==0 and rhs[zk]+margin>0
                if mi==1 and feasible:
                    s=max(lower*1.01,lower+.01)
                    if s>upper:s=lower+(upper-lower)/2
                    if not math.isfinite(s) or s>b['scalar_cap']:continue
                    scalars[row]=s; reconstructed_rows[row]=len(states); states.append(s*directions[row])
        assert counts==[r['zero_interval_nonempty'],r['margin1_interval_nonempty']]
        assert np.array_equal(reconstructed_rows,p['code_rows']); compare(scalars,p['scalars'])
        t=np.array(states,'f8'); compare(p['t'],t); assert np.array_equal(p['t_f32'].view('u4'),p['t'].astype('f4').view('u4'))
        t32=p['t_f32']; t64=t32.astype('f8'); valid=np.flatnonzero(reconstructed_rows>=0); winners=targets[valid]; n=len(t32)
        radius=np.sqrt(np.sum(t64*t64,axis=1)/256+epsilon); v=np.column_stack([t64/radius[:,None],math.sqrt(256*epsilon)/radius]); compare(v,p['v'])
        assert np.all(v[:,-1]>0) and float(np.max(np.abs(np.sum(v*v,axis=1)-256)))<=1e-12
        dll=ctypes.CDLL(b['reference']['dll']['path']); fn=dll.reference_readout
        fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_float,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]; fn.restype=ctypes.c_int
        mapping=np.memmap(r['scores']['path'],mode='r',dtype='<f4',shape=(n,65537)); calls=0; words=0; flags={k:True for k in b['precision']}
        for lo in range(0,n,64):
            guard(); hi=min(lo+64,n); count=hi-lo; norm=np.empty((count,256),'<f4'); inv=np.empty(count,'<f4'); out=np.empty((count,65537),'<f4')
            assert fn(C32.ctypes.data,beta32.ctypes.data,t32[lo:hi].ctypes.data,count,epsilon,norm.ctypes.data,inv.ctypes.data,out.ctypes.data)==0
            assert np.array_equal(norm.view('u4'),p['normalized_f32'][lo:hi].view('u4')) and np.array_equal(inv.view('u4'),p['inverse_f32'][lo:hi].view('u4'))
            assert np.array_equal(out.view('u4'),mapping[lo:hi].view('u4'))
            words+=out.size; calls+=1
            raw=partial(t64[lo:hi],C)+beta; real=raw/radius[lo:hi,None]; delta=out.astype('f8')-real; jj=winners[lo:hi]; rr=np.arange(count)
            logz=np.logaddexp.reduce(real,axis=1); span=delta.max(axis=1)-delta.min(axis=1); absolute=np.max(np.abs(delta),axis=1)
            compare(logz,p['logz'][lo:hi]); compare(span,p['error_range'][lo:hi]); compare(absolute,p['absolute_error'][lo:hi])
            assert np.array_equal(np.argmax(out,axis=1),p['ids'][lo:hi]) and np.array_equal(np.argmax(real,axis=1),p['real_ids'][lo:hi])
            for values,field in [(raw,'raw_margin'),(real,'real_margin'),(out.astype('f8'),'native_margin')]:
                own=values[rr,jj].copy(); other=values.copy(); other[rr,jj]=-np.inf; compare(own-other.max(axis=1),p[field][lo:hi])
            # Only actual native IDs admit decisions;ray primal checks remain real.
            normerr=float(np.linalg.norm(norm.astype('f8')-v[lo:hi,:256])/np.linalg.norm(v[lo:hi,:256])); inverr=float(np.max(np.abs(inv.astype('f8')*radius[lo:hi]-1)))
            batch=next(x for x in r['batches'] if x['lo']==lo); scalar(normerr,batch['norm_relative']); scalar(inverr,batch['inverse_relative']); scalar(absolute.max(),batch['score_absolute'])
            for key,val in [('score_absolute',float(absolute.max())),('norm_relative',normerr),('inverse_relative',inverr)]:flags[key]&=val<=b['precision'][key]
        mapping._mmap.close(); mapping=None; assert calls==r['native_calls'] and words==n*65537 and flags==r['precision_flags']
        assert n==r['candidate_codes'] and np.count_nonzero(p['ids']==winners)==r['native_target_winners'] and np.count_nonzero(p['ids']!=p['real_ids'])==r['native_to_real_ID_changes']
        scalar(p['raw_margin'].min(),r['minimum_raw_margin']); scalar(p['native_margin'].min(),r['minimum_native_margin']); scalar(p['absolute_error'].max(),r['maximum_score_absolute'])
        checked_labels=0
        for s in r['splits']:
            original=load(next(x for x in b['preflight']['splits'] if x['split']==s['split'])['file']); packet=load(s['file']); jj=original['teacher_ids']
            ti=np.searchsorted(targets,jj); rows=reconstructed_rows[ti]; available=rows>=0; successful=np.zeros(len(rows),bool); successful[available]=p['ids'][rows[available]]==jj[available]
            assert np.array_equal(packet['teacher_ids'],jj) and np.array_equal(packet['target_indices'],ti) and np.array_equal(packet['code_rows'],rows) and np.array_equal(packet['available'],available)
            compare(packet['weights'],original['weights'],0.)
            pred=np.full(len(rows),-1,'i8'); pred[available]=p['ids'][rows[available]]; assert np.array_equal(pred,packet['native_ids'])
            loss=np.full(len(rows),np.nan,'f8'); upper=loss.copy(); lower=loss.copy(); used=rows[available]
            # Independent eight-coordinate block sum of sufficient moments.
            linear=np.zeros(len(used),'f8')
            for col in range(0,257,8):linear+=np.sum(original['moments'][available,col:col+8]*v[used,col:col+8],axis=1)
            loss[available]=p['logz'][used]-linear-original['entropy'][available]
            lower[available]=np.maximum(0.,loss[available]-p['error_range'][used]); upper[available]=loss[available]+p['error_range'][used]
            compare(loss,packet['real_KL']); compare(lower,packet['native_KL_lower']); compare(upper,packet['native_KL_upper'])
            w=original['weights']; scalar(w@available,s['case_candidate_fraction']); scalar(w@successful,s['case_native_winner_fraction'])
            scalar(w[available]@loss[available],s['available_weighted_real_KL']); scalar(w[available]@lower[available],s['available_weighted_native_KL_lower']); scalar(w[available]@upper[available],s['available_weighted_native_KL_upper'])
            assert available.sum()==s['candidate_labels'] and successful.sum()==s['native_winner_labels'] and len(rows)==s['rows']; checked_labels+=len(rows)
        assert checked_labels==8808 and not r['general_head_infeasibility_proved'] and not r['causal_encoder_implemented'] and not r['timing_admitted']
        check(); guard()
        result=dict(schema='NATIVE_HEAD_AFFINE_RAYS_AUDIT_V1',source=extent(__file__),freeze=a.freeze,report=extent(a.report),terminal=extent(a.terminal),binding=r['binding'],
                    all2361_full_constraints_zero_and_one_margin_intervals_scalars_codes_verified=True,all8808_associations_moment_losses_native_score_range_bounds_verified=True,
                    native_score_words_bit_exact=words,normalized_words_bit_exact=n*256,inverse_words_bit_exact=n,independent_reference_calls=calls,
                    max_relative=maxrel,max_scalar_delta=maxscalar,before_after_hashes_exact=True,decision='INDEPENDENT_STORED_PASS',seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
                    scope='Oracle positive rays only;complete scalar FMA native-bit and independent geometry/moment audit. No full head infeasibility/causal encoder/distribution preservation/whole-model speed admission.')
        write(a.out,result); receipt.update(exit_code=0,error=None,result=extent(a.out)); print(json.dumps(dict(decision=result['decision'],words=words,seconds=result['seconds'])),flush=True)
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        if mapping is not None:mapping._mmap.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset); write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['report','terminal','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['report-sha','terminal-sha','freeze']:p.add_argument('--'+name,required=True)
    main(p.parse_args())
