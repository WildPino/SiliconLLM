"""All24 source-only native exports and generic-row ABI qualification."""
import argparse,ctypes,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
from token_state import interface as token_interface,arrays,arguments,ptrs
START=time.monotonic()

def interface(b):
    dll,create,destroy,tokens,snapshot=token_interface(b);abi=dll.rs_abi;abi.argtypes=[ctypes.c_int];abi.restype=ctypes.c_int;assert [abi(i) for i in range(6)]==[6704,3072,96,48,2048,3];rows=dll.rs_block;rows.argtypes=[ctypes.c_void_p,ctypes.c_int]+[ctypes.c_void_p]*12;rows.restype=ctypes.c_int;return dll,create,destroy,tokens,rows,snapshot

def prepare(a):
    import numpy as np,psutil
    rp=DOC/'token_state_result_20261013.json';ap=DOC/'token_state_adjudication_20261013.json';r=json.loads(rp.read_bytes());ad=json.loads(ap.read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(rp) and r['numeric_all_pass'] and r['quality_all_pass'];pb=json.loads(Path(r['binding']['path']).read_bytes());gb=json.loads(Path(json.loads(Path(pb['generator_report']['path']).read_bytes())['binding']['path']).read_bytes());bp=DOC/'original_falcon_recurrent_projection_finish_result_20261009.json';ba=DOC/'original_falcon_recurrent_projection_audit_result_20261009.json';br=json.loads(bp.read_bytes());bad=json.loads(ba.read_bytes());assert bad['decision']=='STORED_IDENTITIES_AND_AGGREGATES_VERIFIED' and bad['adopted_bases']==24 and {p['site'] for p in br['bases']}==set(range(24));compiler=Path(pb['compile']['command'][0]);sp=DOC/'row_state_storage_20261013.json';st=json.loads(sp.read_text(encoding='utf-8-sig'));assert not a.out.exists() and not a.directory.exists() and a.directory.resolve()==Path(st['new_family_directory']).resolve();src=B/'row_state.c'
    old=(B/'token_state.c').read_text();common=old[old.index('    memcpy(bits[0],row,'):old.index('API int ts_block(')].split('\n}\n')[0];assert common in src.read_text()
    paths=[Path(__file__),src,B/'row_state_audit.py',B/'token_state.c',B/'token_state.py',B/'full_channel_native.c',B/'integer_readout.c',B/'falcon_auxiliary_s6_transport.py',DOC/'ROW_STATE_PROTOCOL_20261013.md',sp,rp,ap,Path(r['binding']['path']),bp,ba,Path(gb['source_weights']),Path(gb['source_config']),Path(pb['table']['path']),Path(pb['compile']['dll']['path']),compiler,Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),Path(np._core._multiarray_umath.__file__),Path(np.linalg._umath_linalg.__file__),Path(psutil.__file__)];nr=Path(np.__file__).resolve().parent;pr=Path(psutil.__file__).resolve().parent;paths+=list((nr.parent/'numpy.libs').glob('*.dll'))+list(pr.glob('*.pyd'))+list(compiler.parent.glob('*.dll'))+list(compiler.parent.glob('*lld*.exe'))+[Path(p['basis']['path']) for p in br['bases']]+[Path(p['file']['path']) for p in r['records']]+[ROOT/p for p in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']];inputs=[extent(p) for p in dict.fromkeys(paths)];a.directory.mkdir();dll=a.directory/'row_state.dll';log=a.directory/'compile.log';cmd=[str(compiler),'-O3','-mavx2','-mfma','-march=znver2','-ffp-contract=off','-shared',str(src.resolve()),'-o',str(dll.resolve()),'-lm'];co=subprocess.run(cmd,capture_output=True,timeout=45,creationflags=0x08000000);log.write_bytes(co.stdout+co.stderr);assert co.returncode==0;inputs+=[extent(dll),extent(log)];get,stream=tensor_reader(gb['source_weights'],np);exports=[]
    try:
        for p in sorted(br['bases'],key=lambda x:x['site']):
            site=p['site'];pre=f'model.layers.{site}.mamba.'
            with np.load(p['basis']['path'],allow_pickle=False) as z:W=z['W'].copy();V=z['V'].copy()
            assert W.shape==V.shape==(256,96) and np.isfinite(W).all() and np.isfinite(V).all();dual=float(np.max(np.abs(W.T@V-np.eye(96))));assert dual<=1e-8;O=get(pre+'out_proj.weight');assert O.shape==(2048,3072);scale=np.max(np.abs(O),axis=1)/364;ratio=np.divide(O,scale[:,None],out=np.zeros_like(O),where=scale[:,None]!=0);q=np.clip(np.rint(ratio),-364,364).astype('i2')
            data=dict(Ws=W.T.astype('f4').copy(),Vs=V.T.astype('f4').copy(),Alog=get(pre+'A_log').astype('f4'),D=get(pre+'D').astype('f4'),gamma=get(pre+'norm.weight').astype('f4'),cw=get(pre+'conv1d.weight')[:,0],cb=get(pre+'conv1d.bias'),db=get(pre+'dt_bias'),q=q,scale=scale);path=a.directory/f'site{site:02d}.parameters.npz';np.savez(path,**data);file=extent(path);inputs.append(file);exports.append(dict(site=site,basis=p['basis'],dual_error=dual,file=file));print(json.dumps(dict(exported_site=site,dual_error=dual,seconds=time.monotonic()-START)),flush=True)
    finally:stream.close()
    for e in inputs:assert extent(e['path'])==e
    proc=psutil.Process();assert time.monotonic()-START<900 and proc.memory_info().peak_wset<6<<30 and not proc.children(recursive=True);control=next(q for q in pb['records'] if q['split']=='FIT');b=dict(schema='ROW_STATE_BINDING_V1',freeze=a.freeze,python=pb['python'],versions=pb['versions'],parent_report=extent(rp),parent_adjudication=extent(ap),parent_binding=r['binding'],source_weights=gb['source_weights'],source_config=gb['source_config'],basis_report=extent(bp),basis_adjudication=extent(ba),table=pb['table'],reference_records=r['records'],records=pb['records'],exports=exports,operator_control=dict(id=control['id'],tags=control['input_ids'][:16],scope='First16 admitted site0 FIT rows as arbitrary operator inputs for ALL24 parameter sets;not real later-layer trajectories or source-local quality.'),inputs=inputs,compile=dict(command=cmd,source=extent(src),dll=extent(dll),log=extent(log),exit_code=0),output_directory=str(a.directory/'run'),numeric=pb['numeric'],limits=dict(seconds=900,audit_seconds=1200,OS_bytes=6<<30,output_bytes=3<<30),scope='Explicit contextual preconv-row ABI retaining exact token branch arithmetic;ALL24 canonical source/basis/p6 exports and arbitrary-row numeric controls,all48 real initial-layer histories exact to admitted archive. No all24 actual-input/model quality/GPU/speed/useful-n admission.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)

def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and dict(numpy=np.__version__,psutil=psutil.__version__)==b['versions'] and not a.out.exists() and not a.directory.exists() and a.directory.resolve()==Path(b['output_directory']).resolve();proc=psutil.Process();proc.cpu_affinity([0]);contexts=[];stage='startup';records=[];controls=[];receipt=dict(schema='ROW_STATE_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,model_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.parent.rglob('*') if p.is_file())<=b['limits']['output_bytes']
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    def params(e):
        with np.load(e['file']['path'],allow_pickle=False) as z:p={k:z[k].copy() for k in z.files}
        return p,ptrs([p[k] for k in ['Ws','Vs','Alog','D','gamma']]),ptrs([p[k] for k in ['cw','cb','db']])
    def execute(tags,row,p,w,c,incremental=False):
        ctx=create();assert ctx;contexts.append(ctx);out=arrays(len(tags),np)
        if incremental:
            for t in range(len(tags)):
                if t%64==0:guard()
                assert rows(ctx,1,tags[t:t+1].ctypes.data,row[t:t+1].ctypes.data,w,c,p['q'].ctypes.data,p['scale'].ctypes.data,*arguments(out,t,t+1,np))==0
        else:assert rows(ctx,len(tags),tags.ctypes.data,row.ctypes.data,w,c,p['q'].ctypes.data,p['scale'].ctypes.data,*arguments(out,0,len(tags),np))==0
        state=np.empty((3072,96),'f4');ring=np.empty((3,3584),'u2');seen=np.zeros(1,'u8');assert snapshot(ctx,state.ctypes.data,ring.ctypes.data,seen.ctypes.data)==0;destroy(ctx);contexts.remove(ctx);out.update(final_state=state,final_conv=ring,tags=tags.copy(),seen=seen);return out
    try:
        check();a.directory.mkdir();dll,create,destroy,tokens,rows,snapshot=interface(b);table=np.memmap(b['table']['path'],dtype='u2',mode='r',shape=(65537,6704));p,w,c=params(b['exports'][0]);tags=np.array([123],'u4');bad=table[0:1].copy();bad[0,0]=0x7fc0;out=arrays(1,np);ctx=create();assert ctx;contexts.append(ctx);rc=rows(ctx,1,tags.ctypes.data,bad.ctypes.data,w,c,p['q'].ctypes.data,p['scale'].ctypes.data,*arguments(out,0,1,np));st=np.empty((3072,96),'f4');rg=np.empty((3,3584),'u2');seen=np.zeros(1,'u8');assert snapshot(ctx,st.ctypes.data,rg.ctypes.data,seen.ctypes.data)==0;nonfinite=bool(rc==4 and seen[0]==0 and not st.any() and not rg.any());destroy(ctx);contexts.remove(ctx)
        for rec in b['records']:
            stage=rec['id'];guard();tags=np.array(rec['input_ids'],'u4');raw=np.ascontiguousarray(table[tags]);out=execute(tags,raw,p,w,c);inc=execute(tags,raw,p,w,c,True);incremental=all(np.array_equal(out[k].view('u1'),inc[k].view('u1')) for k in out);ref=next(e for e in b['reference_records'] if e['id']==rec['id']);matches={}
            with np.load(ref['file']['path'],allow_pickle=False) as z:
                for k in out:
                    if k=='tags':matches[k]=bool(np.array_equal(out[k],z['token_IDs']))
                    elif k=='seen':matches[k]=bool(out[k][0]==rec['history'])
                    else:matches[k]=bool(np.array_equal(out[k].view('u1'),z[k].view('u1')))
            path=a.directory/(rec['id']+'.row.npz');np.savez(path,**out);records.append(dict(id=rec['id'],history=rec['history'],split=rec['split'],domain=rec['domain'],file=extent(path),reference=ref['file'],all_archive_bits_exact=all(matches.values()),incremental_bits_exact=incremental));print(json.dumps(dict(complete=len(records),archive_bits=all(matches.values()),incremental=incremental,seconds=time.monotonic()-START)),flush=True)
        control_tags=np.array(b['operator_control']['tags'],'u4');row=np.ascontiguousarray(table[control_tags]);assert len(control_tags)==16
        for e in b['exports']:
            stage=f'arbitrary_operator/{e["site"]}';guard();p,w,c=params(e);out=execute(control_tags,row,p,w,c);path=a.directory/f'site{e["site"]:02d}.operator.npz';np.savez(path,**out);controls.append(dict(site=e['site'],file=extent(path),parameters=e['file'],input_scope=b['operator_control']['scope']));print(json.dumps(dict(operator_site=e['site'],seconds=time.monotonic()-START)),flush=True)
        check();guard();result=dict(schema='ROW_STATE_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,records=records,operator_controls=controls,nonfinite_input_rejected=nonfinite,all48_archive_bits_exact=all(e['all_archive_bits_exact'] for e in records),all48_incremental_bits_exact=all(e['incremental_bits_exact'] for e in records),before_after_hashes_exact=True,actual_initial_layer_positions=22547,arbitrary_operator_positions=24*16,source_model_forwards=0,GPU_calls=0,optimizer_updates=0,output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()),seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,decision='COMPLETE_PENDING_INDEPENDENT_AUDIT',scope=b['scope']);write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(result=extent(a.out),all48_archive_bits_exact=result['all48_archive_bits_exact'],all48_incremental_bits_exact=result['all48_incremental_bits_exact'])),flush=True)
    except BaseException as er:
        import traceback
        receipt.update(exit_code=1,error=repr(er),stage=stage,traceback=traceback.format_exc());raise
    finally:
        for ctx in contexts:destroy(ctx)
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    for k in ['binding','directory','out']:p.add_argument('--'+k,type=Path)
    for k in ['binding-sha','freeze']:p.add_argument('--'+k)
    a=p.parse_args()
    if a.run:run(a)
    else:
        import psutil
        proc=psutil.Process();receipt=dict(schema='ROW_STATE_PREPARATION_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,model_admitted=False)
        try:prepare(a);receipt.update(exit_code=0,error=None,result=extent(a.out))
        except BaseException as er:
            import traceback
            receipt.update(exit_code=1,error=repr(er),traceback=traceback.format_exc());raise
        finally:receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.preparation_terminal.json'),receipt)
