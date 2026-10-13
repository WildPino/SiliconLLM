"""Freeze source-only nibble codes and replay exact accepted integer histories."""
import argparse,ctypes,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
START=time.monotonic()

def interface(binding):
    dll=ctypes.CDLL(binding['compile']['dll']['path']);abi=dll.el_abi;abi.argtypes=[ctypes.c_int];abi.restype=ctypes.c_int
    assert [abi(i) for i in range(9)]==[3072,2048,6,4,1536,768,98304,32768,8192]
    fn=dll.el_history;fn.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*7;fn.restype=ctypes.c_int
    return dll,fn

def encode(q,np):
    work=q.astype('i4');out=np.empty((6,768,2048),'u1')
    for p in range(6):
        d=(work+1)%3-1;work=(work-d)//3
        c=(3*(d[:,0::2]+1)+(d[:,1::2]+1)).astype('u1')
        out[p]=(c[:,0::2]|(c[:,1::2]<<4)).T
    assert not work.any();return out

def prepare(a):
    import numpy as np,psutil
    rp=DOC/'integer_readout_result_20261013.json';ap=DOC/'integer_readout_adjudication_v3_20261013.json';cp=DOC/'integer_readout_cost_review_20261013.json'
    r=json.loads(rp.read_bytes());ad=json.loads(ap.read_bytes());cl=json.loads(cp.read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(rp) and ad['numeric_all_pass'] and ad['additional_all_pass'] and all(all(g.values()) for site in ad['donor_gates'].values() for g in site.values()) and cl['parent_adjudication']==extent(ap)
    pb=json.loads(Path(r['binding']['path']).read_bytes());assert extent(r['binding']['path'])==r['binding'];compiler=Path(pb['compile']['command'][0]);stp=DOC/'exact_integer_lut_storage_20261013.json';st=json.loads(stp.read_text(encoding='utf-8-sig'))
    assert not a.out.exists() and not a.directory.exists() and a.directory.resolve()==Path(st['new_family_directory']).resolve();src=B/'exact_integer_lut.c';header=ROOT/'benchmarks/donor_adaptation/nibble_pack.h';engine=ROOT/'benchmarks/phase60/engine.c'
    body=engine.read_text();literal=body[body.index('static inline void acc_add_i8x32('):body.index('// P1 (BRIEF_P1_NIBBLE_PACKING')].strip();assert literal in src.read_text()
    paths=[Path(__file__),src,B/'exact_integer_lut_audit.py',B/'integer_readout.py',B/'falcon_auxiliary_s6_transport.py',DOC/'EXACT_INTEGER_LUT_PROTOCOL_20261013.md',stp,rp,ap,cp,Path(r['binding']['path']),compiler,header,engine,Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),Path(np._core._multiarray_umath.__file__),Path(np.linalg._umath_linalg.__file__),Path(psutil.__file__)]
    nr=Path(np.__file__).resolve().parent;pr=Path(psutil.__file__).resolve().parent
    paths+=list((nr.parent/'numpy.libs').glob('*.dll'))+list(pr.glob('*.pyd'))+list(compiler.parent.glob('*.dll'))+list(compiler.parent.glob('*lld*.exe'))+[Path(e['file']['path']) for e in pb['packets']]+[Path(e['file']['path']) for e in r['records']]
    paths+=[ROOT/p for p in ['benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']];inputs=[extent(p) for p in dict.fromkeys(paths)];a.directory.mkdir();dll=a.directory/'exact_lut.dll';log=a.directory/'compile.log'
    cmd=[str(compiler),'-O3','-mavx2','-mfma','-march=znver2','-ffp-contract=off','-shared',str(src.resolve()),'-o',str(dll.resolve()),'-lm'];co=subprocess.run(cmd,capture_output=True,timeout=45,creationflags=0x08000000);log.write_bytes(co.stdout+co.stderr);assert co.returncode==0;inputs+=[extent(dll),extent(log)];packets=[]
    for packet in pb['packets']:
        with np.load(packet['file']['path'],allow_pickle=False) as z:q=z['q'].copy();scale=z['scale'].copy()
        assert q.shape==(2048,3072) and np.max(np.abs(q.astype('i4')))<=364;codes=encode(q,np);path=a.directory/f'site{packet["site"]:02d}.packed.npz';np.savez(path,codes=codes,scale=scale);packets.append(dict(site=packet['site'],source=packet['file'],file=extent(path)));inputs.append(extent(path))
    for e in inputs:assert extent(e['path'])==e
    proc=psutil.Process();assert time.monotonic()-START<900 and proc.memory_info().peak_wset<6<<30 and not proc.children(recursive=True)
    binding=dict(schema='EXACT_INTEGER_LUT_BINDING_V1',freeze=a.freeze,python=pb['python'],versions=pb['versions'],parent_report=extent(rp),parent_adjudication=extent(ap),parent_binding=r['binding'],parent_closure=extent(cp),records=r['records'],packets=packets,inputs=inputs,compile=dict(command=cmd,source=extent(src),header=extent(header),engine=extent(engine),dll=extent(dll),log=extent(log),exit_code=0),output_directory=str(a.directory/'run'),bounds=dict(byte_low=30,byte_top=16,trit_digit=46080,weighted_low=16773120,weighted_top=8945664,full=36640321536),scratch_bytes=139264,logical_code_vector_load_bytes_per_input=37748736,numeric=dict(all_dot_raw_and_BF16_words_bit_exact=True),limits=dict(seconds=900,audit_seconds=1200,OS_bytes=6<<30,output_bytes=3<<30),scope='Literal original packed two-trit pshufb kernel/six weight planes/four exact input digits. Same accepted integer inputs/scale order/output;no new activation approximation/fit/source/recurrence/GPU/T4/RESERVED/model rate/useful-n/DRAM admission.')
    write(a.out,binding);print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)

def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil'] and not a.out.exists() and not a.directory.exists() and a.directory.resolve()==Path(b['output_directory']).resolve()
    proc=psutil.Process();proc.cpu_affinity([0]);rows=[];controls=[];stage='startup';receipt=dict(schema='EXACT_INTEGER_LUT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir())<b['limits']['output_bytes']
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    def call(c,s,x,aa):
        dots=np.empty((len(x),2048),'i8');raw=np.empty_like(dots,dtype='f8');bits=np.empty_like(dots,dtype='u2');assert fn(len(x),*[v.ctypes.data for v in [c,s,x,aa,dots,raw,bits]])==0;return dots,raw,bits
    try:
        check();a.directory.mkdir();dll,fn=interface(b);stage='exhaustive_digits_tables';digits=np.empty((4,65536),'i1');dg=dll.el_digits;dg.argtypes=[ctypes.c_void_p];dg.restype=ctypes.c_int;assert dg(digits.ctypes.data)==0
        unsigned=np.arange(65536,dtype='u4');gold=np.stack([unsigned&15,(unsigned>>4)&15,(unsigned>>8)&15,(unsigned>>12).astype('i4')-16*(unsigned>=32768)]).astype('i1');assert np.array_equal(digits,gold) and np.array_equal((digits.astype('i8')*np.array([1,16,256,4096])[:,None]).sum(0),unsigned.astype('i8')-65536*(unsigned>=32768))
        pairs=np.stack([unsigned.astype('u2').view('i2'),(unsigned^65535).astype('u2').view('i2')],axis=1).reshape(-1);tables=np.empty((4,32768,16),'i1');tf=dll.el_tables;tf.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p];tf.restype=ctypes.c_int
        # Two halves cover all65536 unsigned words; table entry9..15 neutrality explicit.
        table_paths=[]
        for half in range(2):
            xx=np.ascontiguousarray(pairs[half*65536:(half+1)*65536]);assert tf(32768,xx.ctypes.data,tables.ctypes.data)==0;ind=xx.view('u2').astype('u4');dd=np.stack([ind&15,(ind>>4)&15,(ind>>8)&15,(ind>>12).astype('i4')-16*(ind>=32768)]).astype('i2').reshape(4,32768,2);expected=np.zeros_like(tables)
            for c in range(9):expected[:,:,c]=(c//3-1)*dd[:,:,0]+(c%3-1)*dd[:,:,1]
            assert np.array_equal(tables,expected);p=a.directory/f'control_tables{half}.npz';np.savez(p,pairs=xx,tables=tables);table_paths.append(extent(p))
        p=a.directory/'control_digits.npz';np.savez(p,digits=digits);controls.append(dict(exhaustive_digits=extent(p),exhaustive_tables=table_paths,all65536_inputs=True,all16_entries=True));del digits,gold,pairs,tables,expected,dd
        c=np.empty((6,768,2048),'u1');s=np.ones(2048,'f8');aa=np.ones(1,'f8');xx=np.empty((1,3072),'i2')
        for sw,sx in [(1,1),(1,-1),(-1,1),(-1,-1)]:
            c.fill(0x88 if sw==1 else 0);xx.fill(sx*32767);dots,_,_=call(c,s,xx,aa);ok=bool(np.all(dots==sw*sx*b['bounds']['full']));controls.append(dict(weight_sign=sw,input_sign=sx,all2048_exact=ok))
        c.fill(0x88);c[0,0,0]=0x85;xx.fill(32767);dots,_,_=call(c,s,xx,aa);controls.append(dict(single_coordinate_mutation_exact=bool(dots[0,0]==b['bounds']['full']-32767 and np.all(dots[0,1:]==b['bounds']['full']))));xx.fill(0);dots,raw,bits=call(c,s,xx,aa);controls.append(dict(zero_input_exact=bool(not dots.any() and not raw.any() and not bits.any())));del c,s,aa,xx
        counted=dll.el_counted_once;counted.argtypes=[ctypes.c_void_p]*7;counted.restype=ctypes.c_int64
        for packet in b['packets']:
            with np.load(packet['file']['path'],allow_pickle=False) as z:codes=z['codes'].copy();scale=z['scale'].copy()
            for e in [e for e in b['records'] if e['site']==packet['site']]:
                stage=f'{e["site"]}/{e["id"]}';guard()
                with np.load(e['file']['path'],allow_pickle=False) as z:x=z['input_I16'].copy();aa=z['input_scale'].copy();refdot=z['integer_dots'].copy();refraw=z['scores'].copy();refbits=z['output_BF16'].copy()
                dots,raw,bits=call(codes,scale,x,aa);gates=dict(integer=bool(np.array_equal(dots,refdot)),raw=bool(np.array_equal(raw.view('u8'),refraw.view('u8'))),BF16=bool(np.array_equal(bits,refbits)));path=a.directory/f'{e["id"]}.site{e["site"]:02d}.npz';np.savez(path,integer_dots=dots,scores=raw,output_BF16=bits);rows.append(dict(id=e['id'],site=e['site'],split=e['split'],domain=e['domain'],history=e['history'],file=extent(path),reference=e['file'],numeric_gates=gates))
                if len(rows)==1:
                    cd=np.empty((1,2048),'i8');cy=np.empty((1,2048),'f8');cb=np.empty((1,2048),'u2');loaded=counted(*[v.ctypes.data for v in [codes,scale,x,aa,cd,cy,cb]]);assert np.array_equal(cd[0],refdot[0]) and np.array_equal(cy[0].view('u8'),refraw[0].view('u8')) and np.array_equal(cb[0],refbits[0]);controls.append(dict(logical_code_vector_load_bytes=loaded,expected=b['logical_code_vector_load_bytes_per_input'],exact=loaded==b['logical_code_vector_load_bytes_per_input']))
                print(json.dumps(dict(complete=len(rows),numeric_gates=gates,seconds=time.monotonic()-START)),flush=True)
        check();guard();parent=json.loads(Path(b['parent_report']['path']).read_bytes());result=dict(schema='EXACT_INTEGER_LUT_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,records=rows,controls=controls,numeric_all_pass=all(all(r['numeric_gates'].values()) for r in rows),inherited_additional_all_pass=parent['additional_all_pass'],inherited_donor_gates=parent['donor_gates'],before_after_hashes_exact=True,native_history_calls=144,source_forwards=0,recurrence_calls=0,optimizer_updates=0,output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()),seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,decision='COMPLETE_PENDING_INDEPENDENT_AUDIT',scope=b['scope']);write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(result=extent(a.out),numeric_all_pass=result['numeric_all_pass'])),flush=True)
    except BaseException as er:
        import traceback
        receipt.update(exit_code=1,error=repr(er),stage=stage,traceback=traceback.format_exc());raise
    finally:receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    for k in ['binding','directory','out']:p.add_argument('--'+k,type=Path)
    for k in ['binding-sha','freeze']:p.add_argument('--'+k)
    a=p.parse_args()
    if a.run:run(a)
    else:
        import psutil
        proc=psutil.Process();receipt=dict(schema='EXACT_INTEGER_LUT_PREPARATION_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
        try:prepare(a);receipt.update(exit_code=0,error=None,result=extent(a.out))
        except BaseException as er:
            import traceback
            receipt.update(exit_code=1,error=repr(er),traceback=traceback.format_exc());raise
        finally:receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.preparation_terminal.json'),receipt)
