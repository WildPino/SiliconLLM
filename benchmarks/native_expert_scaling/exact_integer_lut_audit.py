"""Independent complete packed-code/control/I64/raw/BF16 audit; no producer import."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
START=time.monotonic()

def main(a):
    import numpy as np,psutil
    re,te=extent(a.report),extent(a.terminal);assert re['sha256']==a.report_sha and te['sha256']==a.terminal_sha;r=json.loads(a.report.read_bytes());t=json.loads(a.terminal.read_bytes());assert t['exit_code']==0 and t['result']==re and extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil'] and not a.out.exists()
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));stage='startup';rows=[];counts=dict(packed_bytes=0,q_words=0,input_words=0,integer_dot_words=0,integer_mismatches=0,raw_words=0,raw_bit_mismatches=0,BF16_words=0,BF16_mismatches=0,control_digit_words=0,control_table_words=0);receipt=dict(schema='EXACT_INTEGER_LUT_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():assert time.monotonic()-START<=b['limits']['audit_seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
    def check():
        files=[q['file'] for q in r['records']]+[r['controls'][0]['exhaustive_digits']]+r['controls'][0]['exhaustive_tables']
        for e in b['inputs']+[r['binding'],re,te]+files:guard();assert extent(e['path'])==e,e['path']
    def digits(x):
        u=x.astype('i4')%65536;return np.stack([u%16,(u//16)%16,(u//256)%16,(u//4096)-16*(u>=32768)]).astype('i1')
    def bf(x):
        u=x.astype('<f4').view('<u4');top=u>>16;low=u&65535;return (top+((low>32768)|((low==32768)&((top&1)!=0))).astype('u4')).astype('u2')
    try:
        check();parent=json.loads(Path(b['parent_report']['path']).read_bytes());pa=json.loads(Path(b['parent_adjudication']['path']).read_bytes());assert pa['decision']=='INDEPENDENT_STORED_PASS' and pa['report']==b['parent_report'] and parent['binding']==b['parent_binding'];assert r['inherited_additional_all_pass']==parent['additional_all_pass'] and r['inherited_donor_gates']==parent['donor_gates'] and all(r[k]==0 for k in ['source_forwards','recurrence_calls','optimizer_updates'])
        assert b['bounds']==dict(byte_low=30,byte_top=16,trit_digit=46080,weighted_low=16773120,weighted_top=8945664,full=36640321536) and b['bounds']['full']<2**53
        ctl=r['controls'][0];assert ctl['all65536_inputs'] and ctl['all16_entries']
        with np.load(ctl['exhaustive_digits']['path'],allow_pickle=False) as z:dd=z['digits'].copy();assert z.files==['digits']
        signed=np.arange(-32768,32768,dtype='i4');unsigned_order=np.concatenate([signed[32768:],signed[:32768]]);assert dd.dtype==np.dtype('i1') and np.array_equal(dd,digits(unsigned_order));assert np.array_equal(np.einsum('s,su->u',np.array([1,16,256,4096],dtype='i8'),dd.astype('i8')),unsigned_order);counts['control_digit_words']=dd.size
        for half,e in enumerate(ctl['exhaustive_tables']):
            with np.load(e['path'],allow_pickle=False) as z:pairs=z['pairs'].copy();tables=z['tables'].copy();assert set(z.files)=={'pairs','tables'}
            first=unsigned_order[half*32768:(half+1)*32768].astype('i2');second=(-first.astype('i4')-1).astype('i2');assert pairs.shape==(65536,) and np.array_equal(pairs[0::2],first) and np.array_equal(pairs[1::2],second);d=digits(pairs).reshape(4,32768,2);expected=np.zeros((4,32768,16),'i1')
            for code in range(9):expected[:,:,code]=(code//3-1)*d[:,:,0]+(code%3-1)*d[:,:,1]
            assert tables.dtype==np.dtype('i1') and np.array_equal(tables,expected);counts['control_table_words']+=tables.size
        assert len(r['controls'])==8
        for c,(sw,sx) in zip(r['controls'][1:5],[(1,1),(1,-1),(-1,1),(-1,-1)]):assert c==dict(weight_sign=sw,input_sign=sx,all2048_exact=True)
        assert r['controls'][5:]==[dict(single_coordinate_mutation_exact=True),dict(zero_input_exact=True),dict(logical_code_vector_load_bytes=37748736,expected=37748736,exact=True)]
        assert len(r['records'])==144 and len({(e['id'],e['site']) for e in r['records']})==144
        for packet in b['packets']:
            guard();site=packet['site']
            with np.load(packet['file']['path'],allow_pickle=False) as z:codes=z['codes'].copy();scale=z['scale'].copy();assert set(z.files)=={'codes','scale'}
            assert codes.dtype==np.dtype('u1') and codes.shape==(6,768,2048) and scale.dtype==np.dtype('<f8') and scale.shape==(2048,) and np.isfinite(scale).all();q=np.zeros((2048,3072),'i4')
            for p in range(6):
                low=codes[p]&15;high=codes[p]//16;assert np.max(low)<=8 and np.max(high)<=8
                for group,c in [(0,low),(2,high)]:
                    q[:,group::4]+=3**p*(c.T.astype('i4')//3-1);q[:,group+1::4]+=3**p*(c.T.astype('i4')%3-1)
            with np.load(packet['source']['path'],allow_pickle=False) as z:assert np.array_equal(q,z['q']) and np.array_equal(scale.view('u8'),z['scale'].view('u8'))
            assert np.max(np.abs(q))<=364;counts['packed_bytes']+=codes.size;counts['q_words']+=q.size;qd=q.astype('f8')
            for e in [e for e in r['records'] if e['site']==site]:
                stage=f'{site}/{e["id"]}';guard();p=next(q for q in parent['records'] if q['site']==site and q['id']==e['id']);assert e['reference']==p['file'] and all(e[k]==p[k] for k in ['site','id','history','split','domain'])
                with np.load(p['file']['path'],allow_pickle=False) as z:x=z['input_I16'].copy();aa=z['input_scale'].copy();old_d=z['integer_dots'].copy();old_y=z['scores'].copy();old_b=z['output_BF16'].copy()
                with np.load(e['file']['path'],allow_pickle=False) as z:dot=z['integer_dots'].copy();raw=z['scores'].copy();bits=z['output_BF16'].copy();assert set(z.files)=={'integer_dots','scores','output_BF16'}
                assert x.dtype==np.dtype('<i2') and x.shape==(e['history'],3072) and np.max(np.abs(x.astype('i4')))<=32767;assert dot.dtype==np.dtype('<i8') and raw.dtype==np.dtype('<f8') and bits.dtype==np.dtype('<u2') and dot.shape==raw.shape==bits.shape==(e['history'],2048);counts['input_words']+=x.size
                exact=np.empty_like(raw)
                for lo in range(0,len(x),32):exact[lo:lo+32]=np.einsum('td,od->to',x[lo:lo+32].astype('f8'),qd,optimize=True)
                assert np.array_equal(exact,np.rint(exact)) and np.max(np.abs(exact))<=b['bounds']['full'];gold=exact*aa[:,None]*scale;expected_bits=bf(gold);dm=int(np.count_nonzero(dot!=exact.astype('i8')));ym=int(np.count_nonzero(raw.view('u8')!=gold.view('u8')));bm=int(np.count_nonzero(bits!=expected_bits));counts['integer_dot_words']+=dot.size;counts['integer_mismatches']+=dm;counts['raw_words']+=raw.size;counts['raw_bit_mismatches']+=ym;counts['BF16_words']+=bits.size;counts['BF16_mismatches']+=bm
                gates=dict(integer=bool(np.array_equal(dot,old_d)),raw=bool(np.array_equal(raw.view('u8'),old_y.view('u8'))),BF16=bool(np.array_equal(bits,old_b)));assert gates==e['numeric_gates'];rows.append(dict(gates=gates,exact_reference=dm==ym==bm==0))
            print(json.dumps(dict(site=site,complete=len(rows),seconds=time.monotonic()-START)),flush=True)
        numeric=all(all(e['gates'].values()) and e['exact_reference'] for e in rows);assert r['numeric_all_pass']==all(all(e['gates'].values()) for e in rows);check();guard();ad=dict(schema='EXACT_INTEGER_LUT_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,report=re,terminal=te,binding=r['binding'],counts=counts,numeric_all_pass=numeric,inherited_additional_all_pass=r['inherited_additional_all_pass'],inherited_donor_gates=r['inherited_donor_gates'],before_after_hashes_exact=True,decision='INDEPENDENT_STORED_PASS' if numeric else 'INDEPENDENT_STORED_NUMERIC_FAILURE',seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,scope='Every packed coefficient/digit/exhaustive table/exact I64/scaled raw F64/BF16 word independently checked. Native extrema/mutation/count controls are checked producer receipts. No further quality approximation;full accepted archive identity. No cost/model/DRAM/useful-n admission.');write(a.out,ad);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(adjudication=extent(a.out),decision=ad['decision'],counts=counts)),flush=True)
    except BaseException as er:
        import traceback
        receipt.update(exit_code=1,error=repr(er),stage=stage,traceback=traceback.format_exc());raise
    finally:receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['report','terminal','out']:p.add_argument('--'+k,type=Path,required=True)
    for k in ['report-sha','terminal-sha','freeze']:p.add_argument('--'+k,required=True)
    main(p.parse_args())
