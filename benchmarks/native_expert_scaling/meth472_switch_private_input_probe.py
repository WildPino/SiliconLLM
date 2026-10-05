"""ONE frozen rank32 private INPUT probe; fixed107 factors/21 source fallbacks."""
import time
START=time.monotonic()
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',
                  HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import argparse
from datetime import datetime,timezone
import faulthandler
import hashlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
OUT=ROOT/'results/native_expert_scaling/meth472_switch_private_input_probe'
RAW=DOC/'meth472_switch_private_input_probe_result.json'
PROTO=DOC/'METH_472_SWITCH_PRIVATE_INPUT_PROTOCOL_20261005.md'
BIND=DOC/'meth472_prospective_bindings.json'
BIND_SHA='e4405d9ab38524da5cd1915eeca03f16fb00a26fa65a10506c1726f9de170cda'
WIRE=struct.Struct('<H6BHIff32s32s32s')
ENTRY=struct.Struct('<4I14Q')
FORBIDDEN={'torch','transformers','tensorflow','sklearn','pandas','pyarrow','tokenizers','scipy'}


def sha(data):return hashlib.sha256(data).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def committed(path):
    rel=Path(path).resolve().relative_to(ROOT).as_posix()
    assert Path(path).read_bytes()==subprocess.check_output(['git','-c','core.autocrlf=false','cat-file','--filters','--path='+rel,'HEAD:'+rel],cwd=ROOT),rel
def write_new(path,value):
    data=(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').replace('\n','\r\n').encode()
    assert len(data)<=8<<20,('raw8MiB',len(data))
    with Path(path).open('xb') as s:s.write(data)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    assert os.name=='nt' and sys.flags.optimize==0
    assert args.out.resolve()==RAW.resolve() and not RAW.exists() and not RAW.with_suffix('.failure.json').exists() and not OUT.exists()
    stage='standard_library_prospective_bindings';peak=hashed=0;numeric=False;last_scan=-10.;psutil=None
    result={'experiment':'METH472-original128-bank11-ONE-rank32-private-input-function-probe',
            'start_utc':utc(),'argv':sys.argv.copy(),'source_binding_sha256':BIND_SHA,
            'apparatus_gates':{},'experts':[],'native_or_model_commands':0,'updates':0}
    OUT.mkdir();progress=(OUT/'progress.jsonl').open('x',encoding='utf-8')
    fatal=(OUT/'fatal_native.log').open('xb');faulthandler.enable(file=fatal,all_threads=True)
    def checkpoint(**v):
        progress.write(json.dumps({'stage':stage,'utc':utc(),'pid':os.getpid(),'seconds':time.monotonic()-START,**v})+'\n');progress.flush()
    def jobs():
        own={os.getpid(),*(p.pid for p in psutil.Process().parents())};daemons=[]
        for p in psutil.process_iter(['name','cmdline']):
            if p.pid in own:continue
            try:
                name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
                if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():daemons.append(p.pid);continue
                assert not(name.startswith('python') or name=='clang.exe' or(name.startswith('meth') and name.endswith('.exe'))),(p.pid,name)
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        return daemons
    def guard():
        nonlocal peak,last_scan
        assert time.monotonic()-START<=900,'hard900s'
        assert numeric or time.monotonic()-START<=300,'admission300s'
        if psutil is not None:
            info=psutil.Process().memory_info();peak=max(peak,info.rss,getattr(info,'peak_wset',0));assert peak<=8<<30,'hard8GiB'
        if time.monotonic()-last_scan>=1:
            out=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
            out+=RAW.stat().st_size if RAW.exists() else 0
            out+=RAW.with_suffix('.failure.json').stat().st_size if RAW.with_suffix('.failure.json').exists() else 0
            assert out<=768<<20,('hard_new_output768MiB',out);last_scan=time.monotonic()
    def digest(path):
        nonlocal hashed
        h=hashlib.sha256()
        with Path(path).open('rb') as s:
            while b:=s.read(4<<20):h.update(b);hashed+=len(b);guard()
        return h.hexdigest()
    checkpoint()
    try:
        assert digest(BIND)==BIND_SHA;committed(BIND)
        binding=json.loads(BIND.read_text(encoding='utf-8'))
        assert sys.version==binding['runtime']['python'] and Path(sys.executable).resolve()==Path(binding['runtime']['executable']).resolve()
        for path,v in binding['runtime']['files'].items():assert Path(path).stat().st_size==v['bytes'] and digest(path)==v['sha256']
        for pkg,v in binding['runtime']['packages'].items():
            assert metadata.version(pkg)==v['version']
            for path,r in v['files'].items():assert Path(path).stat().st_size==r['bytes'] and digest(path)==r['sha256']
        import psutil as ps
        psutil=ps;parent=psutil.Process();guard()
        result['main_process_instance']={'pid':os.getpid(),'create_time_unix':parent.create_time(),'name':parent.name(),'executable':parent.exe()}
        result['preserved_daemons_before']=jobs()
        result['head_at_execution']=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip()
        assert (ROOT/'.gitattributes').read_bytes()==subprocess.check_output(['git','show','HEAD:.gitattributes'],cwd=ROOT)
        result['scientific_sources']={}
        for path in [Path(__file__),BASE/'meth472_private_input_math.py',PROTO]:
            committed(path);result['scientific_sources'][str(path)]=digest(path)
        for rel,v in binding['helpers'].items():committed(ROOT/rel);assert digest(ROOT/rel)==v['sha256']
        records={}
        for name,v in binding['records'].items():
            p=DOC/name;committed(p);assert p.stat().st_size==v['bytes'] and digest(p)==v['sha256']
            if p.suffix=='.json':records[name]=json.loads(p.read_text(encoding='utf-8'))
        assert digest(binding['preparation_helper']['path'])==binding['preparation_helper']['sha256']
        engine=ROOT/'benchmarks/phase60/engine.c';committed(engine);assert digest(engine)==binding['original_engine']['sha256']
        result['preserved_engine_sha256']=binding['original_engine']['sha256']
        for rel,v in binding['preserved_unrelated_files'].items():assert digest(ROOT/rel)==v['sha256']
        assert subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True,cwd=ROOT).splitlines()==binding['tracked_status']
        for path,v in binding['native_files'].items():assert Path(path).stat().st_size==v['bytes'] and digest(path)==v['sha256']
        capture=records['meth471_switch_development_capture_result.json'];previous=records['meth469_switch_native_domain_capture_result.json']
        assert len(capture['gates'])==13 and all(capture['gates'].values())
        assert all(records['RETENTION_471_20261005.json']['gates'].values()) and all(records['RETENTION_469_20261005.json']['gates'].values())
        assert capture['ledger']['records']==387036 and capture['fixed_future_probe']=={'source':128,'bank':11,'decoder_layer':11,'hypothetical_rank':32,'data_ready':True}
        assert capture['banks'][11]['data_ready_IDs']==107 and capture['banks'][11]['covered_val_natural_executed_queries']==2978
        artifact=binding['original_artifact'];payload=Path(artifact['payload']);initial=[payload.stat().st_size,payload.stat().st_mtime_ns]
        assert initial==[binding['payload_stat_before']['bytes'],binding['payload_stat_before']['mtime_ns']]
        assert digest(payload)==artifact['sha256'] and digest(artifact['manifest'])==artifact['manifest_sha256']
        result['artifact']=artifact;result['payload_stat_before']=initial
        inventory=binding['retained_inventory'];assert len(inventory)==6224 and sum(v['bytes'] for v in inventory)==6895456390
        for origin in ('469','471'):
            rows=[v for v in inventory if v['origin']==origin];folder=Path(rows[0]['path']).parent
            assert {str(p.resolve()) for p in folder.iterdir() if p.is_file()}=={str(Path(v['path']).resolve()) for v in rows}
        for r in inventory:
            p=Path(r['path']);assert p.stat().st_size==r['bytes'] and p.stat().st_mtime_ns==r['mtime_ns'] and digest(p)==r['sha256']
        for r in binding['native_FFN_controls']:assert Path(r['path']).stat().st_size==r['bytes'] and digest(r['path'])==r['sha256']
        for path,r in binding['integer_fixtures'].items():assert digest(path)==r['sha256']
        assert shutil.disk_usage(ROOT).free>=2<<30
        result['apparatus_gates']['strict_frozen_actual_runtime_sources_records_payload_ALL6224_retained_files_controls_SHA']=True
        checkpoint(file_bytes_hashed=hashed)

        stage='actual_runtime_and_original_manifest'
        import numpy as np
        import threadpoolctl as T
        import meth472_private_input_math as M
        for pkg in (np,psutil,T):
            assert str(Path(pkg.__file__).resolve()) in binding['runtime']['packages'][pkg.__name__]['files']
            assert pkg.__version__==binding['runtime']['packages'][pkg.__name__]['version']
        assert Path(M.__file__).resolve()==(BASE/'meth472_private_input_math.py').resolve()
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore')
        parent.cpu_affinity([0]);assert parent.cpu_affinity()==[0]
        limits=T.threadpool_limits(limits=1)
        pools=T.threadpool_info();assert pools and all(v['num_threads']==1 and v['user_api']=='blas' for v in pools)
        for v in pools:assert str(Path(v['filepath']).resolve()) in binding['runtime']['packages']['numpy']['files']
        result['runtime']={'packages':{p:metadata.version(p) for p in ('numpy','psutil','threadpoolctl')},
                           'actual_module_paths':{p.__name__:str(Path(p.__file__).resolve()) for p in (np,psutil,T)},
                           'BLAS':pools,'affinity':[0],'GPU':False}
        export=records['meth380_switch_base128_export_result.json'];entries=export['tensors']
        assert export['artifact']==artifact and all(export['gates'].values())
        manifest=Path(artifact['manifest']).read_bytes();assert manifest[:8]==b'SWI8A001'
        config=struct.unpack_from('<13IfII',manifest,8)
        expected=export['original_config'];fields=('d_model','d_ff','num_heads','d_kv','num_layers','num_decoder_layers','num_experts','expert_capacity','vocab_size','relative_attention_num_buckets','relative_attention_max_distance','encoder_sparse_step','decoder_sparse_step')
        assert config[:13]==tuple(expected[k] for k in fields) and config[13]==np.float32(expected['layer_norm_epsilon']) and config[14:]==(1,len(entries))
        off=72
        def text():
            nonlocal off
            n=struct.unpack_from('<I',manifest,off)[0];off+=4;v=manifest[off:off+n].decode('utf-8');off+=n;return v
        assert Path(text()).resolve()==payload.resolve()
        for name,e in sorted(entries.items()):
            assert text()==name
            values=struct.unpack_from('<5I3Q',manifest,off);off+=44
            assert values==(0,len(e['shape']),e['shape'][0],e['shape'][1] if len(e['shape'])==2 else 1,e['encoding'],e['offset'],e['scale_offset'],e['elements'])
        assert off==len(manifest)
        mapped=np.memmap(payload,mode='r',dtype='u1')
        def weights(expert):
            ans=[]
            for kind,shape in [('wi',(3072,768)),('wo',(768,3072))]:
                name=f'decoder.block.11.layer.2.mlp.experts.expert_{expert}.{kind}.weight';e=entries[name]
                assert tuple(e['shape'])==shape and e['encoding']==1 and e['bytes']==2359296 and e['scale_bytes']==4*shape[0]
                w=np.frombuffer(mapped,dtype='<i1',count=e['elements'],offset=e['offset']).reshape(shape)
                s=np.frombuffer(mapped,dtype='<f4',count=shape[0],offset=e['scale_offset'])
                assert sha(w.tobytes())==e['sha256'] and sha(s.tobytes())==e['scale_sha256'] and np.isfinite(s).all() and np.all(s>=0)
                ans.extend((w,s))
            return tuple(ans)
        result['apparatus_gates']['original_manifest_ALL3320_tensor_entries_payload_and_bank_matrix_parts_exact']=True

        stage='fresh_original_arithmetic_and_native_FFN_controls'
        result['tiny_controls']=M.identity_and_spectral_controls(guard)
        paths=list(binding['integer_fixtures']);fixture=next(p for p in paths if Path(p).name=='integer_cases.bin');answer=next(p for p in paths if Path(p).name=='head-int-dot.bin')
        data=Path(fixture).read_bytes();answers=Path(answer).read_bytes();a=b=12
        assert data[:12]==struct.pack('<8sI',b'SWI8D001',13) and answers[:12]==struct.pack('<8sI',b'SW16R001',13)
        for case in range(13):
            rows,cols=struct.unpack_from('<2I',data,a);a+=8
            w=np.frombuffer(data,'<i1',rows*cols,a).reshape(rows,cols);a+=rows*cols
            si=np.frombuffer(data,'<f4',rows,a);a+=rows*4
            x=np.frombuffer(data,'<f4',cols,a).reshape(1,cols);a+=cols*4
            assert struct.unpack_from('<2I',answers,b)==(rows,cols);b+=8
            y=np.frombuffer(answers,'<f4',rows,b).copy();b+=rows*4
            cq=np.frombuffer(answers,'<i2',cols,b).copy();b+=cols*2
            ca=np.frombuffer(answers,'<f4',1,b).copy();b+=4
            cd=np.frombuffer(answers,'<i8',rows,b).copy();b+=rows*8
            q,alpha=M.quant(x);M.exact(q[0],cq);M.exact(alpha,ca)
            actual=q.astype(np.float64) @ w.astype(np.float64).T
            assert np.array_equal(actual[0],cd.astype(np.float64))
            M.exact(M.projection(q,alpha,w.astype(np.float64),si)[0],y);guard()
        assert a==len(data) and b==len(answers)
        ctype=np.dtype([('position','<u4'),('id','<u4'),('source_tokens','<u4'),('route_index','<u4'),
            ('pre','<f4',(768,)),('input','<f4',(768,)),('wi_scale','<f4'),('wi_codes','<i2',(768,)),
            ('up_raw','<f4',(3072,)),('up','<f4',(3072,)),('wo_scale','<f4'),('wo_codes','<i2',(3072,)),
            ('down','<f4',(768,)),('expert','<i4'),('accepted','<i4'),('probability','<f4'),
            ('post','<f4',(768,)),('final','<f4',(768,)),('head_input','<f4',(768,)),('head_scale','<f4'),('head_codes','<i2',(768,))])
        controls=[]
        for r in binding['native_FFN_controls']:
            d=Path(r['path']).read_bytes();assert d[:32]==struct.pack('<8s6I',b'SWFUN001',768,3072,32128,128,11,1) and len(d)==32+14*ctype.itemsize
            v=np.frombuffer(d,ctype,offset=32).copy();assert np.array_equal(v['position'],np.arange(14)) and v['id'].tolist()==r['decoder_ids']
            assert np.all(v['source_tokens']==29) and np.array_equal(v['route_index'],174+5+6*np.arange(14)) and np.all(v['accepted']==1)
            controls.append(v)
        controls=np.concatenate(controls);assert len(controls)==1344
        for e in range(128):
            ids=np.flatnonzero(controls['expert']==e)
            if not len(ids):continue
            wi,si,wo,so=weights(e);wi64=wi.astype(np.float64);wo64=wo.astype(np.float64)
            for k in range(0,len(ids),64):
                v=controls[ids[k:k+64]];q,alpha=M.quant(v['input'])
                M.exact(q,v['wi_codes']);M.exact(alpha,v['wi_scale'])
                up,h,qh,ah,f=M.source_ffn(q,alpha,wi64,si,wo64,so)
                for actual,name in [(up,'up_raw'),(h,'up'),(qh,'wo_codes'),(ah,'wo_scale'),(f,'down')]:M.exact(actual,v[name])
                guard()
        del controls
        result['apparatus_gates']['13_cached_native_and6rational_identity_extreme_integer_and_ALL1344_cached_selected_FFNs_byte_exact']=True
        checkpoint(cached_native_FFN_positions=1344)

        stage='ALL_fixed_bank11_actual_trace_ledger_inputs_before_geometry'
        qtype=np.dtype([('ledger_record','<u8'),('book','<u2'),('case','u1'),('mode','u1'),('role','u1'),('accepted','u1'),
            ('expert','<u2'),('index','<u4'),('alpha','<f4'),('probability','<f4'),('input','<f4',(768,)),
            ('codes','<i2',(768,)),('input_sha','S32'),('code_sha','S32'),('pair_sha','S32')])
        assert qtype.itemsize==4732 and WIRE.size==118 and ENTRY.size==128
        queries=np.empty(binding['bank_query_count'],dtype=qtype);cursor=records_seen=0
        trace_dtype=np.dtype([('index','<u4'),('phase','<u4'),('input','<f4',(768,)),('scores','<f4',(128,))])
        roles={v['book']:v['split'] for v in capture['immutable_book_roles']}
        assert roles[63]==roles[128]==roles[191]==0 and roles[64]==roles[127]==1
        ledger=Path(capture['ledger']['path']).open('rb')
        assert ledger.read(20)==struct.pack('<8sIQ',b'MQ471L01',118,387036)
        for case_index in range(768):
            bi,ci=divmod(case_index,4);src=previous if bi<128 else capture;local=case_index if bi<128 else case_index-512
            entry=src['cases'][local];assert (entry['book'],entry['case'])==(bi,ci)
            for mi,mode in enumerate(('teacher','natural')):
                c=src['commands'][entry['modes'][mode]];assert c['label']==f'book{bi}.case{ci}.{mode}' and c['returncode']==0
                t=14 if mi==0 else c['native_row']['actual_generated_tokens'];nr=6*(29+t)
                d=Path(c['trace_path']).read_bytes();assert d[:16]==struct.pack('<8sII',b'SWRTA001',128,768) and len(d)==16+nr*trace_dtype.itemsize
                tr=np.frombuffer(d,trace_dtype,offset=16);assert np.array_equal(tr['index'],np.arange(nr))
                whole=Path(c['whole_output_path']).read_bytes();assert whole[:36]==struct.pack('<8s7I',b'SWR32O01',29,t,768,12,12,32128,nr)
                route_offset=36+4*(14*29*768+14*t*768+t*32128);assert len(whole)==route_offset+12*nr
                rr=np.frombuffer(whole,np.dtype([('expert','<i4'),('accepted','<i4'),('p','<f4')]),offset=route_offset)
                inds=174+5+6*np.arange(t);inputs=tr['input'][inds].copy();q,alpha=M.quant(inputs)
                chosen=np.argmax(tr['scores'][inds],axis=1);assert np.array_equal(chosen,rr['expert'][inds]) and np.all(rr['accepted'][inds]==1)
                diff=tr['scores'][inds]-tr['scores'][inds,chosen,None]
                prob=(1/np.exp(diff.astype(np.float64)).astype(np.float32).astype(np.float64).sum(axis=1)).astype('<f4')
                assert np.max(np.abs(prob.astype(np.float64)/rr['p'][inds]-1))<=1e-6
                ld=ledger.read(nr*118);assert len(ld)==nr*118
                for position,index in enumerate(inds):
                    ni,lm,lb,lc,bank,accepted,split,e,ix,la,lp,hi,hq,hp=WIRE.unpack_from(ld,int(index)*118)
                    assert (ni,lm,lb,lc,bank,accepted,split,e,ix)==(128,mi,bi,ci,11,1,roles[bi],int(chosen[position]),int(index))
                    sb=alpha[position].tobytes();qb=q[position].tobytes()
                    assert struct.pack('<f',la)==sb and struct.pack('<f',lp)==rr['p'][index].tobytes()
                    assert hi==hashlib.sha256(inputs[position].tobytes()).digest() and hq==hashlib.sha256(qb).digest() and hp==hashlib.sha256(qb+sb).digest()
                    queries[cursor]=(records_seen+int(index),bi,ci,mi,split,1,e,int(index),la,lp,inputs[position],q[position],hi,hq,hp);cursor+=1
                records_seen+=nr;guard()
            if ci==3 and bi%16==15:checkpoint(completed_book=bi,bank_queries=cursor)
        assert records_seen==387036 and cursor==19962 and not ledger.read(1);ledger.close()
        assert np.sum(queries['role']==0)==13313 and np.sum((queries['role']==1)&(queries['mode']==1))==3065
        ready=set(binding['fixed_ready_IDs']);assert ready=={v['expert'] for v in capture['banks'][11]['experts'] if v['data_ready']}
        dev_codes={};novel={};rederived=[]
        for e,old in enumerate(capture['banks'][11]['experts']):
            by={name:np.flatnonzero((queries['expert']==e)&(queries['role']==split)&(queries['mode']==mi)) for split,role in enumerate(('dev','val')) for mi,mode in enumerate(('teacher','natural')) for name in [role+'_'+mode]}
            for name,ids in by.items():
                v=queries[ids];st=old['by_split_mode'][name]
                assert len(ids)==st['executed']==st['selected'] and st['rejected']==0
                assert len(set(v['code_sha'].tobytes()[i:i+32] for i in range(0,len(v)*32,32)))==st['unique_code_SHA']
                assert len(set((int(a),int(b)) for a,b in zip(v['book'],v['case'])))==st['cases']
                assert sorted(set(map(int,v['book'])))==st['books']
            dev=np.flatnonzero((queries['expert']==e)&(queries['role']==0));val=np.flatnonzero((queries['expert']==e)&(queries['role']==1))
            # Preserve every fixed32byte digest: np.bytes_ may remove trailing zeros.
            dc={queries['code_sha'][i:i+1].tobytes() for i in dev};vc={queries['code_sha'][i:i+1].tobytes() for i in val};dev_codes[e]=dc;novel[e]=vc-dc
            nb=sorted({int(queries['book'][i]) for i in val if queries['code_sha'][i:i+1].tobytes() in novel[e]})
            gates=(len(dc)>=32,len(set(map(int,queries['book'][dev])))>=4,len(novel[e])>=16,len(nb)>=4)
            assert len(dc)==old['dev_union']['unique_code_SHA'] and len(vc)==old['val_union']['unique_code_SHA'] and len(novel[e])==old['validation_novel_codes'] and nb==old['validation_novel_code_books']
            assert all(gates)==old['data_ready']==(e in ready);rederived.append({'expert':e,'dev_codes':len(dc),'novel_val_codes':len(novel[e]),'ready':e in ready})
        np.save(OUT/'query_inputs.npy',queries,allow_pickle=False)
        result['domain']={'total_queries':cursor,'development_queries':13313,'natural_validation_queries':3065,'covered_natural_validation_queries':2978,
                          'ready_IDs':sorted(ready),'fallback_IDs':sorted(set(range(128))-ready),'data_tables':rederived,
                          'query_dtype_descr':queries.dtype.descr,'ledger_records':387036}
        result['apparatus_gates']['ALL19962_bank11_A16_inputs_full_provenance_role_nativewinner_mass_ledger_and128_readiness_tables_exact']=True
        def representatives(indices):
            first={}
            for i in indices:
                key=(int(queries['book'][i]),queries['codes'][i].tobytes(),queries['alpha'][i].tobytes())
                first.setdefault(key,int(i))
            ids=np.asarray(sorted(first.values()),dtype='<u4');books=queries['book'][ids]
            bs,counts=np.unique(books,return_counts=True);lookup=dict(zip(map(int,bs),map(int,counts)))
            weights=np.asarray([1/(len(bs)*lookup[int(book)]) for book in books],dtype='<f8')
            assert len(ids)>0 and abs(float(weights.sum())-1)<=1e-12
            for book in bs:assert abs(float(weights[books==book].sum())-1/len(bs))<=1e-12
            return ids,weights
        columns=['reference_FFN_energy','candidate_FFN_energy','FFN_error_energy','WI_preactivation_error_energy','ReLU_hidden_error_energy',
            'effective_input_energy','ideal_P64_input_error_energy','hidden_decoded_A16_delta_energy','ReLU_gate_changes','hidden_A16_code_changes',
            'reference_hidden_quantizer_error_energy','candidate_hidden_quantizer_error_energy','WO_real_hidden_direction_energy',
            'WO_quantizer_direction_energy','operational_rounding_remainder_energy','direction_quantizer_cross_twice',
            'direction_remainder_cross_twice','quantizer_remainder_cross_twice','routed_FFN_error_energy','routed_reference_energy']
        metrics=np.zeros((cursor,len(columns)),dtype='<f8');reference=np.empty((cursor,768),dtype='<f4');candidate=np.empty_like(reference)
        bound=(64+128*ENTRY.size)+(107*(15360*32+2374656)+21*4733952)+cursor*qtype.itemsize+cursor*768*8+cursor*len(columns)*8+13313*768*8+13313*8+(16<<20)
        assert bound<768<<20
        result['prospective_byte_bounds']={'exact_fixed_bank_bytes':406110272,'query_input_payload_bytes':cursor*qtype.itemsize,
            'two_F32_function_payload_bytes':cursor*768*8,'full_Vt_witness_upper_bytes':13313*768*8,
            'metrics_payload_bytes':cursor*len(columns)*8,'metadata_headers_raw_progress_reserve_bytes':16<<20,'new_output_upper_bytes':bound,
            'max_covariance_workspace_rows':13313,'F64_thin_X_U_each_upper_bytes':13313*768*8,'LAPACK_other_temporary_reserve_bytes':512<<20,
            'CPU_batch_queries':64,'hard_memory_bytes':8<<30}
        numeric=True;guard();stage='ONE_development_only_rank32_and_ALL128_complete_functions'
        result['admission']={'seconds':time.monotonic()-START,'file_bytes_hashed':hashed,'OS_peak_bytes':peak}
        bank_path=OUT/'private_input_bank.bin';bank=bank_path.open('xb+');bank.write(b'\0'*(64+128*ENTRY.size));descriptors=[]
        def bank_array(array):
            offset=bank.tell();data=array.tobytes(order='C');bank.write(data);return [offset,len(data)]
        def reload(offset,bytes_count,dtype,shape):
            bank.flush();here=bank.tell();bank.seek(offset);data=bank.read(bytes_count);bank.seek(here)
            assert len(data)==bytes_count;return np.frombuffer(data,dtype).reshape(shape).copy()
        for e in range(128):
            result['preserved_daemons_last_expert']=jobs()
            wi,si,wo,so=weights(e);wi64=wi.astype(np.float64);wo64=wo.astype(np.float64)
            ids=np.flatnonzero(queries['expert']==e);policy=e in ready;stats={'expert':e,'policy':'factor32' if policy else 'original_I8_fallback','all_query_count':len(ids)}
            sections={k:[0,0] for k in ('P','A','WI','WI_scale','WO','WO_scale')};p32=a32=p64=None
            if policy:
                dev=np.flatnonzero((queries['expert']==e)&(queries['role']==0));di,dw=representatives(dev)
                vi,vw=representatives([i for i in ids if queries['role'][i]==1 and queries['code_sha'][i:i+1].tobytes() in novel[e]])
                x=(queries['codes'][di].astype(np.float64)*queries['alpha'][di].astype(np.float64)[:,None])*np.sqrt(dw)[:,None]
                s,vt,spectral=M.spectrum(x,guard);assert len(s)>=32 and len(di)>=32
                p64=vt[:32].T.copy();p32=p64.astype('<f4');a32=(wi64 @ p32.astype(np.float64)).astype('<f4')
                assert np.isfinite(p32).all() and np.isfinite(a32).all()
                porth=float(np.linalg.norm(p32.astype(np.float64).T @ p32.astype(np.float64)-np.eye(32)))
                assert porth<=2e-6*math.sqrt(32)
                residual=float(np.sum((x-(x @ p64) @ p64.T)**2));tail=float(np.sum(s[32:]**2))
                assert abs(residual-tail)/spectral['energy']<=1e-10
                spectral.update(development_unique_effective_inputs=len(di),development_books=len(set(map(int,queries['book'][di]))),
                    ideal_rank32_development_energy_retained=1-residual/spectral['energy'],stored_P32_orthogonal_residual=porth,
                    singular_boundary_gap=float(s[31]-s[32]) if len(s)>32 else None,
                    boundary_gap_over_s0=float((s[31]-s[32])/s[0]) if len(s)>32 else None)
                np.savez(OUT/f'svd_witness_e{e:03d}.npz',singular_values=s,Vt=vt,development_representatives=di,development_weights=dw,
                         validation_novel_representatives=vi,validation_novel_weights=vw)
                stats['input_spectrum']=spectral
                sections['P']=bank_array(p32);sections['A']=bank_array(a32)
            else:sections['WI']=bank_array(wi)
            sections['WI_scale']=bank_array(si);sections['WO']=bank_array(wo);sections['WO_scale']=bank_array(so)
            M.exact(si,reload(*sections['WI_scale'],'<f4',(3072,)));M.exact(wo,reload(*sections['WO'],'<i1',(768,3072)));M.exact(so,reload(*sections['WO_scale'],'<f4',(768,)))
            if policy:
                M.exact(p32,reload(*sections['P'],'<f4',(768,32)));M.exact(a32,reload(*sections['A'],'<f4',(3072,32)))
            else:M.exact(wi,reload(*sections['WI'],'<i1',(3072,768)))
            desc=[e,int(policy),32 if policy else 0,0]+[v for name in ('P','A','WI','WI_scale','WO','WO_scale') for v in sections[name]]+[0,0]
            assert len(desc)==18;descriptors.append(desc)
            for k in range(0,len(ids),64):
                ii=ids[k:k+64];q=queries['codes'][ii];alpha=queries['alpha'][ii]
                up0,h0,q0,a0,f0=M.source_ffn(q,alpha,wi64,si,wo64,so)
                if policy:upc,hc,qc,ac,fc=M.factor_ffn(q,alpha,p32,a32,si,wo64,so)
                else:upc,hc,qc,ac,fc=up0,h0,q0,a0,f0
                reference[ii]=f0;candidate[ii]=fc
                delta=fc.astype(np.float64)-f0.astype(np.float64);dh=hc.astype(np.float64)-h0.astype(np.float64)
                effective0=q0.astype(np.float64)*a0.astype(np.float64)[:,None];effectivec=qc.astype(np.float64)*ac.astype(np.float64)[:,None]
                de=(effectivec-effective0)-dh
                directional=(dh @ wo64.T)*so.astype(np.float64)[None,:]
                quant_direction=(de @ wo64.T)*so.astype(np.float64)[None,:]
                rho=(delta-directional)-quant_direction
                energy=lambda v:np.sum(v.astype(np.float64)**2,axis=1)
                xin=q.astype(np.float64)*alpha.astype(np.float64)[:,None]
                input_residual=xin-(xin @ p64) @ p64.T if policy else np.zeros_like(xin)
                values=[energy(f0),energy(fc),energy(delta),energy(upc.astype(np.float64)-up0.astype(np.float64)),energy(dh),energy(xin),energy(input_residual),
                    energy(effectivec-effective0),np.sum((up0>0)!=(upc>0),axis=1),np.sum(q0!=qc,axis=1),
                    energy(effective0-h0.astype(np.float64)),energy(effectivec-hc.astype(np.float64)),energy(directional),energy(quant_direction),energy(rho),
                    2*np.sum(directional*quant_direction,axis=1),2*np.sum(directional*rho,axis=1),2*np.sum(quant_direction*rho,axis=1),
                    energy(delta)*queries['probability'][ii].astype(np.float64)**2,energy(f0)*queries['probability'][ii].astype(np.float64)**2]
                metrics[ii]=np.asarray(values,dtype='<f8').T
                reconstructed=metrics[ii,12:18].sum(axis=1)
                cancellation_scale=np.maximum(np.sum(np.abs(metrics[ii,12:18]),axis=1),np.maximum(metrics[ii,2],1e-30))
                assert np.all(np.abs(reconstructed-metrics[ii,2])<=2e-12*cancellation_scale)
                guard()
            if policy:
                stats['novel_validation_FFN']=M.weighted_rms(metrics[vi,2],metrics[vi,0],vw)
                stats['novel_validation_input']=M.weighted_rms(metrics[vi,6],metrics[vi,5],vw)
                stats['per_expert_gate_novel_FFN_RMS_le0p05']=M.within(stats['novel_validation_FFN'],.05)
                stats['all_validation_FFN']=M.weighted_rms(metrics[ids[queries['role'][ids]==1],2],metrics[ids[queries['role'][ids]==1],0])
            else:
                assert np.array_equal(candidate[ids],reference[ids]) and np.all(metrics[ids,2]==0)
            result['experts'].append(stats);checkpoint(completed_expert=e,policy=stats['policy'],all_query_count=len(ids),
                novel_validation_FFN_RMS=stats.get('novel_validation_FFN',{}).get('RMS_relative'))
            guard()
        assert bank.tell()==406110272
        bank.seek(0);bank.write(struct.pack('<8s14I',b'MQ472B01',1,768,3072,128,11,32,128,64,64+128*128,107,21,406093824,0,0))
        for desc in descriptors:bank.write(ENTRY.pack(*desc))
        bank.flush();bank.close()
        assert np.isfinite(metrics).all() and np.isfinite(reference).all() and np.isfinite(candidate).all()
        np.save(OUT/'reference_functions.npy',reference,allow_pickle=False);np.save(OUT/'candidate_functions.npy',candidate,allow_pickle=False);np.save(OUT/'per_query_metrics.npy',metrics,allow_pickle=False)
        result['metric_columns']=columns;result['bank_descriptors']=descriptors
        val=np.flatnonzero(queries['role']==1);vn=np.flatnonzero((queries['role']==1)&(queries['mode']==1));assert len(vn)==3065
        result['full_natural_validation_FFN']=M.weighted_rms(metrics[vn,2],metrics[vn,0])
        result['full_natural_validation_routed_FFN']=M.weighted_rms(metrics[vn,18],metrics[vn,19])
        result['full_teacher_validation_FFN']=M.weighted_rms(metrics[val[queries['mode'][val]==0],2],metrics[val[queries['mode'][val]==0],0])
        books=[]
        for bi in range(64,128):
            ii=vn[queries['book'][vn]==bi];assert len(ii)>0
            books.append({'book':bi,**M.weighted_rms(metrics[ii,2],metrics[ii,0])})
        result['natural_validation_books']=books
        result['storage']={'actual_bank_bytes':bank_path.stat().st_size,'original_bank_array_bytes':605945856,
                           'stored_ratio':bank_path.stat().st_size/605945856,'factor_IDs':107,'original_fallback_IDs':21,
                           'deployable_tensor_header_bytes':64+128*128,'no_original_WI_retained_for107_factor_IDs':True}
        result['recipe_gates']={'actual_complete_bank_bytes_le0p70_original':bank_path.stat().st_size<=.7*605945856,
            'ALL107_novel_validation_equal_book_FFN_RMS_le0p05':all(v.get('per_expert_gate_novel_FFN_RMS_le0p05',True) for v in result['experts']),
            'ALL3065_natural_validation_FFN_RMS_le0p05':M.within(result['full_natural_validation_FFN'],.05),
            'ALL64_natural_validation_book_FFN_RMS_le0p10':all(M.within(v,.10) for v in books)}
        result['apparatus_gates']['ALL107_development_only_uncentered_alpha_squared_equal_book_effective_dedup_SVD_energy_axes_qualified']=True
        result['apparatus_gates']['ALL128_saved_reloaded_actual_factor_or_original_fallback_parts_byte_exact']=True
        result['apparatus_gates']['ALL19962_complete_original_coordinate_source_candidate_functions_finite_and_exact_directional_accounting']=True
        for r in inventory:
            p=Path(r['path']);assert p.stat().st_size==r['bytes'] and p.stat().st_mtime_ns==r['mtime_ns']
        assert [payload.stat().st_size,payload.stat().st_mtime_ns]==initial
        for rel,v in binding['preserved_unrelated_files'].items():assert digest(ROOT/rel)==v['sha256']
        assert subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True,cwd=ROOT).splitlines()==binding['tracked_status']
        result['forbidden_modules_after']=[v for v in sys.modules if v.partition('.')[0] in FORBIDDEN];assert result['forbidden_modules_after']==[]
        result['preserved_daemons_after']=jobs()
        faulthandler.disable();fatal.flush();assert (OUT/'fatal_native.log').stat().st_size==0
        result['apparatus_gates']['original_payload_engine_unrelated3_inputs_immutable_empty_native_fault_no_concurrent_job']=True
        assert len(result['apparatus_gates'])==8 and all(result['apparatus_gates'].values())
        result['decision']='rank32_private_INPUT_recipe_admitted_for_separately_frozen_native_operator_cost_inquiry' if all(result['recipe_gates'].values()) else 'fixed_rank32_private_INPUT_PCA_function_recipe_FAIL_close_this_representation_no_rank_grid_or_validation_selected_fallback'
        result['scope']='Original source128 bank11 diagnostic geometry, fixed107/21 policy. No new complete model/native execution/fresh donor head/generation/task quality/SAMErate/causal useful-n/winner-and-mass/CPU-LUT/physicalDRAM/family/100B proof. Operational rho includes F64 summation/multiplication/reassociation and finalF32 remainder, not solely finalcast.'
        stage='terminal_rank32_complete_function_recipe';result['end_utc']=utc()
        result['output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='progress.jsonl']
        guard();result['resource']={'seconds_before_raw_write':time.monotonic()-START,'OS_peak_bytes_before_raw':peak,
            'file_bytes_hashed_before_raw':hashed,'OUT_bytes_before_terminal_progress_row':sum(p.stat().st_size for p in OUT.iterdir()),
            'hard_seconds':900,'hard_memory_bytes':8<<30,'hard_new_output_bytes':768<<20}
        write_new(RAW,result);rawsha=digest(RAW);guard()
        checkpoint(admitted=all(result['recipe_gates'].values()),raw_sha256=rawsha,raw_bytes=RAW.stat().st_size,
            OS_peak_bytes_after_raw=peak,final_file_bytes_hashed=hashed,OUT_bytes_before_terminal_progress_row=result['resource']['OUT_bytes_before_terminal_progress_row'])
        print(json.dumps({'raw_sha256':rawsha,'apparatus_gates':result['apparatus_gates'],'recipe_gates':result['recipe_gates'],
            'decision':result['decision'],'natural_validation_FFN':result['full_natural_validation_FFN'],'seconds':time.monotonic()-START}),flush=True)
        limits.restore_original_limits()
    except BaseException as e:
        result['failure']={'stage':stage,'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
        result['resource_at_failure']={'seconds':time.monotonic()-START,'OS_peak_bytes':peak,'file_bytes_hashed':hashed}
        result['partial_output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in sorted(OUT.iterdir()) if p.is_file()]
        if not RAW.with_suffix('.failure.json').exists():write_new(RAW.with_suffix('.failure.json'),result)
        checkpoint(failed=True,failure_type=type(e).__name__,message=str(e))
        raise
    finally:
        faulthandler.disable();fatal.close();progress.close()


if __name__=='__main__':main()
