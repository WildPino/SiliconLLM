"""Independent struct joins and exact F64 products; never imports the main or its math."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import struct
import traceback
from meth493_r1_operations import Context, ROOT, DOC

def exact_control_product(a,b,accepted):
    if not accepted: return 0
    sign=(a^b)&0x80000000
    def decode(v):
        e=(v>>23)&255; m=v&0x7fffff; assert e!=255
        return (m,-149) if e==0 else ((1<<23)|m,e-150)
    ma,ea=decode(a); mb,eb=decode(b); m=ma*mb; exponent=ea+eb
    if not m: return sign
    top=m.bit_length()-1+exponent; unit=max(top-23,-149); shift=exponent-unit
    if shift>=0: rounded=m<<shift
    else:
        integer,remainder=divmod(m,1<<-shift); half=1<<(-shift-1)
        rounded=integer+(remainder>half or (remainder==half and integer&1))
    if top < -126: return sign|rounded
    if rounded==1<<24: rounded>>=1; top+=1
    assert top<=127
    return sign|((top+127)<<23)|(rounded-(1<<23))

def descriptor(ctx,path): return {'path':str(path),'bytes':path.stat().st_size,'sha256':ctx.digest(path)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--binding-sha',required=True)
    args=ap.parse_args(); ctx=Context(ROOT/'results/native_expert_scaling/meth493_r1_retention',Path(args.out).resolve(),300,256<<20)
    try:
        binding=ctx.admit(args.binding_sha); ctx.binding=binding
        rawpath=DOC/'meth493_r1_weighted_targets_result.json'; completed=rawpath.exists()
        if not completed: rawpath=rawpath.with_suffix('.failure.json')
        rawdesc=descriptor(ctx,rawpath); raw=json.loads(rawpath.read_bytes())
        regpath=DOC/'meth493_r1_main_sole_first_registration.json'; reg=json.loads(regpath.read_bytes())
        assert reg['attempt']==1 and reg['record_sha256']==rawdesc['sha256'] and reg['process_instance']==raw['process_instance']
        assert reg['actual_exit_code']==0 if completed else reg['actual_exit_code']!=0
        eventpath=ROOT/'results/native_expert_scaling/meth493_r1_windows_terminal.json'; event=json.loads(eventpath.read_bytes())
        assert event['query_available'] and event['query_error'] is None and not event['matching_scientific_events']
        instance=event['instances'][0]
        assert instance['pid']==raw['process_instance']['pid'] and instance['create_time_unix']==raw['process_instance']['create_time_unix']
        for key in ('start_utc','end_utc'): assert datetime.datetime.fromisoformat(instance[key])==datetime.datetime.fromisoformat(raw[key])
        ctx.r['gates']['sole_actual_main_registration_and_available_fault_free_Windows_query']=True
        for v in raw.get('output_inventory',[]):
            assert Path(v['path']).parent==ROOT/'results/native_expert_scaling/meth493_r1_weighted_targets'; ctx.exact(v)
        if not completed:
            result=ctx.finish({'raw':rawdesc,'main_windows_sha256':ctx.digest(eventpath),'main_completed':False,
                'native_calls':0,'optimizer_updates':0,'decision':'FIRST_COMPILER_FAULT_AND_PARTIALS_ADMITTED_NO_TARGET_PROMOTION'})
            print(json.dumps({'gates':result['gates'],'decision':result['decision']})); return
        assert all(raw['gates'].values()) and (raw['unique_inputs'],raw['occurrences'])==(17540,19962)
        terminalpath=Path(raw['terminal_resource_path']); terminal=json.loads(terminalpath.read_bytes())
        assert terminal['raw_sha256']==rawdesc['sha256'] and terminal['raw_path']==str(rawpath)
        assert terminal['resource']['wall_seconds']<=180 and terminal['resource']['parent_peak_bytes']+terminal['resource']['native_peak_bytes']<=256<<20
        import numpy as np
        import threadpoolctl
        threadpoolctl.threadpool_limits(limits=1); ctx.r['numerical_imports']=True
        assert all(v['num_threads']==1 for v in threadpoolctl.threadpool_info())
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore')
        controls=json.loads((ROOT/'results/native_expert_scaling/meth493_r1_weighted_targets/rounding_controls.json').read_bytes())
        fixtures=[(0x3f000000,0x80000000,1,0x80000000),(0x3f000000,1,1,0),(0x3f000000,3,1,2),
            (0x3f000000,0x80000001,1,0x80000000),(0x3f000000,0x80000003,1,0x80000002),
            (0x3f000000,0x01000000,1,0x00800000),(0x3f000000,0x00ffffff,1,0x00800000),
            (0x3f800000,0x7f7fffff,1,0x7f7fffff),(0x3f000001,0x3fc00000,1,0x3f400002),
            (0x3f000003,0x3fc00000,1,0x3f400004),(0x3f000000,0x80000003,0,0)]
        assert len(controls)==len(fixtures)
        for control,(a,b,accepted,expected) in zip(controls,fixtures):
            assert control=={'p_bits':a,'F_bits':b,'accepted':accepted,'expected_bits':expected,'actual_bits':expected}
            assert exact_control_product(a,b,accepted)==expected
        ctx.r['gates']['independent_integer_significand_RNE_proof_of_all_eleven_controls']=True
        U,Q,N,T=238872,387036,17540,19962
        def openwire(name,magic,width,reserved,count):
            stream=Path(binding['data'][name]['path']).open('rb')
            assert stream.read(24)==struct.pack('<8sIIQ',magic,width,reserved,count); return stream
        ctx.phase='independent_original_bank11_canonical_ownership_occurrence_metadata'
        unique={}; owners={}; links={}; fields={}
        with openwire('unique_inputs.bin',b'M479UNI1',3720,768,U) as stream:
            for uid in range(U):
                row=stream.read(3720); assert len(row)==3720; meta=struct.unpack_from('<12I',row)
                assert meta[0]==uid
                if meta[2]==11:
                    p=struct.unpack_from('<d',row,120)[0]; pb=struct.pack('<f',p)
                    assert struct.pack('<d',struct.unpack('<f',pb)[0])==row[120:128]
                    unique[uid]=(meta,struct.unpack('<I',pb)[0],row[48:80])
                if uid%4096==0: ctx.guard()
            assert not stream.read(1)
        assert len(unique)==N
        with openwire('ownership.bin',b'M479OWN1',32,0,U) as stream:
            for uid in range(U):
                row=struct.unpack('<8I',stream.read(32)); assert row[0]==uid
                if uid in unique: owners[uid]=row; assert row[1]==11
            assert not stream.read(1)
        with openwire('query_links.bin',b'M479LNK1',52,0,Q) as stream:
            for li in range(Q):
                row=struct.unpack('<13I',stream.read(52)); assert row[0]==li
                if row[6]==11:
                    assert row[12] in unique and row[9]==unique[row[12]][0][7]; links[li]=row
            assert not stream.read(1)
        with openwire('source_fields.bin',b'M479SRC1',72,0,Q) as stream:
            for li in range(Q):
                row=stream.read(72); assert len(row)==72
                if li in links:
                    assert struct.unpack_from('<12I',row)==links[li][:12]; fields[li]=row[48:]
            assert not stream.read(1)
        assert len(owners)==N and len(links)==len(fields)==T
        out=ROOT/'results/native_expert_scaling/meth493_r1_weighted_targets'
        target=(out/'weighted_targets.bin').open('rb'); savedinfo=(out/'uid_info.bin').open('rb'); occurrence=(out/'occurrences.bin').open('rb')
        assert target.read(24)==struct.pack('<8sIIQ',b'M493Y001',3072,768,N)
        assert savedinfo.read(24)==struct.pack('<8sIIQ',b'M493U001',180,0,N)
        assert occurrence.read(24)==struct.pack('<8sIIQ',b'M493O001',68,0,T)
        local={uid:k for k,uid in enumerate(unique)}; first={}; exposure={uid:[0,0,0,0,set(),set()] for uid in unique}
        byview={}; counts=[0]*N
        query=Path(binding['weighted_source']['query_inputs.npy']['path']).open('rb'); ref=Path(binding['weighted_source']['reference_functions.npy']['path']).open('rb')
        # Header and complete dtype are independently checked against the already qualified metadata, not inferred from payload.
        import ast
        metadata=json.loads((DOC/'meth493_weighted_target_metadata.json').read_bytes())
        for stream,asset in zip((query,ref),metadata['assets']):
            assert stream.read(8)==b'\x93NUMPY\x01\x00'; h=ast.literal_eval(stream.read(struct.unpack('<H',stream.read(2))[0]).decode('ascii'))
            assert json.loads(json.dumps(h))==asset['NPY_header'] and stream.tell()==asset['data_offset']
        qw=struct.Struct('<QH4BHIff3072s1536s32s32s32s'); lw=struct.Struct('<H6BHIff32s32s32s')
        assert qw.size==4732 and lw.size==118
        ledger=Path(binding['weighted_source']['ledger']['path']).open('rb'); assert ledger.read(20)==struct.pack('<8sIQ',b'MQ471L01',118,Q)
        canonical=Path(binding['data']['unique_inputs.bin']['path']).open('rb')
        ctx.phase='ALL19962_independent_struct_source_joins_and_exact_F64_products'
        for qid,li_expected in enumerate(links):
            qb=query.read(qw.size); fb=ref.read(3072)
            li,book,case,mode,role,accepted,expert,index,alpha,p,ib,cb,hi,hc,hp=qw.unpack(qb)
            link=links[li_expected]; assert li==li_expected
            assert (link[1],link[2],link[3],link[4],link[5],link[6],link[8],link[9],link[10])==(128,book,case,mode,2 if book>=128 else role,11,index,expert,accepted)
            assert role==(1 if 64<=book<128 else 0) and accepted==1 and 0<p<=1 and alpha>0
            ledger.seek(20+118*li); lb=ledger.read(118)
            assert lw.unpack(lb)==(128,mode,book,case,11,accepted,role,expert,index,alpha,p,hi,hc,hp)
            assert lb[14:22]==qb[20:28]
            assert hi==hashlib.sha256(ib).digest() and hc==hashlib.sha256(cb).digest() and hp==hashlib.sha256(cb+qb[20:24]).digest()
            uid=link[12]; k=local[uid]; canonical.seek(24+3720*uid); ub=canonical.read(3720)
            assert ub[136:3208]==ib and ub[48:80]==hi==unique[uid][2]
            assert fields[li][8:16]==struct.pack('<d',p) and ub[120:128]==struct.pack('<d',p)
            pb,ab=struct.unpack('<2I',qb[20:28])[1],struct.unpack('<I',qb[20:24])[0]
            assert pb==unique[uid][1]
            f=np.frombuffer(fb,'<f4'); assert np.isfinite(f).all()
            # Both source factors are exactly representable in F64. Their product has <=48 significand bits;
            # even two smallest subnormals remain representable in F64. The ONLY rounding here is castF32.
            y=(f.astype('<f8')*p).astype('<f4'); assert np.isfinite(y).all()
            target.seek(24+3072*k); assert target.read(3072)==y.tobytes(), ('independent_F64_product_BYTE_mismatch',qid,uid)
            expected=struct.pack('<17I',qid,k,*link,ab,pb)
            assert occurrence.read(68)==expected
            hashbytes=hi+hc+hp+hashlib.sha256(fb).digest()
            if uid not in first: first[uid]=(qid,ab,hashbytes)
            else: assert first[uid][1:]==(ab,hashbytes), ('duplicate_source_reference_contradiction',qid,uid)
            counts[k]+=1; e=exposure[uid]; e[0]|=1<<link[5]; e[1]|=1<<mode; e[2]|=1<<accepted; e[3]+=1; e[5 if role==1 else 4].add(book)
            key=(book,mode,accepted); byview[key]=byview.get(key,0)+1
            if qid%256==0: ctx.guard()
        assert not query.read(1) and not ref.read(1) and not occurrence.read(1)
        ctx.r['gates']['ALL19962_ledger_input_codes_alpha_p_expert_accepted_occurrence_BYTE_joins']=True
        ctx.r['gates']['ALL15330816_occurrence_product_coordinates_independent_F64_to_F32_BYTE_exact']=True
        ctx.phase='ALL17540_UID_wire_duplicate_and_exposure_proof'
        dev=val=0
        for uid,k in local.items():
            own=owners[uid]; e=exposure[uid]; qid,ab,hb=first[uid]
            assert own[2:]==(e[0],e[1],e[2],e[3],len(e[4]),len(e[5]))
            assert bool(e[0]&5)!=bool(e[0]&2); dev+=bool(e[0]&5); val+=bool(e[0]&2)
            expected=struct.pack('<13I',k,uid,11,unique[uid][0][7],*own[2:],unique[uid][1],qid,ab)+hb
            assert len(expected)==180 and savedinfo.read(180)==expected
        assert not savedinfo.read(1) and (dev,val)==(11721,5819)
        for stream in (query,ref,ledger,canonical,target,savedinfo,occurrence): stream.close()
        ctx.r['gates']['ALL17540_UID_source_hashes_duplicate_reference_and_owner_roles_modes_controls_book_exposures']=True
        cells=[]
        for expert in range(128):
            selected=[u for u in unique if unique[u][0][7]==expert]
            cells.append({'expert':expert,'unique_count':len(selected),'development':sum(bool(owners[u][2]&5) for u in selected),
                'consumed_validation':sum(bool(owners[u][2]&2) for u in selected),'occurrences':sum(owners[u][5] for u in selected)})
        assert cells==raw['cells']
        views=[{'book':book,'role':0 if book<64 else 1 if book<128 else 2,'mode':mode,'accepted':accepted,
            'count':byview.get((book,mode,accepted),0)} for book in range(192) for mode in range(2) for accepted in range(2)]
        assert views==raw['views'] and len(views)==768 and sum(v['count'] for v in views)==T
        assert raw['target_coordinates']==N*768 and raw['occurrence_product_coordinates']==T*768
        assert raw['duplicate_occurrences']==T-N and raw['accepted_occurrences']==T and raw['rejected_occurrences']==0
        assert raw['native_calls']==raw['optimizer_updates']==raw['SVD_calls']==0
        assert raw['decision']=='FULL_BANK11_WEIGHTED_SOURCE_TARGETS_COMPILED_PENDING_INDEPENDENT_AUDIT'
        ctx.r['gates']['ALL128_cells_768_views_scope_counts_and_no_student_or_quality_promotion']=True
        result=ctx.finish({'raw':rawdesc,'main_windows_sha256':ctx.digest(eventpath),'main_completed':True,
            'main_terminal_resources':descriptor(ctx,terminalpath),'unique_inputs_audited':N,'occurrences_audited':T,
            'target_coordinates_audited':N*768,'occurrence_product_coordinates_audited':T*768,
            'cells_audited':128,'views_audited':768,'native_calls':0,'optimizer_updates':0,'SVD_calls':0,
            'decision':'FULL_BANK11_WEIGHTED_SOURCE_TARGETS_INDEPENDENTLY_ADMITTED',
            'scope':'Only source target compilation. Student/own-state quality/cost/useful n/full goal remain unproved.'})
        print(json.dumps({'gates':result['gates'],'decision':result['decision'],'resource':ctx.terminal_resources['resource']}))
    except BaseException as exc:
        ctx.fail(exc); print(json.dumps({'error':str(exc),'traceback':traceback.format_exc()}),flush=True); raise

if __name__=='__main__': main()
