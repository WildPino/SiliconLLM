"""Independent retained whole outputs, exact integer/AVX head, freshness and decision audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import psutil
import meth511_operations as O

def manifest(path):
    data=Path(path).read_bytes();assert data[:8] in (b'SWI8A001',b'SWI8C001')
    config=struct.unpack_from('<13If',data,8);nf,nt=struct.unpack_from('<2I',data,64);offset=72
    def string():
        nonlocal offset
        n=struct.unpack_from('<I',data,offset)[0];offset+=4;s=data[offset:offset+n].decode();offset+=n;return s
    files=[string() for _ in range(nf)];tensors={}
    for _ in range(nt):
        name=string();row=struct.unpack_from('<5I3Q',data,offset);offset+=44;assert name not in tensors;tensors[name]=row
    assert offset==len(data) and nf==1 and nt==3320;return config,files,tensors
def head(np,ctx,q64,scales,W,states,hybrid):
    # Every integer product and every partial sum is below2^53, so F64 GEMM
    # produces the EXACT integer totals independently of AVX I32/I64 lane order.
    x=(states*np.float32(1/math.sqrt(768))).astype('f4');maximum=np.max(np.abs(x),axis=1)
    a=np.where(maximum==0,np.float32(1),maximum/np.float32(32767)).astype('f4')
    codes=np.clip(np.rint((x/a[:,None]).astype('f4')),-32767,32767).astype('i2')
    totals=codes.astype('f8')@q64.T;assert np.all(totals==np.rint(totals)) and np.max(np.abs(totals))<=768*128*32767
    old=((totals*scales.astype('f8')[None,:])*a.astype('f8')[:,None]).astype('f4');result=old.copy();rows=[]
    ids=np.arange(32128)
    for i in range(len(states)):
        chosen=np.lexsort((ids,-old[i].astype('f8')))[:8]
        if hybrid:
            products=(W[chosen].astype('f8')*x[i].astype('f8')).reshape(8,96,8)
            left=np.zeros((8,4),dtype='f8');right=np.zeros((8,4),dtype='f8')
            for group in range(96):left+=products[:,group,:4];right+=products[:,group,4:]
            joined=left+right;dots=np.zeros(8,dtype='f8')
            for lane in range(4):dots+=joined[:,lane]
            result[i,chosen]=dots.astype('f4')
        winner=int(result[i].argmax());rows.append({'selected_IDs':chosen.tolist(),'old_winner':int(old[i].argmax()),'winner':winner,'tail_overtake':bool(winner not in chosen),
            'insertion_comparisons':insertion_comparisons(np,old[i]) if hybrid else 0,
            'refined_F32_logits':result[i,chosen].tolist() if hybrid else [],'I8_logits_sha256':hashlib.sha256(old[i].tobytes()).hexdigest()})
        ctx.guard()
    return result,rows
def insertion_comparisons(np,values):
    ids=np.arange(32128);rank=np.maximum.accumulate(values.astype('f8'));ge=np.zeros(32128,dtype='i4')
    for _ in range(8):
        previous=np.empty(32128);previous[0]=-np.inf;previous[1:]=rank[:-1]
        ge+=previous>=values;rank=np.maximum.accumulate(np.minimum(values,previous))
    return int(np.minimum(np.minimum(ids,8),ge+1).sum())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding-sha',required=True);ap.add_argument('--cohort-sha',required=True);ap.add_argument('--native-sha',required=True);ap.add_argument('--donor-sha',required=True);args=ap.parse_args();ctx=O.Context('audit')
    try:
        b=ctx.admit(args.binding_sha);prep=ctx.import_result('cohort',args.cohort_sha);native=ctx.import_result('native',args.native_sha);donor=ctx.import_result('donor',args.donor_sha)
        assert native['native_calls']==576 and donor['model_calls']==384
        import numpy as np
        import duckdb
        from tokenizers import Tokenizer
        from threadpoolctl import threadpool_limits
        import meth511_quality as Q
        import meth511_results as R
        pool=threadpool_limits(limits=1);np.seterr(over='raise',invalid='raise',divide='raise',under='ignore')
        cohort=json.loads((O.ROOT/'results/native_expert_scaling/meth511_cohort/cohort.json').read_bytes());assert prep['cohort_sha256']==native['cohort_sha256']==donor['cohort_sha256']==ctx.digest(O.ROOT/'results/native_expert_scaling/meth511_cohort/cohort.json')
        exposed=set()
        for ledger in b['exclusions']['ledgers']:exposed.update(ledger['exposed_rows'])
        assert sorted(exposed)==b['exclusions']['rows']==cohort['excluded_rows']
        seed='meth511-whole-hybrid-head-fresh-511511';assert cohort['seed']==seed
        h=lambda s:hashlib.sha256(s.encode()).hexdigest()
        ranks=sorted(set(range(1243))-exposed,key=lambda row:h(f'{seed}|row={row}'));wanted=set(ranks[:96]);texts={}
        conn=duckdb.connect(config={'threads':'1','memory_limit':'512MB','autoinstall_known_extensions':'false','autoload_known_extensions':'false'});conn.execute('LOAD parquet')
        cursor=conn.execute('SELECT file_row_number,text FROM read_parquet(?,file_row_number=true) ORDER BY file_row_number',[b['corpus']['path']]);count=0
        while batch:=cursor.fetchmany(4):
            for row,text in batch:
                assert row==count;count+=1
                if row in wanted:texts[row]=text
            ctx.guard()
        conn.close();assert count==1243;tok=Tokenizer.from_file(str(Path(b['donor_directory'])/'tokenizer.json'));selected=[];rejects=[]
        for row in ranks[:96]:
            text=texts[row];sid='pg19:data/external/pg19/data/train-00014-of-00023-54b567998cd5eb4b.parquet:row='+str(row)
            if not isinstance(text,str) or len(text)<8192:rejects.append({'row':row,'reason':'less_than8192_characters'});continue
            begin=int(h(seed+'|'+sid),16)%(len(text)-4096);excerpt=text[begin:begin+4096];tokens=tok.encode(excerpt,add_special_tokens=False).ids
            if len(tokens)<512 or len(set(tokens))<128 or not all(1<v<32000 for v in tokens):rejects.append({'row':row,'reason':'token_length_diversity_or_special_id'});continue
            item=cohort['items'][len(selected)];assert item['corpus_row']==row and item['whole_source_utf8_sha256']==h(text)
            assert item['source_id']==sid and item['excerpt_start_character']==begin and item['excerpt']==excerpt and item['excerpt_utf8_sha256']==h(excerpt)
            for i,case in enumerate(item['cases']):
                quarter=len(tokens)//4;offset=i*quarter+int(h(f'{seed}|{sid}|window{i}'),16)%(quarter-31)
                window=tokens[offset:offset+32];source=[];target=[];spans=[]
                for j,value in enumerate(window):
                    if j in (3,10,17,24):
                        which=(3,10,17,24).index(j);source.append(32099-which);spans.append(window[j:j+2]);target.extend([32099-which,*window[j:j+2]])
                    elif j not in (4,11,18,25):source.append(value)
                source.append(1);target.extend([32095,1])
                assert case['index']==i and case['excerpt_token_start']==offset and case['original_window_ids']==window and case['masked_spans_ids']==spans
                assert case['source_ids']==source and case['target_ids']==target and case['decoder_ids']==[0]+target[:-1]
                assert case['source_ids_sha256']==hashlib.sha256(struct.pack('<29i',*source)).hexdigest() and case['target_ids_sha256']==hashlib.sha256(struct.pack('<14i',*target)).hexdigest()
            selected.append(row)
            if len(selected)==24:break
        assert rejects==cohort['source_only_rejections'] and len(selected)==24
        cs,fs,ts=manifest(b['source_manifest']['path']);cc,fc,tc=manifest(b['candidate_manifest']['path']);assert cs==cc and set(ts)==set(tc)
        assert Path(fs[0]).resolve()==Path(b['source_payload']['path']) and Path(fc[0]).resolve()==Path(b['candidate_payload']['path'])
        distinct=0;wo=0
        for name,t in ts.items():
            c=tc[name];assert t[:4]==c[:4] and t[5:]==c[5:]
            column='.mlp.experts.expert_' in name and name.endswith('.wo.weight')
            assert c[4]==(2 if column else t[4]);wo+=column
            if name not in ('encoder.embed_tokens.weight','decoder.embed_tokens.weight','lm_head.weight'):distinct+=t[7]
        assert wo==1536 and distinct==7415217408
        # Three tied source aliases occupy logical records; unique canonical count is unchanged.
        extent=b['head_extents'];W=np.memmap(extent[0]['path'],dtype='<f4',mode='r',offset=extent[0]['offset'],shape=(32128,768))
        q=np.memmap(extent[1]['path'],dtype='i1',mode='r',offset=extent[1]['offset'],shape=(32128,768));q64=q.astype('f8')
        scales=np.memmap(extent[2]['path'],dtype='<f4',mode='r',offset=extent[2]['offset'],shape=(32128,))
        rebuilt=[];head_positions=tail=changed=0;positions=[]
        for ordinal,(record,nrecord) in enumerate(zip(donor['cases'],native['cases'])):
            bi,ci=divmod(ordinal,4);case=cohort['items'][bi]['cases'][ci];assert (record['book'],record['index'])==(bi,ci) and record['case']==case==nrecord['case']
            recompute={'book':bi,'index':ci,'native':record['native']};arrays={}
            for arm in ['source','candidate']:
                r=record['native'][arm];assert r==nrecord['native'][arm]
                for kind in ['teacher','generation','profile']:
                    for row in r[kind]:
                        O.worker(row);assert row['threads']==3 and row['profile']==int(kind=='profile')
                        a=Q.read_output(Path(row['wire']['path']));assert all(np.isfinite(v).all() for v in a)
                        assert a[0].shape==(14,29,768) and a[1].shape==(len(a[2]),14,768) and a[2].shape[1]==32128 and a[3].shape==(174+6*len(a[2]),3)
                        assert np.all((a[3][:,0]>=0)&(a[3][:,0]<128)) and np.isin(a[3][:,1],[0,1]).all() and np.all((a[3][:,2]>0)&(a[3][:,2]<=1))
                        if kind!='teacher':
                            assert row['generated_ids']==a[2].argmax(-1).tolist() and row['actual_generated_tokens']==len(a[2])
                            assert len(a[2])<=64 and (len(a[2])==64 or row['generated_ids'][-1] in (1,32095))
                            assert abs(row['full_generation_seconds']-sum(row[k] for k in ['encoder_seconds','cross_kv_seconds','decode_greedy_seconds']))<=4e-7
                            assert abs(sum(row['step_seconds'])-row['decode_greedy_seconds'])<=1e-5 and row['full_generation_seconds']>0 and row['load_seconds']>0
                        t=len(a[2]);headcost=row['counters'][1][3]
                        assert headcost['calls']==t and headcost['code_bytes']==t*32128*768 and headcost['scale_bytes']==t*32128*4
                        assert headcost['f32_bytes']==(t*8*768*4 if arm=='candidate' else 0)
                        if arm=='candidate':
                            hy=row['hybrid_head'];assert hy['calls']==t and hy['visited']==t*32128 and hy['source_rows']==8*t and hy['source_MACs']==8*t*768 and hy['f32_bytes']==8*t*768*4
                            columns=row['conditional_columns'];assert [c['bank'] for c in columns]==list(range(12))
                            assert all(c['calls']==(29 if c['bank']<6 else t) and 0<=c['maximum']<=3072 and 0<=c['nonzero_sum']<=3072*c['calls'] for c in columns)
                        if kind=='teacher' or (kind=='generation' and row['repetition']==0):arrays[(arm,kind)]=a
                        del a
                assert len({v['wire']['sha256'] for v in r['generation']+r['profile']})==1
                recompute[arm+'_task']=Q.task(r['generation'][0]['generated_ids'],case['masked_spans_ids']);assert recompute[arm+'_task']==record[arm+'_task']
                for kind in ['teacher','generation']:
                    a=arrays[(arm,kind)];expected,rows=head(np,ctx,q64,scales,W,a[1][:,13],arm=='candidate')
                    assert expected.tobytes()==a[2].tobytes(),('independent_full_head_BYTE',bi,ci,arm,kind)
                    if arm=='candidate':
                        total_comparisons=sum(p['insertion_comparisons'] for p in rows)
                        observed=r['teacher'] if kind=='teacher' else r['generation']+r['profile']
                        assert all(z['hybrid_head']['comparisons']==total_comparisons for z in observed)
                        tail+=sum(p['tail_overtake'] for p in rows)
                        for step,p in enumerate(rows):positions.append({'book':bi,'case':ci,'scope':kind,'step':step,**p})
                    head_positions+=len(rows);del expected
            s=arrays[('source','teacher')];c=arrays[('candidate','teacher')];assert all(s[i].tobytes()==c[i].tobytes() for i in [0,1,3])
            sg=arrays[('source','generation')];cg=arrays[('candidate','generation')];assert sg[0].tobytes()==cg[0].tobytes()
            common=next((i for i,(x,y) in enumerate(zip(record['source_task']['generated_ids'],record['candidate_task']['generated_ids'])) if x!=y),min(len(sg[2]),len(cg[2]))-1)+1
            assert sg[1][:common].tobytes()==cg[1][:common].tobytes() and sg[3][:174+6*common].tobytes()==cg[3][:174+6*common].tobytes()
            changed+=int(np.count_nonzero(s[2].view('u4')!=c[2].view('u4')))
            with np.load(record['donor_reference']['path']) as ref:
                assert all(np.isfinite(ref[k]).all() for k in ref.files)
                te=ref['teacher_logits'];ge=ref['generation_logits'];ids=ge.argmax(-1).tolist()
                assert te.shape==(14,32128) and len(ids)<=64 and ref['teacher_encoder'].tobytes()==ref['generation_encoder'].tobytes()
                err=Q.relative(ref['official_teacher_logits'],te);gerr=Q.relative(ref['official_generation_logits'],ge)
                assert err<=1e-6 and gerr<=1e-6 and np.array_equal(ref['official_teacher_logits'].argmax(-1),te.argmax(-1)) and np.array_equal(ref['official_generation_logits'].argmax(-1),ge.argmax(-1))
                recompute.update(donor_prediction=Q.metrics(te,case['target_ids'],case['masked_spans_ids']),candidate_prediction=Q.metrics(c[2],case['target_ids'],case['masked_spans_ids']),
                    donor_task=Q.task(ids,case['masked_spans_ids']),top1_agreement=float(np.mean(c[2].argmax(-1)==te.argmax(-1))),masked_top1_agreement=float(np.mean(c[2].argmax(-1)[Q.MASK]==te.argmax(-1)[Q.MASK])))
                assert err==record['official_teacher_relative_l2'] and gerr==record['official_generation_relative_l2']
            recompute['normalized_prose_edit']=Q.edit_distance(recompute['candidate_task']['prose_ids'],recompute['donor_task']['prose_ids'])/max(len(recompute['candidate_task']['prose_ids']),len(recompute['donor_task']['prose_ids']),1)
            for k,v in recompute.items():assert v==record[k],('independent_rubric',bi,ci,k)
            rebuilt.append(recompute);del arrays,s,c,sg,cg
            if ci==3:ctx.log(audited_book=bi,head_positions=head_positions);print(json.dumps({'phase':'audit','book':bi,'positions':head_positions}),flush=True)
        summary=R.summarize(rebuilt);assert summary==donor['summary']
        # Independent ratio-of-sums and accepted-ID check, without main aggregators.
        for arm in ['source','candidate']:
            times=math.fsum(math.fsum(row['full_generation_seconds'] for row in r['native'][arm]['generation'] if row['repetition']>=0)/3 for r in rebuilt)
            for name,key in [('ordinary','generated_tokens'),('prose','prose_tokens')]:
                numerator=sum(r[arm+'_task'][key] for r in rebuilt if r[arm+'_task']['healthy_complete_nonempty_fields'])
                rate=summary['rates'][arm][name];assert rate['accepted_ids']==numerator and abs(rate['warm_rate']-numerator/times)<=1e-10
        commands=native['commands'];expected=[]
        for bi in range(24):
            for ci in range(4):
                for arm in (['source','candidate'] if (4*bi+ci)%2==0 else ['candidate','source']):
                    expected.extend(f'b{bi:02d}c{ci}.{arm}.{k}' for k in ['teacher','gen','profile'])
        assert len(commands)==576 and [r['label'] for r in commands]==expected
        for command in commands:
            assert command['returncode']==0 and command['descendants']==[] and command['timing_observations']
            own=command['process_instance']
            try:assert abs(psutil.Process(own['pid']).create_time()-own['create_time_unix'])>=.002
            except psutil.NoSuchProcess:pass
        O.write(ctx.out/'independent_head_positions.json',positions)
        ctx.r['gates'].update(independent24_fresh_book_selection96_exact_tasks=True,source_and_transformed_unique7415217408_parameters1536_columns=True,
            all_teacher_and_own_generation_full_head_cells_INTEGER_AVX_F32_BYTE=True,all_forced_teacher_and_common_generation_prefix_states_routes_BYTE=True,
            all_saved_official_donor_bridges_rubrics_and_frozen_decisions_reconstructed=True,all576_ordered_calls_exact_PID_creation_closed=True)
        ctx.finish({'summary':summary,'independent_head_states':head_positions,'candidate_tail_overtakes':tail,'teacher_head_changed_cells':changed,
            'native_model_compiler_calls':0,'physical_DRAM_verified':False,'cohort_sha256':prep['cohort_sha256']})
    except BaseException:ctx.fail();raise
    finally:ctx.close()
if __name__=='__main__':main()
