"""Independent retained-data audit. No main, artifact, quality or results imports."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import meth506_operations as O

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding-sha',required=True);ap.add_argument('--raw-sha',required=True);args=ap.parse_args()
    destination=O.DOC/'RETENTION_506_20261006.json';ctx=O.Context(O.AUDIT,destination)
    try:
        b=ctx.admit(args.binding_sha);assert ctx.digest(O.RAW)==args.raw_sha;ctx.head(O.RAW);raw=json.loads(O.RAW.read_bytes());assert all(raw['gates'].values())
        prepfile=O.DOC/'meth506_preparation_result.json';assert ctx.digest(prepfile)==raw['preparation_sha256'];ctx.head(prepfile);prep=json.loads(prepfile.read_bytes())
        for entry in prep['output_inventory']+raw['output_inventory']:ctx.exact(entry)
        for entry in b['exclusions']['ledgers']:ctx.exact(entry)
        import numpy as np
        import pyarrow.parquet as pq
        from transformers import AutoTokenizer
        from threadpoolctl import threadpool_limits
        pools=threadpool_limits(limits=1);np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.r['numerical_imports']=True
        ctx.phase='independent-manifest-and-whole-inverse'
        def parse(path):
            content=Path(path).read_bytes();config=content[8:64];nf,nt=struct.unpack_from('<II',content,64);i=72
            def text():
                nonlocal i
                n,=struct.unpack_from('<I',content,i);i+=4;v=content[i:i+n].decode('utf8');i+=n;return v
            files=[text() for _ in range(nf)];tensors={}
            for _ in range(nt):
                name=text();values=struct.unpack_from('<5I3Q',content,i);i+=44;assert name not in tensors;tensors[name]=values
            assert i==len(content) and nf==1 and nt==3320
            return content[:8],config,files,tensors
        sm,sc,sf,st=parse(b['source_manifest']['path']);cm,cc,cf,ct=parse(raw['artifact']['manifest']);assert sm==b'SWI8A001' and cm==b'SWI8C001' and sc==cc and set(st)==set(ct)
        changed=[]
        for name,a in st.items():
            c=ct[name];wo='.mlp.experts.expert_' in name and name.endswith('.wo.weight');assert a[:4]==c[:4] and a[5:]==c[5:] and c[4]==(2 if wo else a[4])
            if wo:assert a[1:5]==(2,768,3072,1);changed.append((a[5],a[7],name))
        assert len(changed)==1536 and Path(sf[0]).stat().st_size==Path(cf[0]).stat().st_size==7541946880
        end=0
        with Path(sf[0]).open('rb') as source,Path(cf[0]).open('rb') as candidate:
            def unchanged(stop):
                nonlocal end
                while end<stop:
                    n=min(8<<20,stop-end);x=source.read(n);y=candidate.read(n);assert len(x)==len(y)==n and x==y;end+=n;ctx.guard()
            for offset,n,name in sorted(changed):
                unchanged(offset);original=source.read(n);columns=candidate.read(n);assert len(original)==len(columns)==n
                # Columns encode the original 768x3072 matrix in Fortran order.
                restored=np.frombuffer(columns,dtype='i1').reshape((768,3072),order='F').tobytes(order='C');assert restored==original;end+=n;ctx.guard()
            unchanged(7541946880);assert source.read(1)==candidate.read(1)==b''
        ctx.r['gates']['ALL1536_WO_and_ALL_source_other_extents_scales_padding_inverse_BYTE']=True
        ctx.phase='independent-fresh-source-and-task-reconstruction';cohort=json.loads((O.PREP/'cohort.json').read_bytes());assert ctx.digest(O.PREP/'cohort.json')==raw['cohort_sha256']
        excluded=set(b['exclusions']['rows']);rows=[v['corpus_row'] for v in cohort['items']];assert len(rows)==len(set(rows))==24 and not set(rows)&excluded
        sha=lambda v:hashlib.sha256(v).hexdigest();seed='meth506-whole-conditional-fresh-506506'
        ranks=sorted((i for i in range(1243) if i not in excluded),key=lambda i:sha(f'{seed}|row={i}'.encode()));assert all(i in ranks[:96] for i in rows)
        texts={};i=0
        for batch in pq.ParquetFile(b['corpus']['path']).iter_batches(batch_size=4,columns=['text']):
            for text in batch.column('text').to_pylist():
                if i in ranks[:96]:texts[i]=text
                i+=1
            ctx.guard()
        assert i==1243;tokenizer=AutoTokenizer.from_pretrained(b['donor_directory'],local_files_only=True)
        independently_selected=[];rejections=[]
        for row in ranks[:96]:
            text=texts.get(row,'');sid=f'pg19:data/external/pg19/data/train-00014-of-00023-54b567998cd5eb4b.parquet:row={row}'
            if len(text)<8192:rejections.append({'row':row,'reason':'less_than8192_characters'});continue
            begin=int(sha((seed+'|'+sid).encode()),16)%(len(text)-4096);tokens=tokenizer.encode(text[begin:begin+4096],add_special_tokens=False)
            if len(tokens)<512 or len(set(tokens))<128 or any(v<=1 or v>=32000 for v in tokens):
                rejections.append({'row':row,'reason':'token_length_diversity_or_special_id'});continue
            independently_selected.append(row)
            if len(independently_selected)==24:break
        assert independently_selected==rows and rejections==cohort['source_only_rejections']
        for book in cohort['items']:
            text=texts[book['corpus_row']];assert sha(text.encode())==book['whole_source_utf8_sha256'];begin=int(sha((seed+'|'+book['source_id']).encode()),16)%(len(text)-4096)
            assert begin==book['excerpt_start_character'] and text[begin:begin+4096]==book['excerpt'];tokens=tokenizer.encode(book['excerpt'],add_special_tokens=False);quarter=len(tokens)//4
            assert len(tokens)>=512 and len(set(tokens))>=128 and all(1<x<32000 for x in tokens)
            for case in book['cases']:
                j=case['index'];start=j*quarter+int(sha(f'{seed}|{book["source_id"]}|window{j}'.encode()),16)%(quarter-32+1);original=tokens[start:start+32]
                assert start==case['excerpt_token_start'] and original==case['original_window_ids'];source=[];target=[];last=0;truth=[]
                for k,p in enumerate([3,10,17,24]):
                    truth.append(original[p:p+2]);source+=original[last:p]+[32099-k];target+=[32099-k]+original[p:p+2];last=p+2
                source+=original[last:]+[1];target+=[32095,1]
                assert source==case['source_ids'] and target==case['target_ids'] and [0]+target[:-1]==case['decoder_ids'] and truth==case['masked_spans_ids']
                assert sha(struct.pack('<29i',*source))==case['source_ids_sha256'] and sha(struct.pack('<14i',*target))==case['target_ids_sha256']
        ctx.r['gates']['independent24_fresh_book_bytes_all96_input_answer_masks']=True
        def wire(path):
            data=Path(path).read_bytes();assert data[:8]==b'SWR32O01';s,t,d,e,f,v,n=struct.unpack_from('<7I',data,8);assert (s,d,e,f,v,n)==(29,768,12,12,32128,6*(29+t)) and 0<t<=64
            a=14*29*768;de=t*14*768;lo=t*32128;stop=36+4*(a+de+lo);assert len(data)==stop+12*n
            values=np.frombuffer(data,dtype='<f4',count=a+de+lo,offset=36);assert np.isfinite(values).all()
            route=np.frombuffer(data,dtype=[('expert','<i4'),('accepted','<i4'),('p','<f4')],count=n,offset=stop);assert ((0<=route['expert'])&(route['expert']<128)).all() and np.isin(route['accepted'],[0,1]).all() and ((0<route['p'])&(route['p']<=1)).all()
            return values[a+de:].reshape(t,v)
        # Independent task parser and dynamic programming recurrences.
        def task(ids,truth):
            fields=[[] for _ in range(4)];cursor=-1;expect=32099;bad=False;closed=False
            for token in ids:
                if token==1:break
                if token==0:bad=True
                elif token>=32000:
                    if token!=expect or expect<32095:bad=True
                    else:
                        cursor=32099-token;expect-=1
                        if token==32095:closed=True;cursor=-1
                elif cursor in range(4):fields[cursor].append(token)
                else:bad=True
            if bad:fields=[[] for _ in range(4)]
            count=distinct=0
            for field in fields:
                ts=list(zip(field,field[1:],field[2:]));count+=len(ts);distinct+=len(set(ts))
            repeated=(count-distinct)/max(count,1);health=closed and expect==32094 and not bad and all(fields) and ids[-1]==32095 and repeated<=.5
            scores=[]
            for field,gold in zip(fields,truth):
                grid=np.zeros((len(field)+1,len(gold)+1),dtype='i4')
                for i,x in enumerate(field,1):
                    for j,y in enumerate(gold,1):grid[i,j]=grid[i-1,j-1]+1 if x==y else max(grid[i-1,j],grid[i,j-1])
                scores.append(2*int(grid[-1,-1])/(len(field)+len(gold)))
            correct=sum(sum(j<len(field) and field[j]==gold[j] for j in range(2)) for field,gold in zip(fields,truth));exact=sum(field==gold for field,gold in zip(fields,truth));prose=[x for x in ids if 1<x<32000]
            return {'healthy_complete_nonempty_fields':bool(health),'fields':fields,'correct_fields':exact,'known_token_accuracy':correct/8,'known_field_exact_accuracy':exact/4,'known_field_lcs_f1':float(np.mean(scores)),'generated_ids':ids,'generated_tokens':len(ids),'prose_tokens':len(prose),'prose_ids':prose,'within_field_repeated_trigram_fraction':repeated,'cap_reached_without_terminal':len(ids)==64 and ids[-1] not in [1,32095]}
        def prediction(logits,case):
            x=logits.astype('f8');mx=x.max(axis=1);nll=np.log(np.sum(np.exp(x-mx[:,None]),axis=1))+mx-x[np.arange(14),case['target_ids']];ids=x.argmax(1);mask=np.array([1,2,4,5,7,8,10,11]);right=ids[mask]==np.array(case['masked_spans_ids']).flatten()
            return {'mean_nll':float(np.mean(nll)),'mean_span_nll':float(np.mean(nll[mask])),'correct_span_tokens':int(right.sum()),'correct_fields':int(right.reshape(4,2).all(axis=1).sum()),'all_span_teacher_forced_correct':bool(right.all()),'greedy_teacher_forced_ids':ids.tolist(),'per_target_nll':nll.tolist()}
        def close(actual,expected):
            if isinstance(expected,dict):
                assert set(actual)==set(expected)
                for k,v in expected.items():close(actual[k],v)
            elif isinstance(expected,list):
                assert len(actual)==len(expected)
                for a,e in zip(actual,expected):close(a,e)
            elif isinstance(expected,float):assert abs(actual-expected)<=1e-10*max(abs(expected),1),(actual,expected)
            else:assert actual==expected,(actual,expected)
        def edit(a,b):
            grid=np.empty((len(a)+1,len(b)+1),dtype='i4');grid[:,0]=np.arange(len(a)+1);grid[0,:]=np.arange(len(b)+1)
            for i,x in enumerate(a,1):
                for j,y in enumerate(b,1):grid[i,j]=min(grid[i-1,j]+1,grid[i,j-1]+1,grid[i-1,j-1]+(x!=y))
            return int(grid[-1,-1])/max(len(a),len(b),1)
        ctx.phase='independent-all-retained-whole-wires-and-official-bridges';cases=raw['cases'];assert len(cases)==96;qualified=[]
        for ordinal,c in enumerate(cases):
            bi,ci=divmod(ordinal,4);assert c['book']==bi and c['index']==ci and c['case']==cohort['items'][bi]['cases'][ci]
            teacher=[];generated=[]
            for arm in ['source','candidate']:
                native=c['native'][arm]
                for kind,expected_reps,profile in [('teacher',[0],0),('generation',[-1,0,1,2],0),('profile',[0],1)]:
                    records=native[kind];assert [r['repetition'] for r in records]==expected_reps
                    for row in records:
                        assert row['threads']==3 and row['profile']==profile and [v['actual_mask'] for v in row['worker_affinity']]==[1,4,16] and all(v['group']==0 for v in row['worker_affinity'])
                        ctx.exact(row['wire']);logits=wire(row['wire']['path'])
                        if kind=='teacher':teacher.append(row['wire']['sha256'])
                        else:assert row['generated_ids']==logits.argmax(1).tolist();generated.append(row['wire']['sha256'])
                assert len(native['profile'][0]['conditional_columns'])==12
                if arm=='source':assert all(v['calls']==v['nonzero_sum']==v['maximum']==0 for v in native['profile'][0]['conditional_columns'])
                else:
                    for v in native['profile'][0]['conditional_columns']:assert 0<=v['nonzero_sum']<=3072*v['calls'] and 0<=v['maximum']<=3072
            assert len(set(teacher))==len(set(generated))==1
            cand=wire(c['native']['candidate']['teacher'][0]['wire']['path']);ids=c['native']['candidate']['generation'][0]['generated_ids'];ctx.exact(c['donor_reference'])
            with np.load(c['donor_reference']['path'],allow_pickle=False) as ref:
                for v in ref.values():assert np.isfinite(v).all()
                dl=ref['teacher_logits'];gl=ref['generation_logits'];di=gl.argmax(1).tolist()
                assert ref['teacher_encoder'].tobytes()==ref['generation_encoder'].tobytes()
                for a,d,key in [(ref['official_teacher_logits'],dl,'official_teacher_relative_l2'),(ref['official_generation_logits'],gl,'official_generation_relative_l2')]:
                    error=float(np.linalg.norm(a.astype('f8')-d.astype('f8'))/np.linalg.norm(d.astype('f8')));assert error<=1e-6 and np.array_equal(a.argmax(1),d.argmax(1));close(error,c[key])
                cp=prediction(cand,c['case']);dp=prediction(dl,c['case']);ctask=task(ids,c['case']['masked_spans_ids']);dtask=task(di,c['case']['masked_spans_ids'])
                close(cp,c['candidate_prediction']);close(dp,c['donor_prediction']);close(ctask,c['candidate_task']);close(dtask,c['donor_task'])
                top=float(np.mean(cand.argmax(1)==dl.argmax(1)));mask=np.array([1,2,4,5,7,8,10,11]);mtop=float(np.mean(cand.argmax(1)[mask]==dl.argmax(1)[mask]));ed=edit(ctask['prose_ids'],dtask['prose_ids'])
                close(top,c['top1_agreement']);close(mtop,c['masked_top1_agreement']);close(ed,c['normalized_prose_edit'])
                qualified.append({'cp':cp,'dp':dp,'ct':ctask,'dt':dtask,'top':top,'masked':mtop,'edit':ed})
            ctx.guard()
            if ordinal%4==3:ctx.log(audited_book=bi)
        ctx.r['gates']['ALL96_teacher_960_generation_wires_BYTE_official_bridges_tasks_NLL']=True
        ctx.phase='independent-frozen-book-bootstrap-and-decisions';s=raw['summary'];paired={}
        def interval(fn):
            values=np.array([np.mean([fn(c) for c in qualified[4*i:4*i+4]]) for i in range(24)]);indices=np.random.default_rng(351351).integers(0,24,size=(10000,24));draws=np.mean(values[indices],axis=1)
            return {'mean':float(np.mean(values)),'one_sided_lower95':float(np.quantile(draws,.05)),'one_sided_upper95':float(np.quantile(draws,.95)),'bootstrap_unit':'book','draws':10000,'seed':351351}
        for key in ['mean_nll','mean_span_nll']:paired[key]=interval(lambda c:c['cp'][key]-c['dp'][key])
        paired['teacher_masked_token_accuracy']=interval(lambda c:(c['cp']['correct_span_tokens']-c['dp']['correct_span_tokens'])/8);paired['teacher_field_exact_accuracy']=interval(lambda c:(c['cp']['correct_fields']-c['dp']['correct_fields'])/4)
        for key in ['known_token_accuracy','known_field_exact_accuracy','known_field_lcs_f1','healthy_complete_nonempty_fields','within_field_repeated_trigram_fraction']:paired['generation_'+key]=interval(lambda c:float(c['ct'][key])-float(c['dt'][key]))
        paired['prose_edit']=interval(lambda c:c['edit']);close(paired,s['paired'])
        ix=np.random.default_rng(485485).integers(0,24,size=(10000,24));times={};rates={}
        for arm in ['source','candidate']:
            times[arm]=np.array([sum(np.mean([r['full_generation_seconds'] for r in c['native'][arm]['generation'] if r['repetition']>=0]) for c in cases[4*i:4*i+4]) for i in range(24)])
            loads=[sum(c['native'][arm]['generation'][0]['load_seconds'] for c in cases[4*i:4*i+4]) for i in range(24)];cold=[sum(c['native'][arm]['generation'][0]['load_seconds']+c['native'][arm]['generation'][0]['full_generation_seconds'] for c in cases[4*i:4*i+4]) for i in range(24)]
            rate={'book_seconds':times[arm].tolist(),'sum_case_mean_seconds':float(sum(times[arm])),'sum_load_seconds':sum(loads),'sum_first_request_with_load_seconds':sum(cold),'bootstrap_unit':'book','draws':10000,'seed':485485}
            for label,key in [('ordinary','generated_tokens'),('prose','prose_tokens')]:
                nums=np.array([sum(c['ct'][key] if c['ct']['healthy_complete_nonempty_fields'] else 0 for c in qualified[4*i:4*i+4]) for i in range(24)]);total=int(sum(nums));draws=nums[ix].sum(1)/times[arm][ix].sum(1)
                rate[label]={'accepted_ids':total,'warm_rate':total/float(sum(times[arm])),'warm_lower95':float(np.quantile(draws,.05)),'warm_upper95':float(np.quantile(draws,.95)),'first_request_load_charged_rate':total/sum(cold),'amortized_rates_requests_per_process':{str(k):total/(float(sum(times[arm]))+sum(loads)/k) for k in [1,10,100]}}
            close(rate,s['rates'][arm]);rates[arm]=rate
        ratio=times['candidate']/times['source'];draws=times['candidate'][ix].sum(1)/times['source'][ix].sum(1)
        cost={'ratio_of_sums':float(sum(times['candidate'])/sum(times['source'])),'book_ratios':ratio.tolist(),'bootstrap_lower95':float(np.quantile(draws,.05)),'bootstrap_upper95':float(np.quantile(draws,.95)),'regressing_books':np.where(ratio>1)[0].tolist(),'worst_book_ratio':float(max(ratio)),'seed':485485,'draws':10000};close(cost,s['matched_cost'])
        top=np.mean([c['top'] for c in qualified]);masked=np.mean([c['masked'] for c in qualified]);signal=np.mean([c['dt']['known_field_exact_accuracy'] for c in qualified]);dh=np.mean([c['dt']['healthy_complete_nonempty_fields'] for c in qualified]);ch=np.mean([c['ct']['healthy_complete_nonempty_fields'] for c in qualified])
        for a,key in [(top,'all_top1_agreement'),(masked,'masked_top1_agreement'),(signal,'donor_field_signal'),(dh,'donor_health'),(ch,'candidate_health')]:close(float(a),s[key])
        u=lambda k,v:paired[k]['one_sided_upper95']<=v;l=lambda k,v:paired[k]['one_sided_lower95']>=v
        qg={'all_nll_upper_le.05':u('mean_nll',.05),'masked_nll_upper_le.05':u('mean_span_nll',.05),'teacher_masked_accuracy_lower_ge-.02':l('teacher_masked_token_accuracy',-.02),'teacher_field_accuracy_lower_ge-.05':l('teacher_field_exact_accuracy',-.05),'all_top1_ge.95':top>=.95,'masked_top1_ge.95':masked>=.95,'donor_field_signal_ge.10':signal>=.10,'donor_health_ge.80':dh>=.80,'candidate_health_ge.80':ch>=.80,'generation_token_accuracy_lower_ge-.02':l('generation_known_token_accuracy',-.02),'generation_field_accuracy_lower_ge-.05':l('generation_known_field_exact_accuracy',-.05),'generation_LCS_lower_ge-.02':l('generation_known_field_lcs_f1',-.02),'generation_health_lower_ge-.05':l('generation_healthy_complete_nonempty_fields',-.05),'generation_repeat_upper_le.05':u('generation_within_field_repeated_trigram_fraction',.05),'prose_edit_upper_le.10':u('prose_edit',.10)}
        eg={'whole_mean_ratio_le.95':cost['ratio_of_sums']<=.95,'whole_ratio_upper_le1':cost['bootstrap_upper95']<=1,'all_book_ratio_le1':bool(np.all(ratio<=1)),'candidate_ordinary_warm_and_lower_ge50':rates['candidate']['ordinary']['warm_rate']>=50 and rates['candidate']['ordinary']['warm_lower95']>=50,'candidate_prose_warm_and_lower_ge50':rates['candidate']['prose']['warm_rate']>=50 and rates['candidate']['prose']['warm_lower95']>=50}
        close(qg,s['quality_gates']);close(eg,s['economic_gates']);assert s['whole_recipe_eligible']==(all(qg.values()) and all(eg.values()))
        ctx.r['gates']['independent_book_bootstrap_all_frozen_quality_cost_rate_decisions']=True
        assert all(cmd['returncode']==0 and cmd['observed_modules_prebound_or_actual_build'] for cmd in raw['commands'])
        assert len(raw['commands'])==578 and len(raw['source_shards'])==3
        ctx.r['gates']['retained_compile_controls_576_model_processes_module_checks_and_OS_bounds']=True
        ctx.parent_modules()
        r=ctx.finish({'raw_sha256':args.raw_sha,'preparation_sha256':raw['preparation_sha256'],'quality_gates':qg,'economic_gates':eg,'whole_recipe_eligible':s['whole_recipe_eligible'],'all_bank_inverse_bytes':7541946880,'fresh_books':24,'fresh_cases':96,'scope':'Independent data/format/arithmetic/task/decision rederivation from retained files; no new model or native calls. Physical DRAM and generality remain unverified.'});print(json.dumps({'retention':str(destination),'sha256':ctx.digest(destination),'gates':r['gates'],'resource':r['resource']}),flush=True)
    except BaseException as e:ctx.fail(e);raise
    finally:ctx.done.set();ctx.watchdog.cancel();ctx.watcher.join(timeout=1);ctx.progress.close();ctx.fatal.close()

if __name__=='__main__':main()
