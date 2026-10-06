"""One source-only fresh selection; DuckDB physical row and retained-tokenizer controls."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import meth511_operations as O

SEED='meth511-whole-hybrid-head-fresh-511511'
CORPUS_REL='data/external/pg19/data/train-00014-of-00023-54b567998cd5eb4b.parquet'
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding-sha',required=True);args=ap.parse_args();ctx=O.Context('cohort')
    try:
        b=ctx.admit(args.binding_sha)
        import duckdb
        from tokenizers import Tokenizer
        assert duckdb.__version__=='1.4.4'
        conn=duckdb.connect(config={'threads':'1','memory_limit':'512MB','autoinstall_known_extensions':'false','autoload_known_extensions':'false'})
        conn.execute('LOAD parquet')
        tok=Tokenizer.from_file(str(Path(b['donor_directory'])/'tokenizer.json'))
        sentinels=[tok.token_to_id(f'<extra_id_{i}>') for i in range(5)]
        assert sentinels==[32099,32098,32097,32096,32095] and tok.token_to_id('</s>')==1 and tok.token_to_id('<pad>')==0
        excluded=set(b['exclusions']['rows']);ranks=sorted(set(range(1243))-excluded,key=lambda row:sha(f'{SEED}|row={row}'.encode()))
        assert len(ranks)>=96
        old=json.loads(Path(b['cohort_control']['path']).read_bytes());controls=old['items'][:2]
        wanted=set(ranks[:96])|{r['corpus_row'] for r in controls};texts={};i=0
        cursor=conn.execute('SELECT file_row_number,text FROM read_parquet(?,file_row_number=true) ORDER BY file_row_number',[b['corpus']['path']])
        while batch:=cursor.fetchmany(4):
            for row,text in batch:
                assert row==i;i+=1
                if row in wanted and isinstance(text,str):texts[row]=text
            ctx.guard()
        assert i==1243
        control_receipts=[]
        for item in controls:
            text=texts[item['corpus_row']];assert sha(text.encode())==item['whole_source_utf8_sha256']
            assert text[item['excerpt_start_character']:item['excerpt_start_character']+4096]==item['excerpt']
            tokens=tok.encode(item['excerpt'],add_special_tokens=False).ids
            for case in item['cases']:assert tokens[case['excerpt_token_start']:case['excerpt_token_start']+32]==case['original_window_ids']
            control_receipts.append({'row':item['corpus_row'],'whole_text_utf8_BYTE':True,'all4_windows_original_AutoTokenizer_ids_BYTE':True})
        selected=[];rejected=[]
        for row in ranks[:96]:
            text=texts.get(row,'');source_id=f'pg19:{CORPUS_REL}:row={row}'
            if len(text)<8192:rejected.append({'row':row,'reason':'less_than8192_characters'});continue
            begin=int(sha((SEED+'|'+source_id).encode()),16)%(len(text)-4096);excerpt=text[begin:begin+4096]
            tokens=tok.encode(excerpt,add_special_tokens=False).ids
            if len(tokens)<512 or len(set(tokens))<128 or any(v<=1 or v>=32000 for v in tokens):
                rejected.append({'row':row,'reason':'token_length_diversity_or_special_id'});continue
            cases=[];quarter=len(tokens)//4
            for index in range(4):
                start=index*quarter+int(sha(f'{SEED}|{source_id}|window{index}'.encode()),16)%(quarter-32+1)
                original=tokens[start:start+32];spans=[original[k:k+2] for k in [3,10,17,24]];source=[];target=[];previous=0
                for si,(k,span) in enumerate(zip([3,10,17,24],spans)):
                    source.extend(original[previous:k]);source.append(sentinels[si]);previous=k+2;target.extend([sentinels[si],*span])
                source.extend(original[previous:]);source.append(1);target.extend([32095,1]);decoder=[0]+target[:-1]
                assert len(source)==29 and len(target)==len(decoder)==14
                cases.append({'index':index,'excerpt_token_start':start,'original_window_ids':original,'masked_spans_ids':spans,
                    'source_ids':source,'decoder_ids':decoder,'target_ids':target,
                    'original_window_text':tok.decode(original,skip_special_tokens=False),'source_text':tok.decode(source,skip_special_tokens=False),'target_text':tok.decode(target,skip_special_tokens=False),
                    'source_ids_sha256':sha(struct.pack('<29i',*source)),'target_ids_sha256':sha(struct.pack('<14i',*target))})
            selected.append({'source_id':source_id,'corpus_row':row,'whole_source_utf8_sha256':sha(text.encode()),'source_characters':len(text),
                'excerpt_start_character':begin,'excerpt':excerpt,'excerpt_utf8_sha256':sha(excerpt.encode()),'cases':cases})
            if len(selected)==24:break
        conn.close();assert len(selected)==24 and not {r['corpus_row'] for r in selected}&excluded
        cohort={'seed':SEED,'corpus':b['corpus'],'excluded_rows':sorted(excluded),'items':selected,'source_only_rejections':rejected,
            'binding_sha256':args.binding_sha,'reader_tokenizer_controls':control_receipts,'model_scores_used':0,
            'project_heldout_not_donor_pretraining_claim':True,'decode_scope':'Rust tokenizer decode with skip_special_tokens=False; display-only punctuation cleanup is not applied. Model ID construction retains506 recipe.'}
        O.write(ctx.out/'cohort.json',cohort)
        ctx.r['gates'].update(all1243_physical_rows_and_two_consumed_text_tokenizer_controls=True,all24_project_excluded_books96_source_only_tasks=True,zero_model_native_GPU_calls=True)
        ctx.finish({'cohort_sha256':ctx.digest(ctx.out/'cohort.json'),'selected_rows':[r['corpus_row'] for r in selected],
            'summary':{'books':24,'tasks':96,'excluded_rows':len(excluded),'source_only_rejections':len(rejected)}})
    except BaseException:ctx.fail();raise
    finally:ctx.close()
if __name__=='__main__':main()
