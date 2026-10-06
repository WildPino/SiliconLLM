"""Project-excluded, source-only selection; never imports or scores a model."""
import hashlib
import struct
from pathlib import Path

SEED='meth506-whole-conditional-fresh-506506'
CORPUS_REL='data/external/pg19/data/train-00014-of-00023-54b567998cd5eb4b.parquet'
def sha(data):return hashlib.sha256(data).hexdigest()

def build(ctx,b):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    excluded=set(b['exclusions']['rows']);assert len(excluded)>=110
    corpus=ctx.binding['corpus']['path'];parquet=pq.ParquetFile(corpus);assert parquet.metadata.num_rows==1243
    tokenizer=AutoTokenizer.from_pretrained(b['donor_directory'],local_files_only=True)
    sentinels=[tokenizer.convert_tokens_to_ids(f'<extra_id_{i}>') for i in range(5)]
    assert sentinels==[32099,32098,32097,32096,32095] and (tokenizer.eos_token_id,tokenizer.pad_token_id)==(1,0)
    ranks=sorted((row for row in range(1243) if row not in excluded),key=lambda row:sha(f'{SEED}|row={row}'.encode()))
    wanted=set(ranks[:96]);candidates={};i=0
    for batch in parquet.iter_batches(batch_size=4,columns=['text']):
        for text in batch.column('text').to_pylist():
            if i in wanted and isinstance(text,str):candidates[i]=text
            i+=1
        ctx.guard()
    assert i==1243;selected=[];rejected=[]
    for row in ranks[:96]:
        text=candidates.get(row,'');source_id=f'pg19:{CORPUS_REL}:row={row}'
        if len(text)<8192:rejected.append({'row':row,'reason':'less_than8192_characters'});continue
        begin=int(sha((SEED+'|'+source_id).encode()),16)%(len(text)-4096)
        excerpt=text[begin:begin+4096];tokens=tokenizer.encode(excerpt,add_special_tokens=False)
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
            cases.append({'index':index,'excerpt_token_start':start,'original_window_ids':original,'masked_spans_ids':spans,'source_ids':source,'decoder_ids':decoder,'target_ids':target,
                          'original_window_text':tokenizer.decode(original),'source_text':tokenizer.decode(source),'target_text':tokenizer.decode(target),
                          'source_ids_sha256':sha(struct.pack('<29i',*source)),'target_ids_sha256':sha(struct.pack('<14i',*target))})
        selected.append({'source_id':source_id,'corpus_row':row,'whole_source_utf8_sha256':sha(text.encode()),'source_characters':len(text),'excerpt_start_character':begin,'excerpt':excerpt,'excerpt_utf8_sha256':sha(excerpt.encode()),'cases':cases})
        if len(selected)==24:break
    assert len(selected)==24 and not {v['corpus_row'] for v in selected}&excluded
    return {'seed':SEED,'corpus':b['corpus'],'excluded_rows':sorted(excluded),'exclusion_binding_sha256':ctx.r['binding_sha256'],'items':selected,'source_only_rejections':rejected,'project_heldout_not_donor_pretraining_claim':True,'model_scores_used':0}
