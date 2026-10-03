"""Model-free source-disjoint span reconstruction cohort after346 actual head A16 cost PASS."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import time
import psutil
import pyarrow.parquet as pq
import transformers
from transformers import AutoTokenizer
import meth324_switch_reference as M

PROTOCOL=M.DOC/'METH_348_SWITCH_HEAD_A16_FRESH_MANIFEST_PROTOCOL_20261003.md'
COST=M.DOC/'meth346_switch_head_a16_cost_result.json'
COST_SHA='1d32e982f8392632f4e1745f3fec91b3cb59884aa3f679a085a5307eb1432fcf'
ACQUIRED=M.DOC/'meth326_switch_acquisition_result.json'
ACQUIRED_SHA='39bac2bda18cc6660da77bba8256d1e21fcad5b6fa70796d343b7e139d480711'
CORPUS_REL='data/external/pg19/data/train-00014-of-00023-54b567998cd5eb4b.parquet'
CORPUS_SHA='15f3ecb5e314ce58bbf8eae927b51bf893b9fdabe4a04c8e246547206357a2d5'
SOURCE=M.ROOT/'results/native_expert_scaling/meth326_switch_base256_source'
SEED='meth348-head-a16-new-span-348348'


def sha(data):return hashlib.sha256(data).hexdigest()


def digest(path,start):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(4<<20):h.update(block);guard(start)
    return h.hexdigest()


def guard(start):
    assert time.monotonic()-start<=600 and psutil.Process().memory_info().rss<=4<<30,'source_selection_resource_guard'


def prior_exclusions(start):
    paths=subprocess.check_output(['git','ls-files','--','docs/research/**/*.json','benchmarks/donor_adaptation/**/*.json'],cwd=M.ROOT,text=True).splitlines()
    pattern=re.compile(re.escape(('pg19:'+CORPUS_REL+':row=').encode())+rb'(\d+)')
    generic=re.compile(rb'"source_row"\s*:\s*(\d+)')
    excluded=set();records=[]
    for rel in sorted(paths):
        h=hashlib.sha256();rows=set();generic_rows=set();has_corpus=False;tail=b''
        with (M.ROOT/rel).open('rb') as stream:
            while block:=stream.read(4<<20):
                h.update(block);data=tail+block;has_corpus|=CORPUS_REL.encode() in data
                rows.update(int(v) for v in pattern.findall(data));generic_rows.update(int(v) for v in generic.findall(data))
                tail=data[-512:];guard(start)
        if has_corpus:rows.update(generic_rows)
        excluded.update(rows);records.append({'path':rel,'sha256':h.hexdigest(),'excluded_corpus_rows':sorted(rows)})
    assert records and len(excluded)>=60
    return excluded,records


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists()
    start=time.monotonic();stage='bindings';result={'experiment':'METH-348-source-only-NEW-head-A16-span-cohort'}
    try:
        for path in (Path(__file__),PROTOCOL,COST,ACQUIRED,Path(M.__file__)):M.committed(path)
        assert M.digest(COST)==COST_SHA and all(json.loads(COST.read_text(encoding='utf-8'))['gates'].values())
        assert M.digest(ACQUIRED)==ACQUIRED_SHA and transformers.__version__=='4.57.6'
        assert digest(M.ROOT/CORPUS_REL,start)==CORPUS_SHA
        acquisition=json.loads(ACQUIRED.read_text(encoding='utf-8'));sidefiles=[]
        for row in acquisition['files']:
            if 'pytorch_model-' not in Path(row['path']).name:
                assert digest(row['path'],start)==row['sha256'];sidefiles.append(row)
        tokenizer=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
        first=tokenizer.convert_tokens_to_ids('<extra_id_0>');second=tokenizer.convert_tokens_to_ids('<extra_id_1>')
        assert (first,second,tokenizer.eos_token_id,tokenizer.pad_token_id)==(32099,32098,1,0)
        stage='all_tracked_prior_source_exclusions';excluded,priors=prior_exclusions(start)
        parquet=pq.ParquetFile(M.ROOT/CORPUS_REL);assert parquet.metadata.num_rows==1243
        ranks=sorted((row for row in range(1243) if row not in excluded),key=lambda row:sha(f'{SEED}|row={row}'.encode()))
        wanted=set(ranks[:96]);candidates={};row_index=0
        stage='model_free_corpus_spans'
        for batch in parquet.iter_batches(batch_size=4,columns=['text']):
            for text in batch.column('text').to_pylist():
                if row_index in wanted and isinstance(text,str):candidates[row_index]=text
                row_index+=1
            guard(start)
        assert row_index==1243
        selected=[];rejected=[]
        for row in ranks[:96]:
            text=candidates.get(row,'');source_id=f'pg19:{CORPUS_REL}:row={row}'
            if len(text)<8192:rejected.append({'source_id':source_id,'reason':'less_than8192_characters'});continue
            key=int(sha((SEED+'|'+source_id).encode()),16);begin=key%(len(text)-4096)
            excerpt=text[begin:begin+4096];tokens=tokenizer.encode(excerpt,add_special_tokens=False)
            if len(tokens)<512 or len(set(tokens))<128 or any(v<=1 or v>=32000 for v in tokens):
                rejected.append({'source_id':source_id,'reason':'source_only_token_length_diversity_or_special_id_guard'});continue
            cases=[];quarter=len(tokens)//4
            for index in range(4):
                window_start=index*quarter+int(sha(f'{SEED}|{source_id}|window{index}'.encode()),16)%(quarter-63+1)
                original=tokens[window_start:window_start+63];span_start=12+index*8
                target=[first]+original[span_start:span_start+8]+[second,1]
                source=original[:span_start]+[first]+original[span_start+8:]+[1]
                decoder=[0]+target[:-1]
                assert len(source)==57 and len(target)==len(decoder)==11
                cases.append({'index':index,'excerpt_token_start':window_start,'original_window_ids':original,'span_start':span_start,
                              'masked_span_ids':original[span_start:span_start+8],'source_ids':source,'decoder_ids':decoder,'target_ids':target,
                              'original_window_text':tokenizer.decode(original),'source_text':tokenizer.decode(source),'target_text':tokenizer.decode(target),
                              'source_ids_sha256':sha(struct.pack('<57i',*source)),'target_ids_sha256':sha(struct.pack('<11i',*target))})
            selected.append({'source_id':source_id,'corpus_row':row,'whole_source_utf8_sha256':sha(text.encode()),'source_characters':len(text),
                             'excerpt_start_character':begin,'excerpt':excerpt,'excerpt_utf8_sha256':sha(excerpt.encode()),'cases':cases})
            if len(selected)==24:break
        assert len(selected)==24 and len({v['corpus_row'] for v in selected})==24
        assert not {v['corpus_row'] for v in selected}&excluded
        result.update({'controller_sha256':M.digest(__file__),'protocol_sha256':M.digest(PROTOCOL),'cost346_sha256':COST_SHA,
                       'source_acquisition_sha256':ACQUIRED_SHA,'model':acquisition['model'],'revision':acquisition['revision'],
                       'source_tokenizer_files':sidefiles,'tokenizer_class':type(tokenizer).__name__,'transformers':transformers.__version__,
                       'corpus':CORPUS_REL,'corpus_sha256':CORPUS_SHA,'corpus_rows':1243,'seed':SEED,'prior_tracked_records':priors,
                       'excluded_corpus_rows':sorted(excluded),'source_only_rejections':rejected,'items':selected,
                       'gates':{'all24_new_corpus_rows':True,'all96_source_and_known_span_labels':True,'no_model_scoring':True},
                       'resource':{'main_seconds_excluding_imports':time.monotonic()-start,'end_rss_bytes':psutil.Process().memory_info().rss},
                       'decision':'new_project_source_cohort_frozen_require_original_donor_quality_protocol_before_scores',
                       'scope':'24 disjoint PG19train14 book rows excluded against tracked research/donor JSON source IDs/source-row references. Four separate63-token windows/book, one8-token masked span. No donor-pretraining exclusion guarantee, instruction/chat quality, donor scores, accepted rate or larger-n proof.'})
        guard(start);M.write(args.out,result);print(json.dumps({'sha256':M.digest(args.out),'selected_rows':[v['corpus_row'] for v in selected],
             'excluded_rows':len(excluded),'gates':result['gates'],'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'stage':stage,'error':repr(error),'seconds':time.monotonic()-start});M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
