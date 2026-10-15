"""Bounded source-byte acquisition for one gpt-oss-120b attention layer and used embeddings."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time
import traceback
from urllib.parse import quote
from urllib.request import Request, urlopen

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / 'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path.insert(0, str(SITE))
DOC = ROOT / 'docs/research/PIPELINE_REASSESSMENT_20261015'
PROTOCOL = DOC / 'FIRST_ATTENTION_ACQUISITION_PROTOCOL_20261015.md'
META = DOC / 'large_family_metadata'
CONFIG = META / 'config120.json'
INDEX = META / 'index120.json'
REFERENCE = META / 'reference_model.py'
CORPUS = Path('E:/SiliconLLM/results/native_expert_scaling/chatbot_source_coverage_20261015/corpus.json')
SOURCE_PLAN = ROOT / 'results/native_expert_scaling/chatbot_source_coverage_plan_20261015/cases.json'
OUT = ROOT / 'results/native_expert_scaling/first_attention_source_20261015'
REVISION = 'b5c939de8f754692c1647ca79fbf85e8c1e70f8a'
BASE = 'https://huggingface.co/openai/gpt-oss-120b/resolve/' + REVISION + '/'
CORPUS_SHA = '1c42eb789f52258d7796bcaeb8b52387c648ffe803f8955ecea5c2f3bdbd9ad6'
CONFIG_SHA = '933aeb666a3fd851133ddd7686414f369bc564c4185fb5704416550879f10566'
REFERENCE_SHA = '319ba3c6e4d827eb3115fe4838942af448376b513c02cea4b6b471bf6c21a346'
SOURCE_PLAN_SHA = '67aa70f4b24189e407582262d3a01df57c0ca62bcce4b109f070741dbdb071fa'
CREATED_OUTPUT = False
MAX_SECONDS = 900
MAX_RAM = 2 << 30
MAX_OUTPUT = 512 << 20
MAX_METADATA = 16 << 20
MAX_TOKENIZER = 64 << 20
MAX_WEIGHTS = 256 << 20
MAX_NETWORK = 384 << 20
MAX_CALLS = 2048
MAX_SPAN = 2 << 20
MAX_GAPS = 100 << 20
FIXED_CLOCK_DATE = '2026-10-15'
HOLDOUT_SALT = 'first-attention-source-holdout-v1:'
RANGE = re.compile(r'bytes (\d+)-(\d+)/(\d+)')
TARGETS = ('q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight',
           'q_proj.bias', 'k_proj.bias', 'v_proj.bias', 'o_proj.bias', 'sinks')
EXPECTED_SHAPES = {'model.layers.0.input_layernorm.weight':[2880],
    'model.layers.0.self_attn.q_proj.weight':[4096,2880],
    'model.layers.0.self_attn.k_proj.weight':[512,2880],
    'model.layers.0.self_attn.v_proj.weight':[512,2880],
    'model.layers.0.self_attn.o_proj.weight':[2880,4096],
    'model.layers.0.self_attn.q_proj.bias':[4096],
    'model.layers.0.self_attn.k_proj.bias':[512],
    'model.layers.0.self_attn.v_proj.bias':[512],
    'model.layers.0.self_attn.o_proj.bias':[2880],
    'model.layers.0.self_attn.sinks':[64]}


def desc(path):
    path = Path(path).resolve(); h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def freeze(path, rev):
    blob = subprocess.check_output(['git', 'show', rev + ':' + path.relative_to(ROOT).as_posix()], cwd=ROOT)
    assert blob.replace(b'\r\n', b'\n') == path.read_bytes().replace(b'\r\n', b'\n'), path


def ids_sha(ids):
    h = hashlib.sha256()
    for token in ids: h.update(struct.pack('<I', token))
    return h.hexdigest()


def selected_fields(raw):
    """Decode public-prefix fields only; skip stored Falcon outputs/logits lexically."""
    decoder = json.JSONDecoder(); wanted = {'id','split','domain','group_key','prefix_sha256','messages','upstream'}
    i = raw.index('{') + 1; found = {}
    while True:
        while raw[i].isspace() or raw[i] == ',': i += 1
        if raw[i] == '}': break
        key, i = decoder.raw_decode(raw, i)
        while raw[i].isspace() or raw[i] == ':': i += 1
        begin = i; depth = 0; string = False; escaped = False
        while i < len(raw):
            c = raw[i]
            if string:
                if escaped: escaped = False
                elif c == '\\': escaped = True
                elif c == '"': string = False
            elif c == '"': string = True
            elif c in '[{': depth += 1
            elif c in ']}':
                if depth == 0: break
                depth -= 1
            elif c == ',' and depth == 0: break
            i += 1
        if key in wanted: found[key] = json.loads(raw[begin:i])
        if raw[i] == '}': break
    assert set(found) == wanted
    return found


def corpus_public_rows(path):
    # Hashing reads every byte for custody; only whitelisted public-prefix
    # values are decoded. Falcon output IDs, score paths and text are skipped.
    raw = path.read_text(encoding='utf8')
    marker = '"records"'; at = raw.index(marker); at = raw.index('[', at) + 1
    rows = []
    while True:
        while raw[at].isspace() or raw[at] == ',': at += 1
        if raw[at] == ']': break
        assert raw[at] == '{'
        start = at; level = 0; string = False; escaped = False
        while at < len(raw):
            c = raw[at]
            if string:
                if escaped: escaped = False
                elif c == '\\': escaped = True
                elif c == '"': string = False
            elif c == '"': string = True
            elif c == '{': level += 1
            elif c == '}':
                level -= 1
                if level == 0:
                    at += 1; break
            at += 1
        rows.append(selected_fields(raw[start:at]))
    return rows


class Budget:
    def __init__(self, guard):
        self.guard = guard; self.calls = 0; self.metadata = 0
        self.tokenizer = 0; self.weights = 0; self.events = []
        self.journal = OUT/'http_events.jsonl'
        with self.journal.open('x', encoding='utf8'): pass

    def record(self, event):
        self.events.append(event)
        with self.journal.open('a', encoding='utf8', newline='\n') as stream:
            stream.write(json.dumps(event, allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())

    def add(self, kind, count):
        assert self.calls <= MAX_CALLS
        assert count >= 0
        setattr(self, kind, getattr(self, kind) + count)
        assert self.metadata <= MAX_METADATA and self.tokenizer <= MAX_TOKENIZER
        assert self.weights <= MAX_WEIGHTS
        assert self.metadata + self.tokenizer + self.weights <= MAX_NETWORK
        self.guard()

    def range(self, name, first, last, kind, total=None, etag=None):
        assert 0 <= first <= last and last-first+1 <= MAX_SPAN
        url = BASE + quote(name)
        for retry in range(2):
            self.guard(); assert self.calls < MAX_CALLS, 'HTTP call cap'
            self.calls += 1
            candidate = url if retry == 0 else url + f'?range_start={first}&range_end={last}&range_retry=1'
            req = Request(candidate, headers={'Range': f'bytes={first}-{last}', 'Accept-Encoding':'identity'})
            with urlopen(req, timeout=30) as response:
                value = response.headers.get('Content-Range', '')
                match = RANGE.fullmatch(value)
                header = dict(status=response.status, content_range=value,
                    content_length=response.headers.get('Content-Length'),
                    content_encoding=response.headers.get('Content-Encoding'),
                    etag=response.headers.get('ETag'), final_url=response.url)
                if response.status != 206 or match is None or (int(match[1]),int(match[2])) != (first,last):
                    self.record(dict(name=name, first=first, last=last, retry=retry, header=header,
                                     body_read=False, fault='range refused'))
                    if retry == 0: continue
                    raise RuntimeError('range refused before body: ' + name)
                remote_total = int(match[3]); length = last-first+1
                assert remote_total > last and (total is None or remote_total == total)
                assert header['content_encoding'] in (None, 'identity')
                assert header['content_length'] is None or int(header['content_length']) == length
                assert etag is None or header['etag'] is None or header['etag'] == etag
                self.add(kind, length)
                body = response.read(length+1)
                assert len(body) == length, name
                self.record(dict(name=name, first=first, last=last, retry=retry, header=header,
                                 bytes=len(body), sha256=hashlib.sha256(body).hexdigest()))
                return body, remote_total, header['etag']
        raise AssertionError('range retries exhausted')

    def small(self, name, limit=MAX_TOKENIZER):
        assert self.calls < MAX_CALLS
        self.calls += 1; url = BASE + quote(name)
        limit = min(limit, MAX_TOKENIZER-self.tokenizer,
                    MAX_NETWORK-self.metadata-self.tokenizer-self.weights)
        assert limit > 0, 'no tokenizer/network budget remaining'
        with urlopen(Request(url, headers={'Accept-Encoding':'identity'}), timeout=30) as response:
            assert response.status == 200
            declared = response.headers.get('Content-Length')
            assert declared is None or int(declared) <= limit
            assert response.headers.get('Content-Encoding') in (None,'identity')
            data = response.read(limit+1)
            assert len(data) <= limit
            self.add('tokenizer', len(data))
            self.record(dict(name=name, status=200, final_url=response.url,
                etag=response.headers.get('ETag'), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            return data


def header(name, budget, index):
    prefix, total, tag = budget.range(name, 0, 7, 'metadata')
    length = struct.unpack('<Q', prefix)[0]
    assert 0 < length <= (2 << 20) and length+8 <= total
    raw, again, tag2 = budget.range(name, 8, length+7, 'metadata', total, tag)
    assert again == total
    if tag and tag2: assert tag == tag2
    table = json.loads(raw)
    names = {k for k in table if k != '__metadata__'}
    assert names == {k for k,v in index['weight_map'].items() if v == name}
    ranges = []
    sizes = {'BF16':2,'F16':2,'F32':4,'I8':1,'U8':1}
    for key in names:
        item = table[key]; a,b = item['data_offsets']; shape=item['shape']
        assert item['dtype'] in sizes and all(type(x) is int and x > 0 for x in shape)
        assert 0 <= a < b and b-a == math.prod(shape)*sizes[item['dtype']]
        ranges.append((a,b))
    ranges.sort(); cursor=0
    for a,b in ranges: assert a == cursor; cursor=b
    assert cursor+length+8 == total
    return table, dict(name=name, bytes=total, header_bytes=length,
                       header_sha256=hashlib.sha256(raw).hexdigest(), etag=tag or tag2)


def row_spans(token_ids, row_bytes):
    # Only consecutive rows or gaps <=8 are eligible for bounded coalescing.
    spans=[]
    for token in token_ids:
        if spans and token-spans[-1][1]-1 <= 8 and (token-spans[-1][0]+1)*row_bytes <= MAX_SPAN:
            spans[-1][1]=token
        else:spans.append([token,token])
    if len(spans)>2000:
        raise RuntimeError(f'embedding row plan requires {len(spans)} ranges (>2000); stop before weights')
    gap_rows=sum(b-a+1 for a,b in spans)-len(token_ids)
    assert gap_rows*row_bytes <= MAX_GAPS, 'embedding gap budget exceeded before weight requests'
    return spans, gap_rows*row_bytes


def main(args, process, began):
    global CREATED_OUTPUT
    from transformers import AutoTokenizer
    import transformers, tokenizers, jinja2
    import psutil
    def guard():
        assert time.monotonic()-began <= MAX_SECONDS, 'deadline'
        assert process.memory_info().peak_wset <= MAX_RAM, 'OS RAM cap'
        assert not process.children(recursive=True), 'unexpected child'
        assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) <= MAX_OUTPUT
    assert not OUT.exists(), 'exclusive result exists'
    OUT.mkdir(parents=True); CREATED_OUTPUT = True
    process.cpu_affinity(process.cpu_affinity()[:6])
    write(OUT/'owned_process.json',dict(pid=process.pid,creation_time=process.create_time(),
        command=sys.orig_argv,freeze=args.freeze,limits=dict(seconds=MAX_SECONDS,OS_bytes=MAX_RAM,
        output_bytes=MAX_OUTPUT,network_bytes=MAX_NETWORK)))
    for path in (Path(__file__),PROTOCOL,CONFIG,INDEX,REFERENCE): freeze(path,args.freeze)
    runtime_files=[Path(transformers.__file__),Path(tokenizers.__file__),Path(jinja2.__file__),
                   Path(psutil.__file__)]
    runtime_files += list((SITE/'tokenizers').glob('*.pyd'))+list((SITE/'psutil').glob('*.pyd'))
    # Individual runtime packages are junctions to the existing repository venv.
    # Compare exact expected package entry points after resolving those aliases.
    for module in (transformers, tokenizers, jinja2, psutil):
        assert Path(module.__file__).resolve() == (SITE/module.__name__/'__init__.py').resolve()
    assert runtime_files and all(p.is_file() for p in runtime_files)
    initial = [desc(p) for p in (Path(__file__),PROTOCOL,CONFIG,INDEX,REFERENCE,CORPUS,SOURCE_PLAN,*runtime_files)]
    assert desc(CORPUS)['sha256'] == CORPUS_SHA and desc(CONFIG)['sha256'] == CONFIG_SHA
    assert desc(REFERENCE)['sha256'] == REFERENCE_SHA
    assert desc(SOURCE_PLAN)['sha256'] == SOURCE_PLAN_SHA
    config=json.loads(CONFIG.read_bytes()); index=json.loads(INDEX.read_bytes())
    assert config['hidden_size']==2880 and config['num_hidden_layers']==36
    assert config['num_attention_heads']==64 and config['num_key_value_heads']==8
    rows=corpus_public_rows(CORPUS)
    source_plan=json.loads(SOURCE_PLAN.read_bytes())
    source_by_id={r['id']:r for r in source_plan['cases']}
    assert len(source_by_id)==432 and set(source_by_id)=={r['id'] for r in rows}
    assert len(rows)==432 and Counter(x['split'] for x in rows)==Counter(FIT=360,DEV=72)
    assert len({x['id'] for x in rows})==len({x['group_key'] for x in rows})==432
    for row in rows:
        source=source_by_id[row['id']]
        assert all(row[key]==source[key] for key in ('id','split','messages','group_key','prefix_sha256'))
        assert source['group_scheme']=='normalized_all_user_turns_v2'
        assert source['first_user_key'] and source['conversation_sha256']
    domains=sorted({x['domain'] for x in rows},
        key=lambda value:hashlib.sha256((HOLDOUT_SALT+value).encode('utf8')).hexdigest())
    assert len(domains)==12
    dev_domains=set(domains[:4]);fit_domains=set(domains[4:])
    assert dev_domains=={'smol-magpie-ultra','metamathqa-50k','smol-constraints','self-oss-instruct'}
    cohorts={'FIT':[r for r in rows if r['domain'] in fit_domains and r['split']=='FIT'],
             'DEV':[r for r in rows if r['domain'] in dev_domains and r['split']=='DEV']}
    first_keys={split:{source_by_id[r['id']]['first_user_key'] for r in cohort}
                for split,cohort in cohorts.items()}
    group_keys={split:{r['group_key'] for r in cohort} for split,cohort in cohorts.items()}
    cross_first=first_keys['FIT'] & first_keys['DEV']
    cross_group=group_keys['FIT'] & group_keys['DEV']
    excluded={};chosen=[]
    for domain in domains:
        split='DEV' if domain in dev_domains else 'FIT'
        options=sorted((r for r in cohorts[split] if r['domain']==domain),key=lambda r:r['id'])
        seen_first=set();seen_group=set();kept=[];skipped=[]
        for row in options:
            first_key=source_by_id[row['id']]['first_user_key'];group_key=row['group_key']
            if first_key in cross_first or group_key in cross_group:
                skipped.append(dict(id=row['id'],reason='cross_cohort_key'));continue
            if first_key in seen_first or group_key in seen_group:
                skipped.append(dict(id=row['id'],reason='within_domain_duplicate'));continue
            seen_first.add(first_key);seen_group.add(group_key)
            if len(kept)<3:kept.append(row)
        assert len(kept)==3, f'eligible {domain} {split} has fewer than three distinct records'
        chosen.extend(kept)
        noncohort_ids=sorted(r['id'] for r in rows if r['domain']==domain and r['split']!=split)
        excluded[domain]=dict(split=split,original_cohort_count=len(options),
            noncohort_ids=noncohort_ids,noncohort_count=len(noncohort_ids),
            cross_cohort_exclusions=[x for x in skipped if x['reason']=='cross_cohort_key'],
            within_domain_duplicate_exclusions=[x for x in skipped if x['reason']=='within_domain_duplicate'],
            eligible_distinct_count=len(options)-len(skipped),selected_ids=[x['id'] for x in kept])
    assert len(chosen)==36 and Counter(r['split'] for r in chosen)==Counter(FIT=24,DEV=12)
    assert len({x['group_key'] for x in chosen})==36
    assert len({source_by_id[x['id']]['first_user_key'] for x in chosen})==36
    for key in ('group_key','first_user_key'):
        fit={source_by_id[r['id']][key] for r in chosen if r['split']=='FIT'}
        dev={source_by_id[r['id']][key] for r in chosen if r['split']=='DEV'}
        assert fit.isdisjoint(dev), f'cross-split {key} collision; stop before weights'
    eligibility=dict(schema='FIRST_ATTENTION_ELIGIBILITY_V1',salt=HOLDOUT_SALT,
        domain_order=domains,DEV_domains=domains[:4],FIT_domains=domains[4:],
        cohort_counts={key:len(value) for key,value in cohorts.items()},
        cross_cohort_first_user_keys=sorted(cross_first),cross_cohort_group_keys=sorted(cross_group),
        domain_eligibility=excluded,selected_ids=[row['id'] for row in chosen],
        selected_counts=dict(Counter(row['split'] for row in chosen)))
    write(OUT/'eligibility_plan.json',eligibility)
    budget=Budget(guard)
    tokdir=OUT/'tokenizer';tokdir.mkdir()
    tokenizer_names=('tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja')
    for name in tokenizer_names:
        data=budget.small(name)
        with (tokdir/name).open('xb') as stream:stream.write(data)
    shutil.copyfile(CONFIG,tokdir/'config.json')
    tokenizer=AutoTokenizer.from_pretrained(str(tokdir),local_files_only=True,trust_remote_code=False)
    assert len(tokenizer)==config['vocab_size']
    cases=[];used=set()
    fixed_clock=datetime(2026,10,15,tzinfo=timezone.utc).strftime
    for row in chosen:
        guard()
        canonical=tokenizer.apply_chat_template(row['messages'],tokenize=True,
            add_generation_prompt=True,reasoning_effort='medium',strftime_now=fixed_clock)
        assert isinstance(canonical,list) and canonical and all(type(v) is int and 0<=v<config['vocab_size'] for v in canonical)
        first=canonical[:256];used.update(first)
        cases.append(dict(id=row['id'],split=row['split'],domain=row['domain'],group_key=row['group_key'],
            prefix_sha256=row['prefix_sha256'],messages=row['messages'],upstream=row['upstream'],
            first_user_key=source_by_id[row['id']]['first_user_key'],
            group_scheme=source_by_id[row['id']]['group_scheme'],
            conversation_sha256=source_by_id[row['id']]['conversation_sha256'],
            input_ids=first,canonical_input_ids=canonical,original_length=len(canonical),canonical_ids_sha256=ids_sha(canonical),
            used_ids_sha256=ids_sha(first)))
    common_prefix_length=0
    for column in zip(*(case['input_ids'] for case in cases)):
        if len(set(column))!=1:break
        common_prefix_length+=1
    assert all(len(case['input_ids'])>common_prefix_length for case in cases), 'no variable query within 256'
    for case in cases:case['evaluation_query_start']=common_prefix_length
    token_ids=sorted(used); assert len(token_ids)<=36*256
    spans,gap_bytes=row_spans(token_ids,2880*2)
    plan=dict(schema='FIRST_ATTENTION_SELECTION_PLAN_V1',corpus=desc(CORPUS),source_model='openai/gpt-oss-120b',
        source_revision=REVISION,cases=cases,embedding_token_ids=token_ids,embedding_spans=spans,
        source_holdout=eligibility,eligibility_plan_sha256=desc(OUT/'eligibility_plan.json')['sha256'],
        common_prefix_length=common_prefix_length,
        evaluation_query_start=common_prefix_length,fixed_clock_date=FIXED_CLOCK_DATE,
        host_date_utc=datetime.now(timezone.utc).date().isoformat(),
        canonical_reasoning_effort='medium',
        embedding_unused_gap_bytes=gap_bytes,requested_tensors=['model.layers.0.input_layernorm.weight']+
        ['model.layers.0.self_attn.'+name for name in TARGETS],
        tokenizer_files=[desc(tokdir/name) for name in tokenizer_names])
    write(OUT/'selection_plan.json',plan)
    del tokenizer,rows,chosen
    guard()
    # No weight HTTP operation precedes the frozen selection plan.
    targets=plan['requested_tensors']+['model.embed_tokens.weight']
    shards=sorted({index['weight_map'][name] for name in targets})
    tables={};heads={}
    for shard in shards:tables[shard],heads[shard]=header(shard,budget,index)
    tensors={};tensor_dir=OUT/'tensors';tensor_dir.mkdir()
    for name in plan['requested_tensors']:
        shard=index['weight_map'][name];info=tables[shard][name]
        assert info['dtype']=='BF16' and info['shape']==EXPECTED_SHAPES[name]
        a,b=info['data_offsets'];absolute=8+heads[shard]['header_bytes']+a
        path=tensor_dir/(name.replace('.','_')+'.bf16')
        with path.open('xb') as stream:
            for offset in range(0,b-a,MAX_SPAN):
                first=absolute+offset;last=min(absolute+b-a,first+MAX_SPAN)-1
                body,_,etag=budget.range(shard,first,last,'weights',heads[shard]['bytes'],heads[shard]['etag'])
                stream.write(body)
        tensors[name]=dict(name=name,**desc(path),shape=info['shape'],dtype='BF16',
            source_shard=shard,source_offset=absolute,source_etag=etag)
    embedding_shard=index['weight_map']['model.embed_tokens.weight']
    embed=tables[embedding_shard]['model.embed_tokens.weight']
    assert embed['dtype']=='BF16' and embed['shape']==[config['vocab_size'],2880]
    row_size=2880*2;origin=8+heads[embedding_shard]['header_bytes']+embed['data_offsets'][0]
    extracted={}; selected=set(token_ids)
    for lo,hi in spans:
        first=origin+lo*row_size;last=origin+(hi+1)*row_size-1
        body,_,_=budget.range(embedding_shard,first,last,'weights',heads[embedding_shard]['bytes'],heads[embedding_shard]['etag'])
        for token in range(lo,hi+1):
            if token in selected:
                offset=(token-lo)*row_size;extracted[token]=body[offset:offset+row_size]
    assert sorted(extracted)==token_ids
    embed_path=OUT/'embedding_rows.bf16'
    with embed_path.open('xb') as stream:
        for token in token_ids:stream.write(extracted[token])
    assert desc(embed_path)['bytes']==len(token_ids)*row_size
    embedding_rows=dict(**desc(embed_path),shape=[len(token_ids),2880],dtype='BF16',token_ids=token_ids,
        row_lookup={str(token):i for i,token in enumerate(token_ids)},
        source_shard=embedding_shard,source_row_bytes=row_size)
    del extracted
    pins=initial+[desc(tokdir/name) for name in (*tokenizer_names,'config.json')]
    assert all(desc(p['path'])==p for p in pins)
    assert budget.weights <= MAX_WEIGHTS and budget.calls <= MAX_CALLS
    write(OUT/'manifest.json',dict(schema='FIRST_ATTENTION_SOURCE_V1',source_model='openai/gpt-oss-120b',
        source_revision=REVISION,config=desc(CONFIG),reference=desc(REFERENCE),index=desc(INDEX),
        tensors=tensors,embedding_rows=embedding_rows,cases=cases,input_pins=pins,
        tokenizer_files=[desc(tokdir/name) for name in (*tokenizer_names,'config.json')],
        tokenizer_runtime=dict(transformers=transformers.__version__,tokenizers=tokenizers.__version__,
                               jinja2=jinja2.__version__),
        selection_plan_sha256=desc(OUT/'selection_plan.json')['sha256'],
        eligibility_plan_sha256=desc(OUT/'eligibility_plan.json')['sha256'],
        common_prefix_length=common_prefix_length,evaluation_query_start=common_prefix_length,
        fixed_clock_date=FIXED_CLOCK_DATE,canonical_reasoning_effort='medium',
        host_date_utc=datetime.now(timezone.utc).date().isoformat(),
        source_headers=heads,network=dict(calls=budget.calls,metadata_bytes=budget.metadata,
            tokenizer_bytes=budget.tokenizer,weight_bytes=budget.weights,
            unused_embedding_gap_bytes=gap_bytes,events=budget.events),
        no_experts=True,no_full_head=True,no_full_checkpoint=True,no_model_inference=True,
        seconds=time.monotonic()-began,OS_peak=process.memory_info().peak_wset))
    assert all(desc(p['path'])==p for p in pins)
    assert all(desc(e['path'])['sha256']==e['sha256'] for e in list(tensors.values())+[embedding_rows])
    guard()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',required=True)
    parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args()
    OUT=args.output.resolve()
    import psutil
    proc=psutil.Process();began=time.monotonic()
    try:
        main(args,proc,began)
        result=desc(OUT/'manifest.json')
    except BaseException as error:
        if CREATED_OUTPUT and not (OUT/'terminal.json').exists():
            write(OUT/'terminal.json',dict(exit_code=1,error=repr(error),traceback=traceback.format_exc(),
                pid=proc.pid,creation_time=proc.create_time(),seconds=time.monotonic()-began))
        raise
    else:
        write(OUT/'terminal.json',dict(exit_code=0,error=None,result=result,pid=proc.pid,
            creation_time=proc.create_time(),seconds=time.monotonic()-began,
            OS_peak=proc.memory_info().peak_wset))
        print(json.dumps(dict(manifest=result)),flush=True)
