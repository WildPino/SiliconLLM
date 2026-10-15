"""Offline closure audit for the bounded first-attention source acquisition."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'docs/research/PIPELINE_REASSESSMENT_20261015/FIRST_ATTENTION_ACQUISITION_AUDIT_20261015.md'
DEFAULT_SOURCE = ROOT / 'results/native_expert_scaling/first_attention_source_repair3_20261015'
DEFAULT_OUTPUT = ROOT / 'results/native_expert_scaling/first_attention_source_audit_20261015'
REVISION = 'b5c939de8f754692c1647ca79fbf85e8c1e70f8a'
MAX_SPAN = 2 << 20
SHAPES = {
    'model.layers.0.input_layernorm.weight': [2880],
    'model.layers.0.self_attn.q_proj.weight': [4096, 2880],
    'model.layers.0.self_attn.k_proj.weight': [512, 2880],
    'model.layers.0.self_attn.v_proj.weight': [512, 2880],
    'model.layers.0.self_attn.o_proj.weight': [2880, 4096],
    'model.layers.0.self_attn.q_proj.bias': [4096],
    'model.layers.0.self_attn.k_proj.bias': [512],
    'model.layers.0.self_attn.v_proj.bias': [512],
    'model.layers.0.self_attn.o_proj.bias': [2880],
    'model.layers.0.self_attn.sinks': [64],
}
PREVIOUS = ('first_attention_source_20261015',
            'first_attention_source_repair1_20261015',
            'first_attention_source_repair2_20261015')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return dict(path=str(Path(path).resolve()), bytes=Path(path).stat().st_size, sha256=h.hexdigest())


def seal(item, base=None):
    path = Path(item['path']).resolve()
    if base is not None:
        assert path.is_relative_to(base.resolve()), f'path escaped result: {path}'
    actual = digest(path)
    assert all(actual[key] == item[key] for key in ('path', 'bytes', 'sha256')), path
    return actual


def freeze(path, revision):
    blob = subprocess.check_output(['git', 'show', revision + ':' + path.relative_to(ROOT).as_posix()], cwd=ROOT)
    assert blob.replace(b'\r\n', b'\n') == path.read_bytes().replace(b'\r\n', b'\n'), path


def ids_sha(ids):
    h = hashlib.sha256()
    for token in ids:
        assert type(token) is int and 0 <= token < 1 << 32
        h.update(struct.pack('<I', token))
    return h.hexdigest()


def public_corpus_rows(path):
    """Lexically pass over stored outputs; decode only public input/provenance fields."""
    raw=Path(path).read_text(encoding='utf8')
    decoder=json.JSONDecoder()
    wanted={'id','split','domain','group_key','prefix_sha256','messages','upstream'}
    def spaces(at):
        while at<len(raw) and raw[at].isspace():at+=1
        return at
    def end(at):
        start=at;stack=[];quoted=False;escaped=False
        while at<len(raw):
            char=raw[at]
            if quoted:
                if escaped:escaped=False
                elif char=='\\':escaped=True
                elif char=='"':quoted=False
            elif char=='"':quoted=True
            elif char in '[{':stack.append(char)
            elif char in ']}':
                if stack:
                    expected='[' if char==']' else '{'
                    assert stack.pop()==expected
                else:break
            elif char==',' and not stack:break
            at+=1
        assert at>start and not stack and not quoted
        return at
    marker=raw.index('"records"')
    at=spaces(raw.index('[',marker)+1);result=[]
    while raw[at]!=']':
        assert raw[at]=='{';at+=1;record={}
        while True:
            at=spaces(at)
            if raw[at]=='}':at+=1;break
            key,at=decoder.raw_decode(raw,at)
            at=spaces(at);assert raw[at]==':';at=spaces(at+1)
            stop=end(at)
            if key in wanted:record[key]=json.loads(raw[at:stop])
            at=spaces(stop)
            if raw[at]==',':at+=1;continue
            assert raw[at]=='}';at+=1;break
        assert set(record)==wanted
        result.append(record)
        at=spaces(at)
        if raw[at]==',':at=spaces(at+1)
    return result


def prior_attempts(base):
    found = []
    for name in PREVIOUS:
        path = base/name/'terminal.json'
        assert path.is_file(), f'missing prior receipt {path}'
        item = json.loads(path.read_text(encoding='utf8'))
        assert item['exit_code'] != 0, f'prior attempt unexpectedly succeeded: {name}'
        found.append(dict(name=name, terminal=digest(path), exit_code=item['exit_code'],
                          error=item.get('error'), trace_tail=item.get('traceback', '').splitlines()[-4:]))
    return found


def verify_cases(manifest, selection, eligibility):
    cases = manifest['cases']
    assert cases == selection['cases'] and len(cases) == 36
    assert Counter(c['split'] for c in cases) == Counter(FIT=24, DEV=12)
    assert eligibility['selected_ids'] == [c['id'] for c in cases]
    assert eligibility['selected_counts'] == {'FIT': 24, 'DEV': 12}
    assert selection['source_holdout'] == eligibility
    assert len({c['id'] for c in cases}) == 36
    assert len({c['group_key'] for c in cases}) == 36
    assert len({c['first_user_key'] for c in cases}) == 36
    fit_domains = {c['domain'] for c in cases if c['split'] == 'FIT'}
    dev_domains = {c['domain'] for c in cases if c['split'] == 'DEV'}
    assert fit_domains.isdisjoint(dev_domains) and len(fit_domains) == 8 and len(dev_domains) == 4
    assert fit_domains == set(eligibility['FIT_domains'])
    assert dev_domains == set(eligibility['DEV_domains'])
    assert all(sum(c['domain'] == domain for c in cases) == 3 for domain in fit_domains | dev_domains)
    assert [c['id'] for c in cases] == [case_id for domain in eligibility['domain_order']
                                       for case_id in eligibility['domain_eligibility'][domain]['selected_ids']]
    for domain in eligibility['domain_order']:
        detail = eligibility['domain_eligibility'][domain]
        assert detail['noncohort_count'] == len(detail['noncohort_ids'])
        assert detail['eligible_distinct_count'] >= 3
        assert detail['original_cohort_count'] >= detail['eligible_distinct_count']
    assert manifest['fixed_clock_date'] == selection['fixed_clock_date'] == '2026-10-15'
    assert manifest['canonical_reasoning_effort'] == selection['canonical_reasoning_effort'] == 'medium'
    common = manifest['common_prefix_length']
    assert common == selection['common_prefix_length'] == 62
    assert manifest['evaluation_query_start'] == selection['evaluation_query_start'] == common
    for case in cases:
        full = case['canonical_input_ids']; used = case['input_ids']
        assert case['original_length'] == len(full) and used == full[:256]
        assert 0 < len(used) <= 256 and len(used) > common
        assert case['evaluation_query_start'] == common
        assert ids_sha(full) == case['canonical_ids_sha256']
        assert ids_sha(used) == case['used_ids_sha256']
        assert case['messages'] and case['prefix_sha256'] and case['group_scheme'] == 'normalized_all_user_turns_v2'
    actual_common = 0
    for column in zip(*(c['input_ids'] for c in cases)):
        if len(set(column)) != 1: break
        actual_common += 1
    assert actual_common == common
    token_ids = sorted({token for case in cases for token in case['input_ids']})
    assert token_ids == selection['embedding_token_ids']
    assert len(token_ids) == 1501
    return token_ids


def verify_source_selection(manifest, selection, eligibility):
    corpus_pin = selection['corpus']; seal(corpus_pin)
    corpus = public_corpus_rows(corpus_pin['path'])
    source_plan_path = ROOT/'results/native_expert_scaling/chatbot_source_coverage_plan_20261015/cases.json'
    source_plan_pin = next(p for p in manifest['input_pins'] if Path(p['path']).resolve()==source_plan_path.resolve())
    seal(source_plan_pin)
    source_cases = {r['id']:r for r in json.loads(source_plan_path.read_text(encoding='utf8'))['cases']}
    assert len(corpus)==len(source_cases)==432
    by_id={r['id']:r for r in corpus};assert len(by_id)==432 and set(by_id)==set(source_cases)
    domains=sorted({r['domain'] for r in corpus},
        key=lambda d:hashlib.sha256(('first-attention-source-holdout-v1:'+d).encode('utf8')).hexdigest())
    assert domains==eligibility['domain_order'] and len(domains)==12
    cohorts={'FIT':[r for r in corpus if r['domain'] in domains[4:] and r['split']=='FIT'],
             'DEV':[r for r in corpus if r['domain'] in domains[:4] and r['split']=='DEV']}
    cross_first={source_cases[r['id']]['first_user_key'] for r in cohorts['FIT']} & \
                {source_cases[r['id']]['first_user_key'] for r in cohorts['DEV']}
    cross_group={r['group_key'] for r in cohorts['FIT']} & {r['group_key'] for r in cohorts['DEV']}
    assert eligibility['cohort_counts']=={k:len(v) for k,v in cohorts.items()}
    assert eligibility['cross_cohort_first_user_keys']==sorted(cross_first)
    assert eligibility['cross_cohort_group_keys']==sorted(cross_group)
    chosen=[]
    for domain in domains:
        split='DEV' if domain in domains[:4] else 'FIT'
        options=sorted((r for r in cohorts[split] if r['domain']==domain),key=lambda r:r['id'])
        seen_first=set();seen_group=set();selected=[];excluded_first=[];excluded_duplicate=[]
        for row in options:
            first=source_cases[row['id']]['first_user_key'];group=row['group_key']
            if first in cross_first or group in cross_group:
                excluded_first.append(dict(id=row['id'],reason='cross_cohort_key'));continue
            if first in seen_first or group in seen_group:
                excluded_duplicate.append(dict(id=row['id'],reason='within_domain_duplicate'));continue
            seen_first.add(first);seen_group.add(group)
            if len(selected)<3:selected.append(row)
        detail=eligibility['domain_eligibility'][domain]
        assert detail['original_cohort_count']==len(options)
        assert detail['cross_cohort_exclusions']==excluded_first
        assert detail['within_domain_duplicate_exclusions']==excluded_duplicate
        assert detail['eligible_distinct_count']==len(options)-len(excluded_first)-len(excluded_duplicate)
        assert detail['selected_ids']==[r['id'] for r in selected]
        assert detail['noncohort_ids']==sorted(r['id'] for r in corpus if r['domain']==domain and r['split']!=split)
        chosen.extend(selected)
    assert [r['id'] for r in chosen]==eligibility['selected_ids']
    for case in manifest['cases']:
        original=by_id[case['id']];source=source_cases[case['id']]
        for key in ('id','split','domain','group_key','prefix_sha256','messages','upstream'):
            assert case[key]==original[key]
        for key in ('first_user_key','group_scheme','conversation_sha256'):
            assert case[key]==source[key]


def verify_payload(source, manifest, selection, token_ids):
    tensors = manifest['tensors']; assert set(tensors) == set(SHAPES)
    for name, shape in SHAPES.items():
        item = tensors[name]
        assert item['name'] == name and item['shape'] == shape and item['dtype'] == 'BF16'
        assert item['bytes'] == 2*math.prod(shape)
        seal(item, source)
    rows = manifest['embedding_rows']
    assert rows['shape'] == [1501, 2880] and rows['dtype'] == 'BF16'
    assert rows['bytes'] == 1501*2880*2 and rows['source_row_bytes'] == 2880*2
    assert rows['token_ids'] == token_ids
    assert rows['row_lookup'] == {str(token): i for i, token in enumerate(token_ids)}
    seal(rows, source)
    spans = selection['embedding_spans']
    assert all(0 <= lo <= hi and (hi-lo+1)*5760 <= MAX_SPAN for lo,hi in spans)
    assert sorted(token for lo,hi in spans for token in token_ids if lo <= token <= hi) == token_ids
    gap = (sum(hi-lo+1 for lo,hi in spans)-len(token_ids))*5760
    assert gap == selection['embedding_unused_gap_bytes'] == manifest['network']['unused_embedding_gap_bytes']
    assert gap <= 100 << 20 and len(spans) <= 2000


def verify_http(source, manifest, selection):
    events = [json.loads(line) for line in (source/'http_events.jsonl').read_text(encoding='utf8').splitlines()]
    assert events == manifest['network']['events']
    headers = manifest['source_headers']; assert len(headers) == 2
    success = {}; used = set(); calls = 0; metadata_bytes = weight_bytes = tokenizer_bytes = cached_bytes = 0
    tokenizer_files={Path(item['path']).name:item for item in manifest['tokenizer_files']}
    for index, event in enumerate(events):
        if event.get('kind') == 'verified_local_tokenizer_cache':
            assert event['network_body_bytes'] == 0
            seal(event['source'])
            assert event['source']['sha256']==tokenizer_files[event['name']]['sha256']
            cached_bytes+=event['source']['bytes']
            continue
        calls += 1
        if 'first' not in event:
            assert event['status'] == 200 and event['name'] in ('tokenizer.json','tokenizer_config.json',
                'special_tokens_map.json','chat_template.jinja')
            assert event['bytes'] >= 0 and re.fullmatch('[0-9a-f]{64}',event['sha256'])
            assert event['sha256']==tokenizer_files[event['name']]['sha256']
            tokenizer_bytes += event['bytes']; continue
        if event.get('body_read') is False:
            assert event['retry'] in (0,1)
            assert event.get('fault') in ('HTTP refusal','range refused')
            continue
        head = event['header']; first = event['first'];last = event['last']
        assert head['status'] == 206 and 0 <= first <= last and last-first+1 <= MAX_SPAN
        assert event['bytes'] == last-first+1
        assert head['content_range'] == f'bytes {first}-{last}/{headers[event["name"]]["bytes"]}'
        assert head['content_encoding'] in (None,'identity')
        assert head['content_length'] is None or int(head['content_length']) == event['bytes']
        assert re.fullmatch('[0-9a-f]{64}',event['sha256'])
        expected_etag = headers[event['name']]['etag']
        assert not expected_etag or not head['etag'] or head['etag'] == expected_etag
        key = (event['name'],first,last)
        assert key not in success
        success[key] = (index,event)
    assert calls == manifest['network']['calls'] <= 2048
    for shard, head in headers.items():
        assert head['name'] == shard and head['header_bytes'] <= 2 << 20
        for first,last in ((0,7),(8,7+head['header_bytes'])):
            key=(shard,first,last);assert key in success
            used.add(key);metadata_bytes+=success[key][1]['bytes']
        assert head['header_sha256'] == success[(shard,8,7+head['header_bytes'])][1]['sha256']
    for name, item in manifest['tensors'].items():
        shard=item['source_shard']; assert shard in headers
        assert not item['source_etag'] or not headers[shard]['etag'] or item['source_etag'] == headers[shard]['etag']
        with Path(item['path']).open('rb') as stream:
            for offset in range(0,item['bytes'],MAX_SPAN):
                length=min(MAX_SPAN,item['bytes']-offset)
                key=(shard,item['source_offset']+offset,item['source_offset']+offset+length-1)
                assert key in success and key not in used
                assert hashlib.sha256(stream.read(length)).hexdigest() == success[key][1]['sha256']
                used.add(key);weight_bytes+=length
            assert not stream.read(1)
    embed_shard=manifest['embedding_rows']['source_shard']
    embed_events=sorted((key for key in success if key[0]==embed_shard and key not in used),key=lambda key:key[1])
    spans=selection['embedding_spans'];assert len(embed_events)==len(spans)
    origin=embed_events[0][1]-spans[0][0]*5760
    assert origin >= 8+headers[embed_shard]['header_bytes']
    embedding_rows=manifest['embedding_rows'];token_ids=embedding_rows['token_ids']
    row_index={token:i for i,token in enumerate(token_ids)}
    replayed_spans=replayed_bytes=unreplayable_spans=unreplayable_bytes=0
    with Path(embedding_rows['path']).open('rb') as rowfile:
        for key,(lo,hi) in zip(embed_events,spans):
            assert key == (embed_shard,origin+lo*5760,origin+(hi+1)*5760-1)
            retained=[token for token in token_ids if lo<=token<=hi]
            assert retained and retained[0]==lo and retained[-1]==hi
            if len(retained)==hi-lo+1:
                rowfile.seek(row_index[lo]*5760)
                body=rowfile.read((hi-lo+1)*5760)
                assert hashlib.sha256(body).hexdigest()==success[key][1]['sha256']
                replayed_spans+=1;replayed_bytes+=len(body)
            else:
                unreplayable_spans+=1;unreplayable_bytes+=success[key][1]['bytes']
            used.add(key);weight_bytes+=success[key][1]['bytes']
    assert used == set(success)
    network=manifest['network']
    assert metadata_bytes == network['metadata_bytes'] <= 16 << 20
    assert tokenizer_bytes == network['tokenizer_bytes'] <= 64 << 20
    assert cached_bytes == network['cached_tokenizer_bytes'] <= 64 << 20
    assert weight_bytes == network['weight_bytes'] <= 256 << 20
    assert metadata_bytes+tokenizer_bytes+weight_bytes <= 384 << 20
    return dict(calls=calls,range_successes=len(success),metadata_bytes=metadata_bytes,
                tokenizer_bytes=tokenizer_bytes,weight_bytes=weight_bytes,
                embedding_gap_bytes=network['unused_embedding_gap_bytes'],
                embedding_spans_replayed=replayed_spans,embedding_response_bytes_replayed=replayed_bytes,
                embedding_spans_with_unretained_gaps=unreplayable_spans,
                embedding_response_bytes_with_unretained_gaps=unreplayable_bytes,
                raw_shard_header_replayable=False)


def audit(source):
    assert source.is_dir()
    terminal=json.loads((source/'terminal.json').read_text(encoding='utf8'))
    assert terminal['exit_code']==0 and terminal['error'] is None, 'source terminal incomplete or failed'
    owned=json.loads((source/'owned_process.json').read_text(encoding='utf8'))
    assert (owned['pid'],owned['creation_time'])==(terminal['pid'],terminal['creation_time'])
    assert owned['freeze'] and terminal['result']['path']==str((source/'manifest.json').resolve())
    assert owned['limits']==dict(seconds=900,OS_bytes=2<<30,output_bytes=512<<20,network_bytes=384<<20)
    seal(terminal['result'],source)
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf8'))
    assert manifest['schema']=='FIRST_ATTENTION_SOURCE_V1'
    assert manifest['source_model']=='openai/gpt-oss-120b' and manifest['source_revision']==REVISION
    assert manifest['seconds']<=900 and manifest['OS_peak']<=2<<30
    assert manifest['no_experts'] and manifest['no_full_head'] and manifest['no_full_checkpoint'] and manifest['no_model_inference']
    selection_path=source/'selection_plan.json';eligibility_path=source/'eligibility_plan.json'
    assert digest(selection_path)['sha256']==manifest['selection_plan_sha256']
    assert digest(eligibility_path)['sha256']==manifest['eligibility_plan_sha256']
    selection=json.loads(selection_path.read_text(encoding='utf8'))
    eligibility=json.loads(eligibility_path.read_text(encoding='utf8'))
    assert selection['schema']=='FIRST_ATTENTION_SELECTION_PLAN_V1'
    assert eligibility['schema']=='FIRST_ATTENTION_ELIGIBILITY_V1'
    assert selection['eligibility_plan_sha256']==manifest['eligibility_plan_sha256']
    assert selection['source_revision']==REVISION
    token_ids=verify_cases(manifest,selection,eligibility)
    verify_source_selection(manifest,selection,eligibility)
    for item in manifest['input_pins']:seal(item)
    for key in ('config','reference','index'):seal(manifest[key])
    for item in manifest['tokenizer_files']:seal(item,source)
    verify_payload(source,manifest,selection,token_ids)
    network=verify_http(source,manifest,selection)
    expected={source/name for name in ('terminal.json','owned_process.json','manifest.json',
        'selection_plan.json','eligibility_plan.json','http_events.jsonl')}
    expected.update(Path(item['path']) for item in manifest['tensors'].values())
    expected.add(Path(manifest['embedding_rows']['path']))
    expected.update(Path(item['path']) for item in manifest['tokenizer_files'])
    assert {p for p in source.rglob('*') if p.is_file()}==expected, 'unexpected/missing source file'
    assert sum(p.stat().st_size for p in expected) <= 512 << 20
    for item in manifest['input_pins']:seal(item)
    for item in list(manifest['tensors'].values())+[manifest['embedding_rows']]+manifest['tokenizer_files']:seal(item,source)
    return dict(source_manifest=digest(source/'manifest.json'),selection_plan=digest(selection_path),
        eligibility_plan=digest(eligibility_path),terminal=digest(source/'terminal.json'),
        prior_attempts=prior_attempts(source.parent),cases=36,FIT=24,DEV=12,
        used_embedding_rows=len(token_ids),common_prefix_length=manifest['common_prefix_length'],
        tensors=len(manifest['tensors']),network=network,
        limits_of_offline_replay=['Raw safetensors header bytes are not retained; its digest and keyset check are producer witnesses.',
            'Embedding responses without unused rows are independently hashed against exported rows; spans with unused gap bytes cannot be fully replayed.',
            'The audit verifies saved bytes and receipts, not remote authenticity or inference quality.'])


def write(path,obj):
    with path.open('x',encoding='utf8',newline='\n') as stream:
        json.dump(obj,stream,indent=2,allow_nan=False);stream.write('\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',required=True)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    source=args.source.resolve();out=args.output.resolve();began=time.monotonic()
    assert not out.exists(), 'exclusive audit output exists'
    out.mkdir(parents=True)
    try:
        freeze(Path(__file__).resolve(),args.freeze);freeze(PROTOCOL,args.freeze)
        result=audit(source)
        write(out/'audit.json',dict(schema='FIRST_ATTENTION_ACQUISITION_AUDIT_V1',status='PASS',
                                    source=str(source),freeze=args.freeze,**result))
        write(out/'terminal.json',dict(exit_code=0,result=digest(out/'audit.json'),
                                       seconds=time.monotonic()-began))
    except BaseException as error:
        write(out/'terminal.json',dict(exit_code=1,error=repr(error),traceback=traceback.format_exc(),
                                       seconds=time.monotonic()-began))
        raise
