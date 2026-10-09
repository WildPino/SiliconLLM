"""Build a fixed broad donor-capture cohort from adopted metadata, before any loss."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_falcon_usability import sha,write


def main(a):
    result=json.loads(a.data_result.read_bytes());terminal=json.loads(a.data_result.with_suffix('.terminal.json').read_bytes())
    assert result['decision']=='BROAD_DATA_ADOPTION_PASS' and terminal['exit_code']==0
    assert terminal['result_sha256']==sha(a.data_result)
    corpus=Path(result['corpus']['path']);assert sha(corpus)==result['corpus']['sha256']
    data=json.loads(corpus.read_bytes());records=data['records'];cases=[];deferred={};selection=[]
    for source in sorted(data['source_counts']):
        chosen=[]
        for split in ('FIT','DEV'):
            rows=sorted([r for r in records if r['source']==source and r['split']==split and r['input_length']<=2048],
                        key=lambda r:(r['input_length'],r['id']))
            if len(rows)<2:
                assert source=='longalign' and not rows
                continue
            indices=[math.floor((len(rows)-1)*.25),math.ceil((len(rows)-1)*.75)]
            assert indices[0]!=indices[1]
            for index in indices:
                row=rows[index]
                cases.append(dict(id=row['id'],split=split,domain=source,messages=row['messages'],input_ids=row['input_ids'],
                         input_length=row['input_length'],group_key=row['group_key'],prefix_sha256=row['prefix_sha256'],
                         upstream=dict(repo=data['source_repo'],revision=data['source_revision'],parquet_path=row['parquet_path'],row_index=row['row_index']),
                         context_origin='Upstream public assistant turns, not donor-own-history.',
                         external_reference_assistant=row['external_reference_assistant']))
                chosen.append(row['id'])
        if chosen:selection.append(dict(source=source,selected=chosen))
        else:deferred[source]=dict(reason='All adopted input prefixes exceed this2304-total-token capture stage.',
                                 record_ids=[r['id'] for r in records if r['source']==source])
    counts=Counter(r['split'] for r in cases)
    assert len(cases)==48 and dict(counts)==dict(FIT=24,DEV=24)
    assert len({r['domain'] for r in cases})==12 and len({r['group_key'] for r in cases})==48
    assert all(r['input_length']+256<=2304 for r in cases)
    assert not any(r['split']=='RESERVED' for r in cases)
    spec=dict(schema='BROAD_CHAT_CAPTURE_CASES_V1',data_result_path=str(a.data_result.resolve()),data_result_sha256=sha(a.data_result),
       parent_corpus=dict(path=str(corpus),sha256=sha(corpus)),cases=cases,domains=sorted({r['domain'] for r in cases}),
       max_new_tokens=256,context_limit=2304,selection='25th/75th eligible length-rank per source/split,ID tie break; no model losses.',
       source_selection=selection,deferred_sources=deferred,other_adopted_inputs_unqueried=576,reserved_queries=0,
       max_BF16_logit_bytes=48*256*65537*2,
       scope='New broad donor-output supervision, not external-reference imitation or fresh final quality admission. Longalign is retained for a separate context stage.')
    write(a.out,spec)
    print(json.dumps(dict(path=str(a.out),sha256=sha(a.out),cases=48,FIT=24,DEV=24,
          min_input_ids=min(r['input_length'] for r in cases),max_input_ids=max(r['input_length'] for r in cases),
          max_BF16_logit_bytes=spec['max_BF16_logit_bytes'],deferred_sources=list(deferred))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-result',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
