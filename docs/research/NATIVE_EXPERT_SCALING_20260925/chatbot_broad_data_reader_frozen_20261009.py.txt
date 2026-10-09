"""Isolated single-thread Arrow reader and deterministic dialogue selection."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unicodedata

ROOT=Path(__file__).resolve().parents[2]


def key(text):
    value=' '.join(unicodedata.normalize('NFKC',text).casefold().split())
    return hashlib.sha256(value.encode()).hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def main(a):
    start=time.monotonic();config=json.loads(a.config.read_bytes())
    sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
    sys.path.append(str(ROOT/'.venv/Lib/site-packages'))
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import psutil
    process=psutil.Process();process.cpu_affinity([0]);phase='before_Arrow';counts=Counter();sources={}
    progress=a.directory/'reader.progress.jsonl'
    def guard():
        assert time.monotonic()-start<240,'reader240s'
        assert process.memory_info().peak_wset<4<<30,'reader4GiB'
    def event(**data):
        with progress.open('a',encoding='utf8') as f:
            f.write(json.dumps(dict(phase=phase,seconds=time.monotonic()-start,**data))+'\n');f.flush()
        guard()
    try:
        event()
        import pyarrow as pa
        import pyarrow.parquet as pq
        assert pa.__version__=='24.0.0'
        pa.set_cpu_count(1);pa.set_io_thread_count(1)
        assert not any(v in sys.modules for v in ('torch','transformers','tokenizers'))
        phase='metadata'
        paths={s:Path(config[s]['path']) for s in ('test','train')}
        metadata={s:pq.ParquetFile(p).metadata.num_rows for s,p in paths.items()}
        candidates={s:{} for s in paths};thresholds={s:{} for s in paths}
        test_groups=set();protected=set(config['protected_prompt_keys'])
        def keep(split,source,record):
            group=record['group_key'];pool=candidates[split].setdefault(source,{})
            old=pool.get(group)
            if old and old['conversation_sha256']<=record['conversation_sha256']:return
            pool[group]=record
            if len(pool)>160:
                worst=max(pool,key=lambda k:pool[k]['selection_rank'])
                del pool[worst]
            thresholds[split][source]=max(r['selection_rank'] for r in pool.values())
        for split in ('test','train'):
            phase='read_'+split;parquet=pq.ParquetFile(paths[split]);row_index=0;by_source=Counter()
            assert parquet.schema_arrow.names==['messages','source'],parquet.schema_arrow.names
            for batch in parquet.iter_batches(batch_size=256,columns=['messages','source'],use_threads=False):
                for row in batch.to_pylist():
                    at=row_index;row_index+=1;counts[split+'_rows']+=1
                    messages=row['messages'];source=row['source'];by_source[source]+=1
                    if not isinstance(source,str) or not isinstance(messages,list) or not messages:
                        counts[split+'_invalid_schema']+=1;continue
                    basic_users=[m for m in messages if isinstance(m,dict) and m.get('role')=='user'
                                 and isinstance(m.get('content'),str)]
                    if split=='test' and basic_users:test_groups.add(key(basic_users[0]['content']))
                    if any(not isinstance(m,dict) or m.get('role') not in ('system','user','assistant')
                           or not isinstance(m.get('content'),str) for m in messages):
                        counts[split+'_unsupported_role_or_content']+=1;continue
                    if any(m['role']=='system' for m in messages[1:]):
                        counts[split+'_noninitial_system']+=1;continue
                    users=[i for i,m in enumerate(messages) if m['role']=='user']
                    if not users or not messages[users[0]]['content'].strip():
                        counts[split+'_no_nonempty_user']+=1;continue
                    first=users[0];last=users[-1];group=key(messages[first]['content'])
                    if split=='test':test_groups.add(group)
                    elif group in test_groups:
                        counts['train_group_seen_in_full_test']+=1;continue
                    if any(key(messages[i]['content']) in protected for i in users):
                        counts[split+'_protected_old_query']+=1;continue
                    counts[split+'_structurally_eligible']+=1
                    rank=hashlib.sha256(('20261009-broad-v1\0'+source+'\0'+group).encode()).hexdigest()
                    pool=candidates[split].get(source,{})
                    if len(pool)==160 and group not in pool and rank>thresholds[split][source]:continue
                    raw=json.dumps(messages,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
                    conversation=hashlib.sha256(raw).hexdigest()
                    prefix=[dict(role=m['role'],content=m['content']) for m in messages[:last+1]]
                    canonical=json.dumps(prefix,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
                    reference=messages[last+1]['content'] if last+1<len(messages) and messages[last+1]['role']=='assistant' else None
                    record=dict(source=source,upstream_split=split,parquet_path=str(paths[split]),row_index=at,
                       group_key=group,selection_rank=rank,conversation_sha256=conversation,
                       prefix_sha256=hashlib.sha256(canonical).hexdigest(),messages=prefix,
                       external_reference_assistant=reference,user_turns=len(users),prior_assistant_turns=sum(m['role']=='assistant' for m in prefix))
                    keep(split,source,record)
                event(rows=row_index,expected=metadata[split])
            assert row_index==metadata[split]
            sources[split]=dict(by_source)
        assert len(sources['test'])==len(sources['train'])==13 and set(sources['test'])==set(sources['train'])
        phase='select';selected=[];used=set()
        for split,count in (('test',8),('train',40)):
            for source in sorted(sources[split]):
                accepted=[]
                for row in sorted(candidates[split][source].values(),key=lambda r:r['selection_rank']):
                    if row['group_key'] in used:continue
                    used.add(row['group_key']);accepted.append(row)
                    if len(accepted)==count:break
                assert len(accepted)==count,('quota',split,source,len(accepted))
                for i,row in enumerate(accepted):
                    row['split']='RESERVED' if split=='test' else ('FIT' if i<32 else 'DEV')
                    row['id']='broad_'+row['split'].lower()+'_'+source.replace('-','_')+'_'+str(i).zfill(3)
                    selected.append(row)
        assert len(selected)==len(used)==624
        assert len({r['prefix_sha256'] for r in selected})==624
        assert all(r['group_key'] not in test_groups for r in selected if r['split']!='RESERVED')
        write(a.directory/'selected_raw.json',dict(schema='BROAD_CHAT_SELECTION_V1',records=selected,
             upstream_rows=metadata,upstream_source_rows=sources,counters=dict(counts),test_group_count=len(test_groups),
             protected_group_count=len(protected),pyarrow_version=pa.__version__,peak_OS_snapshot=process.memory_info().peak_wset,
             seconds=time.monotonic()-start,selected=624,independent_Parquet_decoder=False))
        phase='complete';event(selected=624)
    except BaseException as error:
        write(a.directory/'reader.first_failure.json',dict(phase=phase,fault=repr(error),counters=dict(counts),seconds=time.monotonic()-start))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--directory',type=Path,required=True)
    main(p.parse_args())
