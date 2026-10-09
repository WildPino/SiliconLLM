"""Bind a qualified SSD-storage repair, reusing two durable original replies."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from chatbot_falcon_usability import sha,write
from chatbot_broad_capture import worker


def bind(a):
    original_binding=DOC/'chatbot_broad_capture_binding_20261009.json'
    original=json.loads(original_binding.read_bytes())
    old=ROOT/'results/native_expert_scaling/chatbot_broad_capture_20261009'
    failure=json.loads((old/'first_failure.json').read_bytes())
    assert failure['fault']=="AssertionError('GPU allocated cap')" and len(failure['durable_completed'])==2
    qualification=DOC/'chatbot_falcon_ssd_tiles_result_repair1_20261009.json'
    qualified=json.loads(qualification.read_bytes())
    qt=qualification.with_suffix('.terminal.json');receipt=json.loads(qt.read_bytes())
    assert receipt['exit_code']==0 and receipt['result_sha256']==sha(qualification)
    assert qualified['decision']=='TILED_SOURCE_TRAJECTORIES_PASS' and all(qualified['gates'].values())
    assert qualified['labels']==151 and qualified['source_generations']==0
    qb=DOC/'chatbot_falcon_ssd_tiles_binding_repair1_20261009.json'
    q_inputs={v['path']:v['sha256'] for v in json.loads(qb.read_bytes())['inputs']}
    assert q_inputs[str((B/'chatbot_falcon_ssd_tiles.py').resolve())]==sha(B/'chatbot_falcon_ssd_tiles.py')
    source=Path(original['source']);package=json.loads((source/'source_package.json').read_bytes())
    files=[Path(v['path']) for v in original['inputs']]
    files+=[Path(__file__),B/'chatbot_falcon_ssd_tiles.py',original_binding,old/'first_failure.json',
            qualification,qt,qb,DOC/'CHATBOT_BROAD_CAPTURE_REPAIR1_PROTOCOL_20261009.md',
            DOC/'CHATBOT_FALCON_SSD_TILES_RESULT_20261009.md',
            DOC/'chatbot_broad_capture_worker_frozen_20261009.py.txt',
            DOC/'chatbot_broad_capture_result_20261009.launcher_failure.json',
            DOC/'chatbot_broad_capture_result_20261009.worker.log']
    for identifier in failure['durable_completed']:
        path=old/(identifier+'.json');record=json.loads(path.read_bytes())
        assert sha(record['logits']['path'])==record['logits']['sha256']
        assert record['finite'] and record['BF16_lossless'] and record['raw_argmax_equals_generated']
        files += [path,Path(record['logits']['path'])]
    files=list(dict.fromkeys(p.resolve() for p in files))
    inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    by_path={v['path']:v['sha256'] for v in inputs}
    assert all(by_path[str(Path(v['path']).resolve())]==v['sha256'] for v in package['files'])
    write(a.out,dict(schema='BROAD_CHAT_CAPTURE_REPAIR_BINDING_V1',python=str(Path(sys.executable).resolve()),
        worker_path=str(Path(__file__).resolve()),source=original['source'],source_revision=original['source_revision'],
        source_named_elements=original['source_named_elements'],cases=original['cases'],data_result=original['data_result'],
        resume=str(old),qualification=str(qualification),reused_ids=failure['durable_completed'],
        limits=original['limits'],runtime_binding_scope='Original package/cohort/two original packets/qualified source storage variant/current code/runtime/foreign hashes; not full native DLL tree. Old failed family caps remain false.',
        inputs=inputs))
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(inputs))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('bind','worker'),default='worker')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze')
    p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();bind(a) if a.mode=='bind' else worker(a)
