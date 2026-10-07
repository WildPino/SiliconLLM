"""Representation-only repair of Giga's lost over-cap report; no Qwen replay."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

base_path=Path(__file__).with_name('chatbot_operator_census.py')
spec=importlib.util.spec_from_file_location('frozen_census',base_path)
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def compact_write_once(path,obj):
    raw=(json.dumps(obj,separators=(',',':'),allow_nan=False)+'\n').encode('utf8')
    assert len(raw)<=2<<20, 'compact output cap'
    with path.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


base.write_once=compact_write_once
p=argparse.ArgumentParser()
p.add_argument('--binding',type=Path,required=True)
p.add_argument('--expected-binding-sha',required=True)
p.add_argument('--source-commit',required=True)
p.add_argument('--donor',choices=['gigachat'],required=True)
p.add_argument('--out',type=Path,required=True)
args=p.parse_args()
job=None
try:
    job=base.Job(args)
    job.r['repair']=dict(scope='JSON representation only; frozen operator body unchanged',
                         first_fault_file='chatbot_gigachat_operator_census_20261007.failure.json',
                         first_in_memory_counts_lost=True,
                         repeated_failed_metadata_computation=True,
                         completed_Qwen_or_model_or_C_replay=False)
    base.run(job)
except BaseException as error:
    if job is not None:
        job.timer.cancel()
        job.r.update(fault=repr(error),inputs=job.inputs,resource=job.resource())
        try:
            compact_write_once(args.out.with_suffix('.failure.json'),job.r)
        except AssertionError:
            compact_write_once(args.out.with_suffix('.failure.json'),dict(
                scope='COMPACT_SERIALIZER_FAULT_NO_STAGE_ADMISSION',fault=repr(error),
                process_instance=job.r['process_instance'],resource=job.resource(),
                discarded_in_memory_descriptors=True))
    raise
