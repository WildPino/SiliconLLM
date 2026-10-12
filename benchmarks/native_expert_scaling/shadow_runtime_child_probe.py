"""Observe startup descendants with the failed worker's imports; no model."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'benchmarks/native_expert_scaling'),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
import psutil
proc=psutil.Process();started=time.monotonic();observations=[]
def observe(stage):
    children=[]
    for child in proc.children(recursive=True):
        try:children.append(dict(pid=child.pid,ppid=child.ppid(),created=child.create_time(),name=child.name(),cmdline=child.cmdline()))
        except psutil.NoSuchProcess:pass
    row=dict(stage=stage,seconds=time.monotonic()-started,children=children)
    observations.append(row);print(json.dumps(row),flush=True)
observe('before_imports')
import numpy,torch,transformers
from original_engine_chat import ChatTokenizer
observe('after_worker_imports')
torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
proc.cpu_affinity(list(range(6)))
observe('after_worker_setup')
allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
observe('after_cuda_memory_calls')
for i in range(20):
    time.sleep(.1);observe(f'startup_poll{i}')
record=dict(schema='SHADOW_RUNTIME_CHILD_PROBE_V1',pid=proc.pid,created=proc.create_time(),
    command=sys.orig_argv,observations=observations,GPU_allocated=allocated,GPU_reserved=reserved,
    seconds=time.monotonic()-started,no_model_or_numerical_forward=True)
out=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925/shadow_runtime_child_probe_20261012.json'
with out.open('x',encoding='utf8') as f:json.dump(record,f,indent=2);f.write('\n')
