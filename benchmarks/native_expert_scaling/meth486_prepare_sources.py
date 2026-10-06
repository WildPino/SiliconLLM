"""Deterministic source-only preparation; never initializes CUDA or executes model math."""
import hashlib
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'benchmarks/native_expert_scaling'
EXPECTED = 'aa01ce176ad54f1410a5d93eb628de8e887172225ae85a99f611168a9097fa48'
ENGINE = '574bab3e625d4876d976ee0f2a9ecb2cfe8568e59be2f0e023e7e348ea401487'

def modify(source, old, new, count=1):
    assert source.count(old) == count, old
    return source.replace(old, new)

def save(name, text):
    path = BASE / name
    assert not path.exists(), path
    path.write_bytes(text.encode())
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

original = (BASE / 'meth388_switch_three_workers.c').read_bytes()
assert hashlib.sha256(original).hexdigest() == EXPECTED
text = original.decode()
replacements = []
def edit(old, new):
    global text
    text = modify(text, old, new); replacements.append([old, new])

edit('static void mv(Matrix w,', '#include "meth486_shared_integer_backend.h"\nstatic void mv(Matrix w,')
edit('    if(w.q){int16_t codes[4096];float scale=head_activation_codes(x,codes,cols);',
     '    if(g486_eval(w,x,y,rows,cols,1)){if(cost_enabled&&cost_profile)cost_counts[cost_phase][cost_kind].matrix_seconds+=cost_clock()-cost_start;return;}\n    if(w.q){int16_t codes[4096];float scale=head_activation_codes(x,codes,cols);')
edit('    int16_t codes[4096];float scale=head_activation_codes(x,codes,cols);',
     '    if(g486_eval(w,x,y,rows,cols,1)){if(cost_enabled&&cost_profile)cost_counts[cost_phase][cost_kind].matrix_seconds+=cost_clock()-cost_start;return;}\n    int16_t codes[4096];float scale=head_activation_codes(x,codes,cols);')
edit('    int16_t *codes=alloc((size_t)tokens*cols,2);',
     '    if(g486_eval(w,x,y,rows,cols,tokens)){if(cost_enabled&&cost_profile)cost_counts[cost_phase][cost_kind].matrix_seconds+=cost_clock()-cost_start;return;}\n    int16_t *codes=alloc((size_t)tokens*cols,2);')
edit('m->dec[l]=block_weights(m,1,l);return m;', 'm->dec[l]=block_weights(m,1,l);g486_model_setup(m);return m;')
edit('static void unload(Model *m){', 'static void unload(Model *m){g486_teardown();')
edit('int meth388_contract_entry(', 'int meth486_contract_entry(')
reverse = text
for old, new in reversed(replacements):
    reverse = modify(reverse, new, old)
assert reverse.encode() == original
made = [save('meth486_shared_integer_model.c', text)]
for name, kind in [('meth388_switch_three_workers_cost_entry.c','cost'),
                   ('meth388_switch_three_workers_entry.c','generate')]:
    src = (BASE/name).read_text().replace('\r\n','\n')
    dst = src.replace('meth388_switch_three_workers.c','meth486_shared_integer_model.c').replace('meth388_switch_three_workers_cost_entry.c','meth486_shared_integer_cost_entry.c').replace('meth388_contract_entry','meth486_contract_entry').replace('meth388_forced_entry','meth486_forced_entry')
    dst = modify(dst,'memset(cost_counts,0,sizeof(cost_counts));','g486_reset();memset(cost_counts,0,sizeof(cost_counts));')
    dst = modify(dst,'worker_json();printf("}\\n");','worker_json();g486_json();printf("}\\n");')
    if kind == 'generate': dst = modify(dst,'int main(int argc,char **argv){','int meth486_generate_entry(int argc,char **argv){')
    made.append(save('meth486_shared_integer_'+kind+'_entry.c',dst))
engine = ROOT/'benchmarks/phase60/engine.c'; engine_bytes = engine.read_bytes()
assert hashlib.sha256(engine_bytes).hexdigest() == ENGINE
prefix = b'#ifdef SILICON_SWITCH_SHARED_INTEGER_GPU_R1\r\n#include "../native_expert_scaling/meth486_shared_integer_entry.c"\r\n#elif defined(SILICON_SWITCH_SHARED_INTEGER_GPU)'
oldprefix = b'#ifdef SILICON_SWITCH_SHARED_INTEGER_GPU'
assert engine_bytes.startswith(oldprefix)
newengine = prefix + engine_bytes[len(oldprefix):]
assert oldprefix + newengine[len(prefix):] == engine_bytes
engine.write_bytes(newengine)
record = {'experiment':'METH486 reversible source integration', 'source_sha256':EXPECTED,
          'engine_original_sha256':ENGINE,'replacements':replacements,'generated':made,
          'engine_sha256':hashlib.sha256(newengine).hexdigest(),'exact_reverse':True}
out = ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925/meth486_source_derivation.json'
with out.open('xb') as f: f.write((json.dumps(record,indent=2)+'\n').replace('\n','\r\n').encode())
print(json.dumps({'generated':len(made),'exact_reverse':True,'engine':record['engine_sha256']}))
