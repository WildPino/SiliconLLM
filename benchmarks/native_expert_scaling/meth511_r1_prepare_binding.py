"""Operational metadata repair: canonical sources, executable runtime files, empty cache."""
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import meth511_prepare_binding as P
import meth511_operations as O

ORIGINAL=O.DOC/'meth511_binding.failure.json'
assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()=='31614853663796b2895f9f58e2ee386b0709f468fe25bc6b75d540e3cfd3c93d'
assert json.loads(ORIGINAL.read_bytes())['scientific_calls']==0
O.BIND=O.DOC/'meth511_r1_binding.json'
CACHE=str(O.ROOT/'results/native_expert_scaling/meth511_runtime/empty_cache')
assert sys.pycache_prefix==CACHE
Path(CACHE).mkdir(exist_ok=True);assert not any(Path(CACHE).rglob('*'))
out=O.ROOT/'results/native_expert_scaling/meth511_r1_binding';out.mkdir(exist_ok=False)
progress=(out/'progress.jsonl').open('x',encoding='utf8')
setup=json.loads((O.DOC/'meth511_runtime_setup_result.json').read_bytes())
runtime=Path(sys.executable).parent.parent
def add(path,expected=None):
    # All walk roots are resolved ONCE. Their child paths are already canonical;
    # avoid repeatedly traversing deep junction chains for each file.
    p=Path(os.path.abspath(path));key=str(p).lower();P.guard()
    if key not in P.catalog:
        before=p.stat();h=hashlib.sha256()
        with p.open('rb') as f:
            while data:=f.read(8<<20):h.update(data);P.hashed+=len(data);P.guard()
        after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        P.catalog[key]={'path':str(p),'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'sha256':h.hexdigest()}
        if before.st_size>=64<<20:
            progress.write(json.dumps({'path':str(p),'bytes':before.st_size,'seconds':P.time.monotonic()-P.start,'files':len(P.catalog)})+'\n');progress.flush()
    if expected:assert P.catalog[key]['sha256']==expected,str(p)
    return P.catalog[key]
def executable_tree(root,metadata=False,stdlib=False):
    root=Path(root).resolve()
    for parent,dirs,files in os.walk(root):
        dirs[:]=[n for n in dirs if n not in ('__pycache__','include','share','test','tests','testing','idlelib','tkinter','ensurepip','site-packages')]
        for name in files:
            p=Path(parent)/name
            if metadata or p.suffix.lower() in ('.py','.pyd','.dll','.json','.txt','.jinja','.jinja2') or name in ('RECORD','METADATA','WHEEL','entry_points.txt','top_level.txt','LICENSE'):add(p)
        P.guard()
def tree(root,exclude=()):
    root=Path(root)
    if root==runtime:
        add(runtime/'pyvenv.cfg');add(Path(sys.executable).resolve())
        for row in setup['links']:
            p=Path(row['source']).resolve()
            if row['directory']:executable_tree(p,metadata=p.name.endswith('.dist-info'))
            else:add(Path(row['destination']).resolve())
        site=runtime/'Lib/site-packages'
        for p in site.iterdir():
            if p.name.startswith(('duckdb','_duckdb','adbc_driver_duckdb')):
                if p.is_dir():executable_tree(p,metadata=p.name.endswith('.dist-info'))
                else:add(p.resolve())
    else:executable_tree(root,stdlib=True)

P.add=add;P.tree=tree
oldwrite=O.write
def write(path,value):
    if Path(path)==O.BIND:
        value.update(runtime_empty_cache=CACHE,operational_repair={'first_fault':add(ORIGINAL),
            'original_scientific_calls':0,'runtime_scope':'All Python/native/data/metadata executable files in canonical whitelisted source roots; excludes tests/include/share/cache and unused activation scripts. Strict actual-import/image admission remains.',
            'bytecode_scope':'-I -B -X pycache_prefix=absolute_empty_cache; existing foreign bytecode is not read, and no cache is written.',
            'runtime_walk_algorithm':'Canonical roots once; no per-file deep junction resolution. Full SHA still computed freshly on every selected file and whole actual source/candidate/donor.'})
        deriv=json.loads((O.ROOT/'benchmarks/native_expert_scaling/meth511_r1_derivation.json').read_bytes())
        for row in deriv:
            assert add(row['old'],row['old_sha256']) and add(row['new'],row['new_sha256'])
            text=Path(row['old']).read_text(encoding='utf8')
            for change in row['replacements']:
                assert text.count(change['before'])==change['occurrences'];text=text.replace(change['before'],change['after'])
            assert text==Path(row['new']).read_text(encoding='utf8')
        value['literal_operational_derivation']=deriv
        add(O.DOC/'METH_511_R1_RUNTIME_INVENTORY_20261006.md')
        value['catalog']=list(P.catalog.values());value['price']['catalog_files']=len(P.catalog);value['price']['catalog_bytes']=sum(r['bytes'] for r in P.catalog.values())
    oldwrite(path,value)
O.write=write
try:P.main()
finally:progress.close()
