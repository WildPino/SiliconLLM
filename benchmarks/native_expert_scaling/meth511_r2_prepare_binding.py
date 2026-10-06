"""Operational metadata repair: canonical sources, executable runtime files, empty cache."""
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import meth511_prepare_binding as P
import meth511_operations as O

ORIGINAL=O.DOC/'meth511_r2_binding.failure.json'
assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()=='7957801d59ecdfab534034a39b2afa163465355598f714602fdc81c28180f476'
assert json.loads(ORIGINAL.read_bytes())['scientific_calls']==0
O.BIND=O.DOC/'meth511_r2_binding.json'
CACHE=str(O.ROOT/'results/native_expert_scaling/meth511_runtime/empty_cache')
assert sys.pycache_prefix==CACHE
Path(CACHE).mkdir(exist_ok=True);assert not any(Path(CACHE).rglob('*'))
out=O.ROOT/'results/native_expert_scaling/meth511_r2_binding';out.mkdir(exist_ok=False)
progress=(out/'progress.jsonl').open('x',encoding='utf8')
setup=json.loads((O.DOC/'meth511_runtime_setup_result.json').read_bytes())
runtime=Path(sys.executable).parent.parent
previous=json.loads((O.DOC/'meth506_r4_2_binding.json').read_bytes())
donor_directory=Path(previous['donor_directory'])
donor_archives={Path(r['path']).name:r for r in previous['donor'] if Path(r['path']).suffix=='.bin'}
assert len(donor_archives)==3
def add(path,expected=None):
    # All walk roots are resolved ONCE. Their child paths are already canonical;
    # avoid repeatedly traversing deep junction chains for each file.
    p=Path(os.path.abspath(path));key=str(p).lower();P.guard()
    if p.name in donor_archives and p.parent==donor_directory:
        retained=donor_archives[p.name];s=p.stat();assert s.st_size==retained['bytes']
        row={'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':retained['sha256'],'full_sha_freshly_recomputed':False,'identity_scope':'Retained acquisition ZIP SHA; actual all3320 canonical F32 tensor SHAs checked before first donor inference.'}
        if expected:assert expected==retained['sha256']
        P.catalog[key]=row;return row
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
            'original_scientific_calls':0,'donor_archive_scope':'Raw ZIP SHA retained from acquisition; fresh actual all3320 F32 tensor SHA is mandatory before first donor inference, under unchanged3600s/24GiB donor bounds. No fresh whole ZIP SHA claim.','runtime_scope':'All Python/native/data/metadata executable files in canonical whitelisted source roots; excludes tests/include/share/cache and unused activation scripts. Strict actual-import/image admission remains.',
            'bytecode_scope':'-I -B -X pycache_prefix=absolute_empty_cache; existing foreign bytecode is not read, and no cache is written.',
            'runtime_walk_algorithm':'Canonical roots once; no per-file deep junction resolution. Full SHA freshly computed on selected executable files and whole source/candidate/corpus/ledgers; original donor ZIP SHA retained with current stat and write-deny binding, fresh all3320 canonical tensor SHA before inference.'})
        deriv=json.loads((O.ROOT/'benchmarks/native_expert_scaling/meth511_r2_derivation.json').read_bytes())
        for row in deriv:
            assert add(row['old'],row['old_sha256']) and add(row['new'],row['new_sha256'])
            text=Path(row['old']).read_text(encoding='utf8')
            for change in row['replacements']:
                assert text.count(change['before'])==change['occurrences'];text=text.replace(change['before'],change['after'])
            assert text==Path(row['new']).read_text(encoding='utf8')
        value['literal_operational_derivation']=deriv
        value['native_runtime_environment']=previous['runtime_environment']
        assert value['native_runtime_environment']['OMP_NUM_THREADS']=='3'
        value['price']['donor_fresh_canonical_hash_phase']='donor: ALL3320 loaded F32 tensor bytes before first inference; no duplicated whole-ZIP hash'
        value['price']['donor_raw_zip_SHA_scope']='retained acquisition SHA, current size/mtime and stage write-deny handles'
        add(O.DOC/'meth511_binding.failure.json')
        add(O.DOC/'METH_511_R1_RUNTIME_INVENTORY_20261006.md')
        add(O.DOC/'METH_511_R2_CANONICAL_DONOR_20261007.md')
        value['catalog']=list(P.catalog.values());value['price']['catalog_files']=len(P.catalog);value['price']['catalog_bytes']=sum(r['bytes'] for r in P.catalog.values())
    oldwrite(path,value)
O.write=write
try:P.main()
finally:progress.close()
