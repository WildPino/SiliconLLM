"""Sole fresh metadata freeze: source472/476 chains and full inherited provenance."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import psutil

ROOT=Path(__file__).resolve().parents[2]; DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'meth493_binding.json'; FAIL=DEST.with_suffix('.failure.json')
assert not DEST.exists() and not FAIL.exists()
START=time.monotonic(); PROC=psutil.Process(); PROC.cpu_affinity([0]); PEAK=HASHED=0

def guard():
    global PEAK
    PEAK=max(PEAK,PROC.memory_info().peak_wset)
    assert PEAK <= 256 << 20 and time.monotonic()-START <= 90

def item(path, expected=None):
    global HASHED
    path=Path(path).resolve(); before=path.stat(); h=hashlib.sha256()
    with path.open('rb') as stream:
        while data:=stream.read(4<<20): h.update(data); HASHED+=len(data); guard()
    after=path.stat(); assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    result={'path':str(path),'bytes':before.st_size,'sha256':h.hexdigest()}
    if expected: assert result['sha256']==expected, str(path)
    return result

def head(path):
    path=Path(path)
    saved=subprocess.check_output(['git','show','HEAD:'+path.relative_to(ROOT).as_posix()],cwd=ROOT,timeout=5)
    assert path.read_bytes().replace(b'\r\n',b'\n')==saved.replace(b'\r\n',b'\n'); guard()

try:
    inherited=item(DOC/'meth492_r1_binding.json','e1128e2a54874d41e956ff2edd0dca78dfff33530c93e55e26dce82a3fdb88b1')
    binding=json.loads(Path(inherited['path']).read_bytes())
    catalog={v['path']:item(v['path'],v['sha256']) for v in binding['catalog']}
    records=[inherited]; parsed={}
    expected={
        'ADMISSION_492_20261006.json':'368a74840f0f3c8d3e656c24cc73247ae4c0eb07e1806c6bc5abfd499f0e09c0',
        'meth493_weighted_target_metadata.json':'d9f82c1b55f1ebc2f7470aff60d8a24c290dfc58462390831f54d3182a9de50d',
        'meth472_switch_private_input_probe_result.json':'9fc2efb0acde8f354fffa81e8525dcd30940b07eb8f4bbf7fed49321193ae0b3',
        'RETENTION_472_20261005.json':'db9c654df1bfb63ff31f85079a6853d4564ad5c772951c141742c63b51b7eae5',
        'meth476_switch_last_layer_context_result.json':'1f5402b1cf059458ad095c113070c6a5c99a8a6ccf29f280cdfb4c718dd0230f',
        'RETENTION_476_20261005.json':'3f4730698e4bbe3c44b96785379daf7d8bec18b2f2114e933ffcd466dbd9c508',
        'meth476_prospective_bindings.json':'2d3d5b225a0126c293ed716c99cc33125e36f8c7e64b628bf9b9c02db4991dc2'}
    for name,sha in expected.items():
        path=DOC/name; head(path); records.append(item(path,sha)); parsed[name]=json.loads(path.read_bytes())
    for name in ('ADMISSION_492_20261006.json','RETENTION_472_20261005.json','RETENTION_476_20261005.json'):
        assert all(parsed[name]['gates'].values())
    for label in (472,476):
        rawname=f'meth{label}_'+('switch_private_input_probe' if label==472 else 'switch_last_layer_context')+'_result.json'
        assert all(parsed[rawname]['apparatus_gates'].values())
        assert parsed[f'RETENTION_{label}_20261005.json']['raw_sha256']==expected[rawname]
    metadata=parsed['meth493_weighted_target_metadata.json']
    assets={Path(v['path']).name:item(v['path'],v['sha256']) for v in metadata['assets']}
    for asset in assets.values(): catalog[asset['path']]=asset
    capture=DOC/'meth471_switch_development_capture_result.json'; head(capture); records.append(item(capture))
    ledger=json.loads(capture.read_bytes())['ledger']; assets['ledger']=item(ledger['path'],ledger['sha256'])
    assert assets['ledger']['bytes']==20+118*387036
    catalog[assets['ledger']['path']]=assets['ledger']
    # Bind all published 476 source-validation outputs; no native replay is performed.
    for v in parsed['RETENTION_476_20261005.json']['files']: catalog[v['path']]=item(v['path'],v['sha256'])
    new=[]
    paths=sorted((ROOT/'benchmarks/native_expert_scaling').glob('meth493*'))+[
        DOC/'METH_493_WEIGHTED_TARGET_PROTOCOL_20261006.md',
        DOC/'METH_493_WEIGHTED_FUNCTION_TRANSFER_ELIGIBILITY_20261006.md']
    for path in paths: head(path); new.append(item(path))
    for v in records+new: catalog[v['path']]=v
    binding['scientific']+=new; binding['records']+=records
    binding.update(experiment='METH493 full bank11 weighted-function compiler',catalog=list(catalog.values()),
        weighted_source=assets,inherited_output_bytes=0,
        freeze_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip(),
        preparation_seconds=time.monotonic()-START,preparation_peak_bytes=PEAK,preparation_hashed_bytes=HASHED,
        builder_wall_limit_seconds=90,builder_host_limit_bytes=256<<20,new_native_calls=0,new_optimizer_updates=0)
    guard()
    with DEST.open('xb') as stream:
        stream.write((json.dumps(binding,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
    print(json.dumps({'binding':str(DEST),'sha256':item(DEST)['sha256'],
        'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED}))
except BaseException as exc:
    with FAIL.open('xb') as stream:
        stream.write((json.dumps({'error':str(exc),'traceback':traceback.format_exc(),
            'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},indent=2)+'\n').encode())
    raise
