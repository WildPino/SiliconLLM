"""Metadata-only SHA seal of the FIRST528 resource fault and original partials."""
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parent))
from meth528_operations import ROOT,DOC,write


def main():
    start=time.monotonic();proc=psutil.Process();hashed=0
    def item(path,expected=None,offset=0,count=None):
        nonlocal hashed
        p=Path(path).resolve();st=p.stat();h=hashlib.sha256();left=st.st_size-offset if count is None else count
        with p.open('rb') as f:
            f.seek(offset)
            while left:
                data=f.read(min(8<<20,left));assert data;h.update(data);hashed+=len(data);left-=len(data)
                assert time.monotonic()-start<=60 and proc.memory_info().peak_wset<=128<<20
        assert (p.stat().st_size,p.stat().st_mtime_ns)==(st.st_size,st.st_mtime_ns)
        if expected:assert h.hexdigest()==expected,str(p)
        return dict(path=str(p),offset=offset,bytes=st.st_size-offset if count is None else count,sha256=h.hexdigest())
    binding=item(DOC/'meth528_binding.json','9de128cb77373c0e5c35cb24d398b1269d68b79f54fc2e943c2a845236ec49c2')
    fault=item(DOC/'meth528_main_result.failure.json');b=json.loads(Path(binding['path']).read_bytes());r=json.loads(Path(fault['path']).read_bytes())
    assert r['fault']=='hard deadline' and r['resource']['seconds']>240 and r['binding_sha256']==binding['sha256']
    assert len(r['gates'])==4 and all(r['gates'].values()) and not (DOC/'meth528_main_result.json').exists()
    exits=json.loads((DOC/'meth528_original_executor_exits.json').read_bytes());assert exits[0]['exit_code']==0 and exits[1]['exit_code']==1
    for raw in (b,r):
        inst=raw['process_instance']
        try:p=psutil.Process(inst['pid']);assert abs(p.create_time()-inst['create_time_unix'])>.002
        except psutil.NoSuchProcess:pass
    outputs=[item(v['path']) for v in r['partial_outputs']]
    folder=ROOT/'results/native_expert_scaling/meth528_atoms'
    assert {Path(v['path']) for v in outputs}=={p.resolve() for p in folder.iterdir() if p.is_file()}
    paired=[e for e in range(1,128) if all((folder/f'e{e:03d}_a{arm}_hidden_codes.npy').is_file() for arm in (0,1))]
    assert paired==list(range(1,14)) and all((folder/f'e{e:03d}_a{a}.bin').is_file() for e in range(1,128) for a in (0,1))
    legacy=[item(v['path'],v['sha256']) for v in b['legacy_output_inventory']]
    for v in b['source_extents']:item(v['path'],v['sha256'],v['offset'],v['bytes'])
    st=Path(b['payload']['path']).stat();assert (st.st_size,st.st_mtime_ns)==(b['payload_stat']['bytes'],b['payload_stat']['mtime_ns'])
    for rel,sha in b['preserved'].items():item(ROOT/rel,sha)
    assert not any(p.is_file() for p in (ROOT/'results/native_expert_scaling/meth511_runtime/empty_cache').rglob('*'))
    freeze=json.loads((folder/'development_mask_freeze.json').read_bytes());assert len(freeze['cases'])==127
    record=dict(experiment='METH528_FIRST_FAULT_SEAL',binding=binding,original_failure=fault,
        original_main_completed=False,original_main_resource_gate=False,original_executor_exits=exits,
        original_partial_output_inventory=outputs,original_legacy_output_inventory=legacy,
        code_order_guaranteed_completed_parent_prefix=list(range(1,13)),frontier_expert=13,
        frontier_status='UNRESOLVED_COMPLETE_OR_PARTIAL_NO_DURABLE_ROW_CHECKPOINT; codes persist, original arrays unqualified until independent audit',
        ALL127_candidate_rotation_banks_present=True,full_scope_quality_or_eligibility=None,new_source_function_model_native_calls=0,
        no_original_scientific_replay=True,empty_cache_and_foreign_preserved=True,
        metadata_process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        metadata_ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        metadata_resource=dict(seconds=time.monotonic()-start,OS_peak_bytes=proc.memory_info().peak_wset,bytes_hashed=hashed,limits=[60,128<<20]),
        scope='Metadata hashes only, no NumPy/source response/metric reduction/C compilation. Preserve all original bytes BEFORE numbered repair. Bank-only prefix and1..12 code-order evidence do not admit whole original main.')
    dest=DOC/'RETENTION_528_FIRST_FAULT_20261007.json';write(dest,record);print(json.dumps(dict(retention=item(dest),resource=record['metadata_resource'])))


if __name__=='__main__':main()
