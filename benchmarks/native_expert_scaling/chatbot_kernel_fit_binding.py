"""Bind one convex FIT compiler without full source weights or old source J."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once
from chatbot_directional_binding import entry,receipt


def main(args):
    assert not args.out.exists();doc=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';code=ROOT/'benchmarks/native_expert_scaling'
    prior_path=doc/'chatbot_kernel_binding_20261008.json';prior=json.loads(prior_path.read_bytes())
    assert sha(prior_path)=='28c06df3c792251bc2072b2fd13e30e980f4677aaffbdbf76b1cc07243af8810'
    for item in prior['inputs']:
        assert sha(item['path'])==item['sha256'] and Path(item['path']).stat().st_size==item['bytes']
    kernel=doc/'chatbot_kernel_20261008.json';assert sha(kernel)=='d63a24ec90d5c382f9c00655afcecc9f70ca1d43a1e1d876ac744afbd002aa9c'
    audit=doc/'chatbot_kernel_audit_20261008.json';assert sha(audit)=='ef4bee138ac55c1ffc3f8f6ad0cbd9fbeed22226e10c3fab9bfa76c32c0ced73'
    audit_terminal=audit.with_suffix('.terminal.json');term=json.loads(audit_terminal.read_bytes())
    assert term['actual_worker_exit_code']==0 and term['result_sha256']==sha(audit) and all(term['gates'].values())
    kterm=kernel.with_suffix('.terminal.json');kt=json.loads(kterm.read_bytes());paths=[prior_path,kernel,kterm,audit,audit_terminal]
    for item in kt['output_manifest']:
        assert sha(item['path'])==item['sha256'] and Path(item['path']).stat().st_size==item['bytes'];paths.append(item['path'])
    coupled_path=doc/'chatbot_coupled_compile_binding_20261008.json';coupled=json.loads(coupled_path.read_bytes())
    assert sha(coupled_path)=='7b6f172f27a379b01e0fc7c503be463f909fade30fa80d6544994783dc4130d6'
    compiled=doc/'chatbot_coupled_compile_20261008.json';old=json.loads(compiled.read_bytes());oldterm=compiled.with_suffix('.terminal.json')
    oldprobe=doc/'chatbot_coupled_probe_20261008.json';probe=json.loads(oldprobe.read_bytes());probeterm=oldprobe.with_suffix('.terminal.json')
    assert sha(compiled)=='9569e690c5975d01e9b9f9e5bc454be64563e93a2fc6c9efa93648055ab93f66'
    assert sha(oldprobe)=='048e547c82910e49c9a79d6c2edfd31b8ad7e5a9562436eb2cef51e7fe606435'
    ct=json.loads(oldterm.read_bytes());pt=json.loads(probeterm.read_bytes())
    assert ct['actual_worker_exit_code']==pt['actual_worker_exit_code']==0 and ct['result_sha256']==sha(compiled) and pt['result_sha256']==sha(oldprobe)
    original=doc/'chatbot_joint_retained_audit_20261007.json';assert sha(original)=='8eca7c810284a8dd1effa3d64d23a1f4d7010028df4748aeb54a6e0c843c23c8'
    old_core=Path(old['output_arrays']['core_FIT_values_F64']['path']);prefix_core=Path(probe['outputs']['prefix64_shared_only_F64']['path'])
    journal=Path(old['output_arrays']['novel_prefix64_F32']['path']).with_name('novel_prefix64_journal.json')
    receipt(old_core,ct);receipt(prefix_core,pt);receipt(journal,ct)
    prefix=json.loads(journal.read_bytes())
    core_paths={n:str(Path(old['output_arrays']['shared_'+letter+'_BF16']['path'])) for n,letter in (('core_g_path','g'),('core_u_path','u'),('core_b_path','b'))}
    for p in core_paths.values():receipt(p,ct)
    worker=code/'chatbot_kernel_fit.py'
    paths.extend([*(item['path'] for item in prior['inputs']),coupled_path,compiled,oldterm,oldprobe,probeterm,
        coupled['plan_path'],original,old_core,prefix_core,journal,*core_paths.values(),*(item['binary_path'] for item in prefix),
        worker,Path(__file__).resolve(),code/'chatbot_coupled_probe.py',code/'chatbot_directional_plan.py',doc/'CHATBOT_KERNEL_FIT_PROTOCOL_20261008.md'])
    b=dict(schema=prior['schema'],python=prior['python'],kernel_path=str(kernel),adoption_path=prior['adoption_path'],saved_E16_routes=prior['saved_E16_routes'],
        plan_path=coupled['plan_path'],original_response_audit=str(original),old_FIT_core_path=str(old_core),old_prefix_core_path=str(prefix_core),
        prefix_journal_path=str(journal),**core_paths,inputs=[entry(p) for p in dict.fromkeys(paths)],
        capture_runtime_roots=prior['capture_runtime_roots'],interaction_view=prior['interaction_view'],
        job=dict(name='kernel_fit',worker_path=str(worker.resolve()),result_schema='QWEN_FULL_FIT_CONVEX_RESULT_V1',
            accepted_decisions=['CLOSE_FULL_FIT_NUMERICAL_COMPILER','CLOSE_FULL_FIT_CONVEX_NOVEL_FIDELITY','ELIGIBLE_FOR_SEPARATE_NATIVE_COMPOSITION'],
            limits=dict(worker_seconds=300,family_seconds=540,OS_bytes=4<<30,output_bytes=512<<20,log_bytes=2<<20)))
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
