"""Bind one source-null prior compiler to qualified kernel and cached labels."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once
from chatbot_directional_binding import entry,receipt
from chatbot_null_prior_kernel_binding import checked


def main(args):
    assert not args.out.exists();doc=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';code=ROOT/'benchmarks/native_expert_scaling'
    prior_path=doc/'chatbot_null_prior_kernel_binding_20261008.json';assert sha(prior_path)=='2b5b63bb4b801066e74362e471c92f0ab203d407194fce7a3c473a9e3b042ee0'
    prior=json.loads(prior_path.read_bytes())
    kernel=doc/'chatbot_null_prior_kernel_20261008.json';cov,covterm,ct=checked(kernel,'7b84d405b1c86c65a9d3babaa0b898ca8264e5d67eefd789f997b47fb94c8fee')
    assert cov['decision']=='ELIGIBLE_FOR_ONE_NULL_PRIOR_CONVEX_COMPILER' and all(cov['eligibility_gates'].values())
    audit=doc/'chatbot_null_prior_kernel_audit_20261008.json';_,auditerm,_=checked(audit,'7b73c93c93e94d4d0cab1d68a15285b5b75cc1ea74ff2ec77781fe148339f23d')
    closure=doc/'chatbot_null_prior_kernel_terminal_20261008.json';closed=json.loads(closure.read_bytes())
    assert len(closed['instances'])==4 and all(v['owned_instance_closed'] and not v['faults'] for v in closed['instances'])
    paths=[prior_path,kernel,covterm,audit,auditerm,closure,
        *(receipt(cov['outputs'][n]['path'],ct) for n in ('K_H','L','Pi_N','H_leaf','L_H_leaf'))]
    oldkernel_path=Path(prior['old_kernel_path']);old=json.loads(oldkernel_path.read_bytes())
    oldterm=oldkernel_path.with_suffix('.terminal.json');ot=json.loads(oldterm.read_bytes())
    paths.extend(receipt(old['outputs'][n]['path'],ot) for n in ('FIT_x','FIT_U','FIT_W','FIT_weights','mu'))
    cached_path=doc/'chatbot_kernel_fit_20261008.json';cached,cacheterm,cachedterm=checked(cached_path,'1c776995e079e3bbcac256345dcf0702c22c8dfbc50ef0b181cc8ea7fb829a48')
    caudit=doc/'chatbot_kernel_fit_audit_20261008.json';_,caterm,_=checked(caudit,'91f139bbd6fede49f80ba6bcfd916f652f9a12aa5b2add899eddf648471b5533')
    aliases={}
    for field,n in (('FIT_source_y_path','FIT_source_y_BF16'),('FIT_shared_F64_path','FIT_shared_F64'),('FIT_shared_F32_path','FIT_shared_batched_F32'),('prefix_shared_F32_path','novel_prefix64_shared_batched_F32')):
        aliases[field]=str(receipt(cached['outputs'][n]['path'],cachedterm));paths.append(aliases[field])
    cached_binding=doc/'chatbot_kernel_fit_binding_20261008.json';assert sha(cached_binding)=='072ef6ec52538814c3a37efdba373a37b89e198d0a82820fe5f7d14105c48db7'
    cb=json.loads(cached_binding.read_bytes())
    directional=doc/'chatbot_directional_jacobian_20261008.json';_,dterm,dt=checked(directional,'576f0c5a8495456fe6d4dac9b0055e8a32d964546029d00ddc4f89c337356b22')
    source_j=receipt(ROOT/'results/native_expert_scaling/chatbot_directional_jacobian_20261008/source_J_F64.npy',dt)
    compiled=doc/'chatbot_coupled_compile_20261008.json';comp,compterm,cp=checked(compiled,'9569e690c5975d01e9b9f9e5bc454be64563e93a2fc6c9efa93648055ab93f66')
    core_j=receipt(comp['output_arrays']['core_FIT_J_F64']['path'],cp)
    journal=receipt(cb['prefix_journal_path'],cp);records=json.loads(journal.read_bytes())
    probe=doc/'chatbot_coupled_probe_20261008.json';_,probeterm,pt=checked(probe,'048e547c82910e49c9a79d6c2edfd31b8ad7e5a9562436eb2cef51e7fe606435')
    prefix_core=receipt(cb['old_prefix_core_path'],pt)
    pilotterm=doc/'chatbot_joint_pilot_20261007.terminal.json';routes=receipt(cb['saved_E16_routes'],json.loads(pilotterm.read_bytes()))
    worker=code/'chatbot_null_prior_fit.py'
    paths.extend([*(v['path'] for v in prior['inputs']),oldterm,cached_path,cacheterm,caudit,caterm,cached_binding,directional,dterm,source_j,
        compiled,compterm,core_j,journal,probe,probeterm,prefix_core,pilotterm,routes,cb['adoption_path'],*(v['binary_path'] for v in records),
        worker,Path(__file__).resolve(),code/'chatbot_kernel_fit.py',code/'chatbot_coupled_probe.py',code/'chatbot_directional_plan.py',
        doc/'CHATBOT_NULL_PRIOR_FIT_PROTOCOL_20261008.md'])
    b=dict(prior,kernel_path=str(kernel),source_J_path=str(source_j),core_J_path=str(core_j),prefix_journal_path=str(journal),
        prefix_shared_F64_path=str(prefix_core),saved_E16_routes=str(routes),adoption_path=cb['adoption_path'],**aliases,
        inputs=[entry(p) for p in dict.fromkeys(paths)],
        job=dict(name='null_prior_fit',worker_path=str(worker.resolve()),result_schema='QWEN_NULL_PRIOR_CONVEX_RESULT_V1',
            accepted_decisions=['CLOSE_NULL_PRIOR_NUMERICAL_COMPILER','CLOSE_NULL_PRIOR_CONVEX_NOVEL_FIDELITY','NOVEL_PREFIX_INCONCLUSIVE_NO_FIDELITY_ADMISSION'],
            limits=dict(worker_seconds=300,family_seconds=540,OS_bytes=4<<30,output_bytes=512<<20,log_bytes=2<<20)))
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);main(parser.parse_args())
