"""Bind ONE augmented fit, qualified NEW kernel and immutable cached labels."""
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
    kbpath=doc/'chatbot_quadratic_kernel_binding_20261008.json';assert sha(kbpath)=='6b6aad674e0862c80c6969127ff159cb4a86849bc9d5f30407fcee21e637a6bb';kb=json.loads(kbpath.read_bytes())
    kpath=doc/'chatbot_quadratic_kernel_repair1_20261008.json';k,ktpath,kt=checked(kpath,'168bfabd57d6355dc9ac9f9ff42443247570e26b3f0916f8e4cce0e99d6727e6')
    kapath=doc/'chatbot_quadratic_kernel_repair1_audit_20261008.json';ka,katpath,_=checked(kapath,'8dc6cd70e148b8aef8c7537fb8f4048ee6ab117a7eb56770412a75cb7a67758f')
    assert k['decision']=='ELIGIBLE_FOR_ONE_QUADRATIC_CONVEX_COMPILER' and all(k['eligibility_gates'].values()) and ka['decision']=='SAVED_QUADRATIC_KERNEL_INDEPENDENTLY_VERIFIED'
    closure=doc/'chatbot_quadratic_kernel_repair1_terminal_20261008.json';assert sha(closure)=='a841e586880c9ea599fab7db3ad3afad01db5c4f8f82128ad83a90cd1883c707'
    fp=doc/'chatbot_kernel_fit_20261008.json';f,ftpath,ft=checked(fp,'1c776995e079e3bbcac256345dcf0702c22c8dfbc50ef0b181cc8ea7fb829a48')
    fa=doc/'chatbot_kernel_fit_audit_20261008.json';_,fatpath,_=checked(fa,'91f139bbd6fede49f80ba6bcfd916f652f9a12aa5b2add899eddf648471b5533')
    fbpath=doc/'chatbot_kernel_fit_binding_20261008.json';assert sha(fbpath)=='072ef6ec52538814c3a37efdba373a37b89e198d0a82820fe5f7d14105c48db7';fb=json.loads(fbpath.read_bytes())
    opath=Path(kb['old_kernel_path']);old,otpath,ot=checked(opath,'d63a24ec90d5c382f9c00655afcecc9f70ca1d43a1e1d876ac744afbd002aa9c')
    factor=Path(kb['factor_path']);fr,frtpath,frt=checked(factor,'85577acc7a24677cde7b489974b932c97b7ee8244061c13fb41f54fd16288389')
    prefix_probe=doc/'chatbot_coupled_probe_20261008.json';_,pptpath,ppt=checked(prefix_probe,'048e547c82910e49c9a79d6c2edfd31b8ad7e5a9562436eb2cef51e7fe606435')
    compilepath=doc/'chatbot_coupled_compile_20261008.json';_,ctpath,ct=checked(compilepath,'9569e690c5975d01e9b9f9e5bc454be64563e93a2fc6c9efa93648055ab93f66')
    used=[receipt(old['outputs'][n]['path'],ot) for n in ('FIT_x','FIT_U','FIT_W','FIT_weights','mu')]
    used.extend(receipt(k['outputs'][n]['path'],kt) for n in ('G','L','B_Q','Q_raw','sigma','mu_F32','radius_F32'))
    reused={key:str(receipt(f['outputs'][name]['path'],ft)) for key,name in (
        ('FIT_source_y_path','FIT_source_y_BF16'),('FIT_shared_F64_path','FIT_shared_F64'),('FIT_shared_F32_path','FIT_shared_batched_F32'),
        ('RHS_path','dual_RHS'),('prefix_shared_F32_path','novel_prefix64_shared_batched_F32'))}
    reused['prefix_shared_F64_path']=str(receipt(fb['old_prefix_core_path'],ppt))
    reused.update({name+'_path':str(receipt(fr['outputs'][name+'_BF16']['path'],frt)) for name in ('R','T')})
    journal=receipt(fb['prefix_journal_path'],ct);prefix=json.loads(Path(journal).read_bytes());assert len(prefix)==64
    source=Path(fb['original_response_audit']);assert sha(source)=='8eca7c810284a8dd1effa3d64d23a1f4d7010028df4748aeb54a6e0c843c23c8'
    for name in ('adoption_path','saved_E16_routes'):
        item=next(v for v in fb['inputs'] if Path(v['path'])==Path(fb[name]));assert sha(item['path'])==item['sha256']
    worker=code/'chatbot_quadratic_fit.py'
    helpers=['chatbot_interaction_launch.py','chatbot_directional_launch.py','chatbot_directional_binding.py','chatbot_null_prior_kernel_binding.py',
        'chatbot_kernel_fit.py','chatbot_joint_retained_audit.py','chatbot_coupled_probe.py','chatbot_directional_plan.py']
    paths=[kbpath,kpath,ktpath,kapath,katpath,closure,fp,ftpath,fa,fatpath,fbpath,opath,otpath,factor,frtpath,prefix_probe,pptpath,
        compilepath,ctpath,*used,*reused.values(),journal,source,fb['adoption_path'],fb['saved_E16_routes'],*(item['binary_path'] for item in prefix),
        kb['python'],Path(kb['python']).with_name('python312.dll'),worker,Path(__file__).resolve(),*(code/n for n in helpers),
        code/'chatbot_quadratic_fit_audit.py',code/'chatbot_quadratic_fit_audit_binding.py',code/'chatbot_quadratic_terminal_repair1.ps1',
        doc/'CHATBOT_QUADRATIC_FIT_PROTOCOL_20261008.md',doc/'CHATBOT_PIPELINE_REASSESSMENT_20261008.md']
    b=dict(schema=kb['schema'],python=kb['python'],kernel_path=str(kpath),old_kernel_path=str(opath),
        adoption_path=fb['adoption_path'],saved_E16_routes=fb['saved_E16_routes'],prefix_journal_path=str(journal),original_response_audit=str(source),
        **reused,inputs=[entry(p) for p in dict.fromkeys(paths)],capture_runtime_roots=kb['capture_runtime_roots'],interaction_view=kb['interaction_view'],
        job=dict(name='quadratic_fit',worker_path=str(worker),result_schema='QWEN_QUADRATIC_CONVEX_RESULT_V1',
            accepted_decisions=['CLOSE_QUADRATIC_NUMERICAL_COMPILER','CLOSE_QUADRATIC_CONVEX_NOVEL_FIDELITY','ELIGIBLE_FOR_SEPARATE_FULL_NOVEL_SCREEN'],
            limits=dict(worker_seconds=300,family_seconds=540,OS_bytes=4<<30,output_bytes=512<<20,log_bytes=2<<20)))
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
