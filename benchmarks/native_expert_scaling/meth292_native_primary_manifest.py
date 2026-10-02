#!/usr/bin/env python3
"""Model-free new source selection for the explicitly revised native-primary study."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import torch
import meth278_complete_i16_fresh_manifest as T
S=T.S
import meth277_complete_i16_development as D

M=S.M
DEVELOPMENT=M.DOC/'meth277_complete_i16_development_result.json'
DEVELOPMENT_SHA='3d997545fc690a5ebb3681e60fb0d3dc190a2a292e94056e5cf0abcb9ed88b35'
OLD=M.DOC/'meth278_complete_i16_fresh_manifest.json'
OLD_SHA='7fc74c2ff172bcca5d71de685f1e230240d05789e11b0bb3b9290b64a4c8e053'
M.SEED='meth292-native-primary-independent-292292'
M.PRIOR=M.PRIOR+((OLD,OLD_SHA),)
M.MAX_SECONDS=600
POLICY=M.DOC/'METH_292_NATIVE_PRIMARY_EVALUATION_POLICY_20261002.md'
POLICY_SHA='0a3aab8f4733523b3af0b9d6bf919de406b1e94da249d5a3125176a246c57753'
SURGERY=M.DOC/'meth291_operator_surgery_result.json'
SURGERY_SHA='fc740ace69a054eecb9c43a48a2742770ed5767db45a0c7f2cd22e9c115e6483'
PROFILE={'benchmarks/native_expert_scaling/meth285_model_operator.h':'1b71af83d1a86508cba1f78212f3c81ed4a98cba79864293187a43fb19fe780d',
         'benchmarks/native_expert_scaling/meth284_source_operator.h':'1e2a6a7e2e82b25b9968ae8ea206483445d787254a419e89ad875f081a3e7d64',
         'benchmarks/native_expert_scaling/meth284_archive_catalog.h':'15dc10453baf8f928aae8b5a20a1521272cc44cda385184d44a5e34b7da2226c'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();start=time.monotonic();stage='bindings'
    assert all(not p.exists() for p in (args.out,args.out.with_suffix('.failure.json')))
    torch.set_num_threads(6)
    try:
        assert M.digest(POLICY)==POLICY_SHA and M.digest(SURGERY)==SURGERY_SHA
        surgery=json.loads(SURGERY.read_text(encoding='utf-8'))
        assert surgery['gates']['all_CPU_hybrid_exact_original285_tail8'] and surgery['C_fields_bound']==725
        for path,sha in PROFILE.items():assert M.digest(M.ROOT/path)==sha,path
        for path,sha in ((DEVELOPMENT,DEVELOPMENT_SHA),(D.COMPOSITION,D.COMPOSITION_SHA),(OLD,OLD_SHA),(S.OLD,S.OLD_SHA)):
            assert M.digest(path)==sha,str(path)
        development=json.loads(DEVELOPMENT.read_text(encoding='utf-8'))
        composition=json.loads(D.COMPOSITION.read_text(encoding='utf-8'))
        old=json.loads(OLD.read_text(encoding='utf-8'))
        assert development['decision']=='diagnostic_development_pass_requires_fresh_complete_quality'
        assert all(development['gates'].values()) and development['diagnostic_only'] is True
        assert Pinned_archive(composition,development)
        assert M.digest(Path(D.__file__))==development['script_sha256']
        assert M.digest(Path(D.C.__file__))==composition['script_sha256']
        for path,sha in composition['helper_sha256'].items():assert M.digest(Path(path))==sha
        assert old['pg19_parquet']==M.PG19_REL and old['pg19_parquet_sha256']==S.PG_SHA
        assert subprocess.check_output(['git','rev-parse',M.REF],cwd=M.ROOT,text=True).strip()==M.REF==old['source_commit']
        stage='cached_corpus_tokenizer';assert M.digest(M.ROOT/M.PG19_REL)==S.PG_SHA
        tokenizer=M.AutoTokenizer.from_pretrained(M.M42.MODEL,revision=M.M42.REV,local_files_only=True)
        assert M.M15.M13.C.tok_fingerprint(tokenizer)==M.M15.M13.TOK_FP
        stage='all_prior_source_and_fragment_exclusion'
        excluded,base,priors=M.prior_context(tokenizer);initial=len(excluded)
        assert initial==old['excluded_source_count']+24==3224
        for path in (OLD,T.OLD,S.OLD):
            previous=json.loads(path.read_text(encoding='utf-8'))
            assert all(item['source_id'] in excluded for item in previous['items'])
        print(json.dumps({'stage':stage,'excluded_source_count':initial,'runtime':M.budget(start)}),flush=True)
        stage='fixed_hash_rank_source_selection';items,provenance,rows=M.select(tokenizer,excluded,base,start)
        assert rows==old['pg19_rows']==1243 and len(items)==24
        assert len({item['source_id'] for item in items})==24
        assert all(item['source_id'] not in {old_item['source_id'] for old_item in old['items']} for item in items)
        for item in items:
            if item['category']=='prose':item['source_kind']='pg19_train14'
        result={'experiment':'METH-292-native-primary-independent-source-manifest',
            'evaluation_policy_sha256':POLICY_SHA,'native_profile_sha256':PROFILE,'surgery291_result_sha256':SURGERY_SHA,
            'prior_GPU_numeric_fidelity_qualified':False,
            'development_result_sha256':DEVELOPMENT_SHA,'composition_result_sha256':D.COMPOSITION_SHA,
            'artifact_sha256':composition['artifact']['sha256'],'source_sha256':D.P.M57.MODEL_SHA,
            'parent_checkpoint_sha256':D.P.M57.CHECKPOINT_SHA,'child_checkpoint_sha256':D.P.M122.SPECIALIZED_SHA,
            'model':M.M42.MODEL,'revision':M.M42.REV,'tokenizer_fingerprint':M.M15.M13.TOK_FP,
            'source_commit':M.REF,'pg19_parquet':M.PG19_REL,'pg19_parquet_sha256':S.PG_SHA,'pg19_rows':rows,
            'prior_manifest_sha256':priors,'excluded_source_count':initial,'h0_calib_sha256':M.M17.CALIB_SHA,
            'seed':M.SEED,'selected_counts':M.COUNTS,'chat_template':M.M42.REQUEST,'excerpt_characters':384,
            'selection_provenance':provenance,'items':items,'runtime':M.budget(start),
            'script_sha256':M.digest(Path(__file__)),'diagnostic_only':True,'native_promotion_qualified':False,
            'decision':'new_source_manifest_frozen_require_source_only_answerability_before_inference',
            'scope':'New source IDs/project-transfer heldout,including consumed261/278 source exclusions;not a new corpus or certified donor-pretraining exclusion. No model inference/source rejection by scores or quality/rate/useful-n/family promotion.'}
        D.C.I.A.dump(args.out,result)
        print(json.dumps({'decision':result['decision'],'manifest_sha256':M.digest(args.out),
            'source_ids':[item['source_id'] for item in items],'runtime':result['runtime']}),flush=True)
    except BaseException as failure:
        D.C.I.A.dump(args.out.with_suffix('.failure.json'),{'stage':stage,
            'error':type(failure).__name__+': '+str(failure),'seconds':time.monotonic()-start,
            'script_sha256':M.digest(Path(__file__))});raise


def Pinned_archive(composition,development):
    assert all(composition['gates'].values())
    assert composition['diagnostic_only'] is True and composition['native_promotion_qualified'] is False
    assert composition['artifact']['sha256']==development['artifact_sha256']
    assert M.digest(Path(composition['artifact']['path']))==composition['artifact']['sha256']
    return True


if __name__=='__main__':main()
