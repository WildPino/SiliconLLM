#!/usr/bin/env python3
"""Model-free new source selection after complete-core development pass."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import meth260_complete_core_development as D
import meth213_stored_core_fresh_manifest as S

M=S.M
DEVELOPMENT=D.P.DOC/'meth260_complete_core_development_repair1_result.json'
DEVELOPMENT_SHA='f8e102ce698410e43df7a5b1c14a2620a80df185ce65f0d535ba0bf0c4d046f9'
OLD=D.P.DOC/'meth213_stored_core_fresh_manifest.json'
OLD_SHA='bcd8295be1b2d62dceaf103f985182bad11ac50dd739a63cfb92d11400b5af10'
PG_SHA='15f3ecb5e314ce58bbf8eae927b51bf893b9fdabe4a04c8e246547206357a2d5'
M.SEED='meth261-complete-core-independent-261261'
M.PRIOR=M.PRIOR+((OLD,OLD_SHA),)
M.MAX_SECONDS=10*60


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args();assert not args.out.exists();start=time.monotonic();stage='completed_candidate_bindings'
    try:
        for path,sha in ((DEVELOPMENT,DEVELOPMENT_SHA),(D.EXPORT,D.EXPORT_SHA),(D.CORE,D.CORE_SHA),(OLD,OLD_SHA)):
            assert M.digest(path)==sha
        development=json.loads(DEVELOPMENT.read_text(encoding='utf-8'));assert all(development['gates'].values())
        assert development['decision']=='complete_core_development_pass_freeze_new_independent_source_manifest'
        export=json.loads(D.EXPORT.read_text(encoding='utf-8'));assert all(export['gates'].values())
        assert M.digest(Path(D.R.__file__))==export['script_sha256']
        for path,sha in export['helper_sha256'].items():assert M.digest(Path(path))==sha
        old=json.loads(OLD.read_text(encoding='utf-8'));assert old['pg19_parquet']==M.PG19_REL and old['pg19_parquet_sha256']==PG_SHA
        assert subprocess.check_output(['git','rev-parse',M.REF],cwd=M.ROOT,text=True).strip()==M.REF==old['source_commit']
        stage='pinned_cached_parquet_and_tokenizer';assert M.digest(M.ROOT/M.PG19_REL)==PG_SHA
        tokenizer=M.AutoTokenizer.from_pretrained(M.M42.MODEL,revision=M.M42.REV,local_files_only=True)
        assert M.M15.M13.C.tok_fingerprint(tokenizer)==M.M15.M13.TOK_FP
        stage='all_prior_source_fragment_exclusion';excluded,base,priors=M.prior_context(tokenizer);initial=len(excluded)
        assert all(item['source_id'] in excluded for item in old['items'])
        print(json.dumps({'stage':stage,'excluded_sources':initial,'runtime':M.budget(start)}),flush=True)
        stage='frozen_hash_rank_selection';items,provenance,rows=M.select(tokenizer,excluded,base,start)
        assert rows==old['pg19_rows']==1243 and len(items)==24
        assert all(item['source_id'] not in {p['source_id'] for p in old['items']} for item in items)
        for item in items:
            if item['category']=='prose':item['source_kind']='pg19_train14'
        result={'experiment':'METH-261-complete-unique-core-independent-source-manifest',
            'development_result_sha256':DEVELOPMENT_SHA,'export_result_sha256':D.EXPORT_SHA,'artifact_sha256':D.CORE_SHA,
            'source_sha256':D.P.M57.MODEL_SHA,'parent_checkpoint_sha256':D.P.M57.CHECKPOINT_SHA,
            'child_checkpoint_sha256':D.P.M122.SPECIALIZED_SHA,'model':M.M42.MODEL,'revision':M.M42.REV,
            'tokenizer_fingerprint':M.M15.M13.TOK_FP,'source_commit':M.REF,
            'pg19_parquet':M.PG19_REL,'pg19_parquet_sha256':PG_SHA,'pg19_rows':rows,
            'prior_manifest_sha256':priors,'excluded_source_count':initial,'h0_calib_sha256':M.M17.CALIB_SHA,
            'seed':M.SEED,'selected_counts':M.COUNTS,'chat_template':M.M42.REQUEST,'excerpt_characters':384,
            'selection_provenance':provenance,'items':items,'runtime':M.budget(start),
            'script_sha256':M.digest(Path(__file__)),'decision':'new_source_manifest_frozen_require_answerability_before_model_scores',
            'scope':'Model-free source/fragment exclusion against prior transfer fit/evaluation cohorts including213;no claim exclusion from unknown original donor pretraining. No quality inference,source rejection by model scores,n/RAM/native rate or other-donor promotion.'}
        args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(json.dumps({'decision':result['decision'],'manifest_sha256':M.digest(args.out),'counts':M.COUNTS,
            'source_ids':[r['source_id'] for r in items],'runtime':result['runtime']}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':type(error).__name__+': '+str(error),
            'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8');raise


if __name__=='__main__':main()
