#!/usr/bin/env python3
"""Model-free fresh source selection for the diagnostic complete I16 archive."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import torch
import meth261_complete_core_fresh_manifest as S
import meth277_complete_i16_development as D

M=S.M
DEVELOPMENT=M.DOC/'meth277_complete_i16_development_result.json'
DEVELOPMENT_SHA='3d997545fc690a5ebb3681e60fb0d3dc190a2a292e94056e5cf0abcb9ed88b35'
OLD=M.DOC/'meth261_complete_core_fresh_manifest.json'
OLD_SHA='5af457561ad24fb837770735d0dbd0f23baa2f04a0525f38656285c0db7ec050'
M.SEED='meth278-complete-i16-independent-278278'
M.PRIOR=M.PRIOR+((OLD,OLD_SHA),)
M.MAX_SECONDS=600


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();start=time.monotonic();stage='bindings'
    assert all(not p.exists() for p in (args.out,args.out.with_suffix('.failure.json')))
    torch.set_num_threads(6)
    try:
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
        assert initial==old['excluded_source_count']+24==3200
        for path in (OLD,S.OLD):
            previous=json.loads(path.read_text(encoding='utf-8'))
            assert all(item['source_id'] in excluded for item in previous['items'])
        print(json.dumps({'stage':stage,'excluded_source_count':initial,'runtime':M.budget(start)}),flush=True)
        stage='fixed_hash_rank_source_selection';items,provenance,rows=M.select(tokenizer,excluded,base,start)
        assert rows==old['pg19_rows']==1243 and len(items)==24
        assert len({item['source_id'] for item in items})==24
        assert all(item['source_id'] not in {old_item['source_id'] for old_item in old['items']} for item in items)
        for item in items:
            if item['category']=='prose':item['source_kind']='pg19_train14'
        result={'experiment':'METH-278-complete-I16-private128-independent-source-manifest',
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
            'scope':'New source IDs/project-transfer heldout,including consumed261 source exclusions;not a new corpus or certified donor-pretraining exclusion. No model inference/source rejection by scores or quality/rate/useful-n/family promotion.'}
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
