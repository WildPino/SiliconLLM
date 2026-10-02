#!/usr/bin/env python3
"""Validate source-only annotations before any new-manifest model inference."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
MANIFEST=DOC/'meth261_complete_core_fresh_manifest.json'
MANIFEST_SHA='5af457561ad24fb837770735d0dbd0f23baa2f04a0525f38656285c0db7ec050'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    for name in ('annotations','out'):ap.add_argument('--'+name,required=True,type=Path)
    args=ap.parse_args();assert not args.out.exists();assert digest(MANIFEST)==MANIFEST_SHA
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    annotations=json.loads(args.annotations.read_text(encoding='utf-8'));rows=annotations['rows']
    assert len(rows)==len(manifest['items'])==24
    assert annotations['manifest_sha256']==MANIFEST_SHA and annotations['model_outputs_consulted'] is False
    assert len({r['source_id'] for r in rows})==24
    for row,item in zip(rows,manifest['items']):
        assert row['source_id']==item['source_id'] and row['category']==item['category']
        assert type(row['answerable']) is bool and isinstance(row['basis'],str) and len(row['basis'])>=12
        if row['answerable']:
            assert isinstance(row['anchor'],str) and 4<=len(row['anchor'])<=128 and row['anchor'] in item['excerpt']
            assert isinstance(row['supported_summary'],str) and 12<=len(row['supported_summary'])<=240
            assert isinstance(row['supported_detail'],str) and 8<=len(row['supported_detail'])<=240
        else:assert row['anchor'] is None
    count=sum(r['answerable'] for r in rows)
    result={'experiment':'METH-262-source-only-pre-inference-answerability',
        'manifest_sha256':MANIFEST_SHA,'annotations_sha256':digest(args.annotations),
        'model_outputs_consulted':False,'answerable_count':count,'rows':rows,
        'gates':{'all24_sources_preserved_and_annotated':True,'all_positive_literal_anchors_in_visible_excerpt':True,
                 'all24_prompts_answerable_from_source_only':count==24},
        'script_sha256':digest(Path(__file__)),
        'decision':'answerability_pass_freeze_independent_full_prediction' if count==24 else 'answerability_incomplete_preserve_sources_hold_generation',
        'scope':'Source-only annotations,not model correctness,generation or document quality. No source replacement,model-driven anchor selection,n/RAM/native rate or donor scale claim.'}
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('answerable_count','gates','decision')}))


if __name__=='__main__':main()
