#!/usr/bin/env python3
"""Validate source-only annotations before native-primary model scoring."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
MANIFEST=DOC/'meth292_native_primary_manifest.json'
MANIFEST_SHA='99fdb09fe13e8abf8c401718911d08137851aa90c043415dd92eeb7d619f3847'
ARTIFACT_SHA='4f9b9c7a76475d9b6e241947ee590884ea268d4bf7f02097df57ad4b2ac23fe9'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest-sha',required=True)
    for name in ('annotations','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    assert args.manifest_sha==MANIFEST_SHA
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists()
    try:
        assert digest(MANIFEST)==args.manifest_sha
        manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
        assert manifest['experiment']=='METH-292-native-primary-independent-source-manifest'
        assert manifest['artifact_sha256']==ARTIFACT_SHA and manifest['excluded_source_count']==3224
        assert manifest['diagnostic_only'] is True and manifest['native_promotion_qualified'] is False
        annotations=json.loads(args.annotations.read_text(encoding='utf-8'));rows=annotations['rows']
        assert len(rows)==len(manifest['items'])==24
        assert annotations['manifest_sha256']==args.manifest_sha and annotations['model_outputs_consulted'] is False
        assert len({row['source_id'] for row in rows})==24
        for row,item in zip(rows,manifest['items']):
            assert row['source_id']==item['source_id'] and row['category']==item['category']
            assert type(row['answerable']) is bool and isinstance(row['basis'],str) and len(row['basis'])>=12
            if row['answerable']:
                assert isinstance(row['anchor'],str) and 4<=len(row['anchor'])<=128 and row['anchor'] in item['excerpt']
                assert isinstance(row['supported_summary'],str) and 12<=len(row['supported_summary'])<=240
                assert isinstance(row['supported_detail'],str) and 8<=len(row['supported_detail'])<=240
            else:assert row['anchor'] is None
        count=sum(row['answerable'] for row in rows)
        result={'experiment':'METH-293-native-primary-source-only-answerability',
            'manifest_sha256':args.manifest_sha,'artifact_sha256':ARTIFACT_SHA,
            'annotations_sha256':digest(args.annotations),'model_outputs_consulted':False,
            'answerable_count':count,'rows':rows,
            'gates':{'all24_sources_preserved_and_annotated':True,
                'all_positive_literal_anchors_in_visible_excerpt':True,
                'all24_prompts_answerable_from_source_only':count==24},
            'script_sha256':digest(Path(__file__)),'diagnostic_only':True,'native_promotion_qualified':False,
            'decision':'source_answerability_pass_freeze_native_primary_prediction' if count==24 else 'source_answerability_incomplete_preserve_sources_hold_generation',
            'scope':'Source-only annotations,not model output correctness. No source replacement/model-driven anchor,native-rate/useful-n/family proof.'}
        args.out.write_text(json.dumps(result,indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
        print(json.dumps({key:result[key] for key in ('answerable_count','gates','decision')}),flush=True)
    except BaseException as failure:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'error':type(failure).__name__+': '+str(failure),
            'script_sha256':digest(Path(__file__))},indent=2)+'\n',encoding='utf-8');raise


if __name__=='__main__':main()
