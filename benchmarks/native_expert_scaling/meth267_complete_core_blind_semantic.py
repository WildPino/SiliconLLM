#!/usr/bin/env python3
"""Make an anonymous three-arm panel, then score frozen excerpt-only findings."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
MANIFEST=DOC/'meth261_complete_core_fresh_manifest.json'
MANIFEST_SHA='5af457561ad24fb837770735d0dbd0f23baa2f04a0525f38656285c0db7ec050'
ANCHORS=DOC/'meth262_source_answerability_annotations.json'
ANCHORS_SHA='e6e031f674f2b795e30da314e80596fc1e1589149a8d4bae6879dd4199c516bd'
ARMS=('bf16_donor','bf16_e1280','stored_unique_e1280')
METRICS=('unsupported','severe','missing_detail')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):
    assert not path.exists()
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def blind(args):
    assert digest(args.generation)==args.generation_sha
    gen=read(args.generation)
    assert gen['decision']=='full_prefix_generation_health_pass_freeze_task_and_blind' and all(gen['gates'].values())
    assert digest(MANIFEST)==MANIFEST_SHA and digest(ANCHORS)==ANCHORS_SHA
    manifest=read(MANIFEST);anchors=read(ANCHORS)
    assert gen['manifest_sha256']==MANIFEST_SHA and len(manifest['items'])==24
    assert set(gen['arms'])==set(ARMS)
    bank={arm:{r['source_id']:r for r in gen['arms'][arm]} for arm in ARMS}
    assert all(len(rows)==24 for rows in bank.values())
    rows=[];mapping=[];rng=secrets.SystemRandom()
    for item in manifest['items']:
        order=list(ARMS);rng.shuffle(order);sid=item['source_id']
        rows.append({'source_id':sid,'category':item['category'],'excerpt':item['excerpt'],
            **{side:bank[arm][sid]['continuation_text'] for side,arm in zip('ABC',order)}})
        mapping.append({'source_id':sid,**dict(zip('ABC',order))})
    write(args.mapping,{'generation_sha256':args.generation_sha,'rows':mapping})
    write(args.out,{'experiment':'METH-267-three-arm-anonymous-excerpt-panel',
        'generation_sha256':args.generation_sha,'manifest_sha256':MANIFEST_SHA,
        'anchors_sha256':ANCHORS_SHA,'mapping_sha256':digest(args.mapping),
        'rubric':'Excerpt-only unsupported claims,severe central inversions,and absence of any concrete supported detail. Arm identities withheld until findings commit.',
        'rows':rows})
    print(json.dumps({'rows':len(rows),'responses':72,'panel_sha256':digest(args.out),'mapping_sha256':digest(args.mapping)}))


def score(args):
    panel=read(args.panel);verdict=read(args.verdict);mapping=read(args.mapping)
    assert digest(args.mapping)==panel['mapping_sha256']
    assert verdict['panel_sha256']==digest(args.panel)
    assert mapping['generation_sha256']==panel['generation_sha256']
    assert [r['source_id'] for r in verdict['rows']]==[r['source_id'] for r in panel['rows']]==[r['source_id'] for r in mapping['rows']]
    assert len(verdict['rows'])==24
    counts={arm:{key:0 for key in (*METRICS,'ambiguous')} for arm in ARMS}
    paired=[]
    for row,visible,key in zip(verdict['rows'],panel['rows'],mapping['rows']):
        assert set(key[side] for side in 'ABC')==set(ARMS)
        pair={'source_id':row['source_id'],'category':visible['category']}
        for side in 'ABC':
            finding=row[side];arm=key[side];assert isinstance(finding['missing_detail'],bool)
            assert isinstance(finding['unsupported'],list) and isinstance(finding['ambiguous'],list)
            assert isinstance(finding['detail_evidence'],str) and finding['detail_evidence'].strip()
            for claim in finding['unsupported']:
                assert claim['claim'].strip() and claim['evidence'].strip() and isinstance(claim['severe'],bool)
            for claim in finding['ambiguous']:assert claim['claim'].strip() and claim['evidence'].strip()
            counts[arm]['unsupported']+=len(finding['unsupported'])
            counts[arm]['severe']+=sum(c['severe'] for c in finding['unsupported'])
            counts[arm]['missing_detail']+=int(finding['missing_detail'])
            counts[arm]['ambiguous']+=len(finding['ambiguous']);pair[arm]=finding
        paired.append(pair)
    gates={control+'_'+metric:counts[ARMS[2]][metric]<=counts[control][metric]
           for control in ARMS[:2] for metric in METRICS}
    write(args.out,{'experiment':'METH-267-fixed-complete-core-unblinded-semantic-score',
        'panel_sha256':digest(args.panel),'verdict_sha256':digest(args.verdict),
        'mapping_sha256':digest(args.mapping),'generation_sha256':panel['generation_sha256'],
        'counts':counts,'gates':gates,'paired_rows':paired,
        'decision':'anonymous_semantic_pass_pending_joint_native_gate' if all(gates.values()) else 'anonymous_semantic_fail_close_fixed_candidate',
        'scope':'Single-agent arm-anonymous finite excerpt review,not independent human review or broad factual capability certificate. Native production quality/rate and useful RAM-scale capacity still required.'})
    print(json.dumps({'counts':counts,'gates':gates,'result_sha256':digest(args.out)}))


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='mode',required=True)
    b=sub.add_parser('blind')
    for name in ('generation','mapping','out'):b.add_argument('--'+name,type=Path,required=True)
    b.add_argument('--generation-sha',required=True)
    s=sub.add_parser('score')
    for name in ('panel','verdict','mapping','out'):s.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();(blind if args.mode=='blind' else score)(args)


if __name__=='__main__':main()
