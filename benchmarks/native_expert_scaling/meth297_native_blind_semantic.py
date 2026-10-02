#!/usr/bin/env python3
"""Anonymous new-source complete I16 semantic review with committed findings before unblinding."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import subprocess

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
MANIFEST=DOC/'meth292_native_primary_manifest.json'
MANIFEST_SHA='99fdb09fe13e8abf8c401718911d08137851aa90c043415dd92eeb7d619f3847'
ANCHORS=DOC/'meth293_native_source_answerability_annotations.json'
ANCHORS_SHA='9ceb199045da48152eae40784139fd78007550578f2e1e3d070298f317dad21e'
ARTIFACT_SHA='4f9b9c7a76475d9b6e241947ee590884ea268d4bf7f02097df57ad4b2ac23fe9'
ARMS=('bf16_donor','bf16_e1280','native_i16_private128_e1280')
METRICS=('unsupported','severe','missing_detail')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):
    assert not path.exists()
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def blind(args):
    assert digest(args.generation)==args.generation_sha
    gen=read(args.generation)
    assert gen['decision']=='native_generation_health_pass_freeze_anonymous_semantics' and all(gen['gates'].values())
    assert digest(MANIFEST)==MANIFEST_SHA and digest(ANCHORS)==ANCHORS_SHA
    manifest=read(MANIFEST);anchors=read(ANCHORS)
    assert gen['manifest_sha256']==MANIFEST_SHA and len(manifest['items'])==24
    assert gen['experiment']=='METH-296-actual-native-complete-bank-generation-health-and-state-capture'
    assert gen['artifact_sha256']==ARTIFACT_SHA and gen['annotations_sha256']==ANCHORS_SHA
    assert gen['diagnostic_only'] is True and gen['native_promotion_qualified'] is False
    assert gen['native_runtime']['loader']['fields_bound']==725
    assert gen['native_runtime']['loader']['fallback'] is False
    assert gen['native_runtime']['loader']['archive_sha256']==ARTIFACT_SHA
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
    write(args.out,{'experiment':'METH-297-new-source-I16-three-arm-anonymous-excerpt-panel',
        'generation_sha256':args.generation_sha,'manifest_sha256':MANIFEST_SHA,
        'anchors_sha256':ANCHORS_SHA,'mapping_sha256':digest(args.mapping),
        'artifact_sha256':ARTIFACT_SHA,'script_sha256':digest(Path(__file__)),
        'diagnostic_only':True,'native_promotion_qualified':False,
        'rubric':'Excerpt-only unsupported claims,severe central inversions,and absence of any concrete supported detail. Arm identities withheld until findings commit.',
        'rows':rows})
    print(json.dumps({'rows':len(rows),'responses':72,'panel_sha256':digest(args.out),'mapping_sha256':digest(args.mapping)}))


def score(args):
    # Verify committed findings before accessing the arm map.
    for path in (args.verdict,args.panel):
        rel=path.resolve().relative_to(ROOT.resolve()).as_posix()
        blob=subprocess.check_output(['git','-c','core.autocrlf=false','cat-file','--filters','--path='+rel,'HEAD:'+rel],cwd=ROOT)
        assert blob==path.read_bytes(),str(path)+' must be committed before unblinding'
    panel=read(args.panel);verdict=read(args.verdict)
    assert panel['artifact_sha256']==ARTIFACT_SHA and panel['manifest_sha256']==MANIFEST_SHA
    assert panel['script_sha256']==digest(Path(__file__))
    # Validate ALL72 complete findings before any arm-map access.
    assert verdict['panel_sha256']==digest(args.panel)
    assert len(verdict['rows'])==len(panel['rows'])==24
    assert [r['source_id'] for r in verdict['rows']]==[r['source_id'] for r in panel['rows']]
    for row in verdict['rows']:
        for side in 'ABC':
            finding=row[side]
            assert isinstance(finding['missing_detail'],bool)
            assert isinstance(finding['unsupported'],list) and isinstance(finding['ambiguous'],list)
            assert isinstance(finding['detail_evidence'],str) and finding['detail_evidence'].strip()
            for claim in finding['unsupported']:
                assert claim['claim'].strip() and claim['evidence'].strip() and isinstance(claim['severe'],bool)
            for claim in finding['ambiguous']:assert claim['claim'].strip() and claim['evidence'].strip()
    mapping=read(args.mapping)
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
    write(args.out,{'experiment':'METH-297-fixed-complete-I16-unblinded-semantic-score',
        'panel_sha256':digest(args.panel),'verdict_sha256':digest(args.verdict),
        'mapping_sha256':digest(args.mapping),'generation_sha256':panel['generation_sha256'],
        'counts':counts,'gates':gates,'paired_rows':paired,
        'artifact_sha256':ARTIFACT_SHA,'manifest_sha256':MANIFEST_SHA,'anchors_sha256':ANCHORS_SHA,
        'script_sha256':digest(Path(__file__)),'findings_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'diagnostic_only':True,'native_promotion_qualified':False,
        'decision':'anonymous_semantic_pass_requires_PIQA_then_joint_native_gate' if all(gates.values()) else 'anonymous_semantic_fail_close_fixed_native_profile',
        'scope':'Single-agent arm-anonymous finite new-source review,not independent human/broad capability proof. Actual native prediction/health already pass; same-artifact PIQA/CPU K64/accepted rate/useful RAM-scale n/DRAM/family requirements remain;old fixed stops retained.'})
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
