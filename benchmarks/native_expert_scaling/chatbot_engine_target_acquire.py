"""Acquire bounded public config metadata and bind local evidence; never weights."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
SITE = ROOT / 'results/native_expert_scaling/chatbot_source_runtime/site'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def record(path):
    raw = path.read_bytes()
    if len(raw) > 5 << 20:
        raise ValueError('metadata extent exceeds5MiB: ' + str(path))
    return {'path': str(path.absolute()), 'bytes': len(raw), 'sha256': digest(raw)}


def create(path, raw):
    with path.open('xb') as f:
        f.write(raw)


def get(url, start):
    if time.monotonic() - start > 100:
        raise TimeoutError('acquisition budget exceeded')
    req = urllib.request.Request(url, headers={'User-Agent': 'SiliconLLM-metadata-review'})
    with urllib.request.urlopen(req, timeout=20) as response:
        raw = response.read((5 << 20) + 1)
    if len(raw) > 5 << 20:
        raise ValueError('public metadata exceeds5MiB')
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--directory', required=True, type=Path)
    args = ap.parse_args()
    out = args.directory.absolute()
    out.relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    sources = []
    local = [
        ('Qwen/Qwen2.5-0.5B-Instruct', '7ae557604adf67be50417f59c2c2f167def9a775',
         Path('C:/Users/giosa/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/config.json'),
         DOC/'chatbot_qwen_operator_census_20261007.json'),
        ('ai-sage/GigaChat3.1-10B-A1.8B-bf16', '189fff27a1dee68473960c3d5bca53e0e07a3191',
         ROOT/'benchmarks/donor_adaptation/density/results/strat01_gigachat_source_189fff27/config.json',
         DOC/'chatbot_gigachat_operator_census_repair1_20261007.json'),
        ('ibm-granite/granite-4.0-h-small', 'b8c0982bab7fde4eb48110f5a069527c008fab39',
         Path('C:/Users/giosa/.cache/huggingface/hub/models--ibm-granite--granite-4.0-h-small/snapshots/b8c0982bab7fde4eb48110f5a069527c008fab39/config.json'), None),
    ]
    for model, revision, config, census in local:
        sources.append({'model': model, 'revision': revision, 'revision_scope': 'local snapshot/declaration',
                        'config': record(config), 'retained_census': record(census) if census else None})
    for label, model, pinned in [
        ('falcon', 'tiiuae/Falcon-H1-Tiny-90M-Instruct', 'e6389502a0b12cd8da894b395ba5bf7436873b16'),
        ('bitnet', 'microsoft/bitnet-b1.58-2B-4T-bf16', None),
    ]:
        api_url = 'https://huggingface.co/api/models/' + model
        api_raw = get(api_url, start)
        api = json.loads(api_raw)
        revision = pinned or api['sha']
        if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
            raise ValueError('invalid revision')
        create(out/(label+'.api.json'), api_raw)
        url = 'https://huggingface.co/' + model + '/resolve/' + revision + '/config.json'
        raw = get(url, start)
        json.loads(raw)
        path = out/(label+'.config.json')
        create(path, raw)
        sources.append({'model': model, 'revision': revision, 'revision_scope': 'pinned public metadata URL',
                        'url': url, 'config': record(path), 'api': record(out/(label+'.api.json')),
                        'retained_census': None})
    support = [ROOT/'benchmarks/phase60/engine.c', DOC/'chatbot_engine_target_contract_v1.json',
               DOC/'CHATBOT_ENGINE_TARGET_TRIAGE_PROTOCOL_20261008.md',
               Path(__file__).with_name('chatbot_engine_target_triage.py')]
    for family in ('falcon_h1', 'bitnet', 'granitemoehybrid'):
        support.append(SITE/f'transformers/models/{family}/modeling_{family}.py')
    manifest = {'schema': 'engine-target-triage-inputs-v1',
                'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'sources': sources, 'support': [record(p) for p in support],
                'acquirer': record(Path(__file__)), 'seconds': time.monotonic()-start,
                'scope': 'metadata only, no tensor values/model import/inference',
                'partial_policy': 'created directory and downloaded prefixes retained on failure; never overwrite'}
    create(out/'inputs.json', (json.dumps(manifest, indent=2)+'\n').encode())
    print(json.dumps({'inputs': str(out/'inputs.json'), 'sha256': digest((out/'inputs.json').read_bytes()),
                      'sources': len(sources), 'seconds': manifest['seconds']}))


if __name__ == '__main__':
    main()
