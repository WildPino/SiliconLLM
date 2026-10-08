"""Bind the acquired source, cases and selected runtime implementation files."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
B = ROOT / 'benchmarks/native_expert_scaling'


def main(out, kind):
    source = ROOT / 'results/native_expert_scaling/falcon_source_e6389502_20261008'
    site = ROOT / 'results/native_expert_scaling/chatbot_source_runtime/site'
    package = json.loads((source / 'source_package.json').read_bytes())
    files = [Path(v['path']) for v in package['files']] + [source / 'source_package.json', source / 'validated_header.json']
    files += [B / v for v in ('chatbot_falcon_usability.py', 'chatbot_falcon_usability_launch.py',
         'chatbot_falcon_usability_bind.py', 'chatbot_falcon_usability_cases_v1.json')]
    files += [DOC / 'CHATBOT_FALCON_SOURCE_PROTOCOL_20261008.md', Path(sys.executable)]
    files += [site / 'transformers' / v for v in ('models/falcon_h1/modeling_falcon_h1.py',
         'models/falcon_h1/configuration_falcon_h1.py', 'cache_utils.py', 'modeling_utils.py',
         'generation/utils.py', 'generation/configuration_utils.py', 'integrations/hub_kernels.py',
         'utils/import_utils.py', 'tokenization_utils_base.py', 'tokenization_utils_tokenizers.py')]
    foreign = {
      'benchmarks/donor_adaptation/configs/_manifest.json': 'fcb168f0d2004baf6c0f2938f997e095c9210e72b004c0a561add8608500e35d',
      'benchmarks/donor_adaptation/density/build_document_holdout.py': 'c5c9ed40864989592664c15790e69e41a5ebbad73d00ee7ac067d98e2baf15e9',
      'docs/research/RESEARCH_INDEX.md': '99b5b11c865d000f387c17120b918458aa8cac8263628d7d040abddcd7eb5273'}
    files += [ROOT / v for v in foreign]
    if kind == 'scan':
        files += [B / v for v in ('chatbot_falcon_scan_capture.py', 'chatbot_falcon_scan.c', 'chatbot_falcon_scan_compare.py')]
        files += [ROOT / 'benchmarks/phase60/engine.c', DOC / 'CHATBOT_FALCON_SCAN_PROTOCOL_20261008.md']
    def sha(p):
        h = hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''):
                h.update(b)
        return h.hexdigest()
    for v in package['files']:
        assert sha(Path(v['path'])) == v['sha256'] and Path(v['path']).stat().st_size == v['bytes']
    assert all(sha(ROOT / p) == want for p, want in foreign.items())
    result = dict(schema='FALCON_USABILITY_BINDING_V1', python=str(Path(sys.executable).resolve()),
        source_directory=str(source.resolve()), source_revision=package['revision'],
        cases_path=str((B / 'chatbot_falcon_usability_cases_v1.json').resolve()),
        worker_path=str((B / 'chatbot_falcon_usability.py').resolve()),
        runtime_binding_scope='Selected Transformers source files, Python executable, exact package version/import-path checks in isolated local view; full native DLL/runtime tree NOT rehashed',
        inputs=[dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=sha(p)) for p in files])
    if kind == 'scan':
        result.update(schema='FALCON_SCAN_BRIDGE_BINDING_V1', worker_path=str((B/'chatbot_falcon_scan_capture.py').resolve()),
            scan_probe=dict(prompt='What is 8 plus 6? Explain the calculation briefly.', layers=[0,12,23],
                            prefill_calls=1,cached_steps=3),
            decision_limits=dict(per_packet_state_relative_RMS=.01, per_packet_gated_output_relative_RMS=.01))
    with out.open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(binding=str(out), sha256=sha(out), inputs=len(files))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--kind', choices=['usability','scan'], default='usability')
    args=p.parse_args()
    main(args.out,args.kind)
