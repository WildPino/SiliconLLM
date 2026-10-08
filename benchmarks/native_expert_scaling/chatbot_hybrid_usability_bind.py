"""Bind the larger acquired Falcon source without inheriting Tiny contracts."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from chatbot_falcon_usability import sha,write


def main(a):
    source=a.source.resolve()
    package=json.loads((source/'source_package.json').read_bytes())
    assert package['schema']=='HYBRID_ORIGINAL_SOURCE_PACKAGE_V1'
    files=[Path(v['path']) for v in package['files']]+[source/'source_package.json']
    files += [Path(v['path']) for v in package['headers']]
    files += [B/v for v in ('chatbot_hybrid_source.py','chatbot_falcon_usability.py','chatbot_falcon_usability_launch.py',
         'chatbot_hybrid_usability_bind.py','chatbot_falcon_usability_cases_v1.json')]
    files += [DOC/'CHATBOT_HYBRID_TEACHER_PROTOCOL_20261008.md',Path(sys.executable)]
    site=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
    files += [site/'transformers'/v for v in ('models/falcon_h1/modeling_falcon_h1.py',
       'models/falcon_h1/configuration_falcon_h1.py','cache_utils.py','modeling_utils.py','generation/utils.py',
       'generation/configuration_utils.py','integrations/hub_kernels.py','utils/import_utils.py',
       'tokenization_utils_base.py','tokenization_utils_tokenizers.py','utils/chat_template_utils.py')]
    foreign={
       'benchmarks/donor_adaptation/configs/_manifest.json':'fcb168f0d2004baf6c0f2938f997e095c9210e72b004c0a561add8608500e35d',
       'benchmarks/donor_adaptation/density/build_document_holdout.py':'c5c9ed40864989592664c15790e69e41a5ebbad73d00ee7ac067d98e2baf15e9',
       'docs/research/RESEARCH_INDEX.md':'99b5b11c865d000f387c17120b918458aa8cac8263628d7d040abddcd7eb5273'}
    files += [ROOT/v for v in foreign]
    assert all(sha(ROOT/v)==want for v,want in foreign.items())
    assert all(sha(v['path'])==v['sha256'] for v in package['files'])
    result=dict(schema='HYBRID_USABILITY_BINDING_V1',python=str(Path(sys.executable).resolve()),
       source_directory=str(source),source_revision=package['revision'],source_named_elements=package['total_named_elements'],
       cases_path=str((B/'chatbot_falcon_usability_cases_v1.json').resolve()),worker_path=str((B/'chatbot_falcon_usability.py').resolve()),
       limits=dict(OS_bytes=8<<30,GPU_allocated_bytes=9<<30,GPU_reserved_bytes=10<<30),
       runtime_binding_scope='Selected Transformers files and Python executable hashed; exact package versions/paths in isolated view, full native DLL/runtime trees NOT rehashed',
       plain_contract='Original BOS + role/newline/content/im_end/newline, no extra assistant-history newline, both producer EOS',
       inputs=[dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    write(a.out,result)
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(files))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
