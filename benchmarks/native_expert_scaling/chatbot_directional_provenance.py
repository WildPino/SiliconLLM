"""Repair a mislabeled revision by retained Git/file bytes, never repeat anchors."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once


def main():
    doc=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
    commit='ec197f9efdf1b38dfdc6f8a958e2203a354ee6c8'
    binding_path=doc/'chatbot_directional_plan_binding_20261008.json'
    raw_path=doc/'chatbot_directional_plan_20261008.json';terminal_path=raw_path.with_suffix('.terminal.json')
    binding=json.loads(binding_path.read_bytes());raw=json.loads(raw_path.read_bytes());terminal=json.loads(terminal_path.read_bytes())
    assert raw['source_freeze']==terminal['source_freeze']=='9bbeef-placeholder'
    assert terminal['actual_worker_exit_code']==0 and sha(raw_path)==terminal['result_sha256'] and all(terminal['gates'].values())
    rows=[]
    paths=[binding_path,*(Path(v['path']) for v in binding['inputs'] if ROOT in Path(v['path']).parents and 'results' not in Path(v['path']).relative_to(ROOT).parts)]
    for path in dict.fromkeys(paths):
        rel=path.relative_to(ROOT).as_posix()
        blob=subprocess.run(['git','show',commit+':'+rel],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout
        item=(dict(sha256=sha(binding_path),bytes=binding_path.stat().st_size) if path==binding_path else
            next(v for v in binding['inputs'] if Path(v['path'])==path))
        variants=[blob,blob.replace(b'\r\n',b'\n'),blob.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')]
        assert any(len(v)==item['bytes'] and hashlib.sha256(v).hexdigest()==item['sha256'] for v in variants),rel
        rows.append(dict(path=rel,bound_physical_sha256=item['sha256'],Git_blob_sha256=hashlib.sha256(blob).hexdigest(),
            exact_bound_bytes_reconstructed_from_commit_with_only_CRLF_conversion=True))
    write_once(doc/'chatbot_directional_plan_provenance_20261008.json',dict(
        scope='ADMINISTRATIVE_RETAINED_BYTES_REVISION_LABEL_CORRECTION_NO_ANCHOR_REPLAY',
        original_revision_field_VALID=False,original_revision_field=raw['source_freeze'],
        actual_source_freeze=commit,binding_sha256=sha(binding_path),result_sha256=sha(raw_path),terminal_sha256=sha(terminal_path),
        committed_actual_bound_files=rows,raw_and_terminal_preserved=True,new_plan_values=0,new_original_BF16_full_forwards=0))
    print(json.dumps(dict(actual_source_freeze=commit,files=len(rows),new_plan_values=0)))


if __name__=='__main__':main()
