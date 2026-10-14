"""Validate documentation descriptors without rerunning scientific experiments."""
from pathlib import Path
import datetime
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[4]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
SNAP = Path(__file__).resolve().parent
DRAFT = ROOT / 'HANDOFF.next.md'
if not DRAFT.exists():
    DRAFT = ROOT / 'HANDOFF.md'
data = DRAFT.read_bytes()
body = data.decode('utf-8')
paths = [p for p in DOC.iterdir() if p.is_file()]
paths += list((ROOT / 'benchmarks/native_expert_scaling').iterdir())
paths += [ROOT / 'engine.c', ROOT / 'nibble_pack.h']
known = {p.name for p in paths}
mentions = set(re.findall(r'`([^`\n]+\.(?:md|py|json|c|h))`', body))
missing = []
resolved = []
for mention in sorted(mentions):
    if '*' in mention or mention.startswith('*.'):
        continue
    name = mention.replace('\\', '/').rsplit('/', 1)[-1]
    if name in known or (ROOT / mention).exists() or (SNAP / name).exists():
        resolved.append(mention)
    else:
        missing.append(mention)
assert len(body.splitlines()) > 700
assert body.count('```') % 2 == 0
assert not re.search(r'\bTODO\b|for brevity|rest of code', body)
rp = DOC / 'ffn_active_router_result_20261014.json'
report = json.loads(rp.read_bytes())
tp = rp.with_suffix('.terminal.json')
terminal = json.loads(tp.read_bytes())
descriptor = dict(path=str(rp), bytes=rp.stat().st_size,
                  sha256=hashlib.sha256(rp.read_bytes()).hexdigest())
assert terminal['exit_code'] == 0 and terminal['result'] == descriptor
assert descriptor['sha256'] in body
assert len(report['records']) == 576 and len(report['models']) == 3
snapshots = []
for p in sorted(SNAP.iterdir()):
    if p.is_file() and p.name not in ('handoff_document_verification.json',):
        snapshots.append(dict(path=str(p), bytes=p.stat().st_size,
                              sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
result = dict(observed_at=datetime.datetime.now().astimezone().isoformat(),
              document=dict(path=str(DRAFT), bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest(),
                            lines=len(body.splitlines()), headings=len(re.findall(r'^#{1,3} ', body, re.M))),
              producer_descriptor=descriptor,
              producer_terminal_exit=terminal['exit_code'],
              inline_file_references_resolved=resolved,
              inline_file_references_needing_explanation=missing,
              snapshots=snapshots,
              scope='Documentation validation only. No new numeric qualification, no full E artifact hash audit, no model run.')
out = SNAP / 'handoff_document_verification.json'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k not in ('snapshots', 'inline_file_references_resolved')}, ensure_ascii=False, indent=2))
