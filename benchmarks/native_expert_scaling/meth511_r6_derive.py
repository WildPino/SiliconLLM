"""Metadata-only literal copies; retain R3 cohort and dynamically bind idle identities."""
import ast
import hashlib
import json
from pathlib import Path

B = Path(__file__).resolve().parent
records = []

def derive(old, new, replacements):
    text = old.read_text(encoding='utf8')
    changes = []
    for before, after in replacements:
        count = text.count(before)
        assert count > 0, (old.name, before)
        text = text.replace(before, after)
        changes.append({'before': before, 'after': after, 'occurrences': count})
    if new.suffix == '.py':
        ast.parse(text, filename=str(new))
    with new.open('x', encoding='utf8', newline='\n') as f:
        f.write(text)
    records.append({'old': str(old), 'old_sha256': hashlib.sha256(old.read_bytes()).hexdigest(),
                    'new': str(new), 'new_sha256': hashlib.sha256(new.read_bytes()).hexdigest(),
                    'replacements': changes})

for suffix in ['operations', 'native', 'donor', 'retention_audit']:
    replacements = [('meth511_r5_binding.json', 'meth511_r6_binding.json'),
                    ("('meth511_r5_'+kind)", "('meth511_r6_'+kind)"),
                    ("'meth511_r5_'", "'meth511_r6_'")] if suffix == 'operations' else [
                        ('import meth511_r5_operations as O', 'import meth511_r6_operations as O')]
    derive(B / ('meth511_r5_' + suffix + '.py'), B / ('meth511_r6_' + suffix + '.py'), replacements)

derive(B / 'meth511_r5_prepare_binding.py', B / 'meth511_r6_prepare_binding.py', [
    ('import meth511_r5_operations as O', 'import meth511_r6_operations as O'),
    ('meth511_r5*', 'meth511_r6*'),
    ("startswith('meth511_r5')", "startswith('meth511_r6')"),
    ('meth511_r5_derivation.json', 'meth511_r6_derivation.json'),
    ('METH_511_R5_CURRENT_IDLE_IDENTITIES_20261007.md', 'METH_511_R6_DYNAMIC_IDLE_BINDING_20261007.md'),
    ("    assert set(current)=={24880,28180,29836},('current_idle_topology',current)",
     "    # Exact current identities/topology are recorded prospectively, never guessed PIDs.\n"
     "    for row in current.values():\n"
     "        p=psutil.Process(row['pid']);assert p.create_time()==row['create_time_unix']\n"
     "        assert p.name()==row['name'] and p.exe()==row['exe']\n"
     "        assert sum(p.cpu_times()[:2])==row['cpu_seconds'],('idle_CPU_during_binding',row['pid'])\n"
     "        assert sorted((q.pid,q.create_time()) for q in p.children(recursive=True))==row['descendants']"),
    ("    metadata_fault=item(O.DOC/'meth511_r4_binding.failure.json');catalog[metadata_fault['path'].lower()]=metadata_fault",
     "    for name in ['meth511_r4_binding.failure.json','meth511_r5_binding.failure.json']:\n"
     "        metadata_fault=item(O.DOC/name);catalog[metadata_fault['path'].lower()]=metadata_fault"),
    ("'idle_processes':0", "'idle_processes':len(current)"),
])

derive(B / 'meth511_windows_terminal.ps1', B / 'meth511_r6_windows_terminal.ps1', [
    ('meth511_windows_terminal.json', 'meth511_r6_windows_terminal.json'),
    ("    $path=Join-Path $doc ('meth511_'+$kind+'_result.json');$raw=",
     "    $prefix=if ($kind -eq 'cohort') {'meth511_r3_'} else {'meth511_r6_'}\n"
     "    $path=Join-Path $doc ($prefix+$kind+'_result.json');$raw="),
])

with (B / 'meth511_r6_derivation.json').open('x', encoding='utf8') as f:
    json.dump(records, f, indent=2)
    f.write('\n')
