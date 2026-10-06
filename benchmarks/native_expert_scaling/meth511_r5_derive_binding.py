"""Literal metadata binder copy with current exact idle identities; no queries."""
from pathlib import Path
B=Path(__file__).resolve().parent;text=(B/'meth511_r4_prepare_binding.py').read_text(encoding='utf8')
for before,after in [('import meth511_r4_operations as O','import meth511_r5_operations as O'),
 ('meth511_r4*','meth511_r5*'),("startswith('meth511_r4')","startswith('meth511_r5')"),
 ('meth511_r4_derivation.json','meth511_r5_derivation.json'),('METH_511_R4_CLOSED_IDLE_APPS_20261007.md','METH_511_R5_CURRENT_IDLE_IDENTITIES_20261007.md')]:
 assert before in text;text=text.replace(before,after)
before="    assert not live,('current_foreign_scientific_processes',live)\n    b['idle_processes']=[]"
after="    assert {r['name'].lower() for r in live}<={'ollama app.exe','ollama.exe'},('foreign_active_model_process',live)\n    current={}\n    for r in live:\n        p=psutil.Process(r['pid'])\n        for q in [p,*p.children(recursive=True)]:\n            assert q.name().lower() in ('ollama app.exe','ollama.exe','conhost.exe'),('model_descendant',q.pid,q.name())\n            executable=item(q.exe());catalog[executable['path'].lower()]=executable\n            current[q.pid]={'pid':q.pid,'create_time_unix':q.create_time(),'name':q.name(),'exe':q.exe(),'executable':executable,'cpu_seconds':sum(q.cpu_times()[:2]),'descendants':sorted((v.pid,v.create_time()) for v in q.children(recursive=True))}\n    assert set(current)=={24880,28180,29836},('current_idle_topology',current)\n    b['idle_processes']=list(current.values())\n    metadata_fault=item(O.DOC/'meth511_r4_binding.failure.json');catalog[metadata_fault['path'].lower()]=metadata_fault"
assert text.count(before)==1;text=text.replace(before,after)
with (B/'meth511_r5_prepare_binding.py').open('x',encoding='utf8',newline='\n') as f:f.write(text)
