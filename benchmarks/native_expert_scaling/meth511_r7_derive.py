"""Literal unchanged-science copies; dispatch only the 23 never-started C calls."""
import ast,hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent;records=[]
def derive(suffix,replacements,extension='.py',old_prefix='meth511_r6_'):
    old=B/(old_prefix+suffix+extension);new=B/('meth511_r7_'+suffix+extension)
    text=old.read_text(encoding='utf8');changes=[]
    for before,after in replacements:
        n=text.count(before);assert n>0,(suffix,before);text=text.replace(before,after);changes.append({'before':before,'after':after,'occurrences':n})
    if extension=='.py':ast.parse(text,filename=str(new))
    with new.open('x',encoding='utf8',newline='\n') as f:f.write(text)
    records.append({'old':str(old),'old_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'new':str(new),'new_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'replacements':changes})

derive('operations',[
    ('meth511_r6_binding.json','meth511_r7_binding.json'),("('meth511_r6_'+kind)","('meth511_r7_'+kind)"),("'meth511_r6_'","'meth511_r7_'"),
    ("'native':(2400,2<<30,14<<30)","'native':(1100,2<<30,14<<30)"),
    ("        assert size<=output,('output_bytes',size)","        size+=getattr(self,'retained_output_bytes',0)\n        assert size<=output,('output_bytes',size)"),
])
derive('native',[
    ('import meth511_r6_operations as O','import meth511_r7_operations as O\nimport meth511_r7_retained as H'),
    ("b=ctx.admit(args.binding_sha);prep=", "b=ctx.admit(args.binding_sha);H.initialize(ctx,b);prep="),
    ('prefix=ctx.out/label;', 'prefix=H.prefix(ctx,label);'),
    ('rows=ctx.run_native(argv,label);','rows=H.reuse_or_run(ctx,argv,label);'),
    ("assert ctx.r['native_calls']==576 and len(records)==96", "assert ctx.r['native_calls']==23 and ctx.r['retained_native_calls']==553 and len(ctx.r['commands'])==576 and len(records)==96"),
    ("'native_calls':576,", "'native_calls':23,'retained_native_calls':553,'total_native_calls':576,"),
])
derive('donor',[
    ('import meth511_r6_operations as O','import meth511_r7_operations as O'),
    ("assert native['native_calls']==576", "assert native['native_calls']==23 and native['retained_native_calls']==553 and len(native['commands'])==576"),
])
derive('retention_audit',[
    ('import meth511_r6_operations as O','import meth511_r7_operations as O'),
    ("        commands=native['commands'];expected=[]", "        assert native['native_calls']==23 and native['retained_native_calls']==553\n        commands=native['commands'];expected=[]"),
    ("            own=command['process_instance']", "            baseline={(r['pid'],r['create_time_unix']):r['cpu_seconds'] for r in command['idle_start']}\n            assert {(r['pid'],r['create_time_unix']):r['cpu_seconds'] for r in command['idle_end']}==baseline\n            assert all({(r['pid'],r['create_time_unix']):r['cpu_seconds'] for r in o['idle']}==baseline for o in command['timing_observations'])\n            own=command['process_instance']"),
])
derive('prepare_binding',[
    ('import meth511_r6_operations as O','import meth511_r7_operations as O\nimport meth511_r7_retained as H'),
    ('meth511_r6*','meth511_r7*'),("startswith('meth511_r6')","startswith('meth511_r7')"),
    ('meth511_r6_derivation.json','meth511_r7_derivation.json'),
    ('METH_511_R6_DYNAMIC_IDLE_BINDING_20261007.md','METH_511_R7_RETAIN_CLOSED_CALLS_20261007.md'),
    ("    closed=[]", "    def retain_extra(r):\n        h=k.CreateFileW(r['path'],0x80000000,1,None,3,0x80,None);assert h not in (None,ctypes.c_void_p(-1).value),r['path'];handles.append(h)\n        s=Path(r['path']).stat();assert (s.st_size,s.st_mtime_ns)==(r['bytes'],r['mtime_ns']);guard()\n    H.bind(O,b,catalog,item,retain_extra)\n    closed=[]"),
    ("'completed_native_calls':0", "'completed_native_calls':553"),
])
derive('windows_terminal',[
    ('meth511_r6_windows_terminal.json','meth511_r7_windows_terminal.json'),
    ("else {'meth511_r6_'}", "else {'meth511_r7_'}"),
    ("    $start=(Utc $raw.started_utc).AddSeconds(-2);$end=[DateTimeOffset]::UtcNow", "    $stageStart=Utc $raw.started_utc\n    foreach ($v in @($raw.retained_process_instances)) {if ($null -ne $v) {$t=Utc $v.start_utc;if ($t -lt $stageStart) {$stageStart=$t}}}\n    $start=$stageStart.AddSeconds(-2);$end=[DateTimeOffset]::UtcNow"),
    ("    foreach ($cmd in @($raw.commands)) {", "    foreach ($v in @($raw.retained_process_instances)) {if ($null -ne $v) {$instances+=@{label=$v.label;pid=$v.pid;create_time_unix=$v.create_time_unix;start_utc=(Utc $v.start_utc).AddSeconds(-2).ToString('o');end_utc=(Utc $v.end_utc).AddSeconds(2).ToString('o')}}}\n    foreach ($cmd in @($raw.commands)) {"),
],extension='.ps1')
with (B/'meth511_r7_derivation.json').open('x',encoding='utf8') as f:json.dump(records,f,indent=2);f.write('\n')
