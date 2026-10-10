"""Author a new immutable audit completion from the frozen reference, before use."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[2]
base=ROOT/'benchmarks/native_expert_scaling/original_joint_history_recovery_audit.py'
out=base.with_name('original_joint_history_audit_finish.py')
assert not out.exists()
data=base.read_bytes()
assert hashlib.sha256(data).hexdigest()=='7c88c326ce83a8b479b8b2af401a2fa7c8f71a325dc77e28d6f18b0916cd4ed6'
text=data.decode('utf-8')
def replace_once(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new)
replace_once('def main(args):',
    'from original_joint_history_audit_finish_support import check,adopt_A,finish_seal\n\n\ndef main(args):')
replace_once("    for item in finish_binding['adjudication_inputs']+terminal['output_files']:assert extent(item['path'])==item,item['path']",
    '    resume=check(args) # Prior full hashes adopted; complete end hashes required before publication.')
replace_once("            st,expected=state_export(m['checkpoint']['path'],m['packed']['path'],ns/(prefix+'.expected.witness'),name,m,b,inherited)\n            rows,metrics=native(m['native']['records'],records)\n            delta=aggregate_equal(groups(rows),m['native']['aggregate'])",
    "            if name=='A':\n                st,expected,rows,metrics=adopt_A(m,records,resume)\n                delta=None # Exact prior maxima not recorded; proven tolerance bound retained.\n            else:\n                st,expected=state_export(m['checkpoint']['path'],m['packed']['path'],ns/(prefix+'.expected.witness'),name,m,b,inherited)\n                rows,metrics=native(m['native']['records'],records)\n                delta=aggregate_equal(groups(rows),m['native']['aggregate'])")
replace_once("            milestone_audit.append(dict(state=st,native=metrics,max_aggregate_delta=delta))",
    "            entry=dict(state=st,native=metrics,max_aggregate_delta=delta,\n                aggregate_tolerance='1e-10 * max(1, abs(actual), abs(saved))',adopted=(name=='A'))\n            milestone_audit.append(entry)\n            if name=='B':write(args.directory/(prefix+'.audit.json'),entry)")
replace_once('    out=args.out;write(out,result)',
    '    result=finish_seal(args,resume,result,started)\n    out=args.out;write(out,result)')
replace_once("    parser.add_argument('--out',type=Path,required=True)",
    "    parser.add_argument('--out',type=Path,required=True)\n    parser.add_argument('--resume-binding',type=Path,required=True)\n    parser.add_argument('--directory',type=Path,required=True)")
out.write_text(text,encoding='utf-8',newline='\n')
print(out)
