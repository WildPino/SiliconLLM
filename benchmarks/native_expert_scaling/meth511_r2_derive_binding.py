"""Literal metadata binder copy; no scientific imports or queries."""
from pathlib import Path
B=Path(__file__).resolve().parent
old=B/'meth511_r1_prepare_binding.py';text=old.read_text(encoding='utf8')
for before,after in [('meth511_binding.failure.json','meth511_r1_binding.failure.json'),
 ('31614853663796b2895f9f58e2ee386b0709f468fe25bc6b75d540e3cfd3c93d','7957801d59ecdfab534034a39b2afa163465355598f714602fdc81c28180f476'),
 ('meth511_r1_binding.json','meth511_r2_binding.json'),('meth511_r1_binding','meth511_r2_binding'),
 ('meth511_r1_derivation.json','meth511_r2_derivation.json'),('METH_511_R1_RUNTIME_INVENTORY_20261006.md','METH_511_R2_CANONICAL_DONOR_20261007.md')]:
 assert before in text;text=text.replace(before,after)
before="    p=Path(os.path.abspath(path));key=str(p).lower();P.guard()"
after=before+"\n    if p.name in donor_archives and p.parent==donor_directory:\n        retained=donor_archives[p.name];s=p.stat();assert s.st_size==retained['bytes']\n        row={'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':retained['sha256'],'full_sha_freshly_recomputed':False,'identity_scope':'Retained acquisition ZIP SHA; actual all3320 canonical F32 tensor SHAs checked before first donor inference.'}\n        if expected:assert expected==retained['sha256']\n        P.catalog[key]=row;return row"
assert text.count(before)==1;text=text.replace(before,after)
before="runtime=Path(sys.executable).parent.parent"
after=before+"\nprevious=json.loads((O.DOC/'meth506_r4_2_binding.json').read_bytes())\ndonor_directory=Path(previous['donor_directory'])\ndonor_archives={Path(r['path']).name:r for r in previous['donor'] if Path(r['path']).suffix=='.bin'}\nassert len(donor_archives)==3"
assert text.count(before)==1;text=text.replace(before,after)
text=text.replace("'original_scientific_calls':0,'runtime_scope'", "'original_scientific_calls':0,'donor_archive_scope':'Raw ZIP SHA retained from acquisition; fresh actual all3320 F32 tensor SHA is mandatory before first donor inference, under unchanged3600s/24GiB donor bounds. No fresh whole ZIP SHA claim.','runtime_scope'")
text=text.replace('Full SHA still computed freshly on every selected file and whole actual source/candidate/donor.', 'Full SHA freshly computed on selected executable files and whole source/candidate/corpus/ledgers; original donor ZIP SHA retained with current stat and write-deny binding, fresh all3320 canonical tensor SHA before inference.')
before="        value['literal_operational_derivation']=deriv"
after=before+"\n        value['native_runtime_environment']=previous['runtime_environment']\n        assert value['native_runtime_environment']['OMP_NUM_THREADS']=='3'\n        value['price']['donor_fresh_canonical_hash_phase']='donor: ALL3320 loaded F32 tensor bytes before first inference; no duplicated whole-ZIP hash'\n        value['price']['donor_raw_zip_SHA_scope']='retained acquisition SHA, current size/mtime and stage write-deny handles'\n        add(O.DOC/'meth511_binding.failure.json')\n        add(O.DOC/'METH_511_R1_RUNTIME_INVENTORY_20261006.md')"
assert text.count(before)==1;text=text.replace(before,after)
with (B/'meth511_r2_prepare_binding.py').open('x',encoding='utf8',newline='\n') as f:f.write(text)
