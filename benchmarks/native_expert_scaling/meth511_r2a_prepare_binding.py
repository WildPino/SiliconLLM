"""Pre-binder wrapper path correction; no original R2 main or scientific stage began."""
from pathlib import Path
source=Path(__file__).with_name('meth511_r2_prepare_binding.py')
text=source.read_text(encoding='utf8')
before="ORIGINAL=O.DOC/'meth511_r2_binding.failure.json'"
after="ORIGINAL=O.DOC/'meth511_r1_binding.failure.json'"
assert text.count(before)==1
exec(compile(text.replace(before,after),str(source)+'[R2a original-fault-path]', 'exec'))
