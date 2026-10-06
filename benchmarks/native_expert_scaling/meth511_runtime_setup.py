"""Metadata-only isolated runtime assembly; no corpus/model/numerical imports."""
import hashlib
import importlib.metadata as M
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
import venv
import zipfile

ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
OUT=ROOT/'results/native_expert_scaling/meth511_runtime';start=time.monotonic()
NAMES=['numpy','psutil','torch','transformers','tokenizers','safetensors','sentencepiece','protobuf','threadpoolctl',
       'filelock','typing-extensions','networkx','jinja2','fsspec','setuptools','sympy','packaging','pyyaml','regex',
       'huggingface-hub','tqdm','requests','colorama','markupsafe','mpmath','charset-normalizer','idna','certifi','urllib3','hf-xet']
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists();OUT.mkdir()
    try:
        target=OUT/'venv';venv.EnvBuilder(with_pip=False).create(target);site=target/'Lib/site-packages'
        links={};packages={}
        for name in NAMES:
            d=M.distribution(name);packages[name]={'version':d.version,'original_metadata_path':str(d._path.resolve())}
            for rel in d.files or []:
                parts=PurePosixPath(str(rel).replace('\\','/')).parts
                if not parts or parts[0] in ('.','..') or parts[0].lower()=='scripts':continue
                top=parts[0];source=Path(d.locate_file(top)).resolve()
                if source.exists():
                    if top in links:assert links[top]['source']==str(source)
                    else:links[top]={'name':top,'source':str(source),'directory':source.is_dir(),'destination':str(site/top)}
        packet=OUT/'links.json';packet.write_text(json.dumps(list(links.values()),indent=2)+'\n',encoding='utf8')
        # One native PowerShell invocation; no shell-composed delete/move.
        ps="$ErrorActionPreference='Stop'; $rows=Get-Content -LiteralPath '"+str(packet).replace("'","''")+"' -Raw -Encoding utf8 | ConvertFrom-Json; foreach($r in $rows) { if($r.directory) { New-Item -ItemType Junction -Path $r.destination -Target $r.source | Out-Null } else { Copy-Item -LiteralPath $r.source -Destination $r.destination } }"
        r=subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',ps],capture_output=True,text=True,timeout=120)
        (OUT/'links.stdout').write_text(r.stdout,encoding='utf8');(OUT/'links.stderr').write_text(r.stderr,encoding='utf8');assert r.returncode==0
        wheel_name='duckdb-1.4.4-cp312-cp312-win_amd64.whl';wheels=OUT/'wheels';wheels.mkdir()
        env={**os.environ,'PIP_CONFIG_FILE':os.devnull,'PIP_INDEX_URL':'https://pypi.org/simple'}
        r=subprocess.run([sys.executable,'-m','pip','download','--index-url','https://pypi.org/simple','--no-deps','--only-binary=:all:','--dest',str(wheels),'duckdb==1.4.4'],env=env,capture_output=True,text=True,timeout=120)
        (OUT/'wheel.stdout').write_text(r.stdout,encoding='utf8');(OUT/'wheel.stderr').write_text(r.stderr,encoding='utf8');assert r.returncode==0
        wheel=wheels/wheel_name;assert wheel.stat().st_size<=16<<20
        assert digest(wheel)=='d525de5f282b03aa8be6db86b1abffdceae5f1055113a03d5b50cd2fb8cf2ef8'
        with zipfile.ZipFile(wheel) as z:
            for name in z.namelist():
                parts=PurePosixPath(name).parts;assert not PurePosixPath(name).is_absolute() and '..' not in parts
                assert parts[0].startswith(('duckdb','_duckdb','adbc_driver_duckdb')),(name,parts)
                destination=site.joinpath(*parts);assert destination.resolve().is_relative_to(site.resolve())
                assert not destination.exists();z.extract(name,site)
        assert time.monotonic()-start<=240
        result={'experiment':'METH511 isolated runtime assembly','source_sha256':digest(__file__),'python':sys.version,
            'base_executable':sys._base_executable,'runtime_executable':str((target/'Scripts/python.exe').resolve()),
            'packages_reused_without_reinstallation':packages,'links':list(links.values()),
            'duckdb_wheel':{'path':str(wheel),'bytes':wheel.stat().st_size,'sha256':digest(wheel),'PyPI':'https://pypi.org/project/duckdb/1.4.4/'},
            'absent_optional_packages':['pyarrow','pandas','scikit-learn','scipy','datasets','bitsandbytes','accelerate'],
            'runtime_invocation':'-I -X utf8; trusted script explicitly adds its source directory',
            'numerical_corpus_model_native_compiler_calls':0,'seconds':time.monotonic()-start}
        dest=DOC/'meth511_runtime_setup_result.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
        print(json.dumps({'result':str(dest),'sha256':digest(dest),'seconds':result['seconds'],'reused_distributions':len(packages),'links':len(links),'wheel_bytes':wheel.stat().st_size}),flush=True)
    except BaseException:
        import traceback
        p=DOC/'meth511_runtime_setup_result.failure.json';assert not p.exists();p.write_text(json.dumps({'traceback':traceback.format_exc(),'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf8');raise
if __name__=='__main__':main()
