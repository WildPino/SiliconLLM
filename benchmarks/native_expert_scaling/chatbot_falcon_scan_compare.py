"""Compile/check original scan source and compare NEW captured real packets once."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_falcon_usability import sha,write


def main(args):
    start=time.monotonic()
    assert sha(args.binding)==args.binding_sha
    b=json.loads(args.binding.read_bytes())
    assert b['schema']=='FALCON_SCAN_BRIDGE_BINDING_V1'
    for item in b['inputs']:
        assert sha(item['path'])==item['sha256'], item['path']
    assert not args.directory.exists()
    args.directory.mkdir()
    native=ROOT/'benchmarks/native_expert_scaling/chatbot_falcon_scan.c'
    original=(ROOT/'benchmarks/phase60/engine.c').read_text()
    ctext=native.read_text()
    # Original exact-exp update/reduction/gate, whitespace alone normalized.
    update=re.search(r'OMP_PFOR for\(int c=0;c<DN;c\+\+\)\{ const float\* Ac=s->A\+\(size_t\)c\*N; float\* hc=hl\[c\]; float dtc=dt\[c\],xc=xx\[c\],acc=0;.*?y\[c\]=acc\+s->Dskip\[c\]\*xc; \}',ctext,re.S).group()
    normal=lambda text:' '.join(text.split())
    assert normal(update) in normal(original)
    gate='for(int c=0;c<DN;c++) y[c]*=silu(z[c]);'
    assert gate in original and gate in ctext
    clang=Path(args.clang).resolve()
    exe=args.directory/'original_scan.exe'
    command=[str(clang),'-O2','-std=c11','-fno-fast-math','-ffp-contract=off',str(native.resolve()),'-o',str(exe.resolve()),'-lm']
    compile_result=subprocess.run(command,capture_output=True,text=True,timeout=60,creationflags=8)
    write(args.directory/'compile.json',dict(command=command,exit_code=compile_result.returncode,
          stdout=compile_result.stdout,stderr=compile_result.stderr,clang_sha256=sha(clang),
          source_sha256=sha(native),original_engine_sha256=sha(ROOT/'benchmarks/phase60/engine.c'),
          original_update_and_gate_text_match=True))
    assert compile_result.returncode==0
    import numpy as np
    capture=json.loads(args.capture.read_bytes())
    assert capture['schema']=='FALCON_SCAN_CAPTURE_RESULT_V1' and capture['packet_count']==12
    comparisons=[]
    try:
        for packet in capture['packets']:
            assert time.monotonic()-start<=120
            for key in ('input','oracle'):
                assert sha(packet[key]['path'])==packet[key]['sha256']
            output=args.directory/(packet['name']+'.native.f32')
            argv=[str(exe.resolve()),packet['input']['path'],str(output.resolve())]
            native_result=subprocess.run(argv,capture_output=True,text=True,timeout=10,creationflags=8)
            rec=dict(name=packet['name'],call=packet['call'],layer=packet['layer'],tokens=packet['tokens'],
                command=argv,exit_code=native_result.returncode,stdout=native_result.stdout,stderr=native_result.stderr)
            if native_result.returncode!=0:
                write(args.directory/(packet['name']+'.first_failure.json'),rec)
                raise AssertionError(rec)
            produced=np.fromfile(output,dtype='<f4').astype(np.float64)
            oracle=np.fromfile(packet['oracle']['path'],dtype='<f4').astype(np.float64)
            assert produced.shape==oracle.shape==(768*64+packet['tokens']*768,) and np.isfinite(produced).all()
            def metric(left,right):
                error=float(np.sum((left-right)**2))
                energy=float(np.sum(right**2))
                assert energy>0
                return dict(relative_RMS=(error/energy)**.5,squared_error=error,source_energy=energy,
                     max_abs=float(np.max(np.abs(left-right))),pass_1percent=error<=.0001*energy)
            rec.update(state=metric(produced[:768*64],oracle[:768*64]),
                  gated_output=metric(produced[768*64:],oracle[768*64:]),
                  output_sha256=sha(output),source_state_dtype=packet['source_state_dtype'])
            write(args.directory/(packet['name']+'.comparison.json'),rec)
            comparisons.append(rec)
        result=dict(schema='FALCON_ORIGINAL_SCAN_COMPARISON_V1',freeze=args.freeze,binding_sha256=args.binding_sha,
              capture_sha256=sha(args.capture),exe_sha256=sha(exe),comparisons=comparisons,
              decision='ORIGINAL_SCAN_BRIDGE_PASS' if all(v['state']['pass_1percent'] and v['gated_output']['pass_1percent'] for v in comparisons) else 'ORIGINAL_SCAN_BRIDGE_FAIL',
              original_exact_update_and_gate_verified=True,elapsed_seconds=time.monotonic()-start,
              scope='Supplied real source delta/X/B/C/gate, each packet starts at its own actual source state; not free-running whole model, repeated C prefill or native accepted rate',
              native_OS_through_exit_peak='not measured; no rate/resource admission', new_source_forwards=0)
        write(args.out,result)
        print(json.dumps(dict(decision=result['decision'],packets=len(comparisons),
                    max_state_RMS=max(v['state']['relative_RMS'] for v in comparisons),
                    max_gated_RMS=max(v['gated_output']['relative_RMS'] for v in comparisons),seconds=result['elapsed_seconds'])),flush=True)
    except BaseException as error:
        write(args.directory/'first_failure.json',dict(fault=repr(error),completed=[v['name'] for v in comparisons]))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--clang',required=True)
    main(p.parse_args())
