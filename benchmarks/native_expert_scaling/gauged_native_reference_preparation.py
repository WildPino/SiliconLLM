"""Compile separate scalar FMA reference and preserve original DLL disassembly."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from falcon_auxiliary_s6_transport import extent,write


def main(a):
    start=time.monotonic();assert not a.directory.exists() and not a.out.exists()
    prep_path=DOC/'native_scalar_readout_preparation_20261012.json';prep=json.loads(prep_path.read_bytes())
    assert extent(prep['dll']['path'])==prep['dll']
    compiler=Path(prep['compiler']['path']);dump=compiler.parent/'llvm-objdump.exe';source=B/'gauged_native_scalar_reference.c'
    inputs=[extent(p) for p in [Path(__file__),source,compiler,dump,prep_path,Path(prep['dll']['path']),DOC/'GAUGED_NATIVE_STORED_AUDIT_PROTOCOL_20261012.md']]
    a.directory.mkdir();dll=a.directory/'scalar_reference.dll';log=a.directory/'compile.log';assembly=a.directory/'original_scalar_readout.asm'
    command=[str(compiler),'-O3','-mavx2','-mfma','-march=znver2','-fno-vectorize','-fno-slp-vectorize','-ffp-contract=off','-shared',str(source),'-o',str(dll.resolve()),'-lm']
    p=subprocess.run(command,capture_output=True,timeout=45,creationflags=0x08000000);log.write_bytes(p.stdout+p.stderr);assert p.returncode==0
    dcommand=[str(dump),'--disassemble-symbols=scalar_readout',prep['dll']['path']];p=subprocess.run(dcommand,capture_output=True,timeout=30,creationflags=0x08000000);assembly.write_bytes(p.stdout+p.stderr);assert p.returncode==0
    text=assembly.read_text(errors='strict');assert 'vfmadd231ss' in text and 'vfmadd213ss' in text and 'vfmadd231ps' in text
    for x in inputs:assert extent(x['path'])==x
    write(a.out,dict(schema='GAUGED_NATIVE_SCALAR_REFERENCE_PREPARATION_V1',freeze=a.freeze,command=sys.orig_argv,inputs=inputs,compile_command=command,dump_command=dcommand,
        dll=extent(dll),log=extent(log),assembly=extent(assembly),exit_code=0,seconds=time.monotonic()-start,
        scope='Independent scalar explicit FMA reference;no original bodies/SIMD/numeric input/call. Original disassembly confirms sequential norm FMA,eight dot lanes,and final fused bias. '
            'Compile-only;future full output bit audit,not native precision/source quality admission.'))
    print(json.dumps(dict(preparation=extent(a.out),seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--freeze',required=True);main(p.parse_args())
