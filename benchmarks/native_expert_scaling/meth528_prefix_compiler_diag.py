"""ONE metadata-only driver plan and actual observed descendant identities."""
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from meth528_operations import Meter,ROOT,DOC,write


def main():
    ctx=Meter('prefix_compiler_diag',30,512<<20)
    ctx.out=ROOT/'results/native_expert_scaling/meth528_prefix_compiler_diag'
    assert not ctx.out.exists();ctx.out.mkdir()
    children={};p=None
    class Memory(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    k=ctypes.WinDLL('kernel32',use_last_error=True);mem=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
    mem.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD];mem.restype=wintypes.BOOL
    k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
    k.CloseHandle.argtypes=[wintypes.HANDLE];k.CloseHandle.restype=wintypes.BOOL
    def peak(handle):
        value=Memory();value.cb=ctypes.sizeof(value);assert mem(handle,ctypes.byref(value),value.cb)
        return value.PeakWorkingSetSize
    try:
        path=DOC/'meth528_prefix_manifest_repair1_20261007.json'
        assert ctx.digest(path)=='d9b1876eddb78e33ac2172c5c5bdbe5efef22c880e9da8273dcf0620ea7c2733'
        bind=json.loads((DOC/'meth528_binding.json').read_bytes());compiler=bind['compiler']
        for v in compiler['files']:assert ctx.digest(v['path'])==v['sha256']
        source=ROOT/'benchmarks/native_expert_scaling/meth528_verifier.c'
        argv=[compiler['path'],'--sysroot='+compiler['sysroot'],'-target',compiler['target'],'-fintegrated-cc1','-###','-std=c11','-O2','-ffp-contract=off','-fno-fast-math','-c',str(source),'-o',str(ctx.out/'NOT_CREATED.o')]
        import psutil
        with (ctx.out/'driver_plan.log').open('xb') as f:
            p=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,creationflags=0x08000000)
            ctx.compiler_owned[p.pid]=psutil.Process(p.pid).create_time()
            root=dict(pid=p.pid,create_time_unix=ctx.compiler_owned[p.pid],argv=argv)
            while p.poll() is None:
                try:
                    for child in psutil.Process(p.pid).children(recursive=True):
                        if child.pid not in children:
                            h=k.OpenProcess(0x410,False,child.pid)
                            assert h
                            children[child.pid]=dict(handle=h,pid=child.pid,create_time_unix=child.create_time(),name=child.name(),argv=child.cmdline(),parent_pid=child.ppid())
                            ctx.compiler_owned[child.pid]=children[child.pid]['create_time_unix']
                except psutil.NoSuchProcess:pass
                ctx.compiler_peak=max(ctx.compiler_peak,peak(wintypes.HANDLE(int(p._handle)))+sum(peak(v['handle']) for v in children.values()))
                ctx.guard();time.sleep(.001)
        root.update(exit_code=p.returncode,OS_peak_bytes=peak(wintypes.HANDLE(int(p._handle))))
        descendants=[]
        for child in children.values():
            child['OS_peak_bytes']=peak(child['handle']);k.CloseHandle(child.pop('handle'));descendants.append(child)
        ctx.r.update(root=root,observed_descendants=descendants,complete_descendant_capture_guaranteed=False,
                     scope='ONE metadata-only compiler driver plan. No object, controls, response, energy or model call. Observed descendant handles retained; unobserved transients cannot be excluded.',resource=ctx.resources())
        assert p.returncode==0 and not (ctx.out/'NOT_CREATED.o').exists()
        write(ctx.raw,ctx.r);ctx.timer.cancel();print(json.dumps(dict(result_sha256=ctx.digest(ctx.raw),root=root,descendants=descendants,resource=ctx.resources())),flush=True)
    except BaseException:
        ctx.fail()
        raise


if __name__=='__main__':main()
