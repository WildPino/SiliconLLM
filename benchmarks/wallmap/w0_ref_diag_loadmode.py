"""Diagnostic of 29 October after run 1 went VOID, not a timing to report: E44 foreign occupancy of llama.cpp fa_off on
Ling-flash loaded without mmap (-lm none) and with it (-lm mmap). Output: results/w0_ref_diag_loadmode.json."""
import json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, "D:/_THINGS/Progetti/SiliconLLM/benchmarks/donor_adaptation/engine")
import e44_interval as E44

B = "E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/tools/llama.cpp-5b335f4/build-native/bin/llama-bench.exe"
M = "C:/SiliconLLM_stage/f0/ling-flash-2.0-96f9b7a-q4_k_m-noimatrix.gguf"
env = dict(os.environ)
env["PATH"] = os.environ["LOCALAPPDATA"] + "/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.MSVCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin;" + env["PATH"]
out = []
for mm in ("none", "mmap"):
    split = E44.Split()
    p = subprocess.Popen([B, "-m", M, "-t", "6", "-p", "0", "-n", "128", "-d", "0,2048", "-r", "5", "-fa", "off",
                          "-lm", mm, "-o", "json"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    o, e = p.communicate()
    child = E44._process_times(int(p._handle))
    system, foreign = split.close(child)
    rows = json.loads(o.decode(errors="replace"))
    rec = {"mmap": mm, "foreign": foreign, "system": system, "wall": split.wall,
           "ts": [r["samples_ts"] for r in rows]}
    out.append(rec)
    print(json.dumps(rec), flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
