"""S2M: IQ2_KS on the routed experts too, post hoc.  ik_llama.cpp's quantizer writes the organs (as S2L's ks_org) and the routed experts in IQ2_KS with the fit24 imatrix,
then requantizes the file to BF16.  The experts' imatrix is the mean per expert, in_sum2 / counts; an expert the calibration never routed to (count 0) takes the mean of its tensor's
other experts, and the number is recorded.

Controls:
    imatrix arrived   the log loads one entry per organ and expert tensor and names no tensor without weights.  Planted: S2L's ks-im.log, which left the experts out, must fail it.
    organs unchanged  every organ tensor equals S2L's ks-im-bf16.gguf byte for byte: the same quantizer and imatrix give ks_org's values.
    2-bit             every block of 32 expert weights holds at most 4 values (on the first 2^20 blocks of each tensor); relative weight error per expert tensor, reported.

Usage: python -I benchmarks/stage2/s2z_ks.py --json benchmarks/stage2/results/s2z_ks.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
sys.path.insert(0, str(HERE))
from s1t_export import BF16, IMATRIX, gguf  # noqa: E402
from s2y_ks import CUSTOM, IK, Y, imatrix, legacy, run  # noqa: E402

EXPS = "ffn_gate_exps=iq2_ks,ffn_up_exps=iq2_ks,ffn_down_exps=iq2_ks"
KS_ORG = Y / "ks-im-bf16.gguf"


def experts_imatrix() -> tuple[dict[str, np.ndarray], dict[str, int]]:
    r = {t.name: np.asarray(t.data, dtype=np.float32) for t in gguf.GGUFReader(str(IMATRIX)).tensors}
    out, zero = {}, {}
    for k, v in r.items():
        if not (k.endswith(".in_sum2") and "_exps" in k):
            continue
        n = k.removesuffix(".in_sum2")
        c = r[n + ".counts"].reshape(-1)
        s = v.reshape(c.size, -1)
        m = np.where(c[:, None] > 0, s / np.where(c > 0, c, 1)[:, None], 0.0)
        if (c == 0).any():
            m[c == 0] = m[c > 0].mean(0)
        out[n] = m.reshape(-1).astype(np.float32)
        zero[n] = int((c == 0).sum())
    return out, zero


def log_check(text: str, entries: int) -> bool:
    """token_embd and output have no entry in fit24 and stay Q8_0 here: only an expert or organ tensor without weights fails."""
    m = re.search(r"loaded (\d+) importance matrix entries", text)
    missing = re.findall(r"did not find weights for (\S+)", text)
    return bool(m) and int(m.group(1)) == entries and not [x for x in missing if x not in ("token_embd.weight", "output.weight")]


def bf16(t) -> np.ndarray:
    return (np.asarray(t.data).reshape(-1).view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    data = imatrix()
    ex, zero = experts_imatrix()
    data.update(ex)
    dat = Y / "imatrix-all.dat"
    legacy(data, dat)
    ks, bf = Y / "ks-all.gguf", Y / "ks-all-bf16.gguf"
    log = Y / "ks-all.log"
    if not bf.exists():
        run([str(IK), "--imatrix", str(dat), "--ignore-imatrix-rules", "--custom-q", CUSTOM + "," + EXPS, str(BF16), str(ks), "Q8_0", "6"], log)
        run([str(IK), "--allow-requantize", "--pure", str(ks), str(bf), "BF16", "6"], Y / "ks-all-bf16.log")
    arrived = log_check(log.read_text(encoding="utf-8", errors="replace"), len(data))
    planted = log_check((Y / "ks-im.log").read_text(encoding="utf-8", errors="replace"), len(data))

    got = {t.name: t for t in gguf.GGUFReader(str(bf)).tensors}
    org = {t.name: t for t in gguf.GGUFReader(str(KS_ORG)).tensors}
    src = {t.name: t for t in gguf.GGUFReader(str(BF16)).tensors}
    organs = [n for n in src if any(x in n for x in ("attn_qkv", "attn_output", "_shexp")) or n in {f"blk.0.ffn_{p}.weight" for p in ("gate", "up", "down")}]
    same = [n for n in organs if np.asarray(got[n].data).tobytes() == np.asarray(org[n].data).tobytes()]
    experts, worst = {}, 0
    for n in (x for x in src if "_exps" in x):
        d, o = bf16(got[n]), bf16(src[n])
        b = np.sort(d[: min(d.size, 32 << 20)].reshape(-1, 32), axis=1)
        k = int((1 + (np.diff(b, axis=1) != 0).sum(axis=1)).max())
        worst = max(worst, k)
        experts[n] = {"rel": float(((d - o) ** 2).sum() / (o * o).sum()), "max_distinct_per_32": k}
    rec = {"entries": len(data), "experts_zero_count": sum(zero.values()), "imatrix_arrived": arrived, "planted_s2l_log_fails": not planted,
           "organs_equal_ks_org": len(same), "organ_tensors": len(organs), "expert_tensors": len(experts), "max_distinct_per_32": worst,
           "mean_rel_experts": float(np.mean([v["rel"] for v in experts.values()])),
           "control": arrived and not planted and len(same) == len(organs) and worst <= 4, "zero_by_tensor": zero, "experts": experts, "seconds": round(time.time() - t0, 1)}
    a.json.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("experts", "zero_by_tensor")}))
    return 0 if rec["control"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
