"""Per-token byte ledger for candidate donors of the CHATBOT50 roadmap.

DESK MODEL, NOT A MEASUREMENT.  For each donor this prices the bytes a batch-1
decode step must stream (large matrices, router, head, KV/state at a declared
context) under a few explicit precision policies, and converts them into a desk
rate with two declared constants (BW_PLAN, T_FIXED).  Every constant prints its
provenance with --show-sources.  An optional GGUF path prices an actual shipped
artifact from its tensor header (types and shapes only; no tensor values read).

Standard library only.  Usage:
    python -I benchmarks/chatbot50/byte_budget_ledger.py
    python -I benchmarks/chatbot50/byte_budget_ledger.py --gguf <file.gguf> --topk 4
    python -I benchmarks/chatbot50/byte_budget_ledger.py --json out.json --show-sources
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------- constants --
BW_PLAN = 32.0e9  # B/s, planning streamed bandwidth at 6 physical workers
T_FIXED = 2.0e-3  # s/token, non-streaming work (norms, routing, softmax, barriers)
CALIBRATION = 0.85  # measured/desk, complete all-I8 C artifact (see SOURCES)
CALIBRATION_W4 = 0.53  # measured/desk, complete W4G32-FFN C artifact (see SOURCES)
CONTEXTS = (512, 2048, 4096)

SOURCES = {
    "BW_PLAN": "0.88x of E30 BW-CEIL 36.30 GB/s (SPEED_LEDGER s40); E61 int8 stream "
               "34.0-34.9 GB/s (s60.3); METH409 Q6_K head 264.07 MB in 6.695-7.733 ms "
               "= 34.1-39.4 GB/s at six workers.",
    "T_FIXED": "ASSUMPTION to be measured in Phase A cell A1; R1 measured 1.20 ms of "
               "non-matvec work at 0.5B/short context (SPEED_LEDGER s11.1).",
    "CALIBRATION": "Qwen2.5-0.5B-Instruct all-I8 complete recipient, context 2048: "
                   "measured median 44.532/s (RECIPIENT_INT8_BASELINE_RESULT_20261015) "
                   "vs this desk model 52.4/s -> 0.85.  One artifact, one kernel family.",
    "CALIBRATION_W4": "Same source, W4G32 FFN / W8 attention+head complete recipient, "
                      "context 2048: measured median 35.729/s "
                      "(RECIPIENT_MIXED_SCALAR_RESULT_20261015) vs desk 67.8/s -> 0.53. "
                      "The only complete 4-bit C artifact measured so far.",
    "bytes_per_weight": "ggml block sizes: Q4_K 144/256, Q6_K 210/256, Q8_0 34/32, "
                        "IQ3_S 110/256, MXFP4 17/32 (gpt-oss experts).",
}

BPW = {
    "F32": 4.0, "F16": 2.0, "BF16": 2.0, "I8": 1.0,
    "Q8_0": 34 / 32, "Q6_K": 210 / 256, "Q5_0": 22 / 32, "Q4_K": 144 / 256,
    "IQ3_S": 110 / 256, "MXFP4": 17 / 32,
}

# --------------------------------------------------------------- donor shapes --


@dataclass
class Donor:
    name: str
    evidence: str  # VERIFIED-HEADERS | CONFIG | SECONDARY
    attention: int  # always-on attention/mixer coefficients per token
    dense: int  # always-on dense FFN coefficients per token
    shared: int  # always-on shared-expert coefficients per token
    routed: int  # selected routed-expert coefficients per token
    router: int  # router coefficients per token (F32 at source unless noted)
    head: int  # full vocabulary head coefficients
    vocab: int
    d_model: int
    kv_values_per_pos: int  # cached values read per position by full-context layers
    kv_window_per_pos: int = 0  # cached values per position in sliding-window layers
    kv_window: int = 0  # sliding-window length for those layers
    state_bytes: int = 0  # recurrent state read+write per token (bytes, F32)
    routed_format: str = "Q4_K"  # native routed format for P0
    total_params: float = 0.0
    notes: list[str] = field(default_factory=list)


def gqa_attention(layers: int, d: int, hq: int, hkv: int, hd: int) -> int:
    return layers * (d * (hq + 2 * hkv) * hd + hq * hd * d)


LING_MINI = Donor(
    name="Ling-mini-2.0 (16.26B, A1.43B)", evidence="VERIFIED-HEADERS (METH407, rev a810f64)",
    attention=gqa_attention(20, 2048, 16, 4, 128),
    dense=3 * 2048 * 5120, shared=19 * 3 * 2048 * 512, routed=19 * 8 * 3 * 2048 * 512,
    router=19 * 256 * 2048, head=157184 * 2048, vocab=157184, d_model=2048,
    kv_values_per_pos=20 * 2 * 4 * 128, total_params=16.256e9,
    notes=["no MTP layer in released chat checkpoint", "sibling Ling-flash-2.0 shares tokenizer"],
)
GIGACHAT = Donor(
    name="GigaChat3.1-10B-A1.8B", evidence="VERIFIED-HEADERS (STRAT-01, rev 189fff2)",
    attention=26 * 25_001_984, dense=3 * 1536 * 8960, shared=25 * 3 * 1536 * 1280,
    routed=25 * 4 * 3 * 1536 * 1280, router=25 * 64 * 1536, head=128256 * 1536,
    vocab=128256, d_model=1536, kv_values_per_pos=26 * (512 + 64), total_params=11.48e9,
    notes=["MLA latent cache 576/layer", "one MTP layer (256.3M large weights/draft)"],
)
GRANITE_TINY = Donor(
    name="Granite-4.0-H-Tiny (6.94B, A1B)", evidence="VERIFIED-HEADERS (NATIVE_DONOR_PREFLIGHT)",
    attention=526_417_920 + 25_165_824, dense=0, shared=188_743_680, routed=566_231_040,
    router=3_932_160, head=154_140_672, vocab=100352, d_model=1536,
    kv_values_per_pos=4096, state_bytes=113_246_208 + 3_833_856, total_params=6.939e9,
    notes=["attention row includes 526.4M Mamba2 projections", "F32 SSM state r/w per token"],
)
QWEN3_30B = Donor(
    name="Qwen3-30B-A3B", evidence="CONFIG (TARGET_DONOR_DECISION)",
    attention=48 * 18_874_368, dense=0, shared=0, routed=48 * 8 * 4_718_592,
    router=48 * 128 * 2048, head=151936 * 2048, vocab=151936, d_model=2048,
    kv_values_per_pos=48 * 2 * 4 * 128, total_params=30.5e9,
)
GPT_OSS_20B = Donor(
    name="gpt-oss-20b", evidence="CONFIG (local HF cache)",
    attention=gqa_attention(24, 2880, 64, 8, 64), dense=0, shared=0,
    routed=24 * 4 * 3 * 2880 * 2880, router=24 * 32 * 2880, head=201088 * 2880,
    vocab=201088, d_model=2880, kv_values_per_pos=12 * 2 * 8 * 64,
    kv_window_per_pos=12 * 2 * 8 * 64, kv_window=128, routed_format="MXFP4", total_params=20.9e9,
    notes=["12 sliding-window layers capped at 128 positions"],
)
GPT_OSS_120B = Donor(
    name="gpt-oss-120b", evidence="CONFIG (LARGE_FAMILY_SELECTION)",
    attention=gqa_attention(36, 2880, 64, 8, 64), dense=0, shared=0,
    routed=36 * 4 * 3 * 2880 * 2880, router=36 * 128 * 2880, head=201088 * 2880,
    vocab=201088, d_model=2880, kv_values_per_pos=18 * 2 * 8 * 64,
    kv_window_per_pos=18 * 2 * 8 * 64, kv_window=128, routed_format="MXFP4", total_params=116.8e9,
)
LFM2_8B = Donor(
    name="LFM2-8B-A1B (8.3B, A1.5B)", evidence="CONFIG (fetched 2026-10-16, not header-verified)",
    attention=gqa_attention(6, 2048, 32, 8, 64) + 18 * (2048 * 3 * 2048 + 2048 * 2048),
    dense=2 * 3 * 2048 * 7168, shared=0, routed=22 * 4 * 3 * 2048 * 1792,
    router=22 * 32 * 2048, head=65536 * 2048, vocab=65536, d_model=2048,
    kv_values_per_pos=6 * 2 * 8 * 64, total_params=8.3e9,
    notes=["attention row includes 18 gated short-conv mixers (in D->3D, out D->D)"],
)
TRINITY_NANO = Donor(
    name="Trinity-Nano-Preview (6B, A1B)", evidence="CONFIG (fetched 2026-10-16, not header-verified)",
    attention=56 * (1024 * (8 + 2 * 2) * 128 + 2 * 8 * 128 * 1024),
    dense=2 * 3 * 1024 * 3072, shared=54 * 3 * 1024 * 256, routed=54 * 8 * 3 * 1024 * 256,
    router=54 * 128 * 1024, head=200192 * 1024, vocab=200192, d_model=1024,
    kv_values_per_pos=14 * 2 * 2 * 128, kv_window_per_pos=42 * 2 * 2 * 128, kv_window=2048,
    total_params=6.0e9,
    notes=["attention row assumes one D x D output gate per layer (gated attention)",
           "preview checkpoint; 42 sliding-window layers of 2048"],
)
LING_FLASH = Donor(
    name="Ling-flash-2.0 (103B, A6.1B)", evidence="CONFIG (fetched 2026-10-16, not header-verified)",
    attention=gqa_attention(32, 4096, 32, 4, 128), dense=3 * 4096 * 9216,
    shared=31 * 3 * 4096 * 1024, routed=31 * 8 * 3 * 4096 * 1024, router=31 * 256 * 4096,
    head=157184 * 4096, vocab=157184, d_model=4096, kv_values_per_pos=32 * 2 * 4 * 128,
    total_params=102.9e9,
)
OLMOE = Donor(
    name="OLMoE-1B-7B-0924", evidence="CONFIG (local HF cache)",
    attention=gqa_attention(16, 2048, 16, 16, 128), dense=0, shared=0,
    routed=16 * 8 * 3 * 2048 * 1024, router=16 * 64 * 2048, head=50304 * 2048,
    vocab=50304, d_model=2048, kv_values_per_pos=16 * 2 * 16 * 128, total_params=6.92e9,
)
QWEN05_DENSE = Donor(
    name="Qwen2.5-0.5B-Instruct (dense control)", evidence="VERIFIED (recipient baseline)",
    attention=gqa_attention(24, 896, 14, 2, 64), dense=24 * 3 * 896 * 4864, shared=0,
    routed=0, router=0, head=151936 * 896, vocab=151936, d_model=896,
    kv_values_per_pos=24 * 2 * 2 * 64, total_params=0.494e9,
)

DONORS = [LING_MINI, TRINITY_NANO, LFM2_8B, GIGACHAT, GRANITE_TINY, OLMOE, QWEN3_30B,
          GPT_OSS_20B, GPT_OSS_120B, LING_FLASH, QWEN05_DENSE]

# ------------------------------------------------------------------ policies --
SHORTLIST_RANK = 256  # placeholder rank of the approximate head scan (UNVALIDATED)
SHORTLIST_RERANK = 64  # exact candidate rows rescored at the source head precision


@dataclass(frozen=True)
class Policy:
    key: str
    label: str
    core: str
    routed: str | None  # None -> donor native routed format
    head: str  # full head format, or "SHORTLIST"
    router: str
    kv: str


POLICIES = [
    Policy("P0", "Q4_K matrices, Q6_K head, F32 router, F16 KV", "Q4_K", None, "Q6_K", "F32", "F16"),
    Policy("P1", "P0 + int8 KV", "Q4_K", None, "Q6_K", "F32", "I8"),
    Policy("P2", "P1 + approximate head scan + exact rerank", "Q4_K", None, "SHORTLIST", "F32", "I8"),
    Policy("P3", "P2 + IQ3_S routed experts", "Q4_K", "IQ3_S", "SHORTLIST", "F32", "I8"),
    Policy("P4", "P3 + BF16 router", "Q4_K", "IQ3_S", "SHORTLIST", "BF16", "I8"),
]


def head_bytes(d: Donor, fmt: str) -> float:
    if fmt != "SHORTLIST":
        return d.head * BPW[fmt]
    scan = d.vocab * SHORTLIST_RANK * BPW["Q4_K"]
    projection = d.d_model * SHORTLIST_RANK * BPW["F16"]
    rerank = SHORTLIST_RERANK * d.d_model * BPW["Q6_K"]
    return scan + projection + rerank


def ledger(d: Donor, p: Policy, ctx: int) -> dict:
    routed_fmt = p.routed or d.routed_format
    kv_bpv = BPW[p.kv]
    if p.kv == "I8":
        kv_bpv += 2.0 / 128  # one F16 scale per 128-value head vector
    window = d.kv_window_per_pos * min(ctx, d.kv_window)
    organs = {
        "attention_or_mixer": d.attention * BPW[p.core],
        "dense_ffn": d.dense * BPW[p.core],
        "shared_expert": d.shared * BPW[p.core],
        "routed_experts": d.routed * BPW[routed_fmt],
        "router": d.router * BPW[p.router],
        "head": head_bytes(d, p.head),
        "kv_cache": (d.kv_values_per_pos * ctx + window) * kv_bpv,
        "recurrent_state": float(d.state_bytes),
    }
    total = sum(organs.values())
    seconds = total / BW_PLAN + T_FIXED
    return {
        "donor": d.name, "policy": p.key, "context": ctx,
        "organs_bytes": {k: round(v) for k, v in organs.items()},
        "total_bytes": round(total), "desk_tok_s": 1.0 / seconds,
        "calibrated_tok_s": CALIBRATION / seconds,
        "calibrated_w4_tok_s": CALIBRATION_W4 / seconds,
    }


# --------------------------------------------------------------- GGUF header --
GGML_BLOCKS = {  # type id -> (name, elements per block, bytes per block)
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
}
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
            10: "<Q", 11: "<q", 12: "<d"}


class _Reader:
    def __init__(self, fh):
        self.fh = fh

    def take(self, fmt: str):
        size = struct.calcsize(fmt)
        data = self.fh.read(size)
        if len(data) != size:
            raise ValueError("truncated GGUF header")
        return struct.unpack(fmt, data)[0]

    def string(self) -> str:
        n = self.take("<Q")
        return self.fh.read(n).decode("utf-8", errors="replace")

    def value(self, vtype: int):
        if vtype in _SCALARS:
            return self.take(_SCALARS[vtype])
        if vtype == 8:
            return self.string()
        if vtype == 9:
            etype, count = self.take("<I"), self.take("<Q")
            if etype in _SCALARS:  # skip bulk scalar arrays without materializing them
                self.fh.seek(count * struct.calcsize(_SCALARS[etype]), 1)
                return f"<array {count}>"
            items = [self.value(etype) for _ in range(count)]
            return items if count < 64 else f"<array {count}>"
        raise ValueError(f"unknown GGUF value type {vtype}")


def read_gguf_header(path: str) -> tuple[dict, list[dict]]:
    with open(path, "rb") as fh:
        r = _Reader(fh)
        if fh.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version, n_tensors, n_kv = r.take("<I"), r.take("<Q"), r.take("<Q")
        if version != 3:
            raise ValueError(f"unsupported GGUF version {version}")
        meta = {}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = r.value(r.take("<I"))
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.take("<Q") for _ in range(r.take("<I"))]
            ttype, _offset = r.take("<I"), r.take("<Q")
            elems = 1
            for dim in dims:
                elems *= dim
            tname, block, nbytes = GGML_BLOCKS[ttype]
            tensors.append({"name": name, "dims": dims, "type": tname,
                            "elements": elems, "bytes": elems // block * nbytes})
    return meta, tensors


def gguf_token_ledger(path: str, topk: int) -> dict:
    meta, tensors = read_gguf_header(path)
    arch = meta.get("general.architecture", "?")
    n_expert = meta.get(f"{arch}.expert_count")
    used = meta.get(f"{arch}.expert_used_count", topk)
    organs: dict[str, float] = {}
    types: dict[str, dict[str, int]] = {}
    for t in tensors:
        name = t["name"]
        if "norm" in name:
            organ, charged = "small_tensors", t["bytes"]
        elif name.startswith("token_embd"):
            organ, charged = "embedding_row", t["bytes"] / t["dims"][-1]
        elif "_exps" in name:
            organ, charged = "routed_experts", t["bytes"] * used / n_expert
        elif "_shexp" in name:
            organ, charged = "shared_expert", t["bytes"]
        elif "ffn_gate_inp" in name or "exp_probs" in name:
            organ, charged = "router", t["bytes"]
        elif name.startswith("output.") or name == "output.weight":
            organ, charged = "head", t["bytes"]
        elif ".attn_" in name:
            organ, charged = "attention", t["bytes"]
        elif ".ffn_" in name:
            organ, charged = "dense_ffn", t["bytes"]
        else:
            organ, charged = "small_tensors", t["bytes"]
        organs[organ] = organs.get(organ, 0.0) + charged
        types.setdefault(organ, {})
        types[organ][t["type"]] = types[organ].get(t["type"], 0) + 1
    total = sum(organs.values())
    return {"path": path, "architecture": arch, "tensors": len(tensors),
            "expert_count": n_expert, "expert_used_count": used,
            "organs_bytes": {k: round(v) for k, v in sorted(organs.items())},
            "organ_tensor_types": types, "total_weight_bytes_per_token": round(total)}


# ---------------------------------------------------------------------- main --
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", help="price an actual GGUF artifact from its header")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--json", help="write all ledgers to this JSON file")
    ap.add_argument("--show-sources", action="store_true")
    args = ap.parse_args()

    if args.show_sources:
        for key, text in SOURCES.items():
            print(f"{key}: {text}")
        print()

    rows = [ledger(d, p, c) for d in DONORS for p in POLICIES for c in CONTEXTS]
    print(f"DESK MODEL: BW_PLAN={BW_PLAN/1e9:.1f} GB/s, T_FIXED={T_FIXED*1e3:.1f} ms, "
          f"calibrations x{CALIBRATION} (I8 runtime) / x{CALIBRATION_W4} (W4G32 runtime).  "
          f"MB = 1e6 bytes per decoded token.\n")
    print("| donor | evidence | policy | MB/token @2048 | desk tok/s @512 / 2048 / 4096 "
          "| x0.85 / x0.53 @2048 |")
    print("|---|---|---|---:|---|---|")
    for d in DONORS:
        for p in POLICIES:
            r = {c: ledger(d, p, c) for c in CONTEXTS}
            rates = " / ".join(f"{r[c]['desk_tok_s']:.1f}" for c in CONTEXTS)
            print(f"| {d.name} | {d.evidence} | {p.key} | {r[2048]['total_bytes']/1e6:,.1f} | "
                  f"{rates} | {r[2048]['calibrated_tok_s']:.1f} / "
                  f"{r[2048]['calibrated_w4_tok_s']:.1f} |")

    print("\nOrgan split at P0, context 2048 (MB/token):")
    for d in DONORS:
        org = ledger(d, POLICIES[0], 2048)["organs_bytes"]
        print(f"  {d.name}: " + ", ".join(f"{k}={v/1e6:.1f}" for k, v in org.items() if v))

    out = {"constants": {"BW_PLAN": BW_PLAN, "T_FIXED": T_FIXED, "CALIBRATION": CALIBRATION,
                         "CALIBRATION_W4": CALIBRATION_W4, "SHORTLIST_RANK": SHORTLIST_RANK,
                         "SHORTLIST_RERANK": SHORTLIST_RERANK},
           "sources": SOURCES, "policies": [p.__dict__ for p in POLICIES], "rows": rows}
    if args.gguf:
        g = gguf_token_ledger(args.gguf, args.topk)
        out["gguf"] = g
        print(f"\nActual GGUF header ledger: {g['path']}")
        print(f"  arch={g['architecture']} tensors={g['tensors']} "
              f"experts={g['expert_used_count']}/{g['expert_count']}")
        for k, v in g["organs_bytes"].items():
            print(f"  {k:16s} {v/1e6:10.2f} MB  {g['organ_tensor_types'][k]}")
        print(f"  TOTAL weights/token {g['total_weight_bytes_per_token']/1e6:.2f} MB")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
