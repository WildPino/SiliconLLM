#!/usr/bin/env python3
"""64.4/T1 — raw-LM teacher-forced BPB (bits-per-byte) of a candidate teacher on the pinned P62 code-val,
plus generation tok/s on the measuring hardware. Eval-only, $0. The read rule is sealed in the plan:
a teacher that does not clearly beat the cheapest does not earn its generation cost.

Protocol (identical for every teacher, so the numbers are comparable):
  - chat-template OFF: the raw code bytes are fed as-is (add_special_tokens=False), no chat/instruct wrapping.
  - same exact byte slice for all: results/phase62/code_val.txt (1,500,000 bytes; decode_codeval.py).
  - byte-normalized: BPB = (sum NLL in nats / ln2) / total_raw_bytes  -- NOT per-token PPL. Cross-tokenizer fair.
  - sliding window (HF perplexity recipe): window L, stride S; each position scored once with real left-context.
  - quantization declared per model via --quant; --device auto (cuda if present).

Usage:
  python teacher_bpb.py --model <hf-id> --quant {bf16,fp16,8bit,4bit} [--ctx 2048 --stride 1024 --max-bytes N --gen 128]
Output line: MODEL | quant | BPB | gen_tok/s | ctx | n_tokens | bytes
"""
import argparse, time, math, os, sys
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="bf16", choices=["bf16", "fp16", "8bit", "4bit", "fp32"])
    ap.add_argument("--valfile", default="results/phase62/code_val.txt")
    ap.add_argument("--ctx", type=int, default=2048)      # scoring window (<= model max)
    ap.add_argument("--stride", type=int, default=1024)   # non-overlapping scored span per window
    ap.add_argument("--max-bytes", type=int, default=0)   # 0 = full 1.5 MB slice; smaller for a quick smoke
    ap.add_argument("--gen", type=int, default=128)       # tokens to time for gen tok/s (0 = skip)
    ap.add_argument("--attn", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])  # sdpa = memory-efficient (no [T,T] attn matrix); eager if a kernel is missing ("cutlassF: no kernel")
    ap.add_argument("--gpu", type=int, default=-1)   # -1 = auto-pick the largest-VRAM GPU (never split across cards)
    ap.add_argument("--trust-remote-code", action="store_true")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- GPU pinning: NEVER let device_map spread the model across heterogeneous cards. A mixed rig (e.g. a 6 GB
    # Turing GTX 1660 next to a 12 GB Ampere RTX 3060) silently (a) OOMs on the small card and (b) would run bf16 on
    # Turing, which has no bf16. Pin the whole model to the largest-VRAM GPU and verify the dtype is supported there.
    gpu = None
    if dev == "cuda":
        n = torch.cuda.device_count()
        gpu = a.gpu if a.gpu >= 0 else max(range(n), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
        props = torch.cuda.get_device_properties(gpu)
        if n > 1:
            others = ", ".join(f"cuda:{i}={torch.cuda.get_device_properties(i).name}"
                               f"({torch.cuda.get_device_properties(i).total_memory/2**30:.0f}GB)" for i in range(n) if i != gpu)
            print(f"NOTE: {n} GPUs visible; pinning ALL layers to cuda:{gpu} = {props.name} "
                  f"({props.total_memory/2**30:.0f}GB). Ignored: {others}", file=sys.stderr)
        if a.quant == "bf16" and not torch.cuda.is_bf16_supported():
            sys.exit(f"ERROR: --quant bf16 but cuda:{gpu} ({props.name}) has no bf16 support. Use --quant fp16.")
        torch.cuda.set_device(gpu)
    raw = open(a.valfile, "rb").read()
    if a.max_bytes and a.max_bytes < len(raw): raw = raw[:a.max_bytes]
    text = raw.decode("utf-8", errors="replace")
    total_bytes = len(raw)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    dmap = {"": gpu} if dev == "cuda" else None     # pin every layer to the chosen card (no cross-GPU / CPU spill)
    kw = dict(trust_remote_code=a.trust_remote_code, attn_implementation=a.attn)
    if a.quant in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig   # needs bitsandbytes + CUDA
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=(a.quant == "8bit"),
                                                       load_in_4bit=(a.quant == "4bit"),
                                                       bnb_4bit_compute_dtype=torch.float16)
        kw["device_map"] = dmap
    else:
        kw["torch_dtype"] = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.quant]
        kw["device_map"] = dmap
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    model.eval()
    ctx = min(a.ctx, getattr(model.config, "max_position_embeddings", a.ctx) or a.ctx)

    # chat-template OFF: raw bytes, no special tokens
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n_tok = ids.numel()

    # sliding-window NLL (HF recipe): score each token once, with up to (ctx-stride) tokens of left context
    nll = 0.0
    prev_end = 0
    bad = 0          # non-finite windows (fp16 overflow etc) -- must never be silently folded into the total
    with torch.no_grad():
        for begin in range(0, n_tok, a.stride):
            end = min(begin + ctx, n_tok)
            trg_len = end - prev_end            # tokens newly scored in this window
            inp = ids[begin:end].unsqueeze(0).to(model.device)
            tgt = inp.clone()
            tgt[:, :-trg_len] = -100            # ignore the overlap (already scored)
            out = model(inp, labels=tgt)
            # HF returns MEAN loss over scored tokens; multiply back to a summed NLL
            n_scored = (tgt != -100).sum().item()
            l = out.loss.item()
            if not math.isfinite(l):
                bad += 1                        # count, don't poison: a NaN here means the dtype overflowed
            else:
                nll += l * n_scored
            prev_end = end
            if end == n_tok: break
    if bad:
        print(f"WARNING: {bad} non-finite window(s) in {a.model} @ {a.quant} -- BPB is NOT trustworthy. "
              f"Re-run with --quant bf16 (Ampere+) which has fp32 dynamic range.", file=sys.stderr)
    bpb = (nll / math.log(2)) / total_bytes

    # generation tok/s on this hardware (the economic input)
    gen_tps = float("nan")
    if a.gen > 0:
        p = tok("def ", return_tensors="pt").to(model.device)
        prompt, amask = p.input_ids, p.attention_mask
        with torch.no_grad():
            model.generate(prompt, attention_mask=amask, max_new_tokens=8, do_sample=False)   # warm
            torch.cuda.synchronize() if dev == "cuda" else None
            t0 = time.time()
            o = model.generate(prompt, attention_mask=amask, max_new_tokens=a.gen, do_sample=False)
            torch.cuda.synchronize() if dev == "cuda" else None
            gen_tps = a.gen / (time.time() - t0)

    flag = "" if not bad else f" | BAD_WINDOWS={bad} (UNTRUSTWORTHY)"
    print(f"RESULT | {a.model} | {a.quant} | BPB={bpb:.4f} | gen={gen_tps:.1f} tok/s | ctx={ctx} | "
          f"n_tok={n_tok} | bytes={total_bytes} | dev={dev}{flag}")

if __name__ == "__main__":
    main()
