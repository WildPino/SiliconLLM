"""C50-S4 MVE-2 verdict, from the runner's record (`c50_s4_t0_gpu.py --mve2`). The criterion is the S4 design note's,
with its numbers fixed before the run:

- every Q4_K and Q6_K tensor of T0 dequantizes on the GPU to the same fp32 bytes as the pinned gguf-py;
- the fp16 GPU T0 against P0, on MVE-1's 2,044 positions, is within MVE-1's yardstick (P0 against BF16 on the same
  positions): mean |dlogp| <= 0.1448 and top-1 agreement >= 0.9535.

    python -I benchmarks/chatbot50/c50_s4_mve2_verdict.py RECORD.json
"""
import json
import sys

YARD_MEAN_ABS, YARD_TOP1, POSITIONS = 0.1448, 0.9535, 2044

rec = json.load(open(sys.argv[1], encoding="utf-8"))
f = rec["fidelity"]["p0"]
checks = {
    "dequant: all quantized tensors bit-identical": rec["dequant"]["tensors"] == 159 and not rec["dequant"]["mismatched"],
    f"fidelity: {POSITIONS} positions": f["positions"] == POSITIONS,
    f"fidelity: mean|dlogp| {f['mean_abs_dlogp']} <= {YARD_MEAN_ABS}": f["mean_abs_dlogp"] <= YARD_MEAN_ABS,
    f"fidelity: top-1 {f['top1_agree']} >= {YARD_TOP1}": f["top1_agree"] >= YARD_TOP1,
}
for k, v in checks.items():
    print("PASS" if v else "FAIL", k)
print("MVE-2", "PASS" if all(checks.values()) else "FAIL")
