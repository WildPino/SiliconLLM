#!/usr/bin/env python3
# Inventor / S6 - precondition (a) of INVENTORE_04: at what expert count E does FULL-SCAN routing
# (router GEMV ExD + top-8) cost as much as the recall tier's measured ANN query (64.3: 29.05 us/token
# @128K entries t6; 52.4 us t1 scalar)? Above that E, router-as-ANN (the unified lookup) becomes a
# speed argument, not just an elegance argument.
#
# Method: torch fp32 GEMV + topk, SINGLE thread (the t1-comparable figure), best-of over many reps.
# Caveat (declared): python/torch per-call overhead inflates small-E points; the informative region is
# where the GEMV dominates (E >= ~1024). Desk-level instrument, not the C engine.
# Output: tee to docs/in_research/s6_router_crossover_out.txt
import time
import numpy as np
import torch

torch.set_num_threads(1)
D = 256
K = 8
ANN_T1, ANN_T6 = 52.4, 29.05     # us, 64.3 measured @128K entries


def bench(E, reps=200):
    W = torch.randn(E, D)
    x = torch.randn(D)
    for _ in range(10):
        y = W.mv(x); torch.topk(y, K)
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(reps):
            y = W.mv(x)
            torch.topk(y, K)
        dt = (time.perf_counter() - t0) / reps * 1e6
        best = min(best, dt)
    return best


def main():
    print("=" * 90)
    print("S6 precondition (a) - full-scan router cost vs E (torch fp32, 1 thread, D=256, top-8)")
    print("=" * 90)
    print(f"  reference: recall ANN query @128K entries = {ANN_T1} us (t1 scalar) / {ANN_T6} us (t6, SIMD headroom ~18)")
    print(f"  {'E':>8}{'us/token':>10}{'GEMV MB':>9}{'vs ANN t1':>10}")
    for E in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768):
        us = bench(E, reps=max(20, 200000 // E))
        mb = E * D * 4 / 1e6
        print(f"  {E:>8}{us:>10.1f}{mb:>9.2f}{us/ANN_T1:>9.2f}x")
    print("\nreading: the E where us/token crosses ~52 us (t1) bounds where full-scan routing stops being")
    print("free relative to the ANN machinery the engine already has. Bytes matter too: at E=4096 the")
    print("router matrix alone is 4 MB fp32 (vs the recall slot's 1.69 MB searchable @128K). Desk-level;")
    print("the C-engine number would be measured behind the same kernel interface (engine-v2).")


if __name__ == "__main__":
    main()
