#!/usr/bin/env python3
"""WS3 -- the logit self-check, re-run on the CODE corpus, stratified by segment type.

THE CONTROL. A teacher's top-K must contain the token that actually follows. If the window alignment is
off by one, the true token stops appearing in its own top-K and hit@K collapses -- so hit@K is a planted
positive for the alignment, not a statistic about the teacher. At the MVE this read 96.8% with mean
p_true 0.460.

WHY RE-RUN IT RATHER THAN INHERIT IT. Segments now span 1.47-1.57 teacher tokens on code against 1.086
at the MVE. That moves multi-teacher-token segments from roughly 8% of cases to roughly half: teacher
tokens that sit INSIDE a segment rather than at its start have gone from a marginal code path to a
load-bearing one. A path exercised when it was marginal is not tested where it now carries the weight,
so the check is stratified:

  anchor-start   teacher token j is the target of some student anchor -- the path the MVE exercised.
  interior       teacher token j lies inside a segment whose anchor points at an earlier teacher token.
                 This is the path that just became load-bearing.

If the two strata diverge, the span mechanism is weaker exactly where the code corpus puts most of its
mass, and that has to be known before sixteen hours are spent.

Run: python benchmarks/phase64/data/ws3_logit_selfcheck.py --tag probe --windows 40
"""
import argparse, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(ROOT, "results", "phase64", "rung1")
TEACHER = "Qwen/Qwen2.5-Coder-1.5B"
K = 32
MVE_HIT, MVE_PTRUE = 96.8, 0.460


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--vocab", type=int, default=4096, help="student vocab whose anchors define the strata")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                    help="bf16 on Ampere+. fp16 OVERFLOWS this teacher: it returns NaN logits on a "
                         "66-token prompt, and the only visible symptom is a low hit@K -- which reads as "
                         "a data problem. The MVE ran fp16 on T4 because Turing has no bf16.")
    a = ap.parse_args()

    tids = np.fromfile(os.path.join(DATA, f"teacher_ids_{a.tag}.i32"), dtype=np.int32).astype(np.int64)
    anc = np.fromfile(os.path.join(DATA, f"anchors_V{a.vocab}_{a.tag}.i32"), dtype=np.int32).astype(np.int64)
    is_start = np.zeros(len(tids), dtype=bool)
    is_start[anc[anc >= 0]] = True
    print(f"WS3 logit self-check on code   tag={a.tag}  V={a.vocab}  ctx={a.ctx} stride={a.stride} K={K} dtype={a.dtype}")
    print(f"  teacher tokens {len(tids)}   anchor-start {100*is_start.mean():.1f}%  "
          f"interior {100*(1-is_start.mean()):.1f}%")

    from transformers import AutoModelForCausalLM
    t0 = time.time()
    dt = dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)[a.dtype]
    m = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=dt, attn_implementation="eager",
                                             token=os.environ.get("HF_TOKEN")).cuda().eval()
    print(f"  teacher loaded in {time.time()-t0:.0f}s\n")

    starts = list(range(0, a.windows * a.stride, a.stride))
    hit = {True: [0, 0], False: [0, 0]}          # stratum -> [hits, count]
    psum = {True: 0.0, False: 0.0}
    with torch.no_grad():
        for i in range(0, len(starts), a.batch):
            sb = starts[i:i + a.batch]
            x = torch.tensor(np.stack([tids[s:s + a.ctx] for s in sb]), device="cuda")
            lg = torch.log_softmax(m(x).logits.float(), -1)
            # position p predicts token p+1: logits[:, :-1] against inputs[:, 1:]
            tv, ti = lg[:, :-1].topk(K, dim=-1)
            tgt = x[:, 1:]
            inK = (ti == tgt.unsqueeze(-1)).any(-1)
            ptrue = lg[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).exp()
            for b, s in enumerate(sb):
                strat = is_start[s + 1:s + a.ctx]
                ok = inK[b].cpu().numpy(); pt = ptrue[b].float().cpu().numpy()
                for v in (True, False):
                    msk = strat == v
                    hit[v][0] += int(ok[msk].sum()); hit[v][1] += int(msk.sum())
                    psum[v] += float(pt[msk].sum())

    tot_h = hit[True][0] + hit[False][0]; tot_n = hit[True][1] + hit[False][1]
    tot_p = psum[True] + psum[False]
    print(f"  {'stratum':14s} {'positions':>10s} {'hit@'+str(K):>9s} {'mean p_true':>12s}")
    for v, name in ((True, "anchor-start"), (False, "interior")):
        h, n = hit[v]
        if not n: continue
        print(f"  {name:14s} {n:10d} {100*h/n:8.2f}% {psum[v]/n:12.3f}")
    print(f"  {'ALL':14s} {tot_n:10d} {100*tot_h/tot_n:8.2f}% {tot_p/tot_n:12.3f}")
    print(f"  {'MVE reference':14s} {'':10s} {MVE_HIT:8.2f}% {MVE_PTRUE:12.3f}")

    ha, hi = (100 * hit[True][0] / max(hit[True][1], 1)), (100 * hit[False][0] / max(hit[False][1], 1))
    gap = ha - hi
    ok = (100 * tot_h / tot_n) >= 90.0
    print(f"\n  alignment control: hit@{K} = {100*tot_h/tot_n:.2f}%  "
          f"{'PASS (>=90% -- the window alignment is right)' if ok else 'FAIL -- suspect an off-by-one'}")
    print(f"  stratum gap (anchor-start minus interior): {gap:+.2f} points  "
          f"{'-- the two paths behave alike' if abs(gap) < 3 else '-- THE PATHS DIVERGE, investigate before scoring'}")
    print("\nSTOP. No commit.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
