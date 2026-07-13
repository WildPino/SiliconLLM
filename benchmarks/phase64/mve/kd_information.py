#!/usr/bin/env python3
# Phase 64.4 / MVE — the D3 INFORMATION diagnostic (CPU/GPU, $0, no training).
#   magic: "MVED" 0x4D564544
#
# T2 answered WHERE the KD can attach (boundary coincidence: ~40% of student positions, segments tiling 100% of the
# bytes). It did NOT answer HOW MUCH TEACHER INFORMATION SURVIVES the attachment. That is what decides mapping-vs-
# fallback, and it is measurable for $0 before any training.
#
# The sealed span-KD design projects each teacher top-K token onto the FIRST student token of its bytes, summing the
# probabilities that collapse onto the same student token. The collapse is many-to-one and it is NOT benign: " the",
# " then", " they" share a student first-token, so the teacher's uncertainty ABOUT WHICH WORD COMES NEXT is summed
# away. What reaches the student is the entropy of the first token, not the entropy of the teacher.
#
# Measured per student vocab V (the knob T2 examined):
#   anchor%              -- boundary coincidence (the T2 column, reproduced on the pilot corpus)
#   mass on true token   -- does the projected target point at the right place? (sanity: the map is not broken)
#   H(teacher | top-K)   -- the information the teacher actually has, in bits
#   H(q projected)       -- what survives the projection, in bits
#   retention            -- H(q)/H(teacher): the fraction of dark knowledge that reaches the student
#
# Run: .venv/Scripts/python.exe benchmarks/phase64/mve/kd_information.py --tag smoke
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase62"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase55"))
sys.path.insert(0, HERE)
from cartography import Bpe, train_bpe          # noqa: E402
from phase55_ssm import load_meta               # noqa: E402
from mve_data import nl_blocks, BPE_SAMPLE, BLOCK   # noqa: E402

DATA = os.path.join(ROOT, "results", "phase64", "mve")
_B = None
def _init(b):
    global _B; _B = b
def _enc(x):
    return _B.encode(x)


def span_bits(bpe, sid, ends, sel, anc, j0, L_ids, L_pq, tbytes, h_t, V):
    """How much of the teacher's entropy is RECOVERABLE at zero extra teacher cost, by conditioning the SAME stored
    top-K on the bytes the student has already emitted inside the segment.

    Current design distils only at the segment START, where the target is the teacher's marginal over the first
    student token -- and that marginal is nearly one-hot. But by the chain rule the teacher's full uncertainty
    factorizes ACROSS the student tokens of the span: the choice between ' the'/' then'/' they' is not made at ' t',
    it is made at the NEXT student token. Those interior positions are exactly the ones the sealed design supervises
    with a hard CE label. So: walk the segment, keep only the candidates still consistent with the emitted prefix,
    renormalize, and take the entropy of the next-student-token target. Sum over the segment = bits delivered.
    No extra teacher compute: the same K=32 rows, read differently."""
    import math as _m
    tot_anchor = tot_span = n = 0.0
    for p in sel:
        j = int(anc[p]) - j0
        cid, cp = L_ids[j], L_pq[j]
        cb = [tbytes[c] for c in cid]
        w = cp.astype(np.float64); w = w / max(w.sum(), 1e-9)
        true = bpe.exp_bytes                                # student id -> bytes
        b0 = int(ends[p])                                   # byte where the segment starts
        consumed = b""; bits = 0.0; step = 0; first_h = None
        alive = np.ones(len(cid), dtype=bool)
        while step < 8:
            alive = alive & np.array([len(cb[i]) > len(consumed) and cb[i].startswith(consumed) for i in range(len(cid))])
            if not alive.any(): break
            ww = w * alive; ss = ww.sum()
            if ss <= 1e-9: break
            ww = ww / ss
            nxt = {}                                        # next student token -> mass (first token of the REMAINING bytes)
            for i in np.nonzero(alive)[0]:
                e = bpe.encode(cb[i][len(consumed):])
                if not e: continue
                nxt[e[0]] = nxt.get(e[0], 0.0) + ww[i]
            if not nxt: break
            m = np.array(list(nxt.values())); m = m / max(m.sum(), 1e-9)
            h = float(-(m * np.log2(np.clip(m, 1e-12, 1))).sum())
            if first_h is None: first_h = h
            bits += h
            st = int(sid[p + 1 + step])                     # the student token actually emitted here
            consumed = consumed + true[st]
            step += 1
        tot_anchor += (first_h or 0.0); tot_span += bits; n += 1
    if n:
        print(f"{'':>6} {'':>8} {'':>6} {'span-KD (prefix-conditioned, SAME stored top-K):':>50} "
              f"anchor-only {tot_anchor/n:.3f} bits -> full span {tot_span/n:.3f} bits "
              f"({100*(tot_span/n)/max(h_t,1e-6):.0f}% of H(teacher))")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span-n", type=int, default=3000)
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--vocabs", default="1024,2048,4096")
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    meta = json.load(open(os.path.join(DATA, f"meta_{a.tag}.json")))
    N = meta["bytes"]
    V55, el55, id2b = load_meta(os.path.join(ROOT, "results", "phase55", "meta.bin"))
    ids55 = np.fromfile(os.path.join(ROOT, "results", "phase55", "ids.u16"), dtype=np.uint16)
    raw = b"".join(id2b[i] for i in ids55)[:N]

    tids = np.fromfile(os.path.join(DATA, f"teacher_ids_{a.tag}.i32"), dtype=np.int32).astype(np.int64)
    tstart = np.fromfile(os.path.join(DATA, f"teacher_start_{a.tag}.i64"), dtype=np.int64)

    # teacher logits (whatever rows exist)
    d = os.path.join(DATA, f"logits_{a.tag}")
    man = json.load(open(os.path.join(d, "manifest.json")))
    ids_l, pq_l, j0 = [], [], None
    for c in man["chunks"]:
        z = np.load(os.path.join(d, c["path"]))
        if j0 is None: j0 = int(z["j0"])
        ids_l.append(z["ids"].astype(np.int64)); pq_l.append(z["pq"])
    L_ids = np.concatenate(ids_l); L_pq = np.concatenate(pq_l).astype(np.float32) / 255.0
    j1 = j0 + len(L_ids)
    print(f"teacher rows available: [{j0},{j1}) of {len(tids)}  (K={man['K']})")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.teacher)
    Vt = int(tok.vocab_size) + len(tok.get_added_vocab())
    tbytes = []
    for t in range(Vt):
        try: tbytes.append(tok.convert_tokens_to_string([tok.convert_ids_to_tokens(t)]).encode("utf-8"))
        except Exception: tbytes.append(b"")

    print(f"\n==== 64.4 / D3 information diagnostic — corpus {N/2**20:.1f} MiB, teacher {a.teacher} ====")
    print(f"{'V':>6} {'anchor%':>8} {'B/tok':>6} {'mass@true':>10} {'argmax=y':>9} "
          f"{'H(teacher)':>11} {'H(q proj)':>10} {'retained':>9} {'~1-hot':>7}")
    for V in [int(v) for v in a.vocabs.split(",")]:
        bp = os.path.join(DATA, f"bpe{V}_ts.bin")
        bpe = Bpe.load(bp) if os.path.exists(bp) else None
        if bpe is None:
            bpe = train_bpe(raw[:BPE_SAMPLE], V, verbose=False); bpe.save(bp)
        ip = os.path.join(DATA, f"ts_{a.tag}_v{V}.u16")
        if os.path.exists(ip):
            sid = np.fromfile(ip, dtype=np.uint16).astype(np.int64)
        else:
            with Pool(a.jobs, initializer=_init, initargs=(bpe,)) as p:
                parts = p.map(_enc, nl_blocks(raw, BLOCK), chunksize=1)
            sid = np.array([t for pt in parts for t in pt], dtype=np.int64)
            sid.astype(np.uint16).tofile(ip)
        el = np.array(bpe.exp_len, dtype=np.int64)
        ends = np.cumsum(el[sid]); ntok = len(sid)
        b2j = np.full(N + 1, -1, dtype=np.int64); b2j[tstart] = np.arange(len(tids))
        anc = b2j[ends[:-1]]; anc = np.where(anc > 0, anc, -1)
        nanc = int((anc >= 0).sum())

        t2s = np.full(Vt, -1, dtype=np.int64)
        for t in range(Vt):
            if tbytes[t]:
                e = bpe.encode(tbytes[t])
                if e: t2s[t] = e[0]
        t2s_t = torch.from_numpy(t2s).to(dev)

        # anchors whose teacher row we actually have
        sel = np.nonzero((anc >= j0) & (anc < j1))[0]
        sel = sel[np.linspace(0, len(sel) - 1, min(20000, len(sel))).astype(np.int64)]
        rows = anc[sel] - j0
        ti = torch.from_numpy(L_ids[rows]).to(dev); pp = torch.from_numpy(L_pq[rows]).to(dev)
        yt = torch.from_numpy(sid[sel + 1]).to(dev)

        s = t2s_t[ti]; m = (s >= 0).float()
        q = torch.zeros(len(sel), V, device=dev, dtype=torch.float32)
        q.scatter_add_(1, s.clamp_min(0), pp * m)
        q = q / q.sum(1, keepdim=True).clamp_min(1e-8)
        pt = pp / pp.sum(1, keepdim=True).clamp_min(1e-8)
        h_t = float((-(pt * pt.clamp_min(1e-9).log2()).sum(1)).mean())
        h_q = float((-(q * q.clamp_min(1e-9).log2()).sum(1)).mean())
        mass = q.gather(1, yt[:, None]).squeeze(1)
        print(f"{V:>6} {100*nanc/(ntok-1):7.2f}% {N/ntok:6.2f} {float(mass.mean()):10.3f} "
              f"{100*float((q.argmax(1)==yt).float().mean()):8.1f}% {h_t:11.3f} {h_q:10.3f} "
              f"{100*h_q/max(h_t,1e-6):8.0f}% {100*float((mass>0.95).float().mean()):6.0f}%")
        span_bits(bpe, sid, ends, sel[:a.span_n], anc, j0, L_ids, L_pq, tbytes, h_t, V)

    print("\nReading: retention = the fraction of the teacher's own uncertainty that reaches the student through the")
    print("span projection. Low retention => the KD target is a hard label with extra noise => the D3 gate is at risk")
    print("and the sequence-level fallback (or a span-aware target) is the live option. Architect's call.")
    print("STOP. No commit.")


if __name__ == "__main__":
    main()
