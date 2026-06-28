#!/usr/bin/env python3
# Phase 56 STAGE 1 gate (a) CAPACITY FIRST - VSA/sketch superposition associative-memory recall@N.
#   The recall slot stores N (key->value) bindings SUPERPOSED into ONE int8 accumulator per head (the "state",
#   D int8 = D bytes; D=8192 -> 8KB). Binding = bipolar MAP (elementwise product); self-inverse unbind.
#     store : acc = clip_int8( sum_i  key_i (*) value_atom[val_i] )            <- the 8KB int8 state
#     recall: dehat = acc (*) key_i ; pred = argmax_v  dehat . Vc[v]  (cleanup over V_atoms value codebook)
#   Capacity = recall@N before (1) key crosstalk and (2) int8 saturation destroy it. This is the gate that
#   CAN DIE: we run it expecting it might. Keys ORTHOGONAL (Hadamard rows, no crosstalk, needs N<=D) vs
#   RANDOM (+-1, crosstalk ~ sqrt(N/D)). 8:1 chunking: to hold TARGET=128K items use `chunks` accumulators,
#   so the per-accumulator load is TARGET/chunks (=16K at 8:1) -> the gate must reach recall@(TARGET/chunks).
#
#   Sweep D (accumulator dim) x V_atoms (cleanup codebook). Pick the operating point that passes BOTH this gate
#   AND the cost gate (phase56_ivfpq_profile, to be re-measured at these D) jointly on Zen2. Promote iff both.
#   int8 (real state) vs float (headroom) reported side by side -> shows whether int8 saturation is the limiter.
#
# Smoke: python benchmarks/phase56/phase56_vsa_capacity.py --smoke
# Real : python benchmarks/phase56/phase56_vsa_capacity.py  (D 8192,16384,32768 x V 256,512,1024; HOURS)
import argparse, os, json, time
import numpy as np

def popcount32(x):                                   # vectorized popcount on uint32 array
    x = x.astype(np.uint32)
    x = x - ((x >> 1) & np.uint32(0x55555555))
    x = (x & np.uint32(0x33333333)) + ((x >> 2) & np.uint32(0x33333333))
    x = (x + (x >> 4)) & np.uint32(0x0f0f0f0f)
    return (x * np.uint32(0x01010101)) >> 24

def hadamard_rows(row_idx, D):                       # +-1 Sylvester-Hadamard rows WITHOUT the full DxD matrix
    j = np.arange(D, dtype=np.uint32)                # row r, col j = (-1)^popcount(r & j); rows are orthogonal
    out = np.empty((len(row_idx), D), dtype=np.int8)
    for t, r in enumerate(row_idx):
        pc = popcount32(np.uint32(r) & j) & np.uint32(1)
        out[t] = np.where(pc == 0, 1, -1).astype(np.int8)
    return out

def make_keys(atoms, N, D, rng):
    if atoms == "ortho":
        if N > D: return None                        # orthogonal infeasible: can't fit N>D orthogonal keys
        return hadamard_rows(np.arange(1, N + 1), D) # skip row 0 (all-ones) -> use rows 1..N
    return rng.choice(np.array([-1, 1], np.int8), size=(N, D))

def run_cell(D, V, N, atoms, use_int8, trials, blk, rng_seed):
    rng = np.random.RandomState(rng_seed)
    accs = np.zeros(trials); snrs = np.zeros(trials)
    for tr in range(trials):
        r = np.random.RandomState(rng_seed + 1000 * tr + 7)
        K = make_keys(atoms, N, D, r)
        if K is None: return None                    # infeasible (ortho, N>D)
        Vc = r.choice(np.array([-1, 1], np.int8), size=(V, D))   # value cleanup codebook
        val = r.randint(0, V, size=N)                            # each item's true value index
        # ---- STORE: superpose bindings into one accumulator (block over items to bound memory) ----
        acc = np.zeros(D, dtype=np.int32)
        for b in range(0, N, blk):
            e = min(b + blk, N)
            acc += (K[b:e].astype(np.int32) * Vc[val[b:e]].astype(np.int32)).sum(0)
        if use_int8:
            acc = np.clip(acc, -127, 127).astype(np.int32)       # the REAL 8KB int8 state (saturating)
        # ---- RECALL: unbind + cleanup over V_atoms (block over items) ----
        hit = 0; sig = 0.0; cross = 0.0
        VcT = Vc.T.astype(np.float32)                            # cleanup codebook (BLAS-friendly; int @ has no BLAS)
        for b in range(0, N, blk):
            e = min(b + blk, N)
            dehat = (acc[None, :] * K[b:e].astype(np.int32)).astype(np.float32)  # (blk, D); exact <2^24 in int8 state
            scores = dehat @ VcT                                 # (blk, V) cleanup scores via BLAS sgemm
            pred = scores.argmax(1)
            hit += int((pred == val[b:e]).sum())
            rows = np.arange(e - b)
            s = scores[rows, val[b:e]].astype(np.float64)        # signal at true value
            sc = scores.copy(); sc[rows, val[b:e]] = -1 << 30     # mask true -> crosstalk stats
            sig += s.sum(); cross += np.abs(sc.max(1)).astype(np.float64).sum()
        accs[tr] = hit / N
        snrs[tr] = (sig / N) / max(cross / N, 1e-9)
    return float(accs.mean()), float(accs.std()), float(snrs.mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", default="8192,16384,32768")
    ap.add_argument("--vatoms", default="256,512,1024")
    ap.add_argument("--N", default="", help="explicit per-accumulator loads; default derived from target/chunks")
    ap.add_argument("--target", type=int, default=131072, help="total items to hold across chunks (128K context)")
    ap.add_argument("--chunks", type=int, default=8, help="8:1 chunking -> per-acc load = target/chunks")
    ap.add_argument("--atoms", default="ortho,random")
    ap.add_argument("--int8", default="int8,float", help="int8=real saturating state; float=headroom reference")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--blk", type=int, default=1024)
    ap.add_argument("--pass-recall", type=float, default=0.90, help="soft PASS line at N=target/chunks (Architect reads)")
    ap.add_argument("--out-dir", default="results/phase56")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.D="512,1024"; a.vatoms="32,64"; a.N="64,256,512"; a.target=4096; a.chunks=8; a.trials=1
    Ds=[int(x) for x in a.D.split(",")]; Vs=[int(x) for x in a.vatoms.split(",")]
    atoms=a.atoms.split(","); i8s=[s=="int8" for s in a.int8.split(",")]
    per_acc=a.target//a.chunks
    Ns=[int(x) for x in a.N.split(",")] if a.N else sorted({a.target//(a.chunks*2), per_acc, (a.target//a.chunks)*3//2})
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"VSA capacity gate | D={Ds} V_atoms={Vs} | N(per-acc)={Ns} | target={a.target} chunks={a.chunks} -> per-acc load={per_acc}",flush=True)
    print(f"  atoms={atoms} state={['int8' if x else 'float' for x in i8s]} trials={a.trials} | PASS line recall@{per_acc}>={a.pass_recall} (Architect reads)",flush=True)
    rows=[]; t0=time.time()
    def save():
        hdr=f"==== VSA superposition recall@N (per-accumulator) | target={a.target} chunks={a.chunks} per-acc={per_acc} ===="
        lines=[hdr,"  D      V_atoms  atoms   state   N       recall(±std)   SNR    note"]
        for rr in rows:
            note=""
            if rr["N"]==per_acc: note=("PASS@perAcc" if rr["recall"]>=a.pass_recall else "FAIL@perAcc") if rr["recall"]>=0 else "infeasible"
            rec = f'{rr["recall"]*100:5.1f}%+-{rr["std"]*100:4.1f}' if rr["recall"]>=0 else "  --  "
            snr = f'{rr["snr"]:6.2f}' if rr["recall"]>=0 else "  --  "
            lines.append(f'  {rr["D"]:6d} {rr["V"]:6d}  {rr["atoms"]:6s} {rr["state"]:6s} {rr["N"]:7d}  {rec:>14s}  {snr}  {note}')
        txt="\n".join(lines)
        open(os.path.join(a.out_dir,"vsa_capacity.txt"),"w",encoding="utf-8").write(txt)
        json.dump({"config":vars(a),"per_acc":per_acc,"rows":rows}, open(os.path.join(a.out_dir,"vsa_capacity.json"),"w",encoding="utf-8"), indent=1)
        return txt
    for D in Ds:
        for V in Vs:
            for atom in atoms:
                for use_i8 in i8s:
                    for N in Ns:
                        r=run_cell(D,V,N,atom,use_i8,a.trials,a.blk,rng_seed=12345)
                        st="int8" if use_i8 else "float"
                        if r is None:
                            rows.append({"D":D,"V":V,"atoms":atom,"state":st,"N":N,"recall":-1,"std":0,"snr":0})
                            print(f"  D{D} V{V} {atom:6s} {st:5s} N{N:6d} -> infeasible (ortho N>D)",flush=True)
                        else:
                            rec,std,snr=r
                            rows.append({"D":D,"V":V,"atoms":atom,"state":st,"N":N,"recall":rec,"std":std,"snr":snr})
                            print(f"  D{D} V{V} {atom:6s} {st:5s} N{N:6d} -> recall={rec*100:5.1f}% SNR={snr:5.2f}  ({time.time()-t0:.0f}s)",flush=True)
                        save()
    print("\n"+save()+f"\n  [saved -> {a.out_dir}/vsa_capacity.txt + .json]",flush=True)
    print("STOP. capacity gate (a). Pass here AND cost gate on Zen2 -> promote. No commit.",flush=True)

if __name__=="__main__": main()
