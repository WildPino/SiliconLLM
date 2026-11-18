#!/usr/bin/env python3
# Sparse Capacity, phase 1a (D-0): the rung-1 gate 1. Numbers only; sc1a_verdict.py decides.
# Protocol: docs/research/SPARSE_CAPACITY/F1A_RUNG1_GATE_PROTOCOL.md.
#
#   python benchmarks/sparse_capacity/sc1a.py --run [--data-dir ...] [--stages-dir ...]
#
# For each stage checkpoint of the rung-1 main run (stage_D dense ternary, stage_E and stage_F with the E32xh128 top-8
# bank): rebuild the model with the curriculum surgery up to that stage, load the weights strictly, then
#   planted control  BPB on the first 200k tokens of the internal val, the number the trainer recorded at stage exit:
#                    it must come back (the loader, the surgery and the alpha are right);
#   P62              BPB over the whole file (bytes checked against the declared size) and top-1;
#   temperature      a scalar T fitted on the first half of the P62 windows, BPB of the second half at T = 1 and at T.
# CPU only, fp32, a fixed number of threads (the GPU belongs to other work).
import argparse, hashlib, json, math, os, subprocess, sys, time
import numpy as np
import torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase64", "mve"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase55"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase62"))
from mve_model import MVEStudent, S0  # noqa: E402
from cartography import Bpe  # noqa: E402

DATA = "E:/SiliconLLM_Artifacts/workspace/kaggle_rung1/main_run/data"
STAGES_DIR = "D:/_ktmp/mainrun_final/stages_main_V2048_r26_CE"
STAGES = ("D", "E", "F")
SEQ, BATCH, EVAL_TOK = 512, 16, 200_000
TEMPS = (0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3)
NTHR = 4


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", ROOT] + list(a), capture_output=True, text=True).stdout.strip()


def build(ck, V):
    cfg = ck["cfg"]
    m = MVEStudent(V, **S0)
    if cfg.get("xproj_rank", 0) > 0:
        m.lowrank_xproj(cfg["xproj_rank"])
    st = ck["stage"]
    if st in "DEF":
        m.qat_ternary(alpha_sched=cfg["qat_alpha"] > 0)
        if cfg["qat_alpha"] > 0:
            m.set_qat_alpha(1.0)                       # stage D complete: alpha = 1 (the trainer's restore_qat_alpha)
    if st in "EF":
        m.upcycle_moe(dev_type="cpu", load_w=cfg["load_w"], seed=cfg["seed"], sparse=cfg["sparse_moe"])
        if cfg["recall"] == "on":
            m.add_recall(lam_nce=cfg["lam_nce"])
    m.load_state_dict(ck["model"], strict=True)
    m.eval()
    return m


@torch.no_grad()
def bpb_windows(m, stream, el, cap=0, temps=(1.0,), split_half=False):
    """The trainer's bpb_on: full windows of SEQ plus the ragged remainder, every target once. With temps, bits per
    temperature; with split_half, bits and bytes separately for the first and second half of the full windows."""
    n = min(cap, len(stream) - 1) if cap else len(stream) - 1
    full = n // SEQ
    T = torch.tensor(temps).view(-1, 1)
    bits = np.zeros((2, len(temps))); nb = np.zeros(2); top1 = 0; ntok = 0
    def acc(lg, y, h):
        nonlocal top1, ntok
        lp = torch.log_softmax(lg.unsqueeze(0) / T.view(-1, 1, 1), -1)
        g = lp.gather(-1, y.view(1, -1, 1).expand(len(temps), -1, 1)).squeeze(-1)
        bits[h] += (-g.sum(-1) / math.log(2)).numpy()
        nb[h] += float(el[y].sum()); top1 += int((lg.argmax(-1) == y).sum()); ntok += y.numel()
    for i in range(0, full, BATCH):
        k = min(BATCH, full - i)
        st = [(i + j) * SEQ for j in range(k)]
        x = torch.from_numpy(np.stack([stream[p:p + SEQ] for p in st]).astype(np.int64))
        y = torch.from_numpy(np.stack([stream[p + 1:p + 1 + SEQ] for p in st]).astype(np.int64))
        lg, _ = m(x, y)
        for j in range(k):
            acc(lg[j].float(), y[j], 1 if (split_half and i + j >= full // 2) else 0)
    rem = n - full * SEQ
    if rem > 0:
        p = full * SEQ
        x = torch.from_numpy(stream[p:p + rem][None].astype(np.int64)); y = torch.from_numpy(stream[p + 1:p + 1 + rem].astype(np.int64))
        lg, _ = m(x, y[None])
        acc(lg[0].float(), y, 1 if split_half else 0)
    return bits, nb, top1, ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--stages-dir", default=STAGES_DIR)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "sparse_capacity"))
    a = ap.parse_args()
    torch.set_num_threads(NTHR)
    head, upstream = git("rev-parse", "HEAD"), git("rev-parse", "@{u}")
    dirty = git("status", "--porcelain", "--", "benchmarks/sparse_capacity", "docs/research/SPARSE_CAPACITY",
                "benchmarks/phase64/mve", "benchmarks/phase55", "benchmarks/phase62")
    meta = json.load(open(os.path.join(a.data_dir, "meta_m0.json")))
    V = int(meta["V_student"])
    bpe = Bpe.load(os.path.join(a.data_dir, f"bpe{V}_ts.bin"))
    el = torch.tensor(bpe.exp_len, dtype=torch.long)
    p62 = np.fromfile(os.path.join(a.data_dir, "p62_m0.u16"), dtype=np.uint16).astype(np.int64)
    ids = np.memmap(os.path.join(a.data_dir, "ts_m0.u16"), dtype=np.uint16, mode="r")
    val = np.asarray(ids[int(meta["n_train_tok"]):int(meta["n_train_tok"]) + EVAL_TOK + 2]).astype(np.int64)
    ckp = {s: os.path.join(a.stages_dir, f"stage_{s}_ce_anchor_on_m0.pt") for s in STAGES}
    cfg = dict(git_head=head, pushed=(head == upstream), dirty_files=dirty.splitlines(), torch=torch.__version__,
               threads=NTHR, seq=SEQ, eval_tok=EVAL_TOK, temps=TEMPS, data_dir=a.data_dir,
               p62_sha256=sha256(os.path.join(a.data_dir, "p62_m0.u16")), p62_tok=len(p62), p62_bytes=meta["p62_bytes"],
               ckpt_sha256={s: sha256(p) for s, p in ckp.items()})
    for k, v in cfg.items():
        print(f"CONFIG {k} = {v}")
    if dirty or head != upstream:
        sys.exit("REFUSED: the protocol and this runner must be committed and pushed before the reading")
    run_dir = os.path.join(a.out, "f1a_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir)
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    out = open(os.path.join(run_dir, "stages.jsonl"), "w")
    for s in STAGES:
        t0 = time.time()
        ck = torch.load(ckp[s], map_location="cpu", weights_only=False)
        m = build(ck, V)
        nparam = sum(p.numel() for p in m.parameters())
        bv, nv, _, _ = bpb_windows(m, val, el, cap=EVAL_TOK)
        bp, npb, t1, nt = bpb_windows(m, p62, el, temps=TEMPS, split_half=True)
        i1 = TEMPS.index(1.0); fit = int(np.argmin(bp[0] / npb[0]))
        rec = dict(stage=s, gstep=ck["gstep"], params=nparam, recorded_val_bpb=ck["bpb"],
                   val_bpb=float(bv[0, 0] / nv[0]), val_bytes=int(nv[0]),
                   p62_bpb=float(bp[:, i1].sum() / npb.sum()), p62_top1=t1 / nt,
                   p62_bytes_scored=int(npb.sum()), p62_first_tok_bytes=int(el[int(p62[0])]),
                   t_fit=TEMPS[fit], p62_B_bpb_t1=float(bp[1, i1] / npb[1]), p62_B_bpb_tfit=float(bp[1, fit] / npb[1]),
                   seconds=time.time() - t0)
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(json.dumps(rec), flush=True)
    print(f"DONE -> {run_dir}\nverdict: python benchmarks/sparse_capacity/sc1a_verdict.py {run_dir}")


if __name__ == "__main__":
    main()
