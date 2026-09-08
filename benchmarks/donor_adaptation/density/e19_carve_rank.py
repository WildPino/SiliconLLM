#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E19 part B -- does an FFN carve RANK?

Pre-registration: docs/research/donor_adaptation/briefs/BRIEF_E19_DOES_THE_CARVE_RANK.md,
pushed before any arm ran (commit 3173edc).

D0 and D0c scored every carve arm in BPB ONLY.  That is the exact omission E18 caught in T2b, where
five arms spanning 2.4 BPB all turned out to sit at the greedy-agreement floor.  No carve of this
donor has ever been read in generation, at any granularity.

The router is an ORACLE: it picks the k experts holding the largest squared-activation mass AFTER
computing the activations.  No trainable router can beat it.  If the ceiling cannot rank, nothing
below it can.

WHAT IS REUSED AND HOW IT IS PROVEN
  * The PARTITIONS are D0c's own, loaded from results/d0c_labels/labels_E*.npz -- the files its run
    wrote.  Nothing is re-clustered, so the B3 seeding repair and CLUSTER_SEED are inherited by
    construction rather than by re-execution.
  * The HOOK is restated here because d0c_granularity.py loads a model at module level and cannot
    be imported.  It is therefore gated BEHAVIOURALLY instead of textually: G-C2 requires this
    file's BPB to reproduce D0c's published BPB to < 1e-6 on all four arms that have one.  A hook
    that reproduces four BPBs to six decimals is D0c's hook.  (E18's G-L2 is the same move.)
  * The slice, the prompts and the reference are the frozen ones: heldout 24x512 sha a1a48dc9...,
    e6_generate.PROMPTS, results/e6/ref.json.

Env: D_THREADS (6), E19_ONLY (comma list), E19_SMOKE (1 = 2 prompts, 8 new tokens, 2 seqs, 1 layer)
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ENGDIR = os.path.abspath(os.path.join(HERE, "..", "engine"))
sys.path.insert(0, HERE)
sys.path.insert(0, ENGDIR)

import common as C                                          # noqa: E402
import d0_coactivation as DC                                # noqa: E402
from e6_generate import PROMPTS, N_NEW                      # noqa: E402

THREADS = int(os.environ.get("D_THREADS", "6"))
torch.set_num_threads(THREADS)
SMOKE = os.environ.get("E19_SMOKE", "0") == "1"
ONLY = [x.strip() for x in os.environ.get("E19_ONLY", "").split(",") if x.strip()]

HF = "Qwen/Qwen2.5-1.5B"
LABDIR = os.path.join(C.RESULTS, "d0c_labels")
E6REF = os.path.join(ENGDIR, "results", "e6", "ref.json")
OUT = os.path.join(ENGDIR, "results", "e19_carve_rank%s.json" % ("_smoke" if SMOKE else ""))
EXPECT_IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
LN2 = 0.6931471805599453

#                tag      E     k    kind      role
ARMS = [("base",  None, None, None),
        ("FULL",   256,  256, "coact"),
        ("V52",    256,  133, "coact"),
        ("S1",     256,   64, "coact"),
        ("A0",      32,    8, "coact"),
        ("N0",      32,    8, "null"),
        ("D10",    256,   26, "coact")]

# D0c's published BPB, quoted and never re-derived here -- the replication gate G-C2
D0C_BPB = {"base": 0.7675949641196624, "S1": 1.3838675903770394,
           "A0": 1.8582178498441495, "N0": 2.5787313453805165}
REPL_TOL = 1e-6

CHANCE = 4.069819                       # log2(151936) / 4.229452, the frozen heldout slice
FLOOR = 12                              # E18 part A: best constant predictor on this reference
MARGIN = 2                              # E17: a change rewriting 49% of the output moved 2 tokens
AT_FLOOR_MAX = FLOOR + MARGIN           # 14
RANKS_MIN = 80                          # a registered convention, brief s5
ACT_TOL = 0.002                         # D0c s3.2's own tolerance


def log(*a):
    print(*a, flush=True)


def label(m):
    if m <= AT_FLOOR_MAX:
        return "AT-FLOOR"
    if m >= RANKS_MIN:
        return "RANKS"
    return "PARTIAL"


# ================================================================= the carve, as a forward hook
HOOKS = []
STATS = {"kept": 0.0, "ntok": 0}


def install(model, layer_ids, lab_map, k):
    """ORACLE top-k router: picks the k experts holding the largest squared-activation mass, then
    zeroes the rest before down_proj.  An upper bound on any trainable router."""
    for L in layer_ids:
        lab = torch.from_numpy(lab_map[L].astype(np.int64))
        E = int(lab.max()) + 1
        oh = torch.zeros(len(lab), E)
        oh[torch.arange(len(lab)), lab] = 1.0
        sizes = oh.sum(0)

        def mk(lab=lab, oh=oh, sizes=sizes, E=E, k=k):
            def pre(mod, args):
                h = args[0]
                flat = h.reshape(-1, h.shape[-1])
                sel = ((flat ** 2) @ oh).topk(k, dim=1).indices
                keep = torch.zeros(flat.shape[0], E, dtype=torch.bool)
                keep.scatter_(1, sel, True)
                STATS["kept"] += float((keep.float() @ sizes).sum())
                STATS["ntok"] += int(flat.shape[0])
                return ((flat * keep[:, lab]).reshape(h.shape),) + args[1:]
            return pre
        HOOKS.append(model.model.layers[L].mlp.down_proj.register_forward_pre_hook(mk()))


def clear():
    while HOOKS:
        HOOKS.pop().remove()


def load_labels(E):
    """D0c's own partition, read from the cache ITS run wrote.  Nothing is re-clustered."""
    cache = os.path.join(LABDIR, "labels_E%d.npz" % E)
    if not os.path.exists(cache):
        raise SystemExit("missing partition cache %s -- STOP" % cache)
    z = np.load(cache)
    lab = {int(k[1:]): z[k] for k in z.files if k.startswith("c")}
    null = {int(k[1:]): z[k] for k in z.files if k.startswith("n")}
    return lab, null


def greedy(model, prompt_ids, n_new):
    """Exactly e6_generate.stage_ref's loop, and E18 part B's: full forward each step, argmax,
    no KV cache, so the reference is reproduced by construction."""
    ids = list(prompt_ids)
    with torch.no_grad():
        for _ in range(n_new):
            lg = model(torch.tensor([ids])).logits[0, -1].float()
            ids.append(int(torch.argmax(lg)))
    return ids


def main():
    t_start = time.time()
    n_new = 8 if SMOKE else N_NEW
    prompts = PROMPTS[:2] if SMOKE else PROMPTS

    ref = json.load(open(E6REF, encoding="utf-8"))[HF]
    refids = {i: r["ids"][-N_NEW:] for i, r in enumerate(ref)}

    log("== loading %s ==" % HF)
    model, tok = C.load_model(dtype=torch.float32)
    model.eval()
    A = C.arch(model)
    d_ffn, n_layers = A["d_ffn"], A["n_layers"]
    assert model.config.vocab_size == 151936, model.config.vocab_size

    # ---- the eval slice, frozen and hash-checked, same as D0c
    ids_ev, byts_ev, meta_ev = C.get_slice(tok, "heldout", DC.N_EVAL, DC.SEQ_LEN_EVAL, DC.SEED_EVAL)
    if meta_ev["ids_sha256"] != EXPECT_IDS_SHA:
        raise SystemExit("SLICE HASH MISMATCH: %s -- STOP" % meta_ev["ids_sha256"])
    if SMOKE:
        ids_ev, byts_ev = ids_ev[:2], byts_ev[:2]
    B_TOT = float(byts_ev.sum())
    log("slice %dx%d, %d scored bytes  |  ids_sha256 OK  |  d_ffn=%d n_layers=%d"
        % (ids_ev.shape[0], ids_ev.shape[1], int(B_TOT), d_ffn, n_layers))

    CARVE_LAYERS = [27] if SMOKE else list(range(n_layers))

    # ---- prompt ids, checked against the ones E6 actually used
    e6eng = json.load(open(os.path.join(ENGDIR, "results", "e6", "engine.json"), encoding="utf-8"))
    pids = []
    for i, p in enumerate(prompts):
        q = tok(p)["input_ids"]
        assert q == e6eng["prompt_ids"][i], "prompt %d differs from E6's stored ids" % i
        pids.append(q)

    labs = {}
    for E in sorted({a[1] for a in ARMS if a[1]}):
        c, nl = load_labels(E)
        labs[E] = {"coact": c, "null": nl}
        log("  partition E=%d loaded from D0c's cache (%d layers)" % (E, len(c)))

    out = {"brief": "briefs/BRIEF_E19_DOES_THE_CARVE_RANK.md (3173edc)",
           "instrument": "PyTorch fp32 eager; ORACLE top-k router = upper bound on any router",
           "partitions": "D0c's own caches results/d0c_labels/labels_E*.npz -- not re-clustered",
           "hook_provenance": "restated (d0c_granularity.py loads a model at import); gated "
                              "BEHAVIOURALLY by G-C2 -- BPB reproduced to <1e-6 on 4 published arms",
           "model": HF, "smoke": SMOKE, "threads": THREADS, "n_new": n_new,
           "n_prompts": len(prompts), "carve_layers": len(CARVE_LAYERS),
           "bands": {"floor_E18_partA": FLOOR, "margin_E17": MARGIN,
                     "at_floor_max": AT_FLOOR_MAX, "ranks_min": RANKS_MIN,
                     "ceiling": len(prompts) * n_new},
           "chance_bpb": CHANCE, "d0c_published_bpb": D0C_BPB, "arms": {}}
    if os.path.exists(OUT):
        try:
            out["arms"] = json.load(open(OUT, encoding="utf-8")).get("arms", {})
        except Exception:
            pass

    base_ids = None
    todo = [a for a in ARMS if (not ONLY or a[0] in ONLY or a[0] == "base")]
    for tag, E, k, kind in todo:
        if tag in out["arms"] and "matched" in out["arms"][tag]:
            r = out["arms"][tag]
            log("  %-5s CACHED  %d/%d  BPB %.9f" % (tag, r["matched"], r["counted"], r["bpb"]))
            if tag == "base":
                base_ids = r["ids"]
            continue
        t0 = time.time()
        STATS["kept"], STATS["ntok"] = 0.0, 0
        clear()
        if kind is not None:
            install(model, CARVE_LAYERS, labs[E][kind], k)

        # ---- BPB on the frozen slice
        with torch.no_grad():
            nats = []
            for i in range(ids_ev.shape[0]):
                ch = ids_ev[i:i + 1]
                lg = model(ch).logits.float()
                lp = torch.nn.functional.log_softmax(lg[:, :-1], dim=-1)
                nats.append(-lp.gather(-1, ch[:, 1:].unsqueeze(-1)).squeeze(-1)[0].double().numpy())
        bpb = float(np.stack(nats).sum() / (LN2 * B_TOT))
        ach = 1.0 if kind is None else STATS["kept"] / float(STATS["ntok"] * d_ffn)

        # ---- greedy agreement against the donor's OWN continuations
        allids, matched, counted, first_div, per_prompt = [], 0, 0, None, []
        for i in range(len(prompts)):
            ids = greedy(model, pids[i], n_new)
            new = ids[-n_new:]
            allids.append(new)
            theirs = refids[i][:n_new]
            m = sum(1 for a, b in zip(new, theirs) if a == b)
            d = next((j for j, (a, b) in enumerate(zip(new, theirs)) if a != b), None)
            if d is not None and first_div is None:
                first_div = [i, d]
            per_prompt.append({"prompt": i, "matched": m, "of": n_new, "diverges_at": d})
            matched += m
            counted += n_new
        clear()

        rec = {"arm": tag, "E": E, "k": k, "kind": kind,
               "nominal_activation": None if k is None else k / float(E),
               "achieved_activation": ach,
               "bpb": bpb, "vs_chance": bpb - CHANCE,
               "matched": matched, "counted": counted, "agree": matched / float(counted),
               "first_div": first_div, "per_prompt": per_prompt, "ids": allids,
               "seconds": time.time() - t0,
               "text": [tok.decode(x) for x in allids]}

        # G-C4: achieved activation must match nominal
        if k is not None:
            rec["G_C4_activation_matches"] = bool(abs(ach - k / float(E)) <= ACT_TOL)
        # G-C2: replication against D0c's published BPB
        if tag in D0C_BPB and not SMOKE:
            d = abs(bpb - D0C_BPB[tag])
            rec["G_C2"] = {"d0c": D0C_BPB[tag], "here": bpb, "abs_diff": d,
                           "within_tolerance": bool(d < REPL_TOL)}
        # controls
        if tag == "base":
            base_ids = allids
            rec["G_C0"] = "FIRES" if matched == counted else "VOID"
        elif tag == "FULL":
            rec["G_C1_token_identical_to_base"] = bool(allids == base_ids)
            rec["G_C1"] = "FIRES" if allids == base_ids else "VOID"
        if tag not in ("base", "FULL"):
            rec["G_C3"] = label(matched) if not SMOKE else "NOT-APPLICABLE (smoke)"

        out["arms"][tag] = rec
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        log("  %-5s act %.4f  BPB %.6f (%+.6f)  %3d/%-3d %6.2f%%  div %-8s %s  [%.0fs]"
            % (tag, ach, bpb, bpb - CHANCE, matched, counted,
               100.0 * matched / counted, str(first_div),
               rec.get("G_C3", rec.get("G_C0", rec.get("G_C1", ""))), rec["seconds"]))

    # ---- summary
    A = out["arms"]
    void = []
    if "base" in A and A["base"].get("G_C0") != "FIRES":
        void.append("G-C0: base did not reproduce the reference")
    if "FULL" in A and A["FULL"].get("G_C1") != "FIRES":
        void.append("G-C1: the carve path is not lossless at k=E")
    bad = [t for t in D0C_BPB if t in A and "G_C2" in A[t]
           and not A[t]["G_C2"]["within_tolerance"]]
    if bad:
        void.append("G-C2: BPB did not replicate D0c for %s" % ",".join(bad))
    out["VOID"] = void
    out["G_C3"] = {t: A[t]["G_C3"] for t in A if "G_C3" in A[t]}
    out["seconds_total"] = time.time() - t_start

    log("\n  bands: floor %d (E18 part A), margin %d (E17), AT-FLOOR <= %d, RANKS >= %d, ceiling %d"
        % (FLOOR, MARGIN, AT_FLOOR_MAX, RANKS_MIN, len(prompts) * n_new))
    log("  G_C3  %s" % json.dumps(out["G_C3"]))
    log("\n  carve ladder (activation -> BPB vs chance -> agreement):")
    for tag, E, k, kind in ARMS:
        if tag in A:
            r = A[tag]
            log("    %-5s %-6s act %.4f  %+.6f  ->  %3d/%d"
                % (tag, (kind or "-"), r["achieved_activation"], r["vs_chance"],
                   r["matched"], r["counted"]))
    log("\n  VOID: %s" % (", ".join(void) if void else "none"))
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    log("wrote %s  [%.0fs total]" % (OUT, out["seconds_total"]))


if __name__ == "__main__":
    main()
