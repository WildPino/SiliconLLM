#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D0c -- carved BPB as a function of expert granularity E, each arm paired with a
random-partition null at the SAME E.

Pre-registration: docs/research/donor_adaptation/briefs/BRIEF_D0C_CARVE_GRANULARITY.md (4a98c89).
Nothing in this file may be changed to chase an outcome; section 4's thresholds are hard-coded
from the brief and the label is read off them mechanically.

This is a tracked reconstruction of the Part III scratchpad driver
(`d0_carved_bpb_paired.json` -> env.command_line), which was not reproducible from the repo.
The carve, the oracle router, the partitioner call, the B3 seeding repair, the null-shuffle seed
formula and the paired-bootstrap statistics are all carried over unchanged; what is added is the
E sweep, the per-arm achieved-activation measurement (section 3.2), the replication gate (A0/N0
against Part III's standing numbers), and resumability.

Env:
  D_THREADS       torch threads              (default 6 -- Part III's value, keep it for replication)
  D0C_SMOKE       1 = tiny proof-of-path run (2 sequences, 1 layer, no gate, separate outputs)
  D0C_SECONDARY   1 = also run the conditional E=256 arms (see brief section 2: interleaved floor)
  D0C_FORCE       1 = do not abort when the replication gate fails (for diagnosis only)
  D0C_PARTITIONS_ONLY  comma list of E, e.g. "32,64,128": build + cache those partitions and exit.
                  Partitioning is ~10-19 s per layer per E and is pure setup, so it is split out
                  and done while the machine is otherwise busy.  Deterministic given the seed.
"""
import sys, os, json, math, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C            # noqa: E402
import d0_layout as D0        # noqa: E402
import d0_coactivation as DC  # noqa: E402

THREADS = int(os.environ.get("D_THREADS", "6"))
torch.set_num_threads(THREADS)
SMOKE = os.environ.get("D0C_SMOKE", "0") == "1"
SECONDARY = os.environ.get("D0C_SECONDARY", "0") == "1"
FORCE = os.environ.get("D0C_FORCE", "0") == "1"
PART_ONLY = [int(x) for x in os.environ.get("D0C_PARTITIONS_ONLY", "").split(",") if x.strip()]

LN2 = math.log(2.0)
CLUSTER_SEED = DC.CLUSTER_SEED          # 7
SUFFIX = "_smoke" if SMOKE else ""
OUT = os.path.join(C.RESULTS, "d0c_granularity"
                   + ("_partitions" if PART_ONLY else SUFFIX) + ".json")
ARMDIR = os.path.join(C.RESULTS, "d0c_arms" + SUFFIX)
LABDIR = os.path.join(C.RESULTS, "d0c_labels")
os.makedirs(ARMDIR, exist_ok=True)
os.makedirs(LABDIR, exist_ok=True)

# ---------------------------------------------------------------- brief s4, verbatim thresholds
DELTA_THRESH = 0.20        # BPB, = 40 sigma_seed
GAP_THRESH = 0.10          # BPB
SIGMA_SEED = 0.005
BASELINE_STANDING = 0.7675949641196624
STANDING = {               # Part III, results/d0_carved_bpb_paired.json
    "baseline_bpb": 0.7675949641196624,
    "A0_delta": 1.090622885724487,      # all_coact  vs baseline
    "N0_delta": 1.8111363812608539,     # all_null   vs baseline
    "G32": -0.7205134955363668,         # all_coact  vs all_null
}
REPLICATION_TOL = 1e-6

# ---------------------------------------------------------------- arms, brief section 3
#   tag,   E,   k,   kind
PRIMARY = [("baseline", None, None, None),
           ("A0", 32, 8, "coact"), ("N0", 32, 8, "null"),
           ("A1", 64, 16, "coact"), ("N1", 64, 16, "null"),
           ("A2", 128, 32, "coact"), ("N2", 128, 32, "null")]
SECONDARY_ARMS = [("S1", 256, 64, "coact"), ("S1n", 256, 64, "null")]
ARMS = PRIMARY + (SECONDARY_ARMS if SECONDARY else [])
if SMOKE:
    ARMS = [("baseline", None, None, None), ("A0", 32, 8, "coact"), ("A2", 128, 32, "coact")]


def log(*a):
    print(*a, flush=True)


# ================================================================= model + slice
t0 = time.time()
m, tok = C.load_model()
A = C.arch(m)
d_ffn, n_layers = A["d_ffn"], A["n_layers"]
log("loaded in %.0fs  |  threads=%d  |  d_ffn=%d n_layers=%d"
    % (time.time() - t0, torch.get_num_threads(), d_ffn, n_layers))

ids_ev, byts_ev, meta_ev = C.get_slice(tok, "heldout", DC.N_EVAL, DC.SEQ_LEN_EVAL, DC.SEED_EVAL)
EXPECT_IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
if meta_ev["ids_sha256"] != EXPECT_IDS_SHA:
    raise SystemExit("SLICE HASH MISMATCH: %s != %s -- STOP"
                     % (meta_ev["ids_sha256"], EXPECT_IDS_SHA))
if SMOKE:
    ids_ev, byts_ev = ids_ev[:2], byts_ev[:2]
n_seq, seq_len = ids_ev.shape
n_pred = n_seq * (seq_len - 1)
log("slice %dx%d -> %d predicted tokens, %d scored bytes  |  ids_sha256 OK"
    % (n_seq, seq_len, n_pred, int(byts_ev.sum())))

CARVE_LAYERS = [27] if SMOKE else list(range(n_layers))

# ================================================================= partitions, cached per E
K_ACT = int(round(DC.P_PRIMARY * d_ffn))          # 896 = top-10% co-activation mask


def partitions_for(E):
    """(coact labels, null labels) for every layer, at expert count E.  Cached to disk.

    Carried over from the Part III driver unchanged:
      * the mask is the top-K_ACT co-activated neurons per fit token (`d0_masks/fit_L*.npz`);
      * `torch.manual_seed(CLUSTER_SEED)` immediately before `balanced_labels` -- the B3 repair,
        without which `torch.svd_lowrank` makes the partition non-reproducible;
      * the null is a permutation of the SAME label vector, so expert sizes are identical and only
        the assignment is randomised.  Seed formula `1000 + E` is Part III's (it used 1032).
    """
    cache = os.path.join(LABDIR, "labels_E%d.npz" % E)
    if os.path.exists(cache):
        z = np.load(cache)
        lab = {int(k[1:]): z[k] for k in z.files if k.startswith("c")}
        null = {int(k[1:]): z[k] for k in z.files if k.startswith("n")}
        log("  partitions E=%d: loaded from cache (%d layers)" % (E, len(lab)))
        return lab, null
    lab, null = {}, {}
    tp = time.time()
    for L in range(n_layers):
        zz = np.load(os.path.join(DC.MASKDIR, "fit_L%02d.npz" % L))
        idx = zz["idx"][:, :K_ACT].astype(np.int64)
        B = np.zeros((idx.shape[0], d_ffn), dtype=bool)
        B[np.arange(idx.shape[0])[:, None], idx] = True
        torch.manual_seed(CLUSTER_SEED)                       # B3 repair -> deterministic
        lb = np.asarray(D0.balanced_labels(torch.from_numpy(B), E, CLUSTER_SEED))
        lab[L] = lb
        ln = lb.copy()
        np.random.default_rng(1000 + E).shuffle(ln)
        null[L] = ln
        del B, idx
    payload = {}
    payload.update({"c%d" % L: lab[L] for L in lab})
    payload.update({"n%d" % L: null[L] for L in null})
    np.savez_compressed(cache, **payload)
    log("  partitions E=%d: computed in %.0fs and cached -> %s" % (E, time.time() - tp, cache))
    return lab, null


def check_partition(E, lab, null):
    """Every expert must hold exactly d_ffn/E neurons -- otherwise k/E is not an activation
    fraction and section 3.2's matched comparison is void."""
    exp = d_ffn // E
    sizes_ok, nulls_ok = True, True
    for L in lab:
        bc = np.bincount(lab[L], minlength=E)
        bn = np.bincount(null[L], minlength=E)
        if len(bc) != E or not (bc == exp).all():
            sizes_ok = False
        if len(bn) != E or not (bn == exp).all():
            nulls_ok = False
    return {"E": E, "neurons_per_expert_expected": exp,
            "coact_all_experts_exact": bool(sizes_ok), "null_all_experts_exact": bool(nulls_ok)}


# ================================================================= the carve, as a forward hook
HOOKS = []
STATS = {"kept": 0.0, "ntok": 0}


def install(layer_ids, lab_map, k):
    """ORACLE top-k router: picks the k experts holding the largest squared-activation mass, then
    zeroes the rest before down_proj.  An upper bound on any trainable router.  Identical to
    Part III's hook, with an added accumulator for the achieved activation fraction."""
    for L in layer_ids:
        lab = torch.from_numpy(lab_map[L].astype(np.int64))
        E = int(lab.max()) + 1
        oh = torch.zeros(len(lab), E)
        oh[torch.arange(len(lab)), lab] = 1.0
        sizes = oh.sum(0)                                     # neurons per expert

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
        HOOKS.append(m.model.layers[L].mlp.down_proj.register_forward_pre_hook(mk()))


def clear():
    while HOOKS:
        HOOKS.pop().remove()


def run_arm(tag, E, k, kind, lab_map):
    """Per-sequence, per-token nats for one arm.  Cached so the run is resumable."""
    path = os.path.join(ARMDIR, tag + ".npy")
    meta_path = os.path.join(ARMDIR, tag + ".meta.json")
    if os.path.exists(path) and os.path.exists(meta_path):
        a = np.load(path)
        mt = json.load(open(meta_path))
        log("  %-9s CACHED  BPB = %.9f  achieved_activation = %.6f"
            % (tag, a.sum() / (LN2 * float(byts_ev.sum())), mt["achieved_activation"]))
        return a, mt
    STATS["kept"], STATS["ntok"] = 0.0, 0
    clear()
    if kind is not None:
        install(CARVE_LAYERS, lab_map, k)
    t = time.time()
    chunks = []
    for i in range(n_seq):
        ch = ids_ev[i:i + 1]
        lg = m(ch).logits.float()
        lp = torch.nn.functional.log_softmax(lg[:, :-1], dim=-1)
        chunks.append(-lp.gather(-1, ch[:, 1:].unsqueeze(-1)).squeeze(-1)[0].double().numpy())
    clear()
    a = np.stack(chunks)
    secs = time.time() - t
    ach = 1.0 if kind is None else STATS["kept"] / (STATS["ntok"] * d_ffn)
    mt = {"arm": tag, "E": E, "k": k, "kind": kind,
          "nominal_activation": None if k is None else k / E,
          "achieved_activation": float(ach),
          "n_carved_layers": 0 if kind is None else len(CARVE_LAYERS),
          "hook_token_layer_calls": int(STATS["ntok"]), "seconds": float(secs)}
    np.save(path, a)
    json.dump(mt, open(meta_path, "w"), indent=1)
    log("  %-9s BPB = %.9f  achieved_activation = %.6f  (%.0fs)"
        % (tag, a.sum() / (LN2 * float(byts_ev.sum())), ach, secs))
    return a, mt


# ================================================================= paired statistics
byts = byts_ev.numpy()
B_TOT = float(byts.sum())


def bpb_of(a):
    return float(a.sum() / (LN2 * B_TOT))


def paired_d(d, n_boot=20000, seed=7):
    """Paired error on a per-sequence x per-token nats DIFFERENCE array.  The sequence bootstrap
    is the honest one (tokens within a sequence are not independent); the per-token SE is kept
    because Part III reported it and the two must stay comparable."""
    d_tok = d.ravel()
    d_seq = d.sum(1)
    delta = float(d_seq.sum() / (LN2 * B_TOT))
    rng = np.random.default_rng(seed)
    n = len(byts)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        kk = rng.integers(0, n, n)
        vals[b] = d_seq[kk].sum() / (LN2 * byts[kk].sum())
    se_seq = float(np.std(vals, ddof=1))
    bpt = B_TOT / d_tok.size
    se_tok = float(d_tok.std(ddof=1) / math.sqrt(d_tok.size) / (LN2 * bpt))
    return {"delta_bpb": delta, "paired_se_sequence_bootstrap": se_seq,
            "paired_se_per_token": se_tok, "n_sequences": int(n), "n_tokens": int(d_tok.size),
            "z_sequence_paired": delta / se_seq if se_seq > 0 else float("nan"),
            "z_token_paired": delta / se_tok if se_tok > 0 else float("nan"),
            "frac_tokens_worse": float((d_tok > 0).mean())}


def paired(a, b):
    return paired_d(a - b)


# ================================================================= run
res = {
    "brief": "docs/research/donor_adaptation/briefs/BRIEF_D0C_CARVE_GRANULARITY.md @ 4a98c89",
    "config": {
        "arms": [{"tag": t, "E": E, "k": k, "kind": kd,
                  "neurons_per_expert": None if E is None else d_ffn // E}
                 for t, E, k, kd in ARMS],
        "cluster_seed": CLUSTER_SEED,
        "coactivation_mask_density_p": DC.P_PRIMARY,
        "coactivation_mask_topk_neurons": K_ACT,
        "null_seed_formula": "np.random.default_rng(1000 + E)  [Part III used 1032 at E=32]",
        "router": "ORACLE top-k -- upper bound",
        "b3_repair_applied": True,
        "carved_layers": CARVE_LAYERS if SMOKE else "ALL 28",
        "smoke": SMOKE, "secondary_E256_included": SECONDARY,
        "decision_thresholds_from_brief_s4": {"delta_bpb": DELTA_THRESH, "gap_bpb": GAP_THRESH,
                                              "sigma_seed": SIGMA_SEED,
                                              "baseline": BASELINE_STANDING},
        "part_iii_standing_numbers": STANDING,
        "replication_tolerance": REPLICATION_TOL,
    },
    "env": DC.env_manifest("d0c_granularity" + SUFFIX),
    "arch_achieved": A,
    "bpb_eval_slice": meta_ev,
    "n_predicted_tokens": int(n_pred),
}
res["env"]["command_line"] = "python " + " ".join(sys.argv)


def checkpoint():
    json.dump(res, open(OUT, "w"), indent=1, default=float)


checkpoint()

# ---- partitions ---------------------------------------------------------------------------
E_NEEDED = PART_ONLY if PART_ONLY else sorted(set(E for _, E, _, _ in ARMS if E is not None))
LABS = {}
res["partition_checks"] = {}
for E in E_NEEDED:
    lab, null = partitions_for(E)
    LABS[E] = (lab, null)
    chk = check_partition(E, lab, null)
    res["partition_checks"]["E%d" % E] = chk
    log("  partition check E=%d: %s" % (E, chk))

# ---- the E=32 labels must be the ones Part III actually used ------------------------------
ref = os.path.join(C.RESULTS, "d0_carved_labels_E32.npz")
if 32 in LABS and os.path.exists(ref):
    z = np.load(ref)
    lab32, null32 = LABS[32]
    same_c = all(np.array_equal(lab32[L], z["c%d" % L]) for L in sorted(lab32))
    same_n = all(np.array_equal(null32[L], z["n%d" % L]) for L in sorted(null32))
    res["partition_replication_E32"] = {
        "reference": "results/d0_carved_labels_E32.npz (Part III)",
        "coact_labels_identical_all_28_layers": bool(same_c),
        "null_labels_identical_all_28_layers": bool(same_n)}
    log("  E=32 label replication vs Part III cache: coact=%s null=%s" % (same_c, same_n))
checkpoint()

if PART_ONLY:
    log("partitions-only mode: cached E=%s and exiting without running any arm." % PART_ONLY)
    raise SystemExit(0)

# ---- arms ---------------------------------------------------------------------------------
arms, metas = {}, {}
gate_done = False
for tag, E, k, kind in ARMS:
    lab_map = None
    if kind is not None:
        lab_map = LABS[E][0] if kind == "coact" else LABS[E][1]
    a, mt = run_arm(tag, E, k, kind, lab_map)
    arms[tag], metas[tag] = a, mt
    res.setdefault("arm_meta", {})[tag] = mt
    res.setdefault("bpb", {})[tag] = bpb_of(a)
    checkpoint()

    # ---- replication gate, fired the moment A0 and N0 exist --------------------------------
    if not gate_done and not SMOKE and set(["baseline", "A0", "N0"]) <= set(arms):
        gate_done = True
        got = {"baseline_bpb": bpb_of(arms["baseline"]),
               "A0_delta": paired(arms["A0"], arms["baseline"])["delta_bpb"],
               "N0_delta": paired(arms["N0"], arms["baseline"])["delta_bpb"],
               "G32": paired(arms["A0"], arms["N0"])["delta_bpb"]}
        rep = {"tolerance": REPLICATION_TOL, "checks": {}}
        ok = True
        for kk, want in STANDING.items():
            dv = abs(got[kk] - want)
            rep["checks"][kk] = {"part_iii": want, "d0c": got[kk], "abs_diff": dv,
                                 "within_tolerance": bool(dv <= REPLICATION_TOL)}
            ok = ok and dv <= REPLICATION_TOL
        rep["PASS"] = bool(ok)
        res["replication_check"] = rep
        checkpoint()
        log("  REPLICATION GATE: " + ("PASS" if ok else "FAIL"))
        for kk, v in rep["checks"].items():
            log("    %-14s partIII=%.12f  d0c=%.12f  |d|=%.3e  %s"
                % (kk, v["part_iii"], v["d0c"], v["abs_diff"],
                   "ok" if v["within_tolerance"] else "MISMATCH"))
        if not ok and not FORCE:
            log("STOP -- the harness does not reproduce Part III; nothing further is comparable.")
            raise SystemExit(3)

if SMOKE:
    log("smoke complete; wrote " + OUT)
    checkpoint()
    raise SystemExit(0)

# ================================================================= statistics + decision
res["marginal_slice_se"] = {t: float(C.bootstrap_se(v.sum(1) / (LN2 * byts), byts_ev))
                            for t, v in arms.items()}
res["delta_vs_baseline"] = {t: paired(arms[t], arms["baseline"])
                            for t, _, _, kd in ARMS if kd is not None}
E_RUN = sorted(set(E for _, E, _, _ in ARMS if E is not None))
COACT = {32: "A0", 64: "A1", 128: "A2", 256: "S1"}
NULL = {32: "N0", 64: "N1", 128: "N2", 256: "S1n"}
res["gap_coact_vs_null"] = {"E%d" % E: paired(arms[COACT[E]], arms[NULL[E]]) for E in E_RUN}

# achieved-activation match, brief section 3.2
a0_ach = metas["A0"]["achieved_activation"]
res["achieved_activation_match_s3_2"] = {
    "reference_arm": "A0", "reference_achieved": a0_ach, "tolerance": 0.002,
    "per_arm": {t: {"achieved": metas[t]["achieved_activation"],
                    "abs_diff_vs_A0": abs(metas[t]["achieved_activation"] - a0_ach),
                    "matched": bool(abs(metas[t]["achieved_activation"] - a0_ach) <= 0.002)}
                for t, _, _, kd in ARMS if kd is not None}}

# --- the two deciding quantities, with PAIRED errors on the DIFFERENCE ----------------------
dD = paired_d(arms["A2"] - arms["A0"])                                    # Delta(128) - Delta(32)
dG = paired_d((arms["A2"] - arms["N2"]) - (arms["A0"] - arms["N0"]))      # G(128) - G(32)
D32 = res["delta_vs_baseline"]["A0"]["delta_bpb"]
D128 = res["delta_vs_baseline"]["A2"]["delta_bpb"]
G32 = res["gap_coact_vs_null"]["E32"]["delta_bpb"]
G128 = res["gap_coact_vs_null"]["E128"]["delta_bpb"]

dD_v, dD_se = dD["delta_bpb"], dD["paired_se_sequence_bootstrap"]
dG_v, dG_se = dG["delta_bpb"], dG["paired_se_sequence_bootstrap"]
dD_lo, dD_hi = dD_v - 1.96 * dD_se, dD_v + 1.96 * dD_se
dG_lo, dG_hi = dG_v - 1.96 * dG_se, dG_v + 1.96 * dG_se

# INCONCLUSIVE iff the 95% band of a deciding quantity straddles the threshold that decides it.
inconclusive = (dD_lo < -DELTA_THRESH < dD_hi) or (dD_lo < DELTA_THRESH < dD_hi)
if dD_v < -DELTA_THRESH:          # only then does G enter the label at all
    inconclusive = inconclusive or (dG_lo < -GAP_THRESH < dG_hi)

if inconclusive:
    label = "INCONCLUSIVE"
    meaning = "report as inconclusive; do not pick the nearer label"
elif dD_v < -DELTA_THRESH and dG_v < -GAP_THRESH:
    label = "GRANULARITY-BOUND"
    meaning = ("the negative is granularity-local; D0's headline must be restated as an E=32 "
               "statement and the axis reopens at finer E and at larger width")
elif dD_v < -DELTA_THRESH:
    label = "PARTIAL"
    meaning = ("finer carving helps, but not because of co-activation; the mechanism claim does "
               "not strengthen and the engine gets a cheaper knob, not a new lever")
elif abs(dD_v) <= DELTA_THRESH:
    label = "GRANULARITY-INVARIANT"
    meaning = ("D0's negative is about carving, not about granularity. It hardens. Report it as "
               "such and close this axis at 1.5B")
elif dD_v > DELTA_THRESH:
    label = "WORSE"
    meaning = ("finer is worse; say so, and the relerr->BPB link is broken and must itself be "
               "investigated")
else:
    label = "INCONCLUSIVE"
    meaning = "unreachable by construction"

res["decision"] = {
    "rule": "brief section 4, thresholds fixed before the run",
    "Delta_E32": D32, "Delta_E128": D128,
    "Delta_128_minus_Delta_32": dD, "Delta_diff_ci95": [dD_lo, dD_hi],
    "G_E32": G32, "G_E128": G128,
    "G_128_minus_G_32": dG, "G_diff_ci95": [dG_lo, dG_hi],
    "condition_GRANULARITY_BOUND": bool(dD_v < -DELTA_THRESH and dG_v < -GAP_THRESH),
    "condition_PARTIAL": bool(dD_v < -DELTA_THRESH and not (dG_v < -GAP_THRESH)),
    "condition_GRANULARITY_INVARIANT": bool(abs(dD_v) <= DELTA_THRESH),
    "condition_WORSE": bool(dD_v > DELTA_THRESH),
    "band_straddles_threshold": bool(inconclusive),
    "OUTCOME_LABEL": label,
    "OUTCOME_MEANING_VERBATIM_FROM_BRIEF": meaning,
    "sigma_seed": SIGMA_SEED,
    "Delta_in_sigma_seed": {"E%d" % E: res["delta_vs_baseline"][COACT[E]]["delta_bpb"] / SIGMA_SEED
                            for E in E_RUN},
}
res["mem_peak"] = DC._rss_gb()
checkpoint()

log("")
log("=" * 78)
for E in E_RUN:
    log("  E=%4d (n/expert %3d)  Delta=%+.5f +/- %.5f   Delta_null=%+.5f   G=%+.5f +/- %.5f"
        % (E, d_ffn // E,
           res["delta_vs_baseline"][COACT[E]]["delta_bpb"],
           res["delta_vs_baseline"][COACT[E]]["paired_se_sequence_bootstrap"],
           res["delta_vs_baseline"][NULL[E]]["delta_bpb"],
           res["gap_coact_vs_null"]["E%d" % E]["delta_bpb"],
           res["gap_coact_vs_null"]["E%d" % E]["paired_se_sequence_bootstrap"]))
log("  Delta(128)-Delta(32) = %+.5f +/- %.5f   ci95 [%+.5f, %+.5f]" % (dD_v, dD_se, dD_lo, dD_hi))
log("  G(128)-G(32)         = %+.5f +/- %.5f   ci95 [%+.5f, %+.5f]" % (dG_v, dG_se, dG_lo, dG_hi))
log("  OUTCOME: " + label)
log("=" * 78)
log("wrote " + OUT)
