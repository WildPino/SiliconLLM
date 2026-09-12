#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E43 -- is the VOCABULARY a lever, and what is a token actually worth?

Brief: docs/research/donor_adaptation/briefs/BRIEF_E43_IS_THE_VOCABULARY_A_LEVER.md, pushed at
`4b4ebf2` BEFORE this file existed and before any vocabulary had been trained.  Nothing here may
contradict it.

E40 left one term of the floor untouched: the head is `V*D`, it sits OUTSIDE the layer loop, it
is charged in full on every token, and at R128 it is 61.5% of the base.  At a fixed
9,999,220,736 parameters a weight taken out of the head goes into F, which is charged at k/E.
That is the NKV lever's mechanism on a term three times the size.

And the other half, which is the uncomfortable one: the goal is written in tokens/second, and a
token is not a fixed amount of text.  V moves the rate and the bytes-per-token in OPPOSITE
directions and this programme has never measured the denominator.  So E43 measures both and
reports the product.

  python e43_vocabulary.py --phase vocab --smoke    # tiny: instrument check, minutes
  python e43_vocabulary.py --phase vocab            # the 8 vocabularies + the null pair
  python e43_vocabulary.py --phase build            # the 3 new 10 B exports + G-E43B
  python e43_vocabulary.py --phase time --reps 5    # the measurement (IDLE BOX)
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "density")))

from e26_carve_cost import cpu_busy                                  # noqa: E402
from e28_kernel_transfer import bench, winpath                       # noqa: E402
import synth_export as SX                                            # noqa: E402
import e1_bpb_through_engine as E1                                   # noqa: E402

RES = os.path.join(HERE, "results")
VOCDIR = os.path.join(RES, "e43_vocab")
CORPUS = os.path.abspath(os.path.join(HERE, "..", "density", "corpus"))

GOAL, EXCELLENT = 50.0, 100.0
E_GROUPS = 256
KS = [1, 2, 3, 4, 6]
K_IN_FILE = 3
VERDICT_K = 3
TEN_B = 9999220736
NTOK = 40

# --- brief s2: the frozen slice, and the anchor G-E43C reproduces exactly.
SLICE = ("heldout", 24, 512, 1234)
IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
SCORED_BYTES = 51870
QWEN_BPT = 4.22945205479452
QWEN_V = 151936

# --- brief s3.  V32768 IS E40's A10B-R128: the same bytes on disk, re-timed.
ARMS = {
    "V2048":   ("A10B-V2048",     2048),
    "V8192":   ("A10B-V8192",     8192),
    "V32768":  ("A10B-R128",     32768),
    "V131072": ("A10B-V131072", 131072),
}
RANK = 128
CONTROL = "V32768"
E40_R128_K3 = 112.73          # brief s6 G-E43A: E40 run 1, the registered control value
SESSION_TOL = 0.05            # the +-5% bar E39 and E40 both ran on -- a MEASURED tolerance

# --- brief s3.2 / s3.3
VGRID = [2048, 8192, 14336, 20480, 26624, 32768, 51200, 131072]
NULL_VS = [2048, 32768]
NULL_SEED = 43043

# --- brief s5
# --- addendum A rule 3, registered at 3342175 BEFORE the re-run: a rep whose recorded box
# reading is at or above this is discarded before any arm statistic is computed.  25% is not
# taken from E43's numbers -- it is just above the 22.5% maximum that E39's and E40's own
# ACCEPTED sessions ran at.  Fewer than MIN_REPS survivors voids the session.
BUSY_MAX, MIN_REPS = 25.0, 3
SUF = ""

BANDS = [(1.50, "VOCABULARY-IS-A-BIG-LEVER"), (1.15, "VOCABULARY-IS-A-LEVER"),
         (1.05, "VOCABULARY-IS-A-SMALL-LEVER"), (0.0, "VOCABULARY-IS-NOT-A-LEVER")]


def log(*a):
    print(*a, flush=True)


def shape_of(name):
    return dict(zip(("D", "F", "L", "NH", "NKV", "HD", "V", "tied"), SX.SHAPES[name][:8]))


def charged(arm, k):
    s = shape_of(ARMS[arm][0])
    return SX.active_weights(s["D"], s["F"], s["L"], s["NH"], s["NKV"], s["HD"], s["V"],
                             E_GROUPS, k, RANK)


def closed_form(arm, k):
    """Brief s6 G-E43B.  Written here from the shape's own dimensions and WITHOUT calling
    synth_export, so a disagreement is a real disagreement and not the same arithmetic twice."""
    s = shape_of(ARMS[arm][0])
    D, F, L, NH, NKV, HD, V = (s[x] for x in ("D", "F", "L", "NH", "NKV", "HD", "V"))
    QO, KD = NH * HD, NKV * HD
    qo = 2 * (RANK * (QO + D))
    attn = qo + 2 * (KD * D)
    ffn = E_GROUPS * D + 3 * D * (F // E_GROUPS) * k
    return (attn + ffn) * L + V * D


def closed_form_total(arm):
    """Same, for the file's whole parameter count.  Untied, so embedding AND head are counted."""
    s = shape_of(ARMS[arm][0])
    D, F, L, NH, NKV, HD, V = (s[x] for x in ("D", "F", "L", "NH", "NKV", "HD", "V"))
    QO, KD = NH * HD, NKV * HD
    per = 2 * (RANK * (QO + D)) + 2 * (KD * D) + 3 * D * F
    return per * L + 2 * V * D


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# --------------------------------------------------------------------- the units half
def frozen_spans():
    """The 24 scored texts of the frozen slice, reconstructed from the cache.

    make_slice scores `tok.decode(ids[1:])` and charges its UTF-8 length, so THAT text -- 51,870
    bytes of it -- is what every candidate vocabulary has to encode.  The ids sha is asserted
    because density/common.get_slice's cache key omits the tokenizer (a logged defect); asserting
    it here makes the omission harmless for this probe.
    """
    import common as DC
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(DC.MODEL_ID, revision=DC.REVISION)
    ids, byts, meta = DC.get_slice(tok, *SLICE)
    assert meta["ids_sha256"] == IDS_SHA, "the cached slice is not the registered one"
    texts, qtoks = [], 0
    for i in range(ids.shape[0]):
        seq = ids[i, 1:].tolist()
        t = tok.decode(seq)
        assert len(t.encode("utf-8")) == int(byts[i]), "byte accounting moved under the slice"
        texts.append(t)
        qtoks += len(seq)
    return texts, meta, qtoks


def train_bpe(V, corpus_path, out_path):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=V, min_frequency=2, special_tokens=[],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                             show_progress=False)
    t.train([corpus_path], tr)
    t.save(out_path)
    return t


def bpt(t, texts):
    """Bytes per token of `t` over the frozen span: 51,870 bytes / tokens needed to encode it."""
    n = sum(len(t.encode(x).ids) for x in texts)
    return SCORED_BYTES / n, n


def shuffled_corpus(src, dst, nbytes, seed):
    """Same characters, same multiset, word and line structure destroyed, valid UTF-8 kept."""
    with open(src, "rb") as f:
        raw = f.read(nbytes)
    chars = list(raw.decode("utf-8", "ignore"))
    random.Random(seed).shuffle(chars)
    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.write("".join(chars))
    return os.path.getsize(dst)


def sanitised_calib(nbytes):
    """The BPE trainer requires the whole stream to be valid UTF-8, and `calib.txt` is NOT:
    it was assembled from 8,192-byte chunks, so characters are cut in half at chunk joins
    (first bad byte at 184,359).  `make_slice` has always absorbed this with
    `decode("utf-8", "ignore")`, and that is exactly what is done here, on the TRAINING corpus
    only -- the measured span comes from make_slice and is untouched, which is why G-E43C still
    reproduces 4.22945205479452 exactly.  The number of dropped bytes is recorded, not hidden.

    Truncation to `nbytes` happens AFTER sanitising and backs off at most 4 bytes, the longest
    a UTF-8 character can be.  The first version of this backed off one byte at a time until the
    decode succeeded, which on a stream with an INTERIOR bad byte is O(n^2) and ran for ten
    minutes on 8 MB before it was caught.
    """
    src = os.path.join(CORPUS, "calib.txt")
    full = os.path.join(VOCDIR, "calib_utf8.txt")
    if not os.path.exists(full):
        raw = open(src, "rb").read()
        clean = raw.decode("utf-8", "ignore").encode("utf-8")
        with open(full, "wb") as f:
            f.write(clean)
        json.dump({"src_bytes": len(raw), "clean_bytes": len(clean)},
                  open(full + ".json", "w"), indent=1)
    m = json.load(open(full + ".json"))
    drop = m["src_bytes"] - m["clean_bytes"]
    if not nbytes or nbytes >= m["clean_bytes"]:
        return full, drop
    sub = os.path.join(VOCDIR, "calib_utf8_%d.txt" % nbytes)
    if not os.path.exists(sub):
        raw = open(full, "rb").read(nbytes)
        for _ in range(4):
            try:
                raw.decode("utf-8"); break
            except UnicodeDecodeError:
                raw = raw[:-1]
        raw.decode("utf-8")          # sanitised upstream: this cannot fail now
        with open(sub, "wb") as f:
            f.write(raw)
    return sub, drop


def phase_vocab(a):
    os.makedirs(VOCDIR, exist_ok=True)
    out = {"brief": "BRIEF_E43 (pre-registered, 4b4ebf2)", "phase": "vocab", "smoke": a.smoke}
    t0 = time.time()

    log("== E43 phase VOCAB: what is a token worth? ==")
    texts, meta, qtoks = frozen_spans()
    nb = sum(len(x.encode("utf-8")) for x in texts)

    # ---- G-E43C: the units instrument's known positive, exact, no tolerance.
    gc = {"scored_bytes": nb, "bar_bytes": SCORED_BYTES, "qwen_tokens": qtoks,
          "qwen_bpt": nb / qtoks, "bar_bpt": QWEN_BPT, "ids_sha256": meta["ids_sha256"]}
    gc["fires"] = bool(nb == SCORED_BYTES and abs(gc["qwen_bpt"] - QWEN_BPT) < 1e-12
                       and meta["ids_sha256"] == IDS_SHA)
    log("  G-E43C  frozen span %d B (bar %d), Qwen V=%d reads %.14f B/tok (bar %.14f) -> %s"
        % (nb, SCORED_BYTES, QWEN_V, gc["qwen_bpt"], QWEN_BPT,
           "FIRES" if gc["fires"] else "VOID"))
    if not gc["fires"]:
        log("          the instrument cannot read the slice it claims to read.  The whole")
        log("          bytes/token column is void, and with it both cells.")
    out["G_E43C"] = gc

    grid = [2048, 8192] if a.smoke else VGRID
    corp, drop = sanitised_calib(a.train_bytes)
    train_bytes = os.path.getsize(corp)
    out["corpus"] = {"file": os.path.basename(corp), "bytes": train_bytes,
                     "invalid_bytes_dropped": drop, "sha256": sha_file(corp)}
    log("  training on %s (%d B, %d invalid UTF-8 bytes dropped -- see below), measuring on"
        % (os.path.basename(corp), train_bytes, drop))
    log("  the frozen %d B of heldout" % nb)

    rows = {}
    for V in grid:
        p = os.path.join(VOCDIR, "bpe_V%d_%d.json" % (V, train_bytes))
        ts = time.time()
        if os.path.exists(p):
            from tokenizers import Tokenizer
            t = Tokenizer.from_file(p)
            how = "cached"
        else:
            t = train_bpe(V, corp, p)
            how = "%.0fs" % (time.time() - ts)
        b, n = bpt(t, texts)
        rows[str(V)] = {"V": V, "bytes_per_token": b, "tokens": n, "vocab_sha256": sha_file(p),
                        "real_vocab_size": t.get_vocab_size(), "trained": how}
        log("    V=%-7d %8.5f B/tok  (%6d tokens, vocab %d)  [%s]"
            % (V, b, n, t.get_vocab_size(), how))
    out["vocab"] = rows
    out["train_bytes"] = train_bytes

    # ---- G-E43D: the null vocabulary.  ORDINAL, no tolerance (brief s0.1, s6).
    nulls = {}
    sh = os.path.join(VOCDIR, "calib_shuffled_%d_s%d.txt" % (train_bytes, NULL_SEED))
    if not os.path.exists(sh):
        log("  building the character-shuffled corpus (seed %d)..." % NULL_SEED)
        shuffled_corpus(corp, sh, train_bytes, NULL_SEED)
    for V in ([2048] if a.smoke else NULL_VS):
        p = os.path.join(VOCDIR, "bpe_null_V%d_%d.json" % (V, train_bytes))
        if os.path.exists(p):
            from tokenizers import Tokenizer
            t = Tokenizer.from_file(p)
        else:
            t = train_bpe(V, sh, p)
        b, n = bpt(t, texts)
        real = rows[str(V)]["bytes_per_token"]
        nulls[str(V)] = {"V": V, "null_bpt": b, "real_bpt": real,
                         "null_is_worse": bool(b < real), "vocab_sha256": sha_file(p)}
        log("    null V=%-7d %8.5f B/tok  vs real %8.5f  -> null %s"
            % (V, b, real, "worse (as it must be)" if b < real else "NOT worse"))
    gd = {"per_V": nulls, "fires": bool(nulls and all(v["null_is_worse"]
                                                      for v in nulls.values()))}
    log("  G-E43D  character-shuffled BPE strictly worse at every V tested (ORDINAL) -> %s"
        % ("FIRES" if gd["fires"] else "VOID"))
    if not gd["fires"]:
        log("          a vocabulary learned from destroyed structure tied or won: the trainer")
        log("          is not learning corpus structure and the bytes/token column is void.")
    out["G_E43D"] = gd

    out["seconds"] = time.time() - t0
    p = os.path.join(RES, "e43_vocab%s.json" % ("_smoke" if a.smoke else ""))
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    log("  wrote %s  [%.0fs]" % (p, out["seconds"]))
    return 0


# --------------------------------------------------------------------- the speed half
def build(arm, d, reuse):
    shape, _ = ARMS[arm]
    if arm == CONTROL and os.path.exists(reuse):
        log("  have  %-8s <- %s  (E40's OWN artifact, not rebuilt: that is the control)"
            % (arm, reuse))
        return reuse
    out = os.path.join(d, "e43_%s.bin" % arm.lower())
    if os.path.exists(out) and os.path.exists(out + ".json"):
        log("  have  %-8s %s" % (arm, os.path.basename(out)))
        return out
    cmd = [sys.executable, os.path.join(HERE, "synth_export.py"), "--shape", shape,
           "--out", out, "--codes", "mixed", "--seed", "1234",
           "--carve", str(E_GROUPS), "--carve-k", str(K_IN_FILE), "--rank", str(RANK)]
    log("  build %-8s %s" % (arm, shape))
    ts = time.time()
    r = subprocess.run(cmd, capture_output=True)
    txt = r.stdout.decode(errors="replace")
    if r.returncode != 0:
        log(txt[-3000:]); log(r.stderr.decode(errors="replace")[-3000:])
        raise SystemExit("synth_export failed for " + arm)
    for ln in txt.splitlines():
        if "GATE V3" in ln or ln.startswith("wrote"):
            log("        " + ln.strip())
    log("        [%.0fs]" % (time.time() - ts))
    return out


def phase_build(a):
    os.makedirs(a.dir, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()
    log("== E43 phase BUILD: four arms, all exactly %d parameters ==" % TEN_B)
    log("  SPEED ONLY and the weights are NOISE -- the same limit as E36, E39 and E40.")

    files, meta = {}, {}
    for arm in ARMS:
        files[arm] = build(arm, a.dir, a.r128)
        h = E1.read_header(files[arm])
        tot = SX.total_weights(h["D"], h["F"], h["L"], h["NH"], h["NKV"], h["HD"], h["V"],
                               bool(h["tied"]), RANK)
        meta[arm] = {"file": files[arm], "shape": ARMS[arm][0], "V": ARMS[arm][1],
                     "bytes": os.path.getsize(files[arm]),
                     "header": {k: h[k] for k in ("D", "F", "L", "NH", "NKV", "HD", "V", "tied")},
                     "total_from_header": tot, "closed_form_total": closed_form_total(arm),
                     "base": charged(arm, 0), "charged_k3": charged(arm, VERDICT_K),
                     "head": ARMS[arm][1] * h["D"],
                     "head_frac_of_base": ARMS[arm][1] * h["D"] / charged(arm, 0)}

    # ---- G-E43B: exact equality against a closed form written without synth_export.
    gb = {"per_arm": {}}
    for arm in ARMS:
        rows = {"total": {"from_header": meta[arm]["total_from_header"],
                          "closed_form": closed_form_total(arm), "bar": TEN_B}}
        rows["total"]["agrees"] = bool(rows["total"]["from_header"]
                                       == rows["total"]["closed_form"] == TEN_B)
        for k in sorted(set(KS + [0, VERDICT_K])):
            ex, cf = charged(arm, k), closed_form(arm, k)
            rows["k%d" % k] = {"charged": ex, "closed_form": cf, "agrees": bool(ex == cf)}
        rows["all"] = bool(all(r["agrees"] for r in rows.values() if isinstance(r, dict)))
        gb["per_arm"][arm] = rows
        if not rows["all"]:
            log("  G-E43B  %s DISAGREES -- dropped, it is not the object the brief names" % arm)
    gb["fires"] = bool(all(v["all"] for v in gb["per_arm"].values()))
    log("")
    log("  G-E43B  header total == closed form == %d, and charged == closed form at every k,"
        % TEN_B)
    log("          zero tolerance -> %s" % ("FIRES" if gb["fires"] else "VOID"))
    for arm in sorted(ARMS, key=lambda x: ARMS[x][1]):
        m = meta[arm]
        log("    %-8s V=%-7d F=%-6d base %11d  head %11d (%4.1f%%)  charged k=3 %11d"
            % (arm, m["V"], m["header"]["F"], m["base"], m["head"],
               100 * m["head_frac_of_base"], m["charged_k3"]))

    out = {"brief": "BRIEF_E43 (pre-registered, 4b4ebf2)", "phase": "build", "arms": meta,
           "G_E43B": gb, "ten_b_bar": TEN_B, "seconds": time.time() - t0}
    p = os.path.join(RES, "e43_build.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    log("  wrote %s  [%.0fs]" % (p, out["seconds"]))
    return 0


def phase_time(a):
    import numpy as np
    global SUF
    SUF = "_order_reversed" if a.order == "reversed" else ""
    t0 = time.time()
    bp = os.path.join(RES, "e43_build.json")
    if not os.path.exists(bp):
        raise SystemExit("run --phase build first")
    b = json.load(open(bp, encoding="utf-8"))
    files = {k: v["file"] for k, v in b["arms"].items()}
    order = sorted(ARMS, key=lambda x: ARMS[x][1])

    log("== E43 phase TIME: %d arms x %d reps x %d tokens, reps OUTERMOST, order %s =="
        % (len(order), a.reps, a.ntok, a.order))
    plan = [(arm, k) for arm in order for k in KS]
    if a.order == "reversed":
        plan = plan[::-1]

    rates, busy, peaks = {}, [], []
    for rep in range(1, a.reps + 1):
        bz, pk = cpu_busy()
        busy.append(bz); peaks.append(pk)
        for arm, k in plan:
            r = bench(a.engine, winpath(files[arm]), a.ntok, a.threads, ["--carve-k", str(k)])
            rates.setdefault("%s_k%d" % (arm, k), []).append(r)
            log("    rep %d  %-8s k=%-2d %8.2f tok/s" % (rep, arm, k, r))
        log("         box %.1f%% busy, peak core %.1f%%" % (bz, pk))

    # ---- addendum A rule 3: drop contended reps by their OWN recorded box reading, before any
    #      arm statistic exists.  The rule and the number were registered before this ran.
    keep = [i for i, bz in enumerate(busy) if bz < BUSY_MAX]
    dropped = [(i + 1, round(busy[i], 1)) for i in range(len(busy)) if i not in keep]
    session_void = len(keep) < MIN_REPS
    log("")
    log("  rule 3  %d of %d reps survive the %.0f%% box bar%s"
        % (len(keep), len(busy), BUSY_MAX,
           "" if not dropped else "  (dropped: %s)"
           % ", ".join("rep %d @ %.1f%%" % d for d in dropped)))
    if session_void:
        log("          fewer than %d survivors -> THIS SESSION IS VOID (addendum A rule 3)"
            % MIN_REPS)

    arms_out = {}
    for key, rs in rates.items():
        arm, k = key.rsplit("_k", 1); k = int(k)
        kept = [rs[i] for i in keep] if keep else list(rs)
        m = sum(kept) / len(kept)
        arms_out[key] = {"arm": arm, "k": k, "rates": rs, "rates_kept": kept,
                         "mean_tok_s": m, "median_tok_s": sorted(kept)[len(kept) // 2],
                         "spread": (max(kept) - min(kept)) / m, "charged": charged(arm, k)}

    out = dict(b)
    out["phase"] = "time"
    out["speed_arms"] = arms_out
    out["cpu_busy_pct"] = busy
    out["cpu_peak_pct"] = peaks
    out["rule3"] = {"busy_max": BUSY_MAX, "min_reps": MIN_REPS, "kept_reps": [i + 1 for i in keep],
                    "dropped_reps": dropped, "session_void": session_void}
    out["order"] = a.order
    out["reps"] = a.reps
    out["ntok"] = a.ntok

    # ---- G-E43A: the planted control.  E40's own file, re-timed, +-5% (a MEASURED bar).
    ctl = arms_out.get("%s_k%d" % (CONTROL, VERDICT_K))
    rel = ctl["mean_tok_s"] / E40_R128_K3 - 1.0
    ga = {"file": files[CONTROL], "now": ctl["mean_tok_s"], "e40_run1": E40_R128_K3,
          "rel": rel, "tol": SESSION_TOL, "fires": bool(abs(rel) <= SESSION_TOL)}
    log("")
    log("  G-E43A  %s k=3 now %.2f vs E40 run 1's %.2f  (%+.1f%%, bar +-%.0f%%) -> %s"
        % (CONTROL, ctl["mean_tok_s"], E40_R128_K3, 100 * rel, 100 * SESSION_TOL,
           "FIRES" if ga["fires"] else "VOID"))
    if not ga["fires"]:
        log("          the sessions are not comparable.  NO RATE IN E43 COUNTS.")
    ga["session_void_rule3"] = session_void
    out["G_E43A"] = ga

    # ---- the two-term fit per arm (E40's, unchanged), and the floor.
    fits = {}
    for arm in order:
        ks = [k for k in KS if "%s_k%d" % (arm, k) in arms_out]
        t = np.array([1000.0 / arms_out["%s_k%d" % (arm, k)]["mean_tok_s"] for k in ks])
        A = np.vstack([np.ones(len(ks)), np.array(ks, float)]).T
        (c0, c1), *_ = np.linalg.lstsq(A, t, rcond=None)
        F = shape_of(ARMS[arm][0])["F"]
        f = {"ms_base": c0, "ms_per_group": c1, "F": F, "group_size": F // E_GROUPS,
             "floor_tok_s": 1000.0 / c0, "base": closed_form(arm, 0),
             "base_G_w_per_s": closed_form(arm, 0) / c0 / 1e6}
        for goal, nm in ((EXCELLENT, "k_star_100"), (GOAL, "k_star_50")):
            ks_ = (1000.0 / goal - c0) / c1
            f[nm] = ks_
            f[nm + "_pct"] = 100.0 * ks_ * (F // E_GROUPS) / F
        fits[arm] = f
        log("  %-8s time = %.3f ms + %.4f ms/group   floor %7.1f tok/s   base %.1f G-w/s"
            % (arm, c0, c1, f["floor_tok_s"], f["base_G_w_per_s"]))
    out["fits"] = fits

    # ---- the product, and the two cells.  Needs the vocab phase.
    vp = os.path.join(RES, "e43_vocab.json")
    if not os.path.exists(vp):
        log("")
        log("  no e43_vocab.json: rates are written, the CELLS are not computed.")
        out["seconds"] = time.time() - t0
        json.dump(out, open(os.path.join(RES, "e43_vocabulary%s.json" % SUF), "w",
                            encoding="utf-8"), indent=1)
        return 0
    voc = json.load(open(vp, encoding="utf-8"))
    if not (voc["G_E43C"]["fires"] and voc["G_E43D"]["fires"]):
        log("")
        log("  G-E43C/D did not both fire: the bytes/token column is VOID and so are both")
        log("  cells (brief s6).  Rates are written and stand on G-E43A alone.")
        out["cells_void_reason"] = "G-E43C/D"
        out["seconds"] = time.time() - t0
        json.dump(out, open(os.path.join(RES, "e43_vocabulary%s.json" % SUF), "w",
                            encoding="utf-8"), indent=1)
        return 0

    prod = {}
    for arm in order:
        V = ARMS[arm][1]
        bpt_v = voc["vocab"][str(V)]["bytes_per_token"]
        r = arms_out["%s_k%d" % (arm, VERDICT_K)]["mean_tok_s"]
        prod[arm] = {"V": V, "tok_s": r, "bytes_per_token": bpt_v, "bytes_per_s": r * bpt_v,
                     "floor_tok_s": fits[arm]["floor_tok_s"],
                     "floor_bytes_per_s": fits[arm]["floor_tok_s"] * bpt_v}
    out["product"] = prod

    log("")
    log("  %-8s %8s %10s %12s %12s %14s" % ("arm", "V", "tok/s k=3", "B/tok", "BYTES/s",
                                            "floor tok/s"))
    for arm in order:
        p_ = prod[arm]
        log("  %-8s %8d %10.2f %12.5f %12.1f %14.1f"
            % (arm, p_["V"], p_["tok_s"], p_["bytes_per_token"], p_["bytes_per_s"],
               p_["floor_tok_s"]))

    peak = max(order, key=lambda x: prod[x]["bytes_per_s"])
    cell = prod[peak]["bytes_per_s"] / prod[CONTROL]["bytes_per_s"]
    name = next(n for lo, n in BANDS if cell >= lo)
    vs = sorted(order, key=lambda x: ARMS[x][1])
    side = ("at" if peak == CONTROL else
            ("above" if ARMS[peak][1] < ARMS[CONTROL][1] else "below"))
    out["verdict"] = {"cell": "BPS_peak / BPS(V=32768) at k=3", "value": cell, "name": name,
                      "peak_arm": peak, "peak_V": ARMS[peak][1],
                      "control_is": side, "grid": [ARMS[x][1] for x in vs]}
    log("")
    log("  VERDICT  peak is %s (V=%d) at %.1f B/s vs the control's %.1f  ->  cell %.4f  -> %s"
        % (peak, ARMS[peak][1], prod[peak]["bytes_per_s"], prod[CONTROL]["bytes_per_s"],
           cell, name))
    log("  CELL 2   V=32768 is %s the peak of the measured curve (ordinal, no tolerance)"
        % side.upper())
    if not out["G_E43A"]["fires"]:
        log("  ...and G-E43A is VOID, so none of this counts (brief s6).")

    out["seconds"] = time.time() - t0
    p = os.path.join(RES, "e43_vocabulary%s.json" % SUF)
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    log("")
    log("  wrote %s  [%.0fs]" % (p, out["seconds"]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("vocab", "build", "time"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train-bytes", type=int, default=0,
                    help="train vocabularies on the first N bytes of calib.txt (0 = all of it). "
                         "The null pair is ALWAYS trained at the same budget, so G-E43D stays "
                         "matched whatever this is set to.")
    ap.add_argument("--dir", default="D:/_ktmp/e43")
    ap.add_argument("--r128", default="D:/_ktmp/e40/e40_r128.bin")
    ap.add_argument("--engine", default="./donor_engine_e26.exe")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ntok", type=int, default=NTOK)
    ap.add_argument("--order", default="forward", choices=("forward", "reversed"))
    a = ap.parse_args()
    return {"vocab": phase_vocab, "build": phase_build, "time": phase_time}[a.phase](a)


if __name__ == "__main__":
    sys.exit(main())
