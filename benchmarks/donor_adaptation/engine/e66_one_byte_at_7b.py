# -*- coding: utf-8 -*-
"""E66 -- one byte at seven billion.

Brief: docs/research/donor_adaptation/briefs/BRIEF_E66_ONE_BYTE_AT_SEVEN_BILLION.md (+ addenda A-C)

The largest real donor this programme owns has been converted exactly twice: to fp32, where it
works (E15 B0 = 0.674027, 160/160 greedy), and to ternary, where it is rank-dead (E16:
SCORE-CROSSES-RANK-DOES-NOT, 0/160 greedy).  The one-byte rung between them has never been run
at 7 B, although E62 measured it free at 0.5 / 1.5 / 3 B.

Two things this runner does that E64 run 1 did not:

  * every engine invocation ASSERTS its kernel arm and its quant from the engine's own CONFIG
    line, and REFUSES the cell otherwise (ledger 63.3);
  * it never resumes from its own result file.  Stages write separate artefacts under
    results/e66/ and the scoring stage reads those; no control is ever validated against a file
    this runner wrote (E65 run 1's defect).

QUALITY AND RANK ONLY.  No rate.  G-E63d is VOID and OWED and E66 does not touch it.
"""
import argparse, hashlib, json, math, os, re, struct, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "e66")
TMP = r"D:\_ktmp\e66"
EXPORTER = os.path.join(HERE, "qwen_export.py")
ENGINE = os.path.join(HERE, "donor_engine_e63.exe")

# ---- the arm every cell asserts.  E62's int8 ladder was taken on avx4; E66 matches it. -----
ATTN = "avx4"
THREADS = "6"
SEQLEN = 512

# ---- E1's protocol.  Transcribed, not re-derived. ------------------------------------------
SCORED_BYTES, N_PRED = 51870, 12264
BYTES_PER_TOK = SCORED_BYTES / float(N_PRED)
IDS_7B = r"D:\_ktmp\e7\e15_ids_24x512.bin"          # E15's Coder-tokenised frozen slice
IDS_15B = r"D:\_ktmp\e1\ids_qwen25-15b_tqh.bin"     # E1's, for the 1.5 B control
CHANCE_CODER = math.log(152064.0, 2) / BYTES_PER_TOK   # 4.070106
CHANCE_QWEN = math.log(151936.0, 2) / BYTES_PER_TOK    # 4.069819

# ---- published references.  TRANSCRIBED from the result files; a replication that recomputes
#      its own target is not a replication. ------------------------------------------------
E62_15B_I8 = 0.7688588381536873      # e62_third_scale.json references/E60_engine/15b/I8
E16_B2 = 4.017232598                 # probes/E16_R3_AT_7B.md, R3 fold=layers
E15_B0_FP32_7B = 0.674026555         # probes/E15_DOES_THE_7B_PREDICT.md, the fp32 donor
E62_LADDER = {"0.5B": (0.871795, 0.8718768840274047),
              "1.5B": (0.767595, 0.7688588381536873),
              "3B":   (0.724450, 0.7250400705611162)}

G66A_TOL = 1.0e-08                   # E62's own G62A_TOL
G66B_TOL = 1.0e-04                   # E37's standing G-E37A tolerance, registered in the brief
G66B_ALARM = 1.0e-03                 # above this, a dense-path miss is a NEW finding -> stop

# ---- artefacts ------------------------------------------------------------------------------
CTL_15B_I8 = r"D:\_ktmp\e60\qwen25-15b_i8h.bin"
CTL_7B_TERN = r"D:\_ktmp\e7\qwen25-coder7b_p_r3.bin"
FP32_7B = r"D:\_ktmp\e7\qwen25-coder7b_f32.bin"
A1 = os.path.join(TMP, "coder7b_i8_foldlayers.bin")
A2 = os.path.join(TMP, "coder7b_i8_nofold.bin")
CONTROLS_RESULT = os.path.join(RES, "e66_controls.json")

# Addendum C: identities of the committed controls-only checkpoint.  A continuation refuses
# rather than silently repeating or accepting a different control measurement.
PINNED_CONTROLS_SHA256 = "da0dbe3ad667ad017290805647745584f0c709787832650f5177a92fa04cac9d"
PINNED_CONTROL_HEAD = "04216cc391daf481e2a9898a7918082d797f2319"
PINNED_CONTROL_RUNNER_BLOB = "34ded51fa5b54f8a2a1641a6a6746590038ff15e"
PINNED_CONTROL_RUNNER_SHA256 = "8fe0460265a11afc42b30e09d9880f65ae9a0e94fc03fde012e76ca0e90b56a8"
PINNED_ENGINE_SHA256 = "56272fdbe615d61739094605cb026aa308fd74604ba5598fab501c09188ae687"

HF = "Qwen/Qwen2.5-Coder-7B"
REV = "0396a76181e127dfc13e5c5ec48a8cee09938b02"     # pinned from the local snapshot

PROMPTS = ["The capital of France is",
           "def fibonacci(n):",
           "Water boils at",
           "The three laws of motion were formulated by",
           "import numpy as np"]
N_NEW = 32                                            # 5 x 32 = 160, E6/E7/E16's protocol


def log(m, fh=None):
    sys.stdout.write(m + "\n")
    sys.stdout.flush()
    if fh:
        fh.write(m + "\n")
        fh.flush()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def committed_provenance():
    """Addendum B.2: REFUSE a measurement from an uncommitted runner.

    E64 run 2 was pre-registered but its exact apparatus blob reached Git after launch.  E66
    makes the ordering executable: the worktree runner must equal HEAD before any model loads.
    """
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True
        ).strip()
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        rel = os.path.relpath(os.path.abspath(__file__), root).replace("\\", "/")
        worktree_blob = subprocess.check_output(
            ["git", "hash-object", os.path.abspath(__file__)], cwd=root, text=True
        ).strip()
        head_blob = subprocess.check_output(
            ["git", "rev-parse", "HEAD:" + rel], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit("cannot establish committed runner provenance: %s" % e)
    if worktree_blob != head_blob:
        raise SystemExit(
            "RUNNER IS NOT THE COMMITTED HEAD BLOB: worktree %s, HEAD %s. "
            "E66 refuses to start (addendum B.2)." % (worktree_blob, head_blob)
        )
    return {
        "head_commit": head,
        "runner_path": rel,
        "runner_git_blob": worktree_blob,
        "runner_sha256": sha256_file(os.path.abspath(__file__)),
        "engine_sha256": sha256_file(ENGINE),
    }


def load_pinned_controls(current_provenance):
    """Addendum C: reuse only the exact, controls-only committed checkpoint.

    Return the small prior needed by stage_bpb plus an audit record for the treatment result.
    The checkpoint file is opened read-only and is never rewritten by the continuation stage.
    """
    if not os.path.exists(CONTROLS_RESULT):
        raise SystemExit("missing pinned controls checkpoint: " + CONTROLS_RESULT)
    actual_sha = sha256_file(CONTROLS_RESULT)
    if actual_sha != PINNED_CONTROLS_SHA256:
        raise SystemExit("CONTROLS CHECKPOINT SHA256 MISMATCH: %s != %s; refusing continuation"
                         % (actual_sha, PINNED_CONTROLS_SHA256))
    with open(CONTROLS_RESULT, encoding="utf-8") as f:
        d = json.load(f)
    p = d.get("provenance", {})
    expected = {
        "head_commit": PINNED_CONTROL_HEAD,
        "runner_git_blob": PINNED_CONTROL_RUNNER_BLOB,
        "runner_sha256": PINNED_CONTROL_RUNNER_SHA256,
        "engine_sha256": PINNED_ENGINE_SHA256,
    }
    bad = [(k, p.get(k), v) for k, v in expected.items() if p.get(k) != v]
    verdicts = [d.get("controls", {}).get(k, {}).get("verdict")
                for k in ("G_E66a", "G_E66b")]
    if bad or d.get("experiment") != "E66" or d.get("controls_fire") is not True \
            or verdicts != ["FIRES", "FIRES"]:
        raise SystemExit("PINNED CONTROLS ARE NOT ADMISSIBLE: identity=%r experiment=%r "
                         "controls_fire=%r verdicts=%r"
                         % (bad, d.get("experiment"), d.get("controls_fire"), verdicts))
    if current_provenance.get("engine_sha256") != PINNED_ENGINE_SHA256:
        raise SystemExit("ENGINE CHANGED SINCE CONTROLS: %s != %s; refusing continuation"
                         % (current_provenance.get("engine_sha256"), PINNED_ENGINE_SHA256))
    prior = {"controls": d["controls"], "controls_fire": True}
    audit = {"path": os.path.relpath(CONTROLS_RESULT, HERE).replace("\\", "/"),
             "sha256": actual_sha, "producing_provenance": p,
             "seconds_total": d.get("seconds_total")}
    return prior, audit


# ============================================================ engine, with the arm asserted
def run_bpb(weights, ids, want_quant, extra=()):
    """One BPB cell.  REFUSES the cell unless N_PREDICTED is E1's and the engine's own CONFIG
    line confirms BOTH the quant and the kernel arm that were asked for (G-E66c)."""
    cmd = [ENGINE, "--weights", weights, "--threads", THREADS, "--seqlen", str(SEQLEN),
           "--bpb", ids, "--attn", ATTN] + list(extra)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("engine --bpb failed (%d): %s" % (r.returncode, r.stderr[-2000:]))
    m = re.search(r"NATS_PER_TOKEN\s+([0-9.]+)", r.stdout)
    p = re.search(r"N_PREDICTED\s+(\d+)", r.stdout)
    if not m:
        raise SystemExit("no NATS_PER_TOKEN:\n" + r.stdout[-2000:])
    npred = int(p.group(1)) if p else None
    if npred != N_PRED:
        raise SystemExit("N_PREDICTED %r != %d -- not E1's protocol" % (npred, N_PRED))
    cm = re.search(r"CONFIG[^\n]*", r.stdout)
    cfg = cm.group(0) if cm else None
    if not cfg:
        raise SystemExit("no CONFIG line -- the cell cannot confirm its configuration and is "
                         "REFUSED (G-E66c).")
    if not re.search(r"attn=%s\b" % re.escape(ATTN), cfg):
        raise SystemExit("KERNEL ARM MISMATCH: asked --attn %s, engine reports %r. REFUSED "
                         "(G-E66c)." % (ATTN, cfg))
    if not re.search(r"quant=%s\b" % re.escape(want_quant), cfg):
        raise SystemExit("QUANT MISMATCH: this arm requires quant=%s, engine reports %r. "
                         "REFUSED (G-E66c)." % (want_quant, cfg))
    nats = float(m.group(1))
    return {"nats_per_token": nats, "bpb": nats / math.log(2) / BYTES_PER_TOK,
            "n_predicted": npred, "seconds": time.time() - t0, "config": cfg,
            "weights": os.path.basename(weights)}


def confirm_sidecar(weights, want):
    """The ARTEFACT must say what it is.  A cell whose sidecar disagrees with the construction
    this experiment registered is REFUSED -- feedback_control_arm_different_code_path."""
    side = weights + ".json"
    if not os.path.exists(side):
        raise SystemExit("no sidecar for %s" % weights)
    d = json.load(open(side, encoding="utf-8"))
    bad = [(k, d.get(k), v) for k, v in want.items() if d.get(k) != v]
    if bad:
        raise SystemExit("SIDECAR MISMATCH for %s: %s -- the cell is REFUSED"
                         % (os.path.basename(weights), bad))
    return {k: d.get(k) for k in ("model", "revision", "quant", "rule", "fold", "head_ternary",
                                  "calib_seqs", "load_dtype", "vocab", "n_layers", "bytes",
                                  "sha256")}


# ============================================================ gates
def g_e66a(measured):
    d = abs(measured - E62_15B_I8)
    return ("FIRES" if d <= G66A_TOL else "DEAD"), d


def g_e66b(measured):
    d = abs(measured - E16_B2)
    if d > G66B_ALARM:
        return "DEAD-AND-NEW-FINDING", d
    return ("FIRES" if d <= G66B_TOL else "DEAD"), d


def g_e66d(a2_bpb):
    return ("BELOW-CHANCE" if a2_bpb < CHANCE_CODER else "AT-OR-ABOVE-CHANCE",
            a2_bpb - CHANCE_CODER)


def g_e66e(a2_bpb, band=0.05):
    d = a2_bpb - E15_B0_FP32_7B
    return ("ONE-BYTE-IS-NEARLY-FREE" if d <= band else "ONE-BYTE-COSTS", d)


def g_e66f(agree, counted, bar=150):
    return ("RANK-SURVIVES" if agree >= bar else "RANK-DOES-NOT-SURVIVE"), agree, counted


# ============================================================ self-tests
def selftest():
    n = [0]

    def ok(c, m):
        n[0] += 1
        if not c:
            raise SystemExit("SELFTEST FAILED: " + m)

    ok(abs(BYTES_PER_TOK - 4.22945205479452) < 1e-12, "bytes/token is not E1's")
    ok(abs(CHANCE_CODER - 4.070106) < 1e-5, "Coder chance line wrong: %.6f" % CHANCE_CODER)
    ok(abs(CHANCE_QWEN - 4.069819) < 1e-5, "Qwen chance line wrong: %.6f" % CHANCE_QWEN)

    # the planted-control gates must FIRE on their own reference and DIE off it
    ok(g_e66a(E62_15B_I8)[0] == "FIRES", "G-E66a does not fire on its own reference")
    ok(g_e66a(E62_15B_I8 + 1e-7)[0] == "DEAD", "G-E66a does not discriminate at 1e-7")
    ok(g_e66b(E16_B2)[0] == "FIRES", "G-E66b does not fire on its own reference")
    ok(g_e66b(E16_B2 + 1e-3 + 1e-9)[0] == "DEAD-AND-NEW-FINDING",
       "G-E66b does not escalate a large dense-path miss")
    ok(g_e66b(E16_B2 + 5e-4)[0] == "DEAD", "G-E66b does not fail between tol and alarm")

    # the question gates must be able to say NO
    ok(g_e66d(4.5)[0] == "AT-OR-ABOVE-CHANCE", "G-E66d cannot fail")
    ok(g_e66d(0.7)[0] == "BELOW-CHANCE", "G-E66d cannot pass")
    ok(g_e66e(0.674)[0] == "ONE-BYTE-IS-NEARLY-FREE", "G-E66e cannot pass")
    ok(g_e66e(4.0)[0] == "ONE-BYTE-COSTS", "G-E66e cannot fail")
    # E16's actual number must FAIL the score gate -- the gate has to reject the known negative
    ok(g_e66e(E16_B2)[0] == "ONE-BYTE-COSTS", "G-E66e would have passed E16's dead arm")
    # and the RANK gate must reject E16's 0/160 while accepting the fp32 donor's 160/160
    ok(g_e66f(0, 160)[0] == "RANK-DOES-NOT-SURVIVE", "G-E66f would have passed E16's 0/160")
    ok(g_e66f(160, 160)[0] == "RANK-SURVIVES", "G-E66f rejects a perfect arm")
    ok(g_e66f(149, 160)[0] == "RANK-DOES-NOT-SURVIVE", "G-E66f bar is not at 150")

    # the CONFIG assertion must discriminate, not merely accept
    good = "CONFIG  attn=avx4  attnr=none  fexp=libm  mvacc=4  threads=6  quant=int8"
    ok(re.search(r"attn=%s\b" % ATTN, good) and re.search(r"quant=int8\b", good),
       "the CONFIG assertion rejects a correct line")
    for bad in ("CONFIG  attn=serial  quant=int8", "CONFIG  attn=avx4  quant=packed",
                "CONFIG  attn=avx41  quant=int8"):
        ok(not (re.search(r"attn=%s\b" % ATTN, bad) and re.search(r"quant=int8\b", bad)),
           "the CONFIG assertion accepts a wrong line: %r" % bad)

    print("selftest: %d checks passed" % n[0])
    return 0


# ============================================================ stages
def write_prompt_ids():
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(HF, revision=REV)
    os.makedirs(TMP, exist_ok=True)
    out = []
    for i, p in enumerate(PROMPTS):
        ids = tk(p)["input_ids"]
        with open(os.path.join(TMP, "p%d.bin" % i), "wb") as f:
            f.write(struct.pack("<%di" % len(ids), *ids))
        out.append(ids)
    return out


def stage_export(fh):
    """Addendum A: NO --calib-seqs.  Every int8 artefact on disk records calib_seqs: None, and
    passing it would build these cells by a different construction from the rungs they are
    quoted against."""
    os.makedirs(TMP, exist_ok=True)
    for path, fold in ((A1, "layers"), (A2, "none")):
        if os.path.exists(path):
            confirm_sidecar(path, {"quant": "int8", "rule": "R8", "fold": fold,
                                   "head_ternary": True, "calib_seqs": None})
            log("  export SKIP %s (complete sidecar matches registered arm)"
                % os.path.basename(path), fh)
            continue
        cmd = [sys.executable, EXPORTER, "--model", HF, "--revision", REV,
               "--quant", "int8", "--rule", "R8", "--head-ternary", "--fold", fold,
               "--load-dtype", "float32", "--threads", "6", "--out", path]
        log("  export %s  (--fold %s)" % (os.path.basename(path), fold), fh)
        t0 = time.time()
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit("export failed for %s" % path)
        log("    done in %.0f s, %.2f GB" % (time.time() - t0,
                                             os.path.getsize(path) / 1073741824.0), fh)


def stage_bpb(fh, controls_only=False, prior=None):
    out = dict(prior or {})
    if prior is None:
        log("\n-- controls (each must FIRE before any 7 B int8 cell is read)", fh)
        meta_c1 = confirm_sidecar(
            CTL_15B_I8, {"quant": "int8", "rule": "R8", "fold": "none",
                          "head_ternary": True, "calib_seqs": None})
        c1 = run_bpb(CTL_15B_I8, IDS_15B, "int8")
        v1, d1 = g_e66a(c1["bpb"])
        log("  G-E66a  1.5B I8   %.13f  vs E62 %.13f  |d| %.3e  -> %s"
            % (c1["bpb"], E62_15B_I8, d1, v1), fh)

        meta_c2 = confirm_sidecar(
            CTL_7B_TERN, {"quant": "packed", "rule": "R3", "fold": "layers",
                          "head_ternary": True})
        c2 = run_bpb(CTL_7B_TERN, IDS_7B, "packed")
        v2, d2 = g_e66b(c2["bpb"])
        log("  G-E66b  7B tern   %.13f  vs E16 %.9f   |d| %.3e  -> %s"
            % (c2["bpb"], E16_B2, d2, v2), fh)

        out["controls"] = {
            "G_E66a": {"cell": c1, "sidecar": meta_c1, "verdict": v1,
                        "abs_d": d1, "reference": E62_15B_I8},
            "G_E66b": {"cell": c2, "sidecar": meta_c2, "verdict": v2,
                        "abs_d": d2, "reference": E16_B2},
        }
        out["controls_fire"] = (v1 == "FIRES" and v2 == "FIRES")

    if not out.get("controls_fire", False):
        log("\n  CONTROLS DID NOT BOTH FIRE -- no 7 B int8 cell may be read (brief 4.2).", fh)
        log("  Numbers above are printed; no conclusion is drawn.", fh)
        return out
    if controls_only:
        log("\n  controls-only preflight complete -- no E66 treatment cell was read.", fh)
        return out

    log("\n-- the cells", fh)
    cells = {}
    for tag, path, fold in (("A1", A1, "layers"), ("A2", A2, "none")):
        meta = confirm_sidecar(path, {"quant": "int8", "rule": "R8", "fold": fold,
                                      "head_ternary": True, "calib_seqs": None})
        c = run_bpb(path, IDS_7B, "int8")
        c["sidecar"] = meta
        cells[tag] = c
        log("  %s (fold=%-6s) BPB %.12f   vs chance %+.6f   vs fp32 %+.6f   (%.0f s)"
            % (tag, fold, c["bpb"], c["bpb"] - CHANCE_CODER,
               c["bpb"] - E15_B0_FP32_7B, c["seconds"]), fh)
    out["cells"] = cells
    return out


def stage_greedy(fh):
    """E6/E7/E16's protocol: 5 prompts x 32 new tokens, greedy, against the fp32 PyTorch donor
    with the KV cache on.  E7's stored reference is not on disk, so it is regenerated here."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from e6_generate import run as erun, parse_gen

    prompt_ids = write_prompt_ids()
    arms = {}
    for tag, path in (("A1", A1), ("A2", A2)):
        runs = []
        for i in range(len(PROMPTS)):
            pfx = os.path.join(TMP, "%s_p%d" % (tag, i))
            r = parse_gen(erun([ENGINE, "--weights", path, "--threads", THREADS,
                                "--attn", ATTN, "--generate",
                                os.path.join(TMP, "p%d.bin" % i), str(N_NEW), pfx]))
            runs.append({"prompt": i, "ids": r["ids"]})
            log("  %s p%d generated" % (tag, i), fh)
        arms[tag] = runs

    log("  reference: fp32 PyTorch donor, KV cache on (E7's protocol)", fh)
    tk = AutoTokenizer.from_pretrained(HF, revision=REV)
    m = AutoModelForCausalLM.from_pretrained(HF, revision=REV, torch_dtype=torch.float32,
                                             attn_implementation="eager")
    m.eval()
    torch.set_num_threads(int(THREADS))
    ref = []
    for i in range(len(PROMPTS)):
        ids = list(prompt_ids[i])
        with torch.no_grad():
            o = m(torch.tensor([ids]), use_cache=True)
            for _ in range(N_NEW):
                nxt = int(torch.topk(o.logits[0, -1].float(), 1).indices[0])
                ids.append(nxt)
                o = m(torch.tensor([[nxt]]), past_key_values=o.past_key_values, use_cache=True)
        ref.append({"prompt": i, "ids": ids})
        log("  REF p%d done" % i, fh)
    del m

    rr = {x["prompt"]: x for x in ref}
    summary = {}
    for tag, runs in arms.items():
        tot = match = 0
        firstdiv = None
        for r in runs:
            ours, theirs = r["ids"][-N_NEW:], rr[r["prompt"]]["ids"][-N_NEW:]
            for k in range(N_NEW):
                tot += 1
                if ours[k] == theirs[k]:
                    match += 1
                elif firstdiv is None:
                    firstdiv = (r["prompt"], k)
        v, a, c = g_e66f(match, tot)
        summary[tag] = {"matched": match, "counted": tot, "first_div": firstdiv, "verdict": v}
        log("  G-E66f  %s  greedy %d/%d  first divergence %s  -> %s"
            % (tag, match, tot, firstdiv, v), fh)
        log("     text: %s" % json.dumps(tk.decode(arms[tag][0]["ids"][-N_NEW:]))[:160], fh)
    return {"arms": arms, "ref": ref, "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stage", default="all",
                    choices=("all", "controls", "continue", "export", "bpb", "greedy"))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    selftest()

    os.makedirs(RES, exist_ok=True)
    fh = open(os.path.join(HERE, "e66_run.log"), "a", encoding="utf-8")
    log("\n=== E66 -- one byte at seven billion.  QUALITY AND RANK ONLY, no rate. ===", fh)
    log("engine %s   arm --attn %s (ASSERTED from CONFIG)" % (os.path.basename(ENGINE), ATTN), fh)
    log("donor %s @ %s" % (HF, REV[:12]), fh)

    for f in (ENGINE, EXPORTER, IDS_7B, IDS_15B, CTL_15B_I8, CTL_7B_TERN):
        if not os.path.exists(f):
            raise SystemExit("missing: " + f)

    provenance = committed_provenance()
    log("runner HEAD %s  blob %s (worktree match ASSERTED)"
        % (provenance["head_commit"][:12], provenance["runner_git_blob"]), fh)
    log("engine sha256 %s" % provenance["engine_sha256"], fh)

    t0 = time.time()
    res = {"experiment": "E66", "brief": "BRIEF_E66_ONE_BYTE_AT_SEVEN_BILLION.md",
           "addenda": ["A", "B", "C"], "engine": os.path.basename(ENGINE), "attn_arm": ATTN,
           "provenance": provenance,
           "donor": HF, "revision": REV, "threads": int(THREADS),
           "protocol": {"n_predicted": N_PRED, "scored_bytes": SCORED_BYTES,
                        "bytes_per_token": BYTES_PER_TOK,
                        "chance_coder": CHANCE_CODER, "chance_qwen": CHANCE_QWEN},
           "references": {"E62_15B_I8": E62_15B_I8, "E16_B2": E16_B2,
                           "E15_B0_fp32_7B": E15_B0_FP32_7B, "E62_ladder": E62_LADDER},
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    controls = None
    if a.stage in ("all", "controls"):
        controls = stage_bpb(fh, controls_only=True)
        control_res = dict(res)
        control_res.update(controls)
        control_res["seconds_total"] = time.time() - t0
        control_path = os.path.join(RES, "e66_controls.json")
        json.dump(control_res, open(control_path, "w"), indent=1)
        log("\nwrote controls-only result %s in %.0f s"
            % (control_path, control_res["seconds_total"]), fh)
        if not controls.get("controls_fire", False):
            log("E66 STOPS BEFORE EXPORT -- the planted controls did not both fire.", fh)
            return 2
        if a.stage == "controls":
            return 0
    elif a.stage == "continue":
        controls, checkpoint = load_pinned_controls(provenance)
        res["continued_from_controls"] = checkpoint
        log("\n-- pinned controls checkpoint %s (sha256 and producing apparatus ASSERTED)"
            % checkpoint["sha256"], fh)
        log("  G-E66a FIRES; G-E66b FIRES; controls are not rerun or rewritten.", fh)

    if a.stage in ("all", "continue", "export"):
        log("\n-- export (addendum A: NO --calib-seqs on int8)", fh)
        stage_export(fh)
        if a.stage == "export":
            log("\nexport-only stage complete; canonical result was not written.", fh)
            return 0

    if a.stage in ("all", "continue", "bpb"):
        measured = stage_bpb(fh, prior=controls) if controls is not None else stage_bpb(fh)
        res.update(measured)
        json.dump(res, open(os.path.join(RES, "e66_bpb.json"), "w"), indent=1)
        if not res.get("controls_fire", False):
            log("E66 STOPS -- controls failed; canonical result was not written.", fh)
            return 2

    if a.stage == "greedy":
        bpb_path = os.path.join(RES, "e66_bpb.json")
        if not os.path.exists(bpb_path):
            raise SystemExit("greedy stage requires an existing e66_bpb.json")
        previous = json.load(open(bpb_path, encoding="utf-8"))
        if not previous.get("controls_fire") or not previous.get("cells"):
            raise SystemExit("e66_bpb.json is not an admissible scored run")
        old_blob = previous.get("provenance", {}).get("runner_git_blob")
        if old_blob != provenance["runner_git_blob"]:
            raise SystemExit("runner blob differs from e66_bpb.json; refusing a mixed apparatus")
        res = previous
        res["seconds_before_greedy"] = float(previous.get("seconds_total", 0.0))
        res["greedy_continued_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    if a.stage in ("all", "continue", "greedy"):
        log("\n-- greedy (RANK partner, E14 section 3 -- the gate E16 failed at 0/160)", fh)
        res["greedy"] = stage_greedy(fh)
        json.dump(res, open(os.path.join(RES, "e66_full.json"), "w"), indent=1)

    elapsed = time.time() - t0
    if a.stage == "greedy":
        res["seconds_greedy_stage"] = elapsed
        res["seconds_total"] = res["seconds_before_greedy"] + elapsed
    else:
        res["seconds_total"] = elapsed
    json.dump(res, open(os.path.join(RES, "e66_one_byte_at_7b.json"), "w"), indent=1)
    log("\nwrote %s in %.0f s" % (os.path.join(RES, "e66_one_byte_at_7b.json"),
                                  res["seconds_total"]), fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
