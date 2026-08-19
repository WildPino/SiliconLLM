#!/usr/bin/env python3
"""Recover the DECIDING metric offline, because the uploaded package could not compute it.

WHAT WENT WRONG. The data packages uploaded to Kaggle were a generation OLDER than the local ones: they
carried 6 files, not 7, missing p62_s0.u16. The trainer treats P62 as optional (the MVE packages had no
code-val), so with the file absent it printed no [DECIDING] line and still wrote STAGE1-DONE. Both arms
therefore completed cleanly and produced only the tail val -- the one number the prereg forbids as a
decider. The failure is the house pattern: a healthy run answering a different question than the one that
decides.

WHY OFFLINE RECOVERY IS LEGITIMATE, not a patch over the hole. The P62 BPB is a deterministic function of
(trained weights, p62 stream, tokenizer). The stage-C exit checkpoint IS those weights -- stage C is the
only stage these arms ran, so the saved final .pt is exactly the model whose P62 the trainer would have
printed. This recomputes it with the identical code path (bpb_on + the planted byte invariant), so the
number is the one that should have appeared at stage exit, not an approximation of it. It is declared as
recovered-offline because the uploaded artifact was deficient -- a deviation on the record, not a silent
substitution.

ARCHITECTURE GUARD ADDED 2026-08-19 -- AND IT DID NOT RE-DERIVE THE ADOPTED NUMBERS. The V=2048 adoption
and the stage-3 anchor 1.1376 stand exactly as sealed, produced by the generation of this file that had no
such guard. Nothing here re-ran them, and nobody should read this file as though it had: the guard protects
FUTURE invocations. What was measured about the old generation is that its exposure was nil (all three
stage-1 arms carry qat_alpha=0 with stages=C, and all three load strict=True clean) -- the numbers were
right, and they were not protected. Those are different properties, and only the second one changed today.

Run: python benchmarks/phase64/data/ws3_recover_p62.py
"""
import io, json, math, os, sys, zipfile
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase64", "mve"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase62"))
from mve_model import MVEStudent, S0            # noqa: E402
from cartography import Bpe                      # noqa: E402

RES = os.path.join(ROOT, "kaggle_rung1", "results")
# Third field: the tail-val BPB Kaggle actually PRINTED for this arm. It is the planted control on the
# recovery itself -- if the offline pipeline reproduces a number Kaggle emitted, using the same weights
# and the same code, then the P62 number it also produces (which Kaggle could not emit) is trustworthy by
# the same machinery. A recovered decider nobody can cross-check is an assumption.
#
# arm3 carries kaggle_tail=None for a reason worth naming rather than leaving as a blank field: its console
# log was never retained (see below), so there is no printed tail val to reproduce. Its control is instead
# the SAME-RUN one -- arm1 and arm2 reproduce their printed tail vals in this very invocation, so the
# pipeline is demonstrated faithful in the session that also produces arm3's number. A control that fires
# on known-positives beside the unknown is the discipline; a recovered decider nobody can cross-check is
# an assumption.
#
# WHY arm3 IS RE-DERIVED AT ALL (2026-07-24). Its P62 = 1.1376 became load-bearing: it is the i.i.d. anchor
# of the stage-3 domain finding (+0.0338 = 6.8 sigma against CE-on-window). But the number lived ONLY in
# the adjudication documents -- no log under results/. An adjudication is authority, not evidence.
#
# Fields: (arm, V, package-relative-path, kaggle-printed tail val or None, xproj rank)
ARMS = [("arm1_V2048", 2048, "account_1/arm1_V2048", 1.1452, 0),
        ("arm2_V4096", 4096, "account_2/arm2_V4096", 1.1328, 0),
        ("arm3_xproj26_V2048", 2048, "stage2/arm3_xproj26_V2048", None, 26)]
SEQ = 512
BATCH = 8
EVAL_TOK = 200_000        # mve_train --eval-tok default; the cap the stage-exit tail val used


def load_extracted(dirpath):
    """torch.save writes a zip; Kaggle handed it back EXTRACTED (data.pkl + data/ + version). Re-zip in
    memory under a single top-level folder -- the exact layout torch.load expects -- and load that."""
    buf = io.BytesIO()
    root = os.path.basename(dirpath.rstrip("/\\"))
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for base, _, files in os.walk(dirpath):
            for fn in files:
                full = os.path.join(base, fn)
                rel = os.path.relpath(full, dirpath).replace("\\", "/")
                z.write(full, f"{root}/{rel}")
    buf.seek(0)
    return torch.load(buf, map_location="cpu", weights_only=False)


def recall_slot_keys(V):
    """The EXACT key set the recall slot contributes, COMPUTED from the model rather than typed out, so it
    cannot drift from mve_model.py. It replaces `[m for m in missing if "recall" not in m]`, which forgave
    any key that happened to carry the substring -- and an exception is the point at which a guard loses."""
    a = MVEStudent(V, **S0)
    before = set(a.state_dict())
    a.add_recall(lam_nce=0.0)
    return frozenset(set(a.state_dict()) - before)


def build_from_cfg(ck, arm, table_rank, V, dev):
    """THE ARTEFACT IS THE AUTHORITY; THE TABLE IS AN ASSERTION.

    The architecture is derived from the checkpoint's own cfg, the model is built from THAT, and only then
    is the ARMS table asserted against it. Done the other way round -- assert, then build from the table --
    a checkpoint whose config the table does not know would be reconstructed in silence as whatever the
    table says, which is precisely the hole this guard closes.

    WHY IT EXISTS (2026-08-19). This function loaded weights with strict=False and printed, rather than
    raised, on missing/unexpected keys -- and printed only the first four of them. The vocabulary was
    asserted (emb.weight width), which is data identity, not architecture. Measured exposure of the
    numbers already adopted: nil, because all three stage-1 arms carry qat_alpha=0 with stages=C. Measured
    exposure of the MECHANISM: total -- BitLinear158 exposes its parameter as `weight`, the same name and
    shape as nn.Linear, so a ternary state loads into a dense model with zero missing and zero unexpected
    keys and the old code said nothing at all. That case moves the tail val by +0.3112 BPB.

    Key-level checking cannot catch that case by construction, so it is caught here instead, on cfg."""
    cfg = ck.get("cfg")
    if not isinstance(cfg, dict):
        raise SystemExit(
            f"{arm}: the checkpoint carries no cfg dict, so the architecture cannot be derived from the\n"
            "  artefact -- and this tool refuses to derive it from its own table. Recover a checkpoint\n"
            "  saved by a trainer generation that records cfg; do NOT relax this to trust ARMS.")
    for k in ("stages", "xproj_rank", "qat_alpha", "recall"):
        if k not in cfg:
            raise SystemExit(f"{arm}: cfg has no '{k}', so the architecture is not fully determined by the\n"
                             "  artefact. Refusing rather than filling the gap from the table.")

    stages = str(cfg["stages"])
    rank = int(cfg["xproj_rank"])
    ternary = "D" in stages
    unsupported = [s for s in stages if s not in "CD"]
    if unsupported:
        # Stage E upcycles to MoE and inserts the recall slot; reconstructing that faithfully needs
        # load_w/seed/sparse and is not a transcription. Refusing by name beats building something close.
        raise SystemExit(
            f"{arm}: cfg says stages={stages!r}. This tool reconstructs stage C and stage D exit states\n"
            f"  only; {unsupported} would need the stage-E upcycle replayed. Refusing to evaluate an\n"
            "  architecture it cannot rebuild exactly. Extend build_from_cfg deliberately, with the\n"
            "  both-direction exercise, before removing this refusal.")

    model = MVEStudent(V, **S0)
    swapped = ""
    if rank > 0:
        nx, dn, ln = model.lowrank_xproj(rank)
        swapped = (f"x_proj low-rank r={rank}: {nx} swapped | {ln}/{dn} params "
                   f"= {100.0*ln/max(dn,1):.1f}% of dense")
    if ternary:
        n = model.qat_ternary(alpha_sched=int(cfg["qat_alpha"]) > 0)
        if int(cfg["qat_alpha"]) > 0:
            model.set_qat_alpha(1.0)     # stage D is complete in an exit checkpoint -> alpha is 1, not 0
        swapped += (" | " if swapped else "") + f"QAT ternary: {n} MLP linears"
    want_recall = (str(cfg["recall"]) == "on")
    if want_recall:
        raise SystemExit(f"{arm}: cfg says recall=on, but the slot is inserted at stage E and stages="
                         f"{stages!r} never reaches it. Contradictory artefact -- refusing.")

    # ORDER IS LOAD-BEARING: build, then swap, THEN move to the device. The first attempt at this
    # cross-check did .to(dev) first, so the freshly constructed low-rank layers stayed on CPU and the
    # forward died on a device mismatch. The trainer does surgery-then-move; transcribe its order.
    model = model.to(dev)

    # The table is now an ASSERTION against what the artefact said, and it fails closed.
    if rank != table_rank:
        raise SystemExit(
            f"{arm}: the checkpoint's cfg declares xproj_rank={rank}, the ARMS table declares "
            f"{table_rank}.\n  The model was built from the CHECKPOINT ({rank}), which is the authority. "
            "Fix the ARMS\n  entry to match the artefact; do not change the artefact to match the table.")
    return model, swapped, (recall_slot_keys(V) if want_recall else frozenset())


def load_or_refuse(model, ck, arm, allowed_missing):
    """strict, and it REFUSES. Reports the COMPLETE counts, not the first four: the old code truncated at
    [:4], so a hundred discordant keys and four discordant keys printed the same shape of line."""
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    bad = sorted(set(missing) - set(allowed_missing))
    unexpected = sorted(unexpected)
    if bad or unexpected:
        raise SystemExit(
            f"{arm}: ARCHITECTURE MISMATCH -- {len(bad)} missing key(s), {len(unexpected)} unexpected.\n"
            f"  missing    ({len(bad)}): {bad}\n"
            f"  unexpected ({len(unexpected)}): {unexpected}\n"
            f"  allowed missing (recall slot, only when cfg.recall==on): {sorted(allowed_missing)}\n"
            "  The remedy is to build the architecture the checkpoint declares, not to relax this to\n"
            "  strict=False: a randomly-initialised layer evaluates cleanly and answers a different\n"
            "  question. If the cfg is right and this still fires, mve_model.py has moved under it.")
    return len(missing) - len(bad)


def bpb_on(model, stream, V, el, dev, cap=0):
    """The trainer's bpb_on, transcribed verbatim so the number is the one stage-exit would have printed:
    cap=0 = whole stream, full windows batched, ragged remainder alone, bytes from exp_len."""
    model.eval(); bits = 0.0; nb = 0
    with torch.no_grad():
        n = (min(cap, len(stream) - 1) if cap else len(stream) - 1)
        W = SEQ; full = n // W
        for i in range(0, full, BATCH):
            k = min(BATCH, full - i)
            st = [(i + j) * W for j in range(k)]
            x = torch.from_numpy(np.stack([stream[p0:p0+W] for p0 in st])).to(dev)
            y = torch.from_numpy(np.stack([stream[p0+1:p0+1+W] for p0 in st])).to(dev)
            lg, _ = model(x, y)
            bits += F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction="sum").item() / math.log(2)
            nb += int(el[y.reshape(-1).cpu()].sum())
        rem = n - full * W
        if rem > 0:
            p0 = full * W
            x = torch.from_numpy(stream[p0:p0+rem][None]).to(dev)
            y = torch.from_numpy(stream[p0+1:p0+1+rem][None]).to(dev)
            lg, _ = model(x, y)
            bits += F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction="sum").item() / math.log(2)
            nb += int(el[y.reshape(-1).cpu()].sum())
    return bits / max(nb, 1), nb


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"recovering the DECIDING P62 code-val BPB from the saved stage-C checkpoints  (dev={dev})\n")
    rows = []
    for arm, V, pkg, kaggle_tail, xrank in ARMS:
        ckdir = os.path.join(RES, arm)
        if not os.path.isdir(ckdir):
            print(f"  {arm}: no checkpoint at {ckdir} -- skipping"); continue
        ck = load_extracted(ckdir)
        data = os.path.join(ROOT, "kaggle_rung1", pkg, "data")
        p62 = np.fromfile(os.path.join(data, f"p62_s0.u16"), dtype=np.uint16).astype(np.int64)
        ids = np.fromfile(os.path.join(data, f"ts_s0.u16"), dtype=np.uint16).astype(np.int64)
        el = torch.tensor(Bpe.load(os.path.join(data, f"bpe{V}_ts.bin")).exp_len, dtype=torch.long)
        meta = json.load(open(os.path.join(data, "meta_s0.json")))
        p62_bytes = int(meta["p62_bytes"]); ntr = int(meta["n_train_tok"])

        # The checkpoint's own embedding width names its vocabulary: assert it against V BEFORE loading, so
        # a swapped checkpoint is a named refusal, not a silent strict=False shape drop. This is DATA
        # identity; the architecture is a separate question and is settled by build_from_cfg below.
        ck_vocab = ck["model"]["emb.weight"].shape[0]
        assert ck_vocab == V, f"{arm}: checkpoint vocab {ck_vocab} != expected {V} -- wrong file"

        model, swapped, allowed = build_from_cfg(ck, arm, xrank, V, dev)
        cfg = ck["cfg"]
        print(f"  {arm}: arch from cfg -> stages={cfg['stages']!r} xproj_rank={cfg['xproj_rank']} "
              f"qat_alpha={cfg['qat_alpha']} recall={cfg['recall']!r}"
              + (f"\n    {swapped}" if swapped else ""))
        load_or_refuse(model, ck, arm, allowed)

        # PLANTED CONTROL first: reproduce the tail val Kaggle printed, same weights, same code, same cap.
        print(f"  {arm}  V={V}")
        if kaggle_tail is not None:
            tail = ids[ntr:]
            tail_bpb, _ = bpb_on(model, tail, V, el, dev, cap=EVAL_TOK)
            d_tail = abs(tail_bpb - kaggle_tail)
            faithful = d_tail < 0.002
            print(f"    tail-val cross-check : offline {tail_bpb:.4f}  vs Kaggle-printed {kaggle_tail:.4f}  "
                  f"|delta| {d_tail:.4f}  -> "
                  f"{'REPRODUCES (pipeline faithful)' if faithful else 'DIVERGES -- recovery suspect'}")
        else:
            faithful = None      # carried by the same-run controls above, not by a control of its own
            print("    tail-val cross-check : NO PRINTED VALUE (console log not retained) -- this arm leans on"
                  " the same-run arm1/arm2 controls")

        bpb, nb = bpb_on(model, p62, V, el, dev)
        total = nb + int(el[int(p62[0])])
        ok = (total == p62_bytes)
        rows.append((arm, V, bpb, nb, total, p62_bytes, ok and (faithful is not False)))
        print(f"    P62 code-val BPB [DECIDING]  = {bpb:.4f}")
        print(f"    over {nb} bytes + {total-nb} B unscored first token = {total} "
              f"(declared {p62_bytes})  [byte invariant {'HOLDS' if ok else 'FAILED'}]\n")

    by = {r[0]: r[2] for r in rows}
    if "arm1_V2048" in by and "arm2_V4096" in by:
        (a1, b1), (a2, b2) = ("arm1_V2048", by["arm1_V2048"]), ("arm2_V4096", by["arm2_V4096"])
        lo, hi = (a1, b1), (a2, b2)
        if b2 < b1: lo, hi = (a2, b2), (a1, b1)
        print(f"  screening result (P62, the decider): {lo[0]} lower at BPB {lo[1]:.4f} vs {hi[1]:.4f} "
              f"-- delta {hi[1]-lo[1]:+.4f}")
        # THIS LINE ONCE CARRIED THE PROJECT'S OWN MIS-CITATION and printed it every run. It read
        # "decisive if delta > 0.010", i.e. a 2-sigma bar, and therefore labelled the stage-1 delta of
        # 0.0066 "INSIDE noise -- not separable". The sealed prereg SS4 rule is: "adopt the lower code-val
        # BPB if the margin > sigma_seed; if <= sigma_seed, adopt the pre-declared tie-breaker" -- sigma_seed
        # is 0.005, so 0.0066 ADOPTS and the tie-breaker never fires. The 2-sigma thresholds that do exist
        # live in SS4.3 (the KD trend) and in claim-grade language, not in the adoption rule. The Architect
        # caught the same error in a Builder report on 2026-07-22; it survived here because a tool that
        # prints a threshold re-injects it every time it runs. ADOPTION and CLAIM are separate questions and
        # this now prints both, each against its own bar.
        d = abs(b1 - b2)
        print(f"  sigma_seed = 0.005; delta is {d/0.005:.1f} sigma")
        print(f"    ADOPTION (SS4 rule, bar = sigma_seed 0.005): {d:.4f} "
              f"{'>' if d > 0.005 else '<='} 0.005 -> "
              f"{'ADOPT the lower arm; tie-breaker does NOT fire' if d > 0.005 else 'TIE-BREAKER fires'}")
        print(f"    CLAIM-GRADE (separate question, bar = 2 sigma 0.010): {d:.4f} "
              f"{'>' if d > 0.010 else '<='} 0.010 -> "
              f"{'publishable as a difference' if d > 0.010 else 'NOT a claim -- label it a screening'}")
        if not all(r[6] for r in rows):
            print("  !! a byte invariant FAILED -- the recovered number is NOT comparable; investigate before quoting")
    # arm3 is the i.i.d. anchor of the stage-3 domain finding. Re-derive it against the value the
    # adjudication documents carry, and against the stage-3 CE-on-window point, so the 6.8-sigma claim
    # rests on a number this script printed rather than on a citation.
    if "arm3_xproj26_V2048" in by:
        b3 = by["arm3_xproj26_V2048"]
        BRIEF, KD_A0 = 1.1376, 1.1715      # brief SS12 in-run value; stage-3 arm4 log
        d = abs(b3 - BRIEF)
        print(f"  arm3 anchor re-derived: {b3:.4f}  vs brief-recorded {BRIEF:.4f}  |delta| {d:.4f}"
              f"  -> {'CONFIRMS the cited number' if d < 0.002 else 'DISAGREES -- the citation is wrong, stop'}")
        print(f"  domain gap (stage-3 CE-on-window {KD_A0:.4f} - arm3 CE-i.i.d. {b3:.4f}) = "
              f"{KD_A0-b3:+.4f} = {(KD_A0-b3)/0.005:.1f} sigma_seed")
    print("\nSTOP. Recovered offline; no commit. Deciding metric is declared recovered-offline in the report.")


if __name__ == "__main__":
    main()
