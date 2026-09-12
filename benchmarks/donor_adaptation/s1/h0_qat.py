#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H0 step 3 (T4) -- straight-through QAT on the ternary low-rank attention factors.

Plan: docs/research/donor_adaptation/decisions/T4_HEALING_PROPOSAL.md s3 (H0), s5b.
Budget: <= 3 GPU-h.  This is a SIGNAL-DETECTION experiment, not a success experiment.

THE QUESTION.  E22 measured that the budget-feasible object and the working object are not the
same object: rank-512 q/o attention costs +0.052689 BPB in fp32 (E21's QO512, tf 144/160) and
collapses to tf 28/160 once BOTH factors are ternarized, because the two factors' errors
multiply.  Post-hoc conversion of a factored form is therefore dead.  H0 asks the one question
that is left: DO GRADIENTS MOVE THIS OBJECT AT ALL?

WHAT IS TRAINED.  For 28 layers x {q_proj, o_proj}, the fp32 masters A [1536,512], s [512],
B [512,1536] -- 88.1 M parameters, 5.7% of the donor.  EVERYTHING ELSE IS FROZEN: k_proj, v_proj,
all three FFN organs, every norm, the embedding and the tied head.  No FFN carve and no router,
so E23's verdict cannot invalidate this run either way.

THE FORMAT IS APPLIED AT EVERY STEP, NOT APPROXIMATED.  The forward calls the SHIPPED quantizer
(t2_rules.r3_actsearch, per-row threshold searched over D_GRID, activation-weighted) on the live
masters, under no_grad, and passes the gradient straight through:  W_eff = W + (q*a - W).detach().
The search costs ~0.07% of the step, so there is no reason to freeze or approximate it, and
"learn the format" means the format the engine would actually apply.

fp16 ON THE T4.  attn_implementation="sdpa" is MANDATORY and asserted at load: this donor in
fp16 under `eager` goes non-finite on a T4 at model.layers.0.self_attn.o_proj -- one of the two
organs trained here -- while sdpa is finite (proposal s5b(1), measured on the card).  The
quantizer search runs in fp32 outside autocast so the format does not depend on the compute
dtype.

A TRAP THAT WOULD HAVE FAKED THIS EXPERIMENT'S NULL.  `density/common.py` line 28 calls
torch.set_grad_enabled(False) AT IMPORT -- correct for every probe in this programme, all of
which are inference-only, and it saves memory.  But any trainer that imports it, directly or
transitively, silently trains NOTHING: every grad is None, the optimizer steps on nothing, the
loss never moves, and the run reports "gradients do not move this object" -- which is EXACTLY
H0's null hypothesis.  This file therefore (a) imports nothing from the repo, (b) re-enables
grad explicitly and asserts it, and (c) carries G-H0e: after the first update THE SCALER
ACTUALLY APPLIED, the masters must have MOVED, checked against a snapshot, or the run aborts.
A null from this experiment only means something if the instrument can be shown to move.

RUN 1 (2026-09-11) DIED HERE, AND THE GATE WAS THE THING THAT WAS WRONG.  G-H0e originally read
the masters after `step == 1` unconditionally.  But torch.amp.GradScaler starts at
init_scale = 65536 BY DESIGN, and `scaler.step(opt)` DECLINES the update whenever unscale_ finds
an inf in the scaled gradients -- which at that starting scale is ordinary warm-up behaviour and
not a pathology.  A declined step is not an optimizer step, so the masters could not have moved,
so the gate fired on its own unmet precondition and stopped a run with nothing wrong with it.
The corroborating signature in that log is PyTorch's own "lr_scheduler.step() before
optimizer.step()" warning, which is emitted for exactly this reason.

The fix does not weaken the gate -- it makes the code test the sentence the gate was always
written in.  It waits up to G_H0E_WINDOW declined steps for an APPLIED one, reads the masters
after that one, and if the scaler declines every step in the window it aborts with a DIFFERENT
message: "the trainer never got an update in" and "gradients do not move this object" are two
different findings, and only the second one is H0's null.

WHAT THIS SCRIPT DOES NOT DO: measure the gate.  It reports an in-job teacher-forced count in
fp16 on the GPU so the run is watchable, but the H0 gate is re-measured on CPU in fp32 by
h0_eval.py with the same instrument every published number in this programme used.  GPU trains,
CPU measures.

Env: H0_STEPS, H0_BS, H0_LR, H0_ACCUM, H0_CKPT, H0_EVERY, H0_SMOKE, H0_DEV
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]      # t2_rules.D_GRID, verbatim
ORGANS = ("q_proj", "o_proj")


def log(m):
    print(m, flush=True)


# How many DECLINED steps G-H0e will sit through before giving up.  The scaler halves on every
# decline, so from init_scale 65536 it passes through every scale a T4 can represent in 16 of
# them; a window of 25 therefore cannot hide a real "the gradients are inf forever" failure, and
# widening it would not turn a failing run into a passing one -- it would only change which of
# the two abort messages is printed.
G_H0E_WINDOW = 25


def applied_steps(opt):
    """Updates the optimizer has ACTUALLY APPLIED, read out of AdamW's own per-parameter state.

    `scaler.step(opt)` is a no-op when unscale_ found an inf, and it returns None either way, so
    the only honest way to ask "did an update happen" is to ask the optimizer.  AdamW creates
    state["step"] on the first update it applies to a parameter and increments it after that.
    """
    n = 0
    for g in opt.param_groups:
        for prm in g["params"]:
            st = opt.state.get(prm)
            if st and "step" in st:
                v = st["step"]
                n = max(n, int(v.item() if hasattr(v, "item") else v))
    return n


# --------------------------------------------------------------------------------------------
# the shipped ternary rule, reimplemented here ONLY so the bundle is self-contained on Kaggle.
# h0_qat_selftest.py asserts it agrees with t2_rules.r3_actsearch bit-for-bit on the real
# masters; if that assert has not run, this file may not be trusted.
# --------------------------------------------------------------------------------------------
def r3_actsearch(w, act_rms):
    """Per row, the threshold in D_GRID minimising ||(w - alpha q) * act_rms||^2."""
    m = w.abs().mean(dim=1, keepdim=True)
    ww = act_rms.view(1, -1)
    best_err = best_q = best_a = None
    for f in D_GRID:
        d = f * m
        q = torch.sign(w) * (w.abs() > d).to(w.dtype)
        kept = (q != 0).to(w.dtype)
        num = (w * q * ww * ww).sum(dim=1, keepdim=True)
        den = (kept * ww * ww).sum(dim=1, keepdim=True).clamp_min(1e-12)
        a = (num / den).clamp_min(1e-5)
        r = w - a * q
        err = (r * r * ww * ww).sum(dim=1, keepdim=True)
        if best_err is None:
            best_err, best_q, best_a = err, q, a
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_a = torch.where(take, a, best_a)
            best_q = torch.where(take, q, best_q)
    return best_q, best_a


def ste(w, act_rms):
    """Ternarize with the shipped rule; gradient passes straight through to the fp32 master."""
    with torch.no_grad():
        q, a = r3_actsearch(w.detach().float(), act_rms)
        wq = q * a
    return w + (wq - w).detach()


class TernaryLowRank(nn.Module):
    """y = ternarize(A) . diag(s) . ternarize(B) . x  + bias, with A, s, B fp32 masters.

    B is E22's FOLDED factor (c * B) and s starts at ones, so at initialisation this module is
    E22's QO512-TB exactly -- asserted at zero tolerance by h0_factorize.py's G-H0a.
    """

    def __init__(self, A, s, B, rms_in, rms_A, bias):
        super().__init__()
        self.A = nn.Parameter(A.float())
        self.s = nn.Parameter(s.float())
        self.B = nn.Parameter(B.float())
        self.register_buffer("rms_in", rms_in.float())
        self.register_buffer("rms_A", rms_A.float())
        self._ck, self._cv = None, None
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias_buf", bias.clone())
            self.bias = "buf"

    def _quant(self):
        """(At, Bt) for the current masters, cached when nothing has changed.

        The D_GRID search is ~0.07% of a TRAINING step and dominates a CPU EVAL, where the
        masters are frozen and the same 112 searches would be repeated for every one of the 160
        greedy positions.  The key is torch's own _version counter, which increments on the
        in-place write the optimizer makes -- so the cache is exact rather than a mode flag that
        can go stale between an eval, a train and another eval.

        IT IS AN EVAL-ONLY CACHE AND IT IS NEVER SERVED UNDER GRAD, which the first version of
        this method got wrong and G-H0e caught on the CPU smoke of 2026-09-11.  The cached (At,
        Bt) are built under no_grad, so handing them back inside a training step silently takes
        A and B OUT OF THE GRAPH: their .grad stays None, the optimizer skips them, and the only
        master that still learns is s -- 512 numbers of the 88.1 M this experiment is about.
        The step-0 teacher-forced probe runs under no_grad and fills the cache, and the masters
        have not moved since, so the key still MATCHED at step 1: the run would have reported
        "gradients do not move this object" having never sent a gradient to A or B at all.  That
        is H0's null, fabricated.  G-H0g in h0_selftest.py is the planted control and it FIRES
        on the unguarded version.
        """
        key = (self.A._version, self.B._version, self.s._version)
        if self._ck == key and not torch.is_grad_enabled():
            return self._cv
        Bt = ste(self.B, self.rms_in)
        At = ste(self.A, self.rms_A * self.s.detach().abs().clamp_min(1e-8))
        if not torch.is_grad_enabled():
            self._ck, self._cv = key, (At, Bt)
        return At, Bt

    def forward(self, x):
        # quantizer in fp32, outside autocast: the format must not depend on the compute dtype
        with torch.autocast(device_type=("cuda" if x.is_cuda else "cpu"), enabled=False):
            At, Bt = self._quant()
        h = F.linear(x, Bt.to(x.dtype))
        h = h * self.s.to(x.dtype)
        y = F.linear(h, At.to(x.dtype))
        if self.bias is not None:
            y = y + self.bias_buf.to(y.dtype)
        return y


def build(model, bundle, layer_ids, device):
    """Replace q_proj/o_proj with the factored module, for the layers the BUNDLE actually holds.

    Derived from the bundle rather than taken on trust, so a partial (smoke) bundle installs
    exactly what it contains instead of raising halfway through and leaving a half-built model.
    """
    z = np.load(bundle)
    have = sorted({int(k[1:3]) for k in z.files if k.startswith("L")})
    layer_ids = [li for li in layer_ids if li in have]
    n = 0
    for li in layer_ids:
        attn = model.model.layers[li].self_attn
        for nm in ORGANS:
            p = "L%02d.%s" % (li, nm)
            old = getattr(attn, nm)
            mod = TernaryLowRank(
                torch.from_numpy(z[p + ".A"]), torch.from_numpy(z[p + ".s"]),
                torch.from_numpy(z[p + ".B"]), torch.from_numpy(z[p + ".rms_in"]),
                torch.from_numpy(z[p + ".rms_A"]),
                None if old.bias is None else old.bias.data.detach().clone())
            setattr(attn, nm, mod.to(device))
            n += 1
    return n


@torch.no_grad()
def tf_count(model, pids, tgts, device):
    """E20 part B's instrument, run on the GPU in fp16.  A WATCHABLE NUMBER, NOT THE GATE."""
    model.eval()
    tf = 0
    for i in range(len(pids)):
        full = torch.tensor([pids[i] + tgts[i]], device=device)
        lg = model(full).logits[0].float()
        for k in range(len(tgts[i])):
            row = lg[len(pids[i]) - 1 + k]
            tf += int(int(torch.argmax(row)) == tgts[i][k])
    model.train()
    return tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--probe", required=True, help="prompt/target ids for the watchable tf count")
    ap.add_argument("--out", default="h0_trained.npz")
    ap.add_argument("--steps", type=int, default=int(os.environ.get("H0_STEPS", "4000")))
    ap.add_argument("--bs", type=int, default=int(os.environ.get("H0_BS", "2")))
    ap.add_argument("--accum", type=int, default=int(os.environ.get("H0_ACCUM", "8")))
    ap.add_argument("--lr", type=float, default=float(os.environ.get("H0_LR", "2e-4")))
    ap.add_argument("--every", type=int, default=int(os.environ.get("H0_EVERY", "250")))
    ap.add_argument("--max-hours", type=float, default=2.8)
    # RESUME.  --factors accepts any bundle with this key layout, so a previous run's
    # h0_trained.npz IS the resume path for the WEIGHTS.  The batches, however, are drawn
    # i.i.d. from a FIXED rng, so continuing with the same seed would redraw the same
    # sequence the first run already trained on.  Default is 1717 so every existing run
    # reproduces bit for bit; a continuation must pass a different one.  Adam moments are
    # NOT checkpointed and a resume restarts them -- stated, not hidden.
    ap.add_argument("--seed", type=int, default=1717)
    ap.add_argument("--smoke", action="store_true", default=os.environ.get("H0_SMOKE") == "1")
    a = ap.parse_args()

    # see the module docstring: an inference default elsewhere in this repo would silently
    # turn this experiment into its own null.  Explicit, and asserted.
    torch.set_grad_enabled(True)
    if not torch.is_grad_enabled():
        raise SystemExit("AUTOGRAD IS DISABLED -- STOP.  This run would train nothing and "
                         "report H0's null as a result.")

    dev = os.environ.get("H0_DEV", "cuda" if torch.cuda.is_available() else "cpu")
    cuda = dev.startswith("cuda")
    log("== H0 QAT ==  device %s  torch %s" % (dev, torch.__version__))
    if cuda:
        log("   gpu %s  bf16_supported %s" % (torch.cuda.get_device_name(0),
                                              torch.cuda.is_bf16_supported()))

    from transformers import AutoModelForCausalLM
    MODEL_ID = "Qwen/Qwen2.5-1.5B"
    REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION,
        dtype=torch.float16 if cuda else torch.float32,
        attn_implementation="sdpa")            # MANDATORY -- s5b(1); eager is non-finite here
    impl = getattr(model.config, "_attn_implementation", None)
    if impl != "sdpa":
        raise SystemExit("ATTENTION IS %r, NOT sdpa -- STOP.  eager fp16 goes non-finite at "
                         "layers.0.self_attn.o_proj on a T4 (proposal s5b(1))." % impl)
    log("   attn_implementation = %s  (asserted)" % impl)
    model.to(dev)
    for p in model.parameters():
        p.requires_grad_(False)

    n_layers = model.config.num_hidden_layers
    layer_ids = list(range(n_layers))
    n_mod = build(model, a.factors, layer_ids, dev)
    layer_ids = sorted({int(k[1:3]) for k in np.load(a.factors).files if k.startswith("L")})
    train_params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in train_params)
    log("   replaced %d organs, %d trainable params (%.1f%% of donor)"
        % (n_mod, n_train, 100.0 * n_train / 1.543e9))

    # Gradient checkpointing + a fully frozen embedding = checkpointed blocks whose inputs do
    # not require grad, so autograd builds no graph through them and every master grad is None.
    # That failure mode is silent and looks exactly like H0's null; enable_input_require_grads
    # makes the embedding OUTPUT require grad without making the embedding trainable.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    ids = torch.from_numpy(np.load(a.train)["ids"]).long()
    probe = json.load(open(a.probe, encoding="utf-8"))
    pids, tgts = probe["prompt_ids"], probe["target_ids"]
    log("   train stream %s tokens, probe %d prompts x %d"
        % (f"{ids.numel():,}", len(pids), len(tgts[0])))

    opt = torch.optim.AdamW(train_params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05, anneal_strategy="cos")
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)

    tf0 = tf_count(model, pids, tgts, dev)
    log("")
    log("   step 0 teacher-forced (fp16, GPU, WATCHABLE NOT THE GATE): %d/%d"
        % (tf0, len(pids) * len(tgts[0])))
    log("   E22 QO512-TB measured 28/160 on CPU fp32; H0 gate is >= 48 re-measured there")
    log("")

    # G-H0e: the planted control for the TRAINER.  Snapshot three masters; after the first
    # step they must have moved.  An H0 null is only readable if the instrument can move.
    lo, hi = layer_ids[0], layer_ids[-1]
    watch = [("L%02d.q_proj.A" % lo, model.model.layers[lo].self_attn.q_proj.A),
             ("L%02d.o_proj.B" % lo, model.model.layers[lo].self_attn.o_proj.B),
             ("L%02d.q_proj.s" % hi, model.model.layers[hi].self_attn.q_proj.s)]
    snap = [(k, v.detach().clone()) for k, v in watch]

    gh0e = None
    hist = [{"step": 0, "tf_fp16_gpu": tf0, "loss": None, "seconds": 0.0}]
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    nonfinite = 0
    declined = 0
    run_loss, nb = 0.0, 0
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            sel = rng.integers(0, ids.shape[0], size=a.bs)
            batch = ids[torch.from_numpy(sel)].to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                out = model(batch, labels=batch)
                loss = out.loss / a.accum
            if not torch.isfinite(loss):
                nonfinite += 1
                continue
            scaler.scale(loss).backward()
            run_loss += float(loss.detach()) * a.accum
            nb += 1
        scaler.unscale_(opt)
        # after unscale_ this is the TRUE fp32 gradient norm, so an inf here NAMES the reason the
        # scaler is about to decline the step instead of leaving it to be inferred from a warning
        gnorm = float(torch.nn.utils.clip_grad_norm_(train_params, 1.0))
        before = applied_steps(opt)
        scale_used = scaler.get_scale()
        scaler.step(opt)                  # DECLINES the update if unscale_ found an inf
        scaler.update()
        sched.step()                      # OneCycleLR counts ITERATIONS, so it advances on a
                                          # declined step too -- that, and nothing else, is what
                                          # raises PyTorch's "lr_scheduler.step() before
                                          # optimizer.step()" warning during warm-up
        applied = applied_steps(opt) > before

        if gh0e is None:
            if applied:
                moved = {k: float((v.detach() - s0).abs().max())
                         for (k, v), (_, s0) in zip(watch, snap)}
                log("   G-H0e read at step %d, the first update the optimizer APPLIED "
                    "(scale %.0f, grad-norm %.3e, %d declined before it)"
                    % (step, scale_used, gnorm, declined))
                log("   G-H0e masters moved by: %s"
                    % "  ".join("%s %.3e" % (k, m) for k, m in moved.items()))
                if not all(m > 0 for m in moved.values()):
                    raise SystemExit("G-H0e FAILS: a master did not move after an update the "
                                     "optimizer APPLIED.  This run would report H0's null as a "
                                     "result.  STOP.")
                gh0e = {"moved": moved, "first_applied_step": step, "declined_before": declined,
                        "scale_at_first_applied": scale_used,
                        "grad_norm_at_first_applied": gnorm}
            else:
                declined += 1
                log("   step %d DECLINED by the GradScaler (scale %.0f -> %.0f, grad-norm %s) "
                    "-- G-H0e waits, %d of %d"
                    % (step, scale_used, scaler.get_scale(),
                       "inf/nan" if not math.isfinite(gnorm) else "%.3e" % gnorm,
                       declined, G_H0E_WINDOW))
                if declined >= G_H0E_WINDOW:
                    raise SystemExit(
                        "G-H0e CANNOT BE READ: the GradScaler declined all %d of the first "
                        "steps, so no update was ever applied and the masters could not have "
                        "moved.  That is a NUMERICAL-SCALE failure of the TRAINER and it is NOT "
                        "H0's null -- do not report it as one.  STOP." % declined)

        el = time.time() - t0
        if step % a.every == 0 or step == a.steps:
            tf = tf_count(model, pids, tgts, dev)
            ml = run_loss / max(1, nb)
            log("  step %5d/%d  loss %.4f  tf(fp16,gpu) %3d/%d  %5.0fs  lr %.2e  nonfinite %d"
                % (step, a.steps, ml, tf, len(pids) * len(tgts[0]), el,
                   sched.get_last_lr()[0], nonfinite))
            hist.append({"step": step, "tf_fp16_gpu": tf, "loss": ml, "seconds": el})
            run_loss, nb = 0.0, 0
            save(model, layer_ids, a, hist, tf0, nonfinite, el, done=False, gh0e=gh0e,
                 declined=declined)
        if el > a.max_hours * 3600:
            log("  TIME CAP %.1f h reached at step %d -- stopping cleanly" % (a.max_hours, step))
            break

    el = time.time() - t0
    save(model, layer_ids, a, hist, tf0, nonfinite, el, done=True, gh0e=gh0e,
         declined=declined)
    log("")
    log("  wrote %s" % a.out)
    log("  RUN COMPLETE.  The gate is NOT decided here: bring %s back and run h0_eval.py on CPU."
        % os.path.basename(a.out))
    return 0


def save(model, layer_ids, a, hist, tf0, nonfinite, el, done, gh0e=None, declined=0):
    store = {}
    for li in layer_ids:
        attn = model.model.layers[li].self_attn
        for nm in ORGANS:
            m = getattr(attn, nm)
            p = "L%02d.%s" % (li, nm)
            store[p + ".A"] = m.A.detach().float().cpu().numpy()
            store[p + ".s"] = m.s.detach().float().cpu().numpy()
            store[p + ".B"] = m.B.detach().float().cpu().numpy()
            store[p + ".rms_in"] = m.rms_in.detach().float().cpu().numpy()
            store[p + ".rms_A"] = m.rms_A.detach().float().cpu().numpy()
    np.savez(a.out, **store)
    json.dump({"plan": "decisions/T4_HEALING_PROPOSAL.md s3 (H0)",
               "complete": done, "steps_requested": a.steps, "bs": a.bs, "accum": a.accum,
               "lr": a.lr, "seed": a.seed, "resumed_from": a.factors,
               "adam_state_restarted": True,
               "seconds": el, "nonfinite_microbatches": nonfinite,
               "tf_fp16_gpu_step0": tf0, "history": hist, "G_H0e": gh0e,
               "scaler_declined_before_first_applied": declined,
               "gate": "NOT decided here -- h0_eval.py on CPU fp32, tf >= 48",
               "e22_anchor_tf": 28, "e21_fp32_ceiling_tf": 144},
              open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
