#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1 (T4) -- the carve TRAINED rather than APPLIED.

Brief: docs/research/donor_adaptation/briefs/BRIEF_H1_THE_CARVE_TRAINED_NOT_APPLIED.md,
pushed BEFORE this file existed.  Budget: two T4 sessions of 2.8 h (~5.6 GPU-h).

THE QUESTION.  E37 measured the cost of APPLYING the ternary+carved FFN to a trained donor;
H0 measured that a ternary format can be TRAINED INTO.  Nobody has put those two facts in one
run.  H1 does.

WHAT IS BEING HEALED, AND IT IS NOT MOSTLY THE CARVE.  Off e37_sparsity_cost.json, one
instrument, the frozen 51,870-byte slice:  fp32 dense 0.767595 -> ternary FFN with ALL 256
groups on 3.475707 (+2.708) -> carved to k=16 3.986801 (+0.511).  Ternarisation is the big
half; the carve ladder is nearly flat from k=64 (3.927) to k=1 (3.990).  H1 must heal +3.22
BPB combined, 1.6x what H0 healed, on the organ holding nine tenths of the weights.

WHY ONLY 8 LAYERS, AND WHY THESE 8.  Training the whole 1.5B FFN needs 17.2 GB of AdamW state
(1,156,055,040 masters x 16 B) and DOES NOT FIT a 16 GB T4 -- computed in the brief s2 before
the hours were asked for, not discovered mid-session.  8 of 28 layers is 330,301,440 masters =
4.9 GB.  The 8 are 3,6,9,12,15,18,21,24 -- evenly spaced, REGISTERED IN THE BRIEF BEFORE ANY
NUMBER EXISTED, and deliberately NOT E27's MINRES band (12-18), which is the least active
quarter and therefore the easiest place to carve.  Consequence: E37's all-28-layer anchors are
NOT the comparison, and h1_applied.py measures a matched `applied-8L` control on CPU.

THE ROUTER IS TRAINED WITH THE EXPERTS, AND THAT IS THE POINT.  Every carve result in this
programme is post-hoc MoE: in engine/carve_common.py the "experts" are slices of ONE shared
pretrained FFN and the router has only ever been an oracle, a seeded matrix, or a post-hoc
ridge.  E37/E38/E40 close the POST-HOC carve, not the carve.  This is the first jointly
trained router on this branch.

THE GATE IS FORWARD-IDENTITY, ON PURPOSE.  g_e = 1 + E*(p_e - p_e.detach()) with p = softmax
over all E groups.  The forward value is EXACTLY 1, so a selected group is passed through
unscaled and `k = E` reproduces the uncarved ternary FFN BIT-FOR-BIT -- which is what G-H1a
checks and the only thing that rules out a mis-wired mask or a bad group indexing.  The
gradient is E*dp_e and reaches EVERY group through the softmax denominator, including the ones
top-k did not select, so the router can learn to select something new.  A plain hard mask would
give the router no gradient at all; a plain softmax gate would break G-H1a.

NO PERMUTATION HERE, AND THAT IS NOT A DEVIATION.  carve_common.perm_from_labels permutes the F
axis so each group is a contiguous run -- a LAYOUT decision for the engine's prefetcher, and it
"changes nothing the model computes" (that file's own words).  H1 masks by label directly.

THE TRAPS H0 PAID FOR, INHERITED VERBATIM.  (a) density/common.py calls
torch.set_grad_enabled(False) at import, so this file imports NOTHING from the repo and
re-enables grad explicitly.  (b) The quantizer cache is never served under grad -- serving it
would take the masters out of the graph and fabricate the null (H0's G-H0g).  (c) Gradient
checkpointing with a frozen embedding builds no graph; enable_input_require_grads fixes it.
(d) G-H1b waits for an update the optimizer ACTUALLY APPLIED, not for step == 1, because
GradScaler legitimately refuses the first few.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]      # t2_rules.D_GRID, verbatim
QO_ORGANS = ("q_proj", "o_proj")
FFN_ORGANS = ("gate_proj", "up_proj", "down_proj")
H1_LAYERS = (3, 6, 9, 12, 15, 18, 21, 24)        # registered in the brief s2, before any number
E_GROUPS = 256
K_DEFAULT = 16                                    # 6.25% -- brief s3


def log(m):
    sys.stdout.write(m + "\n")
    sys.stdout.flush()


# --------------------------------------------------------------------------------------------
# the shipped ternary rule, reimplemented here ONLY so the bundle is self-contained on Kaggle
# (same reason h0_qat.py carries its own copy; h1_selftest.py asserts they agree).
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
    """H0's factored attention organ, verbatim -- H1 resumes q/o from h0_trained3.npz."""

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
        key = (self.A._version, self.B._version, self.s._version)
        if self._ck == key and not torch.is_grad_enabled():
            return self._cv
        Bt = ste(self.B, self.rms_in)
        At = ste(self.A, self.rms_A * self.s.detach().abs().clamp_min(1e-8))
        if not torch.is_grad_enabled():
            self._ck, self._cv = key, (At, Bt)
        return At, Bt

    def forward(self, x):
        with torch.autocast(device_type=("cuda" if x.is_cuda else "cpu"), enabled=False):
            At, Bt = self._quant()
        h = F.linear(x, Bt.to(x.dtype))
        h = h * self.s.to(x.dtype)
        y = F.linear(h, At.to(x.dtype))
        if self.bias is not None:
            y = y + self.bias_buf.to(y.dtype)
        return y


class TernaryCarvedFFN(nn.Module):
    """One layer's SwiGLU FFN, ternary under STE, with the E=256 group carve ON and a router
    trained jointly with the experts.

        h  = silu(gate(x)) * up(x)                 [.., F]
        h  = h * mask                              mask is 0 outside the top-k groups, and
                                                   EXACTLY 1 inside them (see the docstring)
        y  = down(h)

    `labels` is the per-layer D0c partition (F,) in [0, E), equal groups of F/E.  Masking by
    label is mathematically identical to the engine's permuted layout.

    TRAINING COST IS THE DENSE COST.  The full FFN is computed and then masked, so this buys
    quality information, not speed.  H1 takes no timing and the brief s9 says so; the engine's
    sparse path is E26's object and is already measured.
    """

    def __init__(self, gate_w, up_w, down_w, labels, rms_in, rms_h, k, E):
        super().__init__()
        self.gate = nn.Parameter(gate_w.float())
        self.up = nn.Parameter(up_w.float())
        self.down = nn.Parameter(down_w.float())
        self.register_buffer("rms_in", rms_in.float())
        self.register_buffer("rms_h", rms_h.float())
        lab = labels.long()
        self.register_buffer("labels", lab)
        # one-hot [F, E] as float: neuron-to-group scatter, built once
        self.register_buffer("onehot", F.one_hot(lab, E).float())
        D = gate_w.shape[1]
        self.router = nn.Parameter(torch.zeros(E, D))          # init 0 -> uniform p, no bias
        self.k, self.E = int(k), int(E)
        self._ck, self._cv = None, None

    def _quant(self):
        key = (self.gate._version, self.up._version, self.down._version)
        if self._ck == key and not torch.is_grad_enabled():
            return self._cv
        gq = ste(self.gate, self.rms_in)
        uq = ste(self.up, self.rms_in)
        dq = ste(self.down, self.rms_h)
        if not torch.is_grad_enabled():
            self._ck, self._cv = key, (gq, uq, dq)
        return gq, uq, dq

    def group_mask(self, x):
        """[.., F] gate: 0 outside the top-k groups, a LEARNED positive weight inside them.

            p   = softmax(router . x)                over all E groups
            g_e = k * p_e / sum_{selected} p_j       on the selected groups, 0 elsewhere

        This is Mixtral's renormalised top-k gate, rescaled by k so the gates AVERAGE 1
        instead of summing to 1.  Two properties are load-bearing:

        1.  AT INITIALISATION the router is zeros, so p is exactly uniform, so every selected
            g_e is exactly 1.0 -- and `k == E` therefore reproduces the uncarved ternary FFN
            BIT-FOR-BIT.  That is G-H1a, and it is a WIRING check run at init.
        2.  The forward genuinely DEPENDS on the router, which is what makes this a real MoE
            and what makes the router trainable.

        THE FIRST VERSION OF THIS WAS `g = sel * (1 + E*(p - p.detach()))`, forward-identical
        to a hard mask at every step so that G-H1a would hold at ANY time, not just at init.
        Its planted control DID NOT FIRE: with the forward value pinned to 1 the loss is
        completely insensitive to the router, the only gradient is a "what if I rescaled this
        group" proxy, and on a known-positive task (reproduce the uncarved output, experts
        frozen) recall went 0.2754 -> 0.2568 against a chance of 0.2500 -- it got WORSE.  The
        gate is bounded on purpose: sum_sel g = k, so a concentrating router cannot blow a
        selected group up by a factor of E.
        """
        scores = F.linear(x.float(), self.router)              # [.., E]
        p = torch.softmax(scores, dim=-1)
        if self.k >= self.E:
            sel = torch.ones_like(p)
        else:
            idx = torch.topk(scores, self.k, dim=-1).indices
            sel = torch.zeros_like(p).scatter_(-1, idx, 1.0)
        ps = p * sel
        g = self.k * ps / ps.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return g @ self.onehot.t()                             # [.., E] -> [.., F]

    def forward(self, x):
        with torch.autocast(device_type=("cuda" if x.is_cuda else "cpu"), enabled=False):
            gq, uq, dq = self._quant()
            m = self.group_mask(x)
        h = F.silu(F.linear(x, gq.to(x.dtype))) * F.linear(x, uq.to(x.dtype))
        h = h * m.to(h.dtype)
        return F.linear(h, dq.to(x.dtype))


def build_qo(model, bundle, device):
    """Install H0's trained q/o organs.  Derived from the bundle, not taken on trust."""
    z = np.load(bundle)
    have = sorted({int(k[1:3]) for k in z.files if k.startswith("L")})
    n = 0
    for li in have:
        attn = model.model.layers[li].self_attn
        for nm in QO_ORGANS:
            p = "L%02d.%s" % (li, nm)
            if p + ".A" not in z.files:
                continue
            old = getattr(attn, nm)
            mod = TernaryLowRank(
                torch.from_numpy(z[p + ".A"]), torch.from_numpy(z[p + ".s"]),
                torch.from_numpy(z[p + ".B"]), torch.from_numpy(z[p + ".rms_in"]),
                torch.from_numpy(z[p + ".rms_A"]),
                None if old.bias is None else old.bias.data.detach().clone())
            setattr(attn, nm, mod.to(device))
            n += 1
    return n, have


def build_ffn(model, labels_npz, stats_npz, layers, k, E, device):
    """Replace the MLP of `layers` with the carved ternary FFN.

    `stats_npz` carries the activation RMS per organ, the same quantity h0_factorize.py
    measured for q/o: the R3 rule is activation-weighted and using ones here would silently
    change the format from the one the engine ships.
    """
    lz = np.load(labels_npz)
    sz = np.load(stats_npz) if stats_npz else None
    n = 0
    for li in layers:
        mlp = model.model.layers[li].mlp
        lab = torch.from_numpy(lz["c%d" % li])
        if sz is not None and ("L%02d.rms_in" % li) in sz.files:
            rin = torch.from_numpy(sz["L%02d.rms_in" % li])
            rh = torch.from_numpy(sz["L%02d.rms_h" % li])
        else:
            rin = torch.ones(mlp.gate_proj.weight.shape[1])
            rh = torch.ones(mlp.down_proj.weight.shape[1])
        mod = TernaryCarvedFFN(
            mlp.gate_proj.weight.data.detach().clone(),
            mlp.up_proj.weight.data.detach().clone(),
            mlp.down_proj.weight.data.detach().clone(),
            lab, rin, rh, k, E)
        model.model.layers[li].mlp = mod.to(device)
        n += 1
    return n


# --------------------------------------------------------------------------------------------
# the planted controls
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def g_h1a(mod, x):
    """G-H1a: with k == E the carved forward is BIT-IDENTICAL to the uncarved ternary forward.

    The ONLY check that rules out a mis-wired mask, a bad label vector or a group-indexing
    error -- the same trick E27 used with L28-PASSTHROUGH.  Zero tolerance, not a bar.
    """
    k0 = mod.k
    try:
        mod.k = mod.E
        got = mod.forward(x)
        gq, uq, dq = mod._quant()
        h = F.silu(F.linear(x, gq.to(x.dtype))) * F.linear(x, uq.to(x.dtype))
        want = F.linear(h, dq.to(x.dtype))
    finally:
        mod.k = k0
    same = torch.equal(got, want)
    return same, (got - want).abs().max().item()


@torch.no_grad()
def carve_is_live(mod, x):
    """Companion to G-H1a: at the REAL k the output must DIFFER.  A mask that is wired right
    but never actually masks would pass G-H1a and measure nothing."""
    k0 = mod.k
    try:
        mod.k = mod.E
        full = mod.forward(x)
        mod.k = k0
        cut = mod.forward(x)
    finally:
        mod.k = k0
    return (not torch.equal(full, cut)), (full - cut).abs().max().item()


@torch.no_grad()
def router_recall(mod, x):
    """DIAGNOSTIC ONLY -- NOT a gate.  Agreement with the top-k groups by activation MASS.

    Addendum A: a jointly trained router is not trying to match mass, and on the toy this sat
    at chance WHILE the router was beating the exhaustive best fixed pair on loss.  E38 also
    retired mass-ranking as a yardstick -- it handed the carve a per-token mass oracle and the
    result still read above chance.  The router's gate is `G-H1e` (addendum B): beat STATIC on
    held-out tokens.  This function stays because the number is worth LOOKING at; nothing may
    be decided on it.
    """
    gq, uq, _ = mod._quant()
    h = F.silu(F.linear(x, gq.to(x.dtype))) * F.linear(x, uq.to(x.dtype))
    mass = (h.abs().float() @ mod.onehot)                      # [.., E] true group mass
    true_idx = torch.topk(mass, mod.k, dim=-1).indices
    scores = F.linear(x.float(), mod.router)
    got_idx = torch.topk(scores, mod.k, dim=-1).indices
    t = torch.zeros_like(mass).scatter_(-1, true_idx, 1.0)
    g = torch.zeros_like(mass).scatter_(-1, got_idx, 1.0)
    recall = (t * g).sum(-1).mean().item() / mod.k
    got_mass = (mass * g).sum(-1)
    all_mass = mass.sum(-1).clamp_min(1e-9)
    return recall, (got_mass / all_mass).mean().item()


def applied_steps(opt):
    """How many updates AdamW has actually applied -- read from its own state, not step count.

    GradScaler legitimately refuses the first few steps while it halves the loss scale; H0's
    G-H0e fired on its own unverified precondition because it read the step counter instead.
    """
    n = 0
    for g in opt.param_groups:
        for p in g["params"]:
            st = opt.state.get(p)
            if st and "step" in st:
                s = st["step"]
                n = max(n, int(s.item() if torch.is_tensor(s) else s))
    return n
