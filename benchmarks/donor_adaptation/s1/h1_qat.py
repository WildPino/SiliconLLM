#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1 (T4) -- the carve TRAINED rather than APPLIED.

Brief: docs/research/donor_adaptation/briefs/BRIEF_H1_THE_CARVE_TRAINED_NOT_APPLIED.md,
pushed BEFORE this file existed.  Budget: two T4 sessions of 2.8 h (~5.6 GPU-h).

THE QUESTION.  E37 measured the cost of APPLYING the ternary+carved FFN to a trained donor;
H0 measured that a ternary format can be TRAINED INTO.  Nobody has put those two facts in one
run.  H1 does.

WHAT IS BEING HEALED -- MEASURED ON THE 8 LAYERS, NOT QUOTED FROM 28.  The "+3.22 BPB
combined" this docstring used to claim is RETIRED (addendum C): it was E37's ALL-28-LAYER
figure, and H1 does not build that object.  On the 8 layers a 16 GB T4 affords, measured on
the frozen 51,870-byte slice by h1_applied.py:

    h0-run3     0.810022     H1's start line -- q/o trained, FFN untouched
    ternary-8L  0.947851     + ternary FFN, k = E, no carve        (+0.137829)
    applied-8L  1.096636     + the carve at k=16, E37's router     (+0.148785)

So the whole prize is 0.286614 BPB, of which the carve is 0.148785 -- about 30 sigma against
this programme's seed constant (0.005), so a small claim but not a weak instrument.  E37's
28-layer numbers (3.475707, 3.986801) are NOT the comparison and are not thresholds here; the
damage is plainly not additive in layers, and nothing H1 measures may be multiplied up to 28
layers or to 10 B.

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

THE GATE, AND THE ONE THAT WAS REPLACED BEFORE ANY GPU HOUR WAS SPENT.  The gate in force is
Mixtral's renormalised top-k, rescaled by k so the gates AVERAGE 1 instead of summing to 1:
g_e = k * p_e / sum_{selected} p_j on the selected groups and 0 elsewhere, p = softmax over all
E.  At init the router is zeros, so p is uniform, so every selected g_e is EXACTLY 1.0 and
`k = E` reproduces the uncarved ternary FFN BIT-FOR-BIT -- that is G-H1a, a WIRING check run at
init.  The forward genuinely DEPENDS on the router, which is what makes the router trainable.

The FIRST version was g = sel * (1 + E*(p - p.detach())), forward-identical to a hard mask at
EVERY step rather than only at init.  ITS PLANTED CONTROL DID NOT FIRE: pinning the forward
value to 1 makes the loss completely insensitive to the router, and on a known-positive task
recall went 0.2754 -> 0.2568 against a chance of 0.2500 -- it got WORSE.  Addendum A records it.
group_mask's own docstring carries the detail; this paragraph exists so the header cannot drift
back to describing a gate the code does not have.

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
LN2 = 0.6931471805599453
# H0's G-H0e window: how many opening steps the GradScaler may decline before the
# planted control is declared UNREADABLE.  A scale failure is not H1's null.
G_H1B_WINDOW = 40


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
        self._static = None        # [F] mask when the STATIC control of G-H1e is forced on
        self._aux = None           # last forward's load-balancing term, read by the trainer
        self._occ = None           # last forward's per-group occupancy, DIAGNOSTIC
        self._mass = None          # [E] accumulator for STATIC's calibration, when armed

    def invalidate(self):
        """Drop the quantisation cache.  MUST be called after ANY write through `.data`.

        `_quant` keys its cache on the parameters' autograd `_version`, and a `.data.copy_()`
        deliberately does NOT bump that -- bypassing version tracking is what `.data` is for.
        So a module that has already run a forward under no_grad will keep serving the
        weights it was just told to replace, silently and with no error anywhere.

        Found by h1_selftest T7, which is the reason that check exists: the failure is
        invisible in h1_qat's own flow (resume happens before the first forward) and would
        have surfaced for the first time in a SECOND T4 session, as a continuation that
        trained the donor again while reporting that it had resumed.
        """
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
        if self._static is not None:
            # G-H1e's control: a FIXED set of k groups, gates exactly 1, no router in the path.
            self._aux, self._occ = None, None
            return self._static.to(x.dtype if x.is_floating_point() else torch.float32)
        xr = x.float()
        if getattr(self, "route_norm", False):
            # OFF BY DEFAULT, AND THE HYPOTHESIS THAT MOTIVATED IT DID NOT HOLD.  The idea:
            # scores = router . x scales with ||x||, so the softmax temperature is set by the
            # activation magnitude, which differs per layer; routing on x/rms(x) would make the
            # router scale-invariant.  MEASURED on the toy, 12 lr x steps settings per cell,
            # counting how often the router beats STATIC out-of-sample:
            #     activation scale   x1.0     x3.0     x10.0
            #     route_norm off     11/12     2/12     7/12
            #     route_norm on       8/12     8/12     6/12
            # It rescues the x3.0 cell and COSTS the other two; 20/36 against 22/36 overall is
            # not a fix.  And the collapse is not monotone in scale at all -- x10.0 off reads
            # better than x3.0 off -- which is what actually kills the scale story.  Kept, off,
            # so a successor does not spend the same afternoon rediscovering it.
            xr = xr / xr.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        scores = F.linear(xr, self.router)                     # [.., E]
        p = torch.softmax(scores, dim=-1)
        if self.k >= self.E:
            sel = torch.ones_like(p)
        else:
            idx = torch.topk(scores, self.k, dim=-1).indices
            sel = torch.zeros_like(p).scatter_(-1, idx, 1.0)
        if getattr(self, "hard_gate", False):
            # E23/E37's APPLIED carve: hard selection, gates EXACTLY 1, no gate weighting.
            # This is what `applied-8L` must use to be a matched control -- the engine's carve
            # masks and does not reweight.  The router gets no gradient in this mode, which is
            # correct: applied-8L is not trained.
            self._aux, self._occ = None, sel.reshape(-1, self.E).mean(dim=0).detach()
            return sel @ self.onehot.t()
        ps = p * sel
        g = self.k * ps / ps.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        # Switch's load-balancing term, generalised to top-k and normalised so a UNIFORM router
        # reads EXACTLY 1.0: f_e = fraction of tokens selecting e (sums to k), P_e = mean prob
        # (sums to 1), aux = (E/k) * sum_e f_e * P_e.  Addendum A registered a separate router
        # LR and this coefficient as the two things the CPU smoke on the real donor must fix.
        f = sel.reshape(-1, self.E).mean(dim=0)
        P = p.reshape(-1, self.E).mean(dim=0)
        self._aux = (float(self.E) / float(self.k)) * (f * P).sum()
        self._occ = f.detach()
        return g @ self.onehot.t()                             # [.., E] -> [.., F]

    @torch.no_grad()
    def set_static(self, idx):
        """Force the STATIC control of `G-H1e` (addendum B): the same k groups for EVERY token.

        `idx` is a [k] long tensor of group ids, or None to hand the selection back to the
        router.  Gates are exactly 1 inside the chosen groups, so STATIC at `k = E` is also
        bit-identical to the uncarved forward and `G-H1a` still holds under it.
        """
        if idx is None:
            self._static = None
            return
        m = torch.zeros(self.E, device=self.onehot.device, dtype=self.onehot.dtype)
        m[idx.to(self.onehot.device).long()] = 1.0
        self._static = m @ self.onehot.t()                     # [E] -> [F]

    def forward(self, x):
        with torch.autocast(device_type=("cuda" if x.is_cuda else "cpu"), enabled=False):
            gq, uq, dq = self._quant()
            m = self.group_mask(x)
        h = F.silu(F.linear(x, gq.to(x.dtype))) * F.linear(x, uq.to(x.dtype))
        if self._mass is not None:                 # armed only by arm_mass(), for STATIC
            with torch.no_grad():
                flat = h.detach().abs().float().reshape(-1, h.shape[-1])
                self._mass += flat.sum(0) @ self.onehot
        h = h * m.to(h.dtype)
        return F.linear(h, dq.to(x.dtype))

    def arm_mass(self, on=True):
        """Start/stop accumulating UNMASKED per-group activation mass, for STATIC's calibration.

        The mass is taken BEFORE the mask, which is the point: STATIC picks its k groups once
        by global activation mass over the calibration stream, exactly E23's `V52-STATIC`.
        """
        self._mass = torch.zeros(self.E, device=self.onehot.device) if on else None

    @torch.no_grad()
    def static_from_mass(self):
        """Top-k groups by the accumulated mass -> the STATIC index set.  Does not install it."""
        if self._mass is None:
            raise RuntimeError("arm_mass() was never called -- no calibration mass to rank")
        return torch.topk(self._mass, self.k).indices.clone()


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
    if n == 0:
        raise SystemExit(
            "build_qo installed ZERO organs from %s.  Its keys carry no '.A', so this is not "
            "an H0 factor bundle -- most likely it is an H1 OUTPUT bundle passed to --factors. "
            "The FFN resume path is --resume; --factors is the q/o base and must be H0's. "
            "Continuing would train H1's FFN on the RAW donor's attention while the log said "
            "otherwise.  STOP." % bundle)
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


def ffn_mods(model, layers):
    """The carved FFNs, in layer order.  One place that knows where they live."""
    return [(li, model.model.layers[li].mlp) for li in layers]


def resume_ffn(mods, bundle, labels_npz, device):
    """Continue a previous H1 session: load its FFN masters AND its router.

    The brief s7 says the second session is not optional, so this path is load-bearing and not
    a convenience.  Everything is asserted rather than assumed -- in particular the bundle's
    own `labels`, because a run resumed onto a DIFFERENT partition would train happily, report
    a BPB, and be incomparable with the applied-8L control the gate argues with.

    Adam moments are deliberately NOT restored: they are not in the bundle, the optimizer is
    rebuilt, and save() records adam_state_restarted so the discontinuity is on the record.
    """
    z = np.load(bundle)
    lz = np.load(labels_npz)
    n = 0
    for li, m in mods:
        p = "L%02d" % li
        need = [p + s for s in (".gate", ".up", ".down", ".router", ".labels")]
        miss = [q for q in need if q not in z.files]
        if miss:
            raise SystemExit("--resume bundle %s is missing %s -- STOP" % (bundle, miss))
        lab_b = torch.from_numpy(z[p + ".labels"]).long()
        lab_f = torch.from_numpy(lz["c%d" % li]).long()
        if not torch.equal(lab_b, lab_f):
            raise SystemExit(
                "layer %d: the resumed bundle's partition DIFFERS from %s.  The continuation "
                "would be a different experiment from the one it claims to continue.  STOP."
                % (li, labels_npz))
        for nm in ("gate", "up", "down", "router", "rms_in", "rms_h"):
            if p + "." + nm not in z.files:
                continue
            src = torch.from_numpy(z[p + "." + nm]).float()
            dst = getattr(m, nm)
            if tuple(src.shape) != tuple(dst.shape):
                raise SystemExit("layer %d %s is %s, module wants %s -- STOP"
                                 % (li, nm, tuple(src.shape), tuple(dst.shape)))
            dst.data.copy_(src.to(dst.dtype).to(device))
        m.invalidate()
        n += 1
    return n


@torch.no_grad()
def heldout_nats(model, ids, dev, cuda, bs=1):
    """Sum of token nats and the token count on a held-out stream.  fp16 on GPU is fine here:
    this is a PROGRESS metric, and the GATE is re-measured on CPU fp32 by h1_eval.py."""
    tot, n = 0.0, 0
    for i in range(0, ids.shape[0], bs):
        ch = ids[i:i + bs].to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
            lg = model(ch).logits
        lp = F.log_softmax(lg[:, :-1].float(), dim=-1)
        tot += float(-lp.gather(-1, ch[:, 1:].unsqueeze(-1)).squeeze(-1).double().sum())
        n += int(ch[:, 1:].numel())
    return tot, n


@torch.no_grad()
def g_h1e(model, mods, ids_cal, ids_ev, dev, cuda):
    """G-H1e (addendum B) -- ORDINAL, no tolerance: on HELD-OUT tokens the jointly trained
    router must be strictly better than STATIC.

    STATIC = the same model with the top-k groups chosen ONCE by global activation mass over
    the calibration stream and used for EVERY token.  That is E23's `V52-STATIC` comparison,
    and it prices the only thing a router can sell -- the per-token decision.  If the router
    cannot beat a fixed selection it is not earning its 5.1% of the charged weights.

    Returns (router_nats_per_token, static_nats_per_token, fires).
    """
    for _, m in mods:
        m.arm_mass(True)
    heldout_nats(model, ids_cal, dev, cuda)             # calibration pass, mass only
    picks = {}
    for li, m in mods:
        picks[li] = m.static_from_mass()
        m.arm_mass(False)

    r_tot, r_n = heldout_nats(model, ids_ev, dev, cuda)   # router
    for li, m in mods:
        m.set_static(picks[li])
    try:
        s_tot, s_n = heldout_nats(model, ids_ev, dev, cuda)   # STATIC
    finally:
        for _, m in mods:
            m.set_static(None)
    r, s = r_tot / max(1, r_n), s_tot / max(1, s_n)
    return r, s, bool(r < s), {li: picks[li].tolist() for li in picks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", required=True,
                    help="H0 run 3's bundle -- the trained q/o organs, INSTALLED AND FROZEN")
    ap.add_argument("--labels", required=True, help="density/results/d0c_labels/labels_E256.npz")
    ap.add_argument("--stats", default=None, help="per-organ activation RMS; ones if absent")
    ap.add_argument("--train", required=True)
    ap.add_argument("--heldout", required=True, help="npz with 'ids' -- the PROGRESS stream")
    ap.add_argument("--calib", default=None, help="npz with 'ids' for STATIC; --heldout if absent")
    ap.add_argument("--bpt", type=float, default=4.22945205479452,
                    help="bytes per token of the held-out stream; the frozen slice's own value")
    ap.add_argument("--out", default="h1_trained.npz")
    ap.add_argument("--layers", default=",".join(str(x) for x in H1_LAYERS))
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--groups", type=int, default=E_GROUPS)
    ap.add_argument("--steps", type=int, default=int(os.environ.get("H1_STEPS", "4000")))
    ap.add_argument("--bs", type=int, default=int(os.environ.get("H1_BS", "2")))
    ap.add_argument("--accum", type=int, default=int(os.environ.get("H1_ACCUM", "8")))
    ap.add_argument("--lr", type=float, default=float(os.environ.get("H1_LR", "2e-4")),
                    help="the EXPERTS' lr (H0's value on the same shape)")
    ap.add_argument("--router-lr", type=float, default=float(os.environ.get("H1_RLR", "0")),
                    help="the ROUTER's own lr -- addendum A consequence 1.  0 = NOT SET, and "
                         "the run refuses to start: the CPU smoke on the real donor fixes it")
    ap.add_argument("--aux", type=float, default=float(os.environ.get("H1_AUX", "-1")),
                    help="load-balancing coefficient -- same rule, -1 = NOT SET")
    ap.add_argument("--every", type=int, default=int(os.environ.get("H1_EVERY", "250")))
    ap.add_argument("--max-hours", type=float, default=2.8)
    # SESSION 2 RESUMES HERE, NOT THROUGH --factors.  H0 could reuse --factors as its resume
    # path because its trainer rebuilt the same organs it wrote; H1 cannot -- an H1 bundle
    # carries FFN keys (L**.gate/up/down/router) and no '.A', so build_qo would install NOTHING
    # from it and the run would train on the RAW donor's attention.  build_qo now refuses that
    # outright, and this is the flag that actually continues a run.
    ap.add_argument("--resume", default=None,
                    help="a previous H1 bundle -- continues its FFN masters AND router")
    # Batches come from a FIXED rng, so a continuation MUST pass a different seed or it
    # retrains the same draws.  Adam moments are NOT checkpointed.
    ap.add_argument("--seed", type=int, default=1717)
    ap.add_argument("--dev", default=os.environ.get("H1_DEV", None))
    a = ap.parse_args()

    if a.router_lr <= 0 or a.aux < 0:
        raise SystemExit(
            "--router-lr and --aux are NOT SET.  Addendum A registered both as fixed by a CPU "
            "smoke on the REAL donor before any GPU hour is spent, because the jointly trained "
            "router DIVERGED at every obvious setting on the toy.  Refusing to guess them on a "
            "T4.  STOP.")

    # density/common.py calls torch.set_grad_enabled(False) at import; this file imports nothing
    # from the repo, and asserts anyway -- an inference default here fabricates H1's null.
    torch.set_grad_enabled(True)
    if not torch.is_grad_enabled():
        raise SystemExit("AUTOGRAD IS DISABLED -- STOP.  This run would train nothing and "
                         "report H1's null as a result.")

    dev = a.dev or ("cuda" if torch.cuda.is_available() else "cpu")
    cuda = dev.startswith("cuda")
    layers = [int(x) for x in a.layers.split(",") if x.strip() != ""]
    log("== H1 QAT ==  device %s  torch %s" % (dev, torch.__version__))
    log("   layers %s   k %d of E %d  (%.2f%% activation)"
        % (layers, a.k, a.groups, 100.0 * a.k / a.groups))
    if cuda:
        log("   gpu %s" % torch.cuda.get_device_name(0))

    from transformers import AutoModelForCausalLM
    MODEL_ID = "Qwen/Qwen2.5-1.5B"
    REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION,
        dtype=torch.float16 if cuda else torch.float32,
        attn_implementation="sdpa")        # MANDATORY -- eager fp16 is non-finite on a T4
    impl = getattr(model.config, "_attn_implementation", None)
    if impl != "sdpa":
        raise SystemExit("ATTENTION IS %r, NOT sdpa -- STOP (H0 s5b(1))." % impl)
    model.to(dev)
    for p in model.parameters():
        p.requires_grad_(False)

    n_qo, qo_layers = build_qo(model, a.factors, dev)
    for p in model.parameters():
        p.requires_grad_(False)            # q/o are RESUMED and FROZEN -- brief s3
    n_ffn = build_ffn(model, a.labels, a.stats, layers, a.k, a.groups, dev)
    mods = ffn_mods(model, layers)
    if a.resume:
        n_res = resume_ffn(mods, a.resume, a.labels, dev)
        log("   RESUMED the FFN masters and routers of %d layers from %s" % (n_res, a.resume))

    expert_params, router_params = [], []
    for _, m in mods:
        expert_params += [m.gate, m.up, m.down]
        router_params.append(m.router)
    for p in expert_params + router_params:
        p.requires_grad_(True)
    n_exp = sum(p.numel() for p in expert_params)
    n_rt = sum(p.numel() for p in router_params)
    log("   q/o installed %d organs over layers %s -- FROZEN" % (n_qo, qo_layers))
    log("   FFN carved on %d layers: %s masters + %s router = %.2f GB of AdamW state"
        % (n_ffn, f"{n_exp:,}", f"{n_rt:,}", 16.0 * (n_exp + n_rt) / 2**30))

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()     # frozen embedding + checkpointing = no graph at all
    model.train()

    ids = torch.from_numpy(np.load(a.train)["ids"]).long()
    ev = torch.from_numpy(np.load(a.heldout)["ids"]).long()
    cal = torch.from_numpy(np.load(a.calib)["ids"]).long() if a.calib else ev
    log("   train %s tokens, heldout %s tokens, calib %s tokens"
        % (f"{ids.numel():,}", f"{ev.numel():,}", f"{cal.numel():,}"))

    # Addendum A consequence 1: the router is its OWN param group with its OWN lr.
    opt = torch.optim.AdamW([{"params": expert_params, "lr": a.lr},
                             {"params": router_params, "lr": a.router_lr}],
                            weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[a.lr, a.router_lr], total_steps=a.steps, pct_start=0.05,
        anneal_strategy="cos")
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)

    # ---- G-H1a, run BEFORE a single step: k = E must be bit-identical to uncarved ---------
    probe_x = torch.randn(2, 8, model.config.hidden_size, device=dev,
                          dtype=torch.float16 if cuda else torch.float32)
    gates = {}
    li0, m0 = mods[0]
    same, dmax = g_h1a(m0, probe_x)
    live, ldmax = carve_is_live(m0, probe_x)
    log("   G-H1a  k=E bit-identical to uncarved : %s  (max |diff| %.3e)"
        % ("FIRES" if same else "FAILS", dmax))
    log("   carve actually masks at k=%d        : %s  (max |diff| %.3e)"
        % (a.k, "FIRES" if live else "FAILS", ldmax))
    if not same:
        raise SystemExit("G-H1a FAILS -- the mask is not wired to what the measurement claims. "
                         "H1 has no result and the hours are not spent.  STOP.")
    if not live:
        raise SystemExit("The carve does not mask at the real k.  A mask that is wired right "
                         "but never masks passes G-H1a and measures NOTHING.  STOP.")
    gates["G_H1a"] = {"bit_identical": bool(same), "max_abs_diff": dmax}
    gates["carve_is_live"] = {"differs": bool(live), "max_abs_diff": ldmax}

    n0, t0n = heldout_nats(model, ev, dev, cuda)
    bpb0 = n0 / (LN2 * a.bpt * t0n)
    log("")
    log("   step 0 held-out BPB (fp16, GPU, PROGRESS not the gate): %.6f" % bpb0)
    log("   the GATE is trained BPB < applied-8L, re-measured on CPU fp32 by h1_eval.py")
    log("")

    # ---- G-H1b / G-H1c: the first update the optimizer APPLIED, and the masters moved ------
    watch = [("L%02d.gate" % mods[0][0], mods[0][1].gate),
             ("L%02d.down" % mods[-1][0], mods[-1][1].down),
             ("L%02d.router" % mods[0][0], mods[0][1].router)]
    snap = [(kk, v.detach().clone()) for kk, v in watch]

    hist = [{"step": 0, "bpb_fp16_gpu": bpb0, "loss": None, "seconds": 0.0}]
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    nonfinite, declined, gh1bc = 0, 0, None
    run_loss, run_aux, nb = 0.0, 0.0, 0
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            sel = rng.integers(0, ids.shape[0], size=a.bs)
            batch = ids[torch.from_numpy(sel)].to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                out = model(batch, labels=batch)
                aux = sum(m._aux for _, m in mods if m._aux is not None)
                loss = (out.loss + a.aux * aux) / a.accum
            if not torch.isfinite(loss):
                nonfinite += 1
                continue
            scaler.scale(loss).backward()
            run_loss += float(out.loss.detach())
            run_aux += float(aux.detach()) if torch.is_tensor(aux) else float(aux)
            nb += 1
        scaler.unscale_(opt)
        gnorm = float(torch.nn.utils.clip_grad_norm_(expert_params + router_params, 1.0))
        before = applied_steps(opt)
        scale_used = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        sched.step()
        applied = applied_steps(opt) > before

        if gh1bc is None:
            if applied:
                moved = {kk: float((v.detach() - s0).abs().max())
                         for (kk, v), (_, s0) in zip(watch, snap)}
                log("   G-H1b read at step %d, the first update the optimizer APPLIED "
                    "(scale %.0f, grad-norm %.3e, %d declined before it)"
                    % (step, scale_used, gnorm, declined))
                log("   G-H1c masters moved by: %s"
                    % "  ".join("%s %.3e" % (kk, mv) for kk, mv in moved.items()))
                if not all(mv > 0 for mv in moved.values()):
                    raise SystemExit(
                        "G-H1c FAILS: a master did not move after an update the optimizer "
                        "APPLIED.  This run would report H1's null as a result.  STOP.")
                gh1bc = {"moved": moved, "first_applied_step": step,
                         "declined_before": declined, "scale": scale_used, "grad_norm": gnorm}
                gates["G_H1b"] = {"first_applied_step": step, "declined_before": declined}
                gates["G_H1c"] = {"moved": moved, "fires": True}
            else:
                declined += 1
                log("   step %d DECLINED by the GradScaler (scale %.0f -> %.0f) -- G-H1b waits, "
                    "%d of %d" % (step, scale_used, scaler.get_scale(), declined, G_H1B_WINDOW))
                if declined >= G_H1B_WINDOW:
                    raise SystemExit(
                        "G-H1b CANNOT BE READ: the GradScaler declined all %d of the first "
                        "steps, so no update was ever applied and the masters could not have "
                        "moved.  That is a NUMERICAL-SCALE failure of the TRAINER and it is NOT "
                        "H1's null -- do not report it as one.  STOP." % declined)

        el = time.time() - t0
        if step % a.every == 0 or step == a.steps:
            nt, tn = heldout_nats(model, ev, dev, cuda)
            bpb = nt / (LN2 * a.bpt * tn)
            occ = [float(m._occ.max()) for _, m in mods if m._occ is not None]
            log("  step %5d/%d  loss %.4f  aux %.3f  BPB %.6f  %5.0fs  lr %.2e/%.2e  "
                "occ_max %.3f  nonfinite %d"
                % (step, a.steps, run_loss / max(1, nb), run_aux / max(1, nb), bpb, el,
                   sched.get_last_lr()[0], sched.get_last_lr()[1],
                   max(occ) if occ else float("nan"), nonfinite))
            hist.append({"step": step, "bpb_fp16_gpu": bpb, "loss": run_loss / max(1, nb),
                         "aux": run_aux / max(1, nb), "occ_max": max(occ) if occ else None,
                         "seconds": el})
            run_loss, run_aux, nb = 0.0, 0.0, 0
            save(model, mods, a, layers, hist, bpb0, nonfinite, el, False, gates, declined)
        if el > a.max_hours * 3600:
            log("  TIME CAP %.1f h reached at step %d -- stopping cleanly" % (a.max_hours, step))
            break

    el = time.time() - t0
    model.eval()
    r, s, fires, picks = g_h1e(model, mods, cal, ev, dev, cuda)
    log("")
    log("  G-H1e  router %.6f vs STATIC %.6f nats/token  ->  %s"
        % (r, s, "FIRES" if fires else "FAILS"))
    gates["G_H1e"] = {"router_nats": r, "static_nats": s, "fires": fires,
                      "static_groups": picks}
    save(model, mods, a, layers, hist, bpb0, nonfinite, el, True, gates, declined)
    log("  wrote %s" % a.out)
    log("  THE GATE IS NOT DECIDED HERE.  Bring %s back and run h1_eval.py on CPU fp32: "
        "G-H1 is trained BPB < applied-8L, both on the frozen slice."
        % os.path.basename(a.out))
    return 0


def save(model, mods, a, layers, hist, bpb0, nonfinite, el, done, gates, declined):
    store = {}
    for li, m in mods:
        p = "L%02d" % li
        store[p + ".gate"] = m.gate.detach().float().cpu().numpy()
        store[p + ".up"] = m.up.detach().float().cpu().numpy()
        store[p + ".down"] = m.down.detach().float().cpu().numpy()
        store[p + ".router"] = m.router.detach().float().cpu().numpy()
        store[p + ".rms_in"] = m.rms_in.detach().float().cpu().numpy()
        store[p + ".rms_h"] = m.rms_h.detach().float().cpu().numpy()
        store[p + ".labels"] = m.labels.detach().cpu().numpy()
    np.savez(a.out, **store)
    json.dump({"brief": "briefs/BRIEF_H1_THE_CARVE_TRAINED_NOT_APPLIED.md",
               "complete": done, "layers": layers, "k": a.k, "E": a.groups,
               "steps_requested": a.steps, "bs": a.bs, "accum": a.accum,
               "lr": a.lr, "router_lr": a.router_lr, "aux": a.aux, "seed": a.seed,
               "resumed_qo_from": a.factors, "resumed_ffn_from": a.resume,
               "adam_state_restarted": True,
               "bytes_per_token": a.bpt, "seconds": el,
               "nonfinite_microbatches": nonfinite,
               "scaler_declined_before_first_applied": declined,
               "bpb_fp16_gpu_step0": bpb0, "history": hist, "gates": gates,
               "gate": "NOT decided here -- h1_eval.py on CPU fp32, BPB < applied-8L",
               # addendum C: the bands are the MATCHED 8-layer ladder, measured by
               # h1_applied.py.  E37's 28-layer numbers are kept for reference ONLY and are
               # not thresholds -- using one as a band edge is the defect addendum C repaired.
               "anchors": {"dense_fp32": 0.767595, "chance": 4.069819,
                           "h0_run3": 0.810022, "ternary_8L": 0.947851,
                           "applied_8L": 1.096636,
                           "e37_28L_reference_only": {"ternary_all_groups": 3.475707,
                                                      "carved_k16": 3.986801}},
               "bands": {"CARVE-NOT-TRAINABLE": ">= 1.096636",
                         "TRAINING-HELPS": "[0.947851, 1.096636)",
                         "CARVE-IS-TRAINABLE": "(0.810022, 0.947851)",
                         "CARVE-IS-FREE": "<= 0.810022"}},
              open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
