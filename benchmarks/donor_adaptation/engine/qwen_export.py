#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export a Qwen2.5 donor into a flat binary that donor_engine.c can execute.

TWO MODES, and the order matters:

  --quant fp32     every weight fp32.  This exists so the C runtime can be proved CORRECT
                   against PyTorch BEFORE any quantization is involved.  If the first thing
                   we build is the ternary path and it produces garbage, we cannot tell a bug
                   in the C from the cost of the conversion.  fp32 first, always.
  --quant ternary  weights as int8 codes {-1,0,+1} + one fp32 scale per output row, exactly
                   engine.c's format.  The rule is imported from t1_ternarize.ternarize() --
                   ONE definition of the conversion, shared with the T1 probe, never re-derived.

Norms, biases and the embedding table stay fp32 in both modes: engine.c's own convention is
"ternary WEIGHT-CODE arrays only; fp32 tensors, per-row scales and activations excluded"
(engine.c:263).

FORMAT (little-endian, no padding):
    char[8]  "QWENDON1"
    int32    d_model, d_ffn, n_layers, n_heads, n_kv_heads, head_dim, vocab, tied, quant
    float32  rms_eps, rope_theta
    fp32     embed_tokens            [vocab, d_model]
    per layer:
      fp32   input_layernorm.weight  [d_model]
      W      q_proj [q_out, d_model] ; fp32 q_bias [q_out]
      W      k_proj [kv_out, d_model]; fp32 k_bias [kv_out]
      W      v_proj [kv_out, d_model]; fp32 v_bias [kv_out]
      W      o_proj [d_model, q_out]
      fp32   post_attention_layernorm.weight [d_model]
      W      gate_proj [d_ffn, d_model]
      W      up_proj   [d_ffn, d_model]
      W      down_proj [d_model, d_ffn]
    fp32     model.norm.weight       [d_model]
    (lm_head is tied on every Qwen2.5 size we use; if tied==0 a final W follows)

  where W is:  quant==0 -> fp32 [out, in]
               quant==1 -> int8 [out, in] codes, then fp32 [out] scales
               quant==2 -> packed base-3 g=2 [out, in/2] bytes, then fp32 [out] scales
               quant==3 -> TAGGED: int32 kind, then
                             kind 0 (packed)   : exactly quant==2's payload
                             kind 2 (fp32)     : exactly quant==0's payload
                             kind 1 (factored) : int32 rank, then A [out,rank] as kind 0/2,
                                                 then fp32 s [rank],
                                                 then B [rank,in] as kind 0/2

THE TAGGED KIND EXISTS BECAUSE THE CONFIGURATIONS THIS PROGRAMME MEASURES ARE MIXED.  E21, E22
and E23 all left k_proj/v_proj fp32 and cut only q_proj/o_proj, and H0 trains exactly that
object; a single global quant flag cannot express it, so until quant==3 existed every rank
result was a PyTorch number with no runnable artifact behind it.  The factored kind is also the
first time the low-rank form is stored AS a factored form: E21 and E22 computed A.B and
installed the dense product, which prices the cut's quality and never its cost.

Usage:
    python qwen_export.py --model Qwen/Qwen2.5-0.5B --quant fp32    --out qwen05b_fp32.bin
    python qwen_export.py --model Qwen/Qwen2.5-0.5B --quant ternary --out qwen05b_tern.bin
"""
import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "ternary")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "density")))
from t1_ternarize import ternarize  # noqa: E402  -- the single definition of the conversion
import t2_rules as T2               # noqa: E402  -- the alternative rules, same definitions
from t3_rotation import fold_norms  # noqa: E402  -- ONE definition of the fold, shared with T3

MAGIC = b"QWENDON1"


def w_fp32(fh, w, rule="R0", act_rms=None):   # rule/act_rms ignored: fp32 keeps every weight
    # .float() is a no-op under --load-dtype float32 and the widening under bfloat16.
    # bf16 -> fp32 is EXACT (it only pads the mantissa), so the bytes are the same either
    # way; the 0.5 B sha256 control in E7 is what makes that a checked claim.
    fh.write(np.ascontiguousarray(w.float().numpy(), dtype="<f4").tobytes())


def quantize(w, rule="R0", act_rms=None):
    """(codes int8 in {-1,0,+1}, scale fp32 [out]) under the chosen rule.

    Every rule returns the SAME format. The rules themselves are imported from t2_rules so the
    exporter, the T2 probe and the parity gate cannot drift apart -- one definition each.
    """
    w = w.float()                 # the rules are defined on fp32; see w_fp32 above
    if rule == "R0":
        q, a = T2.r0_bitlinear(w)
    elif rule == "R1":
        q, a = T2.r1_twn(w)
    elif rule == "R2":
        q, a = T2.r2_search(w)
    elif rule == "R3":
        assert act_rms is not None, "R3 needs calibration activations"
        q, a = T2.r3_actsearch(w, act_rms)
    else:
        raise ValueError(rule)
    return q.to(torch.int8), a.squeeze(1)


def w_tern(fh, w, rule="R0", act_rms=None):
    """int8 codes + fp32 per-row scales."""
    q, scale = quantize(w, rule, act_rms)
    fh.write(np.ascontiguousarray(q.numpy(), dtype="i1").tobytes())
    # quantize() already returns the scale as a [out] vector. This line used to squeeze it a
    # SECOND time, which raises IndexError on a 1-D tensor -- so --quant ternary has been dead
    # since --rule landed. --quant packed never had the bug, which is why nothing caught it.
    fh.write(np.ascontiguousarray(scale.numpy(), dtype="<f4").tobytes())
    return float((q == 0).float().mean())


def write_packed_pair(fh, q, scale):
    """Write already-quantized (codes, per-row scale) in engine.c's packed layout.

    Split out of w_packed so the FACTORED writer can hand it a pair produced by h0_qat's
    quantizer instead of re-deriving one here -- the factors must be quantized by the same code
    that trained them, or the artifact is not the object that was measured.
    """
    out_f, in_f = q.shape
    assert in_f % 2 == 0, "in_features must be even to pack 2 trits per byte"
    qn = q.numpy().astype(np.int16) + 1                      # {0,1,2}
    packed = (qn[:, 0::2] + 3 * qn[:, 1::2]).astype(np.uint8)
    assert packed.max() <= 8
    fh.write(np.ascontiguousarray(packed).tobytes())
    fh.write(np.ascontiguousarray(scale.numpy(), dtype="<f4").tobytes())
    return float((q == 0).float().mean())


def w_packed(fh, w, rule="R0", act_rms=None):
    """base-3 g=2: TWO trits per byte, 4 bits/weight -- engine.c's own packing.

    Byte value v = (t0+1) + 3*(t1+1) in [0,8], where t0 is the weight for input feature 2j and
    t1 for 2j+1. Lossless with respect to the int8 codes, and exactly half the bytes; on a
    bandwidth-bound path that is a free 2x, which is why engine.c uses it.
    """
    q, scale = quantize(w, rule, act_rms)
    return write_packed_pair(fh, q, scale)


# ---------------------------------------------------------------------- the tagged writers
MK_PACKED, MK_FACTORED, MK_F32 = 0, 1, 2


def w_tag_packed(fh, w, rule="R0", act_rms=None):
    fh.write(struct.pack("<i", MK_PACKED))
    return w_packed(fh, w, rule, act_rms)


def w_tag_f32(fh, w, rule="R0", act_rms=None):
    fh.write(struct.pack("<i", MK_F32))
    w_fp32(fh, w)
    return None


def w_tag_factored(fh, A, s, B, rms_in, rms_A, H0):
    """kind 1: A [out,r] packed, s [r] fp32, B [r,in] packed.

    The two ternarizations are h0_qat's OWN calls, not a re-derivation: A is quantized against
    rms_A * |s| and B against rms_in, exactly as TernaryLowRank._quant does at every training
    step and at every eval position.  If these two lines and that method ever disagree, the
    exported model is not the model the gate was measured on, and the parity gate is what
    catches it.
    """
    import torch
    r = int(s.shape[0])
    assert A.shape[1] == r and B.shape[0] == r
    qA, aA = H0.r3_actsearch(A.float(), rms_A.float() * s.float().abs().clamp_min(1e-8))
    qB, aB = H0.r3_actsearch(B.float(), rms_in.float())
    fh.write(struct.pack("<ii", MK_FACTORED, r))
    # A and B are themselves TAGGED matrices -- the reader calls read_mat recursively with
    # quant=3 and expects a kind tag on each factor, which is what lets one factor be fp32 and
    # the other packed.  Omitting these two tags writes a file that parses as far as the first
    # factor and then dies on a byte that is a CODE being read as a KIND.
    fh.write(struct.pack("<i", MK_PACKED))
    zA = write_packed_pair(fh, qA.to(torch.int8), aA.squeeze(1))
    fh.write(np.ascontiguousarray(s.float().numpy(), dtype="<f4").tobytes())
    fh.write(struct.pack("<i", MK_PACKED))
    zB = write_packed_pair(fh, qB.to(torch.int8), aB.squeeze(1))
    return 0.5 * (zA + zB)


def capture_act_rms(m, tk, calib_seqs, L):
    """Per-input activation RMS for rule R3, captured over the pinned calibration slice.

    ONE definition, imported by both the exporter and E1's runner: a Gate-A check that
    re-derived the calibration itself would be comparing two implementations, not the
    exporter against the rule.

    The calibration corpus half is DISJOINT from the eval half by construction
    (build_calib.py) and the sha256s are asserted different, exactly as D4 and T2 do.
    """
    import common as CD
    ids_cal, _, meta_cal = CD.get_slice(tk, "calib", 32, 512, 42424)
    _, _, meta_ev = CD.get_slice(tk, "heldout", 24, 512, 1234)
    assert meta_cal["corpus_sha256"] != meta_ev["corpus_sha256"], \
        "calib and eval must be different corpus halves"
    ids_cal = ids_cal[: calib_seqs]
    if ids_cal.shape[0] != 32:
        print("  WARNING: --calib-seqs %d != 32. T2's numbers were measured at 32; this "
              "export is a DIFFERENT quantization and its BPB is unmeasured."
              % ids_cal.shape[0], flush=True)
    # T2 applied the rule to the FFN organs ONLY (84 tensors = 28 layers x 3). This exporter
    # applies it to the attention projections as well, which is what a runnable model needs
    # and what BRIEF_T2 s4 explicitly forbids extrapolating to. The organ list is recorded.
    print("  R3: capturing activations over %d calib sequences..." % ids_cal.shape[0],
          flush=True)
    sums, cnts, hooks, act = {}, {}, [], {}

    def mk(key):
        def f(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            sums[key] = (x * x).sum(0) if key not in sums else sums[key] + (x * x).sum(0)
            cnts[key] = cnts.get(key, 0) + x.shape[0]
        return f
    for li, lay in enumerate(m.model.layers):
        for nm, mod in (("q_proj", lay.self_attn.q_proj), ("o_proj", lay.self_attn.o_proj),
                        ("gate_proj", lay.mlp.gate_proj), ("down_proj", lay.mlp.down_proj)):
            hooks.append(mod.register_forward_hook(mk((li, nm))))
    hl = m.lm_head.register_forward_hook(mk(("head", "head")))
    with torch.no_grad():
        for i in range(ids_cal.shape[0]):
            m(ids_cal[i:i + 1])
    for h in hooks:
        h.remove()
    hl.remove()
    for k, v in sums.items():
        act[k] = torch.sqrt(v / cnts[k]).clamp_min(1e-8)
    for li in range(L):                      # organs that share an input
        act[(li, "k_proj")] = act[(li, "q_proj")]
        act[(li, "v_proj")] = act[(li, "q_proj")]
        act[(li, "up_proj")] = act[(li, "gate_proj")]
    print("  R3: captured %d activation vectors" % len(act), flush=True)
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--quant", choices=("fp32", "ternary", "packed", "tagged"), default="fp32")
    ap.add_argument("--factors", default=None,
                    help="npz of fp32 low-rank masters in h0_factorize.py's layout "
                         "(L%%02d.<organ>.{A,s,B,rms_in,rms_A}).  Every organ it holds is "
                         "written as the FACTORED kind; everything else follows --quant. "
                         "Requires --quant tagged and --fold none: the masters were fitted on "
                         "the UNFOLDED donor, and folding a norm gain into q would make the "
                         "exported matrix a different matrix from the one that was measured.")
    ap.add_argument("--fp32-organs", default="",
                    help="comma-separated organs written as the fp32 kind in --quant tagged "
                         "(e.g. k_proj,v_proj).  This is not a convenience: E21/E22/E23 all "
                         "left k/v fp32 and their numbers are the numbers for THAT object.")
    ap.add_argument("--load-dtype", choices=("float32", "bfloat16"), default="float32",
                    help="how the donor is held in RAM. float32 is what every export before "
                         "E7 used. bfloat16 halves the peak (7.6 B: 30 GB -> 15 GB) and is "
                         "the checkpoint's own precision; each tensor is widened to fp32 at "
                         "write time, so the artifact is unchanged. Refuses --fold and --rule "
                         "R3, whose arithmetic would then run in bf16.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rule", choices=("R0", "R1", "R2", "R3"), default="R0",
                    help="ternarization rule. R0 = BitLinear158, the engine's original and T1's. "
                         "R1/R2/R3 are T2's alternatives; ALL of them land in the same format "
                         "(codes in {-1,0,+1} + one fp32 scale per output row), so the runtime "
                         "executes any of them unchanged -- which is the point of T2 and the "
                         "reason a better rule needs no engine change. R3 needs per-input "
                         "activation RMS and therefore a calibration pass (--calib-seqs).")
    ap.add_argument("--calib-seqs", type=int, default=32,
                    help="calibration sequences for --rule R3, from the DISJOINT corpus half. "
                         "The default is 32 because that is T2's registered operating point "
                         "(t2_rules.py N_CAL=32, SEQ_CAL=512, SEED_CAL=42424), inherited from "
                         "D4. Exporting at any other budget produces a DIFFERENT quantization "
                         "from the one T2 measured, so the value used is written to the sidecar "
                         "and a mismatch is warned about on stderr. D4b was written to sweep "
                         "this knob and has never been run: 32 is a registered point, not an "
                         "optimum.")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch thread count. NOT a speed knob: R3's calibration forward pass "
                         "changes its REDUCTION ORDER with the thread count, so act_rms moves in "
                         "its last bits and alpha=num/den moves with it. Measured on 0.5B: 6 vs 6 "
                         "threads gives 0 differing activations, 6 vs 1 gives 102,123 (worst "
                         "1.9e-06), which propagates to <=6 ulp on the stored scales. The CODES "
                         "never moved (0 of 357,826,560), so the quantization is not in doubt -- "
                         "but two exports of the same command produce different sha256 unless "
                         "this matches. 0 leaves torch alone; the value used is recorded.")
    ap.add_argument("--head-ternary", action="store_true",
                    help="UNTIE the output head and store it ternary. The embedding table stays "
                         "fp32 because it is a row LOOKUP (3.5 KB/token) and costs nothing to "
                         "stream; the HEAD is a dense GEMV over the whole vocabulary and is 40.4%% "
                         "of per-token time when left fp32 (measured, donor_engine --profile).")
    ap.add_argument("--fold", choices=("none", "layers", "all"), default="layers",
                    help="fold RMSNorm gains into the linears that read them before quantizing. "
                         "'layers' folds the 2L per-layer gains (input_layernorm -> q/k/v, "
                         "post_attention_layernorm -> gate/up); 'all' additionally folds "
                         "model.norm into lm_head, which UNTIES the head. T3 measured the fold "
                         "at -0.220 BPB on the 1.5B; see BRIEF_E2_RMSNORM_FOLD.md.")
    a = ap.parse_args()

    if a.threads > 0:
        torch.set_num_threads(a.threads)

    from transformers import AutoModelForCausalLM
    # attn_implementation="eager" is NOT cosmetic and NOT a speed choice: it is what
    # common.load_model uses, and therefore what T1/T2/T2b/T3 measured. HF's default here is
    # sdpa, and sdpa vs eager moves R3's calibration act_rms on 142,977 elements (worst rel
    # 8.2e-06) -- which moved 132,844 stored scales by up to 6 ulp while leaving all
    # 357,826,560 CODES identical. Nothing about the quantization was ever in doubt; the
    # ARTIFACT simply was not the one the probes measured. Same class as the S1 fp16 bug:
    # reproduce the CONFIGURATION, not just the model.
    if a.load_dtype != "float32":
        # The fold multiplies a gain into every row and R3 runs forward passes: both would
        # then be bf16 arithmetic, not bf16 STORAGE, and the artifact would differ. Refuse
        # rather than quietly produce a different quantization.
        if a.fold != "none":
            sys.exit("--load-dtype %s needs --fold none: folding in bf16 is bf16 ARITHMETIC"
                     % a.load_dtype)
        if a.rule == "R3":
            sys.exit("--load-dtype %s cannot run R3: its calibration forwards would be bf16"
                     % a.load_dtype)
    m = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision,
                                             dtype=getattr(torch, a.load_dtype),
                                             attn_implementation="eager").eval()
    c = m.config
    D = c.hidden_size
    F = c.intermediate_size
    L = c.num_hidden_layers
    NH = c.num_attention_heads
    NKV = c.num_key_value_heads
    HD = getattr(c, "head_dim", D // NH)
    V = c.vocab_size
    tied = int(bool(getattr(c, "tie_word_embeddings", False)))
    if a.head_ternary:
        tied = 0          # write an explicit head; the embedding table is still written fp32
    quant = {"fp32": 0, "ternary": 1, "packed": 2, "tagged": 3}[a.quant]
    fp32_organs = tuple(x for x in a.fp32_organs.split(",") if x)
    if a.factors and a.quant != "tagged":
        sys.exit("--factors needs --quant tagged: no other layout can carry a factored matrix")
    if a.factors and a.fold != "none":
        sys.exit("--factors needs --fold none: the masters were fitted on the UNFOLDED donor")
    if fp32_organs and a.quant != "tagged":
        sys.exit("--fp32-organs needs --quant tagged")
    fac, H0 = None, None
    if a.factors:
        sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "s1")))
        import h0_qat as H0                                       # noqa: F811
        fac = np.load(a.factors)
        have = sorted({(int(k[1:3]), k.split(".")[1]) for k in fac.files if k.startswith("L")})
        print("  --factors: %d organs, ranks %s"
              % (len(have), sorted({int(fac["L%02d.%s.s" % lo].shape[0]) for lo in have})))

    # The fold has to happen BEFORE the calibration capture, not after: folding a gain into
    # q/k/v changes what those linears SEE, so R3's per-input activation RMS is a different
    # vector under a folded model. Calibrating a folded model on unfolded activations would be
    # a different -- and worse -- experiment than the one E2 pre-registers.
    n_folded, untied_by_fold = 0, False
    if a.fold != "none":
        # A packed/ternary file has ONE quant flag and the engine reads the head with it
        # (donor_engine.c:397). So an untied head in a quantized file is necessarily quantized:
        # "fold everything but keep an fp32 head" is not expressible in this format, and the
        # exporter refuses rather than silently ternarizing a head nobody asked to ternarize.
        if a.fold == "all" and quant != 0 and not a.head_ternary:
            sys.exit("--fold all unties the head, and a quantized file has a single quant flag: "
                     "the engine would read that head with it. Use --fold all --head-ternary, "
                     "or --fold layers, or --quant fp32.")
        n_folded, untied_by_fold = fold_norms(m, fold_final=(a.fold == "all"))
        if untied_by_fold:
            tied = 0
        print("  folded %d RMSNorm gains (--fold %s), untied=%s"
              % (n_folded, a.fold, untied_by_fold))
    W = {0: w_fp32, 1: w_tern, 2: w_packed, 3: w_tag_packed}[quant]

    def emit(fh, li, organ, w, rule, act):
        """One matrix, in whichever kind this organ gets.  The only place that decides."""
        if fac is not None:
            pre = "L%02d.%s" % (li, organ)
            if pre + ".A" in fac.files:
                import torch as _t
                return w_tag_factored(fh, _t.from_numpy(fac[pre + ".A"]),
                                      _t.from_numpy(fac[pre + ".s"]),
                                      _t.from_numpy(fac[pre + ".B"]),
                                      _t.from_numpy(fac[pre + ".rms_in"]),
                                      _t.from_numpy(fac[pre + ".rms_A"]), H0)
        if quant == 3 and organ in fp32_organs:
            return w_tag_f32(fh, w)
        return W(fh, w, rule, act)

    print("exporting %s  D=%d F=%d L=%d heads=%d/%d hd=%d V=%d tied=%d quant=%s rule=%s"
          % (a.model, D, F, L, NH, NKV, HD, V, tied, a.quant, a.rule))

    # R3 needs the per-input activation RMS, so it needs a calibration pass. The calibration
    # corpus half is DISJOINT from the eval half by construction (build_calib.py) and the
    # sha256s are asserted different, exactly as D4 and T2 do.
    act = {}
    if a.rule == "R3":
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
        act = capture_act_rms(m, tk, a.calib_seqs, L)

    zeros = []
    with open(a.out, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<9i", D, F, L, NH, NKV, HD, V, tied, quant))
        fh.write(struct.pack("<2f", float(c.rms_norm_eps), float(c.rope_theta)))
        w_fp32(fh, m.model.embed_tokens.weight.data)
        for li in range(L):
            lay = m.model.layers[li]
            w_fp32(fh, lay.input_layernorm.weight.data)
            for name in ("q_proj", "k_proj", "v_proj"):
                mod = getattr(lay.self_attn, name)
                r = emit(fh, li, name, mod.weight.data, a.rule, act.get((li, name)))
                if r is not None:
                    zeros.append(r)
                assert mod.bias is not None, "%s has no bias -- Qwen2 should" % name
                w_fp32(fh, mod.bias.data)
            r = emit(fh, li, "o_proj", lay.self_attn.o_proj.weight.data, a.rule,
                     act.get((li, "o_proj")))
            if r is not None:
                zeros.append(r)
            assert lay.self_attn.o_proj.bias is None
            w_fp32(fh, lay.post_attention_layernorm.weight.data)
            for name in ("gate_proj", "up_proj", "down_proj"):
                mod = getattr(lay.mlp, name)
                assert mod.bias is None
                r = emit(fh, li, name, mod.weight.data, a.rule, act.get((li, name)))
                if r is not None:
                    zeros.append(r)
            if (li + 1) % 8 == 0 or li == L - 1:
                print("  layer %d/%d  (%.2f GB written)"
                      % (li + 1, L, fh.tell() / 2**30), flush=True)
        w_fp32(fh, m.model.norm.weight.data)
        if not tied:
            hw = m.lm_head.weight.data
            r = W(fh, hw, a.rule, act.get(("head", "head")))
            if r is not None:
                zeros.append(r)
            print("  head written explicitly: %s %s" % (tuple(hw.shape), a.quant))

    size = os.path.getsize(a.out)

    # ---- GATE E25-L: does the file have the size the FORMAT says it should?
    # e1_bpb_through_engine.layout_bytes_tagged walks the layout from the header alone, as a
    # third party would, and knows nothing about the loop above.  The first tagged export this
    # gate would have caught was real: w_tag_factored wrote A's and B's payloads without their
    # own kind tags, and the file parsed until the engine read a weight CODE as a KIND.  The
    # engine's own "consumed exactly N bytes" found it, but only after a 3-minute write.
    if quant == 3:
        import e1_bpb_through_engine as E1

        def spec_of(name, o, i):
            organ = name.split(".")[-1]
            li = int(name.split(".")[0][1:]) if name.startswith("L") else -1
            if fac is not None and ("L%02d.%s.A" % (li, organ)) in fac.files:
                return ("factored", int(fac["L%02d.%s.s" % (li, organ)].shape[0]))
            return "f32" if organ in fp32_organs else "packed"
        want = E1.layout_bytes_tagged(D, F, L, NH, NKV, HD, V, tied, spec_of)
        assert want == size, ("GATE E25-L FAILED: the format says %d bytes, the file is %d "
                              "(%+d)." % (want, size, size - want))
        print("  GATE E25-L: %d bytes, matches E1's independent tagged layout exactly" % size)

    h = hashlib.sha256()
    with open(a.out, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    meta = {"model": a.model, "revision": a.revision, "quant": a.quant,
            "load_dtype": a.load_dtype,
            "d_model": D, "d_ffn": F, "n_layers": L, "n_heads": NH, "n_kv_heads": NKV,
            "head_dim": HD, "vocab": V, "tied": tied,
            "rms_eps": float(c.rms_norm_eps), "rope_theta": float(c.rope_theta),
            "head_ternary": bool(a.head_ternary), "rule": a.rule,
            # part of the artifact's identity: a folded export is a DIFFERENT quantization,
            # because the fold changes what q/k/v/gate/up see and therefore R3's thresholds.
            "fold": a.fold, "n_gains_folded": n_folded, "untied_by_fold": bool(untied_by_fold),
            "bytes": size, "sha256": h.hexdigest(),
            "mean_ternary_zero_fraction": (float(np.mean(zeros)) if zeros else None),
            # The seam between what T2 MEASURED and what this file CONTAINS. Recorded so a
            # BPB gap between the two can be attributed instead of guessed at.
            "calib_seqs": (a.calib_seqs if a.rule == "R3" else None),
            # part of the artifact's IDENTITY, not a performance note: see --threads.
            "torch_threads": (a.threads if a.threads > 0 else torch.get_num_threads()),
            "torch_threads_pinned": bool(a.threads > 0),
            "attn_implementation": m.config._attn_implementation,
            "calib_matches_t2_operating_point": (a.calib_seqs == 32 if a.rule == "R3" else None),
            "rule_applied_to": (["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"]
                                + (["lm_head"] if a.head_ternary else [])),
            "t2_measured_organs": ["gate_proj", "up_proj", "down_proj"],
            "factors": a.factors,
            "factored_organs": (["L%02d.%s" % lo for lo in have] if fac is not None else []),
            "fp32_organs": list(fp32_organs)}
    json.dump(meta, open(a.out + ".json", "w", encoding="utf-8"), indent=1)
    print("wrote %s  (%.2f GB)  sha256 %s" % (a.out, size / 2**30, h.hexdigest()[:16]))
    if zeros:
        print("  mean ternary zero-fraction across %d matrices: %.4f"
              % (len(zeros), float(np.mean(zeros))))


if __name__ == "__main__":
    main()
