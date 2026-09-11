#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E26 -- the ONE definition of the carve's layout decisions.

Brief: docs/research/donor_adaptation/briefs/BRIEF_E26_WHAT_AN_ACTIVATED_WEIGHT_COSTS.md,
pushed before this file existed.

The exporter and the parity reference must agree on three things or the gate is meaningless:
the neuron PERMUTATION, the ROUTER weights, and the TIE rule in top-k.  They live here so
neither side can re-derive one of them slightly differently -- the same discipline that put
`quantize` in qwen_export and the rules in t2_rules.

WHY A PERMUTATION AT ALL.  D0c's E=256 partition at Qwen2.5-1.5B has all 256 groups at 35
neurons and NOT ONE of them a contiguous index range.  `gate`/`up` are [F, D] so a kept neuron
is a whole row and scattering only costs the prefetcher; `down` is [D, F] so a kept neuron is a
COLUMN -- half a byte inside a 64-byte line -- and a scattered carve saves nothing there at all.
Permuting the F axis makes every group a contiguous run.  It is EXACT: it permutes the rows of
gate/up and the columns of down and changes nothing the model computes.
"""
import numpy as np


def perm_from_labels(lab, E):
    """Group-major permutation of the F axis: perm[t] is the original neuron at new index t.

    Requires equal-sized groups, because the engine indexes group g as
    [g*GSZ, (g+1)*GSZ) with GSZ = F/E and has no per-group offset table.  D0c's E=256
    partition satisfies that exactly (35 neurons, all 256 groups).
    """
    lab = np.asarray(lab).astype(np.int64)
    assert lab.min() >= 0 and lab.max() < E, "labels outside [0, E)"
    cnt = np.bincount(lab, minlength=E)
    assert cnt.min() == cnt.max(), \
        "the engine's carve needs equal group sizes; got min %d max %d" % (cnt.min(), cnt.max())
    assert lab.shape[0] % E == 0 and cnt[0] == lab.shape[0] // E
    # stable so the order WITHIN a group is the original neuron order -- one definition, and a
    # rerun of the exporter cannot silently produce a different artifact.
    return np.argsort(lab, kind="stable").astype(np.int64)


def router_weights(D, E, seed, li):
    """The synthetic router for layer `li`: [E, D] fp32, deterministic in (seed, li).

    E26 prices the carve; it does not evaluate the router.  A trained router is E24's object
    and is not what this measures -- but the router must still be EXECUTED and CHARGED, or the
    measurement flatters the carve.  Generating it from a seed (rather than storing it) is what
    lets the parity reference hold the same matrix without reading the artifact under test.
    """
    rng = np.random.default_rng([int(seed), int(li)])
    return (rng.standard_normal((E, D), dtype=np.float32) / np.sqrt(D)).astype(np.float32)


def topk_lower_ties(scores, k):
    """The engine's selection: k largest, ties to the LOWER index, returned sorted ASCENDING.

    np.argsort(-x, kind="stable") breaks ties by lower index, which is exactly what
    donor_engine.c's `sc[i] > bv` (strict) does in its partial selection.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    if k >= n:
        return np.arange(n, dtype=np.int64)
    return np.sort(np.argsort(-scores, kind="stable")[:k]).astype(np.int64)
