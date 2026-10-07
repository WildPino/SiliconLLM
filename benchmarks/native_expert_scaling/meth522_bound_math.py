"""Exact integer bitsets: one deterministic incompatibility clique, not a packing."""


def clique(bits, uids, width, H):
    assert len(bits) == len(uids) and len(set(uids)) == len(uids)
    size = [v.bit_count() for v in bits]
    order = sorted(range(len(bits)), key=lambda j: (-size[j], uids[j]))
    picked = []; trace = []; union = 0; total = 0
    for at in order:
        blocker = -1; cardinality = 0; checked = 0
        for other in picked:
            checked += 1; cardinality = (bits[at] | bits[other]).bit_count()
            if cardinality <= width: blocker = uids[other]; break
        accept = blocker == -1
        trace.append((uids[at], size[at], int(accept), checked, blocker, cardinality if not accept else 0))
        if accept: picked.append(at); union |= bits[at]; total += size[at]
    minimum_pair_union = min(((bits[a] | bits[b]).bit_count() for k, a in enumerate(picked) for b in picked[:k]), default=None)
    assert minimum_pair_union is None or minimum_pair_union > width
    return {'ordered_UIDs': [uids[j] for j in order], 'clique_UIDs': [uids[j] for j in picked],
            'clique_support_copy_sum': total, 'clique_support_union': union.bit_count(),
            'lower_bound_incidence_copies': H + total - union.bit_count(),
            'lower_bound_branches': max(len(picked), (H + width - 1) // width),
            'minimum_clique_pair_union': minimum_pair_union}, trace, union


def controls():
    bits = [sum(1 << j for j in at) for at in ((0, 1, 2), (0, 3, 4), (0, 5, 6), (0, 1))]
    record, trace, union = clique(bits, list(range(4)), 3, 8)
    assert record['clique_UIDs'] == [0, 1, 2] and record['clique_support_copy_sum'] == 9
    assert record['clique_support_union'] == 7 and record['lower_bound_incidence_copies'] == 10
    assert record['lower_bound_branches'] == 3 and record['minimum_clique_pair_union'] == 5
    assert trace[-1] == (3, 2, 0, 1, 0, 3) and union == 127
    wide = sum(1 << j for j in (0, 1, 2, 3)); assert wide.bit_count() > 3
    record2, _, _ = clique([wide], [11], 3, 8)
    assert record2['lower_bound_incidence_copies'] == 8
    return {'H8_B3_clique': record, 'trace': trace, 'union_integer': union,
            'support_exceeds_width': {'UID': 11, 'support': 4, 'width': 3},
            'rho_budget9_obstructed_by_exact_lower10': True}
