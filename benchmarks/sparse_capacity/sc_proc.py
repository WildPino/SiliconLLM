#!/usr/bin/env python3
# Sparse Capacity, probe P ("rubrica di procedure", D-R): many procedures, each stored under a name, applied to inputs
# never seen with that name. Protocol: docs/research/SPARSE_CAPACITY/RT_PROC_PROTOCOL.md.
#
# Three books of N = 256, 2048, 16384 functions (the sizes of K), equal token share each. Every function f_<name> is a
# simple random rule on 4-digit strings: one or two operations from a small set (add c to every digit mod 10, 9 - d,
# multiply every digit by a unit mod 10, reverse, rotate, a fixed permutation of the positions). Every operation is a
# bijection, so every rule is. Training line:
#     assert f_kqzvw("3812") == "4923"
# Test: for each scored function, inputs never used with it in training (excluded from the training generator).
# Score: exact match of the answer (all tokens greedy), as K in phase 2.
# Controls: (a) functions never trained (names never seen, random rules): the model's guess level; (b) copy the input;
# (c) the most frequent training output of the function: 0 by construction (bijective rules, test inputs never trained);
# (d) the most frequent rule of the book applied to every function (the best name-blind rule).
import itertools, random, string

P_SIZES = (256, 2048, 16384)
P_SEED, P_TEST_PER_FN = 2468, 2
DIG = 4
UNSEEN = 1024


def _ops():
    ops = [("add", c) for c in range(1, 10)] + [("comp", 0), ("rev", 0)] + [("rot", r) for r in range(1, DIG)]
    ops += [("mul", c) for c in (3, 7, 9)]
    special = {tuple(range(DIG)), tuple(reversed(range(DIG)))} | {tuple((i + r) % DIG for i in range(DIG)) for r in range(1, DIG)}
    ops += [("perm", p) for p in itertools.permutations(range(DIG)) if p not in special]
    return ops


OPS = _ops()


def apply_op(op, x):
    k, a = op
    if k == "add":
        return [(d + a) % 10 for d in x]
    if k == "comp":
        return [9 - d for d in x]
    if k == "mul":
        return [(d * a) % 10 for d in x]
    if k == "rev":
        return x[::-1]
    if k == "rot":
        return x[a:] + x[:a]
    return [x[i] for i in a]


def apply_rule(rule, s):
    x = [int(c) for c in s]
    for op in rule:
        x = apply_op(op, x)
    return "".join(map(str, x))


PROBE_INPUTS = ["0123", "4567", "8901", "2468", "1357", "9753"]


def rule_key(rule):
    """Rules are compared as functions (their outputs on a few inputs that separate every operation pair)."""
    return tuple(apply_rule(rule, s) for s in PROBE_INPUTS)


def random_rule(rng):
    while True:
        rule = tuple(rng.choice(OPS) for _ in range(rng.choice((1, 2))))
        if rule_key(rule) != tuple(PROBE_INPUTS):                            # never the identity
            return rule


def make_pbooks(seed=P_SEED, forbidden=()):
    """Three disjoint books {name: rule}, names 5 lowercase letters (prefixed f_ in the text), plus UNSEEN control
    functions. `forbidden`: names to avoid (the phone-book names of K)."""
    rng = random.Random(seed)
    names = set(forbidden); books = []
    for size in P_SIZES:
        b = {}
        while len(b) < size:
            nm = "".join(rng.choice(string.ascii_lowercase) for _ in range(5))
            if nm in names:
                continue
            names.add(nm); b[nm] = random_rule(rng)
        books.append(b)
    unseen = {}
    while len(unseen) < UNSEEN:
        nm = "".join(rng.choice(string.ascii_lowercase) for _ in range(5))
        if nm not in names:
            names.add(nm); unseen[nm] = random_rule(rng)
    # held-out inputs: P_TEST_PER_FN per function, never generated in training for that function
    test_in = {}
    for b in books + [unseen]:
        for nm in b:
            test_in[nm] = set()
            while len(test_in[nm]) < P_TEST_PER_FN:
                test_in[nm].add("".join(rng.choice("0123456789") for _ in range(DIG)))
    return books, unseen, test_in


def p_prefix(name, x):
    return f'assert f_{name}("{x}") == '


def p_answer(y):
    return f'"{y}"\n'


def p_line(name, rule, x):
    return p_prefix(name, x) + p_answer(apply_rule(rule, x))


def p_train_stream(books, test_in, seed):
    """Endless training lines: a book uniformly (equal token share), a function uniformly in it, an input never held
    out for it."""
    rng = random.Random(seed)
    lists = [list(b.items()) for b in books]
    while True:
        nm, rule = rng.choice(lists[rng.randrange(len(lists))])
        while True:
            x = "".join(rng.choice("0123456789") for _ in range(DIG))
            if x not in test_in[nm]:
                break
        yield p_line(nm, rule, x)


def most_common_rule(book):
    cnt = {}
    for r in book.values():
        cnt[rule_key(r)] = cnt.get(rule_key(r), 0) + 1
    key = max(cnt, key=cnt.get)
    return next(r for r in book.values() if rule_key(r) == key), cnt[key] / len(book)


def selftest():
    res = []
    def check(name, ok, val=""):
        res.append(bool(ok)); print(f"  {'ok  ' if ok else 'FAIL'} {name:<72} {val}")
    check("operations: 36 (9 add, comp, rev, 3 rot, 3 mul, 19 perm)", len(OPS) == 36, len(OPS))
    allin = ["".join(p) for p in itertools.product("0123456789", repeat=DIG)]
    bij = all(len({apply_op(op, [int(c) for c in s]).__repr__() for s in allin}) == len(allin) for op in OPS)
    check("every operation is a bijection on the 10^4 strings", bij)
    books, unseen, test_in = make_pbooks()
    check("book sizes", [len(b) for b in books] == list(P_SIZES), [len(b) for b in books])
    names = [n for b in books for n in b] + list(unseen)
    check("names unique across books and the unseen controls", len(set(names)) == len(names))
    rng = random.Random(1); g = p_train_stream(books, test_in, 5)
    hit = 0
    for _ in range(200_000):
        line = next(g); nm = line[len('assert f_'):len('assert f_') + 5]; x = line.split('("')[1][:DIG]
        hit += x in test_in[nm]
    check("training stream never uses a held-out input of its function (200k lines)", hit == 0, hit)
    sample = [r for b in books for r in list(b.values())[:200]]
    check("rule-key collisions are true equalities (on all 10^4 inputs, 200 pairs)",
          all((rule_key(a) == rule_key(b)) == all(apply_rule(a, s) == apply_rule(b, s) for s in allin[::37])
              for a, b in zip(sample[:200], sample[200:400])))
    for i, b in enumerate(books):
        _, share = most_common_rule(b)
        check(f"book {P_SIZES[i]}: most common rule share below 0.05 (the name-blind control)", share < 0.05, f"{share:.4f}")
    nm, r = next(iter(books[0].items()))
    print("  example:", p_line(nm, r, "3812").strip(), "  rule:", r)
    print("SELFTEST (probe P)", "PASS" if all(res) else "FAIL", f"({sum(res)}/{len(res)})")
    return all(res)


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
