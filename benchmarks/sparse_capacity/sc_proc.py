#!/usr/bin/env python3
# Sparse Capacity, probe P ("rubrica di procedure", D-R): many procedures, each stored under a name, applied to inputs
# never seen with that name. Protocol: docs/research/SPARSE_CAPACITY/RT_PROC_PROTOCOL.md.
#
# Three books of N = 64, 512, 4096 named functions, equal token share each, plus an EXPLICIT book: the planted positive,
# where the rule is written in the call (nothing to remember, only to execute). Every function is a simple random rule
# on 4-digit strings: one or two operations from a small set (add c to every digit mod 10, 9 - d, multiply every digit
# by a unit mod 10, reverse, rotate, a fixed permutation of the positions). Every operation is a bijection, so every rule
# is. Digits are written separated by spaces, so that every digit is one token (48-57) whatever its neighbours: the BPE
# merges digit runs irregularly ("3812" -> 3|8|12), which makes per-digit execution nearly impossible. Training lines:
#     assert f_kqzvw("3 8 1 2") == "4 9 2 3"
#     assert rot_1__add_3("3 8 1 2") == "1 4 5 6"        (explicit: operations joined by "__", parameters after "_")
# Test: named functions on inputs never used with them in training; explicit rules on (rule, input) pairs never trained.
# Score: exact match of the answer (all tokens greedy), as K in phase 2.
# Controls: (a) named functions never trained (names never seen, random rules): the model's guess level; (b) copy the
# input; (c) the most frequent training output of the function: 0 by construction (bijective rules, test inputs never
# trained); (d) the most frequent rule of the book applied to every function (the best name-blind rule); for the explicit
# book, (e) the call with another rule's name (scored against the true answer): does the model read the name?
import itertools, random, string

P_SIZES = (64, 512, 4096)
P_SEED, P_TEST_PER_FN = 2468, 4
E_TEST = 2048                                  # explicit book: held-out (rule, input) pairs
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


def rand_digits(rng):
    return "".join(rng.choice("0123456789") for _ in range(DIG))


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
    test_in = {}                                    # held-out inputs, never generated in training for that function
    for b in books + [unseen]:
        for nm in b:
            test_in[nm] = set()
            while len(test_in[nm]) < P_TEST_PER_FN:
                test_in[nm].add(rand_digits(rng))
    return books, unseen, test_in


def spaced(x):
    return " ".join(x)


def p_prefix(name, x):
    return f'assert f_{name}("{spaced(x)}") == '


def p_answer(y):
    return f'"{spaced(y)}"\n'


def p_line(name, rule, x):
    return p_prefix(name, x) + p_answer(apply_rule(rule, x))


def op_name(op):
    k, a = op
    if k in ("comp", "rev"):
        return k
    if k == "perm":
        return "perm_" + "_".join(map(str, a))
    return f"{k}_{a}"


def e_name(rule):
    return "__".join(op_name(op) for op in rule)


def e_prefix(rule, x):
    return f'assert {e_name(rule)}("{spaced(x)}") == '


def p_train_stream(books, test_in, seed):
    """Endless named training lines: a book uniformly (equal token share), a function uniformly in it, an input never
    held out for it."""
    rng = random.Random(seed)
    lists = [list(b.items()) for b in books]
    while True:
        nm, rule = rng.choice(lists[rng.randrange(len(lists))])
        while True:
            x = rand_digits(rng)
            if x not in test_in[nm]:
                break
        yield p_line(nm, rule, x)


def e_test_set(seed=P_SEED + 1, n=E_TEST):
    """Held-out (rule, input) pairs of the explicit book; the explicit training stream never generates them."""
    rng = random.Random(seed); out, keys = [], set()
    while len(out) < n:
        rule = random_rule(rng); x = rand_digits(rng)
        k = (e_name(rule), x)
        if k not in keys:
            keys.add(k); out.append((rule, x))
    return out, keys


def e_train_stream(keys, seed):
    rng = random.Random(seed)
    while True:
        rule = random_rule(rng); x = rand_digits(rng)
        if (e_name(rule), x) in keys:
            continue
        yield e_prefix(rule, x) + p_answer(apply_rule(rule, x))


def most_common_rule(book):
    cnt = {}
    for r in book.values():
        cnt[rule_key(r)] = cnt.get(rule_key(r), 0) + 1
    key = max(cnt, key=cnt.get)
    return next(r for r in book.values() if rule_key(r) == key), cnt[key] / len(book)


def check_tokens(enc):
    """With the phase-2 BPE: every digit of an answer is its own token 48-57 whatever the neighbours, the split
    prefix | answer tokenizes as the whole line, the input digits too, and the explicit names keep every parameter digit
    a token. Returns the number of failures over 3000 random lines of each kind."""
    rng = random.Random(0); bad = 0
    for _ in range(3000):
        x, y = rand_digits(rng), rand_digits(rng)
        rule = random_rule(rng)
        for pre in (p_prefix("abcde", x), e_prefix(rule, x)):
            if enc(pre) + enc(p_answer(y)) != enc(pre + p_answer(y)):
                bad += 1
        if [i for i in enc(p_answer(y)) if 48 <= i <= 57] != [48 + int(c) for c in y]:
            bad += 1
        if [i for i in enc(f'("{spaced(x)}")') if 48 <= i <= 57] != [48 + int(c) for c in x]:
            bad += 1
        want = [c for op in rule for c in (map(str, op[1]) if op[0] == "perm" else [] if op[0] in ("comp", "rev") else [str(op[1])])]
        if [i for i in enc(e_name(rule)) if 48 <= i <= 57] != [48 + int(c) for c in want]:
            bad += 1
    return bad


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
    g = p_train_stream(books, test_in, 5); hit = 0
    for _ in range(200_000):
        line = next(g); nm = line[len('assert f_'):len('assert f_') + 5]; x = line.split('("')[1][:2 * DIG - 1].replace(" ", "")
        hit += x in test_in[nm]
    check("named stream never uses a held-out input of its function (200k lines)", hit == 0, hit)
    et, ek = e_test_set(); eg = e_train_stream(ek, 5); leak = 0
    for _ in range(100_000):
        line = next(eg)
        leak += (line[len("assert "):line.index('("')], line.split('("')[1][:2 * DIG - 1].replace(" ", "")) in ek
    check("explicit stream never generates a held-out (rule, input) pair (100k lines)", leak == 0, leak)
    r0 = (("rot", 1), ("add", 3))
    check("explicit name and rule agree: rot_1__add_3 on 3812 gives 1456 (8123 + 3)",
          e_name(r0) == "rot_1__add_3" and apply_rule(r0, "3812") == "1456", apply_rule(r0, "3812"))
    sample = [r for b in books for r in list(b.values())[:200]]
    check("rule-key collisions are true equalities (200 pairs, 271 inputs each)",
          all((rule_key(a) == rule_key(b)) == all(apply_rule(a, s) == apply_rule(b, s) for s in allin[::37])
              for a, b in zip(sample[:200], sample[200:400])))
    for i, b in enumerate(books):
        _, share = most_common_rule(b)
        check(f"book {P_SIZES[i]}: most common rule share below 0.10 (the name-blind control)", share < 0.10, f"{share:.4f}")
    nm, r = next(iter(books[0].items()))
    print("  example:", p_line(nm, r, "3812").strip(), "  rule:", r)
    print("  example:", (e_prefix(r, "3812") + p_answer(apply_rule(r, "3812"))).strip())
    print("SELFTEST (probe P)", "PASS" if all(res) else "FAIL", f"({sum(res)}/{len(res)})")
    return all(res)


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
