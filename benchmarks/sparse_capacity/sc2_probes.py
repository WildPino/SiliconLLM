#!/usr/bin/env python3
# Sparse Capacity, phase 2 (D-0, D-R): the two probes, mixed into the training stream as Python text.
# Protocol: docs/research/SPARSE_CAPACITY/F2_BANK_PROTOCOL.md, section 4. Tasks after Mixture of Parrots
# (arXiv 2410.19034 v2, section 4.1): a phone book (memorization) and shortest paths on random directed graphs
# (reasoning), here written as Python lines so the code tokenizer reads them.
#
#   K (knowledge)  three fixed phone books of 256, 2048 and 16384 entries, equal token share each, so the exposures per
#                  entry are about 222, 28 and 3.5 over the run: one of the three is in the discriminating range
#                  whatever the model. Line: tel["kqzvw"] = 48213907
#                  Score: exact match of the 8 digits given the prefix, on training entries (memorization by design).
#   R (reasoning)  fresh graphs every time (never repeated, test disjoint from train): n in {6, 9} vertices, query with a
#                  UNIQUE shortest path of length L in {1, 2, 3, 4}. Line pair:
#                      g = {0: [3, 5], 1: [2], ...}
#                      assert path(g, 4, 7) == [4, 2, 7]
#                  Score: exact match of the path list given the prefix. Primary = mean over n x L>=2 (L = 1 is a lookup).
#   Controls       K: names never in a book (must be ~0). R: the same queries scored against a shuffled graph (the edge
#                  list of another instance), and a no-model heuristic (walk the first listed edge).
import random, string
from collections import deque

K_SIZES = (256, 2048, 16384)
R_N = (6, 9)
R_L = (1, 2, 3, 4)
R_PRIMARY = ((9, 3), (9, 4))          # the cells a one-hop lookup does not solve (review item 6)
R_OOD = ((12, 5),)                    # never trained: a learned procedure vs memorized fragments (a reading)
K_SEED, R_TEST_SEED = 12345, 777


# ---------------------------------------------------------------- K
def make_books(seed=K_SEED):
    """Three disjoint books {name: number}; names 5 lowercase letters, numbers 8 digits, all unique."""
    rng = random.Random(seed)
    names, nums = set(), set()
    books = []
    for size in K_SIZES:
        b = {}
        while len(b) < size:
            nm = "".join(rng.choice(string.ascii_lowercase) for _ in range(5))
            nu = rng.randrange(10_000_000, 100_000_000)
            if nm in names or nu in nums:
                continue
            names.add(nm); nums.add(nu); b[nm] = nu
        books.append(b)
    unseen = []
    while len(unseen) < 2048:
        nm = "".join(rng.choice(string.ascii_lowercase) for _ in range(5))
        if nm not in names:
            names.add(nm); unseen.append((nm, rng.randrange(10_000_000, 100_000_000)))
    return books, unseen


def k_prefix(name):
    return f'tel["{name}"] = '


def k_answer(num):
    return f"{num}\n"


# ---------------------------------------------------------------- R
def bfs_paths(adj, s):
    """dist and number of shortest paths from s (counts capped at 2), plus one parent per vertex."""
    n = len(adj)
    dist = [-1] * n; cnt = [0] * n; par = [-1] * n
    dist[s] = 0; cnt[s] = 1
    q = deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1; cnt[v] = cnt[u]; par[v] = u; q.append(v)
            elif dist[v] == dist[u] + 1:
                cnt[v] = min(2, cnt[v] + cnt[u])
    return dist, cnt, par


def r_instance(rng, n, L, p=None):
    """Random directed graph on n vertices and a query (s, t) whose shortest path has length L and is unique."""
    p = p if p is not None else 2.8 / n                   # first-edge heuristic solves 8-29% at L >= 2 (16-50% at 1.6/n)
    for _ in range(10_000):
        adj = [sorted(v for v in range(n) if v != u and rng.random() < p) for u in range(n)]
        order = list(range(n)); rng.shuffle(order)
        for s in order:
            dist, cnt, par = bfs_paths(adj, s)
            ts = [t for t in range(n) if dist[t] == L and cnt[t] == 1]
            if ts:
                t = rng.choice(ts)
                path = [t]
                while path[-1] != s:
                    path.append(par[path[-1]])
                return adj, s, t, path[::-1]
    raise RuntimeError("no instance")


def r_graph_text(adj):
    return "g = {" + ", ".join(f"{u}: [{', '.join(map(str, vs))}]" for u, vs in enumerate(adj)) + "}\n"


def r_prefix(adj, s, t):
    return r_graph_text(adj) + f"assert path(g, {s}, {t}) == "


def r_answer(path):
    return "[" + ", ".join(map(str, path)) + "]\n"


def r_key(adj, s, t):
    return (tuple(tuple(v) for v in adj), s, t)


def r_heuristic(adj, s, t, maxlen=4):
    """No-model control: from s follow the first listed out-edge until t or maxlen steps."""
    path = [s]
    while path[-1] != t and len(path) <= maxlen:
        if not adj[path[-1]]:
            break
        path.append(adj[path[-1]][0])
    return path


def r_lookahead(adj, s, t, rng, maxlen=5):
    """No-model control, the strongest local rule: step to t if adjacent, else to an unvisited neighbour that has t
    adjacent, else to any unvisited neighbour (random choices). Solves L = 2 always (review item 6)."""
    path = [s]
    while path[-1] != t and len(path) <= maxlen:
        u = path[-1]
        if t in adj[u]:
            path.append(t); continue
        c = [v for v in adj[u] if v not in path and t in adj[v]] or [v for v in adj[u] if v not in path]
        if not c:
            break
        path.append(rng.choice(c))
    return path


def r_walk(adj, s, t, rng, maxlen=5):
    """No-model control: random walk that never revisits a vertex, stops at t or after maxlen steps."""
    path = [s]
    while path[-1] != t and len(path) <= maxlen:
        nxt = [v for v in adj[path[-1]] if v not in path]
        if not nxt:
            break
        path.append(t if t in nxt else rng.choice(nxt))
    return path


def r_test_set(per_cell=500, seed=R_TEST_SEED):
    """Fixed test instances per (n, L), trained cells plus the OOD cell. Keys are excluded from training generation."""
    rng = random.Random(seed)
    out, keys = [], set()
    for n, L in [(n, L) for n in R_N for L in R_L] + list(R_OOD):
        if True:
            got = 0
            while got < per_cell:
                adj, s, t, path = r_instance(rng, n, L)
                k = r_key(adj, s, t)
                if k in keys:
                    continue
                keys.add(k); got += 1
                out.append(dict(n=n, L=L, adj=adj, s=s, t=t, path=path))
    return out, keys


def r_train_stream(seed, test_keys):
    """Endless fresh training instances, uniform over (n, L), never a test instance."""
    rng = random.Random(seed)
    while True:
        n = rng.choice(R_N); L = rng.choice(R_L)
        adj, s, t, path = r_instance(rng, n, L)
        k = r_key(adj, s, t)
        if k in test_keys:
            continue
        yield k, r_prefix(adj, s, t) + r_answer(path)


def selftest():
    import itertools
    rng = random.Random(1)
    bad = 0
    for n, L in itertools.product(R_N, R_L):
        for _ in range(200):
            adj, s, t, path = r_instance(rng, n, L)
            # brute force: every simple path s->t of length <= L, the shortest must be unique and equal to path
            best = []
            def dfs(u, pth):
                if len(pth) - 1 > L:
                    return
                if u == t:
                    best.append(list(pth)); return
                for v in adj[u]:
                    if v not in pth:
                        pth.append(v); dfs(v, pth); pth.pop()
            dfs(s, [s])
            m = min(len(b) for b in best) - 1
            short = [b for b in best if len(b) - 1 == m]
            if m != L or len(short) != 1 or short[0] != path:
                bad += 1
    books, unseen = make_books()
    allnames = set().union(*[set(b) for b in books])
    ok_books = (sum(len(b) for b in books) == len(allnames) and not (allnames & {u for u, _ in unseen}))
    test, keys = r_test_set(per_cell=50)
    gen = r_train_stream(5, keys)
    leak = sum(next(gen)[0] in keys for _ in range(2000))
    return dict(r_bad_instances=bad, k_books_disjoint=ok_books, r_test=len(test), leak=leak)


if __name__ == "__main__":
    print(selftest())
