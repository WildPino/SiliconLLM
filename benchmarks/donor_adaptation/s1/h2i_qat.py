#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H2I Phase B: train the E37 carve while the FFN remains one-byte R8.

This is deliberately a thin, auditable adaptation of h1_qat.py.  H0 q/o factors stay
frozen; only the fp32 masters behind R8 and the E37-initialised routers are trainable.
The first session is fresh by design: ``--resume`` is rejected.  The full T4 run is not
performed by this file during development; ``--self-test`` and ``--cpu-smoke`` are cheap
apparatus checks.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DENSDIR = os.path.abspath(os.path.join(HERE, "..", "density"))
TERNDIR = os.path.abspath(os.path.join(HERE, "..", "ternary"))
ENGDIR = os.path.abspath(os.path.join(HERE, "..", "engine"))
for _path in (DENSDIR, TERNDIR, ENGDIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import common as C  # noqa: E402
import h1_qat as H1  # noqa: E402
import h2i_applied as H2A  # noqa: E402

torch.set_num_threads(int(os.environ.get("D_THREADS", "6")))
torch.set_grad_enabled(True)

LAYERS = [3, 6, 9, 12, 15, 18, 21, 24]
SMOKE_LAYERS = [3, 24]
E = 256
K = 16
MODEL = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
LN2 = 0.6931471805599453
EXPECTED_PHASE_A = 1.0190764652622473
EXPECTED_IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
EXPECTED_BYTES = 51870
GATE_WINDOW = 40

DEFAULTS = {
    "factors": "D:/_ktmp/h0_run3_final/h0_trained3.npz",
    "labels": os.path.join(DENSDIR, "results", "d0c_labels", "labels_E256.npz"),
    "stats": os.path.join(HERE, "results", "h1", "h1_actstats.npz"),
    "routers": "D:/_ktmp/e37/e37_routers_E256.npz",
    "phase_a": os.path.join(HERE, "results", "h2i", "h2i_applied_8L.json"),
    "train": os.path.join(HERE, "_h1_bundle", "h1_train.npz"),
    "heldout": os.path.join(HERE, "results", "h1", "h1_heldout.npz"),
    "calib": os.path.join(HERE, "results", "h1", "h1_calib.npz"),
    "out": os.path.join(HERE, "results", "h2i", "h2i_qat.npz"),
}

PINNED = {
    "factors": (352967686, "dd62482d65accb34129a7b7a3f8baf75b0ccf3c4fdf2481233f1b9ce823927d8"),
    "labels": (859537, "c39d0740b7f754daf168d811c77120c5394ae677e3882fef208d331d8c333c9c"),
    "stats": (1189630, "49fd2f659a37237ead850ea86057664b74db4f2286dc7146abf87e197e6887fe"),
    "routers": (44046858, "42b12cb9d4bda2da7833cd0e179a9dab8e3f43264ce640e2b5db31d5fd043d8a"),
    "phase_a": (12239, "c367acb45113e9ce68fb898de0b0833db61b6a1b84990b7ccc6467f82d2ef8cc"),
    "train": (64000260, "0cdaa28f405c3a8c6a8589af8e88788b33131bc7d4f1096da6c1fedf78caa9d9"),
    "heldout": (98564, "110a90ab358efcacdc090fdf0fce62bb74d6d06df02d1083cf4cf7e752529d35"),
    "calib": (131332, "b8d4184db0988f4d86ae3089e99ba167b8e05e6dcdbbc504f7679729fe7aa019"),
}


def log(msg):
    print(msg, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def apparatus_files():
    """Every Python source whose behavior this thin runner inherits."""
    paths = [__file__, H1.__file__, H2A.__file__, C.__file__, H2A.T2.__file__,
             os.path.join(ENGDIR, "e6_generate.py")]
    return list(dict.fromkeys(os.path.abspath(path) for path in paths))


def committed_provenance():
    """Pin apparatus to HEAD locally; Kaggle's no-git copy is pinned by its manifest."""
    try:
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=HERE,
                                       text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return {"mode": "manifest-only-no-git", "head_commit": None,
                "files": {os.path.basename(path): {"sha256": sha256_file(path)}
                          for path in apparatus_files()}}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    files = apparatus_files()
    rows = {}
    for path in files:
        rel = os.path.relpath(path, root).replace("\\", "/")
        work = subprocess.check_output(["git", "hash-object", path], cwd=root, text=True).strip()
        try:
            committed = subprocess.check_output(["git", "rev-parse", "HEAD:" + rel], cwd=root,
                                                text=True, stderr=subprocess.STDOUT).strip()
        except subprocess.CalledProcessError as exc:
            raise SystemExit("APPARATUS NOT TRACKED AT HEAD: %s" % rel) from exc
        if work != committed:
            raise SystemExit("APPARATUS IS NOT THE COMMITTED HEAD BLOB: %s" % rel)
        rows[rel] = {"git_blob": work, "sha256": sha256_file(path)}
    return {"mode": "git-head-plus-manifest", "head_commit": head, "files": rows}


def validate_manifest(path, inputs):
    """Validate a bundle manifest without trusting paths or recorded hashes."""
    if not path:
        raise SystemExit("--manifest is required for a training launch; use --self-test for toy checks")
    if not os.path.isfile(path):
        raise SystemExit("missing H2I bundle manifest: %s" % path)
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise SystemExit("H2I manifest has no files map -- STOP")
    for key, value in inputs.items():
        name = os.path.basename(value)
        candidates = [name, key, os.path.abspath(value)]
        entry = next((files[c] for c in candidates if c in files), None)
        if entry is None:
            raise SystemExit("manifest does not list %s (%s) -- STOP" % (key, name))
        expected_size = entry.get("bytes") if isinstance(entry, dict) else None
        expected_sha = entry.get("sha256") if isinstance(entry, dict) else entry
        actual_size, actual_sha = os.path.getsize(value), sha256_file(value)
        if expected_size is not None and int(expected_size) != actual_size:
            raise SystemExit("manifest bytes mismatch for %s" % key)
        if expected_sha != actual_sha:
            raise SystemExit("manifest sha256 mismatch for %s" % key)
    return {"path": os.path.abspath(path), "sha256": sha256_file(path),
            "files": len(files), "validated": True}


def validate_inputs(paths):
    actual = {}
    for key, path in paths.items():
        if not os.path.isfile(path):
            raise SystemExit("missing pinned H2I input %s: %s" % (key, path))
        size, digest = os.path.getsize(path), sha256_file(path)
        want_size, want_digest = PINNED[key]
        if (size, digest) != (want_size, want_digest):
            raise SystemExit("PINNED INPUT MISMATCH %s: %d %s (expected %d %s)" %
                             (key, size, digest, want_size, want_digest))
        actual[key] = {"path": os.path.abspath(path), "bytes": size, "sha256": digest}
    with open(paths["phase_a"], encoding="utf-8") as fh:
        phase_a = json.load(fh)
    if phase_a.get("status") != "PHASE_B_ELIGIBLE" or \
            phase_a.get("bpb", {}).get("applied-int8-8L") != EXPECTED_PHASE_A:
        raise SystemExit("Phase A result is not the pinned eligible H2I object -- STOP")
    if phase_a.get("eval_slice", {}).get("ids_sha256") != EXPECTED_IDS_SHA or \
            phase_a.get("eval_slice", {}).get("total_scored_bytes") != EXPECTED_BYTES:
        raise SystemExit("Phase A slice identity drifted -- STOP")
    return actual, phase_a


def _assert_r8():
    """B.3.1: exact parity with h2i_applied's shipped R8 implementation."""
    torch.manual_seed(271828)
    planted = torch.tensor([[-8.0, -3.25, -1.0, -0.125, 0.0, 0.25, 2.75, 7.0],
                            [-0.031, -0.017, -0.004, 0.0, 0.003, 0.011, 0.019, 0.029]])
    got = H2A.r8_ste(planted)
    tq, ta = H2A.T2.r8_int8_rtn(planted)
    want = tq * ta
    exact = torch.equal(got, want)
    distinct = int(torch.unique(H2A.r8_codes(planted)[0]).numel())
    if not exact or distinct < 5:
        raise SystemExit("G-H2Ib forward parity/distinct-code control failed")
    return {"r8_exact_t2_rules": exact, "distinct_planted_codes": distinct}


def _toy_controls():
    """B.3.2 plus gate-boundary controls, no model or scientific input loaded."""
    d, f, groups = 6, 140, 4
    labels = torch.arange(f) % groups
    mod = H2A.Int8CarvedFFN(torch.randn(f, d) / 5, torch.randn(f, d) / 5,
                            torch.randn(d, f) / 5, labels, torch.ones(d), torch.ones(f), 2, groups)
    mod.router.data.normal_(0, 0.1)
    x = torch.randn(2, 3, d)
    with torch.enable_grad():
        mod(x).square().mean().backward()
    gradients = {n: bool(getattr(mod, n).grad is not None and
                         torch.isfinite(getattr(mod, n).grad).all() and
                         torch.count_nonzero(getattr(mod, n).grad) > 0)
                 for n in ("gate", "up", "down")}
    if not all(gradients.values()):
        raise SystemExit("G-H2Ib STE gradient control failed: %r" % gradients)
    mod.hard_gate = True
    same_h, dmax_h = H1.g_h1a(mod, x)
    active = mod.group_mask(x).sum(-1)
    live, live_dmax = H1.carve_is_live(mod, x)
    mod.hard_gate = False
    saved_router = mod.router.detach().clone()
    mod.router.data.zero_()
    mod.invalidate()
    same_s, dmax_s = H1.g_h1a(mod, x)
    mod.router.data.copy_(saved_router)
    mod.invalidate()
    if not same_h or dmax_h != 0.0 or not same_s or dmax_s != 0.0 \
            or not torch.all(active == 70) or not live:
        raise SystemExit("G-H2Ic toy carve controls failed")
    return {"ste_gradients_finite_nonzero": gradients,
            "hard_k_equals_E_any_router": same_h, "hard_max_abs_diff": dmax_h,
            "soft_k_equals_E_zero_router": same_s, "soft_max_abs_diff": dmax_s,
            "hard_active_neurons_each": active.tolist(), "expected_each": 70,
            "carve_live": live, "carve_max_abs_diff": live_dmax}


def score_fires(value):
    return value < EXPECTED_PHASE_A


def _seeded_probe(shape, device, dtype, seed=90210):
    """Draw reproducibly on CPU, then transfer; CPU generators cannot drive CUDA randn."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    probe = torch.randn(*shape, generator=generator, dtype=torch.float32, device="cpu")
    return probe.to(device=device, dtype=dtype)


def _seeded_probe_selftest():
    """Plant determinism and global-RNG isolation checks for the cross-device-safe helper."""
    before = torch.random.get_rng_state().clone()
    first = _seeded_probe((2, 3, 5), "cpu", torch.float32)
    middle = torch.random.get_rng_state().clone()
    second = _seeded_probe((2, 3, 5), torch.device("cpu"), torch.float32)
    after = torch.random.get_rng_state()
    exact = bool(torch.equal(first, second))
    isolated = bool(torch.equal(before, middle) and torch.equal(before, after))
    shape_ok = tuple(first.shape) == (2, 3, 5)
    finite = bool(torch.isfinite(first).all())
    if not (exact and isolated and shape_ok and finite):
        raise SystemExit("seeded device-transfer probe self-test failed")
    return {"source_device": "cpu", "transfer_target": "caller",
            "repeat_exact": exact, "global_rng_unchanged": isolated,
            "shape_ok": shape_ok, "finite": finite}


def selftest():
    return {"G_H2Ib": _assert_r8(), "G_H2Ic_toy": _toy_controls(),
            "device_safe_seeded_probe": _seeded_probe_selftest(),
            "gate_boundary_test": {
                "equal_threshold_fails": not score_fires(EXPECTED_PHASE_A),
                "below_threshold_fires": score_fires(EXPECTED_PHASE_A - 1e-12),
                "fires": (not score_fires(EXPECTED_PHASE_A)
                          and score_fires(EXPECTED_PHASE_A - 1e-12))}}


def _install(model, factors, labels, stats, routers, layers, device):
    H1.build_qo(model, factors, device)
    for p in model.parameters():
        p.requires_grad_(False)
    H2A.build_ffn_int8(model, labels, stats, layers, K, E, device)
    mods = H1.ffn_mods(model, layers)
    with np.load(routers) as z:
        for li, mod in mods:
            key = "r%d" % li
            if key not in z.files or tuple(z[key].shape) != tuple(mod.router.shape):
                raise SystemExit("E37 router %s is missing or has the wrong shape" % key)
            source = torch.from_numpy(z[key]).float().to(device)
            mod.router.data.copy_(source)
            mod._h2i_router_init_exact = bool(torch.equal(mod.router.detach(), source))
            mod.invalidate()
    for _, mod in mods:
        mod.hard_gate = False
        mod.route_norm = False
        mod.k = K
        mod.router.requires_grad_(True)
        for name in ("gate", "up", "down"):
            getattr(mod, name).requires_grad_(True)
    return mods


@torch.no_grad()
def _real_controls(model, mods):
    """B.3.2/3 on a real layer: both identities, exact k16 cardinality and live mask."""
    mod = mods[0][1]
    x = _seeded_probe((1, 6, model.config.hidden_size), mod.gate.device, mod.gate.dtype)
    mod.hard_gate = True
    same_h, dmax_h = H1.g_h1a(mod, x)
    active = mod.group_mask(x).sum(-1)
    live, live_dmax = H1.carve_is_live(mod, x)
    mod.hard_gate = False
    saved_router = mod.router.detach().clone()
    mod.router.data.zero_()
    mod.invalidate()
    same_s, dmax_s = H1.g_h1a(mod, x)
    mod.router.data.copy_(saved_router)
    mod.invalidate()
    exact_router = all(getattr(item, "_h2i_router_init_exact", False) for _, item in mods)
    cardinality = bool(torch.all(active == 560))
    fires = bool(same_h and dmax_h == 0.0 and same_s and dmax_s == 0.0
                 and cardinality and live and exact_router)
    if not fires:
        raise SystemExit("B.3 real-shape wiring/router-init control failed")
    return {"layer": mods[0][0], "hard_kE_any_router": same_h,
            "hard_max_abs_diff": dmax_h, "soft_kE_zero_router": same_s,
            "soft_max_abs_diff": dmax_s, "hard_k16_active": active.tolist(),
            "expected_active": 560, "cardinality_fires": cardinality,
            "carve_live": live, "carve_max_abs_diff": live_dmax,
            "router_init_exact": exact_router, "fires": fires}


def _exact_reload(mods, path):
    """Save/reload control: every master/router/label is exact and cache is dropped."""
    before = {}
    by_layer = {li: mod for li, mod in mods}
    with np.load(path) as z:
        for li, mod in mods:
            p = "L%02d" % li
            for name in ("gate", "up", "down", "router", "labels"):
                before[p + "." + name] = np.array(z[p + "." + name], copy=True)
    # Plant a mutation and a populated no-grad cache so the reload must genuinely restore both.
    li0, mod0 = mods[0]
    mod0.gate.data.view(-1)[0].add_(1.0)
    with torch.no_grad():
        mod0._quant()
    if mod0._ck is None:
        return False
    for li, mod in mods:
        p = "L%02d" % li
        with np.load(path) as z:
            for name in ("gate", "up", "down", "router"):
                getattr(mod, name).data.copy_(torch.from_numpy(z[p + "." + name]).to(mod.gate.device))
            if not torch.equal(mod.labels.cpu(), torch.from_numpy(z[p + ".labels"])):
                return False
        mod.invalidate()
        if mod._ck is not None or mod._cv is not None:
            return False
    return all(np.array_equal(before[k], getattr(by_layer[int(k[1:3])], k.split(".")[1]).detach().cpu().numpy())
               for k in before if ".labels" not in k)


def save_bundle(path, mods, meta):
    store = {}
    for li, mod in mods:
        p = "L%02d" % li
        for name in ("gate", "up", "down", "router", "rms_in", "rms_h", "labels"):
            value = getattr(mod, name)
            store[p + "." + name] = (value.detach().cpu().numpy() if name == "labels"
                                      else value.detach().float().cpu().numpy())
    np.savez(path, **store)
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=float)


@torch.no_grad()
def heldout_nats(model, ids, dev, cuda, bs=1):
    total, count = 0.0, 0
    for i in range(0, ids.shape[0], bs):
        chunk = ids[i:i + bs].to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
            logits = model(chunk).logits
        lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
        total += float(-lp.gather(-1, chunk[:, 1:].unsqueeze(-1)).squeeze(-1).double().sum())
        count += int(chunk[:, 1:].numel())
    return total, count


def _smoke(model, mods, train_ids, device, cuda):
    """One real-donor update on layers 3 and 24; apparatus only, not a quality result."""
    opt = torch.optim.AdamW([{"params": [p for _, m in mods for p in (m.gate, m.up, m.down)], "lr": 2e-4},
                             {"params": [m.router for _, m in mods], "lr": 3e-4}],
                            weight_decay=0.0, betas=(0.9, 0.95))
    watch = [("L03.gate", mods[0][1].gate), ("L24.down", mods[-1][1].down),
             ("L03.router", mods[0][1].router)]
    snap = [v.detach().clone() for _, v in watch]
    opt.zero_grad(set_to_none=True)
    for j in range(2):
        batch = train_ids[j:j + 1].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
            out = model(batch, labels=batch)
            aux = sum(m._aux for _, m in mods if m._aux is not None)
            loss = (out.loss + 0.01 * aux) / 2
            if not torch.isfinite(loss):
                raise SystemExit("B.3 real CPU smoke produced a non-finite microbatch")
            loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0))
    before = H1.applied_steps(opt)
    opt.step()
    applied = H1.applied_steps(opt) > before
    moved = {name: float((value.detach() - old).abs().max()) for (name, value), old in zip(watch, snap)}
    if not applied or not all(v > 0 and np.isfinite(v) for v in moved.values()):
        raise SystemExit("B.3 real CPU smoke did not apply/move all watched parameters")
    return {"applied": applied, "applied_steps": H1.applied_steps(opt), "moved": moved,
            "grad_norm": grad_norm, "nonfinite_microbatches": 0,
            "layers": SMOKE_LAYERS}


def main():
    ap = argparse.ArgumentParser()
    for key in ("factors", "labels", "stats", "routers", "phase_a", "train", "heldout", "calib", "out"):
        ap.add_argument("--" + key.replace("_", "-"), default=DEFAULTS[key])
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--router-lr", type=float, default=3e-4)
    ap.add_argument("--aux", type=float, default=0.01)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--bpt", type=float, default=4.22945205479452)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--cpu-smoke", action="store_true",
                    help="one real-donor update on layers 3 and 24; no full training")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    if args.self_test:
        log(json.dumps(selftest(), indent=1))
        return 0
    if args.resume:
        raise SystemExit("--resume is forbidden for the first H2I Phase B session")
    if args.steps != 4000 or args.bs != 2 or args.accum != 8 or args.lr != 2e-4 or \
            args.router_lr != 3e-4 or args.aux != 0.01 or args.every != 250 or \
            args.seed != 4242 or args.max_hours != 11.0 or args.bpt != 4.22945205479452:
        raise SystemExit("Phase B fixed controls were overridden -- STOP")
    paths = {k: getattr(args, k) for k in PINNED}
    log("== H2I QAT preflight ==")
    inputs, phase_a = validate_inputs(paths)
    manifest_inputs = dict(paths)
    for index, path in enumerate(apparatus_files()):
        manifest_inputs["apparatus_%02d" % index] = path
    manifest = validate_manifest(args.manifest, manifest_inputs)
    provenance = committed_provenance()
    controls = selftest()
    checkpoint = os.path.splitext(args.out)[0] + ".checkpoint.npz"
    smoke_result = os.path.splitext(args.out)[0] + "_cpu_smoke.json"
    reserved = [args.out, os.path.splitext(args.out)[0] + ".json"]
    reserved += [smoke_result] if args.cpu_smoke else [checkpoint,
                                                       os.path.splitext(checkpoint)[0] + ".json"]
    existing = [path for path in reserved if os.path.exists(path)]
    if existing:
        raise SystemExit("fresh H2I output path already exists; refusing overwrite: %s"
                         % existing)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    log("pre-GPU controls and pinned inputs FIRE; loading Qwen with SDPA")
    dev = args.dev or ("cuda" if torch.cuda.is_available() else "cpu")
    cuda = dev.startswith("cuda")
    if args.cpu_smoke and cuda:
        raise SystemExit("--cpu-smoke must run with --dev cpu; GPU smoke is not B.3's control")
    if not args.cpu_smoke and not cuda:
        raise SystemExit("full H2I Phase B requires CUDA; use --cpu-smoke --dev cpu locally")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                                   dtype=torch.float16 if cuda else torch.float32,
                                                   attn_implementation="sdpa")
    impl = getattr(model.config, "_attn_implementation", None)
    if impl != "sdpa":
        raise SystemExit("attention implementation is %r, not sdpa" % impl)
    model.to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    smoke_layers = SMOKE_LAYERS if args.cpu_smoke else LAYERS
    mods = _install(model, args.factors, args.labels, args.stats, args.routers, smoke_layers, dev)
    controls["G_H2Ic_real"] = _real_controls(model, mods)
    train_ids = torch.from_numpy(np.load(args.train)["ids"]).long()
    if args.cpu_smoke:
        smoke_started = time.time()
        model.train()
        controls["G_H2I_smoke"] = _smoke(model, mods, train_ids, dev, cuda)
        log(json.dumps(controls["G_H2I_smoke"], indent=1))
        tmp = args.out + ".smoke.npz"
        save_bundle(tmp, mods, {"smoke": True, "step": 1})
        controls["save_reload_exact"] = _exact_reload(mods, tmp)
        if not controls["save_reload_exact"]:
            raise SystemExit("B.3 save/reload exact control failed")
        os.remove(tmp)
        os.remove(os.path.splitext(tmp)[0] + ".json")
        with open(smoke_result, "x", encoding="utf-8") as fh:
            json.dump({"brief": "BRIEF_H2I_TRAIN_THE_CARVE_AT_ONE_BYTE.md addendum B",
                       "status": "CPU_SMOKE_PASS", "provenance": provenance,
                       "inputs": inputs, "manifest": manifest, "controls": controls,
                       "layers": SMOKE_LAYERS, "device": dev,
                       "seconds": time.time() - smoke_started}, fh, indent=1)
        log("CPU smoke FIRE; full training intentionally not launched")
        log("wrote %s" % smoke_result)
        return 0
    # Full Phase B loop: dense training with soft gate, hard evaluation at checkpoints.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    ev = torch.from_numpy(np.load(args.heldout)["ids"]).long()
    cal = torch.from_numpy(np.load(args.calib)["ids"]).long()
    experts = [p for _, m in mods for p in (m.gate, m.up, m.down)]
    routers = [m.router for _, m in mods]
    opt = torch.optim.AdamW([{"params": experts, "lr": args.lr}, {"params": routers, "lr": args.router_lr}],
                            weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.router_lr],
                                                 total_steps=args.steps, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    rng = np.random.default_rng(args.seed)
    soft0, n0 = heldout_nats(model, ev, dev, cuda)
    previous_hard = [mod.hard_gate for _, mod in mods]
    for _, mod in mods:
        mod.hard_gate = True
    hard0, _ = heldout_nats(model, ev, dev, cuda)
    for (_, mod), value in zip(mods, previous_hard):
        mod.hard_gate = value
    hist = [{"step": 0, "bpb_soft": soft0 / (LN2 * args.bpt * n0),
             "bpb_hard": hard0 / (LN2 * args.bpt * n0), "seconds": 0.0}]
    nonfinite, declined = 0, 0
    watch = [("L03.gate", mods[0][1].gate), ("L24.down", mods[-1][1].down),
             ("L03.router", mods[0][1].router)]
    watch_snap = [v.detach().clone() for _, v in watch]
    first_moved = None
    start = time.time()
    stop_reason, applied_updates = "steps-exhausted", 0
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        run_loss, run_aux, micro = 0.0, 0.0, 0
        for _ in range(args.accum):
            choose = rng.integers(0, train_ids.shape[0], size=args.bs)
            batch = train_ids[torch.from_numpy(choose)].to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                out = model(batch, labels=batch)
                aux = sum(m._aux for _, m in mods if m._aux is not None)
                loss = (out.loss + args.aux * aux) / args.accum
            if not torch.isfinite(loss):
                nonfinite += 1
                raise SystemExit("non-finite Phase B microbatch at step %d -- numerical "
                                 "failure, not H2I's null" % step)
            scaler.scale(loss).backward()
            run_loss += float(out.loss.detach()); run_aux += float(aux.detach()); micro += 1
        scaler.unscale_(opt)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(experts + routers, 1.0))
        before = H1.applied_steps(opt); scaler.step(opt); scaler.update(); sched.step()
        applied = H1.applied_steps(opt) > before
        if applied:
            applied_updates += 1
        elif first_moved is None:
            declined += 1
            if declined >= GATE_WINDOW:
                raise SystemExit("no optimizer update applied in the first %d attempts -- "
                                 "numerical failure, not H2I's null" % GATE_WINDOW)
        if applied and first_moved is None:
            first_moved = {name: float((value.detach() - old).abs().max())
                           for (name, value), old in zip(watch, watch_snap)}
            if not all(value > 0 and np.isfinite(value) for value in first_moved.values()):
                raise SystemExit("B.3 optimizer applied but a watched master/router did not move")
        elapsed = time.time() - start
        if step % args.every == 0 or step == args.steps:
            soft_n, n = heldout_nats(model, ev, dev, cuda)
            prev = [m.hard_gate for _, m in mods]
            for _, m in mods: m.hard_gate = True
            hard_n, _ = heldout_nats(model, ev, dev, cuda)
            for (_, m), value in zip(mods, prev): m.hard_gate = value
            occ = [float(m._occ.max()) for _, m in mods if m._occ is not None]
            row = {"step": step, "bpb_soft": soft_n / (LN2 * args.bpt * n),
                   "bpb_hard": hard_n / (LN2 * args.bpt * n), "loss": run_loss / max(1, micro),
                   "aux": run_aux / max(1, micro), "occupancy_max": max(occ) if occ else None,
                   "optimizer_applied_updates": applied_updates, "optimizer_declined_updates": declined,
                   "grad_norm": grad_norm, "seconds": elapsed}
            hist.append(row); log(json.dumps(row, sort_keys=True))
            meta = {"brief": "BRIEF_H2I_TRAIN_THE_CARVE_AT_ONE_BYTE.md addendum B",
                    "phase": "B", "status": "in-progress", "layers": LAYERS, "E": E, "k": K,
                    "ffn_rule": "R8 per-output-row amax/127 RTN, codes [-127,127], STE",
                    "steps_requested": args.steps, "steps_completed": step, "stop_reason": "in-progress",
                    "seed": args.seed, "lr": args.lr, "router_lr": args.router_lr, "aux": args.aux,
                    "bs": args.bs, "accum": args.accum, "max_hours": args.max_hours,
                    "provenance": provenance, "inputs": inputs, "manifest": manifest,
                    "phase_a_threshold_hard_bpb": EXPECTED_PHASE_A, "history": hist,
                    "nonfinite_microbatches": nonfinite, "optimizer_applied_updates": applied_updates,
                    "scaler_declined_before_first_applied": declined,
                    "optimizer_moved_at_first_applied": first_moved,
                    "router_init_sha256": PINNED["routers"][1],
                    "resumed_ffn_from": None}
            save_bundle(checkpoint, mods, meta)
        if elapsed >= args.max_hours * 3600:
            stop_reason = "time-cap"; break
    elapsed = time.time() - start
    r, s, fires, picks = H1.g_h1e(model, mods, cal, ev, dev, cuda)
    final = {"brief": "BRIEF_H2I_TRAIN_THE_CARVE_AT_ONE_BYTE.md addendum B", "phase": "B",
             "complete": True, "model": MODEL, "revision": REVISION, "layers": LAYERS, "E": E, "k": K,
             "ffn_rule": "R8 per-output-row amax/127 RTN, codes [-127,127], STE", "sdpa": impl,
             "provenance": provenance, "inputs": inputs, "manifest": manifest,
             "steps_requested": args.steps, "steps_completed": step, "stop_reason": stop_reason,
             "seconds": elapsed, "seconds_per_step": elapsed / max(1, step), "seed": args.seed,
             "lr": args.lr, "router_lr": args.router_lr, "aux": args.aux, "bs": args.bs, "accum": args.accum,
              "nonfinite_microbatches": nonfinite, "optimizer_applied_updates": applied_updates,
              "scaler_declined_before_first_applied": declined,
              "optimizer_moved_at_first_applied": first_moved,
              "router_init_sha256": PINNED["routers"][1], "resumed_ffn_from": None,
             "progress": hist, "router_vs_STATIC": {"router_nats": r, "static_nats": s,
                                                         "fires": fires, "static_groups": picks},
             "controls": controls}
    save_bundle(args.out, mods, final)
    log("stopped after %d steps (%s); wrote NPZ+JSON" % (step, stop_reason))
    return 0


if __name__ == "__main__":
    sys.exit(main())
