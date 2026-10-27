"""C50-S4 Kaggle kernel: finds the datasets on the mount, prints the code manifest, runs TASK, and fails loudly.

TASK is set by `c50_s4_kaggle.py push <task>`.
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

TASK = "__TASK__"


def find(pattern: str) -> Path:
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert len(hits) == 1, (pattern, hits)
    return Path(hits[0])


code = find("c50_s4_t0_gpu.py").parent
gguf = find("ling-mini-2.0-a810f64-q4_k_m.gguf")
print("TASK", TASK, "code", code, "gguf", gguf, flush=True)
print("MANIFEST", (code / "MANIFEST.json").read_text(encoding="utf-8"), flush=True)
subprocess.run(["nvidia-smi"], check=False)
env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
           PYTHONPATH=os.pathsep.join([str(code), str(code / "gguf-py")]))
out = Path("/kaggle/working")
if TASK == "mve2":
    cmd = [sys.executable, "-u", str(code / "c50_s4_t0_gpu.py"), "--mve2", "--gguf", str(gguf), "--refs",
           str(code / "refs"), "--fingerprints", str(code / "refs/t0_dequant_fp32_sha256.json"),
           "--out", str(out / "mve2.json")]
elif TASK == "mve3":
    cmd = [sys.executable, "-u", str(code / "c50_s4_student.py"), "--mve3", "--gguf", str(gguf),
           "--teacher", str(find("teacher_v1/manifest.jsonl").parent), "--corpus", str(find("teacher.tsv")),
           "--cap-min", "20", "--out", str(out / "mve3.json")]
elif TASK == "bank":
    cmd = [sys.executable, "-u", str(code / "c50_s4_bank.py"), "--gguf", str(gguf),
           "--teacher", str(find("teacher_v1/manifest.jsonl").parent), "--corpus", str(find("teacher.tsv")),
           "--out", str(out / "bank")]
elif TASK.startswith("train-"):
    # train-<arm>[+resume]: resumes from a `ckpt.pt` mounted from the previous session's kernel, if there is one;
    # with +resume a missing checkpoint fails at once instead of training silently from step 0.
    arm = TASK[len("train-"):]
    need_ck = arm.endswith("+resume")
    arm = arm.removesuffix("+resume")
    cmd = [sys.executable, "-u", str(code / "c50_s4_train.py"), "--arm", arm, "--gguf", str(gguf),
           "--teacher", str(find("teacher_v1/manifest.jsonl").parent), "--corpus", str(find("teacher.tsv")),
           "--out", str(out / "train")]
    ck = sorted(glob.glob("/kaggle/input/**/train/ckpt.pt", recursive=True))
    assert len(ck) <= 1, ck
    assert ck or not need_ck, "a resumed session with no train/ckpt.pt mounted"
    if ck:
        cmd += ["--resume", ck[0]]
    if arm == "b":
        cmd += ["--bank", str(find("bank/bank_values.f16").parent)]
elif TASK.startswith("t1train-"):
    # t1train-<only>-<tau>[+resume] (C50-T1 part 2b): the anchored LoRA, resumed as the S4 trainings are.
    name = TASK[len("t1train-"):]
    need_ck = name.endswith("+resume")
    only, tau = name.removesuffix("+resume").split("-")
    cmd = [sys.executable, "-u", str(code / "c50_t1_train.py"), "--only", only, "--tau", tau, "--gguf", str(gguf),
           "--teacher", str(find("teacher_v1/manifest.jsonl").parent), "--corpus", str(find("teacher.tsv")),
           "--out", str(out / "train")]
    # from the previous session's kernel output (train/ckpt.pt), or from a checkpoint moved to this account as a
    # dataset: Kaggle flattens a dataset whose only content is one folder, so there it sits at the dataset's root
    ck = sorted(glob.glob("/kaggle/input/**/train/ckpt.pt", recursive=True)
                + glob.glob("/kaggle/input/**/c50-t1-*-ckpt/ckpt.pt", recursive=True))
    assert len(ck) <= 1, ck
    assert ck or not need_ck, "a resumed session with no train/ckpt.pt mounted"
    if ck:
        # a checkpoint moved to another account as a dataset is checked against its sha256 on arrival
        print("CKPT", ck[0], hashlib.sha256(Path(ck[0]).read_bytes()).hexdigest(), flush=True)
        cmd += ["--resume", ck[0]]
elif TASK.startswith("t2"):
    # C50-T2 option N, on v0 (`t1_a1_q4_k_m.gguf`, dataset c50-t2-v0): t2mve; t2n[+resume] (the N arm);
    # t2e2[+resume] (the extra-epoch control: C50-T1's anchored LoRA r48 trainer, unchanged, on v0);
    # t2eval-<v0|n|e2>[_shuffle] (the weights come from a mounted training kernel's train/final.pt).
    # C50-T2 N-fire: t2nfmve (the CPU controls, then a 150-step MVE of the fire cost); t2nf[+resume] (arm NF, fire
    # target 0.25); t2eval-nf[_shuffle] (NF's weights in the N arm); t2eval-ndocs (N on the documents only, for the
    # per-position fire count).
    v0 = find("t1_a1_q4_k_m.gguf")
    print("V0", v0, hashlib.sha256(v0.read_bytes()).hexdigest(), flush=True)
    teach = ["--teacher", str(find("teacher_v1/manifest.jsonl").parent), "--corpus", str(find("teacher.tsv"))]
    name = TASK.removesuffix("+resume")
    need_ck = TASK.endswith("+resume")
    nf = ["--fire-target", "0.25"]
    if name == "t2mve":
        cmd = [sys.executable, "-u", str(code / "c50_t2_n.py"), "--mve", "--gguf", str(v0), *teach,
               "--out", str(out / "mve")]
    elif name == "t2nfmve":
        ctl = [sys.executable, "-u", str(code / "c50_t2_n.py"), "--controls", "--gguf", str(v0), *teach,
               "--out", str(out / "controls")]
        print("CMD", json.dumps(ctl), flush=True)
        subprocess.run(ctl, env=env, check=True)
        cmd = [sys.executable, "-u", str(code / "c50_t2_n.py"), "--mve", "--mve-steps", "150", *nf, "--gguf", str(v0),
               *teach, "--out", str(out / "mve")]
    elif name in ("t2n", "t2e2", "t2nf"):
        if name in ("t2n", "t2nf"):
            cmd = [sys.executable, "-u", str(code / "c50_t2_n.py"), "--gguf", str(v0), *teach, "--out", str(out / "train")]
            cmd += nf if name == "t2nf" else []
        else:
            cmd = [sys.executable, "-u", str(code / "c50_t1_train.py"), "--only", "all", "--tau", "0.5", "--gguf", str(v0),
                   *teach, "--out", str(out / "train")]
        ck = sorted(glob.glob("/kaggle/input/**/train/ckpt.pt", recursive=True)
                    + glob.glob("/kaggle/input/**/c50-t2-*-ckpt/ckpt.pt", recursive=True))
        assert len(ck) <= 1, ck
        assert ck or not need_ck, "a resumed session with no train/ckpt.pt mounted"
        if ck:
            print("CKPT", ck[0], hashlib.sha256(Path(ck[0]).read_bytes()).hexdigest(), flush=True)
            cmd += ["--resume", ck[0]]
    elif name.startswith("t2eval-"):
        arm, _, ctrl = name[len("t2eval-"):].partition("_")
        docs_only = arm == "ndocs"
        arm = "n" if arm in ("nf", "ndocs") else arm
        cmd = [sys.executable, "-u", str(code / "c50_t2_n_eval.py"), "--gguf", str(v0), "--arm", arm,
               "--a4-inputs", str(code / "refs/inputs"), "--fresh", str(code / "refs/fresh"),
               "--split", str(code / "refs/c50_s4_split.json"), "--out", str(out / "eval")]
        if arm != "v0":
            cmd += ["--weights", str(find("train/final.pt"))]
        if ctrl:
            cmd += ["--n-mode", ctrl]
        if docs_only:
            cmd += ["--panels", "docs"]
    else:
        raise SystemExit(f"unknown TASK {TASK}")
elif TASK.startswith("m-"):
    # C50 cell M (the metric): m-<s4|t2>-<arm>[_<mode>][+smoke]. One pass per arm: the temperature fit on the
    # validation shards and the documents' target log-probabilities at every temperature (`c50_m_eval.py`).
    name = TASK[2:].removesuffix("+smoke")
    fam, _, rest = name.partition("-")
    arm, _, mode = rest.partition("_")
    g = gguf if fam == "s4" else find("t1_a1_q4_k_m.gguf")
    print("GGUF", g, hashlib.sha256(g.read_bytes()).hexdigest(), flush=True)
    cmd = [sys.executable, "-u", str(code / "c50_m_eval.py"), "--family", fam, "--gguf", str(g), "--arm", arm,
           "--teacher", str(find("teacher_v1/manifest.jsonl").parent), "--corpus", str(find("teacher.tsv")),
           "--a4-inputs", str(code / "refs/inputs"), "--split", str(code / "refs/c50_s4_split.json"),
           "--out", str(out / "m")]
    if arm not in ("t0", "v0"):
        cmd += ["--weights", str(find("train/final.pt"))]
    if arm == "b":
        cmd += ["--bank", str(find("bank/bank_values.f16").parent)]
    if mode:
        cmd += ["--n-mode", mode]
    if TASK.endswith("+smoke"):
        cmd += ["--limit", "2"]
elif TASK.startswith("inj-"):
    # C50 injection cell, on v0. The inputs are refs/inj (`c50_inj_data.py build`); the dev pilot's and the MVE's picks
    # are refs/inj/params.json. Tasks:
    #   inj-sample                    v0's 8 draws per pool question (the "v0 does not know" filter)
    #   inj-mve                       arm e's lambda0 grid (`c50_inj_train.py --mve`)
    #   inj-dev-<e|l>-<lr>            the dev pilot: train on the dev facts at this lr, then evaluate
    #   inj-train-<arm>-<seed>[+resume]  arm e|edense|s|qa1|l, seed 1 or 2, lr and lambda0 from params.json
    #   inj-eval-<arm>-<mark>[_rot]   the injection sets on train/<mark>.pt (mark ck25|ck50|final), or v0 (+ open book)
    #   inj-general-<arm>             docs, gsm8k and chat (`c50_t2_n_eval.py`) on train/final.pt, or v0
    v0 = find("t1_a1_q4_k_m.gguf")
    print("V0", v0, hashlib.sha256(v0.read_bytes()).hexdigest(), flush=True)
    teach = ["--teacher", str(find("teacher_v1/manifest.jsonl").parent), "--corpus", str(find("teacher.tsv"))]
    inj = code / "refs/inj"
    name = TASK[len("inj-"):].removesuffix("+resume")
    params = json.loads((inj / "params.json").read_text(encoding="utf-8")) if (inj / "params.json").exists() else {}
    print("PARAMS", json.dumps(params), flush=True)
    seeds = {"1": "20261016", "2": "20261017"}
    if name == "sample":
        cmd = [sys.executable, "-u", str(code / "c50_inj_eval.py"), "--gguf", str(v0), "--arm", "v0", "--inputs", str(inj),
               "--out", str(out / "eval"), "--sample", "8", "--temp", "0.7"]
    elif name == "mve":
        cmd = [sys.executable, "-u", str(code / "c50_inj_train.py"), "--arm", "e", "--mve", "--lr", "3e-4",
               "--gguf", str(v0), *teach, "--inputs", str(inj), "--out", str(out / "mve")]
    elif name.startswith("dev-"):
        arm, lr = name[len("dev-"):].split("-", 1)
        tr = [sys.executable, "-u", str(code / "c50_inj_train.py"), "--arm", arm, "--dev", "--lr", lr, "--gguf", str(v0),
              *teach, "--inputs", str(inj), "--out", str(out / "train")]
        if arm == "e":
            tr += ["--lambda0", str(params["lambda0"])]
        print("CMD", json.dumps(tr), flush=True)
        subprocess.run(tr, env=env, check=True)
        cmd = [sys.executable, "-u", str(code / "c50_inj_eval.py"), "--gguf", str(v0), "--arm", arm, "--weights",
               str(out / "train/final.pt"), "--inputs", str(inj), "--out", str(out / "eval"), "--sets", "rew,orig,cloze"]
    elif name.startswith("train-"):
        arm, seed = name[len("train-"):].split("-")
        cmd = [sys.executable, "-u", str(code / "c50_inj_train.py"), "--arm", arm, "--seed", seeds[seed],
               "--lr", str(params["lr_l" if arm == "l" else "lr_e"]), "--gguf", str(v0), *teach, "--inputs", str(inj),
               "--out", str(out / "train")]
        if arm in ("e", "s", "qa1"):
            cmd += ["--lambda0", str(params["lambda0"])]
        ck = sorted(glob.glob("/kaggle/input/**/train/ckpt.pt", recursive=True))
        assert len(ck) <= 1, ck
        assert ck or not TASK.endswith("+resume"), "a resumed session with no train/ckpt.pt mounted"
        if ck:
            print("CKPT", ck[0], hashlib.sha256(Path(ck[0]).read_bytes()).hexdigest(), flush=True)
            cmd += ["--resume", ck[0]]
    elif name.startswith("eval-"):
        rest, _, ctrl = name[len("eval-"):].partition("_")
        arm, _, mark = rest.partition("-")
        cmd = [sys.executable, "-u", str(code / "c50_inj_eval.py"), "--gguf", str(v0),
               "--arm", "l" if arm == "l" else "v0" if arm == "v0" else "e", "--inputs", str(inj), "--out", str(out / "eval")]
        if arm == "v0":
            cmd += ["--sets", "rew,orig,cloze,known,open,open_rew"]
        else:
            cmd += ["--weights", str(find(f"train/{mark}.pt"))]
        if ctrl == "rot":
            cmd += ["--rot", "--sets", "rew,orig,cloze"]
    elif name.startswith("general-"):
        arm = name[len("general-"):]
        cmd = [sys.executable, "-u", str(code / "c50_t2_n_eval.py"), "--gguf", str(v0),
               "--arm", "v0" if arm == "v0" else "e2" if arm == "l" else "n", "--a4-inputs", str(code / "refs/inputs"),
               "--fresh", str(code / "refs/fresh"), "--split", str(code / "refs/c50_s4_split.json"),
               "--panels", "docs,gsm8k,chat", "--out", str(out / "general")]
        if arm != "v0":
            cmd += ["--weights", str(find("train/final.pt"))]
    else:
        raise SystemExit(f"unknown TASK {TASK}")
elif TASK.startswith("eval-"):
    # eval-<arm>[_<control>]: t0 (MVE-5), a, b, c48, c496, a_shuffle, b_remove, b_shuffle, b_mask.
    name = TASK[len("eval-"):]
    arm, _, ctrl = name.partition("_")
    cmd = [sys.executable, "-u", str(code / "c50_s4_eval.py"), "--gguf", str(gguf), "--inputs",
           str(code / "refs/inputs"), "--split", str(code / "refs/c50_s4_split.json"), "--arm", arm,
           "--out", str(out / "eval")]
    if arm != "t0":
        cmd += ["--weights", str(find("train/final.pt"))]
    if arm == "b":
        cmd += ["--bank", str(find("bank/bank_values.f16").parent), "--bank-mode", ctrl or "full"]
    if arm == "a":
        cmd += ["--a-mode", ctrl or "full"]
else:
    raise SystemExit(f"unknown TASK {TASK}")
print("CMD", json.dumps(cmd), flush=True)
subprocess.run(cmd, env=env, check=True)
print("DONE", TASK, flush=True)
