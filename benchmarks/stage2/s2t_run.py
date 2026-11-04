"""Stage 2 test 2b runner.  It prints numbers, never a verdict: s2t_verdict.py decides.

    dump --set fit|dev|gen|docs|tern   the scorer writes each position's hidden state, its records and its top-32 into
        C:/SiliconLLM_fast/s2t/h/.  docs is the panel's documents on P0: its records must equal ref's byte for byte.
        tern is the same documents on stage 1 (b)'s tern-q4.gguf, the model whose experts are ternary.
    panel --name N --k K --tail-offset O   the documents and chat panels with the head in two passes, into stage 1's work
        directory.  N is a pinned preview: a file that differs from its sha stops the run.  --docs-limit 2 --panels docs
        are the controls' prefix runs.
    controls   C4 from the prefix arms: replay (K = the vocabulary) against ref's bytes, plant's top-1 against ref's.
Each process's CONFIG must confirm its cell, or the run stops.  Resumable: a set or panel already COMPLETE is skipped.

Usage: python -I benchmarks/stage2/s2t_run.py dump --set docs --out benchmarks/stage2/results/s2t_run.json
       python -I benchmarks/stage2/s2t_run.py panel --arm two_pass --name tern_tr --k 1024 --tail-offset 1 --out F
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
from s1_run import INPUTS, MAX_PROJECTED_S, STOP, check_inputs, sha256  # noqa: E402

WORK = Path("C:/SiliconLLM_fast/s2t")
P0 = WORK / "m" / "p0.gguf"
P0_SHA = "44de73157f11165c0551fd06714f4f043913292a79b07683e6a34b11d3b0d2f4"
TERN = Path("C:/SiliconLLM_fast/export/tern-q4.gguf")
REF = Path("E:/SiliconLLM_Artifacts/stage1/s1a/ref")
STAGE1_WORK = Path("E:/SiliconLLM_Artifacts/stage1/s1a")
V = 157184
PREFIX_DOCS, PLANT_K = 2, 1024                 # the controls' prefix and the planted arm's K
# a preview differs from P0 only in output.weight; the plant, test 1's shuffled head, also in output_norm.weight
PREVIEWS = {"tern_tr": (WORK / "m" / "org_head.gguf", "cba1156d07ffd2b4862d92154777b753ac268106166523130221eef22519ce8d"),
            "wsvd256": (WORK / "m" / "wsvd256-q4.gguf", "1d2ab05e164dbccd66f5b85766893255bf087ece8ee1a62a8a49373e4794ef73"),
            "plant": (Path("E:/SiliconLLM_Artifacts/stage2/s2h/models/head_shuf-q4.gguf"), "4cbbd37d21df00261c9a163cdcdbd896bc1f02765ec74d51f5f608cab12fb17f")}
POS_RECORD = 32                                # doc, pos, target, top1, lp_t, lp_1, lse
TOP_RECORD = 32 + 32 * 8                       # the head, then 32 entries of id and logp
SCORE = HERE / "bin" / "s2t_llama_score_avx2.exe"
GEN = HERE / "bin" / "s2t_llama_generate_avx2.exe"
SETS = {"fit": (WORK / "sets" / "fit.tsv", P0), "dev": (WORK / "sets" / "dev.tsv", P0),
        "gen": (WORK / "sets" / "gen.tsv", P0), "docs": (INPUTS["a4"] / "docs.tsv", P0),
        "tern": (INPUTS["a4"] / "docs.tsv", TERN)}


def launch(cmd: list[str], need: list[str]) -> dict:
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    lines = proc.stdout.splitlines()
    config = [ln for ln in lines if ln.startswith("CONFIG ")]
    ok = proc.returncode == 0 and len(config) == 1 and all(x in config[0] for x in need)
    return {"cmd": cmd, "exe_sha256": sha256(Path(cmd[0])), "returncode": proc.returncode, "wall_s": round(time.time() - t0, 1),
            "config": config, "two_pass": [ln for ln in lines if ln.startswith("TWOPASS ")], "stdout_tail": lines[-3:],
            "stderr_tail": proc.stderr[-600:], "status": "COMPLETE" if ok else "STOPPED"}


def load(out: Path) -> dict:
    return json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"cell": "S2T-two-pass-head", "dumps": {}, "arms": {}}


def save(out: Path, report: dict) -> None:
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


def dump(a: argparse.Namespace) -> int:
    report = load(a.out)
    if report["dumps"].get(a.set, {}).get("status") == "COMPLETE":
        print(a.set, "already COMPLETE")
        return 0
    ids, model = SETS[a.set]
    pre = WORK / "h" / a.set
    cmd = [str(SCORE), "--model", str(model), "--ids", str(ids), "--out", str(pre), "--threads", "6", "--ubatch", "512",
           "--ctx", "8192", "--top", "32", "--moe-mode", "none", "--dump-h", f"{pre}.h.f32"]
    rec = launch(cmd, ["two_pass_k=0", "top=32", "moe_mode=none", f"model={model}"])
    rec["ids_sha256"], rec["model_sha256"] = sha256(ids), sha256(model)
    if a.set == "docs" and rec["status"] == "COMPLETE":
        # the dump must not have touched the numbers: every record equals ref's
        rec["equals_ref"] = {x: sha256(Path(f"{pre}.{x}.bin")) == sha256(REF / f"docs.{x}.bin") for x in ("pos", "top")}
        if not all(rec["equals_ref"].values()):
            rec["status"] = "STOPPED: records differ from ref"
    report["dumps"][a.set] = rec
    save(a.out, report)
    print(a.set, rec["status"], rec["wall_s"], rec.get("equals_ref"))
    return 0 if rec["status"] == "COMPLETE" else 1


def subset(n: int) -> tuple[Path, Path]:
    """The panel's first n documents and ref's records of them: the controls run on a prefix."""
    ids, top = WORK / "sets" / f"docs{n}.tsv", WORK / "sets" / f"docs{n}.top.bin"
    lines = (INPUTS["a4"] / "docs.tsv").read_text(encoding="ascii").splitlines()[:n]
    ids.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    keep = sum(len(x.split("\t")[1].split(",")) - 1 for x in lines)
    top.write_bytes((REF / "docs.top.bin").read_bytes()[:keep * TOP_RECORD])
    return ids, top


def controls(a: argparse.Namespace) -> int:
    """C4, from the two prefix arms: the replay's records equal ref's byte for byte; the planted arm's top-1 falls."""
    report = load(a.out)
    lines = (INPUTS["a4"] / "docs.tsv").read_text(encoding="ascii").splitlines()[:PREFIX_DOCS]
    n = sum(len(x.split("\t")[1].split(",")) - 1 for x in lines)
    for name, k, preview in (("replay", V, "tern_tr"), ("plant", PLANT_K, "plant")):
        arm = report["arms"][name]
        if ((arm["k"], arm["docs_limit"], arm["preview_sha256"]) != (k, PREFIX_DOCS, PREVIEWS[preview][1])
                or arm["panels"]["docs"]["status"] != "COMPLETE"):
            raise SystemExit(f"{name}: not the registered control")
    out = {}
    for x, size in (("pos", POS_RECORD), ("top", TOP_RECORD)):
        got = (STAGE1_WORK / "replay" / f"docs.{x}.bin").read_bytes()
        out[f"replay_{x}_bytes"] = len(got)
        out[f"replay_{x}_equal"] = len(got) == n * size and got == (REF / f"docs.{x}.bin").read_bytes()[:len(got)]
    out["replay_equal"] = out["replay_pos_equal"] and out["replay_top_equal"]
    ref = (REF / "docs.pos.bin").read_bytes()
    got = (STAGE1_WORK / "plant" / "docs.pos.bin").read_bytes()
    if len(got) != n * POS_RECORD:
        raise SystemExit("plant: not the whole prefix")
    same = sum(got[i * POS_RECORD + 12:i * POS_RECORD + 16] == ref[i * POS_RECORD + 12:i * POS_RECORD + 16] for i in range(n))
    out["plant_positions"] = n
    out["plant_top1"] = same / max(n, 1)
    report["controls"] = out
    save(a.out, report)
    print(json.dumps(out))
    return 0


def panel(a: argparse.Namespace) -> int:
    report = load(a.out)
    man = {k: check_inputs(p) for k, p in INPUTS.items()}
    report.setdefault("inputs_manifest_sha256", man)
    if report["inputs_manifest_sha256"] != man:
        raise SystemExit("inputs changed since the run began")
    if sha256(P0) != P0_SHA:
        raise SystemExit("P0 differs from its pin")
    preview, pin = PREVIEWS[a.name]
    got = sha256(preview)
    if got != pin:
        raise SystemExit(f"{a.name}: the file differs from its pin")
    arm = report["arms"].setdefault(a.arm, {"panels": {}, "preview": str(preview), "preview_name": a.name, "preview_sha256": got,
                                            "k": a.k, "tail_offset": a.tail_offset, "docs_limit": a.docs_limit})
    if (arm["preview_sha256"], arm["k"], arm["tail_offset"], arm["docs_limit"]) != (got, a.k, a.tail_offset, a.docs_limit):
        raise SystemExit(f"{a.arm}: the cell differs from the one this record began")
    (STAGE1_WORK / a.arm).mkdir(parents=True, exist_ok=True)
    two = ["--preview", str(preview), "--two-pass-k", str(a.k), "--tail-offset", str(a.tail_offset)]
    need = [f"preview={preview}", f"two_pass_k={a.k}", f"tail_offset={a.tail_offset}", "moe_mode=none", f"model={P0}"]
    ids, ref_top = INPUTS["a4"] / "docs.tsv", REF / "docs.top.bin"
    if a.docs_limit:
        ids, ref_top = subset(a.docs_limit)
    for name, cmd in (
        ("docs", [str(SCORE), "--model", str(P0), "--ids", str(ids), "--ctx", "8192", "--top", "32",
                  "--ref-top", str(ref_top)]),
        ("chat", [str(GEN), "--model", str(P0), "--prompts", str(INPUTS["a4"] / "chat.tsv"), "--max-new", "256",
                  "--stop", STOP])):
        if name not in a.panels.split(",") or arm["panels"].get(name, {}).get("status") == "COMPLETE":
            continue
        cmd += ["--out", str(STAGE1_WORK / a.arm / name), "--threads", "6", "--ubatch", "512",
                "--max-projected-s", str(MAX_PROJECTED_S), "--moe-mode", "none"] + two
        arm["panels"][name] = launch(cmd, need)
        save(a.out, report)
        print(a.arm, name, arm["panels"][name]["status"], arm["panels"][name]["wall_s"], arm["panels"][name]["two_pass"])
        if arm["panels"][name]["status"] != "COMPLETE":
            return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("dump")
    d.add_argument("--set", required=True, choices=sorted(SETS))
    d.add_argument("--out", type=Path, required=True)
    d.set_defaults(fn=dump)
    p = sp.add_parser("panel")
    p.add_argument("--arm", default="two_pass")
    p.add_argument("--name", required=True, choices=sorted(PREVIEWS))
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--tail-offset", type=int, required=True)
    p.add_argument("--panels", default="docs,chat")
    p.add_argument("--docs-limit", type=int, default=0, help="only the panel's first N documents: the controls' prefix")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=panel)
    c = sp.add_parser("controls")
    c.add_argument("--out", type=Path, required=True)
    c.set_defaults(fn=controls)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
