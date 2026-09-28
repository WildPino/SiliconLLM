#!/usr/bin/env python3
"""Gate native shortlisted-head choices and matched greedy CPU timing."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling"
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
CORE = ART / "meth127_qwen05b_instruct_fp32.bin"
BANK = ART / "meth126_shared_a_factor_bank.bin"
HEAD = ART / "meth129_r8_head_sidecar.bin"
IDS64 = ART / "meth127_parity64_ids.bin"
PREFIX = ART / "meth127_parity_ids.bin"
BASE_LOGITS = ART / "meth127_centered64_logits.bin"
HASHES = {
    CORE: "6b2be143303510f15785783542649026b719488407f48b350de67f429e206029",
    BANK: "1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1",
    HEAD: "5df1038b7224cdc44b9c62686780e08162c9e5da1ab8c998d16d999ade74dd43",
    IDS64: "990924bdcf23204fc28ac091e960834dc775efc09cd6ce9a38f73f307ee13db7",
    PREFIX: "b9ffb0827e01fec4ef13027d46e04a61421d2ef62c571cc3827aa049bd3013ed",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(exe, args, log, *, expected=0):
    cmd = [str(exe), "--weights", str(CORE), "--factor-bank", str(BANK), *map(str, args)]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=1200)
    log.write_text(p.stdout, encoding="utf-8")
    if expected == 0:
        assert p.returncode == 0, (p.returncode, str(log), p.stdout[-1000:])
    else:
        assert p.returncode != 0, (p.returncode, str(log))
    return p.stdout


def number(out, key):
    m = re.search(rf"(?m)^{re.escape(key)} ([0-9.]+)$", out)
    assert m, (key, out[-1000:])
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", type=Path, default=ART / "meth129_donor_engine.exe")
    ap.add_argument("--out", type=Path, default=DOC / "meth129_native_head_result.json")
    args = ap.parse_args()
    start = time.monotonic()
    exe = args.exe.resolve()
    for path, want in HASHES.items():
        assert sha(path) == want, path
    report = {"experiment": "METH-129-native-exact-head", "artifact_sha256": {
        path.name: want for path, want in HASHES.items()}, "executable_sha256": sha(exe)}

    bad = ART / "meth129_bad_head_magic.bin"
    shutil.copyfile(HEAD, bad)
    with bad.open("r+b") as f:
        first = f.read(1)
        f.seek(0)
        f.write(bytes([first[0] ^ 1]))
    negative = run(exe, ["--head-shortlist", bad, "--threads", "6", "--head-choice",
                         IDS64, "64", ART / "meth129_bad_choices.bin"],
                   ART / "meth129_bad_head.log", expected=1)
    assert "head shortlist magic or dimensions mismatch" in negative
    report["corrupted_magic_rejected"] = True

    chosen_path = ART / "meth129_choices64.bin"
    choice_log = ART / "meth129_choice64.log"
    choice = run(exe, ["--head-shortlist", HEAD, "--threads", "6", "--head-choice",
                       IDS64, "64", chosen_path], choice_log)
    exact = np.memmap(BASE_LOGITS, mode="r", dtype="<f4", shape=(64, 151936))
    reference = exact.argmax(axis=1).astype("<i4")
    chosen = np.fromfile(chosen_path, dtype="<i4")
    assert chosen.shape == reference.shape
    mismatches = np.flatnonzero(chosen != reference).tolist()
    report["prompt_positions"] = 64
    report["prompt_choice_mismatches"] = mismatches
    report["prompt_choice_sha256"] = sha(chosen_path)
    report["prompt_peak_rss_bytes"] = int(number(choice, "PEAK_RSS_BYTES"))
    assert not mismatches

    runs = []
    for threads in (1, 6):
        for two_pass in (False, True):
            tag = f"meth129_{'two_pass' if two_pass else 'full'}_t{threads}"
            prefix = ART / tag
            flags = ["--threads", str(threads), "--profile", "--generate", PREFIX, "64", prefix]
            if two_pass:
                flags = ["--head-shortlist", HEAD, *flags]
            out = run(exe, flags, ART / f"{tag}.log")
            path = ART / f"{tag}.ids.bin"
            ids = np.fromfile(path, dtype="<i4")
            ref = np.fromfile(ART / f"meth127_centered_t{threads}_gen.ids.bin", dtype="<i4")
            assert ids.shape == (72,) and ref.shape == (72,)
            mismatch = np.flatnonzero(ids != ref).tolist()
            assert not mismatch, (tag, mismatch)
            assert 151645 not in ids[8:], (tag, "early EOS")
            organs = {name: number(out, f"GEN_ORGAN_{name}_MS_PER_TOKEN")
                      for name in ("qkv_proj", "rope", "attention", "o_proj", "ffn", "head", "norm+glue")}
            runs.append({"name": tag, "threads": threads, "two_pass": two_pass,
                         "decode_tokens_per_second": number(out, "GEN_DECODE_TOKS"),
                         "decode_seconds": number(out, "GEN_DECODE_S"),
                         "peak_rss_bytes": int(number(out, "GEN_PEAK_RSS_BYTES")),
                         "organ_ms_per_token": organs,
                         "ids_sha256": sha(path), "mismatches_vs_m127": mismatch})
            print(tag, runs[-1]["decode_tokens_per_second"], "tok/s", flush=True)
    report["runs"] = runs
    baseline = next(x for x in runs if x["threads"] == 6 and not x["two_pass"])
    variant = next(x for x in runs if x["threads"] == 6 and x["two_pass"])
    report["six_thread_speedup_fraction"] = variant["decode_tokens_per_second"] / baseline["decode_tokens_per_second"] - 1
    report["five_percent_speed_gate"] = report["six_thread_speedup_fraction"] >= .05
    report["sidecar_bytes"] = HEAD.stat().st_size
    report["nominal_head_active_bytes"] = 151936 * 896 + 151936 * 4 + 64 * 896 * 4
    report["original_full_head_active_bytes"] = 151936 * 896 * 4
    report["elapsed_seconds"] = time.monotonic() - start
    assert report["elapsed_seconds"] <= 1200
    assert max(x["peak_rss_bytes"] for x in runs) <= 16 * (1 << 30)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"six_thread_speedup_fraction": report["six_thread_speedup_fraction"],
                      "five_percent_speed_gate": report["five_percent_speed_gate"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
