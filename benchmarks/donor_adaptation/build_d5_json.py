#!/usr/bin/env python3
"""
build_d5_json.py -- turn a `gemv_donor_bench d5` run's stdout log into the D5 deliverable JSON.

Why this exists: BRIEF_D5_LUT_RATE_AT_DONOR_WIDTH.md sec.7 wants
`benchmarks/donor_adaptation/results/d5_lut_rate_donor_width.json` with the rate(D,threads) table, working
set + l3_resident per point, all four control outcomes, and a reproducibility manifest. The C harness prints
its numbers as `D5CSV,...` / `D5VERDICT,...` / `D5NOTE,...` / `D5SUMMARY,...` lines (documented in
gemv_donor_bench.c mode_d5) rather than JSON directly, so hand-transcription is never needed -- this script
is the one and only place that number goes from stdout text to the JSON file. Run it on the log AFTER a
real (uncontended) `d5` run; it does no measurement of its own.

Usage:
    bin\\gemv_donor_bench.exe d5 --reps 5 > benchmarks/donor_adaptation/results/d5_raw.log 2>&1
    python benchmarks/donor_adaptation/build_d5_json.py benchmarks/donor_adaptation/results/d5_raw.log \
        --out benchmarks/donor_adaptation/results/d5_lut_rate_donor_width.json \
        --git-rev <rev> --build-cmd "<exact clang command>" --machine "AMD Ryzen 5 3600X, 80 GB DDR4, Windows 11"
"""
import argparse, json, re, subprocess, sys, datetime, os

CSV_RE = re.compile(r"^D5CSV,(.*)$")
VERDICT_RE = re.compile(r"^D5VERDICT,(\w+),(PASS|FAIL),(.*)$")
NOTE_RE = re.compile(r"^D5NOTE,(\w+),(.*)$")
SUMMARY_RE = re.compile(r"^D5SUMMARY,controls,(.*)$")
STOP_RE = re.compile(r"^D5STOP: (.*)$")
QUIET_HDR_RE = re.compile(r"^# QUIESCENCE CHECK: threshold=([\d.]+) GB resident working set \| (\d+) process")
QUIET_PROC_RE = re.compile(r"^#\s+pid=(\d+)\s+ws=\s*([\d.]+) GB\s+(\S+)")
MODE_HDR_RE = re.compile(r"^# gemv_donor_bench \| mode=(\S+) reps=(\d+) check=(\d+) gather=(\S+) force_unclean=(\d+)")
OMP_HDR_RE = re.compile(r"^# OpenMP ON, max_threads=(\d+) \| OMP_PROC_BIND=(\S+) OMP_PLACES=(\S+)")


def kv_parse(s):
    """Parse a comma-separated k=v list where values may themselves contain '=' (e.g. dev=+0.12)."""
    out = {}
    for part in s.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


L3_BYTES = 16 * 1048576  # the project's banked L3 cliff, ~16 MiB exact -- matches L3_BYTES in gemv_donor_bench.c


def ternary_block_bytes(M, K):
    """Mirrors gemv_donor_bench.c: Mpad=(M+31)&~31, T=K/2, EB=T*Mpad. Pure arithmetic, no measurement."""
    Mpad = (M + 31) & ~31
    T = K // 2
    return T * Mpad


LLAMA70B_CLASS_SHAPES = [
    ("llama70b_qo", 8192, 8192),
    ("llama70b_kv", 1024, 8192),
    ("llama70b_gateup", 28672, 8192),
    ("llama70b_down", 8192, 28672),
]


def shape_arithmetic_llama70b():
    """Derived, NOT measured: whether a single ternary-packed block of each Llama-3-70B-class organ fits in
    the banked 16 MiB L3 cliff on its own. True regardless of whether any timing loop ever ran -- this is
    why it is computed here directly rather than only parsed from a (possibly absent/truncated) log."""
    out = []
    for name, M, K in LLAMA70B_CLASS_SHAPES:
        eb = ternary_block_bytes(M, K)
        out.append({
            "organ": name, "M": M, "K": K,
            "single_block_bytes": eb, "single_block_MB": round(eb / 1048576.0, 2),
            "l3_bytes": L3_BYTES,
            "l3_resident_possible": eb <= L3_BYTES,
        })
    return out


def num(x):
    try:
        if x is None:
            return None
        if x.endswith("x") or x.endswith("%"):
            x = x[:-1]
        return float(x)
    except (ValueError, AttributeError):
        return x


def parse_log(lines):
    rows = []          # raw D5CSV rows, section-tagged
    verdicts = {}
    notes = []
    summary = None
    stops = []
    header = {}
    quiescence = {"threshold_gb": None, "heavy_processes": []}

    for line in lines:
        line = line.rstrip("\n")
        m = MODE_HDR_RE.match(line)
        if m:
            header["mode"], header["reps"], header["check"], header["gather"], header["force_unclean"] = m.groups()
            continue
        m = OMP_HDR_RE.match(line)
        if m:
            header["max_threads"], header["omp_proc_bind"], header["omp_places"] = m.groups()
            continue
        m = QUIET_HDR_RE.match(line)
        if m:
            quiescence["threshold_gb"] = float(m.group(1))
            quiescence["n_heavy"] = int(m.group(2))
            continue
        m = QUIET_PROC_RE.match(line)
        if m:
            quiescence["heavy_processes"].append(
                {"pid": int(m.group(1)), "working_set_gb": float(m.group(2)), "name": m.group(3)}
            )
            continue
        m = CSV_RE.match(line)
        if m:
            fields = m.group(1).split(",")
            rows.append(fields)
            continue
        m = VERDICT_RE.match(line)
        if m:
            control, verdict, rest = m.groups()
            verdicts[control] = {"verdict": verdict, **{k: num(v) for k, v in kv_parse(rest).items()}}
            continue
        m = NOTE_RE.match(line)
        if m:
            section, rest = m.groups()
            notes.append({"section": section, **{k: num(v) for k, v in kv_parse(rest).items()}})
            continue
        m = SUMMARY_RE.match(line)
        if m:
            summary = {k: v for k, v in kv_parse(m.group(1)).items()}
            continue
        m = STOP_RE.match(line)
        if m:
            stops.append(m.group(1))
            continue

    return header, quiescence, rows, verdicts, notes, summary, stops


def csv_row_to_dict(fields):
    """Column layout depends on section (documented in gemv_donor_bench.c mode_d5):
    mlp_point sections (control1, lsens, dsweep):
        section,tag,D,HID,L,threads,working_set_bytes,l3_resident,gbps_codes,us_per_tok,cv_pct,gbps_codes_plus_scales
    lut_one_point sections (control23, organs):
        section,tag,M,K,0,threads,working_set_bytes,l3_resident,gbps,us_per_touch,cv_pct
    control4 (fp32):
        control4,tag,rows,in,0,threads,working_set_bytes,l3_resident,gbps,us,cv_pct
    """
    section = fields[0]
    d = {
        "section": section,
        "tag": fields[1],
        "dim1": int(fields[2]),
        "dim2": int(fields[3]),
        "L_or_zero": int(fields[4]),
        "threads": int(fields[5]),
        "working_set_bytes": int(fields[6]),
        "l3_resident": bool(int(fields[7])),
        "gbps": float(fields[8]),
        "time_us": float(fields[9]),
        "cv_pct": float(fields[10]),
    }
    if len(fields) > 11:
        d["gbps_codes_plus_scales"] = float(fields[11])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", help="stdout+stderr captured from `gemv_donor_bench.exe d5 ...`")
    ap.add_argument("--out", required=True)
    ap.add_argument("--git-rev", default=None, help="if omitted, read from `git rev-parse HEAD` at run time")
    ap.add_argument("--build-cmd", required=True)
    ap.add_argument("--run-cmd", required=True)
    ap.add_argument("--machine", default="AMD Ryzen 5 3600X, 6c/12t, 80 GB DDR4, Windows 11")
    ap.add_argument("--compiler", default="clang 21.1.8 (x86_64-w64-windows-gnu)")
    args = ap.parse_args()

    with open(args.logfile, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    header, quiescence, rows, verdicts, notes, summary, stops = parse_log(lines)

    git_rev = args.git_rev
    if git_rev is None:
        try:
            git_rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__) or ".").decode().strip()
        except Exception:
            git_rev = "UNKNOWN"

    parsed_rows = [csv_row_to_dict(r) for r in rows]

    def section_rows(name):
        return [r for r in parsed_rows if r["section"] == name]

    out = {
        "brief": "docs/research/donor_adaptation/briefs/BRIEF_D5_LUT_RATE_AT_DONOR_WIDTH.md",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "manifest": {
            "git_revision": git_rev,
            "build_command": args.build_cmd,
            "run_command": args.run_cmd,
            "compiler": args.compiler,
            "machine": args.machine,
            "harness_header": header,
            "quiescence_check": quiescence,
            "note": "quiescence_check.n_heavy must be 0 (or force_unclean=1, which invalidates the run's "
                    "GB/s numbers as evidence of the kernel) for any number below to be trustworthy.",
        },
        "controls": {
            "control1_reproduce_banked_D1536": verdicts.get("control1"),
            "control2_forced_slowdown": verdicts.get("control2"),
            "control3_forced_speedup": verdicts.get("control3"),
            "control4_fp32_crosscheck": verdicts.get("control4"),
            "summary": summary,
            "stop_messages": stops,
        },
        "l_sensitivity_check": {
            "description": "Reported result, not merely a justification for a shortcut: L=2 vs L=28 at the "
                            "real donor shape (D=1536, HID=8960, skip=0, t6). Used to license L=2 in the "
                            "D-sweep below -- valid if the ratio is ~1, because every layer's working set "
                            "already exceeds L3 regardless of L (20.6 MB/layer at this shape).",
            "points": [n for n in notes if n["section"] == "lsensitivity"],
        },
        "rate_curve_D_threads": {
            "description": "engine-integrated dense ternary MLP, ffn:D ratio fixed at 3.5 (Llama-3-70B-class), "
                            "L=2 (see l_sensitivity_check for why L=2 is valid here). This is the INTEGRATED "
                            "rate (quant + LUT build + matvecs + per-row fp32 scales + dReLU) -- the same "
                            "quantity as the banked 21.25 GB/s at D=1536, NOT the same quantity as the "
                            "kernel-pure numbers in fp32_vs_ternary_matched_shapes below.",
            "points": section_rows("dsweep"),
            "notes": [n for n in notes if n["section"] == "dsweep"],
        },
        "fp32_vs_ternary_matched_shapes": {
            "description": "THE compute-bound-vs-bandwidth-bound discriminator. Kernel-pure ternary LUT rate "
                            "(matvec_lut_full only) vs kernel-pure fp32 streamed rate (honest matvec only), "
                            "at the IDENTICAL (HID x D) gate/up shape for every D in the sweep -- no "
                            "integration-overhead confound on either arm. Read: ratio_fp32_over_ternary > 1 "
                            "means fp32 DRAM streaming outruns the ternary kernel even fully streamed, i.e. "
                            "the LUT path is still compute-bound at donor width and denser packing "
                            "(ADAPTER_MEMO_01 sec.2.4) would NOT buy bandwidth headroom. <= 1 means the LUT "
                            "path has become bandwidth-bound and denser packing becomes the ~2.5x lever. "
                            "ternary_integrated_t6_forcontext in each note is the Part-A integrated number "
                            "at that D, carried through for reference ONLY -- it is a different quantity "
                            "(see rate_curve_D_threads.description) and is NOT part of the ratio.",
            "notes": [n for n in notes if n["section"] == "fp32_vs_ternary"],
            "raw_points": section_rows("fp32_vs_ternary"),
        },
        "control1_raw_points": section_rows("control1"),
        "control23_raw_points": section_rows("control23"),
        "control4_raw_points": section_rows("control4"),
        "organs_llama70b_class": {
            "description": "d_model=8192 d_ffn=28672: q/o 8192x8192, k/v 1024x8192, gate/up 28672x8192, "
                            "down 8192x28672. Kernel-pure ternary LUT rate (matvec_lut_full only, no engine "
                            "integration overhead) vs working-set size.",
            "points": section_rows("organs"),
            "notes_from_run": [n for n in notes if n["section"] == "shape_arithmetic"],
        },
        "shape_arithmetic_llama70b_class": {
            "description": "DERIVED, NOT MEASURED -- pure byte arithmetic on the ternary block-packing "
                            "formula, true independent of any timing run. At Llama-3-70B-class organ shapes, "
                            "a single ternary-packed block of gate/up or down is 112 MB and q/o is 32 MB -- "
                            "each EXCEEDS the 16 MiB L3 cliff on its own, so those organs are always in the "
                            "streamed regime at donor scale (l3_resident_possible=false). Only k/v at 4 MB/"
                            "block can ever be L3-resident. This establishes the data must come from DRAM; "
                            "it does NOT by itself establish the kernel is bandwidth-bound -- see "
                            "fp32_vs_ternary_matched_shapes for that question.",
            "shapes": shape_arithmetic_llama70b(),
        },
        "donor_budget_curve_l3_resident_points_excluded": [
            r for r in (section_rows("dsweep") + section_rows("organs")) if r["l3_resident"]
        ],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")
    print(f"controls: {[ (k, v['verdict'] if v else None) for k,v in [('c1',verdicts.get('control1')),('c2',verdicts.get('control2')),('c3',verdicts.get('control3')),('c4',verdicts.get('control4'))] ]}")
    if stops:
        print("STOP messages present in log:")
        for s in stops:
            print("  " + s)


if __name__ == "__main__":
    main()
