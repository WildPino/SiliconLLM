#!/usr/bin/env python3
"""
phase56_parallel.py — Run MQAR benchmark arms in parallel across available GPUs.
Each arm gets its own GPU via CUDA_VISIBLE_DEVICES and its own output directory.
Output is streamed live with [arm] prefixes. At the end, results are merged.

Multi-pairs mode: --pairs 32,64 generates cross-product of arms × pairs values.
Each job gets --pairs P --queries P --topk P and out-dir <out>/sz<P>/<arm>.
With 2 GPUs and 4 jobs, 2 run in parallel; when one finishes, the next starts.

Usage (scale-up: sparse+simvq at pairs 32 and 64, 2×T4):
    !python phase56_parallel.py \
        --arm sparse,simvq --pairs 32,64 --ctx 384 --dim 128 --layers 3 --heads 4 \
        --vatoms 256 --nprobe 4 --batch 32 --steps 24000 --lr 3e-3 --bf16 \
        --no-ckpt --scan-chunk 128 --eval-every 250 --grok-thr 0.6 --patience 6 \
        --ckpt-every 1000 --save-models --time-budget-min 330 \
        --infonce-w 0.05 --out-dir /kaggle/working/phase56_scale

Usage (single pairs, backward compatible):
    !python phase56_parallel.py \
        --arm ramp --ctx 384 --pairs 16 --queries 16 --dim 128 --layers 3 --heads 4 \
        --batch 32 --steps 12000 --lr 3e-3 --bf16 --no-ckpt --scan-chunk 128 \
        --out-dir /kaggle/working/phase56_p16 --save-models --time-budget-min 330

Resume from flat folder (download all checkpoints from Kaggle into one dir):
    !python phase56_parallel.py \
        --arm sparse,simvq --pairs 32,64 ... \
        --resume /kaggle/working/resume_ckpts \
        --out-dir /kaggle/working/phase56_scale
"""
import subprocess
import sys
import os
import time
import threading
import queue
import json

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase56_mqar.py")


def merge_and_save_results(out_dir_val, job_list):
    """Merge results from all jobs. job_list = [(arm, P, job_out_dir), ...]"""
    master_results = {}
    config = {}
    for arm, P, job_out in job_list:
        arm_json_path = os.path.join(job_out, "mqar_results.json")
        if os.path.exists(arm_json_path):
            try:
                with open(arm_json_path, 'r') as f:
                    data = json.load(f)
                if "results" in data:
                    for k, v in data["results"].items():
                        # Tag with pairs for multi-pairs runs
                        label = f"{k}/p{P}" if len(set(p for _, p, _ in job_list)) > 1 else k
                        master_results[label] = v
                if "config" in data and not config:
                    config = data["config"]
            except Exception as e:
                print(f"Error reading {arm_json_path}: {e}", flush=True)

    if not master_results:
        print("\nNo results found to merge.", flush=True)
        return

    # BINS definition matching phase56_mqar.py
    BINS = [(0, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 1 << 20)]

    # Determine which bins have data
    active = []
    for bi in range(len(BINS)):
        has_data = False
        for arm in master_results:
            tot_dict = master_results[arm].get("tot", {})
            val = tot_dict.get(str(bi), tot_dict.get(bi, 0))
            if val > 0:
                has_data = True
                break
        if has_data:
            active.append(bi)

    lines = [
        "==== RECALL @ key->query distance (accuracy) ====",
        "  arm          params  steps overall "
        + " ".join(f"{BINS[bi][0]}-{BINS[bi][1] if BINS[bi][1]<(1<<20) else 'inf'}".rjust(10) for bi in active),
    ]
    for arm in master_results:
        r = master_results[arm]
        cells = []
        for bi in active:
            hit_dict = r.get("hit", {})
            tot_dict = r.get("tot", {})
            h_val = hit_dict.get(str(bi), hit_dict.get(bi, 0))
            t_val = tot_dict.get(str(bi), tot_dict.get(bi, 0))
            cells.append((f"{h_val/t_val*100:.0f}%" if t_val > 0 else "-").rjust(10))
        cells_str = " ".join(cells)
        lines.append(f"  {arm:12s} {r.get('npar',0)/1e6:5.3f}M {r.get('steps',0):5d} {r.get('ov',0)*100:5.1f}%  {cells_str}")
    txt = "\n".join(lines)

    # Write merged files at the parent out_dir_val
    os.makedirs(out_dir_val, exist_ok=True)
    txt_path = os.path.join(out_dir_val, "mqar_recall.txt")
    json_path = os.path.join(out_dir_val, "mqar_results.json")

    with open(txt_path, "w") as f:
        f.write(txt)

    # Ensure we format keys consistently for the final json
    formatted_results = {}
    for arm, r in master_results.items():
        formatted_results[arm] = {
            "npar": r.get("npar"),
            "steps": r.get("steps"),
            "ov": r.get("ov"),
            "hit": {str(k): v for k, v in r.get("hit", {}).items()},
            "tot": {str(k): v for k, v in r.get("tot", {}).items()},
        }

    with open(json_path, "w") as f:
        json.dump({"config": config, "results": formatted_results}, f, indent=1)

    print("\nMerged results:", flush=True)
    print(txt, flush=True)
    print(f"\n[Saved merged outputs to {txt_path} and {json_path}]", flush=True)


def _job_key(arm, pairs):
    """Unique key for a (arm, pairs) job — used in resume_map."""
    return f"{arm}@{pairs}"


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Parallel runner for phase56_mqar.py. Distributed across available GPUs.")
        print("Usage:")
        print("  python phase56_parallel.py [any phase56_mqar.py arguments]")
        print("\nMulti-pairs: --pairs 32,64 generates cross-product of arms × pairs.")
        print("  Each job gets --pairs P --queries P --topk P, out-dir <out>/sz<P>/<arm>.")
        print("\nMulti-arm: --arm sparse,simvq (comma-separated list of arms).")
        print("\nUnderlying script help:")
        if os.path.exists(SCRIPT):
            subprocess.run([sys.executable, SCRIPT, "--help"])
        else:
            print(f"Error: {SCRIPT} not found.")
        sys.exit(0)

    # Parse args: extract --arm, --out-dir, --resume, --pairs, --queries, --topk
    # These are handled by the runner (not forwarded). Everything else is forwarded.
    args = sys.argv[1:]
    arm_val = "all"
    out_dir_val = "results/phase56_parallel"
    resume_dir = ""
    pairs_str = ""        # will be extracted; if empty, forwarded from forward_args
    queries_str = ""
    topk_str = ""
    steps_val = 24000     # default; extracted for pre-filter (skip only if ck.step >= this)

    forward_args = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--arm":
            arm_val = args[i + 1]; i += 2
        elif arg.startswith("--arm="):
            arm_val = arg.split("=", 1)[1]; i += 1
        elif arg == "--out-dir":
            out_dir_val = args[i + 1]; i += 2
        elif arg.startswith("--out-dir="):
            out_dir_val = arg.split("=", 1)[1]; i += 1
        elif arg == "--resume":
            resume_dir = args[i + 1]; i += 2
        elif arg.startswith("--resume="):
            resume_dir = arg.split("=", 1)[1]; i += 1
        elif arg == "--pairs":
            pairs_str = args[i + 1]; i += 2
        elif arg.startswith("--pairs="):
            pairs_str = arg.split("=", 1)[1]; i += 1
        elif arg == "--queries":
            queries_str = args[i + 1]; i += 2
        elif arg.startswith("--queries="):
            queries_str = arg.split("=", 1)[1]; i += 1
        elif arg == "--topk":
            topk_str = args[i + 1]; i += 2
        elif arg.startswith("--topk="):
            topk_str = arg.split("=", 1)[1]; i += 1
        elif arg == "--steps":
            steps_val = int(args[i + 1]); forward_args += [arg, args[i + 1]]; i += 2
        elif arg.startswith("--steps="):
            steps_val = int(arg.split("=", 1)[1]); forward_args.append(arg); i += 1
        else:
            forward_args.append(arg); i += 1

    # --- Expand arms ---
    ARM_EXPANSIONS = {
        "all":       ["ssm", "swa", "sparse", "attn"],
        "simvq-all": ["simvq", "simvq-kmeans", "simvq-random"],
        "ramp":      ["sparse", "simvq", "simvq-random"],
    }
    # Support comma-separated arm list: --arm sparse,simvq
    if arm_val in ARM_EXPANSIONS:
        arms_list = ARM_EXPANSIONS[arm_val]
    elif "," in arm_val:
        arms_list = [a.strip() for a in arm_val.split(",")]
    else:
        arms_list = [arm_val]

    # --- Parse pairs (may be comma-separated: "32,64") ---
    if "," in pairs_str:
        pairs_list = [int(p.strip()) for p in pairs_str.split(",")]
        multi_pairs = True
    elif pairs_str:
        pairs_list = [int(pairs_str)]
        multi_pairs = False
    else:
        pairs_list = [None]   # None = don't override, use whatever is in forward_args
        multi_pairs = False

    # --- Generate job list: cross-product of arms × pairs ---
    # Each job = (arm_name, pairs_value, job_out_dir)
    job_list = []
    for P in pairs_list:
        for arm in arms_list:
            if multi_pairs or len(pairs_list) > 1:
                job_out = os.path.join(out_dir_val, f"sz{P}", arm)
            elif P is not None:
                job_out = os.path.join(out_dir_val, arm)
            else:
                job_out = os.path.join(out_dir_val, arm)
            job_list.append((arm, P, job_out))

    # --- Build resume_map: scan --resume dir for .pt files, index by (arm, pairs) ---
    # Supports FLAT folder (all checkpoints dumped together) and structured subdirs
    resume_map = {}   # _job_key(arm, pairs) -> path to checkpoint file
    if resume_dir and os.path.isdir(resume_dir):
        import torch as _t

        def _scan_pt(fpath, label=""):
            try:
                ck = _t.load(fpath, map_location='cpu')
                if isinstance(ck, dict) and 'arm' in ck:
                    ck_arm = ck['arm']
                    ck_pairs = ck.get('cfg', {}).get('pairs', None)
                    key = _job_key(ck_arm, ck_pairs)
                    if key not in resume_map:
                        resume_map[key] = fpath
                        print(f"  [resume-scan] {label}{os.path.basename(fpath)} -> arm='{ck_arm}' pairs={ck_pairs} step={ck.get('step')} done={ck.get('done')}", flush=True)
            except Exception as e:
                print(f"  [resume-scan] skip {label}{os.path.basename(fpath)}: {e}", flush=True)

        # 1) Top-level .pt files (flat folder)
        for fname in os.listdir(resume_dir):
            if fname.endswith('.pt'):
                _scan_pt(os.path.join(resume_dir, fname))
        # 2) Subdirs (structured: <resume>/sz32/simvq/ckpt_simvq.pt or <resume>/simvq/ckpt_simvq.pt)
        for d1 in os.listdir(resume_dir):
            d1path = os.path.join(resume_dir, d1)
            if not os.path.isdir(d1path):
                continue
            for item in os.listdir(d1path):
                item_path = os.path.join(d1path, item)
                if item.endswith('.pt') and os.path.isfile(item_path):
                    _scan_pt(item_path, f"{d1}/")
                elif os.path.isdir(item_path):
                    for fname in os.listdir(item_path):
                        if fname.endswith('.pt'):
                            _scan_pt(os.path.join(item_path, fname), f"{d1}/{item}/")
        if resume_map:
            print(f"  [resume] found {len(resume_map)} checkpoint(s): {list(resume_map.keys())}", flush=True)
        else:
            print(f"  [resume] WARNING: no valid checkpoints found in {resume_dir}", flush=True)

    # --- Detect GPUs ---
    try:
        import torch
        num_gpus = torch.cuda.device_count()
    except ImportError:
        num_gpus = 0
    gpus = [str(g) for g in range(num_gpus)] if num_gpus > 0 else ["cpu"]

    t0 = time.time()
    print(f"Parallel runner | SCRIPT={SCRIPT}", flush=True)
    print(f"GPUs detected: {gpus}", flush=True)
    print(f"Jobs ({len(job_list)} total):", flush=True)
    for arm, P, jout in job_list:
        print(f"  {arm} pairs={P} -> {jout}", flush=True)
    print(f"Forwarded args: {forward_args}", flush=True)
    if resume_dir:
        print(f"Resume from: {resume_dir} ({len(resume_map)} checkpoints found)", flush=True)

    # --- Pre-filter: skip jobs whose checkpoint is marked DONE ---
    jobs_pending = []
    for arm, P, job_out in job_list:
        key = _job_key(arm, P)
        ckpt_path = resume_map.get(key)
        if ckpt_path:
            try:
                import torch as _t
                ck = _t.load(ckpt_path, map_location="cpu")
                if ck.get('done') and ck.get('results') is not None and ck.get('step', 0) >= steps_val:
                    print(f"[{arm}/p{P}] Already CONVERGED (step {ck['step']}, recall {ck['results']['ov']*100:.1f}%) -> skip", flush=True)
                    continue
            except Exception as e:
                print(f"[{arm}/p{P}] Could not read checkpoint ({e}) -> will run", flush=True)
        jobs_pending.append((arm, P, job_out))

    if not jobs_pending:
        print("\nAll jobs already converged! Nothing to run.", flush=True)
    else:
        print(f"\nJobs to launch ({len(jobs_pending)}):", flush=True)
        for arm, P, jout in jobs_pending:
            print(f"  {arm} pairs={P} -> {jout}", flush=True)

    # --- Job queue ---
    job_queue = queue.Queue()
    for job in jobs_pending:
        job_queue.put(job)

    threads = []

    def worker(gpu_id):
        while True:
            try:
                arm, P, job_out = job_queue.get_nowait()
            except queue.Empty:
                break

            os.makedirs(job_out, exist_ok=True)
            job_label = f"{arm}/p{P}" if P is not None else arm

            # Build command
            arm_cmd = [sys.executable, SCRIPT, "--arm", arm, "--out-dir", job_out]
            # Inject per-job --pairs, --queries, --topk
            if P is not None:
                arm_cmd += ["--pairs", str(P), "--queries", str(P), "--topk", str(P)]
            elif pairs_str:
                arm_cmd += ["--pairs", pairs_str]
                if queries_str:
                    arm_cmd += ["--queries", queries_str]
                if topk_str:
                    arm_cmd += ["--topk", topk_str]

            # Resume: copy checkpoint into job_out so child finds it
            key = _job_key(arm, P)
            ckpt_src = resume_map.get(key)
            if ckpt_src:
                import shutil
                ckpt_dst = os.path.join(job_out, os.path.basename(ckpt_src))
                if os.path.abspath(ckpt_src) != os.path.abspath(ckpt_dst):
                    shutil.copy2(ckpt_src, ckpt_dst)
                    print(f"[{job_label}] Copied checkpoint -> {job_out}", flush=True)
                arm_cmd += ["--resume", job_out]

            cmd = arm_cmd + forward_args

            env = os.environ.copy()
            if gpu_id != "cpu":
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

            print(f"\n[{job_label}] Launching on GPU {gpu_id} -> {job_out}\nCommand: {' '.join(cmd)}\n", flush=True)
            t_launch = time.time()
            p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            for line in iter(p.stdout.readline, b""):
                print(f"[{job_label}] {line.decode().rstrip()}", flush=True)

            p.wait()
            elapsed = (time.time() - t_launch) / 60
            status = "OK" if p.returncode == 0 else f"FAIL(rc={p.returncode})"
            print(f"[{job_label}] DONE {status} (elapsed {elapsed:.1f} min)", flush=True)

    # Start threads (one per GPU) — each thread pulls jobs from the queue until empty
    for gpu in gpus:
        t = threading.Thread(target=worker, args=(gpu,), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"\nAll jobs finished. Total time: {(time.time()-t0)/60:.1f} min", flush=True)

    # Merge results
    merge_and_save_results(out_dir_val, job_list)


if __name__ == "__main__":
    main()
