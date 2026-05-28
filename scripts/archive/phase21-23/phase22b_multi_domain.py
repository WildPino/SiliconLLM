import os
import subprocess
import csv

DATA_DIR = "data"
RESULTS_DIR = "results"

datasets = {
    "c_code": os.path.join(DATA_DIR, "c_code.c"),
    "prose": os.path.join(DATA_DIR, "promessi_sposi.txt"),
    "markdown": os.path.join(DATA_DIR, "markdown_docs.md"),
    "shuffled": os.path.join(DATA_DIR, "shuffled.bin") # Shuffled might not exist yet if audit_compression makes it on the fly, but wait, audit_compression runs with `--shuffled` flag on c_code!
}

def create_multi_domain(files_order, out_name):
    print(f"Creating {out_name}...")
    offsets = []
    current_offset = 0
    out_path = os.path.join(DATA_DIR, out_name)
    
    with open(out_path, "wb") as f_out:
        for name in files_order:
            if name == "shuffled":
                # We need to generate shuffled data explicitly since it's just a file now
                # Let's shuffle c_code
                with open(datasets["c_code"], "rb") as f_in:
                    data = bytearray(f_in.read())
                import random
                random.seed(42)
                for i in range(len(data)-1, 0, -1):
                    j = random.randint(0, i)
                    data[i], data[j] = data[j], data[i]
                
                start = current_offset
                f_out.write(data)
                end = current_offset + len(data)
                offsets.append((name, start, end))
                current_offset = end
            else:
                with open(datasets[name], "rb") as f_in:
                    data = f_in.read()
                start = current_offset
                f_out.write(data)
                end = current_offset + len(data)
                offsets.append((name, start, end))
                current_offset = end
                
    with open(os.path.join(RESULTS_DIR, out_name + "_manifest.txt"), "w") as f:
        for name, start, end in offsets:
            f.write(f"{name}: {start} -> {end}\\n")
            print(f"  {name}: {start} -> {end}")
    
    return out_path, offsets

def run_exhaustive_sweep():
    print("\\n--- Exhaustive Fixed Lambda Sweep ---")
    results = []
    datasets_to_test = ["c_code.c", "promessi_sposi.txt", "markdown_docs.md", "c_code.c"] # last is shuffled
    dnames = ["c_code", "prose", "markdown", "shuffled"]
    
    with open(os.path.join(RESULTS_DIR, "exhaustive_lambda_sweep.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "lambda", "phys_bpb"])
        
        for dname, fname in zip(dnames, datasets_to_test):
            print(f"Testing {dname}...")
            lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, "moe"]
            dpath = os.path.join(DATA_DIR, fname)
            
            for lam in lambdas:
                out_bin = "tmp_sweep.bin"
                cmd = [
                    "coder.exe", "--encode", dpath, out_bin,
                    "--weights", "weights/entropy_weights.bin",
                    "--profile", "fast",
                    "--blend", str(lam)
                ]
                if dname == "shuffled":
                    cmd.append("--shuffled")
                    
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                eval_len = os.path.getsize(dpath)
                compressed_bits = os.path.getsize(out_bin) * 8.0 if os.path.exists(out_bin) else 0.0
                phys_bpb = compressed_bits / eval_len if eval_len > 0 else 0.0
                
                writer.writerow([dname, str(lam), f"{phys_bpb:.4f}"])
                print(f"  Lambda {lam:>3}: {phys_bpb:.4f} BPB")
                
                if os.path.exists(out_bin):
                    os.remove(out_bin)

def run_moe_multi_domain(out_name):
    print(f"\\n--- Running MoE on {out_name} ---")
    in_path = os.path.join(DATA_DIR, out_name)
    out_bin = in_path + ".see"
    dec_bin = in_path + ".dec"
    tel_path = os.path.join(RESULTS_DIR, out_name + "_telemetry.csv")
    
    cmd_enc = [
        "coder.exe", "--encode", in_path, out_bin,
        "--weights", "weights/entropy_weights.bin",
        "--profile", "fast",
        "--blend", "moe",
        "--telemetry", tel_path
    ]
    subprocess.run(cmd_enc, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Test SHA-256 roundtrip
    cmd_dec = [
        "coder.exe", "--decode", out_bin, dec_bin,
        "--weights", "weights/entropy_weights.bin"
    ]
    subprocess.run(cmd_dec, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    import hashlib
    def get_sha(p):
        with open(p, "rb") as f: return hashlib.sha256(f.read()).hexdigest()
        
    orig_sha = get_sha(in_path)
    dec_sha = get_sha(dec_bin)
    if orig_sha == dec_sha:
        print(f"  SHA-256 Match OK! {orig_sha}")
    else:
        print(f"  WARNING: SHA-256 Mismatch!")
        
    if os.path.exists(out_bin): os.remove(out_bin)
    if os.path.exists(dec_bin): os.remove(dec_bin)

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Create multi-domain files
    order1 = ["c_code", "prose", "shuffled", "markdown"]
    create_multi_domain(order1, "multi_domain_1.bin")
    
    order2 = ["shuffled", "c_code", "markdown", "prose"]
    create_multi_domain(order2, "multi_domain_2.bin")
    
    # 2. Run MoE on multi-domain files
    run_moe_multi_domain("multi_domain_1.bin")
    run_moe_multi_domain("multi_domain_2.bin")
    
    # 3. Exhaustive Lambda Sweep
    run_exhaustive_sweep()
